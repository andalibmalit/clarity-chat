# Nondeterminism and replay

Research note, 2026-11-07. Question: even at temperature 0, LLM inference can differ from run to run. What does that mean for Clarity's replay ("identical", or a list of differences)?

Labels: **[src]** read in the llama.cpp checkout at `f477f795cb07` (paths relative to `vendor/llama.cpp/`). **[web]** a cited page says it. **[inf]** inference, not verified. A second agent re-checked every [src] and [web] claim on 2026-11-07. Corrections are folded in, and claims that could not be checked are marked **[unverifiable]**.

## Bottom line

- On one machine with one build, Clarity's setup is close to the case where llama.cpp is deterministic. With one slot and no prompt cache, each call's prompt is split into ubatches the same way, lands in the same KV cells, and runs the same kernels on the same shapes. The Metal kernels this model uses have no floating-point atomics. So the same token IDs and settings should give the same output IDs, even after a server restart or other requests in between. **[src + inf]** The live tests (`tests/test_live.py:97`, `tests/test_olmo.py:63`) confirm this only for "same running server, back to back".
- Output can differ when anything changes how the logits are computed: llama.cpp commit, `batch-size`/`ubatch-size`, flash attention, KV cache type, GPU family, macOS version (Metal compiles the kernels at server start), some `GGML_*` environment variables, or the GGUF. Clarity already logs almost all of these.
- At temperature 0.6 with a fixed seed, arithmetic noise is tiny next to seed-to-seed sampling variance. Bit-exactness makes a call auditable and replayable. It does not make a one-sample comparison statistically meaningful (see "Check 0").

## 1. Known causes (general)

- **Floating-point addition is not associative.** The order of a reduction changes the last bits, and one changed bit can flip a near-tied argmax, after which the texts diverge. **[web]** [He / Thinking Machines, 2025-09-10](https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/)
- **On servers, the main cause is a lack of batch invariance, not atomics.** The forward pass "involves no operations that require atomic adds". Outputs vary because server load changes the batch size, and the reduction strategy in RMSNorm, matmul and attention depends on batch size. In their test (Qwen3-235B, temperature 0), 1000 completions gave 80 unique outputs, first diverging at token 103. With batch-invariant kernels, all 1000 were identical. Cost on vLLM with Qwen3-8B: 26 s by default, 42 s with their improved attention kernel (55 s unoptimized). **[web]** same source.
- **Prefill and decode use different kernels.** The README warns that logits are "**not** guaranteed to be bit-for-bit identical for different batch sizes (prompt processing vs. token generation)", so `cache_prompt` "can cause nondeterministic results". **[src]** `tools/server/README.md:598`
- **Changing only `-ub` can change greedy output.** This was reported on master with speculation off (backend not stated) and reproduced by a maintainer on CUDA. The thread has no `-ub`-only test on Metal. The issue is about speculative decoding and is still open. A maintainer asked whether bitwise batch invariance is "something we want to guarantee?", and it hasn't been answered. **[web]** [llama.cpp #25618](https://github.com/ggml-org/llama.cpp/issues/25618). An open PR adds batch-invariant batched matmul for the x86 CPU backend only: [#31504](https://github.com/ggml-org/llama.cpp/pull/31504).
- **Hosted APIs** also change models and routing. One study measured accuracy differences of up to 15% between runs at "deterministic" settings. **[web]** [Atil et al., arXiv 2408.04667](https://arxiv.org/abs/2408.04667)

## 2. llama.cpp at `f477f795cb07`

**What temperature 0 does.** The chain ends in the `greedy` sampler instead of `dist` when `n_probs == 0` and either the last sampler is `temperature` with `temp == 0` and `dynatemp_range == 0`, or the last is `top_k` with `top_k == 1`. **[src]** `common/sampling.cpp:403-406`. Greedy takes the first maximum (strict `>`), so ties always break the same way. **[src]** `src/llama-sampler.cpp:1054-1061`. All Clarity profiles put `temperature` last, so temperature 0 is a pure argmax and the seed is unused.
- If `n_probs > 0`, the chain ends in `dist` instead. With `temp <= 0`, the temperature sampler keeps only the first maximum logit and sets the rest to −inf **[src]** `src/llama-sampler.cpp:266-287`, so `dist` must pick that same token. One RNG draw per token is still consumed. **[src + inf]**

**Sampling RNG (temperature > 0).** It uses `std::mt19937`, seeded per request, and `std::uniform_real_distribution<double>`. **[src]** `src/llama-sampler.cpp:1129,1168`. The C++ standard leaves that distribution's algorithm to the library, so libstdc++ (Linux) and libc++ (macOS) builds may draw different numbers from the same seed. **[inf]**

**Metal kernels.** The only atomics are integer ones, in `argsort.metal` (a histogram top-k) and `misc.metal` (`count_equal`). **[src]** `grep -rn atomic ggml/src/ggml-metal/kernels`. A dense model like OLMo 3 with `backend_sampling: false` shouldn't use them. **[inf]** Reductions use `simd_sum` and fixed loops, and the flash-attention split-KV merge is fixed-order. **[src]** `kernels/fa_aux.metal:364-404`. So the kernels should be deterministic run to run for fixed shapes. **[inf]** The live test supports this.

**Kernel choice depends on shapes and the device:**
- Flash attention uses the "vec" kernel when a ubatch has fewer than 20 query rows (decode). **[src]** `ggml-metal-ops.cpp:3200-3207`
- The vec kernel splits the KV into `nwg = 32` workgroups, and `nsg` grows with KV length. **[src]** `ggml-metal-ops.cpp:4234-4238`
- The vec kernel's `Q`/`NE` comes from a tuned table keyed by **Apple GPU family** and KV-length buckets (<1024 uses the baseline, then 4096 and 16384). **[src]** `ggml-metal-tuning.cpp:276` (Apple8 = M2), `:1140`
- Matmul switches between mat-vec and mat-mat paths on the token count (`ne11`), so the size of the last, partial ubatch matters. **[src]** `ggml-metal-ops.cpp:2742-2863`
- The Metal 4 tensor API is on by default only for M5, M6, A19 and A20. `GGML_METAL_TENSOR_ENABLE` turns it on elsewhere. **[src]** `ggml-metal-device.m:1146-1162`

So for the same call, everything is fixed. A different chip family, or an M5, may use different kernels. **[inf]**

**Kernels are compiled at server start.** `GGML_METAL_EMBED_LIBRARY=ON` in `build/CMakeCache.txt`, so the `.metal` source is compiled by the OS's Metal compiler through `newLibraryWithSource`. The only options set are preprocessor macros, plus the language version on tensor-API chips; everything else is Metal's default. **[src]** `ggml-metal-device.m:346-353`. So a macOS update can change the machine code even when the llama.cpp build hasn't changed. **[inf]**

**Earlier requests with `cache_prompt: false`.**
- `n_past = 0`, so the slot's whole sequence is removed. **[src]** `tools/server/server-context.cpp:4202`, `:4380`
- `seq_rm` moves the KV head back to the first freed cell. `find_slot` restarts at 0 when the head is far past the used cells. After each ubatch, the head moves to just past the last cell written. **[src]** `src/llama-kv-cache.cpp:461-477`, `:1057`, `:1234-1237`
- `n_kv` is the highest used cell + 1, padded to 256. Cell placement therefore changes `n_kv`, which changes the vec-kernel configuration above. **[src]** `:1318-1326`

Every call fills cells from 0 upward. OLMo 3's sliding-window cache is a ring of `pad(n_swa + 512, 256)` cells (4608 if `n_swa` is 4096) **[src]** `src/llama-kv-cache-iswa.cpp:73`. When it wraps, it overwrites cells in place starting at 0, so cell 0 is always in use when a call ends. Clearing the slot therefore always resets the head to 0, and the next call gets the same layout as on a fresh server. **[inf]** This hasn't been tested live after a long intervening call.

**Deterministic mode.** There is none. Grepping `common`, `src`, `ggml/src/ggml-metal`, `tools/server` and `docs` for "determinis" or "batch invariant" finds only unrelated or model-specific comments (DeepSeek top-k ordering). **[src]** Upstream's `test_consistent_result_same_seed` sends `cache_prompt: False`, and `test_cache_vs_nocache_prompt` is skipped ("fails on linux"). **[src]** `tools/server/tests/unit/test_completion.py:154-215`

**Environment variables that change numerics for Clarity's config.** Examples: `GGML_METAL_FUSION_DISABLE`, `GGML_METAL_GRAPH_OPTIMIZE_DISABLE`, `GGML_METAL_TENSOR_ENABLE/DISABLE`, and `GGML_METAL_BF16_DISABLE` (the GGUFs are bf16). `LLAMA_ATTN_ROT_DISABLE` acts only on a quantized KV cache, and `LLAMA_KV_ROWS_DISABLE` only on a unified one, so neither applies to `server.toml` today. **[src]** `grep getenv`, `src/llama-kv-cache.cpp:303-314`, `src/llama-context.cpp:401-407`. Clarity logs only the **names** of the `GGML_*`/`LLAMA_*` variables it keeps, not their values. **[src]** `src/clarity_chat/backends/llama_cpp.py:238,361`. Some of these variables are read by value (`atoi(...) != 0`), so the name alone doesn't show the setting.

## 3. Transformers on MPS (step 5)

- `torch.use_deterministic_algorithms(True)` makes no MPS operation deterministic. On MPS it only makes some operations raise errors: `scatter_reduce`, `index_put(accumulate=True)`, `index_reduce` (prod/mean), `kthvalue`, and several backward passes. "This setting alone is not always enough to make an application reproducible." **[web]** [PyTorch 2.15 docs](https://docs.pytorch.org/docs/2.15/generated/torch.use_deterministic_algorithms.html)
- In the SDPA backends, MATH is deterministic in both directions. Flash, efficient and cuDNN are deterministic in the forward pass and nondeterministic only in the backward pass. MPS isn't mentioned. **[web]** [PyTorch randomness notes](https://docs.pytorch.org/docs/2.15/notes/randomness.html)
- Transformers' `enable_full_determinism` sets CUDA, Ascend and flash-attention variables, the cuDNN flags and `use_deterministic_algorithms`. It has no MPS branch. **[web]** [trainer_utils.py](https://raw.githubusercontent.com/huggingface/transformers/main/src/transformers/trainer_utils.py)
- MPS forward kernels are Apple's MPS/MPSGraph code. Which kernel runs is opaque and may change with macOS and torch versions. Expect run-to-run determinism at batch size 1 on one setup, but check it with the same live test as llama.cpp. Don't expect the two backends to match bit for bit (the brief already treats that comparison as informational). **[inf]**

## 4. Answers for Clarity

### Q1. Can Clarity control or log every source?

| Source | Status |
|---|---|
| Batching with other requests | **Controlled:** `parallel = 1`, and Clarity owns the server |
| Prompt cache, KV reuse | **Controlled:** `cache_prompt` false, `cache-reuse 0`, `cache-ram 0`, `ctx-checkpoints 0`. **Checked** per call: `cache_n == 0` |
| Leftover KV state from earlier calls | **Controlled** indirectly: the slot is cleared and the head returns to 0 (section 2). Not yet tested live |
| How the prompt is split into ubatches | **Controlled and logged:** fixed `batch-size`/`ubatch-size` in `server_start.flags`, so the same prompt is always split the same way |
| Prefill vs decode kernels | **Inherent.** Matters only when a reply's tokens are re-fed as the next turn's prompt, never within one replay |
| Flash attention, KV type, GPU layers | **Controlled and logged** (`server.toml`, `server_start`) |
| GPU atomics, split-K | **Not used** by the Metal kernels this model runs (section 2) |
| CPU threads | **Controlled and logged** (`threads = 8`). With every layer on the GPU, the CPU does the embedding lookup and the sampling, both of which are thread-count-independent **[inf]** |
| Greedy ties | **Inherent,** but they break the same way every time |
| Sampling RNG | **Controlled:** concrete seed, `mt19937`. A Linux build may draw differently at temperature > 0 **[inf]** |
| llama.cpp commit | **Controlled and logged** (pin plus a `build_info` check). CMake options are **not logged** |
| GPU family | **Logged** (device description from the startup output). Not controllable |
| macOS / Metal compiler | **Logged** via `platform.platform()` and `os_version` (`logs.py:103-108`). Not controllable |
| `GGML_*`/`LLAMA_*` environment | **Partly logged:** names only |
| Weights | **Controlled and logged** (sha256) |
| MPS kernels (step 5) | **Neither.** Log the torch and macOS versions and test live |

### Q2. When should Clarity warn?

**[inf]**
- **Never per call.** The user can't act on a per-call warning.
- **At replay, only when the output differs.** Show the differing facts (the brief already requires this list). If the output differs and no output-relevant fact differs (only the client commit, which can't change the output for the same IDs), say so loudly. That is the brief's "stop and tell me" case.
- **Identical despite differences:** say so in one line.
- **At config time: nothing.** The options that make nondeterminism likely (`parallel > 1`, the prompt cache) are already fixed.

### Q3. How do others handle it?

- **OpenAI:** `seed` gives "mostly deterministic" outputs when the seed, parameters and `system_fingerprint` match. "Determinism is not guaranteed." **[web]** [OpenAI cookbook](https://developers.openai.com/cookbook/examples/reproducible_outputs_with_the_seed_parameter). Clarity's `server_start` record does the job of `system_fingerprint` with the facts themselves instead of an opaque ID.
- **vLLM:** `VLLM_BATCH_INVARIANT=1` (beta; NVIDIA compute capability 8.0+, AMD Instinct, Intel XPU) makes outputs "independent of the batch size or the order of requests in a batch", possibly at some cost in speed. **[web]** [vLLM docs](https://docs.vllm.ai/en/latest/features/batch_invariance/)
- **SGLang:** `--enable-deterministic-inference` works with FlashInfer, FA3 or Triton attention, and supports a per-request `sampling_seed`. **[web]** [SGLang docs](https://docs.sglang.io/advanced_features/deterministic_inference.html)
- Both of those modes solve nondeterminism across requests that share a batch. Clarity avoids the problem with one slot. **[inf]**
- **lm-evaluation-harness:** `--seed` (default `0,1234,1234,1234` for python, numpy, torch and few-shot), `--log_samples` for every input and output, plus a git hash, `transformers_version`, torch `pretty_env_info` and the tokenizer's special tokens. **[web]** [interface.md](https://raw.githubusercontent.com/EleutherAI/lm-evaluation-harness/main/docs/interface.md), [loggers/utils.py](https://raw.githubusercontent.com/EleutherAI/lm-evaluation-harness/main/lm_eval/loggers/utils.py)
- **HELM:** records `cached` and `request_datetime` for each request (because the backend can change), and uses a `random` field to ask for distinct samples. **[web]** [request.py](https://raw.githubusercontent.com/stanford-crfm/helm/main/src/helm/common/request.py). Reproducing results therefore means re-reading cached outputs. Clarity's replay also recomputes them. **[inf]**
- **Leon Chlon (Hassana Labs) LinkedIn posts:** one describes training-free "stable decoding" with a margin and noise-radius guarantee and 1.0–1.2× overhead. It was read through a summary of the page. **[unverifiable]** Its linked `hallbayes/.../stable_decode.py` returns 404, and the repo is now about another tool ("Berry"). **[web]** No paper was found. The other post is commentary on the Thinking Machines blog.

### Q4. What would be overengineering?

**[inf]**
- Batch-invariant kernels or llama.cpp patches. With one slot, Clarity already avoids the problem they solve.
- Margin-based stable decoding. It targets batch noise on providers' servers.
- Running every call twice, or hashing logits per call.
- Per-call warnings, or an opaque fingerprint hash next to the facts.
- Claims of bit-exactness across machines, or live tests on several GPUs.

## Check 0: arithmetic noise vs sampling variance

**[inf]** This is input for the user's decision, not a recommendation to change the brief.
- At temperature 0.6 and top_p 0.95, a different seed usually changes the reply within its first few dozen tokens.
- On this machine and build, the live test shows no arithmetic noise. Where noise exists, it changes a sampled token only when the draw lands next to a probability boundary, or when a logit sits at a top_k/top_p/min_p cutoff. Either way the result is another valid sample from almost the same distribution. Noise breaks exact replay but shouldn't bias a comparison between setups.
- With one reply per opening per profile, the difference between profiles is mixed with seed-to-seed variance. Using the same seed in every profile gives every profile the same random draws. That correlates the replies only weakly, and only until their tokens first differ, because different prompts give different distributions.
- More seeds per (opening, profile), or more openings, buy statistical power. Bit-exactness buys auditability. The cost is rater time: each extra seed is another reply for both raters.

## Recommendations (small)

1. **Live test: replay after a restart.** Make a call, stop and start the server, send the same request, and expect identical output IDs. This is the normal case for replaying a log from an earlier session.
2. **Live test: replay after a different, longer call.** Make call A, then call B with a longer prompt, then repeat A and expect identical output IDs. For OLMo, make B's prompt longer than 4608 tokens so the sliding-window ring wraps during prefill, which is fast.
3. **Log the values of the kept `GGML_*`/`LLAMA_*` environment variables, not just their names.**
4. **In the replay report**, when the output differs and the only differing fact is the client commit (or none), mark it as the unexplained case.
5. **Step 5 (MPS):** gate on the same back-to-back and restart live tests, and log the torch and macOS versions. Don't rely on `use_deterministic_algorithms(True)`: it makes no MPS operation deterministic.

## Not verified

- Run-to-run determinism after a restart, and after a long intervening call, on this machine (recommendations 1 and 2).
- Whether libc++ and libstdc++ `uniform_real_distribution<double>` give different draws for `mt19937`.
- That `simd_sum` has a fixed reduction order in hardware. Apple doesn't document it; the live test supports it.
- MPS run-to-run behavior. No specific PyTorch issue on MPS forward nondeterminism was found.
- OLMo 3's `n_swa` value (4096 assumed; read from the GGUF).
- The content of the LinkedIn posts and of Chlon's `stable_decode` code (now 404).
