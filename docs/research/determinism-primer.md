# Determinism primer and complexity audit

Brief, 2026-11-07, checked by a second agent. Questions: should Clarity turn on PyTorch's deterministic mode, how does run-to-run nondeterminism work, and did the nondeterminism work add complexity worth worrying about?

## Summary

- **Don't turn on `torch.use_deterministic_algorithms`, by default or as an option.** On MPS in the current release (2.15), it makes no operation deterministic; it only makes some MPS operations raise errors. On MPS and CPU, nothing it affects is in a language model's forward pass or sampling at batch size 1. Log whether it is on instead, since a library can turn it on.
- **The nondeterminism work added little code:** about 60 lines of source and 80 lines of tests, in one commit (`68c5bdb`). Each piece earns its place. The overkill options were considered and not built.
- **Most of what keeps Clarity deterministic predates that work**, because the spec already required it: one slot, no prompt cache, fixed batch sizes, a fixed seed, and every sampler value sent and checked.
- **One step-5 addition:** PyTorch reads `PYTORCH_MPS_*` environment variables that change MPS numerics. Give them the same warn-and-log treatment as `GGML_*`.

Labels: **[web]** a cited page says it. **[src]** read in source code. **[inf]** inference, not verified. All web and source pages were read through a fetch tool that summarizes or quotes on request; `curl` to `raw.githubusercontent.com` was denied, so nothing was grepped directly. Claims that rest on a single summarized read are marked **(summarized)**.

## 1. Foundations

**Floating-point addition isn't associative.** Every addition rounds, so the grouping changes the result. In Python, `(0.1 + 0.2) + 0.3` is `0.6000000000000001` and `0.1 + (0.2 + 0.3)` is `0.6`. A single dot product in a model sums thousands of terms, so its last bits depend on the order of the sum.

**Parallel hardware picks the order.** A GPU splits a long sum across threads and combines the partial sums. Two things can make that order vary:
- **Atomics.** Threads add into one memory cell in whatever order they finish, so the result can change from run to run on identical input.
- **Shape-dependent strategies.** Without atomics, the order is fixed for a given kernel and shape. But libraries pick a different kernel or split for a different shape, so a 1-row matrix and a 512-row matrix may be summed in different orders.

**Batch invariance.** On a busy server your request is batched with others, so its shapes depend on the load. Thinking Machines found this, not atomics, is the main cause of nondeterminism on inference servers: 1000 identical temperature-0 requests to Qwen3-235B gave 80 distinct completions. **[web]** [He, Thinking Machines, 2025](https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/)

**Small differences become big ones.** A last-bit change rarely matters, unless the two most likely tokens are nearly tied. Then the chosen token flips, and every later token sees a different prompt.

**The sampling RNG** is the other source. At temperature > 0, the next token is a random draw. The same seed and RNG implementation give the same draws; a different seed, library or device gives different ones.

| Kind of "same" | What must match | Who offers it |
|---|---|---|
| **Run to run** | Same machine, build, settings and shapes | Usually achievable when nothing uses atomics. Clarity tests this. |
| **Across machines, builds or backends** | Different chip, OS, library or runtime | No inference stack. PyTorch says so explicitly (section 3). |
| **Across batch sizes or load** | Same request, different company in the batch | Batch-invariant kernels (vLLM, SGLang). Clarity avoids the problem with one slot. |

## 2. PyTorch's deterministic mode

### What it does

`torch.use_deterministic_algorithms(True)` sets a global flag. Operations on one list switch to a deterministic implementation. Operations on a second list, which have no deterministic implementation, raise a `RuntimeError`; with `warn_only=True` they warn and run as before. Everything else is unaffected. The docs say: "This setting alone is not always enough to make an application reproducible." **[web]** [PyTorch 2.15 docs](https://docs.pytorch.org/docs/2.15/generated/torch.use_deterministic_algorithms.html)

It also makes some operations fill newly allocated memory, while `torch.utils.deterministic.fill_uninitialized_memory` is on (the default), which "is detrimental to performance." **[web]** same page; [reproducibility notes](https://docs.pytorch.org/docs/2.15/notes/randomness.html)

### What it covers

- **CUDA:** most of the first list, including convolutions, `scatter`, `scatter_add_`, `index_add`, floating-point `cumsum`, and several backward passes. **[web]** 2.15 docs
- **CPU:** a few indexing entries: `index_put` and `put_` with `accumulate=True`, `index_copy`, and the backward of `__getitem__` with tensor indices. **[web]** same
- **MPS (2.15):** nothing on the first list. On the error list: `scatter_reduce`, `index_put(accumulate=True)` and `index_reduce` (prod/mean) on float tensors, `kthvalue`, and the backward of `Embedding`, `EmbeddingBag`, `grid_sample`, some pooling and `interpolate`. **[web]** 2.15 docs, quoted per item
- **MPS (unreleased `main`):** five MPS entries have moved to the first list: `scatter_reduce`, `index_reduce`, and the backward of `grid_sample`, pooling and `interpolate`. **[src]** [`torch/__init__.py`](https://raw.githubusercontent.com/pytorch/pytorch/main/torch/__init__.py), quoted per item (two fetches; one paraphrase disagreed on `Embedding`, the per-item quotes put it on the error list). None is used in inference, so the conclusion holds.

**What a language model uses at batch size 1:** an embedding lookup, matmuls, RMSNorm, rotary embeddings, attention (SDPA), elementwise ops, and for sampling: softmax, sort, `cumsum`, `scatter` and `multinomial`. **[src]** Transformers' top-p warper calls `cumsum` and `scatter` ([`logits_process.py`](https://raw.githubusercontent.com/huggingface/transformers/main/src/transformers/generation/logits_process.py)). Of these, only `cumsum` and `scatter` are listed, and only for CUDA. **[web]** 2.15 docs. For SDPA, the reproducibility notes say every CUDA backend's forward pass is deterministic; MPS isn't mentioned. **[web]** reproducibility notes

So on MPS and CPU, deterministic mode changes nothing in Clarity's inference. Its only possible effect is an error if some code path hits an operation on the error list. **[inf]** On CUDA it would affect sampling.

### Related settings

The cuDNN flags (`torch.backends.cudnn.deterministic`, `benchmark`) and `CUBLAS_WORKSPACE_CONFIG` are CUDA-only and don't apply to Clarity. Whether current PyTorch still requires `CUBLAS_WORKSPACE_CONFIG` in deterministic mode is **unverified**: the 2.15 docs don't mention it and the old check isn't in `main`'s `Context.cpp`, but it may have moved.

`torch.manual_seed(n)` matters. It seeds the CPU generator and, when present, CUDA, MPS, XPU and MTIA. **[src]** [`torch/random.py`](https://raw.githubusercontent.com/pytorch/pytorch/main/torch/random.py). Clarity will call it per call with the logged seed.

Transformers adds nothing for MPS. `set_seed` seeds Python, NumPy and torch, and its `deterministic=True` argument calls `use_deterministic_algorithms(True)`. `enable_full_determinism` also sets CUDA, Ascend and flash-attention variables and the cuDNN flags. **[src]** [`trainer_utils.py`](https://raw.githubusercontent.com/huggingface/transformers/main/src/transformers/trainer_utils.py)

### What does change MPS numerics

PyTorch reads environment variables that change how MPS computes, like `GGML_*` for llama.cpp. **[web]** [MPS environment variables](https://docs.pytorch.org/docs/2.15/mps_environment_variables.html)
- `PYTORCH_MPS_FAST_MATH=1` turns on Metal fast math.
- `PYTORCH_MPS_PREFER_METAL=1` uses Metal kernels instead of MPSGraph for matmul.
- `PYTORCH_MPS_METAL_VERSION` caps the Metal language version; below 4.0 it disables the MetalPerformancePrimitives kernels "even on macOS 26". So the macOS version selects kernels in PyTorch too.
- `PYTORCH_ENABLE_MPS_FALLBACK=1` runs unsupported ops on the CPU: a silent fallback in Clarity's terms.

The docs describe what each variable does, not whether results change. Fast math and a different kernel choice change results almost certainly **[inf]**; the fallback changes which device computes an op.

Beyond these, MPS is opaque. Most kernels are Apple's MPS and MPSGraph code, and PyTorch doesn't document their reduction order. **[web]** [MPS backend notes](https://docs.pytorch.org/docs/2.15/notes/mps.html). No report of run-to-run forward nondeterminism on MPS at fixed shapes was found. The nearest is [#84936](https://github.com/pytorch/pytorch/issues/84936), about MPS and CPU giving different results (closed in 2024 after an `exp` kernel fix), which is cross-device, not run to run.

### Recommendation

**Never call `use_deterministic_algorithms` in Clarity.**
- For: on CUDA it would make sampling's `cumsum` and `scatter` deterministic.
- Against: Clarity has no CUDA backend. On MPS and CPU it changes nothing in inference, can add errors and the memory-filling cost, and would suggest a guarantee that doesn't exist.
- If a CUDA backend is ever added, revisit it there.

**What to log at each backend start (step 5):**
- `torch.are_deterministic_algorithms_enabled()` and `torch.is_deterministic_algorithms_warn_only_enabled()`. Warn if on, since it can raise errors mid-reply.
- All `PYTORCH_*` variables and their values, with a start warning when any is set, reusing the `GGML_*` logic. Consider refusing `PYTORCH_ENABLE_MPS_FALLBACK=1` outright, as Clarity refuses llama.cpp config files.
- Torch version, device, dtype, `attn_implementation`, and `torch.get_num_threads()`. On CPU, the thread count may change how sums are split **[inf]**, so set it explicitly, as `threads = 8` does for llama-server.

## 3. What Clarity can guarantee

**llama.cpp on this Mac: identical output for the same build, flags, environment, macOS and machine.** Same output token IDs and stop type, at any temperature, because the seed is always fixed. Live tests check it back to back, after a longer call in between (5000 tokens on OLMo), and after a server restart, on SmolLM2 and OLMo 3. It holds because one slot, no prompt cache and fixed batch sizes give every kernel the same shapes, and the Metal kernels these models use have no floating-point atomics. Section 1 of `docs/research/nondeterminism.md` lists what can change it; Clarity logs each.

**Transformers on MPS: expected, not guaranteed.** Batch size 1 with fixed shapes should be deterministic run to run, but PyTorch doesn't promise it. Step 5 runs the same three live tests. If they pass, say "tested" with the torch and macOS versions; if not, warn once at backend start and point to `device = "cpu"`.

**Transformers on CPU: likely deterministic with a fixed thread count.** **[inf]** Run the same tests.

**Across backends: never bit-identical.** PyTorch: "Completely reproducible results are not guaranteed across PyTorch releases, individual commits, or different platforms," and results "may not be reproducible between CPU and GPU executions, even when using identical seeds." **[web]** reproducibility notes. The RNGs differ too, so the same seed gives different draws. The brief already treats cross-backend comparison as information only.

## 4. Complexity audit

The nondeterminism work is two commits: `bd5a77a` (the research note) and `68c5bdb` (code, tests and docs: 173 lines added, 20 removed, 13 files). Sizes below are from `git show --numstat 68c5bdb`.

| Piece | What it does | Size | Verdict |
|---|---|---|---|
| `start_warnings()`, `backends/llama_cpp.py:243` | Start warnings: build mismatch (predates, moved here), uncommitted llama.cpp changes, inherited `GGML_*`/`LLAMA_*` variables | 23 lines, ~11 new | **Keep.** Each warning is actionable. Step 5 can reuse it for `PYTORCH_*`. |
| `server_env()` logs values | Logs `GGML_*`/`LLAMA_*` values, not just names | 1 line | **Keep.** Some are read by value, so a name alone was ambiguous. `LLAMA_API_KEY` is removed first, so it isn't logged. |
| `show_server_warnings()`, `cli.py:179` | Shows start warnings after `/new` too, once per server | ~12 lines | **Keep.** Otherwise a restart from switching models hid its warnings. |
| `_macos()`, `_command_output()`, `logs.py:121` | macOS version, build and chip in the machine record | 22 lines | **Keep.** Metal compiles kernels at server start, so a macOS update can change output. The chip will be the only chip record for the Transformers backend. |
| `@cache` on `machine()` | Reads machine facts once per process | 1 line | **Keep.** Avoids two subprocesses per record. |
| Replay live tests, `test_live.py:108`, `test_olmo.py:73` | Replay after a longer call and after a restart | 33 lines | **Keep.** This is the evidence for the "same machine, same output" claim. Costs a 5000-token prefill and a reload in `-m olmo`. |
| Offline tests (`test_llama_cpp`, `test_cli`, `test_logs`, `fakes`) | Cover the warnings and macOS record | 45 lines | **Keep.** |
| `docs/design.md`, "Determinism and replay" | The guarantee, what can change it, warning policy, what's not attempted | 21 lines | **Keep.** "Not attempted" records what was deliberately not built. |
| README note | One paragraph | 4 lines | **Keep.** |
| `docs/research/nondeterminism.md` | Sources and llama.cpp source reading | 134 lines | **Keep as reference; fix three stale lines.** The Q1 table says env vars are logged by "names only" (values since `68c5bdb`). Lines 54 and 123 say the mode "makes no MPS operation deterministic"; add "in 2.15", or say "nothing in inference". |

**Planned, not built:** step 4 replay flags a difference when nothing that could explain it changed (the brief asks for this); step 5 gates MPS on the live tests and adds the `PYTORCH_*` handling above. Both **keep**.

**Considered and correctly not built** (Q4 of the research note): batch-invariant kernels, margin-based "stable decoding", running every call twice, hashing logits, per-call warnings, an opaque fingerprint, and cross-machine claims.

**Predates the nondeterminism work.** The skeleton (`c25a0e1`) and steps 1 and 2 already had what makes llama.cpp deterministic here, because the brief requires it:
- `server.toml`: `parallel = 1`, `cache-reuse 0`, `cache-ram 0`, `ctx-checkpoints 0`, `no-cache-idle-slots`, fixed batch sizes, `no-context-shift`, fixed threads.
- `cache_prompt: false` on every request, checked with `cache_n == 0` (`llama_cpp.py:72`, `:586`).
- Every sampler value sent and echo-checked; concrete seeds.
- `LLAMA_ARG_*` removed, config files refused, and the checkout pinned to a commit (a hard error if it differs).

**Verdict:** the work is cheap, about 60 simple source lines, each tied to an actionable fact or a documented claim. Nothing is overkill and nothing needs cutting. The conceptual weight sits in the research note, which is reference material. The design didn't change: the guarantee comes from choices the spec already made.

## Not verified

- Exact PyTorch docstring lists were read through the fetch tool, item by item, not grepped.
- Whether `CUBLAS_WORKSPACE_CONFIG` is still required in deterministic mode (irrelevant to Clarity today).
- MPS run-to-run determinism at fixed shapes, and whether torch's MPS `multinomial` draws repeat for a fixed seed. Torch isn't installed and the sandbox hides Metal; step 5's live tests will settle both.
- CPU thread-count effects in PyTorch.
