# Design

Why Clarity works the way it does: the choices, what each one costs, and what is out of scope. The spec is `docs/brief.md`; the build plan and status are in `docs/plan.md`.

## Principles

- **What you see is exactly what the model got, and every call can be replayed.**
- **No silent fallbacks.** If something can't be done exactly as configured, Clarity stops with a clear error.
- **Clarity gets out of the way.** Being exact must not make common experiments awkward. Where a strict rule gets in the way, the answer is a small, explicit shortcut that is itself logged, not more ceremony. Example: `use_recommended = true` (below).

## Choices and tradeoffs

**The client owns the prompt, down to the token IDs.** It renders the prompt and tokenizes it with the model's Hugging Face tokenizer at a pinned commit, then sends token IDs. No server applies a template or tokenizes.
- Cost: the tokenizer and template files must be in the Hugging Face cache (`clarity fetch` puts them there), and `transformers` is a dependency (no torch).
- Check: the server's own decoding of the IDs must equal the client's prompt text, which shows the tokenizer and the GGUF agree.

**BOS is decided per mode and logged.** Text mode follows `add_bos_token` in `tokenizer_config.json`, cross-checked by probing the tokenizer; if they disagree, the profile won't load. Chat mode never prepends BOS: the chat template decides.

**Chat templates come from a pinned repo and are logged by hash.** `template_repo` and `template_revision` (default: the model's tokenizer) supply only the template; the model's own tokenizer always does the tokenizing. Every call logs the template's repo, commit, kwargs and sha256, plus the `transformers`, `tokenizers` and `jinja2` versions, because rendering depends on them. The order is fixed: system (if set), priming, then the conversation.

**Text that looks like a control token is a control token.** If your message or a prompt file contains, say, `<|im_end|>`, the tokenizer reads it as that control token, as Hugging Face does everywhere. Clarity keeps this standard behaviour, so both backends get the same IDs. It flags such calls (`special_tokens_in_content`, with where the text was found) and warns when your new message contains one.

**Line endings are part of the prompt.** Prompt files are read byte for byte, so `.gitattributes` stops git from converting line endings in `prompts/`, `profiles/` and the test fixtures. A Windows checkout therefore sends the same bytes.

**Every sampling value is sent, every time.** Settings come from the profile, then `defaults.toml` (neutral values). Every other field llama-server reports in `generation_settings` gets a fixed neutral value. After each call the client checks that the server echoes what was sent (floats at float32 precision) and that no unknown field appeared.
- Why: llama-server's defaults are not neutral, and since the converter copies `generation_config.json` into the GGUF, the server's defaults can differ per model.
- Cost: the neutral table is tied to the pinned llama.cpp commit. An upgrade that adds a sampler fails the "unaccounted settings" check until the table is updated.
- A setting only acts if its sampler is in `samplers`, so a non-neutral setting whose sampler is missing is an error when the profile loads, rather than a value that silently does nothing.

**The model's recommended settings are opt-in.** `use_recommended = true` in a profile fills every setting the profile leaves out from the model's pinned `generation_config.json`. These values are logged with origin `model`, and `/settings` shows them either way.
- Why opt-in: with the shared neutral defaults, two models are compared under the same settings unless you choose otherwise.
- Only sampling values map (`repetition_penalty` to `repeat_penalty`; `do_sample: false` to temperature 0). Everything else, including `max_new_tokens`, is listed as not applied.

**Clarity starts its own llama-server.** Every output-affecting flag is set in `server.toml`, nothing is left on `auto`, and environment variables and config files that llama.cpp would read are removed or refused. After startup and after every call, the client checks `/props` (model path, context size, process alive).
- Cost: one model at a time. Switching to a profile with another model restarts the server.
- The server never parses the model's chat template (`no-jinja`, with `chat-template = "chatml"`): Clarity doesn't use it, and OLMo 3's fails to parse in llama.cpp's template engine, which would stop the server from starting.
- Startup output is read at `log-verbosity` 4, because that is the level at which llama.cpp logs where the weights went. With GPU layers requested, a model that lands on the CPU is an error. The token-embedding table stays on the CPU by design.

**No prompt cache, no context shift.** `cache_prompt` is off and checked (`cache_n == 0`), so every call processes its whole prompt. That is slower for long conversations, but a cached prefix can change results. Context shift is off, so tokens are never dropped silently.

**The context guard allows a call only if prompt + `max_tokens` < context.** llama-server returns at most context − prompt tokens. At exactly prompt + `max_tokens` = context, it returns every token but marks the reply truncated. The guard therefore refuses that case, so an allowed call is never marked truncated (checked live). A warning appears above 80% of the context.

**A model's end token comes from its own files, not from assumptions.** OLMo 3's template closes the last assistant message with `<|endoftext|>`, not `<|im_end|>`, and the model does the same. Both are end tokens in its `generation_config.json`, and llama.cpp stops on either.

**Seeds are always concrete.** `seed = "random"` makes the client draw one and log it, because llama-server doesn't report a seed it picks itself.

**Models are pinned and converted locally.** `models.toml` names a repo and a 40-character commit. `fetch` converts with the llama.cpp checkout's own converter, in its own environment, and records everything it resolved in `models.lock`: sha256, converter package versions, and the model's recommended settings.
- `models.lock` is committed, so its paths are relative to the repo or written as `~/...`.
- The full sha256 is computed at fetch; server start checks only size and modified time; `clarity verify` re-hashes on demand.
- Converting SmolLM2 twice gave a byte-identical GGUF.

**Hand-written and generated files stay apart.** The tool never edits `models.toml`, profiles, `defaults.toml` or `server.toml`.

**Logs are append-only JSONL, one file per conversation** (a batch run puts all its conversations in one `calls.jsonl`), with a `schema_version`. Every call writes a record, including failures and Ctrl-C (`stop_type` `cancelled`). Readers ignore unknown keys, so later additions (token probabilities, reasoning text) need no format change. `logs/` and `runs/` are private and gitignored.

**Batch runs are blind, checked first and kept apart.** `clarity run` sends every opening to every profile, each call in a new conversation, and writes `blind.md` (openings and shuffled, labelled replies, nothing else), `key.json` (which label is which) and `calls.jsonl` to `runs/<run-id>/`.
- Before any model loads, every profile is prepared and every opening is rendered against it: unequal `max_tokens` (reply length would give the setup away), a profile listed twice, or an opening that doesn't fit the context stops the run.
- Seeds come from the run, not the profile: 333, then 334 and so on with `--seeds N`, logged with origin `run`. A fixed start makes two runs comparable; several seeds show a setup's spread rather than one sample.
- Openings are split on lines that are exactly `---` and stripped. CRLF becomes LF first, so an editor's line endings never change the token IDs. Replies are stripped in `blind.md` only; the log keeps them as generated.
- Calls are grouped by model, so each model loads once. The terminal shows progress and warnings only, never replies.
- Nothing goes to `logs/`, not even the server's output, so held-out openings never end up in a transcript export.

**Failure policy.** Configuration problems are errors before anything is sent. A failed after-call check is a flag on the record and a warning in the terminal; in batch mode it stops the run.

## Determinism and replay

**The same call on the same machine and build gives the same output.** This holds at any temperature, because the seed is always fixed. Live tests check it back to back, after a longer call in between, and after a server restart, for SmolLM2 and OLMo 3. It holds because Clarity removes what usually varies: one request at a time (one slot), no prompt cache, and fixed batch sizes, so every GPU kernel sees the same shapes every time.

**Some changes can change the output, and that's expected.** Floating-point sums depend on their order, and when the top tokens are nearly tied, one flipped token changes the rest of the reply. These change the order:
- the llama.cpp build, batch or micro-batch size, flash attention or KV cache type
- the GPU family, or the macOS version (Metal compiles llama.cpp's GPU kernels when the server starts)
- some `GGML_*` environment variables, and the GGUF itself

Clarity logs all of these, and replay (step 4) will list the ones that differ.

**Warnings are for things you can act on.**
- Per call: a failed check, a nearly full context, or control-token text in your message.
- Per server start: a build that doesn't match the pinned commit, uncommitted changes in the llama.cpp checkout, or inherited `GGML_*`/`LLAMA_*` variables (their values are logged).

There is no general "results may vary" notice. The avoidable causes are already removed, and the rest is logged. Replay will flag the one case that needs attention: output that differs although nothing that could explain it changed.

**Not attempted:** identical output across machines, builds or backends, which no inference stack offers; llama.cpp's experimental batch-invariant kernels, which fix a variation Clarity's single-slot setup doesn't have; and a determinism guarantee for the Transformers backend on Apple GPUs, which PyTorch doesn't provide (step 5 runs the same live tests instead). Research and sources: `docs/research/nondeterminism.md`.

## Usage patterns

- **Look at what the model got:** `/prompt` after any reply; `/prompt repr` shows whitespace and newlines, including a trailing newline your editor added to a prompt file.
- **Compare two setups:** make two profiles that differ in one key. Everything else, including settings you didn't set, is the same, because it comes from `defaults.toml`.
- **Try a model's own settings:** add `use_recommended = true`; `/settings` shows which values came from the model.
- **Check reproducibility:** the same call gives identical output token IDs on the same machine and build (tested live).
- **Compare setups fairly:** a fixed seed makes a reply replayable, not representative. At temperature 0.6 another seed gives a different reply, so compare setups over several openings or seeds.
- **Blind test:** `clarity run --openings FILE --profiles A B C` (add `--seeds 3` for three replies each). Give raters `blind.md`; open `key.json` only after rating.
- **Same model, chat versus plain document:** `4-raw-instruct` sends the instruct model a plain document with no template; compare it with a chat profile to see what the template contributes.

## Not in scope

From the brief: no GUI or web UI; no tools, retrieval or agents; no metrics or LLM judges; no multi-user features; no Ollama backend; no llama-server router mode; no MLflow export. The design stays open to a small local web page and an MLflow exporter.

Also left out:
- Running two models side by side (memory: two 7B models don't fit together on a 32 GB Mac).
- Speculative decoding, LoRA and control vectors (refused in `server.toml`).
- Windows and Linux are covered by offline CI only; the live tests have run on macOS (Metal) only.
