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

**Every sampling value is sent, every time.** Settings come from the profile, then `defaults.toml` (neutral values). Every other field llama-server reports in `generation_settings` gets a fixed neutral value. After each call the client checks that the server echoes what was sent (floats at float32 precision) and that no unknown field appeared.
- Why: llama-server's defaults are not neutral, and since the converter copies `generation_config.json` into the GGUF, the server's defaults can differ per model.
- Cost: the neutral table is tied to the pinned llama.cpp commit. An upgrade that adds a sampler fails the "unaccounted settings" check until the table is updated.
- A setting only acts if its sampler is in `samplers`, so a non-neutral setting whose sampler is missing is an error when the profile loads, rather than a value that silently does nothing.

**The model's recommended settings are opt-in.** `use_recommended = true` in a profile fills every setting the profile leaves out from the model's pinned `generation_config.json`. These values are logged with origin `model`, and `/settings` shows them either way.
- Why opt-in: with the shared neutral defaults, two models are compared under the same settings unless you choose otherwise.
- Only sampling values map (`repetition_penalty` to `repeat_penalty`; `do_sample: false` to temperature 0). Everything else, including `max_new_tokens`, is listed as not applied.

**Clarity starts its own llama-server.** Every output-affecting flag is set in `server.toml`, nothing is left on `auto`, and environment variables and config files that llama.cpp would read are removed or refused. After startup and after every call, the client checks `/props` (model path, context size, process alive).
- Cost: one model at a time. Switching to a profile with another model restarts the server.
- Startup output is read at `log-verbosity` 4, because that is the level at which llama.cpp logs where the weights went. With GPU layers requested, a model that lands on the CPU is an error. The token-embedding table stays on the CPU by design.

**No prompt cache, no context shift.** `cache_prompt` is off and checked (`cache_n == 0`), so every call processes its whole prompt. That is slower for long conversations, but a cached prefix can change results. Context shift is off, so tokens are never dropped silently.

**The context guard allows a call only if prompt + `max_tokens` < context.** llama-server returns at most context − prompt tokens. At exactly prompt + `max_tokens` = context, it returns every token but marks the reply truncated. The guard therefore refuses that case, so an allowed call is never marked truncated (checked live). A warning appears above 80% of the context.

**Seeds are always concrete.** `seed = "random"` makes the client draw one and log it, because llama-server doesn't report a seed it picks itself.

**Models are pinned and converted locally.** `models.toml` names a repo and a 40-character commit. `fetch` converts with the llama.cpp checkout's own converter, in its own environment, and records everything it resolved in `models.lock`: sha256, converter package versions, and the model's recommended settings.
- `models.lock` is committed, so its paths are relative to the repo or written as `~/...`.
- The full sha256 is computed at fetch; server start checks only size and modified time; `clarity verify` re-hashes on demand.
- Converting SmolLM2 twice gave a byte-identical GGUF.

**Hand-written and generated files stay apart.** The tool never edits `models.toml`, profiles, `defaults.toml` or `server.toml`.

**Logs are append-only JSONL, one file per conversation**, with a `schema_version`. Every call writes a record, including failures and Ctrl-C (`stop_type` `cancelled`). Readers ignore unknown keys, so later additions (token probabilities, reasoning text) need no format change. `logs/` and `runs/` are private and gitignored.

**Failure policy.** Configuration problems are errors before anything is sent. A failed after-call check is a flag on the record and a warning in the terminal; in batch mode it stops the run.

## Usage patterns

- **Look at what the model got:** `/prompt` after any reply; `/prompt repr` shows whitespace and newlines, including a trailing newline your editor added to a prompt file.
- **Compare two setups:** make two profiles that differ in one key. Everything else, including settings you didn't set, is the same, because it comes from `defaults.toml`.
- **Try a model's own settings:** add `use_recommended = true`; `/settings` shows which values came from the model.
- **Check reproducibility:** a fixed seed on the same server gives identical output token IDs (tested live).

## Not in scope

From the brief: no GUI or web UI; no tools, retrieval or agents; no metrics or LLM judges; no multi-user features; no Ollama backend; no llama-server router mode; no MLflow export. The design stays open to a small local web page and an MLflow exporter.

Also left out:
- Running two models side by side (memory: two 7B models don't fit together on a 32 GB Mac).
- Speculative decoding, LoRA and control vectors (refused in `server.toml`).
- Windows and Linux are covered by offline CI only; the live tests have run on macOS (Metal) only.
