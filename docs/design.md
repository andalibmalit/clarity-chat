# Design

Why Clarity works the way it does: each choice, what it costs, and what is out of scope. For what each key and command does, see [Reference](reference.md); for how to use them, see [Guides](guides.md).

## Principles

- **What you see is exactly what the model got, and every call can be replayed.**
- **No silent fallbacks.** If something can't be done exactly as configured, Clarity stops with a clear error.
- **Clarity gets out of the way.** Being exact must not make common experiments awkward. Where a strict rule gets in the way, the answer is a small, explicit shortcut that is itself logged, such as `use_recommended = true`, not more ceremony.

## The prompt

**The client owns the prompt, down to the token IDs.** It renders the prompt and tokenizes it with the model's Hugging Face tokenizer at a pinned commit, then sends token IDs. No server applies a template or tokenizes.
- Cost: the tokenizer and template files must be in the Hugging Face cache, and `transformers` is a dependency (without torch).
- Check: after each call, the server's own decoding of the IDs must equal the prompt text, which shows that the tokenizer and the GGUF agree.

**BOS is decided per mode and logged.** Text mode follows `add_bos_token` in `tokenizer_config.json`, cross-checked by probing the tokenizer; if they disagree, the profile won't load. Chat mode never adds BOS, because the chat template decides.

**Chat templates come from a pinned repo and are logged by hash.** Each call logs the template's repo, commit, arguments and sha256, plus the `transformers`, `tokenizers` and `jinja2` versions, because rendering depends on all of them.

**Text that looks like a control token is a control token.** If a message or prompt file contains, say, `<|im_end|>`, the tokenizer reads it as that token, as Hugging Face does everywhere. Clarity keeps this standard behaviour so both backends get the same IDs, and flags and warns about such calls instead.
- Cost: there is no way to send such text as plain text.

**Line endings are part of the prompt.** Prompt files are read byte for byte, and `.gitattributes` stops git from converting line endings in `prompts/`, `profiles/` and the test fixtures, so a Windows checkout sends the same bytes.

**A model's end token comes from its own files, not from assumptions.** OLMo 3's template closes the last assistant message with `<|endoftext|>`, not `<|im_end|>`, and the model does the same. Both are end tokens in its `generation_config.json`, so llama.cpp stops on either.

## Settings

**Every sampling value is sent, every time.** Each setting comes from the profile, then from the model's recommended values if the profile opts in, then from `defaults.toml`, whose values turn every sampler off. Every other field llama-server reports gets a fixed neutral value.

After each call, Clarity checks that the server echoed what was sent and reported no field Clarity doesn't know.
- Why: llama-server's defaults aren't neutral, and since the converter copies `generation_config.json` into the GGUF, they can differ per model.
- Cost: the table of neutral values is tied to the pinned llama.cpp commit. An upgrade that adds a sampler fails the check until the table is updated.
- A setting acts only if its sampler is in `samplers`, so a non-neutral setting whose sampler is missing is an error when the profile loads, not a value that silently does nothing.

**The model's recommended settings are opt-in.** With the shared neutral defaults, two models are compared under the same settings unless you choose otherwise. `use_recommended = true` is the one-line way to choose otherwise, and its values are logged with origin `model`.

**Seeds are always concrete.** `seed = "random"` makes Clarity draw a seed and log it, because llama-server doesn't report a seed it picks itself, and that call couldn't be replayed.

## The server

**Clarity starts its own llama-server.** Every output-affecting flag is set in `server.toml`, nothing is left on `auto`, and the environment variables and config files llama.cpp would read are removed or refused. After startup and after each call, Clarity checks `/props` to confirm that the process it started is still running, with the same model and context size.
- Cost: one model at a time. Switching to a profile with another model restarts the server.
- The server never parses the model's chat template (`no-jinja`). Clarity doesn't use it, and OLMo 3's fails to parse in llama.cpp, which would stop the server from starting.
- With GPU layers requested, a model that lands on the CPU is an error. Clarity reads where the weights went from llama.cpp's startup output, which reports it only at `log-verbosity` 4.

**Each server start gets a new random key.** llama-server answers any program that can reach its port, including a web page open in your browser: by default its CORS headers let any site read the replies. Clarity passes a new key in the server's environment at each start (`LLAMA_API_KEY`), sends it with every request, and turns off the `/slots` endpoint and the built-in web UI, which would show prompts or stay public. Only `/health` answers without the key.
- The key is never logged. At `log-verbosity` 4, llama-server prints its last four characters; Clarity replaces that line in the startup output and the server log.
- Cost: the key is in the server process's environment, which other programs running as you can read. The key keeps out web pages, not your own programs.

**The context size is per profile.** A long document needs a large context, and a large context takes memory a short chat doesn't need. So a profile can set `ctx_size`; the rest use `server.toml`'s `ctx-size`. The size goes into the server's flags, so argv, the `/props` check and the log agree, and each conversation logs the size and its origin; each call logs `n_ctx`.
- Cost: switching between profiles of different sizes restarts llama-server, which reloads the model.
- Replay runs a call at its logged size, so the size is never a hidden difference between a call and its replay.
- llama-server quietly caps a size above the model's training context; Clarity refuses it instead, naming the limit.
- On a Mac, Clarity refuses a size whose memory doesn't fit. After llama-server starts, Clarity adds up what it put on the GPU (model, context cache, compute buffers) from its startup log, and compares that with what the GPU had free: Metal's `recommendedMaxWorkingSetSize`, which Apple describes as roughly what the GPU can allocate without affecting its performance, minus what was already in use. Measured, not modelled: on OLMo 3 7B the context cache took 2,752 MiB at 8,192 tokens and 5,824 MiB at 32,768, and the compute buffers 237 and 645 MiB with flash attention (609 and 2,193 MiB without). Other GPUs log the figures only, until the Linux release.

**A cancel works from any thread, and never adds an empty reply.** Whatever starts a call, such as the web page or a script, makes a cancel signal for that call alone, so a late cancel can't stop the next call, and one sent while the prompt is still being built stops the call before anything is sent. The llama.cpp backend checks it between stream lines, and llama-server sends a ping each second while it reads the prompt, so the call returns within about a second even then. Closing the stream tells llama-server to stop, which it does once it finishes the piece of the prompt it's working on (`batch-size` tokens). Measured on OLMo 3 7B on an M2 Max: a cancel half a second into a 6,000-token prompt freed the server after 4.3 s; the whole prompt takes 14 s. So a cancel never needs a server restart.
- A cancel before the first token is logged like a failed call, with `cancelled_during: "prompt"`, and your message is offered back. A reply with no text would be a turn nobody wrote.
- Cost: the Transformers backend checks only as each token arrives, so a cancel during its prompt pass takes effect at the first token.

**No prompt cache, no context shift.** A cached prefix can change results, and context shift drops tokens without saying so.
- Cost: every call processes its whole prompt, which is slower in long conversations.

**The context guard allows a call only if prompt + `max_tokens` < context.** At exactly prompt + `max_tokens` = context, llama-server returns every token but marks the reply truncated, so the guard refuses that case (checked live).

## Models and files

**Models are pinned and converted locally.** `models.toml` names a repo and a full commit hash. `fetch` converts with the llama.cpp checkout's own converter, in its own environment, and records everything it resolved in `models.lock`.
- `models.lock` is committed, so its paths are relative to the repo or written as `~/...`.
- The full sha256 is computed at fetch. A server start checks only size and modification time, which is fast; `clarity verify` re-hashes on demand.
- A ready-made GGUF takes its tokenizer from the model it was made from, and the check after each call shows whether the two agree.

**Shipped files stay read-only, and your files live in `local/`.** Clarity never edits `models.toml`, `profiles/`, `defaults.toml` or `server.toml`. Your own profiles go in `local/profiles/`, which git ignores, so a `git pull` never touches them. A profile name that exists in both folders is an error, because the name must say which file it means. Clarity writes only into `local/`, `logs/`, `runs/`, `models/`, `models.lock` and the Hugging Face cache, and a new kind of write must be visible to you when it happens.

**Every chat command is in one table,** which the terminal, `clarity chat --json` and the web interface all run. Each command returns its result as data and reports what happens while it runs as events: reply text as it streams, a model starting, and every record written to the log. So a script or another program can do anything you can do in the terminal, and see what you'd see.

**One running Clarity per project.** `chat`, `run` and `replay` hold a lock on `local/clarity.lock` while they run; a second one stops and says which process holds it. Two would load two models, more memory than a laptop has, and could write to the same log. The operating system releases the lock however the process ends, so a crash never leaves it stuck. `clarity mark` and `clarity note` only append a line, so they don't take it.
- Who holds the lock is in a separate file, `local/clarity.json`, because on Windows a locked file can't be read by another process.

**A crash doesn't leave a model in memory.** llama-server runs in its own process group, so it outlives a Clarity that is killed or crashes, holding 14 GB or more. Two things prevent that:
- SIGTERM, and SIGHUP from a closed terminal, exit Clarity through its normal cleanup, which stops the server.
- While a server runs, its port, process ID and key are in `local/clarity.json`. If Clarity dies anyway (`kill -9`, a power cut), the next `chat`, `run` or `replay` finds them there and asks that port's `/props`, with that key. Only an answer naming the same model proves it's the server that was left, and only then is it stopped; Clarity says so. Anything else is reported and left alone. No process list is read, so this works the same in a sandbox and on Windows.
- Cost: a per-start key sits on disk while the server runs, readable only by you. It guards the server against web pages, not against your own programs.

**Logs are append-only JSONL,** one file per chat conversation and one `calls.jsonl` per batch run, with a schema version. Every call writes a record, including failed and cancelled ones. Readers ignore keys they don't know, so later additions need no format change. `logs/` and `runs/` are private and gitignored.

**Failure policy.** Configuration problems are errors before anything is sent. A failed check after a call is a flag on the record and a warning in the terminal; in a batch run, it stops the run.

## The Transformers backend

**It runs the same weights in-process.** It loads the official weights the GGUF was converted from, in bf16, and is given the same prompt token IDs as llama-server, so the two backends can be compared, and a call's activations can be read in PyTorch.
- Every output-affecting `GenerationConfig` field is set on every call, and the model's own generation config is replaced by an empty one at load. Otherwise Transformers fills unset fields from the model's file and then from its own defaults (`top_k` 50, for one). Only `eos_token_id` comes from the model's file, and it is logged.
- The attention implementation is set to `sdpa` explicitly, because Transformers' default falls back to `eager` without saying so.
- Some settings can't run on Transformers as written (see [Settings](reference.md#settings)). They are recorded as not applied rather than refused, so a profile still runs on both backends and the record says how they differed.
- `PYTORCH_ENABLE_MPS_FALLBACK` is refused on MPS, because it would run unsupported operations on the CPU without notice.
- PyTorch's deterministic mode is never turned on: in inference it makes nothing deterministic on MPS or CPU and can only raise errors. Its state is logged.
- Only one model is ever in memory. If a new model fails to start, the previous one is started again, so a bad `/new` or `/profile` leaves you where you were.

## Conversations and runs

**Branches never overwrite.** `/regen`, `/edit` and `/undo` start new branches, and every turn stays in the log with its parent. The k-th `/regen` uses the first reply's seed plus k: with the same seed it would only repeat the reply, and checking that is `/replay`'s job.

**Replay re-sends the logged request, not a rebuilt one.** It sends the logged token IDs and request body, so a later change to how Clarity builds requests can't change a replay. It then lists every recorded fact that differs from the original call, so a different output comes with its likely causes.
- Cost: replay can't show whether a later fix to how requests are built changes a reply; that takes a new call.

**Batch runs are blind, checked first and kept apart.**
- Before any model loads, every opening is rendered against every profile. Unequal `max_tokens` stops the run, because reply length would give a setup away. The cost: all profiles in a run share one `max_tokens`.
- Seeds come from the run, starting at 333, so two runs with the same profiles are comparable, and `--seeds N` shows a setup's spread rather than one sample.
- Openings files have their line endings normalized, so an editor's line endings never change the token IDs.
- A server restart mid-run stops the run, so no run mixes server processes.
- Nothing goes to `logs/`, not even the server's output, so held-out openings never end up in a transcript export.

## Determinism and replay

**The same call on the same machine and build gives the same output,** at any temperature, because the seed is always fixed. Live tests check this back to back, after a longer call in between, and after a server restart, on SmolLM2 and OLMo 3. The Transformers backend passes the same checks (back to back and after a reload) on MPS for SmolLM2 and OLMo 3 Instruct, and on the CPU for SmolLM2. It holds because Clarity removes what usually varies: one request at a time, no prompt cache, and fixed batch sizes, so every GPU kernel sees the same shapes every time.

**Some changes can change the output, and that's expected.** Floating-point sums depend on their order, and when the top tokens are nearly tied, one flipped token changes the rest of the reply. The order changes with:
- the llama.cpp build, batch or micro-batch size, flash attention or KV cache type
- the GPU family, or the macOS version, since Metal compiles llama.cpp's GPU kernels when the server starts
- some `GGML_*` environment variables, and the GGUF itself

Clarity logs all of these, and replay lists the ones that differ.

**Warnings are for things you can act on.**
- Per call: a failed check, a nearly full context, or control-token text in your message.
- Per server start: a build that doesn't match the pinned commit, uncommitted changes in the llama.cpp checkout, or inherited `GGML_*` or `LLAMA_*` variables (or `PYTORCH_*` and `TORCH_*` on the Transformers backend), whose values are logged unless the name looks like a secret.

There is no general "results may vary" notice. Replay flags the one case that needs attention: output that differs although nothing that could explain it changed.

**Not attempted:** identical output across machines, builds or backends, which no inference stack offers; llama.cpp's experimental batch-invariant kernels, which fix a variation Clarity's single-slot setup doesn't have; and a determinism guarantee for the Transformers backend on Apple GPUs, which PyTorch doesn't provide. The live tests check that case instead.

## Not in scope

Left out from the start: tools or agents; metrics or LLM judges; multi-user features; an Ollama backend; llama-server's router mode; MLflow export.

Also left out:
- Running two models side by side. Two 7B models in bf16 need about 29 GB, more than a 32 GB laptop can spare.
- Speculative decoding, LoRA and control vectors, which `server.toml` refuses.
- Live tests on Windows and Linux. Offline CI covers them; the live tests have run on macOS (Metal) only.

## Changed decisions

When a choice above changes, the old one, the new one and the reason go here, newest first.

| Date | Was | Now | Why |
|---|---|---|---|
| 2026-11-10 | Clarity never edits any file you write. | Shipped files stay read-only; your own files live in `local/`, and Clarity writes there only visibly. | The web interface needs to save profiles and settings you make in it. Keeping them in `local/` keeps a `git pull` from touching them, and keeps the shipped files a fixed reference. |
| 2026-11-10 | A web interface and retrieval from documents were out of scope. | Both are planned: the web interface in v0.3.0, documents in v0.3.1 to v0.3.3 ([Roadmap](../ROADMAP.md)). | Making local models easy to use for people who don't use a terminal, without giving up exactness. Every feature still works from the terminal. |
