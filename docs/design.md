# Design

Why Clarity works the way it does: each choice, what it costs, and what is out of scope. For what each key and command does, see [Reference](reference.md); for how to use them, see [Guides](guides.md).

## Principles

- **What you see is exactly what the model got, and every completed call can be replayed.**
- **No silent fallbacks.** If something can't be done exactly as configured, Clarity stops with a clear error. Two cases are flagged instead, on the record and on screen: a check that fails after a call, and a setting the Transformers backend can't apply.
- **What Clarity trusts.** It acts on these as written: its own code and locked dependencies, your llama.cpp build, the files it ships, your `local/` folder, and what you type or pass on the command line. Everything else is untrusted: logs and exports (yours may be hand-edited, or someone else's), profiles and prompt files from someone else, downloaded model files and their chat templates, everything llama-server sends back and, from v0.3.0, every request from the browser. Untrusted input may be refused, but it must never reach a command line, a file outside the project folder or a network setting unchecked. So logged flags go through the same checks as `server.toml`'s, and a profile's prompt files must be inside the project folder.
- **Clarity gets out of the way.** Being exact must not make common experiments awkward. Where a strict rule gets in the way, the answer is a small, explicit shortcut that is itself logged, such as `use_recommended = true`, not more ceremony.

## The prompt

**The client owns the prompt, down to the token IDs.** It renders the prompt and tokenizes it with the model's Hugging Face tokenizer at a pinned commit, then sends token IDs. No server applies a template or tokenizes.
- Cost: the tokenizer and template files must be in the Hugging Face cache, and `transformers` is a dependency (without torch).
- Check: after each call, the server's own decoding of the IDs must equal the prompt text, which shows that the tokenizer and the GGUF agree.

**BOS is decided per mode and logged.** Text mode follows `add_bos_token` in `tokenizer_config.json`, cross-checked by probing the tokenizer; if they disagree, the profile won't load. Chat mode never adds BOS, because the chat template decides.

**Chat templates come from a pinned repo and are logged by hash.** Each call logs the template's repo, commit, arguments and sha256, plus the `transformers`, `tokenizers` and `jinja2` versions, because rendering depends on all of them.

**Text that looks like a control token is a control token.** If a message or prompt file contains, say, `<|im_end|>`, the tokenizer reads it as that token, as Hugging Face does everywhere. Clarity keeps this standard behaviour so both backends get the same IDs, and flags and warns about such calls instead.
- Cost: there is no way to send such text as plain text.

**Each call records where each message sits in its prompt.** A message's tokens differ from call to call: OLMo 3 ends only the last reply with `<|endoftext|>`, Gemma 4 ends the prompt with an empty thinking block when thinking is off and strips thinking from earlier replies, and Qwen rewrites earlier reasoning. So the spans come from each call as it was rendered, never from rendering the conversation again later. To find them, Clarity renders the call a second time with each message replaced by a unique placeholder, which shows the template's own text around each message, then puts the messages back and checks that the result equals the real prompt exactly. A template that trims a message's edge whitespace is matched by trying that message stripped, and the span says `trimmed`. A template that rewrites, drops or repeats a message gets no spans for that call, with the reason, never a guess; a call never fails over its spans.
- Tokens come from the fast tokenizer's character offsets, checked to give the call's own IDs and to cover the text without gaps. A token that crosses an edge belongs to the span it starts in, which says `edge_token`. In text mode, `User: ` is the profile's text, so a tokenizer that joins the space to the first word gives that token to the profile, not the message.
- Checked over 26 saved chat templates: plain messages give spans on all 26; with edge whitespace, 25; with a `<think>` block in a reply, 21 (the Qwen3.5 and 3.6 family and Nemotron rewrite it).
- Cost: a second template render and a second tokenization per call.

**Line endings are part of the prompt.** Prompt files are read byte for byte, and `.gitattributes` stops git from converting line endings in `prompts/`, `profiles/` and the test fixtures, so a Windows checkout sends the same bytes.

**A model's end token comes from its own files, not from assumptions.** OLMo 3's template closes the last assistant message with `<|endoftext|>`, not `<|im_end|>`, and the model does the same. Both are end tokens in OLMo 3 Instruct's `generation_config.json`, and llama.cpp stops on either.
- The two backends can stop on different tokens. Transformers stops on the IDs in `generation_config.json` (or the tokenizer's end token, if the file lists none). llama.cpp stops on the end token in the GGUF and also on tokens it recognizes by name, such as `<|im_end|>`, `<|endoftext|>` and some fill-in-the-middle tokens (`vendor/llama.cpp/src/llama-vocab.cpp`). For SmolLM2 that is {0, 2, 4} on llama.cpp (4 is `<reponame>`) against {2} on Transformers; for OLMo 3 base, {100257, 100265} against {100257}. Clarity keeps each backend's own set and doesn't merge them, because merging would change what one backend does. Instead each call's end tokens are a fact (`end_tokens`) that replay, resume and the export check compare, so a difference is shown.

## Settings

**Every sampling value is sent, every time.** Each setting comes from the profile, then from the model's recommended values if the profile opts in, then from `defaults.toml`, whose values turn every sampler off. Every other field llama-server reports gets a fixed neutral value.

After each call, Clarity checks that the server echoed what was sent and reported no field Clarity doesn't know.
- Why: llama-server's defaults aren't neutral, and since the converter copies `generation_config.json` into the GGUF, they can differ per model.
- Cost: the table of neutral values is tied to the pinned llama.cpp commit. An upgrade that adds a sampler fails the check until the table is updated.
- A setting acts only if its sampler is in `samplers`, so a non-neutral setting whose sampler is missing is an error when the profile loads, not a value that silently does nothing.

**The model's recommended settings are opt-in.** With the shared neutral defaults, two models are compared under the same settings unless you choose otherwise. `use_recommended = true` is the one-line way to choose otherwise, and its values are logged with origin `model`.

**Seeds are always concrete, and the default is 42.** With a fixed default, on the same machine and build, the same message with the same history and settings gets the same reply, so a difference between two replies comes from what you changed. `seed = "random"` makes Clarity draw a seed and log it, because llama-server doesn't report a seed it picks itself, and that call couldn't be replayed.

## The server

**Clarity starts its own llama-server.** Every output-affecting flag is set in `server.toml`, nothing is left on `auto`, `server.toml` accepts only a fixed list of flags, and the environment variables and config files llama.cpp would read are removed or refused. After startup, Clarity checks `/props` to confirm that the server it started has the expected model and context size, and that its build is the pinned commit; a build from another commit is an error. After each call, it checks that the process is still running. Nothing can change a running server's model or context: `server.toml` allows no flag that would, and the per-start key keeps other programs out.
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
- On a Mac, Clarity refuses a size whose memory doesn't fit. After llama-server starts, Clarity adds up what it put on the GPU (model, context cache, compute buffers) from its startup log, and compares that with Metal's `recommendedMaxWorkingSetSize`, which Apple describes as an approximation of how much memory the GPU can allocate without affecting its performance. Other apps' memory isn't counted, so the check catches a size that can't fit, not one that a busy machine can't fit right now. Measured, not modelled: on OLMo 3 7B the context cache took 2,752 MiB at 8,192 tokens and 5,824 MiB at 32,768, and the compute buffers 237 and 645 MiB with flash attention (609 and 2,193 MiB without). Other GPUs log the figures only, until the Linux release.

**A cancel works from any thread, and never adds an empty reply.** Whatever starts a call, such as the web page or a script, makes a cancel signal for that call alone, so a late cancel can't stop the next call, and one sent while the prompt is still being built stops the call before anything is sent. The llama.cpp backend checks it between stream lines, and Clarity asks llama-server for a ping each second while it reads the prompt (`sse_ping_interval`; the server's default is 30 s), so the call returns within about a second even then. Closing the stream tells llama-server to stop, which it does once it finishes the piece of the prompt it's working on (`batch-size` tokens). Measured on OLMo 3 7B on an M2 Max: a cancel half a second into a 6,000-token prompt freed the server after 4.3 s; the whole prompt takes 14 s. So a cancel never needs a server restart.
- A cancel before the first token is logged like a failed call, with `cancelled_during: "prompt"`, and your message is offered back. A reply with no text would be a turn nobody wrote.
- Cost: the Transformers backend checks only as each token arrives, so a cancel during its prompt pass takes effect at the first token.

**No prompt cache, no context shift.** A cached prefix can change results, and context shift drops tokens without saying so.
- Cost: every call processes its whole prompt, which is slower in long conversations.

**The context guard allows a call only if prompt + `max_tokens` < context.** At exactly prompt + `max_tokens` = context, llama-server returns every token but marks the reply truncated, so the guard refuses that case (checked live).

## Models and files

**Models are pinned and converted locally.** `models.toml` names a repo and a full commit hash. `fetch` converts with the llama.cpp checkout's own converter, in its own environment, and records everything it resolved in `models.lock`.
- `models.lock` is committed, so its paths are relative to the repo or written as `~/...`.
- The full sha256 is computed at fetch. A server start checks only size and modification time, which is fast; `clarity verify` re-hashes on demand.
- A running server is kept for the next profile only if its file is the same: same path, inode, size and modification time. `fetch` replaces a file by rename, which changes the inode, so a model fetched again while its server runs restarts the server. A file overwritten in place could keep all four and isn't covered.
- A ready-made GGUF takes its tokenizer from the model it was made from, and the check after each call shows whether the two agree.

**Shipped files stay read-only, and your files live in `local/`.** Clarity never edits `models.toml`, `profiles/`, `defaults.toml` or `server.toml`. Your own profiles go in `local/profiles/`, which git ignores, so a `git pull` never touches them. A profile name that exists in both folders is an error, because the name must say which file it means. Clarity writes into `local/`, `logs/`, `runs/`, `models/`, `models.lock` and the Hugging Face cache, and appends to a log you name (`clarity replay`, `clarity mark`, `clarity note`; replay also puts the server's output next to it). A new kind of write must be visible to you when it happens.

**Every chat command is in one table,** which the terminal, `clarity chat --json` and the web interface all run. Each command returns its result as data and reports what happens while it runs as events: reply text as it streams, a model starting, and every record written to the log. So a script or another program can do anything you can do in the terminal, and see what you'd see.

**One running Clarity per project.** `chat`, `run` and `replay` hold a lock on `local/clarity.lock` while they run; a second one stops and says which process holds it. Two would load two models, more memory than a laptop has, and could write to the same log. The operating system releases the lock however the process ends (on Windows, possibly after a short delay), so a crash never leaves it stuck. `clarity mark` and `clarity note` only append a line, so they don't take it.
- Who holds the lock is in a separate file, `local/clarity.json`, because on Windows a locked file can't be read by another process.

**A crash doesn't leave a model in memory.** llama-server runs in its own process group, so it outlives a Clarity that is killed or crashes, holding 14 GB or more. Two things prevent that:
- SIGTERM, and SIGHUP from a closed terminal, exit Clarity through its normal cleanup, which stops the server.
- While a server runs, its port, process ID and key are in `local/clarity.json`. If Clarity dies anyway (`kill -9`, a power cut), the next `chat`, `run` or `replay` finds them there and asks that port's `/props`, with that key. Only an answer naming the same model proves it's the server that was left, and only then is it stopped; Clarity says so. Anything else is reported and left alone. No process list is read, so this works the same in a sandbox and on Windows.
- Cost: a per-start key sits on disk while the server runs, readable only by you.

**Logs are append-only JSONL,** one file per chat conversation and one `calls.jsonl` per batch run, with a schema version. Every call writes a record, including failed and cancelled ones. Readers ignore keys they don't know, so later additions need no format change. A line cut off by a crash or a full disk stays as it was: the next write ends it and adds a `torn_line` record naming it, and Clarity says so. Until then, readers treat a last line with no line end as not yet written, so reading a log never writes to it. A line a `torn_line` names is never a record, even if it parses, so a log reads the same before and after it's mended; any other damage still stops them. `logs/` and `runs/` are private and gitignored.
- Cost: a chat keeps every call record in memory, about 4 to 5 times its size in the log. Each prompt is capped at the context size, so this grows with the number of calls: 400 calls at an 8,192-token context hold about 200 MB. A 32k context with `logprobs` on comes nearer 1 GB, and on Transformers that's in the same process as the model.

**Failure policy.** Configuration problems are errors before anything is sent. A failed check after a call is a flag on the record and a warning in the terminal; in a batch run, it stops the run.

## The Transformers backend

**It runs the same weights in-process.** It loads the official weights the GGUF was converted from, in bf16, and is given the same prompt token IDs as llama-server, so the two backends can be compared, and a call's activations can be read in PyTorch.
- Every output-affecting `GenerationConfig` field is set on every call, and the model's own generation config is replaced by an empty one at load. Otherwise Transformers fills unset fields from the model's file and then from its own defaults (`top_k` 50, for one). Only `eos_token_id` comes from the model: from its `generation_config.json`, or, if that lists none, from its tokenizer's end token, which Clarity says at start. The IDs and their source are logged and shown in `/settings`. Some models need the second: OLMo 3 base's file lists no end token, and Qwen3.5-9B ships no file. Transformers' own fallback, `config.json`, would give Qwen 248044 (`<|endoftext|>`), not its chat stop `<|im_end|>` (248046). A model with an end token in neither is an error.
- The attention implementation is set to `sdpa` explicitly, because Transformers' default falls back to `eager` without saying so.
- Some settings can't run on Transformers as written (see [Settings](reference.md#settings)). They are recorded as not applied rather than refused, so a profile still runs on both backends and the record says how they differed.
- `PYTORCH_ENABLE_MPS_FALLBACK` is refused on MPS, because it would run unsupported operations on the CPU without notice.
- PyTorch's deterministic mode is never turned on: in inference it makes nothing deterministic on MPS or CPU and can only raise errors. Its state is logged.
- **Token probabilities come from the raw scores, on Transformers first.** With `logprobs` on, `generate` also returns each step's logits, taken before any sampler, and Clarity turns them into log-probabilities after the reply is done. So the numbers say what the model itself gave each token, not what temperature or `top_k` did to it. One view is recorded, and the record names it. llama.cpp profiles refuse the setting until v0.4.0: its probabilities need a change to how it samples (`n_probs`), which must first be tested for its effect on the reply, and a reply that quietly lacks the numbers you asked for would be a silent fallback.
  - The log-softmax runs on the CPU in float32. On MPS, the same row gave slightly different results (in about the 7th significant digit) depending on how many rows were processed together; on the CPU it didn't. From that test we infer that only the logits depend on the device, and a recompute that reduces the same way gets exactly the same numbers from the same logits. Measured on SmolLM2 on CPU and MPS: the logged probabilities equal a step-by-step recompute with the cache exactly, and turning `logprobs` on doesn't change the reply.
  - Cost: the scores stay in memory until the reply is done, about 0.4 MB per token for a 100,000-token vocabulary, and the log grows by about 0.8 MB per 4,000-token reply at `logprobs = 5` (estimated from the width of the stored numbers).
- **The recompute check gives facts and numbers, not a verdict.** A notebook that recomputes an export's probabilities under other conditions gets other numbers, so `clarity_chat.recompute.check` lists each condition that differs and measures the largest gap and the changed top choices. It recomputes step by step with the cache, as generation ran, because in bf16 one pass over the whole sequence differs even under the same conditions: the sums run in another order and round differently (one 60-token SmolLM2 run, CPU, bf16: a gap of 0.139 and 2 changed choices, against 0 step by step; in another such run, 0.177 in bf16 and 0.0000145 with no changed choices in float32). It sets no threshold: what matters depends on the experiment.
- **Steering adds a vector made elsewhere, on Transformers first.** A profile names a control-vector file (llama.cpp's format, as repeng writes it) by path and sha256; Clarity applies it and never makes one. For each call, a forward hook adds the scale times `direction.N` to the output of block N at every position, and comes off when the call ends, however it ends. The vector, scale and blocks are in the call's request, so a replay applies the same ones, and a check confirms every forward pass of the call ran each steered block once. On MPS, Transformers decides to stop one step late and runs one more pass, which it then undoes, so a call can have one more pass than reply tokens; that pass is steered too.
  - The hooks run before the ones Transformers uses to record `hidden_states`, so a notebook reading activations sees the steered values.
  - The scaled vector is computed in float32 and cast once to bf16, as repeng computes it (casting first and scaling in bf16 rounds differently), then added in bf16. llama.cpp adds it in float32 to float32 hidden states: the same sums, not the same numbers. How far apart they land isn't measured.
  - Block 0 is refused, only because llama.cpp has no vector slot for block 0 and its loader refuses `direction.0`; Clarity keeps a vector usable on both. The last block can be steered, but repeng, which steers it by default, reads that block's output after the model's final normalization (Transformers' last `hidden_states` entry), so such a vector is trained on a different quantity than the one it's added to. The guide says to leave it out.
  - Steering runs on model types `llama` and `olmo3`, whose blocks Clarity has checked return a plain tensor from `model.model.layers`. Other types are refused until checked.
  - Clarity reads the file with its own small reader rather than the `gguf` package: it parses the bytes it hashed, in one read, and every malformed file gets a message saying what was wrong.
- Only one model is ever in memory. If a new model fails to start, the previous one is started again, so a bad `/new` or `/profile` leaves you where you were.

## Conversations and runs

**Branches never overwrite.** `/regen`, `/edit` and `/undo` start new branches, and every turn stays in the log with its parent. A `/regen` uses the highest seed used for the message plus 1, which for replies made only by `/regen` is the first reply's seed plus k: with the same seed and settings it would only repeat a reply, and checking that is `/replay`'s job. `/regen same` reuses the seeds on purpose, under changed settings, so the change is the only difference between two replies with the same seed.

**`/compare` groups replies by their logged settings, not their requests.** Each backend logs its own request body, so two calls with the same settings can have requests with different keys. The settings table uses the same names on every backend, with the seed left out, since the seed is what differs between replies made to compare.

**Replay re-sends the logged request, not a rebuilt one.** It sends the logged token IDs and request body, so a later change to how Clarity builds requests can't change a replay. It then lists every recorded fact that differs from the original call, so a different output comes with its likely causes.
- Cost: replay can't show whether a later fix to how requests are built changes a reply; that takes a new call.

**Resume continues a conversation as it was logged.** `clarity chat --resume LOG` uses the profile's logged text and files, its resolved settings, its `/set` values and the context size of its last call, not today's files. Then it says what differs: a changed profile file, settings today's files would resolve differently, the facts replay compares (server build, flags, machine, libraries), and whether the current reply's prompt still renders to the same token IDs with today's template and libraries. `/switch` runs that last check again for the reply it goes to. Some things can only be today's: the model's `models.toml` entry, `server.toml`'s flags other than the context size, the llama.cpp build and the libraries. Each is compared with the log and listed when it differs.
- The conversation continues where the latest `turn` or `head` record says; a message whose call never finished means the turn before it. v0.1.0 logged neither `/undo` nor a failed call's move back, so for its logs that place is a guess, and Clarity says so.
- Cost: a resumed chat keeps old values even when you'd rather use today's. `/set` or a new conversation changes them, visibly.
- A batch run's `calls.jsonl` can't be resumed: each of its calls is a conversation of its own.

**Batch runs are blind, checked first and kept apart.**
- Before any model loads, every opening is rendered against every profile. Unequal `max_tokens` stops the run, because reply length would give a setup away. The cost: all profiles in a run share one `max_tokens`.
- Seeds come from the run, starting at 42 like the default seed, so two runs with the same profiles are comparable, and `--seeds N` shows a setup's spread rather than one sample.
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
- Per server start: a `llama-server` binary built from another commit than the pinned one (a checkout at another commit is refused instead), uncommitted changes in the llama.cpp checkout, or inherited `GGML_*` or `LLAMA_*` variables (or `PYTORCH_*` and `TORCH_*` on the Transformers backend), whose values are logged unless the name looks like a secret.

There is no general "results may vary" notice. Replay flags the one case that needs attention: output that differs although nothing that could explain it changed.

**Not attempted:** identical output across machines, builds or backends; llama.cpp's experimental batch-invariant kernels, which fix a variation Clarity's single-slot setup doesn't have; and a determinism guarantee for the Transformers backend on Apple GPUs, which PyTorch doesn't provide. The live tests check that case instead.

## The web layer

**The page is a view of the same command table.** Every action and every read goes through the commands the terminal and `--json` use, so the web interface can't do or show something a script can't. One worker thread owns the session: commands queue and run in order, as with `--json`; the ones the table lets run during another (`state`, `cancel`, `chats`, `profiles`, and `settings` during a reply) don't wait.

**The server trusts nothing but its own page.** It listens on `127.0.0.1` only and checks the exact Host on every request, which stops DNS rebinding. The threats it's built against are a page on another site, or on another `127.0.0.1` port, in the same browser; a rebinding page; and another user on the same computer.
- **A secret per start, in the link's fragment, sent as a header.** The fragment never reaches a server, a log or a `Referer`. A page from another origin can't read it, and can't send an `Authorization` header to `127.0.0.1` without a CORS preflight, which the server refuses, so a cross-site request fails before anything else is read. A cookie would be simpler for the page but worse: browsers send it to every port on `127.0.0.1`, so any local web page could use it.
- **Fetch metadata and Origin** are checked as well, as a second layer, and every response carries a strict content security policy (scripts, styles and fonts from the page's own origin only; no framing).
- **Untrusted text is escaped.** Replies, notes, labels, titles, descriptions and error sentences go into the page through autoescaped templates or as plain text; the page's script has one way to put server HTML into the page and none to build HTML from text, and the content security policy's Trusted Types make the browser refuse any other way, where it supports them.
- **The port is Clarity's before the link exists.** `clarity ui` binds and listens on it before printing or opening the link, so no other program can take the port in between and receive the secret.
- **Its limits:** anyone who can read the terminal, the browser's stored session or this user's files can find the secret while Clarity runs; another user on the same computer can reach the port but not the secret. For now Clarity assumes one user per computer.

## Not in scope

Left out from the start: tools or agents; metrics or LLM judges; multi-user features; an Ollama backend; llama-server's router mode; MLflow export.

Also left out:
- Running two models side by side. Two 7B models in bf16 need about 29 GB, more than a 32 GB laptop can spare.
- Speculative decoding, LoRA and llama.cpp's control vectors, which `server.toml` refuses. Steering runs on the Transformers backend.
- Live tests on Windows and Linux. Offline CI covers them; the live tests have run on macOS (Metal) only.

## Changed decisions

When a choice above changes, the old one, the new one and the reason go here, newest first.

| Date | Was | Now | Why |
|---|---|---|---|
| 2026-11-14 | The default seed was `"random"`, and a batch run's seeds started at 333. | The default seed is 42, and a run's seeds start at 42. | With a random default, sending the same message twice gave two replies, which looked like a difference worth studying. A fixed seed makes a chat repeat until you change something; `"random"` is still a value you can choose. |
| 2026-11-11 | Steering vectors were left out: `server.toml` refuses llama.cpp's control-vector flags. | A pinned vector made by another tool is added by a hook on the Transformers backend. llama.cpp's control-vector route stays refused. | The owner's experiments need a chat with a steered model that can be replayed. A hook on the in-process backend can be applied, logged, checked and replayed exactly. |
| 2026-11-11 | A llama-server built from another commit was a warning, as the brief asked. | It's an error at start. | The binary is what runs. A checkout at another commit was already an error, so a binary built from one shouldn't pass with a warning. |
| 2026-11-11 | `server.toml` refused a list of known-bad flags. | It accepts only the flags it ships with; any other is refused. | A list of bad flags misses new flags and other names for the same flag: `gpu-layers`, `override-tensor` and others got through. A new flag now joins on purpose, with a test. |
| 2026-11-10 | Clarity never edits any file you write. | Shipped files stay read-only; your own files live in `local/`, and Clarity writes there only visibly. | The web interface needs to save profiles and settings you make in it. Keeping them in `local/` keeps a `git pull` from touching them, and keeps the shipped files a fixed reference. |
| 2026-11-10 | A web interface and retrieval from documents were out of scope. | Both are planned: the web interface in v0.3.0, documents in v0.3.1 to v0.3.3 ([Roadmap](../ROADMAP.md)). | Making local models easy to use for people who don't use a terminal, without giving up exactness. Every feature still works from the terminal. |
