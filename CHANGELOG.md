# Changelog

All notable changes to Clarity are listed here. The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versions follow [Semantic Versioning](https://semver.org/).

Two headings are Clarity's own, used when they apply:
- **Migration:** files on disk or the log format changed, and what to do about it.
- **Output may change:** a pinned version or a default moved, so the same call can give a different reply. Replay lists such differences.

## [Unreleased]

### Security
- `clarity replay --logged-flags` passed a log's flags to llama-server unchecked, and Clarity's own `--host 127.0.0.1`, `--no-slots` and `--no-webui` came before them, where a later flag wins. A shared or hand-edited log could make the replayed server listen on the network, re-enable `/slots` and the web UI, add a key, or turn on its shell tool. Logged flags now go through the allowed-list checks, and Clarity's own flags come last.

### Added
- Each call records `message_spans`: where each message, and the text the chat template or profile wrote around it, sits in the prompt, in characters and in tokens. A template that rewrites a message gets none for that call, with the reason.
- Token probabilities on the Transformers backend: the `logprobs` setting (0 to 20, off by default) records each reply token's log-probability and its N likeliest alternatives, from the model's raw scores before any sampler. `/probs` summarizes the current reply's least likely tokens. A llama.cpp profile refuses the setting until v0.4.0.
- `/regen N` makes N more replies to your last message, each a new branch with its own logged seed. With `--json`, a `reply_start` event (`k`, `n`, `seed`) comes before each.

### Fixed
- With `--json`, an unexpected error in a command (a bug, or a full disk) ended the session with a traceback and no `error` event. It now sends an `error` event naming the command and the error, quits, and exits with status 1. A line nested too deeply to parse stopped Clarity reading input; it's now an `InputError` like any other line that isn't a command.
- Replay and resume of a call whose log lacks its `server_start` record (a trimmed or hand-edited log) compared the wrong backend's facts and reported empty ones as differences, which could hide an unexplained difference. The call's own backend now decides, and what the log lacks is listed as not recorded.
- A Transformers replay judged its stop type, and recorded `eos_token_id`, from the end tokens in today's model files rather than the logged ones it was generated with. A model revision that changed only `generation_config.json` made an identical replay report `stop_type: none`.
- `/say` in the terminal sent its text with the spaces at its start and end removed, while a plain line and `--json` sent them; now all three send the text exactly. Through `--json`, spaces around another command's arguments made `/set` name an empty setting and `/mark` store a label `/unmark` couldn't find; they're now ignored, as in the terminal.
- If llama-server had to restart for `/replay` (it had stopped, or its model file was fetched again) and the restart failed, the next message crashed the chat. Now the conversation's model starts again, or the chat says to start a new conversation.
- In the terminal, Ctrl-C during a call cancels it wherever the call is. Before, it cancelled only while the reply streamed; at any other moment (building the prompt, the checks after the reply, computing token probabilities) it quit the chat, and the reply on screen could be missing from the log. During a command that can't be cancelled, such as a model start, Ctrl-C still stops Clarity.
- The Transformers backend refused models whose `generation_config.json` lists no end token (OLMo 3 base) or that ship none (Qwen3.5-9B). It now takes the tokenizer's end token, says so at start, and logs where the IDs came from (`server_start.eos_token_id_from`); `/settings` shows it.
- A model fetched again while its llama-server was running kept the old server, so calls ran the old file while the record named the new one. A server is now kept only if its file is unchanged (same inode, size and modification time); otherwise it restarts.
- `server.toml` now accepts only a fixed list of flags, the ones it ships with, and refuses any other. Before, it refused a list of known-bad flags, so flags such as `gpu-layers` (another name for `n-gpu-layers`), `grp-attn-n`, `override-tensor`, `special` and `docker-repo` were passed to llama-server.
- After a resume, each call recorded the model's `recommended` values from today's `models.lock`, while its settings came from the log. If a later fetch had changed them, a setting with origin `model` didn't match the `recommended` beside it. A resumed conversation's calls now record the logged values.
- A batch run (`clarity run`) logged a call's flags, such as `incomplete_output_ids`, but didn't show them, so a flagged sample went unnoticed unless someone read `calls.jsonl`. Each flagged call is now named on screen with its flags (never its reply).
- `server.toml` accepted `false` for `kv-unified`, `no-kv-unified` and `no-jinja`, which left the flag out so llama-server's default applied. It also accepted `threads` or `threads-batch` of 0 or less, which llama-server reads as every core. Both are now refused.
- `server.toml` now refuses every adapter and control-vector flag. Before, only `lora` and `control-vector` were refused, and `lora-scaled`, `lora-init-without-apply`, `control-vector-scaled` and `control-vector-layer-range` were passed to llama-server. A reply made with one of them has the flag in its `server_start` record.

### Changed
- A llama-server built from a commit other than the one `server.toml` pins is now an error at start, not a warning. The binary is what runs, and a checkout at another commit was already an error.
- After each call, Clarity checks that the llama-server process it started is still running, in place of a second `/props` request. `/props` is still checked once after start; nothing can change a running server's model or context.
- Replay and resume compare one more fact, `end_tokens`: the IDs a reply stops on. The backends can differ here (llama.cpp also stops on tokens it recognizes by name), and the difference is now shown rather than hidden. Logs from v0.1.0 made with llama.cpp list it under `not_recorded`; v0.2.0 llama.cpp logs and every Transformers log already hold it.
- `/history`, `/mark`, `/unmark`, `/note`, `clarity mark` and `clarity note` read only the records they need, not every call record, which made them 3 to 5 times faster on long logs in a measurement on made-up logs of 45 to 225 MB. A damaged line of another type no longer stops them; resume and replay still refuse it.
- Docs: the README, the design doc and the citation abstract now state the limits of their claims: replay covers completed calls, the same-output tests ran on two model families on one Apple Silicon Mac, and "No silent fallbacks" names its two flagged cases. The README, the package description and the citation title now share one tagline.

### Migration
- With `--json`, the `regen` command's result is now `{"replies": [{"call", "warnings"}, …]}`, also for a plain `/regen`, in place of one `call` and `warnings`. Read `replies[0]` where a script read the result itself.
- The check after each llama.cpp call is renamed: `checks.server_props` is now `checks.server_running` (with `exit_code` when the process has exited). `checks.not_echoed` is gone: it could never fail, and the same list follows from the logged request and `generation_settings`. A reader that looks for either key finds it missing in new logs.

## [0.2.0] - 2026-11-10

Live tests on an M2 Max (macOS), 2026-11-10: `-m live` 15 passed (including crash cleanup against a real llama-server), `-m olmo` 7 passed, `-m hf` 7 passed (greedy output identical on llama.cpp and Transformers), `-m embed` 5 passed: bge-small's embeddings were bit-identical back to back, after a restart, whatever else the request held, and on the CPU at a fixed thread count (and between 8 and 4 threads, 0 of 25 vectors differed).

### Security
- llama-server now needs a random key, new at each start, so a web page in your browser can't send it requests. Its `/slots` endpoint and built-in web UI are off. The key is never logged.

### Added
- The tokens that end a reply, as llama.cpp read them from the model file, are logged with each server start and shown in `/settings`.
- Each server start logs how many pieces llama.cpp split the compute graph into, and replay compares it. A change between a call and its replay can mean work moved between devices.
- The status line after each reply shows how much of the context the conversation now uses, such as "context 12,400 of 32,768".
- A `template_kwargs` key that the chat template never reads, such as a misspelt one, is now a warning when the conversation starts, and is logged. Before, it changed nothing silently.
- `/say TEXT` and `/cancel` chat commands, for scripts. All chat commands now come from one table, which `/help` lists.
- If Clarity is killed or crashes, the next start stops the llama-server it left running, after checking that the server's port answers with its key and model. On Linux and macOS, SIGTERM and a closed terminal now stop the server too.
- `clarity chat --json` runs every chat command from a script or another program: commands in as JSON lines, and events out, including every log record as it's written. `cancel` works while a reply is being written.
- `clarity chat --resume ID` and `/resume ID` continue a logged conversation with its profile, settings and context size as logged, and say what differs today, including whether the current prompt still renders to the same token IDs.
- `clarity chats` and `/chats` list your conversations, newest first, with a title: the first words of the first message, or one you set with `/title`. No model is asked for a title.
- `/set NAME VALUE` changes a setting for one chat, checked like a profile's value and logged with origin `chat`. `/settings` shows it.
- `/history` shows the conversation so far on the current branch, with each reply's marks and notes, so a script or the web interface can draw it without reading the log.
- `/branches` lists where a conversation branches, and `/switch TURN_ID` goes to another branch. Every move of the current turn other than a new reply (undo, switch, a failed call) is logged as a `head` record.
- `/unmark [LABEL]` and `clarity mark --remove` take a mark back. A new note on a reply replaces its note, and Clarity says so. `/mark` and `clarity mark` without a label mark the reply `interesting`. Marks and notes now have IDs; the log keeps every one and records removals and replacements.
- A log line cut off by a crash or a full disk no longer makes the log unreadable. The next write ends it and adds a `torn_line` record naming it, and Clarity says so; the cut line is kept as it was.
- Only one `chat`, `run` or `replay` runs in a project at a time; a second one stops and says which process is running.
- Your own profiles go in `local/profiles/`, which git ignores, next to the shipped ones in `profiles/`. A name in both folders is an error.
- On a Mac, a context size whose memory doesn't fit on the GPU is refused, with the figures. Every server start logs its memory use.
- `ctx_size` in a profile sets its context size; profiles without it keep `ctx-size` from `server.toml`. Replay runs a call at the size it was logged with, and a size above the model's training context is refused instead of quietly capped.
- A call can be cancelled from any thread, which the web interface and `clarity chat --json` will use. Each cancelled call logs `cancelled_during`: `prompt` or `reply`.
- `clarity mark LOG CALL_ID LABEL` and `clarity note LOG CALL_ID TEXT` label or annotate any logged reply after the chat, without starting a model. Marks and notes now record whether they were made live or in review.
- The Transformers backend sets `attn_implementation = "sdpa"` explicitly, checks it after loading, and logs it.
- `uv run pytest -m embed` tests whether llama-server's embeddings of bge-small, a new entry in `models.toml`, come back bit-identical.
- User docs: getting started, guides, reference and contributing.
- `CHANGELOG.md`, `SECURITY.md`, `CONTRIBUTORS.md`, `CODE_OF_CONDUCT.md` and `ROADMAP.md`; a release checklist and a Credit section in `CONTRIBUTING.md`.

### Changed
- A call cancelled before its first token no longer adds an empty reply. It's logged like a failed call, and your message comes back in the input.

### Migration
- `server.toml` can no longer set `api-key-file`, `slots`, `ui`, `webui` or their `no-` forms, or any `ui-` or `webui-` flag: Clarity sets the server key, `--no-slots` and `--no-webui` itself. A file that sets one stops with an error naming it.
- Logs keep `schema_version` 1. They gain record types (`head`, `unmark`, `settings`, `title`, `resume`, `torn_line`) and fields, which readers that ignore unknown records and keys skip. Clarity 0.1.0 reading a 0.2.0 log shows removed marks as still there and every note, including replaced ones.
- A profile name in both `profiles/` and `local/profiles/` is an error. Move your own profiles to `local/profiles/` so a `git pull` never touches them.
- No change in this release is known to change a reply: the new server flags don't affect sampling, and replay compares the server's flags, not its command line. Replay lists any difference it finds.

### Fixed
- On CUDA and Vulkan, a model whose weights went only to pinned system memory (`CUDA_Host`, `Vulkan_Host`) passed the GPU placement check. It's now an error, like any model that lands on the CPU.

## [0.1.0] - 2026-11-09

The first release.

### Added
- Chat mode, through the model's chat template, and text mode, where the model continues a plain document. Clarity renders the prompt and builds the token IDs itself.
- Every sampling value is sent on every call, and the server's echo is checked.
- Branches in a chat (`/regen`, `/edit`, `/undo`), and marks and notes on replies.
- An append-only JSONL log per conversation, and replay of any logged call with a list of what changed since.
- Blind batch runs for comparing profiles.
- Models at a pinned commit: converted from the official weights, or a ready-made GGUF checked by sha256.
- A llama-server that Clarity starts and checks itself, and a Transformers backend for the same weights.

[Unreleased]: https://github.com/andalibmalit/clarity-chat/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/andalibmalit/clarity-chat/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/andalibmalit/clarity-chat/releases/tag/v0.1.0
