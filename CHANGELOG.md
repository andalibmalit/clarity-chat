# Changelog

All notable changes to Clarity are listed here. The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versions follow [Semantic Versioning](https://semver.org/).

Two headings are Clarity's own, used when they apply:
- **Migration:** files on disk or the log format changed, and what to do about it.
- **Output may change:** a pinned version or a default moved, so the same call can give a different reply. Replay lists such differences.

## [Unreleased]

### Fixed
- `server.toml` now accepts only a fixed list of flags, the ones it ships with, and refuses any other. Before, it refused a list of known-bad flags, so flags such as `gpu-layers` (another name for `n-gpu-layers`), `grp-attn-n`, `override-tensor`, `special` and `docker-repo` were passed to llama-server.
- `server.toml` now refuses every adapter and control-vector flag. Before, only `lora` and `control-vector` were refused, and `lora-scaled`, `lora-init-without-apply`, `control-vector-scaled` and `control-vector-layer-range` were passed to llama-server. A reply made with one of them has the flag in its `server_start` record.

### Changed
- Docs: the README, the design doc and the citation abstract now state the limits of their claims: replay covers completed calls, the same-output tests ran on two model families on one Apple Silicon Mac, and "No silent fallbacks" names its two flagged cases. The README, the package description and the citation title now share one tagline.

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
