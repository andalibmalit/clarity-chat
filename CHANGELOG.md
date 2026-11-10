# Changelog

All notable changes to Clarity are listed here. The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versions follow [Semantic Versioning](https://semver.org/).

Two headings are Clarity's own, used when they apply:
- **Migration:** files on disk or the log format changed, and what to do about it.
- **Output may change:** a pinned version or a default moved, so the same call can give a different reply. Replay lists such differences.

## [Unreleased]

### Security
- llama-server now needs a random key, new at each start, so a web page in your browser can't send it requests. Its `/slots` endpoint and built-in web UI are off. The key is never logged.

### Added
- The tokens that end a reply, as llama.cpp read them from the model file, are logged with each server start and shown in `/settings`.
- Each server start logs how many pieces llama.cpp split the compute graph into, and replay compares it. More than one piece can mean part of the work runs on the CPU.
- The status line after each reply shows how much of the context the conversation now uses, such as "context 12,400 of 32,768".
- A `template_kwargs` key that the chat template never reads, such as a misspelt one, is now a warning when the conversation starts, and is logged. Before, it changed nothing silently.
- `/say TEXT` and `/cancel` chat commands, for scripts. All chat commands now come from one table, which `/help` lists.
- If Clarity is killed or crashes, the next start stops the llama-server it left running, after checking that the server's port answers with its key and model. SIGTERM and a closed terminal now stop the server too.
- A log line cut off by a crash or a full disk no longer makes the log unreadable. The next write ends it and adds a `torn_line` record naming it, and Clarity says so; the cut line is kept as it was.
- Only one `chat`, `run` or `replay` runs in a project at a time; a second one stops and says which process is running.
- Your own profiles go in `local/profiles/`, which git ignores, next to the shipped ones in `profiles/`. A name in both folders is an error.
- On a Mac, a context size whose memory doesn't fit on the GPU is refused, with the figures. Every server start logs its memory use.
- `ctx_size` in a profile sets its context size; profiles without it keep `ctx-size` from `server.toml`. Replay runs a call at the size it was logged with, and a size above the model's training context is refused instead of quietly capped.
- A call can be cancelled from any thread, which the web interface and `clarity chat --json` will use. Each cancelled call logs `cancelled_during`: `prompt` or `reply`.
- `clarity mark LOG CALL_ID LABEL` and `clarity note LOG CALL_ID TEXT` label or annotate any logged reply after the chat, without starting a model. Marks and notes now record whether they were made live or in review.
- The Transformers backend sets `attn_implementation = "sdpa"` explicitly, checks it after loading, and logs it.
- User docs: getting started, guides, reference and contributing.
- `CHANGELOG.md`, `SECURITY.md`, `CONTRIBUTORS.md`, `CODE_OF_CONDUCT.md` and `ROADMAP.md`; a release checklist and a Credit section in `CONTRIBUTING.md`.

### Changed
- A call cancelled before its first token no longer adds an empty reply. It's logged like a failed call, and your message comes back in the input.

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

[Unreleased]: https://github.com/andalibmalit/clarity-chat/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/andalibmalit/clarity-chat/releases/tag/v0.1.0
