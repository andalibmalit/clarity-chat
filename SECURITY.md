# Security

## Supported versions

Only the latest release gets security fixes.

## Report a vulnerability

Report it privately through **Report a vulnerability** on the repository's [Security tab](https://github.com/andalibmalit/clarity-chat/security), not in a public issue. Include the version or commit, your operating system, and the steps to reproduce. Use made-up text: never attach anything from your `logs/` or `runs/` folders.

Once a fix is released, the report is credited in `CHANGELOG.md` unless you'd rather not be named.

## What Clarity exposes

- **llama-server.** Clarity starts it on `127.0.0.1` only. It listens on a local port that other programs on the same machine can reach.
- **Logs.** `logs/` and `runs/` hold your conversations in plain text. Clarity never sends them anywhere, and git ignores both folders.
- **Downloads.** `clarity fetch` downloads model files from Hugging Face at a pinned commit and records their sha256 in `models.lock`; `clarity verify` re-checks them.

A way to read another user's conversations, reach llama-server from another machine, or make Clarity load files other than the pinned ones is a vulnerability. Please report it.
