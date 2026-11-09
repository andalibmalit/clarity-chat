# Contributing

## Set up

Follow [Getting started](docs/getting-started.md) to install Clarity and build llama.cpp. For the Transformers backend, also run `uv sync --extra hf`.

## Test

| Command | Needs | Covers |
|---|---|---|
| `uv run pytest` | nothing: no network, model or server | everything that can run offline; this is what CI runs on Linux, macOS and Windows |
| `uv run pytest -m live` | the built llama-server and the fetched SmolLM2; downloads a 145 MB GGUF once | real calls, the context boundary, replay, determinism and batch runs |
| `uv run pytest -m olmo` | both OLMo 3 models fetched (about 58 GB) | chat templates, end tokens and determinism on OLMo 3 |
| `uv run pytest -m hf` | the `hf` extra and the fetched SmolLM2 (one test also uses OLMo 3 Instruct, if fetched) | the Transformers backend on MPS and CPU, and that both backends get the same prompt token IDs |

Before every commit, run the checks CI runs:

```
uv run ruff check && uv run ruff format --check && uv run ty check && uv run pytest
```

Tests use made-up text. Never put anything from `logs/` or `runs/` in a test or an issue; they hold private conversations.

## Keep the docs current

Update the docs in the same commit as the change. Choices, tradeoffs and scope go in [`docs/design.md`](docs/design.md); keys, commands and record fields in [`docs/reference.md`](docs/reference.md). `tests/test_docs.py` fails if a profile key, setting, CLI command or chat command is missing from the reference, or a relative link is broken.

## Add a backend

A backend turns token IDs plus settings into a reply. Write a module in `src/clarity_chat/backends/` with a class that implements the `Backend` protocol in `backends/__init__.py`. Then add its name to `BACKENDS` in `profiles.py` and to `Session.backend_for`. Other code branches on the backend's name, such as `Session.weights_for`, which picks the files a backend loads; `grep -rn '"transformers"' src/` finds each place.

- Map every Clarity setting to the backend's own, and set every other output-affecting value to a fixed neutral one.
- List any setting it can't apply in `not_applied`, with the reason; never drop one silently.
- Return a `Result`: output token IDs including stop-string and end tokens, text without the stop string.
- For replay, send a logged request exactly as logged, or raise `BackendError`.

`backends/llama_cpp.py` and `backends/transformers_hf.py` are the two examples.
