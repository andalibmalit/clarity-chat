# Clarity

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.24068236.svg)](https://doi.org/10.5281/zenodo.24068236)

**Clarity**: a simple chat client for transparent, reproducible experiments with local LLMs.

**What you see is exactly what the model got, and every call can be replayed.**

## The problem

When you prompt a local model through a server or chat app, much of what shapes a reply is hidden:

- which chat template ran, and how the text was tokenized
- whether a beginning-of-sequence token was added
- sampler defaults from the server or from the model file, and samplers that were skipped silently
- prompt caching, and context shift dropping tokens
- flags picked up from environment variables or config files

So when two replies differ, you can't tell whether your prompt caused it, and you can't rerun the call to find out.

Clarity removes that hidden layer, without making experiments harder to run:

- it builds the prompt and its token IDs itself
- it sends every sampling value and checks that the server used them
- it logs everything that affects the output
- it can replay any call

## Reproducibility

The same call on the same machine and llama.cpp build gives the same output; live tests check this, and the same holds for the Transformers backend on MPS and CPU. A different build, GPU, macOS version or batch setting can change it, because GPU arithmetic depends on the order of operations. Clarity logs all of these, and `clarity replay` (or `/replay` in a chat) re-sends a call and lists any that differ. See [`docs/design.md`](docs/design.md#determinism-and-replay).

## Status

Built so far: fetching and converting a model, the managed llama-server, chat mode (the model's chat template) and text mode (a plain document), the chat loop with branches (`/regen`, `/edit`, `/undo`), marks and notes, replay, batch runs for blind comparisons, downloading a ready-made GGUF as an alternative to converting, and a Transformers backend that runs the same weights in-process (`uv sync --extra hf`; MPS by default).

## Quick start

Needs [uv](https://docs.astral.sh/uv/), CMake and a C++ compiler.

```
uv sync
git clone https://github.com/ggml-org/llama.cpp vendor/llama.cpp
git -C vendor/llama.cpp checkout <llama_cpp_commit from server.toml>
cmake -S vendor/llama.cpp -B vendor/llama.cpp/build -DCMAKE_BUILD_TYPE=Release -DLLAMA_CURL=OFF
cmake --build vendor/llama.cpp/build --config Release -j 8 --target llama-server

uv run clarity fetch smollm2-135m-instruct   # download at the pinned commit, convert to GGUF
uv run clarity chat --profile smol-text      # chat; /help lists the commands
uv run clarity run --openings FILE --profiles smol-chat smol-text   # blind test
uv run clarity replay logs/CONVERSATION.jsonl CALL_ID              # re-send a call, compare
uv run clarity mark logs/CONVERSATION.jsonl CALL_ID LABEL          # label a past reply
uv run clarity note logs/CONVERSATION.jsonl CALL_ID "TEXT"         # add a note to a past reply
```

## Configuration

- `models.toml`: models, each at a pinned commit: converted from the official weights, or a ready-made GGUF checked by sha256.
- `server.toml`: every llama-server flag.
- `defaults.toml`: the neutral value of every setting.
- `profiles/*.toml`: one experimental setup each (model, mode, prompt pieces and settings).

Add `use_recommended = true` to a profile to use the model's own recommended sampling settings for anything the profile doesn't set.

The tool never edits these files; what it resolves goes to `models.lock`.

A batch run (`clarity run`) reads openings from a text file, separated by lines that are exactly `---`. It writes `blind.md` for raters, `key.json` (which reply came from which profile) and the call log to a new `runs/<run-id>/` folder.

## From profile to token IDs

1. **Load the profile.** Unknown keys and setting names are errors. Each setting comes from the profile, then (with `use_recommended`) the model's `generation_config.json`, then `defaults.toml`, and its origin is recorded. Prompt files are read byte for byte.
2. **Render the text.** Chat mode runs the chat template from the pinned tokenizer repo on the system message, priming and conversation. Text mode writes a plain document: header, priming, then `User: …` and `Assistant: …` lines, ending with `Assistant:`.
3. **Tokenize.** The model's own tokenizer, at its pinned commit, turns the text into token IDs without adding special tokens. Text mode then prepends BOS if the tokenizer config says so; chat mode leaves that to the template.
4. **Check and send.** The prompt plus `max_tokens` must fit the context. The IDs go to the backend with every sampling value; llama-server applies no template and does no tokenizing.
5. **Verify.** After the call, the server's own decoding of the IDs must equal the prompt text, and every value it reports must equal what was sent.

`/prompt` shows the text and IDs of the current reply's call; `/prompt repr` makes whitespace visible.

## What is logged, and why

Each conversation is one JSONL file in `logs/`, append-only. Every call writes a record, even when it fails or you cancel it, holding everything needed to reproduce it or explain a difference:

- the profile's full text and files, and every setting with its origin
- the prompt text, its token IDs and the BOS decision; the template's repo, commit and hash
- the request sent and the server's response, including the values it echoed and what wasn't applied
- the output text and token IDs, stop type, timings and any failed check
- the model file's sha256, the client's commit, and the library versions that render prompts

Each server start is logged once per file: its command line, `/props`, build, environment variables and machine. Turns, marks, notes and replays are their own records. Logs hold private conversations, so `logs/` and `runs/` are gitignored. Records name their schema version, and readers ignore fields they don't know.

## Adding a backend

A backend turns token IDs plus settings into a reply. Write a module in `src/clarity_chat/backends/` with a class that has `start`, `complete`, `close`, `describe`, `n_ctx` and `server_record` (see the `Backend` protocol in `backends/__init__.py`), then add its name to `BACKENDS` in `profiles.py` and to `Session.backend_for`.

- Map every Clarity setting to the backend's own, and set every other output-affecting value to a fixed neutral one.
- List any setting it can't apply in `not_applied`, with the reason; never drop one silently.
- Return a `Result`: output token IDs including stop-string and end tokens, text without the stop string.
- For replay, send a logged request exactly as logged, or raise `BackendError`.

`backends/llama_cpp.py` and `backends/transformers_hf.py` are the two examples.

## More

- [`docs/design.md`](docs/design.md): design choices, tradeoffs, usage patterns and what is out of scope.
- [`docs/brief.md`](docs/brief.md): the spec.
- [`docs/plan.md`](docs/plan.md): the build plan and status.

## License and citation

Clarity is licensed under Apache 2.0; see [`LICENSE`](LICENSE) and [`NOTICE`](NOTICE). The OLMo 3 tokenizer files in `tests/fixtures/` come from AllenAI under the same license.

To cite Clarity, use **Cite this repository** on its GitHub page, which reads [`CITATION.cff`](CITATION.cff). Each release is archived on Zenodo with its own DOI; [10.5281/zenodo.24068236](https://doi.org/10.5281/zenodo.24068236) covers all versions. For reproducible work, cite the DOI of the release you used.
