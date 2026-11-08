# Clarity

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

The same call on the same machine and llama.cpp build gives the same output; live tests check this. A different build, GPU, macOS version or batch setting can change it, because GPU arithmetic depends on the order of operations. Clarity logs all of these, and `clarity replay` (or `/replay` in a chat) re-sends a call and lists any that differ. See [`docs/design.md`](docs/design.md#determinism-and-replay).

## Status

Built so far: fetching and converting a model, the managed llama-server, chat mode (the model's chat template) and text mode (a plain document), the chat loop with branches (`/regen`, `/edit`, `/undo`), marks and notes, replay, and batch runs for blind comparisons. Coming next: downloading a ready-made GGUF, and a Transformers backend.

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
```

## Configuration

- `models.toml`: models, each at a pinned commit.
- `server.toml`: every llama-server flag.
- `defaults.toml`: the neutral value of every setting.
- `profiles/*.toml`: one experimental setup each (model, mode, prompt pieces and settings).

Add `use_recommended = true` to a profile to use the model's own recommended sampling settings for anything the profile doesn't set.

The tool never edits these files; what it resolves goes to `models.lock`.

A batch run (`clarity run`) reads openings from a text file, separated by lines that are exactly `---`. It writes `blind.md` for raters, `key.json` (which reply came from which profile) and the call log to a new `runs/<run-id>/` folder.

## More

- [`docs/design.md`](docs/design.md): design choices, tradeoffs, usage patterns and what is out of scope.
- [`docs/brief.md`](docs/brief.md): the spec.
- [`docs/plan.md`](docs/plan.md): the build plan and status.
