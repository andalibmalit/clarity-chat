# CLAUDE.md

Clarity: a terminal chat client for reproducible prompting experiments with local LLMs. Spec: `docs/brief.md`. Design choices, tradeoffs and scope: `docs/design.md`. Build plan and current status: `docs/plan.md`.

## The problem it solves
When you prompt a local model through a server or chat app, much of what shapes a reply is hidden: which chat template ran and how the text was tokenized, whether a BOS token was added, sampler defaults from the server or the model file, samplers skipped silently, prompt caching, context shift dropping tokens, flags picked up from environment variables or config files. A difference between two replies can come from any of these rather than from your prompt, and you can't rerun the call to find out. Clarity removes that hidden layer: it builds the token IDs itself, sends every value, checks the server used them, logs everything and can replay any call.

## Commands

```
uv sync                                   # install (add --extra hf for the Transformers backend)
uv run clarity fetch NAME                 # download a model from models.toml, convert it, write models.lock
uv run clarity verify [NAME]              # re-hash GGUF files against models.lock
uv run clarity chat --profile NAME        # chat with a profile from profiles/
uv run pytest                             # offline tests (no network, model or server)
uv run pytest -m live                     # SmolLM2 via the managed llama-server
uv run pytest -m olmo                     # OLMo 3 (needs both OLMo models fetched)
uv run ruff check && uv run ruff format --check && uv run ty check
```

Build llama.cpp (checkout at the commit in `server.toml`):

```
cmake -S vendor/llama.cpp -B vendor/llama.cpp/build -DCMAKE_BUILD_TYPE=Release -DLLAMA_CURL=OFF
cmake --build vendor/llama.cpp/build --config Release -j 8 --target llama-server
```

## Principle
What you see is exactly what the model got, and every call can be replayed.
No silent fallbacks: if something can't be done exactly as configured, stop with a clear error.
Clarity should get out of the way: being exact must not make common experiments awkward. Prefer a small, explicit, logged shortcut (like `use_recommended = true`) to extra ceremony.

## How to work
- State your assumptions in each step report. Ask before decisions that are hard to undo.
- A step is done when its tests pass. Write tests with the change, and run ruff, ty and pytest before every commit.
- Keep `docs/design.md` and the README current when a choice, tradeoff or scope decision changes. Say each thing once, briefly.
- Use the same names everywhere: profile, conversation, turn, call, mark and note mean the same thing in code, profiles, logs and commands.
- Never open files in `logs/` or `runs/` unless I ask; they hold private conversations. Tests use made-up text.

## Python style
- Follow PEP 8 and the Zen of Python as defaults, not hard rules. Readability is the goal.
- The project's own conventions and configured tools (ruff for lint and format, ty for types) override PEP 8. Match the surrounding code.
- Break a rule when following it would make the code harder to read. Let `ruff format` do the wrapping; don't shorten a good name just to fit a line.
- Keep names that mirror an external API (e.g. llama-server's `n_predict`, `tokens_evaluated` in log records, or camelCase from a library you're wrapping).
- Don't reformat or restyle code you aren't otherwise changing. Keep diffs focused.
- Put more care into clear design than into cosmetic compliance. Passing the linter doesn't make code Pythonic.
- Don't quote the Zen to justify a choice. Explain the actual tradeoff.

## Sandbox notes
- `vendor/llama.cpp` can't be cloned or updated from inside the sandbox (it blocks writes to `.git/hooks`); the user runs those git commands.
- Delete files with Python's `shutil`; `rm -rf` is denied.
- Run `uv run pytest -m live -v` (or `-m olmo`, `-m hf`) on its own: no pipe, `cd` or `&&`. Only then does the `excludedCommands` entry take it out of the sandbox, which hides the GPU.
