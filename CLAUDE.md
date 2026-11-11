# CLAUDE.md

Clarity: a chat client for exact, replayable prompting experiments with local LLMs. Design choices, tradeoffs and scope: `docs/design.md`. Local-only notes in `notes/` (gitignored, so not in clones or CI): the spec `notes/brief.md`, the build plan and current status `notes/plan.md`, and research in `notes/research/`. `docs/` holds only docs for other readers.

## The problem it solves
When you prompt a local model through a server or chat app, much of what shapes a reply is hidden: which chat template ran and how the text was tokenized, whether a BOS token was added, sampler defaults from the server or the model file, samplers skipped silently, prompt caching, context shift dropping tokens, flags picked up from environment variables or config files. A difference between two replies can come from any of these rather than from your prompt, and you can't rerun the call to find out. Clarity takes over that layer: it builds the token IDs itself, sends every value, checks the server used them, logs every setting and environment fact known to affect the output, and can replay any completed call.

## Commands

```
uv sync                                   # install (add --extra hf for the Transformers backend)
uv run clarity fetch NAME                 # download a model from models.toml, convert it, write models.lock
uv run clarity verify [NAME]              # re-hash GGUF files against models.lock
uv run clarity chat --profile NAME        # chat with a profile from profiles/ or local/profiles/
uv run clarity chat --resume ID           # continue a conversation (IDs from `clarity chats`)
uv run clarity chat --json [--profile NAME]  # commands as JSON lines on stdin, events out
uv run clarity chats                      # list conversations in logs/
uv run clarity run --openings FILE --profiles A B [--seeds N]   # blind test into runs/RUN_ID/
uv run clarity replay LOG CALL_ID [--logged-flags]                # re-send a call and compare
uv run clarity mark LOG CALL_ID [LABEL] [--remove]                # label a logged reply, or remove a label (note: TEXT)
uv run pytest                             # offline tests (no network, model or server)
uv run pytest -m live                     # SmolLM2 via the managed llama-server
uv run pytest -m olmo                     # OLMo 3 (needs both OLMo models fetched)
uv run pytest -m measure -v -s            # timings and memory figures behind design choices
uv run pytest -m embed -v -s              # embedding determinism on bge-small (decides v0.3.3)
uv run ruff check && uv run ruff format --check && uv run ty check
```

Build llama.cpp (checkout at the commit in `server.toml`):

```
cmake -S vendor/llama.cpp -B vendor/llama.cpp/build -DCMAKE_BUILD_TYPE=Release -DLLAMA_CURL=OFF
cmake --build vendor/llama.cpp/build --config Release -j 8 --target llama-server
```

## Principle
What you see is exactly what the model got, and every completed call can be replayed.
No silent fallbacks: if something can't be done exactly as configured, stop with a clear error. Two cases are flagged instead, on the record and on screen: a check that fails after a call, and a setting the Transformers backend can't apply.
Clarity should get out of the way: being exact must not make common experiments awkward. Prefer a small, explicit, logged shortcut (like `use_recommended = true`) to extra ceremony.
The backend does what the UI leads people to expect. A difference between them is a silent fallback: show it.
No terminal commands in the UI. Command-line usage belongs in the docs and `help`.
Teaching text, anything that explains how models or settings work, is checked against authoritative sources before it ships. Each explanation lives in one place, the settings table, which the UI, `help` and the docs reuse.
Clarity writes files only visibly. Shipped files stay read-only; the user's own files live in `local/`.

## How to work
- State your assumptions in each step report. Ask before decisions that are hard to undo.
- A step is done when its tests pass. Write tests with the change, and run ruff, ty and pytest before every commit.
  Show the command and its output in the step report.
- For behaviour the plan fixes (record fields, refusals, command results and events, replay, resume, reading old logs), write the test first and see it fail; the step report quotes that failure.
- An offline test never skips in CI. A test that needs a model or torch goes on a live tier, and the plan marks its behaviour "live only".
- A bug fix starts with a failing test, committed before the fix.
- Each release follows its approved plan in `notes/plans/`. If the build departs from the plan, update the plan in the same step and say so.
- A release is done when `CONTRIBUTING.md`'s release checklist is complete. I may run the live tests while building, to catch my own bugs, and ask before `-m olmo` (it loads 15 GB). The owner runs them again for the release checklist, pushes and tags.
- A change to the llama.cpp commit, `tokenizers`, `transformers` or `jinja2` can change output: give it its own commit and a changelog line, and ask the owner to run the live tests.
- Add user-visible changes to `CHANGELOG.md` under Unreleased in the same commit.
- If you make a mistake that an earlier session also made, propose one line for this file.
- One commit does one plan item, with its tests and docs. A release is a closed list the owner can use and judge in one sitting; scope sets its size, not a line count.
- The plan names review checkpoints: after each group that later items build on, and the whole branch before the release is used. Stop for an extra review if the diff since the last one passes about 400 source lines, or an item lands at over 1.5 times its estimate. These are triggers, not budgets.
- Reviews check the diff against the plan and look for bloat: code no item asked for, abstractions with one user, unrequested options, tests for cases that can't happen. Skip style. Prefer findings that delete code.
- Keep the docs current as you go: update the docs in the same commit as a change to a choice, tradeoff, scope or usage: `docs/design.md` for choices and tradeoffs, `docs/reference.md` for keys, commands and record fields, `docs/guides.md` and `docs/getting-started.md` for usage, and the README. `tests/test_docs.py` catches some drift, not all. Write for human readers, in Google developer-docs style: lead with the point, use short sentences and plain words, and use lists only for parallel items. Say each thing once. Keep caveats and sources; cut padding.
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
