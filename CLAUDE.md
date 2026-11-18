# CLAUDE.md

Clarity: a chat client for exact, replayable prompting experiments with local LLMs. A reply is shaped by much that servers and chat apps hide (the chat template, tokenization, BOS, sampler defaults, prompt caching, context shift, flags from the environment), so a difference between two replies may not come from the prompt. Clarity builds the token IDs itself, sends every value, checks the server used them, logs every setting and environment fact known to affect the output, and can replay any completed call.

Where things are: choices, tradeoffs and scope in `docs/design.md`; keys, commands and record fields in `docs/reference.md`; test tiers and the release checklist in `CONTRIBUTING.md`. `notes/` is local only (gitignored): the release plans in `notes/plans/`, the sprint's rules and status in `notes/sprint.md`, research in `notes/research/`.

## Commands

```
uv sync                       # install (add --extra hf for the Transformers backend)
uv run clarity --help         # the CLI; every command is in docs/reference.md
uv run pytest                 # offline tests: no network, model or server
uv run pytest -m live         # SmolLM2 via the managed llama-server; other tiers: CONTRIBUTING.md's Test table
uv run ruff check && uv run ruff format --check && uv run ty check
```

Build llama.cpp at the commit in `server.toml` with Getting started's build lines (`docs/getting-started.md`).

## Principle
What you see is exactly what the model got, and every completed call can be replayed.
No silent fallbacks: if something can't be done exactly as configured, stop with a clear error. Two cases are flagged instead, on the record and on screen: a check that fails after a call, and a setting the Transformers backend can't apply.
Clarity should get out of the way: being exact must not make common experiments awkward. Prefer a small, explicit, logged shortcut (like `use_recommended = true`) to extra ceremony.
The backend does what the UI leads people to expect. A difference between them is a silent fallback: show it.
The UI doesn't show terminal equivalents of its controls; command-line usage belongs in the docs and `help`.
Teaching text, anything that explains how models or settings work, is checked against authoritative sources before it ships. Each explanation lives in one place, the settings table, which the UI, `help` and the docs reuse.
Clarity writes files only visibly. Shipped files stay read-only; the user's own files live in `local/`.

## How to work
- Never open files in `logs/`, `runs/` or `local/documents/` unless I ask; they hold private conversations and documents. Tests use made-up text.
- Each release follows its approved plan in `notes/plans/`. If the build departs from the plan, update the plan in the same step and say so. Reviews and the release process: `CONTRIBUTING.md`.
- State your assumptions in each step report, and ask before decisions that are hard to undo. Run the checks above before every commit, the failing-test commit included, and quote their output in the report.
- For behaviour the plan fixes (record fields, refusals, command results and events, replay, resume, reading old logs), write the test first and see it fail on its own assertion; the report quotes the failure. For tests written with their code, stash the source change, run the new tests, and quote the ones that fail. A bug fix starts with a failing test, committed before the fix; a fix commit that changes a test from that commit says what changed and why.
- An offline test never skips in CI. A test that needs a model or torch goes on a live tier.
- Before `-m olmo` or `-m llama` (each loads 15–16 GB), check that no model is loaded and about 16 GB is free; otherwise ask.
- A change to the llama.cpp commit, `tokenizers`, `transformers` or `jinja2` can change output: give it its own commit and a changelog line, and ask the owner to run the live tests.
- One commit does one plan item, with its tests, its docs and its `CHANGELOG.md` line under Unreleased.
- Docs are for human readers, in Google developer-docs style: lead with the point, use short sentences and plain words, and use lists only for parallel items. Say each thing once. Keep caveats and sources; cut padding.
- Use the same names everywhere: profile, conversation, turn, call, mark and note mean the same thing in code, profiles, logs and commands.
- If you make a mistake an earlier session also made, propose one line for this file. Check it against the bar this file keeps (true on every task, not learnable from the code), and say whether a hook or deny rule would serve better, or which line it replaces.

## Python style
- PEP 8 and the Zen of Python are defaults, not hard rules: readability is the goal, so break one when following it would read worse. ruff and ty (configured in `pyproject.toml`) override PEP 8; match the surrounding code. Let `ruff format` wrap lines; don't shorten a good name to fit one. Keep names that mirror an external API (llama-server's `n_predict`, `tokens_evaluated`), and don't restyle code you aren't otherwise changing.
- Put more care into clear design than into passing the linter. Explain a choice by its actual tradeoff, not by quoting the Zen of Python.

## Sandbox notes
- `vendor/llama.cpp` can't be cloned or updated from inside the sandbox; the owner runs its git commands.
- Delete files with Python's `shutil`; `rm -rf` is denied.
- Run a live tier (`uv run pytest -m live -v`, `-m olmo`, `-m hf`) on its own: no pipe, `cd` or `&&`. Only then does its `excludedCommands` entry take it out of the sandbox, which hides the GPU.
