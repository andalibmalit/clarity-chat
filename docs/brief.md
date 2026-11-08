# Brief: Clarity

**Clarity**: a simple chat client for transparent, reproducible experiments with local LLMs.

Paste this into Claude Code. This is a new, empty repo named `clarity`: build at the repo root. Names: the project and README title is **Clarity**; the distribution package is `clarity-chat` (`clarity` is taken on PyPI); the import package is `clarity_chat`; the command is `clarity`. The README opens with the tagline above and the principle below.

## Principle

**What you see is exactly what the model got, and every call can be replayed.** Every design choice serves this. Keep everything else small.

## Goal

A terminal chat client for systematic prompting experiments with local models, both base and instruct. It supports normal chat and raw "text continuation" (the model continues a plain document). It logs everything that affects the model's output, so any reply can be reproduced and compared. It should be simple, cross-platform, and easy to extend with new backends.

Reference: my earlier project https://github.com/andalibmalit/lord-gana-poem-gen solved related problems with Ollama. Read `docs/inference-settings.md` (what affects output, hidden defaults, how to control them) and skim `src/poemgen/records.py` before designing. Reuse ideas, not the package.

## Core design: the client owns the prompt, down to the token IDs

- The client builds the full prompt itself, **tokenizes it with the model's Hugging Face tokenizer (named in `models.toml`; text mode uses it too) with `add_special_tokens=False`, and sends token IDs**. No server ever applies a template or tokenizes.
- **Chat mode:** `AutoTokenizer.from_pretrained(repo, revision=...)` then `apply_chat_template(messages, tokenize=False, add_generation_prompt=True, **template_kwargs)`. Template kwargs (for example `enable_thinking`) are part of the profile.
- **Text mode:** a plain document (format under Profiles). Stop sequence `"\nUserName:"`.
- Decide BOS (beginning-of-sequence) handling explicitly, from the tokenizer's config, and log it. Both backends must receive identical token IDs.
- Store the model's output exactly as returned, plus its output token IDs. Parse anything from it (for example reasoning text) separately, and later.

## Backends

One small interface: `complete(token_ids: list[int], settings: dict, stop: list[str]) -> Result` (output text, output token IDs, token counts, stop reason, timings, raw response).

1. **llama.cpp server** (first). The client **starts and stops `llama-server` itself** from `server.toml`, so every server flag is known and logged.
   - Set every flag that affects output explicitly: model path, context size, GPU layers, flash attention on or off, KV cache types, batch and micro-batch sizes, `--fit off`, context shift off, one parallel slot.
   - Start it with every `LLAMA_ARG_*` variable removed from its environment. Stop with an error if its startup output shows it loaded a config file (llama.cpp reads `config.ini` files from system and user folders before the flags).
   - Bind `127.0.0.1` on a free port and wait for `/health`. Then check that `GET /props` shows the expected `model_path` and that the process the client started is still running, so the client never talks to a stale server. Log `/props`. Stop the server on exit, including Ctrl-C.
   - Generate with `POST /completion`:
     - `prompt` as a token-ID array
     - an explicit `samplers` list (order and membership; any sampler left out doesn't run)
     - every sampling value, per request. Also send a fixed neutral value for every other sampling field that `generation_settings` reports (for example `mirostat` 0, `dynatemp_range` 0, `logit_bias` [], `grammar` "", `ignore_eos` false, `min_keep` 0, `n_probs` 0), so no server default applies. Log these with origin `backend`. `mirostat` above 0 replaces the whole `samplers` chain, and `logit_bias` and `grammar` act even when they're not in `samplers`.
     - `return_tokens: true`
     - `cache_prompt: false`; the llama.cpp docs warn that prompt caching can make results nondeterministic
     - streaming. Collect the output token IDs from each chunk and check that their count equals the final `tokens_predicted`. llama-server sends no chunk for a token that ends in the middle of a character, so a shorter list is logged as incomplete.
   - Log `generation_settings`, `stop_type`, `stopping_word`, `tokens_evaluated`, `tokens_cached`, `truncated` and `timings` from the response. `timings.cache_n` (prompt tokens reused from cache) must be 0. `tokens_cached` is not that count; it is the slot's whole length after the call.
   - After every call, check three things and flag any difference: every sampling value sent appears unchanged in `generation_settings` (floats at float32 precision); its `n_ctx` is the context size I asked for; and the response's `prompt` field (the server's own decoding of the IDs) equals the client's prompt text. The last check shows that the tokenizer and the GGUF agree.
   - Switching to a profile with another model restarts the server. Don't use router mode.
2. **Transformers** (step 5): in-process. Model and tokenizer at a pinned revision, bf16, on MPS, CUDA or CPU. Feed the same token IDs. Put torch behind an optional extra (`uv sync --extra hf`).
   - Set every `GenerationConfig` field explicitly (`do_sample`, `top_k=0` for off, `max_new_tokens`). Never inherit the model's `generation_config.json` or the library fallbacks (`top_k` 50 and greedy decoding are not neutral), except `eos_token_id`, which you take from `generation_config.json` (OLMo 3 Instruct: 100265 and 100257) and log.
   - Hugging Face applies its processors in a fixed order (repetition penalty, temperature, top_k, top_p, min_p). A different `samplers` order is recorded as not applied, with the order used. `presence_penalty` and `frequency_penalty` don't exist there: not applied. Seed per call.
   - Cut the text at the stop string the way llama-server does, so the backends' outputs compare.
   - Never load both 7B models at once on my 32 GB Mac.

**Settings** use one set of names in profiles (temperature, top_k, top_p, min_p, repeat_penalty, repeat_last_n, presence_penalty, frequency_penalty, seed, max_tokens, samplers), mapped per backend. A setting a backend can't apply is recorded as not applied, never dropped silently. A setting only acts if its sampler is in `samplers`: `repeat_penalty`, `presence_penalty` and `frequency_penalty` need `penalties`; `top_k`, `top_p`, `min_p` and `temperature` need their own names. A non-neutral setting whose sampler is missing is an error when the profile loads.

**Seeds:** the client always sends a concrete seed. `seed = "random"` makes the client draw one and log it, because llama-server doesn't report the seed it picks for -1, so that call couldn't be replayed. `/regen` uses the turn's seed plus 1, then plus 2, and so on, logged with origin `regen`, so each regenerated reply is a new sample that can still be replayed. The profile seed is unchanged for new turns. (With the same seed, a regen would return the same reply; checking that is `/replay`'s job.)

## Models: where the weights come from

Models are listed by hand in `models.toml`: a name, the route, the repo and pinned commit (and the file, for a GGUF), and the Hugging Face repo and commit of the tokenizer (for a converted model, its source repo and commit). `fetch NAME` resolves one entry. Two routes:

1. **Convert from official weights** (the default when no official GGUF exists, as for OLMo 3). `fetch` downloads the official repo (for example `allenai/Olmo-3-7B-Instruct`, `allenai/Olmo-3-1025-7B`) at its pinned commit, then runs llama.cpp's `convert_hf_to_gguf.py` to a bf16 GGUF. The Transformers backend uses the same repo and commit, so both backends run the same weights.
2. **Download a GGUF** (when an official or trusted one exists): repo, file and commit, verified by sha256.

**llama.cpp:** one checkout, built from source at a recorded commit (its path in `server.toml`), provides both `llama-server` and `convert_hf_to_gguf.py`. The converter pins its own torch and transformers versions and needs the whole checkout, so `fetch` runs it with `uv run --no-project --with-requirements <checkout>/requirements/requirements-convert_hf_to_gguf.txt python <checkout>/convert_hf_to_gguf.py …`. The converter's own requirements file is the single source of its dependencies; nothing is copied into this project's `pyproject.toml`. Warn if `/props` `build_info` doesn't match the checkout's commit.

`fetch` writes what it resolved (repo, commit, file paths, sha256, the llama.cpp commit, and the converter's resolved torch and transformers versions) to a generated `models.lock`. It never edits hand-written files. Models live in a gitignored `models/` folder or the Hugging Face cache. The sha256 is computed once, at fetch. At server start, check only the file size and modified time against `models.lock`. A `verify` command does a full re-hash on demand.

## Profiles and config (TOML, read with the standard library's `tomllib`)

- `defaults.toml`: a neutral or "off" value for every profile setting. Requests are filled from the profile, then from these defaults, so server and library defaults never apply. Record the origin of every value (profile, default, backend or regen).
- **Reject unknown setting names when a profile loads.**
- `server.toml`: the llama.cpp checkout path and every launch flag.
- One profile per setup:
  - `backend` and `model` (a name from `models.toml`)
  - `mode`: `chat` or `text`
  - chat: `template_repo` and `template_revision` (default: the model's tokenizer), `template_kwargs`; `system`: absent = no system message (the template may inject its own; OLMo 3's does), or a string or file (`" "` = a blank system turn); `priming`: earlier messages to prepend, as a list of `{role, content}` tables, placed after the system message
  - text: `header`, `user_name`, `assistant_name`, optional priming text. The document is the header, then the priming text, then one line per turn (`UserName: my text`; for a model turn, `AssistantName:` followed by its output exactly as returned, which usually starts with a space), then `AssistantName:` with no trailing space. Header and priming text are used verbatim and carry their own newlines.
  - `settings`
- Files (system prompt, header, priming) resolve relative to the profile and are used byte for byte, so a trailing newline my editor adds is part of the prompt. `/prompt` shows it.
- Example profiles for my Check 0: `1a-stock`, `1b-blank`, `2-system`, `3-primed`, `4-raw-instruct` (text mode, instruct model), `5-raw-base` (text mode, base model). OLMo 3 instruct settings: temperature 0.6 and top_p 0.95 (OLMo's `generation_config.json`), top_k 0, min_p 0, repeat_penalty 1.0, max_tokens 1024, seed 42, context 8192, samplers `["top_k", "top_p", "min_p", "temperature"]`. All Check 0 profiles use the same `max_tokens`.
- Profiles `2-system` and `3-primed` point at `prompts/system.txt` and `prompts/priming.toml`. Create these with clearly marked placeholder text; I'll replace them before the blind run.

## Context

Count prompt tokens before each call. Warn above 80% of the context size, and refuse with a clear message if prompt plus `max_tokens` would exceed it. Context shift stays off on the server, so nothing is ever silently dropped.

## Commands

`/prompt` (the exact prompt text, its token IDs and their count; `/prompt repr` makes whitespace and newlines visible), `/regen`, `/edit`, `/undo`, `/new`, `/profile NAME` (starts a new conversation with that profile), `/settings`, `/mark LABEL` (for example `landed`, `dropped`) and `/note TEXT` on the current turn, `/replay`, `/quit`. Multi-line input.

Regenerating or editing creates a branch, never an overwrite. Every turn records its parent and its origin (the model, me, an edit, or priming), and the log records which branch I continued from.

## Logging

One JSONL file per conversation in `logs/`, with a `schema_version`. **Add `logs/`, `models/` and `runs/` to `.gitignore`: logs are private conversations, and `runs/` holds held-out test material.**

**Every model call, by default:**
- call, parent and conversation IDs; timestamp
- profile name and full contents; every setting sent, with its origin; settings not applied
- backend and its version (llama.cpp build or commit; Transformers and torch versions)
- the model's route, repo, commit and GGUF sha256
- template repo, revision and kwargs; the BOS decision
- the exact prompt text and its token IDs; the client's token count and the server's `tokens_evaluated` and `tokens_cached`
- the output text as returned (llama-server's `content` leaves out the stopping word and special tokens) and the output token IDs; stop type and stopping word; timings
- flags: cut off by length, context full, empty reply, incomplete output IDs, any failed after-call check
- the client's git commit plus a dirty flag

**Every call writes a record, including failures:** an error gets an `error` field. Ctrl-C during generation cancels it, and the partial reply is logged with stop reason `cancelled`.

**Once per server start:** the exact launch command and flags, the OS and machine, and the full `GET /props` response.

**Marks and notes:** separate records pointing at a call ID.

**Hooks only, not built now:** token probabilities (`n_probs`, `post_sampling_probs`) as an opt-in flag; splitting out reasoning text; a "does this setting change the output?" check; export to MLflow. Keep the record format open to these.

## Replay

`replay LOG CALL_ID` re-sends the exact token IDs and settings and compares the new output with the logged one. It reports "identical" (same output token IDs and stop type; same text if the logged IDs are incomplete) or shows the differences. A difference is a finding, not a crash. It lists every recorded fact that differs from now: llama.cpp build, launch flags, `/props`, machine, client commit. A replay is a new log record pointing at the original call.

## Batch mode (for my blind test)

`run --openings runs/check0-openings.txt --profiles 1a-stock 1b-blank 2-system 3-primed` collects each profile's first reply to each opening. Openings are separated by a line containing only `---`. It writes to a new `runs/<run-id>/` folder:
- `blind.md`: each opening verbatim, then its replies shuffled and labelled A, B, C…, each followed by empty `Rating:` and `Note:` lines. Nothing else: no profile names, call IDs or timestamps. Replies are trimmed of leading and trailing whitespace; the log keeps them untrimmed.
- `key.json`: label to profile and call ID, the shuffle seed, and the trim rule.
- `calls.jsonl`: the log records for these calls, kept out of `logs/` so a later transcript export never picks up held-out openings.

`run` refuses if the profiles' `max_tokens` differ, since reply length would give the setup away. While it runs, the terminal shows progress only ("12/40"), never replies. Any error stops the run before `blind.md` is written. Both friends get the same `blind.md`.

## Tests (must pass before calling a step done)

1. Unit tests with a fake backend: chat rendering, text-mode building, stops, branches, `/undo` and `/edit`, the context guard, defaults and origins, rejection of unknown settings, sampler-membership errors.
2. Live tests against llama-server. **Development tests use a tiny model, `HuggingFaceTB/SmolLM2-135M-Instruct` (Apache 2.0), converted the same way as OLMo, so they exercise the whole pipeline including the default model route.** Tests that need OLMo 3 are opt-in (`pytest -m olmo`) and skip if OLMo isn't fetched. The 37/15 counts and the exact prompts below need only OLMo's tokenizer: commit OLMo 3 Instruct's `tokenizer.json`, `tokenizer_config.json` and `chat_template.jinja` (about 7 MB, Apache 2.0, with their source commit) under `tests/fixtures/`, so CI checks them with no network.
   - OLMo 3 Instruct, chat mode, a single user message "hi": with no system message the prompt is **37** tokens; with system `" "` it is **15**.
   - Exact prompts (Python string literals). No system message: `"<|im_start|>system\nYou are a helpful function-calling AI assistant. You do not currently have access to any functions. <functions></functions><|im_end|>\n<|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant\n"`. System `" "`: `"<|im_start|>system\n <|im_end|>\n<|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant\n"`. Don't compare with the server's `/apply-template`: llama.cpp leaves `tools` undefined, OLMo 3's template then takes its function-calling branch, and the difference is not a client bug.
   - An over-long prompt is refused by the client before sending.
   - A text-mode reply stops before writing my next line.
   - An OLMo 3 Instruct chat reply ends with stop type `eos` and last output token ID 100265 (`<|im_end|>`).
   - Replaying a fixed-seed call on the same server gives identical output token IDs. If it doesn't, stop and tell me: no replay result means anything until it does.
3. Transformers backend: identical input token IDs to llama-server for the same profile and message. Compare short greedy outputs across the two backends for information only, since numbers differ across runtimes.

## Constraints

- Python 3.13, pinned in `.python-version`; `uv` with a lockfile. Few dependencies: `httpx`, `transformers` (tokenizer only in the core), `huggingface_hub`, optionally `prompt_toolkit`. Torch only in the `hf` extra.
- Small modules in `src/clarity_chat/` (`backends/`, `models.py`, `render.py`, `session.py`, `logs.py`, `replay.py`, `cli.py`). Type hints, no clever abstractions.
- Works on macOS, Linux and Windows.
- A short README: how to build llama.cpp and fetch a model, how to run it, how a profile becomes the exact token IDs, what is logged and why, how to add a backend.

## Engineering practices

For reference, `lord-gana-poem-gen` uses this setup (its `pyproject.toml`, `.github/workflows/tests.yml` and `tests/conftest.py`):

- **Layout and tooling:** `src/` layout built with `uv_build`, and the console command `clarity`. Ruff for linting and formatting, line length 100, rules `E`, `F`, `I`, `UP`, `B`, `ANN` (every function in `src/` has type hints; tests exempt) and `PLW1514` (every `open()` passes `encoding="utf-8"`). `ty` checks the types. Dev dependencies: pytest, hypothesis, ruff, ty.
- **CI:** GitHub Actions on Linux, Windows and macOS. Run `uv sync --locked`, `ruff check`, `ruff format --check`, `ty check` and `pytest`. CI tests need no network and no model.
- **Tests:** offline by default. Anything needing a real server or model is behind a pytest marker. A `conftest.py` builds a small, self-contained config for each test. Tokenizers for unit tests load from `tests/fixtures/`, never from the network. Use hypothesis for round-trips: a log record written and read back is identical, including unusual Unicode.
- **File safety:** write files atomically (temp file, then `os.replace`), except the JSONL logs, which are only appended to. Write text verbatim (`newline=""`, so Windows doesn't change line endings). Read JSONL by iterating over the file's lines, never with `str.splitlines()`, which also splits on characters like U+2028 that can appear in model output. Use `pathlib` throughout, and no OS-specific code.
- **Append-only logs:** never rewrite a log line. A correction, for example to a mark, is a new record pointing at the old one.
- **Hand-written and generated files stay separate:** the tool never edits `models.toml`, profiles, `defaults.toml` or `server.toml`. What it resolves goes in `models.lock`.

## CLAUDE.md (write it in step 1)

Keep it short. It should contain the commands and this content:

```
## Principle
What you see is exactly what the model got, and every call can be replayed.
No silent fallbacks: if something can't be done exactly as configured, stop with a clear error.

## How to work
- State your assumptions in each step report. Ask before decisions that are hard to undo.
- A step is done when its tests pass. Write tests with the change, and run ruff, ty and pytest before every commit.
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
```

## Sandbox

I'll allowlist what the build needs, and you run the live tests yourself. Expect to need: local binding for llama-server (`sandbox.network.allowLocalBinding: true`), Hugging Face (`huggingface.co`, `*.hf.co`, including `cas-bridge.xethub.hf.co`), GitHub (to clone llama.cpp), PyPI (`pypi.org`, `files.pythonhosted.org`), `download.pytorch.org` (the converter's torch), and writes to `~/.cache/huggingface` and `~/.cache/uv`. Building llama.cpp needs `cmake` and Apple's command-line tools. Tell me the exact settings to add before the first step that needs them.

## Not now

No GUI or web UI, no tools, retrieval or agents, no metrics or LLM judges, no multi-user features, no Ollama backend, no llama-server router mode, no MLflow export. Keep the design open to a small local web page and an MLflow exporter later.

## How to work

Build in this order, and make each step usable before the next:
1. The repo skeleton (`pyproject.toml`, ruff and ty config, the CI workflow, `CLAUDE.md`, `.gitignore`), the llama.cpp checkout and the convert command, `fetch` by conversion with `models.lock`, the managed llama-server with `/props`, defaults and origins, logging, text mode, and a minimal chat loop (multi-line input, `/prompt`, `/settings`, `/new`, `/quit`; pick the profile with `--profile NAME`). All tested with the tiny model.
2. Chat mode via template rendering, and the context guard. Then fetch OLMo 3 Instruct and Base (**ask me first**: about 29 GB on disk per model, source plus GGUF), and run the 37/15, exact-prompt and OLMo live tests. After this step I can try all six Check 0 setups by hand.
3. Batch mode.
4. `/regen`, `/edit`, `/undo`, `/profile`, `/mark`, `/note`, branches, replay, and the GGUF-download route.
5. The Transformers backend and the cross-backend checks.

Call, parent and conversation IDs go in the log from step 1, so branches and replay need no format change later.

After each step, tell me in two or three sentences what you built and any design choice I should know about.

**Commits.** Commit at each working point: at least once per step, and whenever the tests pass after a meaningful change. Small commits, clear messages. Don't push unless I ask.

**Reviews.** Use the code-review skill (if it isn't installed, use reviewer subagents on Opus and tell me), scoped to the changes since the last review:
- After step 2 (template rendering and tokenization: a subtle bug there would quietly corrupt every experiment) and after step 3 (batch mode and blinding). Medium effort. Fix what holds up, rerun the tests, commit.
- No separate reviews for steps 1, 4 and 5 unless one turns out much bigger than expected (several hundred lines).
- **Final review** after step 5: high effort, all review agents on Opus (never Fable), covering the whole codebase. Then fix, test and commit. Max effort only if the high-effort review finds serious problems.
