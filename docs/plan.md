# Plan: Clarity

## Context

New, empty repo, currently at `~/git/clarity`; it will be renamed to `~/git/clarity-chat` to match the GitHub repo `clarity-chat` (the product is still **Clarity**; package names in the brief are unchanged). Where the brief says the repo is named `clarity`, read `clarity-chat`. The brief (`docs/brief.md` after step 0) is the spec. This plan records the decisions the brief leaves open, the order of work, and what to check first.

Clarity is a terminal chat client for reproducible prompting experiments with local models (OLMo 3 Instruct and Base first). The client builds the prompt, tokenizes it with the model's Hugging Face tokenizer, starts its own `llama-server`, sends every sampling value explicitly, and logs enough to replay any call.

Ideas reused from `~/git/lord-gana-poem-gen` (ideas only, no imports):
- `docs/inference-settings.md`: library defaults are not neutral; send every value; reject unknown names.
- `src/poemgen/records.py`: `read_text` (verbatim, `newline=""`), `write_text_atomic` (temp file, `os.replace`, Windows retry), append-only JSONL under a lock.
- `pyproject.toml`, `.github/workflows/tests.yml`, `tests/conftest.py`: uv_build, the ruff rule set, ty, pytest markers in `addopts`, CI on 3 OSes, a tmp repo per test.
- Its CLAUDE.md sandbox note: Python HTTPS fails here because the global `**/*.pem` read deny blocks certifi's CA bundle; `rm -rf` is denied, so delete with `shutil`.

## Step 0: hand-off (right after approval)

1. Save this plan to `docs/plan.md`, and the brief verbatim as pasted to `docs/brief.md`.
2. Commit both on `main` ("Add brief and plan"). Don't push.
3. Stop and wait. The user renames the folder to `clarity-chat` and resumes (or a fresh session reads both files and starts step 1). Never rename the folder myself.

## Repo layout

```
.python-version (3.13)  pyproject.toml  uv.lock  .gitignore  CLAUDE.md  README.md
.github/workflows/tests.yml
defaults.toml  server.toml  models.toml  models.lock (generated JSON, committed)
profiles/  1a-stock 1b-blank 2-system 3-primed 4-raw-instruct 5-raw-base smol-chat smol-text (.toml)
prompts/   system.txt  priming.toml  header.txt   (clearly marked placeholders)
docs/      brief.md  plan.md
vendor/llama.cpp   (gitignored checkout, inside the repo so the sandbox can write it)
logs/ models/ runs/ vendor/   (gitignored)
src/clarity_chat/
  cli.py        chat --profile, fetch, verify, replay, run
  profiles.py   TOML loading and validation; settings with origins
  render.py     tokenizer loading, chat rendering, text document, BOS decision, token IDs
  session.py    conversation tree, context guard, calls
  logs.py       records, append-only JSONL, IDs, environment facts
  models.py     fetch (convert; later gguf), models.lock, verify
  replay.py     replay and environment diff
  batch.py      run, blind.md, key.json
  files.py      verbatim read, atomic write, sha256
  backends/__init__.py       Result, Backend protocol, registry
  backends/llama_cpp.py      LlamaServer (process) + LlamaCppBackend
  backends/transformers_hf.py (step 5)
tests/  conftest.py  fakes.py  fixtures/olmo3-instruct/{tokenizer.json, tokenizer_config.json, chat_template.jinja, SOURCE.md}
```

Ruff, ty and pytest exclude `vendor/`, `models/`, `logs/` and `runs/` explicitly (the llama.cpp checkout contains Python). `testpaths = ["tests"]`.

## Design decisions

**Hand-written config (the tool never edits it).**
- `server.toml`: `llama_cpp_dir`, `llama_cpp_commit` (checkout HEAD must match, else error), `startup_timeout_s`, and `[flags]` passed exactly as written, long names only (`true` = bare flag).
  - Required flags: `ctx-size`, `n-gpu-layers` (a number), `flash-attn` (`on`/`off`), `cache-type-k`/`cache-type-v`, `batch-size`, `ubatch-size`, `fit off`, `parallel 1`, `no-context-shift`, `kv-unified` (explicit), `swa-full` (explicit; OLMo 3 uses sliding-window attention), `ctx-checkpoints 0`, `cache-ram 0`, `cache-reuse 0`, `threads`, `threads-batch`, backend sampling off, speculative decoding off.
  - Rejected: `auto` values, `ctx-size 0`, short aliases, and `model`, `hf-*`, `models-*`, `override-kv`, `host`, `port` (the client adds model/host/port).
  - The exact names and values are confirmed against the vendored commit in step 1.
- **Context size** lives only in `server.toml` (8192 for Check 0). A different ctx means a server restart; it's logged with every call.
- `models.toml`: `[name]` with `route` (`convert` | `gguf`), `repo`, `revision` (40-hex commit required), `file` and `sha256` for gguf, `tokenizer_repo`, `tokenizer_revision`.
- `defaults.toml`: temperature 1.0, top_k 0, top_p 1.0, min_p 0.0, repeat_penalty 1.0, repeat_last_n 64, presence/frequency 0.0, seed "random", max_tokens 1024, samplers `["penalties","top_k","top_p","min_p","temperature"]`.
- Profile: `backend`, `model`, `mode`.
  - Chat keys: `template_repo` and `template_revision` (default: the model's tokenizer), `template_kwargs`, `system` (absent, a string, or `{ file = "..." }`), `priming` (a list of `{role, content}` or `{ file = "..." }`).
  - Text keys: `header`, `user_name`, `assistant_name`, `priming` (a string or `{file}`).
  - `[settings]`.
  - Unknown top-level keys and setting names are errors. Files resolve relative to the profile and are read byte for byte.
- Check 0 profiles share one settings block (OLMo values from the brief). 4 and 5 use the same settings, so only mode and model differ.

**Settings and origins.** Each resolved setting is `{value, origin}`, origin `profile | default | backend | regen`.
- A code table defines neutral values. A non-neutral setting whose sampler is missing from `samplers` is a load error (`penalties` covers repeat, presence and frequency).
- `samplers` accepts only `penalties, top_k, top_p, min_p, temperature`, without duplicates. llama-server silently skips unknown names, so the client validates and then checks the echoed list.
- Seeds must be in 0…2³²−2 (0xFFFFFFFF means random to llama.cpp).
- `seed = "random"`: the client draws one, and the record shows `seed_drawn: true`.
- `/regen` seed = the original reply's seed + k, where k = 1 + the number of earlier regens of that user turn. It refuses at the top of the range rather than wrap.
- Each backend maps the names and returns `not_applied: [{name, value, reason}]`.

**Rendering and token IDs (`render.py`).** `AutoTokenizer.from_pretrained(repo, revision=commit)`; pinned commits make cache or network loads equivalent.
- Chat: `apply_chat_template(messages, tokenize=False, add_generation_prompt=True, **template_kwargs)`, in the order system, priming, turns.
- Text: `header + priming + per turn (f"{user}: {t}\n" or f"{asst}:{out}\n") + f"{asst}:"`, stop `["\n" + user_name + ":"]`.
- Tokenize with `add_special_tokens=False`.
- **BOS rule, per mode, logged every call:**
  - Text mode: prepend BOS iff the tokenizer config says to (`add_bos_token`, cross-checked by probing the ID prefix with `add_special_tokens=True`; disagreement = load error).
  - Chat mode: the template is authoritative (HF's own chat tokenization adds nothing), so nothing is prepended. The log records whether the text starts with BOS.
  - Record `{mode, bos_token, bos_token_id, config_add_bos, probe_adds_bos, text_starts_with_bos, prepended, reason}`. The 37/15 fixtures confirm the chat rule.
- Every call logs the transformers, tokenizers and jinja2 versions, because rendering depends on them.

**Backend interface.** `complete(token_ids, settings, stop) -> Result`.
- Result fields: `text, token_ids, n_prompt, n_predicted, stop_type, stopping_word, timings, raw, request, backend_settings, not_applied, flags, checks`.
- Also `start(model)`, `close()`, `describe()`.
- Ctrl-C inside `complete` closes the stream and returns a partial Result with `stop_type = "cancelled"`. Stream reads use timeouts so Ctrl-C also works on Windows.
- Errors, including a mid-stream error event, propagate. The session writes a call record with `error`, then reports it.

**llama-server (`backends/llama_cpp.py`).**
- **Before launch:**
  - Error if a `config.ini` exists in llama.cpp's system or user config paths.
  - Build the environment without `LLAMA_ARG_*`. Log which `LLAMA_*`/`GGML_*` names were present.
  - Launch with `127.0.0.1`, a free port (logged retry if the bind races), a new process group (one `os.name` branch for Windows), and the absolute resolved GGUF path.
- **Startup:** a drain thread decodes output as UTF-8 with `errors="replace"` and writes it to `logs/server-<id>.log`. The startup output also goes in the record. Error on `using config file:`. Poll `/health`.
- **After `/health`:**
  - `/props.model_path` must equal the resolved path, and the process must be alive.
  - `/props.default_generation_settings.n_ctx` must equal `ctx-size`.
  - Warn if `build_info` ≠ the checkout's HEAD.
  - The GGUF size and mtime must match `models.lock`.
- **Stop:** `terminate`, wait, `kill`, via a context manager plus `atexit`. A model switch stops the old server before starting the new one, so two 7B models are never loaded.
- **Request:** `prompt` (IDs), `n_predict`, the mapped settings, explicit `samplers`, `stop`, `stream`, `return_tokens`, `cache_prompt: false`, `backend_sampling: false`.
  - It also sends a neutral value for every other sampling key in `generation_settings`: mirostat (+tau, eta), dynatemp_range/exponent, typical_p, xtc_*, dry_* (incl. sequence breakers), top_n_sigma, adaptive_*, logit_bias [], grammar "", ignore_eos false, min_keep 0, n_probs 0, post_sampling_probs false, n_keep, n_discard, and so on.
  - Each key is classed as sent-neutral or log-only (chat_format, reasoning_*, lora, speculative, grammar_triggers, …). The table comes from the vendored commit's `server-task.cpp`.
- **After-call checks** (a failure adds a flag and prints a terminal warning):
  - Every sent value equals the echoed `generation_settings` (floats at float32 precision), including `samplers`.
  - `n_ctx` matches, where the response reports it.
  - The response `prompt` equals `tokenizer.decode(sent_ids, skip_special_tokens=False)`, which is the rendered text plus any prepended BOS. Both strings are logged.
  - `tokens_evaluated` == len(IDs).
  - `timings.cache_n` == 0.
  - No `generation_settings` key is unaccounted for (catches samplers added in new llama.cpp versions).
  - The returned `logit_bias` is logged; llama.cpp merges in the model's suppress tokens.
  - `tokens_cached` is logged only, never checked; it is the slot length.
- **Output IDs:** these include stop-string tokens and EOS, while the text excludes the stop string (llama-server behavior; documented in the log schema). The `incomplete_output_ids` flag is set when streamed IDs < `tokens_predicted` (a token ending mid-character).
- **Flags:** `length`, `context_full`, `empty_reply` (based on the text), `incomplete_output_ids`, `check_failed:<name>`.

**Context guard.** Count IDs. Warn when the prompt is above 80% of ctx, and refuse before sending when the call would exceed the server's real limit. llama-server stops when `n_tokens + 1 >= n_ctx`, so the boundary case is tested live in step 1 and the guard set to match it exactly (documented).

**Logging (`logs.py`).**
- Format: `logs/<conversation_id>.jsonl`, append-only, `newline="\n"`, flushed per record. Read by iterating the file's lines.
- Every record has `schema_version`, `type`, `time`, `conversation_id`.
- Record types:
  - `conversation`: profile name, raw text and parsed, client commit and dirty flag, machine.
  - `server_start`: argv, flags, env names, llama.cpp dir, commit and dirty flag, build_info, full `/props`, startup output, OS and machine. Written at the top of each conversation file that uses that server.
  - `turn`: turn_id, parent_id, role, origin `user | model | edit | priming`, text, call_id.
  - `call`: everything on the brief's list, plus the full profile contents embedded so `calls.jsonl` and replay stand alone, `server_id`, `regen_of`.
  - `mark` and `note`: point at a call. A correction points at the record it replaces.
  - `replay`.
- Readers ignore unknown keys, so the hooks (n_probs, reasoning split, MLflow) need no format change.
- The client commit and dirty flag are read once per process.

**Session.** Turns form a tree, with `head` at the current turn. The prompt is built from the path from the root to `head`.
- `/edit` creates a sibling user turn (origin `edit`) prefilled with the last message, and calls the model.
- `/undo` moves `head` back past the last exchange.
- `/new` starts a new conversation file.
- `/profile NAME` starts a new conversation, restarting the server only if the model changes.
- `/prompt` shows the last call's prompt text, IDs and count. `/prompt repr` shows `repr()` per line.

**CLI input (decided).** prompt_toolkit, a core dependency. Enter sends; Alt+Enter (or Esc, then Enter) inserts a newline; pasted text keeps its newlines. Text is sent as typed. `/cmd` runs a command; `//` sends a literal `/`. Tests use pipe input, never a real terminal session.

**Fetch (`models.py`).**
1. `snapshot_download(repo, revision)` into the HF cache, plus the tokenizer files at their pin.
2. Convert with `uv run --no-project --with-requirements <ckpt>/requirements/requirements-convert_hf_to_gguf.txt python <shim> …`. The shim writes the environment's full package freeze (including torch and transformers) to JSON, then runs `convert_hf_to_gguf.py` via `runpy`, so the recorded versions are the ones that converted.
3. Output goes to `models/<name>/<name>-bf16.gguf.partial`, then renamed. sha256 is computed once.
4. `models.lock` is written as atomic JSON. `verify` re-hashes.

**Replay.**
- Inputs: the call, its `server_start`, and the embedded profile.
- Default runs the current `server.toml`; `--logged-flags` uses the logged command line.
- Sends the logged request body and IDs.
- "Identical" means the same output IDs and stop type. If the logged IDs were incomplete, it compares text and says so explicitly.
- Lists differences in build_info, flags, `/props` (deep diff), machine, client commit and GGUF sha256.
- Appends a `replay` record to the same log. `/replay` uses the running server.

**Batch (decided).**
- Refuses if `max_tokens` differ.
- Openings are split on lines that are exactly `---` (CRLF-tolerant). Each opening is `str.strip()`ed; empty openings are an error. The rule is recorded in `key.json` and in each call.
- Each (opening, profile) pair is its own conversation. Calls are grouped by model, and progress shows `12/40` only.
- Logs go to `runs/<id>/calls.jsonl`. Any error, including a failed after-call check, stops the run before `blind.md`.
- Per opening, replies are shuffled with a drawn seed and labelled A, B, ….
- Writes `key.json`, then `blind.md` (atomic).
- Reply trim: `str.strip()`. This also hides text mode's leading space. Replies cut off at `max_tokens` remain visible; that's inherent.

**Dependencies.** Core: httpx, transformers (no torch), tokenizers, huggingface_hub, jinja2 (needed by chat templates; transformers doesn't install it), prompt_toolkit. `hf` extra: torch. Dev: pytest, hypothesis, ruff, ty (pinned). uv.lock pins transformers.

## Steps

Each step ends with ruff, ty, pytest, the live tests, a commit, and a 2–3 sentence report with assumptions.

1. **Foundation.**
   - Skeleton, `CLAUDE.md` (the brief's text plus commands), `.gitignore`.
   - Shallow-clone llama.cpp into `vendor/`, record the commit, build `llama-server`.
   - **First, confirm against the vendored source:** flag names, the `generation_settings` keys, sampler names, the config-file paths and message, the `prompt` field, EOS in streamed tokens.
   - Pin the SmolLM2-135M-Instruct commit, `fetch` by conversion, `models.lock`.
   - Managed server and `/props`, defaults and origins, logging, text mode, minimal chat loop.
   - Live: context boundary, same-seed determinism.
2. **Chat mode.**
   - Chat rendering and the context guard.
   - Commit the OLMo tokenizer fixtures (pinned commit, Apache 2.0) and the 37/15 and exact-prompt tests.
   - **Ask before fetching OLMo 3 Instruct and Base (~58 GB total; 149 GiB free).**
   - `pytest -m olmo`.
   - Code review (medium), fix, commit.
3. **Batch mode.** Code review (medium), fix, commit.
4. `/regen`, `/edit`, `/undo`, `/profile`, `/mark`, `/note`, replay, the gguf-download route.
5. **Transformers backend** (`hf` extra).
   - Explicit `GenerationConfig`; `eos_token_id` from `generation_config.json`.
   - Fixed processor order: any other order goes to `not_applied`, as do presence and frequency.
   - Cut at the stop string the way llama-server does; a seed per call.
   - One model at a time, never alongside a 7B llama-server.
   - Cross-backend tests.
   - Final review (high, Opus reviewers), fix, commit.

If same-seed replay on the same server isn't identical, stop and report before going further.

## Sandbox settings to add before step 1

- `sandbox.network.allowLocalBinding: true`
- Network: `github.com`, `huggingface.co`, `*.hf.co`, `*.xethub.hf.co`, `cas-bridge.xethub.hf.co`, `cdn-lfs.hf.co`, `pypi.org`, `files.pythonhosted.org`, `download.pytorch.org`
- Write: `~/.cache/huggingface`
- Read inside the `**/*.pem` deny: `~/git/clarity-chat/.venv` (certifi's CA bundle for Python HTTPS). It's the same mechanism as an existing `.venv` entry for another project. The alternative is excluding `uv run clarity fetch` from the sandbox.
- To watch for: Metal inside the Seatbelt sandbox. If it's blocked, SmolLM2 tests can use a CPU test config, but OLMo needs Metal, which means a sandbox exclusion for `llama-server` (your call).

## Verification

- `uv run ruff check && uv run ruff format --check && uv run ty check && uv run pytest` (offline; CI sets `HF_HUB_OFFLINE=1`).
- `uv run pytest -m live` (SmolLM2 via llama-server); `-m olmo` after step 2; `-m hf` after step 5.
- Manual check: `uv run clarity chat --profile smol-chat`, then `/prompt repr`, `/settings`, and Ctrl-C mid-reply; then read the JSONL record.

## Status and decisions since approval (updated 2026-11-08, during step 3)

**Where we are:** steps 1 and 2 are done and reviewed.
- Step 2 (`d6a3efc`): chat mode; OLMo 3 Instruct and Base fetched; 7 OLMo GPU tests pass. The medium code review's ten findings are fixed. `server.toml` sets `no-jinja` and `chat-template = "chatml"`, because OLMo 3's template stopped llama-server from starting. OLMo 3 ends replies with `<|endoftext|>` (its template does too); the brief expected `<|im_end|>`.
- Nondeterminism (user request): researched and checked by a second agent (`docs/research/nondeterminism.md`). Done: start-up warnings you can act on (build mismatch, uncommitted llama.cpp changes, inherited `GGML_*`/`LLAMA_*` variables with values logged), server warnings after `/new`, macOS build and chip in the machine record, and live replay tests after a longer call and after a restart (identical on both models). `docs/design.md` has a "Determinism and replay" section; the README a short note.

**Overnight run (authorized by the user on 2026-11-07, before sleeping):** finish steps 3 to 5 autonomously, without pushing. Record every decision made without the user under "Decisions to review in the morning" below, and keep each easy to reverse. An hourly session check-in (CronCreate) resumes the work if a usage limit pauses it.

**User decisions for steps 3 to 5 (2026-11-07):**
- `run --seeds N` (default 1): N replies per opening and profile, each still shuffled and blind. The seeds start at 333 (333, 334, …), confirmed by the user: "333" is the seed value, not the count. In batch runs these replace the profile's seed and are logged with their own origin.
- Transformers backend: MPS by default, with a `device = "cpu"` option. If MPS fails the run-to-run replay test, log and document it, and warn once at backend start (actionable: switch to CPU). The user wants to discuss later whether to move experiments to Linux, and whether Macs need a further warning.

**Decisions to review in the morning:** (append as they're made)
- **Batch openings: CRLF becomes LF** before splitting, so an editor's line endings never change the token IDs (otherwise a Windows-edited file would send `\r\n`). Reverse: drop the `replace` in `batch.split_openings`.
- **Batch logs stay in the run folder,** including llama-server's output log (`runs/<id>/server-*.log`), so nothing from a held-out run lands in `logs/`.
- **Split and trim rules are in the `run` record and `key.json`, not repeated in every call.** Each call has `run: {run_id, opening}`, and its `run` record is the first line of the same file.
- **Pre-flight before any model loads:** each profile is prepared (model, lock, weights, tokenizer, template) and every opening is rendered and checked against the context. A profile listed twice is refused.
- **What stops a run:** any error, a `check_failed:*` flag, or Ctrl-C. Other flags (`length`, `empty_reply`, `special_tokens_in_content`) don't; they're in the log, and warnings print once.
- **`blind.md` layout:** `# Opening N`, the opening, then `## A` and so on, each reply followed by `Rating:` and `Note:`. Labels continue `AA`, `AB` after `Z`. One shuffle seed per run (`random.Random`), recorded in `key.json` with the rule.
- **Secret-looking variable values are never logged** (review finding): a kept `LLAMA_*`/`GGML_*` variable whose name contains KEY, TOKEN, SECRET, PASS, AUTH or CREDENTIAL is logged by name only. The server still receives it.
- **A server restart mid-run stops a batch run,** so a run never mixes server processes.
- **Replay sends the logged request body** (plus the logged IDs), not one rebuilt from the settings, so a later change to the client can't alter it. Failed and cancelled calls can't be replayed. `clarity replay` exits 1 when the output differs.
- **"Can't change a replay's output":** only the client commit and the rendering libraries (the IDs are sent as logged). Any other difference (build, flags, `/props`, env, machine, GGUF hash) counts as a possible explanation, so `unexplained_difference` stays off.
- **`/props` `media_marker` is ignored in replay diffs:** it is random at every server start (seen live) and only marks images.
- **`/edit` with no text** prefills the input with your last message; `/edit TEXT` replaces it directly. An edited message uses the profile's seed (it is a new message); only `/regen` adds k.
- **`/undo` writes no log record:** the next turn's `parent_id` shows the branch, which is what the brief asks the log to record.
- **Downloaded GGUFs stay in the Hugging Face cache** (`models.lock` records `~/.cache/...`), rather than being copied into `models/`. Recommended settings come from the tokenizer repo's `generation_config.json`.
- **The GGUF route's live test uses a third-party file** (`bartowski/SmolLM2-135M-Instruct-GGUF`, Q8_0, pinned by commit and sha256), because no official SmolLM2 GGUF exists. It lives only in the test's temporary project, not in `models.toml`. The test downloads 145 MB once.
- **`Session.prepare`** now holds the profile checks that `new_conversation` used to do inline, so batch runs and `/new` share them.
- **PyTorch deterministic mode (your question tonight):** still not planned for step 5. An agent is writing `docs/research/determinism-primer.md` (foundations, what the mode does on CUDA, CPU and MPS, and an audit of the complexity the nondeterminism work added); a second agent will check it before you read it.

**Next actions, in order:**
1. Step 3 is done: batch mode, reviewed (medium), nine of ten findings fixed. The tenth (a run's single server record carries the first conversation's ID) is documented: replay finds servers by `server_id`.
2. Step 4 is done: branches (`/regen`, `/edit`, `/undo`), `/profile`, `/mark`, `/note`, replay (`clarity replay`, `/replay`, with `unexplained_difference`) and the GGUF-download route, all tested live.
3. Step 5: no `use_deterministic_algorithms` (it does nothing on MPS); gate the MPS backend on the same live replay tests; log torch and macOS versions.

**Docs (user request, 2026-11-07):** README and CLAUDE.md state the problem Clarity solves; `docs/design.md` holds choices, tradeoffs, usage patterns and scope, briefly. Keep both current each step. Still owed in the README at the end: how a profile becomes token IDs, what is logged and why, how to add a backend.

**Environment and sandbox (the user's machine):**
- An M2 Max with 32 GB; 149 GiB free disk; Python 3.13.12 (miniforge), which uv picked for `.venv`.
- llama.cpp is shallow-cloned at `vendor/llama.cpp`, commit `f477f795cb07…`, pinned in `server.toml`, with `llama-server` built. `build_info` is `b1-f477f79`.
- **Git clones and pulls into `vendor/` fail inside the sandbox** (writes to `.git/hooks` are blocked, even with `git clone *` excluded). The user runs them with `!`.
- **Python HTTPS inside the sandbox** needs `sandbox.enableWeakerNetworkIsolation: true`, which lets commands reach `com.apple.trustd.agent`; it is now set. The error without it was `OSStatus -26276`.
- **The sandbox hides Metal** (`ggml_metal_init: failed to create command queue`). The user chose GPU live tests, so `excludedCommands` has `uv run pytest -m live*`, `uv run pytest -m olmo*` and `uv run pytest -m hf*`. This works, but **only for the bare command**: every command in a call must match, so `uv run pytest -m live -v | tail` stays sandboxed. `uv run pytest -m live -v` ran unsandboxed; `uv run pytest -m live -v -k placement` did not, for reasons not yet understood. Run the whole suite, unpiped.
- **httpx loads certifi's CA bundle even for plain HTTP**, and the sandbox's `**/*.pem` read deny blocks it there. Local calls pass `verify=False` (they are plain HTTP to 127.0.0.1).
- `curl` piped to Python was denied by the permission system. Pinned commits were looked up with WebFetch.
- When giving the user a settings command, check against the Claude Code docs that it does what it claims first; the user asked for this.

**Pinned models (models.toml):** SmolLM2-135M-Instruct `12fd25f7…`, OLMo-3-7B-Instruct `6e5971d9…`, OLMo-3-1025-7B (base) `a81bae42…`. The commits were read through WebFetch's summarizer; `fetch` will catch it if one is wrong.

**Facts checked against llama.cpp `f477f79` (see `backends/llama_cpp.py`):**
- A token-ID prompt gets no BOS from the server.
- The response `prompt` is the server's detokenization of the IDs, special tokens included.
- Stop-string tokens and EOS are streamed with their IDs, and the text leaves out the stop string. Only a token that ends mid-character gets no chunk.
- If a reply hits `max_tokens` while text is held back as a possible stop string, that text is never streamed. The client then sets the flag `text_differs_from_ids` and logs `decoded_output_text`.
- With context shift off, the server returns at most `n_ctx - n_prompt` tokens and marks the reply `truncated` once prompt + generated reaches `n_ctx`, even when that token was also the `max_tokens`-th (checked live). The client's guard allows a call only if `n_prompt + max_tokens < n_ctx`, so an allowed call is never marked truncated.
- Device and model-buffer lines are logged at llama.cpp's trace level (library info maps to verbosity 4), so `server.toml` must set `log-verbosity` of at least 4 for the GPU placement check. The token-embedding table stays in a `CPU_Mapped` buffer by design.
- `/props.default_generation_settings.n_ctx` holds the context size, and `generation_settings` has no `n_ctx`. The client re-reads `/props` after every call.
- Config files: `/etc/llama.cpp/config.ini` and `${XDG_CONFIG_HOME:-~/.config}/llama.cpp/config.ini`, logged as `using config file:`. The client checks for both before launch and in the startup output.
- Environment variables read as flags: `LLAMA_ARG_*`, `LLAMA_API_KEY`, `HF_TOKEN` and `MTMD_BACKEND_DEVICE`, all removed.
- Flags with no "off" form: `--swa-full` and `--backend-sampling`; both default to off.
- `--spec-type` appends to a list, so it's never passed; each call checks that `speculative.types == "none"`.
- `--cache-idle-slots` defaults to on, so `no-cache-idle-slots = true`.
- Request parsing clamps some values silently (for example `top_p` above 1). The client validates ranges itself, and the echo check catches anything else.
- An empty `stop` list falls back to the command-line stop strings, which are empty.
- `dry_sequence_breakers` must not be empty.
- `sse_ping_interval: 1` keeps reads short, so Ctrl-C is handled promptly.
- llama.cpp's converter copies `generation_config.json` into GGUF `general.sampling.*`, and llama-server uses those as its defaults. The client overrides them by sending every value.

**Decisions made with the user since approval:**
- **Model-recommended settings (opt-in):** `use_recommended = true` in a profile fills unset settings from the model's pinned `generation_config.json`, which `fetch` records in `models.lock` under `recommended`. They are logged with origin `model`.
  - Resolution order: profile, then model, then `defaults.toml`.
  - Mapping: `repetition_penalty` becomes `repeat_penalty`; `do_sample: false` becomes temperature 0. Every other key is listed as `not_applied`, and `max_new_tokens` never becomes `max_tokens`.
  - The Check 0 profiles keep the explicit numbers from the brief.
  - The user's intent: clarity should get out of the way, not be a rigidity that obstructs experimentation.
- **GPU check:** with GPU layers requested, a server start fails if the weights land on the CPU. The device and buffers are parsed from the startup output and logged as `placement`.
- **Local requests** use `trust_env=False`, so the sandbox's proxy variables never apply to `127.0.0.1`.
- **`models.lock` is committed**, so it records paths relative to the repo or as `~/...`, never with the user name.
- **Placeholders** (`prompts/system.txt`, `priming.toml`, `header.txt`) and the Check 0 profiles are still to be written in step 2.
