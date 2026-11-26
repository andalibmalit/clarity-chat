# Contributing

## Set up

Follow [Getting started](docs/getting-started.md) to install Clarity and build llama.cpp. For the Transformers backend, also run `uv sync --extra hf`. To run every offline test, install both extras, as CI does: `uv sync --extra hf --extra inspect` (a plain `uv sync` removes an extra installed earlier, and the tests that need it then skip).

## Test

| Command | Needs | Covers |
|---|---|---|
| `uv run pytest` | nothing: no network, model or server | everything that can run offline; this is what CI runs on Linux, macOS and Windows |
| `uv run pytest -m live` | the built llama-server and the fetched SmolLM2; downloads a 145 MB GGUF once | real calls, the context boundary, replay, determinism and batch runs; CI runs it on a Linux CPU build, without the `gpu` test |
| `uv run pytest -m olmo` | both OLMo 3 models fetched (about 58 GB) | chat templates, end tokens and determinism on OLMo 3 |
| `uv run pytest -m llama` | both Llama 3.1 8B models fetched (about 64 GB; gated), and the `hf` extra | one BOS, the template's dated system header, the same prompt token IDs on both backends, and steering on Llama 3.1 Instruct |
| `uv run pytest -m hf` | the `hf` extra and the fetched SmolLM2 (one test also uses OLMo 3 Instruct, if fetched) | the Transformers backend on MPS and CPU, and that both backends get the same prompt token IDs |
| `uv run pytest -m measure -v -s` | the built llama-server and OLMo 3 Instruct fetched | prints timings and memory figures that design choices rest on (cancel, the memory check); asserts almost nothing |
| `uv run pytest -m browser` | `uv sync --group browser`, and Chrome on the Mac (CI uses Playwright's Chromium, and runs the tier again in WebKit on macOS and Firefox on Linux, allowed to fail, with the counts in the job summary; locally, add `--browser webkit` or `--browser firefox` after `uv run playwright install webkit firefox`) | the web interface in a real browser: its secret, a page on another port, the live tab, a reply joined after a reload |
| `uv run pytest -m embed -v -s` | the built llama-server, bge-m3 fetched (`uv run clarity fetch bge-m3`: a 2.27 GB download, and about 3.4 GB kept, since the download stays in the Hugging Face cache), and the `hf` extra (`uv sync --extra hf`; without it the reference test skips) | that bge-m3's vectors come back bit-identical on Metal, from a few tokens to 8,192, and match the source model's |
| `uv run pytest -m setup -v -s` | network to GitHub, `uv sync --group setup`, about 5 minutes | `clarity setup`'s own steps for real, into a temporary folder: llama.cpp's source fetched at the pin from GitHub, and `llama-server` built from it, reporting the pin; CI runs it on macOS (Apple silicon) when a file setup depends on changes (`.github/workflows/setup.yml`) |
| `CLARITY_GATE_ROOT=ROOT GATE_MODEL=NAME HF_HOME=HF uv run pytest -m gate -v -s` | a scratch root made by `notes/tools/gate_root.py` and fetched with an empty `HF_HOME` | a first model's gate: it loads, thinking off, no date, its banned tokens recorded, a call replayed Identical, and both servers' memory printed. With no variables set, the same checks on the shipped `gemma-4-e4b`, once fetched (`uv run pytest -m gate -v -s`) |

The offline tier can also run in parallel, in about a minute instead of six: `uv run pytest -n 8 --dist loadfile`. CI and the live tiers run serially, because the live tiers share one server, GPU or model.

Offline tests run on a fake backend (`tests/fakes.py`). To keep it honest, `-m live` and `-m hf` each record the shape of the call records their backend writes for every way a reply ends, in `tests/fixtures/backend_shapes.json`. If a shape changes, the live test rewrites the file and fails. Read the diff and commit it. Offline, `tests/test_contract.py` holds the fake to llama.cpp's shapes, and allows only the differences written down in `tests/contract.py`.

Before every commit, run the checks CI runs:

```
uv run ruff check && uv run ruff format --check && uv run ty check && uv run pytest
```

Tests use made-up text. Never put anything from `logs/` or `runs/` in a test or an issue; they hold private conversations.

## Keep the docs current

Update the docs in the same commit as the change. Choices, tradeoffs and scope go in [`docs/design.md`](docs/design.md); keys, commands and record fields in [`docs/reference.md`](docs/reference.md); usage in [`docs/guides.md`](docs/guides.md), [`docs/getting-started.md`](docs/getting-started.md) and the README. `tests/test_docs.py` fails if a profile key, setting, CLI command or chat command is missing from the reference, or a relative link is broken.

## Add a backend

A backend turns token IDs plus settings into a reply. Write a module in `src/clarity_chat/backends/` with a class that implements the `Backend` protocol in `backends/__init__.py`. Then add its name to `BACKENDS` in `profiles.py` and to `Session.backend_for`. Other code branches on the backend's name, such as `Session.weights_for`, which picks the files a backend loads; `grep -rn '"transformers"' src/` finds each place.

- Map every Clarity setting to the backend's own, and set every other output-affecting value to a fixed neutral one.
- List any setting it can't apply in `not_applied`, with the reason; never drop one silently.
- Return a `Result`: output token IDs including stop-string and end tokens, text without the stop string.
- For replay, send a logged request exactly as logged, or raise `BackendError`.

`backends/llama_cpp.py` and `backends/transformers_hf.py` are the two examples.

## How decisions are made

The maintainer decides scope and direction. Each release follows a written plan, and [`ROADMAP.md`](ROADMAP.md) lists what's next. Before a large change, open an issue to discuss it, so your work fits the plan.

Every design choice, with its cost, is in [`docs/design.md`](docs/design.md). When a choice changes, the old one, the new one and the reason are added to its list of changed decisions, so you can see why things are the way they are.

Everyone taking part follows the [Code of Conduct](CODE_OF_CONDUCT.md). Report security problems privately, as [`SECURITY.md`](SECURITY.md) describes.

## Credit

Every merged contribution of any kind counts: code, docs, design, testing, research or substantial review.

- **[`CONTRIBUTORS.md`](CONTRIBUTORS.md)** lists everyone with a merged contribution, with their roles from the [CRediT taxonomy](https://credit.niso.org/), such as Software, Validation or Writing – review & editing. Your pull request can add you, or the maintainer will.
- **Authors in [`CITATION.cff`](CITATION.cff)**, and so in the citation and the Zenodo record, are people who made a substantial contribution: a feature, a meaningful part of a release, or sustained work. The bar is generous. You can ask to be added, decline, or add your ORCID.
- **Order:** the maintainer first, then others in the order of their first substantial contribution.
- **AI tools** are disclosed in an `Assisted-by: TOOL:MODEL` trailer on each commit they helped with (such as `Assisted-by: Claude Code:claude-opus-5-5`), not credited as authors, because a tool can't take an author's responsibility for the work. Clarity is built with AI assistance. The maintainer sets the plans' direction and reads the review findings; from v0.2.3 to 0.8.0, in one sprint, the plans are approved under the maintainer's delegation and the maintainer uses the whole after 0.8.0, through extended use, rather than each release.

## Reviews

A release is a closed list the maintainer can judge in one sitting. Its plan names the review points: after each group that later items build on, after any item that changes record fields, resume or old-log behaviour, and a whole-branch review before the release is used. An extra review is due when the diff since the last one passes about 400 source lines (review fixes don't count: the fix review reads them).

Each plan sets an estimate and a cut order. When an item lands over 1.5 times its estimate and at least 100 lines over, or the running total passes the estimate, the next entry in the cut order is cut before the next item starts. A review doesn't replace the cut.

Reviews check the diff against the plan and look for bloat: code no item asked for, abstractions with one user, unrequested options, tests for cases that can't happen. They skip style and prefer findings that delete code. A must is a wrong prompt or output, lost data, a broken exactness or replay promise, a security hole, or no way out. Any other finding may delete or fix, but shouldn't grow what a user must learn: a command, control, state or sentence the product shows in normal use goes to the maintainer with its price. Three things aren't growth and are judged on their merits: a fix that makes an approved decision work as planned; hardening of security, exactness or data safety that a user meets only when something is wrong; a sentence in the docs, changelog or help.

## Release checklist

A release is done when every item is checked.

1. Every item in the release's plan is built, and the branch has had its reviews: a fresh-context review against the plan, and one review of the whole release for bugs, with each finding checked and the confirmed ones fixed or listed.
2. CI passes on Linux, macOS and Windows.
3. The maintainer runs the live tests on their own machine: `uv run pytest -m live -v`, plus `-m olmo`, `-m llama` and `-m hf` if a backend, a chat template or one of the exactness dependencies changed. The results go in the release notes. CI runs the offline tests on all three systems and the SmolLM2 live tier on a Linux CPU build, without its GPU check (the `gpu` marker); the GPU path, the other models and the Transformers backend are tested only by these runs.
4. A golden log for this release is added to `tests/fixtures/golden/`: a small log written by this release's code with the fake backend and made-up text. `tests/test_golden.py` checks that current code still reads and replays every one.
5. A shipped model fetched at a new pin, or newly added, has its entry copied to `models.lock` with `uv run python scripts/refresh_lock.py NAME …`, committed on its own.
6. A new llama.cpp commit in `server.toml` is on ggml-org's `master`: GitHub serves a fork's commit through the parent repository's URL, so the hash fixes the code but not whose it is.
7. `CHANGELOG.md`: Unreleased becomes the new version, with **Migration** and **Output may change** where they apply. The entry says when the release moves llama.cpp's pinned commit or converts a model again, since Getting started's steps for moving to a new release don't cover either.
8. The version is bumped in `pyproject.toml`, `install.sh`'s `VERSION` and `CITATION.cff` (`version` and `date-released`) together, the install command in `README.md` and Getting started names the new tag (`tests/test_install.py` checks both), and `ROADMAP.md` is updated.
9. The maintainer uses the release as a user would, following its by-hand list. During the sprint to 0.8.0 (v0.2.3 onward) this is done once, for all the sprint's releases, as extended use after 0.8.0, except the first runs on real Linux and Windows machines (v0.6.0, v0.7.0); until then the live tests and the QA pass carry the weight.
10. A pull request merges the release branch into `main`, and the maintainer tags it `vX.Y.Z`.
11. After the tag, the maintainer pastes the README's install command on a fresh macOS account, so the script as shipped runs once from start to end.
12. Only for a version worth citing: a GitHub Release, which mints a Zenodo DOI; that DOI goes in `CITATION.cff`.

The exactness dependencies are the llama.cpp commit, `tokenizers`, `transformers` and `jinja2`: a change to any of them can change output. Each bump gets its own commit, a changelog line under **Output may change**, and the live tests. `tests/test_dependencies.py` holds the locked versions of these three and `pypdf` in a table, so a change to `uv.lock` that moves one fails until that table is edited, in the bump's own commit.

`inspect-ai` is pinned exactly, and an export for Inspect refuses any other version. A bump gets its own commit and a changelog line, updates `INSPECT_AI` in `src/clarity_chat/export_inspect.py`, and reruns the tests of what Inspect does with Clarity's logs: `tests/test_inspect_gate.py` offline, and `tests/test_browser_inspect.py` and `tests/test_browser_inspect_export.py` in the browser tier (`uv sync --group browser --extra inspect`).
