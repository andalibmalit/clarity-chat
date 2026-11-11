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
| `uv run pytest -m measure -v -s` | the built llama-server and OLMo 3 Instruct fetched | prints timings and memory figures that design choices rest on (cancel, the memory check); asserts almost nothing |
| `uv run pytest -m embed -v -s` | the built llama-server and bge-small fetched (`uv run clarity fetch bge-small-en-v1.5`, about 135 MB once) | whether llama-server's embeddings come back bit-identical, which decides whether documents and meaning search (v0.3.3) go ahead |

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

## How decisions are made

The maintainer decides scope and direction. Each release follows a written plan, and [`ROADMAP.md`](ROADMAP.md) lists what's next. Before a large change, open an issue to discuss it, so your work fits the plan.

Every design choice, with its cost, is in [`docs/design.md`](docs/design.md). When a choice changes, the old one, the new one and the reason are added to its list of changed decisions, so you can see why things are the way they are.

Everyone taking part follows the [Code of Conduct](CODE_OF_CONDUCT.md). Report security problems privately, as [`SECURITY.md`](SECURITY.md) describes.

## Credit

Every merged contribution of any kind counts: code, docs, design, testing, research or substantial review.

- **[`CONTRIBUTORS.md`](CONTRIBUTORS.md)** lists everyone with a merged contribution, with their roles from the [CRediT taxonomy](https://credit.niso.org/), such as Software, Validation or Writing – review & editing. Your pull request can add you, or the maintainer will.
- **Authors in [`CITATION.cff`](CITATION.cff)**, and so in the citation and the Zenodo record, are people who made a substantial contribution: a feature, a meaningful part of a release, or sustained work. The bar is generous. You can ask to be added, decline, or add your ORCID.
- **Order:** the maintainer first, then others in the order of their first substantial contribution.
- **AI tools** are disclosed in commit trailers (`Co-Authored-By`), not credited as authors, because a tool can't take an author's responsibility for the work. Clarity is built with AI assistance: the maintainer approves each plan, reads the review findings, runs the live tests and uses each release.

## Release checklist

A release is done when every item is checked.

1. Every item in the release's plan is built, and the branch has had its reviews: a fresh-context review against the plan, and one review of the whole release for bugs, with each finding checked and the confirmed ones fixed or listed.
2. CI passes on Linux, macOS and Windows.
3. The maintainer runs the live tests on their own machine: `uv run pytest -m live -v`, plus `-m olmo` and `-m hf` if a backend, a chat template or one of the exactness dependencies changed. The results go in the release notes. CI runs only offline tests, so this is the only test of the code that makes Clarity exact.
4. A golden log for this release is added to `tests/fixtures/golden/`: a small log written by this release's code with the fake backend and made-up text. `tests/test_golden.py` checks that current code still reads and replays every one.
5. `CHANGELOG.md`: Unreleased becomes the new version, with **Migration** and **Output may change** where they apply.
6. The version is bumped in `pyproject.toml` and `CITATION.cff` (`version` and `date-released`) together, and `ROADMAP.md` is updated.
7. From v0.3.0, the maintainer uses the release as a user would, following its by-hand list. Releases before the web interface skip this; their live tests carry the weight.
8. A pull request merges the release branch into `main`, and the maintainer tags it `vX.Y.Z`.
9. Only for a version worth citing: a GitHub Release, which mints a Zenodo DOI; that DOI goes in `CITATION.cff`.

The exactness dependencies are the llama.cpp commit, `tokenizers`, `transformers` and `jinja2`: a change to any of them can change output. Each bump gets its own commit, a changelog line under **Output may change**, and the live tests.
