# Clarity

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.24068236.svg)](https://doi.org/10.5281/zenodo.24068236)

**Clarity**: a chat client for exact, replayable prompting experiments with local LLMs.

**What you see is exactly what the model got.**

## The problem

When you prompt a local model through a server or chat app, much of what shapes a reply is hidden:

- which chat template ran, and how the text was tokenized
- whether a beginning-of-sequence token was added
- sampler defaults from the server or from the model file, and samplers that were skipped silently
- prompt caching, and context shift dropping tokens
- flags picked up from environment variables or config files

So when two replies differ, you can't tell whether your prompt caused it, and you can't rerun the call to find out.

Clarity takes over that layer and shows it to you, without making experiments harder to run:

- it builds the prompt and its token IDs itself
- it sends every sampling value and checks that the server used them
- it logs every setting and environment fact known to affect the output
- it can re-send any completed call, and reports whether the output matched and what changed

## Reproducibility

The same call on the same machine and llama.cpp build gives the same output. Live tests check this for SmolLM2 and OLMo 3 on one Apple Silicon Mac, and it holds for the Transformers backend on MPS and CPU too. A different build, GPU, macOS version or batch setting can change the output, because GPU arithmetic depends on the order of operations. Clarity logs all of these, and `clarity replay` (or `/replay` in a chat) re-sends a call and lists any that differ. See [Design](docs/design.md#determinism-and-replay).

## Features

- Chat mode, through the model's chat template, and text mode, where the model continues a plain document.
- Branches in a chat (`/regen`, `/edit`, `/undo`), and marks and notes on replies, during a chat or afterwards.
- Replay of any completed call, with a list of everything recorded that changed since.
- Export of a conversation to one JSON Lines file a notebook can load, with exact token IDs, where each message sits in the prompt, and marks and notes.
- Export of a conversation or a batch run as Inspect logs, to browse and compare replies in Inspect's viewer (`uv sync --extra inspect`; [Browse a run in Inspect's viewer](docs/guides.md#browse-a-run-in-inspects-viewer)).
- Blind batch runs for comparing setups.
- Attach a document to a message (text, HTML or PDF), whole or in part: Clarity shows what it costs and what it leaves before anything is sent, and logs its text with the message.
- Search your documents by meaning (with bge-m3, a second local model) or by word: each message you send adds the best parts of the documents a chat searches, within a token budget, and logs which parts and why ([Search your documents](docs/guides.md#search-your-documents)).
- A web interface on your own computer (`clarity ui`), over the same commands and logs: the exact prompt, Replay, Settings, Compare, branches and chats.
- Models at a pinned commit: converted from the official weights, or a ready-made GGUF checked by sha256.
- Profiles for SmolLM2-135M, OLMo 3 7B (Instruct and base) and Llama 3.1 8B (Instruct and base; Meta gates its download, see [Fetch a gated model](docs/guides.md#fetch-a-gated-model)).
- A llama-server that Clarity starts and checks itself, and a Transformers backend that runs the same weights in-process (`uv sync --extra hf`).
- Steering on the Transformers backend: a vector made by `clarity vector make` or another tool, pinned by its hash, logged with every reply and replayed.

## The web interface

These are from `clarity ui` with SmolLM2-135M, the smallest model Clarity ships, which is why the replies are weak.

A reply, and under it why it stopped:

![A message asking for three colours in a sunset, the model's numbered reply, and under it the line "stopped at the token limit (max_tokens 128)" with a link to raise max_tokens in Settings.](docs/images/chat.png)

Details shows exactly what the model got, with the text the chat template wrote highlighted:

![The Details panel beside the chat: the prompt with its chat template markers highlighted, 30 tokens starting with a start token, then a summary of why the reply stopped, its length, the model, the profile and the call ID, and a Replay button.](docs/images/details.png)

Settings for one chat, each value with where it comes from:

![The Settings panel beside the reply: the Common group with temperature 0.6, top_p 0.95, max_tokens 128 and seed 42, each marked as from the profile, and the start of the Documents group with the search method, meaning or words.](docs/images/settings.png)

Compare lines up replies by seed, so a change of setting is the only difference in each row:

![Compare: four replies to one message in a grid, seeds 42 and 43 down the side and temperature 0.6 and 1.2 across the top, with a mark where each reply at 1.2 first differs from the one at 0.6.](docs/images/compare.png)

## Install on a Mac

On a Mac with Apple silicon and 16 GB of memory or more, paste this into Terminal:

```
curl --proto '=https' --tlsv1.2 -fsSL https://raw.githubusercontent.com/andalibmalit/clarity-chat/v0.4.4/install.sh | sh
```

It checks the Mac, installs uv if it's missing, puts Clarity in `~/Clarity`, builds llama.cpp, fetches a first chat model (Gemma 4 E4B) and the model meaning search uses, and puts a `Clarity.command` launcher on your Desktop. It asks once, with the sizes, before it downloads anything. See [Getting started](docs/getting-started.md#install-with-one-command) for what it does and writes, and how to read the script first.

## Quick start (by hand)

On Linux, an Intel Mac or a Mac with less memory, or to make your own build. Needs [uv](https://docs.astral.sh/uv/), CMake, a C++ compiler and a GPU that llama.cpp supports (or see [Getting started](docs/getting-started.md#before-you-begin) to run on the CPU). Live tests have run on Apple Silicon only; other GPUs are untested. Getting started also walks through these steps.

```
uv sync
git clone https://github.com/ggml-org/llama.cpp vendor/llama.cpp
git -C vendor/llama.cpp checkout b1b770d90d94cad9e9ea5b993bdbc383a522fb28
cmake -S vendor/llama.cpp -B vendor/llama.cpp/build -DCMAKE_BUILD_TYPE=Release -DLLAMA_CURL=OFF -DLLAMA_USE_PREBUILT_UI=OFF
cmake --build vendor/llama.cpp/build --config Release -j 8 --target llama-server

uv run clarity fetch smollm2-135m-instruct   # download at the pinned commit, convert to GGUF
uv run clarity chat --profile smol-text      # chat; /help lists the commands
uv run clarity ui                            # the same, in your browser
uv run clarity run --openings FILE --profiles smol-chat smol-text  # blind test; FILE holds openings separated by --- lines
uv run clarity replay logs/CONVERSATION_ID.jsonl CALL_ID           # re-send a call, compare
uv run clarity chats                                                # list your conversations
uv run clarity chat --json --profile smol-chat                     # chat from a script: JSON lines in and out
uv run clarity mark logs/CONVERSATION_ID.jsonl CALL_ID LABEL       # label a past reply
uv run clarity note logs/CONVERSATION_ID.jsonl CALL_ID "TEXT"      # add a note to a past reply
```

## Documentation

- [Getting started](docs/getting-started.md): from a fresh clone to a first chat and a first blind test.
- [Guides](docs/guides.md): write a profile, add a model, run a blind test, replay a call, read the logs, and fix common errors.
- [Reference](docs/reference.md): every command, profile key, setting, log record and flag.
- [Design](docs/design.md): the choices, their tradeoffs, and what is out of scope.
- [Contributing](CONTRIBUTING.md): tests, checks, how to add a backend, credit, and the release checklist.
- [Changelog](CHANGELOG.md) and [Roadmap](ROADMAP.md): what changed in each release, and what's next.

## Configuration

You set up every experiment in these files. Clarity doesn't edit the ones it ships; your own go in `local/`, which git ignores. What `fetch` resolves goes to `local/models.lock`; what each call resolves goes to its log.

- `models.toml`: the models, each pinned to a commit. Put your own in `local/models.toml`.
- `server.toml`: every llama-server flag.
- `defaults.toml`: the neutral value of every setting.
- `profiles/*.toml`: one experimental setup each (model, mode, prompt pieces and settings). Put your own in `local/profiles/`.

To use a model's own recommended sampling settings, add `use_recommended = true` to a profile. [Reference](docs/reference.md) lists every key.

## From profile to token IDs

Clarity builds every prompt itself. It resolves each setting and records where it came from, renders the chat template or a plain document, and tokenizes the text with the model's own tokenizer at its pinned commit. It sends the token IDs with every sampling value, and after the call checks that the server used exactly those. `/prompt` shows the text and IDs behind any reply, and `/prompt repr` makes whitespace visible.

One call on the llama.cpp backend, end to end:

```mermaid
sequenceDiagram
    actor You
    participant C as Clarity
    participant T as Tokenizer<br/>(Hugging Face, pinned commit)
    participant S as llama-server<br/>(started by Clarity)
    participant L as Log<br/>(logs/*.jsonl)

    You->>C: message
    C->>C: resolve settings: profile, then model (opt-in), then defaults.toml, each with its origin
    C->>T: render the chat template (or the plain text document)
    T-->>C: prompt text
    C->>T: tokenize, adding no special tokens, and decide BOS
    T-->>C: token IDs
    C->>C: context guard: prompt + max_tokens must fit the context
    C->>S: POST /completion with the token IDs, every sampling value, the sampler order, cache_prompt false
    S-->>C: streamed reply text and token IDs
    S-->>C: final response: the settings it used, its own decoding of the prompt, timings
    C->>S: GET /props
    S-->>C: model path, context size (process still alive)
    C->>C: check: every value echoed unchanged, decoded prompt equals the text, no cache used
    C->>L: append the call record: inputs, request, response, output, checks
    C-->>You: reply, then any warnings, flags and a status line
```

The Transformers backend runs the model in-process instead of through llama-server, and checks its effective generation config after each call.

## What is logged, and why

Each conversation is one JSONL file in `logs/`, append-only. Every call writes a record, even when it fails or you cancel it, holding everything needed to reproduce it or explain a difference:

- the profile's full text and files, and every setting with its origin
- the prompt text, its token IDs and the BOS decision; the template's repo, commit and hash
- the request sent and the server's response, including the values it echoed and what wasn't applied
- the output text and token IDs, stop type, timings and any failed check
- the model file's sha256, the client's commit, and the library versions that render prompts

Each server start is logged once per file: its command line, `/props`, build, environment variables and machine. Turns, marks, notes and replays are their own records. Logs hold private conversations, so `logs/` and `runs/` are gitignored.

## License and citation

Clarity is licensed under Apache 2.0; see [`LICENSE`](LICENSE) and [`NOTICE`](NOTICE). The OLMo 3 tokenizer files in `tests/fixtures/` come from AllenAI under the same license. The web interface's fonts, Rasa and IBM Plex, ship unmodified under the SIL Open Font License 1.1; each family's licence is beside its files in `src/clarity_chat/web_static/fonts/`.

To cite Clarity, use **Cite this repository** on its GitHub page, which reads [`CITATION.cff`](CITATION.cff). Each release is archived on Zenodo with its own DOI; [10.5281/zenodo.24068236](https://doi.org/10.5281/zenodo.24068236) covers all versions. For reproducible work, cite the DOI of the release you used.
