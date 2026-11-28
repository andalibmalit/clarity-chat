# Guides

Short recipes for common tasks. [Getting started](getting-started.md) covers installation and a first chat. [Reference](reference.md) lists every key and command.

## Write a profile

A profile is one experimental setup: a model, a mode, the prompt pieces and the settings. Save it as `local/profiles/NAME.toml` (create the folder the first time) and start it with `uv run clarity chat --profile NAME`. The profiles in `profiles/` ship with Clarity; copy one into `local/profiles/` under a new name to change it.

A chat-mode profile uses the model's chat template:

```toml
backend = "llama.cpp"
model = "smollm2-135m-instruct"   # a name from models.toml or local/models.toml
mode = "chat"
system = "You answer in one sentence."    # or { file = "PATH" }; leave out for none
priming = [                                # earlier messages placed before the conversation
  { role = "user", content = "What's a token?" },
  { role = "assistant", content = "A piece of text the model reads as one unit." },
]

[settings]
temperature = 0.6
top_p = 0.95
max_tokens = 256
seed = 42
samplers = ["top_k", "top_p", "min_p", "temperature"]
```

A text-mode profile has the model continue a plain document instead, with no template:

```toml
backend = "llama.cpp"
model = "smollm2-135m-instruct"
mode = "text"
header = "A conversation between User and Assistant.\n\n"
user_name = "User"
assistant_name = "Assistant"
```

The model sees the header, the priming text, one line per turn, and a final `Assistant:`. Generation stops at a new line starting with `User:`. [Profile keys](reference.md#profile-keys) gives the exact format.

Tips:
- Unknown keys and setting names are errors, so most typos stop the profile from loading. Keys inside `template_kwargs` can't be checked against a fixed list, because the chat template defines them. Instead, Clarity warns when a conversation starts if the template never reads one of them.
- File paths are relative to the profile, and files are read byte for byte. A trailing newline your editor added is part of the prompt; `/prompt repr` shows it.
- A priming file holds only `[[messages]]` tables, each with `role` (`"user"` or `"assistant"`) and `content`:

  ```toml
  [[messages]]
  role = "user"
  content = "What's a token?"
  ```
- Template options such as `enable_thinking` go in a `[template_kwargs]` table.
- Settings you leave out come from the model if `use_recommended = true`, and otherwise from `defaults.toml`, whose values turn every sampler off.

## Use another repo's chat template

Set `template_repo` and `template_revision` (a full commit hash) together in a chat-mode profile. The template comes from that repo; the model's own tokenizer still does the tokenizing.

`fetch` doesn't download that repo, so put its tokenizer files in the Hugging Face cache yourself:

```
uv run python -c "from huggingface_hub import snapshot_download; from clarity_chat.render import TOKENIZER_FILES; snapshot_download('ORG/REPO', revision='COMMIT', allow_patterns=TOKENIZER_FILES)"
```

## Compare two setups

Make two profiles that differ in one key, and keep everything else the same. Settings you don't set come from `defaults.toml` for both, so the only difference is the one you made.

To try a value without writing a profile, use `/set` in a chat, for example `/set temperature 0.9`. The change lasts for that conversation, each call logs it with origin `chat`, and `/set temperature` with no value goes back to the profile's. For a comparison you'll report, put the value in a profile so it's written down.

A fixed seed makes a reply replayable, not representative: at temperature 0.6, another seed gives another reply. Compare setups over several openings or seeds with a [blind test](#run-and-rate-a-blind-test).

To see what a chat template contributes, compare a text profile with a chat profile on the same instruct model. `4-raw-instruct` sends OLMo 3 Instruct a plain document; `1a-stock` sends it the same conversation through its chat template.

## Use a model's recommended settings

Add `use_recommended = true` to a profile. Every setting the profile leaves out then comes from the model's own `generation_config.json`, at the commit pinned in the model's entry (`models.toml` or `local/models.toml`), before falling back to `defaults.toml`.

`/settings` marks these values with origin `model`, and lists what the model recommends whether or not you use it. Only sampling values carry over: `repetition_penalty` becomes `repeat_penalty`, and `do_sample = false` becomes temperature 0. Everything else, including `max_new_tokens`, is listed as not applied.

## Add a model

Add an entry to `local/models.toml` (create it if it isn't there), then run `uv run clarity fetch NAME`. Its lock entry goes to `local/models.lock`; `models.toml` is the shipped list, and a name can't be in both. Lock entries are kept by name, so an entry you move to `local/models.toml` or rename is fetched again under its new name (a `convert` entry is converted again). Every revision is a full 40-character commit hash, so a name always means the same files.

To find a repo's current commit:

```
uv run python -c "from huggingface_hub import HfApi; print(HfApi().model_info('ORG/REPO').sha)"
```

Clarity can get a model in three ways.

**Convert the official weights** (`route = "convert"`). Clarity downloads the repo and converts it to a bf16 GGUF with the llama.cpp checkout's own converter. Use this when no official GGUF exists. The Transformers backend can then run the same weights.

```toml
[my-model]
route = "convert"
repo = "ORG/REPO"
revision = "<40-character commit>"
tokenizer_repo = "ORG/REPO"
tokenizer_revision = "<the same commit>"
```

**Download the official weights alone** (`route = "safetensors"`). Clarity downloads the repo's safetensors, config and tokenizer, no pickle, and makes no GGUF. It records each file's sha256, which `clarity verify` checks. Use this for a model the converter can't convert, or one you only run on the Transformers backend or make vectors on. A llama.cpp profile on it is refused.

```toml
[my-model-official]
route = "safetensors"
repo = "ORG/REPO"
revision = "<40-character commit>"
tokenizer_repo = "ORG/REPO"
tokenizer_revision = "<the same commit>"
```

**Download a ready-made GGUF** (`route = "gguf"`). Clarity downloads one file and checks it against the sha256 you give. The tokenizer, chat template and recommended settings come from `tokenizer_repo`, usually the model the GGUF was made from.

```toml
[my-model-q8]
route = "gguf"
repo = "ORG/REPO-GGUF"
revision = "<40-character commit>"
file = "model-Q8_0.gguf"
sha256 = "<64 lowercase hex characters>"
tokenizer_repo = "ORG/REPO"
tokenizer_revision = "<40-character commit>"
```

Hugging Face shows a file's sha256 on its page. Or look it up:

```
uv run python -c "from huggingface_hub import HfApi; print(HfApi().get_paths_info('ORG/REPO-GGUF', ['model-Q8_0.gguf'], revision='COMMIT')[0].lfs.sha256)"
```

After every call, Clarity checks that the server's own decoding of the token IDs matches the prompt text. If a GGUF and its `tokenizer_repo` don't agree, that check fails.

## Fetch a gated model

Some repos are gated: Hugging Face gives their files only to accounts the publisher has let in. Meta's Llama 3.1 repos, which `llama-3.1-8b-instruct` and `llama-3.1-8b-base` come from, are gated.

1. Sign in at [huggingface.co](https://huggingface.co), open the model's page ([Instruct](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct), [base](https://huggingface.co/meta-llama/Llama-3.1-8B)) and accept its licence. Access isn't always immediate; the page says when you have it.
2. Run `uv run hf auth login` (again, if your login has expired) and follow its prompts. It saves a token where the Hugging Face library finds it. Setting `HF_TOKEN` in your shell works too.
3. Run `uv run clarity fetch NAME`.

Clarity never asks for or stores the token; the Hugging Face library reads it. Without access, `clarity fetch` stops and says what to do.

Llama 3.1 comes under Meta's [Llama 3.1 Community License](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct/blob/main/LICENSE). Its "Built with Llama" attribution applies if you distribute or make available the model, a file converted from it, or a product or service that contains it; `clarity fetch` only downloads it to your computer. If you share the files it downloads or converts, the licence's terms for distributing apply to you. Meta's [Acceptable Use Policy](https://llama.meta.com/llama3_1/use-policy) applies to how you use the model.

## Run and rate a blind test

1. Write the openings to a text file, separated by lines that are exactly `---`. Each opening is stripped of surrounding whitespace.
2. Run every profile on every opening:

   ```
   uv run clarity run --openings openings.txt --profiles A B C
   ```

   Add `--seeds 3` for three replies per opening and profile, with seeds 42, 43 and 44. The run's seeds replace the profiles' seeds, so two runs with the same profiles are comparable.

3. Give raters `runs/RUN_ID/blind.md`. Each reply has empty `Rating:` and `Note:` lines.
4. After rating, open `key.json`. It says which profile and seed each label came from. Each label's call ID, under `openings[].labels.<label>.call_id`, is what you pass to [`clarity mark`](#label-replies-after-the-fact) or [`clarity replay`](#replay-a-call).

Before any model loads, Clarity checks every profile and renders every opening. It stops if the profiles' `max_tokens` differ (reply length would give a setup away), a profile is listed twice, or an opening doesn't fit the context. During the run, any error or failed check stops it, and no `blind.md` is written.

Each call is its own conversation. Profiles are read once, at the start, so editing one during a run changes nothing. `blind.md` strips the whitespace around each reply; `calls.jsonl` keeps replies exactly as generated.

## Browse a run in Inspect's viewer

[Inspect](https://inspect.aisi.org.uk/)'s viewer lists a run's replies, puts every profile's reply to one opening side by side, and can add a score later. Clarity writes the logs; Inspect only reads them.

1. Install the extra: `uv sync --extra inspect`. Add `--extra hf` if you use the Transformers backend, since `uv sync` removes extras you don't name.
2. Run the profiles and mark replies, as in [Run and rate a blind test](#run-and-rate-a-blind-test) and [Label replies after the fact](#label-replies-after-the-fact).
3. Export the run:

   ```
   uv run clarity export runs/RUN_ID --format inspect
   ```

   It writes one log per profile, `runs/RUN_ID/inspect/1-A.eval`, `2-B.eval` and so on, readable only by you on Linux and macOS. A run that stopped early isn't exported. Exporting again replaces the files, so mark first and export last. What goes where is in [Export for Inspect](reference.md#export-for-inspect).

4. Open the viewer:

   ```
   uv run inspect view --log-dir runs/RUN_ID/inspect
   ```

   It doesn't open a browser: open the address it prints. A second `inspect view` on the same port stops the first, so view a run and a chat one at a time, or give one of them `--port`.

**Compare profiles.** Choose **Samples**, at the top right beside Tasks and Folders, for one grid of every log's replies. Each opening is a sample, numbered as in `key.json`, and each seed an epoch of it. Sort by **Id** so each opening's replies from every profile sit together, with **Model** at the row's left and **MARK** as the last column. **Input** is the opening as the model got it, so a profile that searches documents shows the parts search added there too. MARK is the reply's labels, or "unmarked". The grid has no column for the reply itself, so open a row to read it; a log's own Samples list shows it as **Answer**, one profile at a time, and can filter on MARK; with 10 or more distinct values it stops offering MARK's values and stops sorting on it. To read the replies side by side in a terminal instead, run this with `uv run python`:

```python
from pathlib import Path

from inspect_ai.log import read_eval_log

replies = {}
for path in sorted(Path("runs/RUN_ID/inspect").glob("*.eval")):
    log = read_eval_log(str(path))
    for sample in log.samples or []:
        mark = sample.scores["mark"].value if sample.scores else ""
        replies.setdefault((sample.id, sample.epoch), []).append(
            (path.name, mark, sample.metadata["output_text"])
        )
for (opening, epoch), rows in sorted(replies.items()):
    print(f"Opening {opening}, epoch {epoch}")
    for name, mark, text in rows:
        print(f"  {name} [{mark}]: {text!r}")
```

**See the exact text.** The logs ask the viewer to show their text as written, not as Markdown, as Clarity's page does: on a sample's Messages tab, leading spaces, a line starting with `* ` and tags like `<think>` show as they are, and a hidden character, like a zero-width space, shows as a marker such as ⟨U+200B⟩. The **Answer** above the tabs shows the reply on one line. A call with no system message of its own shows the text its chat template wrote of its own before the first message as a system message, like the system message OLMo 3's template adds when a profile has none, or Llama 3.1's system header with its dates. The messages leave out the role markers and BOS the template adds; `prompt_text` has them. The record is the sample's metadata: `prompt_text`, `prompt_ids` and `output_text`, with every setting and the facts replay compares. Each message's own metadata, its span of the prompt as logged, shows folded on the user's message but not on the system message; `read_eval_log` reads it on both.

**Know who made the calls.** Each log's model is `clarity/PROFILE`, and its tags, `clarity` and `not-run-by-inspect`, show in the log's header and on its Task tab, not in the grid. Its Info tab shows `inspect_note`, which says Clarity made the calls; below it, Clarity's export header is folded under `clarity`: click ▸ to open it. To make a call again, use [`clarity replay`](#replay-a-call).

**Score later.** `inspect score` adds a score beside `mark` and keeps it. A scorer is a Python file; its inner function needs a return annotation (`-> Score`), or `inspect score` refuses it:

```python
from inspect_ai.scorer import Score, Target, accuracy, scorer
from inspect_ai.solver import TaskState


@scorer(metrics=[accuracy()])
def mentions_blue():
    async def score(state: TaskState, target: Target) -> Score:
        return Score(value=1 if "blue" in state.output.completion.lower() else 0)

    return score
```

Save it as `blue.py`, then score a copy outside `inspect/`, so the viewer and the loop above don't show A's replies twice, and make the copy readable only by you:

```
uv run inspect score runs/RUN_ID/inspect/1-A.eval --scorer blue.py@mentions_blue --model none --action append --output-file runs/RUN_ID/1-A-scored.eval
chmod 600 runs/RUN_ID/1-A-scored.eval
```

`--model none` is for a scorer that calls no model; a model-graded scorer needs a real model, passed with `--model` or in the scorer's own `model=`. `--action append` keeps `mark`. Inspect writes the scored copy, which holds the whole conversation, with your default file permissions, readable by others under the usual umask, so `chmod 600` it. A new export of the run replaces `1-A.eval` but leaves the scored copy as it was, so score again after exporting again. If the scored copy exists, `inspect score` asks whether to overwrite it; add `--overwrite` to say yes. `inspect score` also makes a `logs/` folder where you run it, if there isn't one, and leaves `blue.py`'s `__pycache__/` there. In Clarity's folder, `logs/` is your chats folder; Inspect writes nothing in it.

**A chat.** `/export inspect` in the chat, or `uv run clarity export ID --format inspect` afterwards, writes one conversation as `logs/exports/ID.eval`, ID as `clarity chats` lists it; open it with `uv run inspect view --log-dir logs/exports`. Each reply is its own sample, named by the message it answers and its place among that message's replies, so sort by **Id** to keep a message's regenerated replies together.

**What Inspect keeps.** On macOS, `inspect view` leaves an empty trace file in `~/Library/Application Support/inspect_ai/traces/` and a pid file in `view/ports/`; `inspect score` leaves a trace file in `traces/` that names the log files it read and wrote. With `inspect-ai` 0.3.291, none holds reply text.

## Attach a document

In a chat-mode profile, `/attach notes.md` puts the file's text at the start of your next message, after a framing line that says it's reference material. It stays in the chat like a long paste: later replies see it, and the log keeps its text. Clarity reads PDFs, HTML, and UTF-8 text such as Markdown or code ([what it reads](reference.md#what-clarity-reads-from-a-file)). A line before each prompt lists what's waiting; `/detach` removes one, and `clarity chat --attach FILE` attaches a file once the model is ready.

`/attach notes.md check` counts the file without attaching it: what the whole takes, what it leaves for your message, and where it would fit. If the whole doesn't fit, the refusal names the largest start it found that still leaves at least `max_tokens` for your message, as a command to copy, such as `/attach notes.md lines 1-19`. Attach part of a file with `pages A-B` for a PDF or `lines A-B` for anything else: only the pages or lines you choose are sent, after a line that says they're an excerpt unless the profile sets no marker. Pages are counted from the file's first page, not the numbers printed on them. A range with no text in it is refused. A new chat (`/new`) has more room, and a lower `max_tokens` leaves more for the document. Clarity never trims a document to make it fit.

In the web interface, the paperclip beside the message box opens your computer's file picker. The sheet that opens shows the same numbers as `check`: choose **Attach whole**, **Attach part of it** (a file that doesn't fit opens on the part that still leaves room for your message, and the range boxes take any other), or **New chat with this document** when only an empty chat has room. **Show extracted text** shows the start and end of what will go in; for a PDF or a web page it's open from the start, since it's the first check that the text came out as you expect. A PDF can hold text a viewer doesn't show, so the whole is in **Details** after you send, and in `/attach FILE check`'s result with `--json`. The attachments waiting for your message show as chips above the box; × removes one. Your sent message lists each one with what it cost when attached. **Edit** keeps them, each as a chip whose × leaves it out of the edited message (`/edit --drop N` in the terminal). In **Details**, the framing line and documents Clarity put in your message have a highlight of their own, and **Attachments** lists each one with where its block sits in the prompt's tokens.

The framing line comes from the profile. To change it, set `framing` in the profile's `[documents]` table; `framing = ""` sends none. `excerpt_marker` sets the excerpt's line the same way ([profile keys](reference.md#profile-keys)).

## Search your documents

Give Clarity a document once, and a chat can search it with every message you send. Add it to your library:

```
uv run clarity document add glossary.md
```

Clarity reads the file as `/attach` does, then writes `local/documents/glossary/`: a copy of the file, its text as `text.md`, its parts and a manifest. Each part is a paragraph or a few, up to 250 words, titled by the Markdown heading above it. `clarity document list` lists your documents. Adding a file under a name that's taken replaces that document, and Clarity says so before and after.

In a chat, `/search glossary` searches it from your next message, or set `search = ["glossary"]` under `[documents]` in a chat-mode profile to search it from the start. Each message you send is searched by meaning: Clarity finds the parts closest in meaning to what you typed, with a second local model, bge-m3 (fetch it once: `uv run clarity fetch bge-m3`). The best 8 parts at or above the minimum similarity (0.5 by default; `min_similarity` in the profile) go into your message, best first (parts that tie go by document name and number), until the next one would pass the budget (1,500 tokens by default; `budget` in the profile). A part already in the chat isn't added again. The first search of a document prepares it, and says so; later ones reuse that. `/try TEXT` shows what a message would find, with no model call, and when nothing reaches the minimum, how close the best part came. After each reply, a line under the status line says what search added, and `/history` shows what each message found. The parts are logged with the message, so a regeneration sends them again without searching, and replay sends them exactly.

To search by words instead, set `search_method = "words"` under `[documents]` in your own profile and start a new chat: each message is then searched with the words you typed, ranked by SQLite FTS5's `bm25` ([Word search](reference.md#word-search)). A chat you resume keeps the method it searched with.

In the web interface, **Search it** on the paperclip's sheet, beside Attach whole, adds the file to your documents and searches it in this chat from your next message; a document that already has the name in its box is replaced, as the sheet says. The sheet says how parts are chosen: by meaning, with the minimum similarity, or by words; and Search it warns when the chat searches by meaning and bge-m3 isn't fetched. **Searching** beside Settings lists what the chat searches, ticked, with the rest of your library: tick or untick a document to change the list. Search it's **Name** box holds the name the copy gets, filled in from the file's; change it to keep both documents. To see the whole library and where it's kept, or to try a search without sending, use `/documents` and `/try` in a terminal chat.

A reply to a searched message opens with **From documents**: how many parts search added and their tokens. Press it for each part's rank and its similarity (by words: the words it matched, what was left out and why, and the words searched and ignored); **Show in Details** opens the reply's Documents section, with each part's tokens and where it sits in the prompt. For a meaning search, Details names the model and the minimum, and what was left out; when no part reaches the minimum, From documents and Details say so. Regenerating reuses the parts found when you sent the message. If Send is refused, the box says why: for each document, with **Stop searching it**, or for the chat (bge-m3 not fetched, the embedder failed, or a message too long for meaning search). Once each is fixed, send again.

### Change a document

To give a document's parts titles, or take out text that shouldn't be searched, edit your own file (add `## ` headings, say) and add it again under the same name: `clarity document add FILE --name glossary` (in a chat, `/documents add PATH glossary`). It replaces the document. Editing the copy in `local/documents/` changes nothing, and if its `parts.jsonl` isn't the one Clarity built, a chat that searches it refuses to send until you add the file again.

### Compare meaning and words

To see which finds better parts for your documents, make two profiles that differ only in `search_method`, both searching the same documents, and run a blind test with both ([Run and rate a blind test](#run-and-rate-a-blind-test)). Each searches by its own method; the run says when they got different parts.

### Compare placements

`placement = "start"` (the default) puts the parts before your message, and `placement = "end"` after it. To see which a model answers better from, make two profiles that differ only in `placement`, both searching the same documents, and run a blind test ([Run and rate a blind test](#run-and-rate-a-blind-test)):

```
uv run clarity run --openings questions.txt --profiles parts-first parts-last
```

Each opening is searched once for each profile, and every seed gets those parts. The run says when two profiles got different parts for an opening, and what differs between them.

## Follow branches

`/regen` and `/edit` add a branch next to the reply or message they replace, and `/undo` steps back so your next message starts one. Nothing is overwritten. `/branches` lists every place on the current path where the conversation branches, with each option's turn ID, and `/switch TURN_ID` goes to that branch's newest reply. Switching sends nothing; your next message continues from there.

## Sample several replies

`/regen 4` asks for four more replies to your last message, one after another. Each is a new branch with its own logged seed: the first reply's seed plus 1, 2, 3 and 4, so each can be replayed. The terminal shows a header before each reply and its seed, and `/branches` then lists them all. Ctrl-C stops the reply being written and skips the rest. With `--json`, a `reply_start` event comes before each reply, so a script can tell their text apart.

## Compare replies

To see what one setting does, hold the randomness fixed. Ask for a few replies (`/regen 2`), change the setting (`/set temperature 0.9`), then `/regen same`: it makes the message's replies again with the same seeds under the new value. `/compare` then lists every reply to the message, grouped by its settings, and for each pair with the same seed and different settings it shows the token and character where the two replies part. On the same machine and build, with the same seed and prompt, the setting is the only difference between their inputs. Where they part is where the new setting first changed a pick; it doesn't show how much the setting matters in general.

`/compare CALL_ID` does the same for an earlier message, from the ID of any of its replies, without moving you there. `/branches all` lists every branch end in the conversation, and every marked reply, so you can find one again.

## See a reply's token probabilities

On a Transformers profile, `/set logprobs 5` records, for each token of the next replies, the probability the model gave it and its five likeliest alternatives. `/probs` then summarizes the current reply: the ten tokens the model thought least likely, in reply order, each with the model's likeliest other tokens. The probabilities are the model's own, from its raw scores before any sampler: a token picked at temperature 1.2 shows the probability the model gave it, not the reshaped one. Every number is in the call's `logprobs` field in the log, and `/probs` with `--json` returns all of them. `/set logprobs 0` turns it off. llama.cpp profiles refuse the setting for now. In the web interface, the same numbers are in **Details**, under **Probabilities**; [Use the Transformers backend](#use-the-transformers-backend) shows the steps.

## Take a conversation to a notebook

`/export`, or `uv run clarity export ID` afterwards, writes the conversation to `logs/exports/ID.jsonl`: exact token IDs, where each message sits in the prompt, marks, notes and settings for every reply. Its fields are in [Export](reference.md#export). In the web interface, choose **Export for a Python notebook** from the chat's title menu: the sheet says where the file is, and **Download** saves a copy where your browser keeps downloads.

Numbers you recompute in a notebook are only comparable with the logged ones under the same conditions: the same weights, precision, device, attention implementation, library versions, PyTorch environment variables, deterministic-algorithms setting, steering and machine. So run Clarity's recompute check first, with the model you loaded:

```python
from clarity_chat.recompute import check

report = check("logs/exports/ID.jsonl", model, repo="ORG/NAME", revision="COMMIT")
print(report)
```

Load Gemma 4 as Clarity does, with its audio and vision configs set to None, from the model's folder:

```python
config = AutoConfig.from_pretrained(folder)
config.audio_config = config.vision_config = None
model = AutoModelForCausalLM.from_pretrained(
    folder, config=config, dtype=torch.bfloat16, attn_implementation="sdpa", use_safetensors=True
)
```

With the two configs left in, that call fails, because the audio tower has no `sdpa` attention.

The check lists each condition that differs from the export's. Transformers doesn't keep the commit a model was loaded from, so pass `revision`, the commit in the export's header, or the weights read "unknown". Pass `repo` too when you loaded the model from a local folder. A llama.cpp conversation ran on a GGUF file, so its weights never read as the same as a Transformers model's. For replies logged with `logprobs` on, it also recomputes each reply token's log-probability and reports the largest gap from the logged value, and how many steps' top choice changed. It recomputes step by step with the cache, as generation ran. In bf16, one pass over the whole reply gives different numbers even with nothing changed, because the two ways add things up in a different order and bf16 rounds differently: in one 60-token run of SmolLM2 on a CPU, the gap was 0.139, about as large as loading the model in float32 (0.133). In float32 it's far smaller: in another 60-token SmolLM2 run on a CPU, one pass gave a gap of 0.177 in bf16 and 0.0000145 in float32. With the same weights, precision, device, attention implementation and library versions on the same CPU, the gap was exactly 0 in our tests. The check gives facts and numbers, not a verdict: whether a difference matters depends on your experiment. For a steered conversation, run the check inside `clarity_chat.steering.applied` with the same vector, scale and blocks; the check compares them with the export's and lists steering if they differ. It needs the `hf` extra.

## Resume a conversation

`clarity chats` lists your conversations. To continue one, pass its ID, the log file's name:

```
uv run clarity chat --resume ID
```

A path to its log file works too. In a chat, `/resume ID` does the same, by ID only. The conversation continues where it left off, with its profile, settings, `/set` values and context size as they were logged, and new records go to the same log. Clarity lists what differs today, such as an edited profile file or a new llama.cpp build, and checks that the current reply's prompt still renders to the same token IDs. `/history` shows the conversation so far. To pick up today's profile instead, start a new conversation with `/new`.

## Label replies after the fact

`/mark` and `/note` label the current reply during a chat. To label any logged reply later, for example while reading a blind run's results, use:

```
uv run clarity mark runs/RUN_ID/calls.jsonl CALL_ID landed
uv run clarity note runs/RUN_ID/calls.jsonl CALL_ID "Drifts off topic after the first line."
```

These append to the log, with `origin` `review` (in-chat ones say `live`), and start no server. A failed call has no reply, so it can't be marked.

A mark without a label is `interesting`. To take a mark back, use `/unmark` in a chat or `clarity mark --remove LOG CALL_ID [LABEL]`; name the label if the reply has marks with different labels. A reply has one note: a new note replaces the old one, and Clarity says so (notes from before v0.2.0 have no ID, so they all stay). Nothing is deleted. The log keeps every mark and note, and records each removal and replacement.

## Replay a call

In a chat, `/replay` re-sends the current reply's call. In the web interface, open **Details** under a reply and press **Replay**; **Stop** stops it, and the result shows under the button, with what changed since the call. For any logged call, use:

```
uv run clarity replay logs/CONVERSATION_ID.jsonl CALL_ID
```

The call ID is on each reply's status line, and in the `call_id` field of the log. For a batch run, pass `runs/RUN_ID/calls.jsonl`.

Replay sends the logged token IDs and request body exactly, then reports one of:
- `Identical`: the same output token IDs and stop type.
- `Different`: where the output first differs, with both versions.

It then lists every recorded fact that differs from the original call, such as the llama.cpp build, a server flag, the GGUF hash, an environment variable or the machine. If the output differs and nothing recorded does, it says so: keep that log, as it shows nondeterminism Clarity didn't remove.

Replay uses the current `server.toml`. Add `--logged-flags` to start llama-server with the flags the call was made with. `clarity replay` exits with status 1 when the output differs. Failed and cancelled calls can't be replayed.

## Script a chat

`clarity chat --json` takes commands as JSON lines and answers with JSON lines, so a script, a test or another program can drive a chat:

```
printf '%s\n' '{"command": "say", "args": "Hello", "id": 1}' '{"command": "quit"}' \
  | uv run clarity chat --json --profile smol-chat
```

Each command's `result` carries its `id`. Every record written to the log also arrives as a `record` event, so a script sees exactly what was logged. [Scripting with --json](reference.md#scripting-with---json) lists the events.

## Use the Transformers backend

The Transformers backend runs the official weights in-process with PyTorch. It is given the same prompt token IDs as llama-server, but its output can still differ, because the arithmetic and the sampler code differ.

1. Install the extra: `uv sync --extra hf`.
2. Fetch the model with the `convert` or `safetensors` route; a ready-made GGUF has no official weights.
3. Set `backend = "transformers"` in a profile. It runs on MPS by default; add `device = "cpu"` or `device = "cuda"` to change it.

In the web interface, the new profile is in the picker once you save it under `local/profiles/`: open the picker under the message box and choose it. To see how likely each word was, open **Settings**, set `logprobs` to 5 under Probabilities, send a message, then open **Details** under the reply and open **Probabilities**: the reply is shaded token by token, and picking a token shows its chance and the likeliest others.

Some settings can't run on Transformers exactly as written; [Settings](reference.md#settings) lists which. When that can change the reply, the call is flagged `not_applied`, and `/settings` gives the reason. The most common case is sampler order: `defaults.toml` and most profiles, including `smol-chat`, list temperature last, so with temperature ≠ 1 and `top_p` < 1, every Transformers call is flagged.

The context size comes from the profile's `ctx_size` or `server.toml` here too, so the context guard is the same on both backends. A model with fewer positions than that won't start.

## Make a steering vector

A vector is one way to change how a model replies, and often not the best one. AxBench ([arXiv 2501.17148](https://arxiv.org/abs/2501.17148)), which tested steering on Gemma 2 (2B and 9B) with 500 concepts and an LLM judge, found that "for steering, … prompting outperforms all existing methods, followed by finetuning". Treat a vector as something to test.

`clarity vector make` makes a vector from a file of pairs of prompts. In each pair, the two sides have different system messages: the positive one asks for the behaviour you want more of, and the negative one for its opposite. Clarity makes the vector as the average difference between the two sides. Across seven behaviours on one model, one comparison found this beat directions found by PCA or by a classifier on most tasks; a classifier did better on three of them in open-ended replies (Im and Li, [arXiv 2502.02716](https://arxiv.org/abs/2502.02716)).

1. Write the pairs file. Describe the behaviour you want to a coding agent or a chat app, and give it `docs/pairs-prompt.md`, a prompt that writes a file in the form `make` reads. Or write one yourself: [The pairs file](reference.md#the-pairs-file) has the form and an example. Only the system message differs between the two sides.
2. Read the file before you make. The vector carries every way the two sides differ in the same direction across pairs, so a word on one side of every pair becomes part of it. Check that each system pair differs only in the behaviour, and not also in length, topic or tone you didn't mean.
3. Make the vector with a Transformers chat profile of the model you'll steer:

   ```
   uv run clarity vector make local/pairs/brief.toml --profile NAME
   ```

   Clarity reads each prompt at its last token, after the reply's opening words, where the model is about to write the reply. It writes `local/vectors/brief-NAME.gguf` and its record, then prints a readout.
4. Read the readout. It shows the first pair exactly as the model read it, so you can check the chat template and the system message. Then, for each block, three numbers: the activation size, the gap and the agreement ([what each means](reference.md#what-make-writes-and-prints)). Agreement is a hint, not a guarantee. In one study (Llama 2 7B Chat, 36 multiple-choice datasets, one block), datasets whose pairs' differences pointed more nearly the way of their average steered better (Braun et al., [arXiv 2505.22637](https://arxiv.org/abs/2505.22637)).
5. Paste the lines the readout ends with into a profile of your own of the same model in `local/profiles/`, on either backend (to use a shipped profile, copy it there first: [Write a profile](#write-a-profile)), and add `steering_scale = 1.0` to its `[settings]`. The vector's length at each block is the gap in the readout, so its scale isn't a repeng vector's ([why](reference.md#making-a-vector)). The lines steer one block, the middle one. Steering more blocks adds a push at each one, and later blocks read the sum. How much that changes a reply hasn't been measured: widen layers a block at a time and compare.
6. Compare steered and unsteered replies, as in [Steer a chat with a vector](#steer-a-chat-with-a-vector), step 6: `/set steering_scale 0`, then `/regen same` and `/compare`. Try a few values and compare.

To make one for Gemma 4 E4B, install the `hf` extra (`uv sync --extra hf`) and fetch `gemma-4-e4b-official`, Google's own weights: a 16.0 GB download. Clarity loads its text model only, which in one test took 14.93 GB of GPU memory on a Mac. These weights have no GGUF, so only the Transformers backend runs them, and the readout's lines say a Transformers profile. To steer Gemma on llama.cpp, paste the lines into a profile of `gemma-4-e4b`, the Q4_0 file a setup install runs. Its weights differ from the official ones by more than Q4_0's rounding, but its architecture, width and blocks are the same, so the vector's check passes.

In one test, the Q4_0 file followed such a vector closely on the first token. The vector was made from 8 made-up pairs on the official weights (Transformers, bf16) and added at block 21, at strength 1.0, to the Q4_0 file on llama.cpp (Metal). For each of 12 made-up prompts, the test took the official weights' likeliest first tokens, found how steering changed each one's log-probability in each file, and compared the two files' changes by rank correlation (1 when they put the tokens in the same order). It was +0.90 to +0.98 per prompt. With a random direction of the same length in the Q4_0 file instead, compared with the vector's changes on the official weights, it was +0.09 to +0.44, averaged over the prompts for each of 19 directions: below the vector's average for all 19. Reversed, at strength -1.0, the two files matched again (+0.96 on average), and neither file's reversed run matched the other's forward run (-0.13 and -0.32 on average). Of the official weights' 10 likeliest steered first tokens, 4 to 8 were among the Q4_0 file's 10. The two files also moved alike under each random direction (+0.89 to +0.98 on average), so they respond alike to a push at that block, not to this vector alone. The test covers the first token only, for one vector at one block, and the two runs differ in the backend, the weights and the quantisation at once, so it can't say which of them a difference comes from. Compare replies on the file you steer.

Even a vector that works moves some prompts the wrong way. Across 40 datasets on two models, some datasets did so for almost half their inputs, and how well a vector steered depended more on the dataset than on the model (Tan et al., [arXiv 2407.12404](https://arxiv.org/abs/2407.12404)). A vector that steers isn't proof that it is the idea you meant. In one study on two models, at one middle block each, adding a random direction at right angles to the vector changed little, so different vectors can steer the same way ([arXiv 2602.06801](https://arxiv.org/abs/2602.06801)).

## Steer a chat with a vector

A steering vector nudges a model's replies in a direction, such as calm rather than hurried, by adding a fixed vector to the output of some of its blocks. Clarity applies a vector it made ([Make a steering vector](#make-a-steering-vector)) or one made by another tool. Steering runs on both backends, on Gemma 4, OLMo 3 and Llama-architecture models such as SmolLM2 and Llama 3.1.

1. Make a vector with `clarity vector make` ([Make a steering vector](#make-a-steering-vector)), which writes it to `local/vectors/` and prints the lines for steps 4 and 5; then go on at step 6. Or make one elsewhere: [repeng](https://github.com/vgel/repeng) trains one from pairs of contrasting prompts and writes a file in llama.cpp's control-vector layout, which Clarity reads (repeng's own docs call its export work in progress). Install repeng from git in a separate environment: its PyPI release needs numpy below 2, which Clarity's dependencies don't allow. `tests/fixtures/steering/README.md` has a script that makes a small one for SmolLM2.
2. With repeng, leave the model's last block out (pass `hidden_layers` that stop before it). repeng includes it by default, but it reads that block's output after the model's final normalization (Transformers' last `hidden_states` entry), which isn't what the vector gets added to.
3. Copy the file into your project, for example to `local/vectors/calm.gguf`.
4. Add a `[steering]` table to a profile of your own in `local/profiles/`, on either backend (to steer a shipped profile, copy it there first: [Write a profile](#write-a-profile)), and set `steering_scale` (`4.0` here is for a repeng vector):

   ```toml
   [steering]
   file = "local/vectors/calm.gguf"

   [settings]
   steering_scale = 4.0
   ```

5. Start a chat with it. Clarity refuses the first time, because the vector isn't pinned yet, and gives the line to add to `[steering]`: `sha256 = "…"`. Add it. From then on, a changed file is refused by a new chat, a resumed one and a replay.
6. Chat. The status line after each reply ends with the scale, such as `steered ×4.0`, and `/settings` shows the vector's file, hash and blocks. To compare with an unsteered reply, `/set steering_scale 0`, then `/regen same`: it makes the message's replies again with their seeds, unsteered, each a branch next to its steered one. `/set steering_scale` with no value steers again at the profile's scale. `/compare` then groups the replies by scale and shows where each pair with the same seed parts ([Compare replies](#compare-replies)). A negative scale steers the other way. No scale is recommended: it depends on the model and the vector, so try small values and compare. A vector Clarity makes isn't scaled like a repeng vector, so the same scale is a different push ([how they compare](reference.md#making-a-vector)). On llama.cpp, each new strength starts the model again, since llama-server reads the vector when it starts. In two tests on an M2 Max, `gemma-4-e4b` started again in 0.7 s and 1.0 s after `/set steering_scale 0`. The chat keeps its turns, and the new strength is the chat's once the start works.

To steer only some of the vector's blocks, add `layers = [10, 11]` to the table. Block 0 can't be steered ([why](reference.md#steering-vectors)).

Each steered reply logs the vector's file, sha256, scale and blocks, and replays steered. `/replay` in a chat on llama.cpp runs a call only at the steering its model runs, and says which strength to set if that's the difference. From the terminal, `clarity replay LOG CALL_ID` starts llama-server with the call's own vector, strength and blocks. In a notebook, apply the same steering to a Transformers model with `clarity_chat.steering` before recomputing or reading activations:

```python
from pathlib import Path

from clarity_chat.recompute import check
from clarity_chat.steering import applied, load

vector = load(Path("local/vectors/calm.gguf"), "SHA256")
with applied(model, vector, 4.0):
    print(check("logs/exports/ID.jsonl", model, repo="ORG/NAME", revision="COMMIT"))
```

## Change the context size or a server flag

For one chat, choose a size under Context in Settings, or type `/set ctx_size 16384`. llama-server starts again at that size, and the chat keeps its turns; `/set ctx_size` with no value goes back to the profile's size. For one profile, set `ctx_size = 32768` at the top of the profile. Other profiles keep `ctx-size` from `server.toml`. Starting a conversation with a profile of another size restarts llama-server at that size, and `/settings` shows the size and where it came from.

For every profile, edit `[flags]` in `server.toml`. The next `clarity` command uses the new values, and every call logs them. A chat that is already running keeps the flags it started with, even if `/profile` restarts its server, so quit and start it again.

`clarity replay` runs a call at the context size it was logged with. `/replay` in a chat refuses a call logged at another size, because it would restart the chat's server at that size. To replay it there, set the chat to the call's size first, as the refusal says (`/set ctx_size 4096`, or that size in Settings), then replay.

Some flags are required, some values are fixed, and some flags are refused; see [server.toml](reference.md#servertoml) in the reference.

## When the model doesn't fit

In a running chat, three refusals have a way out in the chat itself: in Settings, or with `/set` in the terminal. The chat keeps its turns, and the change is logged with origin `chat`.

**Your message doesn't fit the context.** Nothing was sent, and your message is back in the box. The refusal ends "or raise ctx_size". Choose a larger size under Context in Settings, or type `/set ctx_size 16384`, then send again. A size larger than the model was trained on is refused.

**Meaning search's model doesn't fit beside the chat model.** bge-m3 runs on a second llama-server, and on a Mac the two together can need more than Metal's recommended maximum for the GPU. The refusal ends with its way out, `set search_method to "words"`. Choose words under Documents in Settings, or type `/set search_method words`, then send again. This chat searches by words from then on. When the refusal also names a lower `ctx_size`, a smaller context can make room instead.

**A size you just chose doesn't fit.** The refusal says why: the model would need more than Metal's recommended maximum, or the size is more than the model was trained on. The old size keeps running. Choose a smaller size, or type `/set ctx_size` to go back to the profile's size (in a resumed chat, the logged one).

A new chat whose model can't start at its profile's size has no chat to change yet. Lower `ctx_size` in its profile, or choose a profile with a smaller one; see [Change the context size or a server flag](#change-the-context-size-or-a-server-flag).

## Read your logs

Each chat conversation is one JSONL file in `logs/`, one record per line. `clarity chats`, or `/chats` in a chat, lists them newest first with their titles; `/title TEXT` names the current one. These [jq](https://jqlang.org/) queries are a starting point:

```
# each reply: call ID, stop reason, text
jq -c 'select(.type == "call") | {call_id, stop_type, output_text}' logs/CONVERSATION_ID.jsonl

# the settings of one call, with their origins
jq 'select(.call_id == "CALL_ID" and .type == "call") | .settings' logs/CONVERSATION_ID.jsonl

# calls with flags
jq -c 'select(.type == "call" and (.flags | length) > 0) | {call_id, flags}' logs/CONVERSATION_ID.jsonl

# marks, removed marks and notes
jq -c 'select(.type == "mark" or .type == "unmark" or .type == "note")' logs/CONVERSATION_ID.jsonl
```

[Log records](reference.md#log-records) in the reference lists the record types and their fields.

## Read activations from a logged call

Clarity doesn't read activations itself, but a logged call holds everything you need to read them in PyTorch.

1. Pick the call. A call made on the Transformers backend gives the closest match.
2. Load the weights named by `model.repo` and `model.revision`. These are the official weights only for a model fetched with the `convert` or `safetensors` route; a `gguf`-route model's source is usually `model.tokenizer_repo`.
3. For a Transformers call, match its `backend.dtype`, `backend.device` and `backend.attn_implementation`, and the `backend.torch` and `backend.transformers` versions.
4. Run one forward pass over `prompt_ids` followed by `output_ids`, so nothing is re-tokenized.
5. Check the reading, as below.

For a steered call, run the pass inside `clarity_chat.steering.applied` with the call's `request.steering`; its hooks run before the ones Transformers uses for `output_hidden_states`, so you read the steered values.

Even for a Transformers call, the reading is a new forward pass over the whole sequence, not the one that generated the reply, so small numeric differences are possible. A llama.cpp call is further off, because llama.cpp ran a GGUF (`model.gguf_sha256`) with its own kernels.

To check a reading, confirm that it predicts the logged output IDs, and report how often it agrees. For a greedy call (temperature 0), the top token at each position should be the logged one. For a sampled call, the top token differs wherever the sampler picked another, so check instead that each logged token is one the call's sampler settings would keep.

## Troubleshooting

Clarity's errors say what to do. These entries add context where a message can't.

**`n-gpu-layers = 'all', but llama-server put the model on the CPU …`** llama.cpp found no usable GPU. On a machine without one, set `n-gpu-layers = 0` in `server.toml`.

**`llama.cpp config file(s) … would apply settings before any flag.`** llama.cpp reads `/etc/llama.cpp/config.ini` and `~/.config/llama.cpp/config.ini` before its flags. Remove or rename them.

**`llama-server inherits [...] from your environment.`** A `GGML_*` or `LLAMA_*` variable is set in your shell. Some change how llama.cpp computes, so unset them unless you mean to keep them. Clarity removes `LLAMA_ARG_*` variables itself, since they act as flags.

**`PyTorch inherits [...] from your environment.`** The same, for `PYTORCH_*` or `TORCH_*` variables on the Transformers backend.

**`Tokenizer … is not in the Hugging Face cache.`** `clarity fetch` downloads a model's own tokenizer, but not a `template_repo`'s; see [Use another repo's chat template](#use-another-repos-chat-template).

**`The prompt is N tokens; with max_tokens M it needs more than the context …`** Nothing was sent, and your text and any attachments for it are back. Besides removing one of those attachments, a new conversation or a lower `max_tokens`, you can raise the chat's context size: choose a larger size under Context in Settings, or type `/set ctx_size 16384`. See [When the model doesn't fit](#when-the-model-doesnt-fit).

**`! flags: check_failed:…`** After the call, the server reported something other than what was sent. The reply is logged with the failed check; see [Call flags](reference.md#call-flags).
