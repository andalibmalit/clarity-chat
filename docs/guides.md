# Guides

Short recipes for common tasks. [Getting started](getting-started.md) covers installation and a first chat. [Reference](reference.md) lists every key and command.

## Write a profile

A profile is one experimental setup: a model, a mode, the prompt pieces and the settings. Save it as `local/profiles/NAME.toml` (create the folder the first time) and start it with `uv run clarity chat --profile NAME`. The profiles in `profiles/` ship with Clarity; copy one into `local/profiles/` under a new name to change it.

A chat-mode profile uses the model's chat template:

```toml
backend = "llama.cpp"
model = "smollm2-135m-instruct"   # a name from models.toml
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

Add `use_recommended = true` to a profile. Every setting the profile leaves out then comes from the model's own `generation_config.json`, at the commit pinned in `models.toml`, before falling back to `defaults.toml`.

`/settings` marks these values with origin `model`, and lists what the model recommends whether or not you use it. Only sampling values carry over: `repetition_penalty` becomes `repeat_penalty`, and `do_sample = false` becomes temperature 0. Everything else, including `max_new_tokens`, is listed as not applied.

## Add a model

Add an entry to `models.toml`, then run `uv run clarity fetch NAME`. Every revision is a full 40-character commit hash, so a name always means the same files.

To find a repo's current commit:

```
uv run python -c "from huggingface_hub import HfApi; print(HfApi().model_info('ORG/REPO').sha)"
```

Clarity can get a model in two ways.

**Convert the official weights** (`route = "convert"`). Clarity downloads the repo and converts it to a bf16 GGUF with the llama.cpp checkout's own converter. Use this when no official GGUF exists. The Transformers backend can then run the same weights.

```toml
[my-model]
route = "convert"
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

## Run and rate a blind test

1. Write the openings to a text file, separated by lines that are exactly `---`. Each opening is stripped of surrounding whitespace.
2. Run every profile on every opening:

   ```
   uv run clarity run --openings openings.txt --profiles A B C
   ```

   Add `--seeds 3` for three replies per opening and profile, with seeds 333, 334 and 335. The run's seeds replace the profiles' seeds, so two runs with the same profiles are comparable.

3. Give raters `runs/RUN_ID/blind.md`. Each reply has empty `Rating:` and `Note:` lines.
4. After rating, open `key.json`. It says which profile and seed each label came from. Each label's call ID, under `openings[].labels.<label>.call_id`, is what you pass to [`clarity mark`](#label-replies-after-the-fact) or [`clarity replay`](#replay-a-call).

Before any model loads, Clarity checks every profile and renders every opening. It stops if the profiles' `max_tokens` differ (reply length would give a setup away), a profile is listed twice, or an opening doesn't fit the context. During the run, any error or failed check stops it, and no `blind.md` is written.

Each call is its own conversation. Profiles are read once, at the start, so editing one during a run changes nothing. `blind.md` strips the whitespace around each reply; `calls.jsonl` keeps replies exactly as generated.

## Follow branches

`/regen` and `/edit` add a branch next to the reply or message they replace, and `/undo` steps back so your next message starts one. Nothing is overwritten. `/branches` lists every place on the current path where the conversation branches, with each option's turn ID, and `/switch TURN_ID` goes to that branch's newest reply. Switching sends nothing; your next message continues from there.

## Sample several replies

`/regen 4` asks for four more replies to your last message, one after another. Each is a new branch with its own logged seed: the first reply's seed plus 1, 2, 3 and 4, so each can be replayed. The terminal shows a header before each reply and its seed, and `/branches` then lists them all. Ctrl-C stops the reply being written and skips the rest. With `--json`, a `reply_start` event comes before each reply, so a script can tell their text apart.

## See a reply's token probabilities

On a Transformers profile, `/set logprobs 5` records, for each token of the next replies, the probability the model gave it and its five likeliest alternatives. `/probs` then summarizes the current reply: the ten tokens the model thought least likely, in reply order, each with the model's likeliest other tokens. The probabilities are the model's own, from its raw scores before any sampler: a token picked at temperature 1.2 shows the probability the model gave it, not the reshaped one. Every number is in the call's `logprobs` field in the log, and `/probs` with `--json` returns all of them. `/set logprobs 0` turns it off. llama.cpp profiles refuse the setting for now.

## Take a conversation to a notebook

`/export`, or `uv run clarity export ID` afterwards, writes the conversation to `logs/exports/ID.jsonl`: exact token IDs, where each message sits in the prompt, marks, notes and settings for every reply. Its fields are in [Export](reference.md#export).

Numbers you recompute in a notebook are only comparable with the logged ones under the same conditions: the same weights, precision, device, attention implementation, library versions, PyTorch environment variables, deterministic-algorithms setting, steering and machine. So run the check first, with the model you loaded:

```python
from clarity_chat.recompute import check

report = check("logs/exports/ID.jsonl", model, repo="ORG/NAME", revision="COMMIT")
print(report)
```

It lists each condition that differs from the export's. Transformers doesn't keep the commit a model was loaded from, so pass `revision`, the commit in the export's header, or the weights read "unknown". Pass `repo` too when you loaded the model from a local folder. A llama.cpp conversation ran on a GGUF file, so its weights never read as the same as a Transformers model's. For replies logged with `logprobs` on, it also recomputes each reply token's log-probability and reports the largest gap from the logged value, and how many steps' top choice changed. It recomputes step by step with the cache, as generation ran. In bf16, one pass over the whole reply gives different numbers even with nothing changed, because the two ways add things up in a different order and bf16 rounds differently: in one 60-token run of SmolLM2 on a CPU, the gap was 0.139, about as large as loading the model in float32 (0.133). In float32 it's far smaller: in another 60-token SmolLM2 run on a CPU, one pass gave a gap of 0.177 in bf16 and 0.0000145 in float32. With the same weights, precision, device, attention implementation and library versions on the same CPU, the gap was exactly 0 in our tests. The check gives facts and numbers, not a verdict: whether a difference matters depends on your experiment. For a steered conversation, run the check inside `clarity_chat.steering.applied` with the same vector, scale and blocks; the check compares them with the export's and lists steering if they differ. It needs the `hf` extra.

## Resume a conversation

`clarity chats` lists your conversations. To continue one, pass its ID:

```
uv run clarity chat --resume CONVERSATION_ID
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

In a chat, `/replay` re-sends the current reply's call. For any logged call, use:

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
2. Fetch the model with the `convert` route; a ready-made GGUF has no official weights.
3. Set `backend = "transformers"` in a profile. It runs on MPS by default; add `device = "cpu"` or `device = "cuda"` to change it.

Some settings can't run on Transformers exactly as written; [Settings](reference.md#settings) lists which. When that can change the reply, the call is flagged `not_applied`, and `/settings` gives the reason. The most common case is sampler order: `defaults.toml` and most profiles, including `smol-chat`, list temperature last, so with temperature ≠ 1 and `top_p` < 1, every Transformers call is flagged.

The context size comes from the profile's `ctx_size` or `server.toml` here too, so the context guard is the same on both backends. A model with fewer positions than that won't start.

## Steer a chat with a vector

A steering vector nudges a model's replies in a direction, such as calm rather than hurried, by adding a fixed vector to the output of some of its blocks. Clarity applies a vector made by another tool; it never makes one. Steering runs on the Transformers backend, on OLMo 3 and on Llama-architecture models such as SmolLM2.

1. Make a vector elsewhere. [repeng](https://github.com/vgel/repeng) trains one from pairs of contrasting prompts and writes it in llama.cpp's control-vector format, which Clarity reads. Install repeng from git in a separate environment: its PyPI release needs numpy below 2, which Clarity's dependencies don't allow. `tests/fixtures/steering/README.md` has a script that makes a small one for SmolLM2.
2. Leave the model's last block out (pass `hidden_layers` that stop before it). repeng includes it by default, but on Transformers 5 it reads that block's output after the model's final normalization, which isn't what the vector gets added to.
3. Copy the file into your project, for example to `local/vectors/calm.gguf`.
4. Add a `[steering]` table to a Transformers profile, and set `steering_scale`:

   ```toml
   [steering]
   file = "local/vectors/calm.gguf"

   [settings]
   steering_scale = 4.0
   ```

5. Start a chat with it. Clarity refuses the first time, because the vector isn't pinned yet, and gives the line to add to `[steering]`: `sha256 = "…"`. Add it. From then on, a changed file is refused by a new chat, a resumed one and a replay.
6. Chat. The status line after each reply ends with `steered ×4.0`, and `/settings` shows the vector's file, hash and blocks. To compare with an unsteered reply, use a fixed seed or temperature 0, `/set steering_scale 0`, then `/edit` with the same message: the new reply is a branch next to the steered one. A negative scale steers the other way.

To steer only some of the vector's blocks, add `layers = [10, 11]` to the table. Block 0 can't be steered: llama.cpp's control-vector loader refuses it, and Clarity keeps a vector usable on both.

Each steered reply logs the vector's file, sha256, scale and blocks, and replays steered. In a notebook, apply the same steering with `clarity_chat.steering` before recomputing or reading activations:

```python
from pathlib import Path

from clarity_chat.recompute import check
from clarity_chat.steering import applied, load

vector = load(Path("local/vectors/calm.gguf"), "SHA256")
with applied(model, vector, 4.0):
    print(check("logs/exports/ID.jsonl", model, repo="ORG/NAME", revision="COMMIT"))
```

## Change the context size or a server flag

For one profile, set `ctx_size = 32768` at the top of the profile. Other profiles keep `ctx-size` from `server.toml`. Starting a conversation with a profile of another size restarts llama-server at that size, and `/settings` shows the size and where it came from.

For every profile, edit `[flags]` in `server.toml`. The next `clarity` command uses the new values, and every call logs them. A chat that is already running keeps the flags it started with, even if `/profile` restarts its server, so quit and start it again.

`clarity replay` runs a call at the context size it was logged with. `/replay` in a chat refuses a call logged at another size, because it would restart the chat's server at that size.

Some flags are required, some values are fixed, and some flags are refused; see [server.toml](reference.md#servertoml) in the reference.

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
2. Load the weights named by `model.repo` and `model.revision`. These are the official weights only for a model fetched with the `convert` route; a `gguf`-route model's source is usually `model.tokenizer_repo`.
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

**`The prompt is N tokens; with max_tokens M it needs more than the context …`** Besides a new conversation or a lower `max_tokens`, you can raise the profile's `ctx_size` (or `ctx-size` in `server.toml`) and restart the chat.

**`! flags: check_failed:…`** After the call, the server reported something other than what was sent. The reply is logged with the failed check; see [Call flags](reference.md#call-flags).
