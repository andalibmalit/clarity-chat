# Guides

Short recipes for common tasks. [Getting started](getting-started.md) covers installation and a first chat. [Reference](reference.md) lists every key and command.

## Write a profile

A profile is one experimental setup: a model, a mode, the prompt pieces and the settings. Save it as `profiles/NAME.toml` and start it with `uv run clarity chat --profile NAME`. Unknown keys and setting names are errors, so most typos stop the profile from loading. Keys inside `template_kwargs` aren't checked, because the chat template defines them; a misspelled one is silently ignored.

A chat-mode profile uses the model's chat template:

```toml
backend = "llama.cpp"
model = "smollm2-135m-instruct"   # a name from models.toml
mode = "chat"
system = { file = "../prompts/system.txt" }   # or a string; leave out for none
priming = { file = "../prompts/priming.toml" } # earlier messages placed before the conversation

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

The document is the header, the priming text, one `User: …` or `Assistant: …` line per turn, and a final `Assistant:`. Generation stops at a new line starting with `User:`.

Tips:
- File paths are relative to the profile, and files are read byte for byte. A trailing newline your editor added is part of the prompt; `/prompt repr` shows it.
- A priming file holds only `[[messages]]` tables, each with `role` (`"user"` or `"assistant"`) and `content`. See `prompts/priming.toml`.
- Template options such as `enable_thinking` go in a `[template_kwargs]` table.
- To use another repo's chat template, set `template_repo` and `template_revision` (a full commit hash) together. The model's own tokenizer still does the tokenizing. `fetch` doesn't download that repo, so put its tokenizer files in the Hugging Face cache yourself:

  ```
  uv run python -c "from huggingface_hub import snapshot_download; from clarity_chat.render import TOKENIZER_FILES; snapshot_download('ORG/REPO', revision='COMMIT', allow_patterns=TOKENIZER_FILES)"
  ```

- Every setting you leave out comes from `defaults.toml`, which turns every sampler off (see [Settings](reference.md#settings)), unless `use_recommended` fills it from the model first.

## Compare two setups

Make two profiles that differ in one key, and keep everything else the same. Settings you don't set come from `defaults.toml` for both, so the only difference is the one you made.

A fixed seed makes a reply replayable, not representative: at temperature 0.6, another seed gives another reply. Compare setups over several openings or seeds with a [blind test](#run-and-rate-a-blind-test).

To see what a chat template contributes, compare a chat profile with a text profile on the same instruct model, like the `2-system` and `4-raw-instruct` profiles.

## Use a model's recommended settings

Add `use_recommended = true` to a profile. Every setting the profile leaves out then comes from the model's own `generation_config.json`, at the commit pinned in `models.toml`, before falling back to `defaults.toml`.

`/settings` marks these values with origin `model` and lists what the model recommends whether or not you use it. Only sampling values carry over: `repetition_penalty` becomes `repeat_penalty`, and `do_sample = false` becomes temperature 0. Everything else, including `max_new_tokens`, is listed as not applied.

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

Clarity checks on every call that the server's own decoding of the token IDs matches the prompt text. If a GGUF and its `tokenizer_repo` don't agree, that check fails.

## Run and rate a blind test

1. Write the openings to a text file, separated by lines that are exactly `---`. Each opening is stripped of surrounding whitespace.
2. Run every profile on every opening:

   ```
   uv run clarity run --openings openings.txt --profiles A B C
   ```

   Add `--seeds 3` for three replies per opening and profile, with seeds 333, 334 and 335. The run's seeds replace the profiles' seeds, so two runs with the same profiles are comparable.

3. Give raters `runs/<run ID>/blind.md`. Each reply has empty `Rating:` and `Note:` lines.
4. Open `key.json` only after rating. It says which label came from which profile and seed, and gives each label's call ID (under `openings[].labels.<label>.call_id`), which you need to [label](#label-replies-after-the-fact) or [replay](#replay-a-call) a reply.

Before any model loads, Clarity checks every profile and renders every opening. It stops if the profiles' `max_tokens` differ (reply length would give a setup away), a profile is listed twice, or an opening doesn't fit the context. During the run, any error or failed check stops it, and no `blind.md` is written.

## Label replies after the fact

`/mark` and `/note` label the current reply during a chat. To label any logged reply later, for example while reading a blind run's results, use:

```
uv run clarity mark runs/<run ID>/calls.jsonl CALL_ID landed
uv run clarity note runs/<run ID>/calls.jsonl CALL_ID "Drifts off topic after the first line."
```

These append one record to the log, with `origin` `review` (in-chat ones say `live`), and start no server. A failed call has no reply, so it can't be marked.

## Replay a call

In a chat, `/replay` re-sends the current reply's call. For any logged call, use:

```
uv run clarity replay logs/<conversation ID>.jsonl CALL_ID
```

The call ID is on each reply's status line, and in the `call_id` field of the log. For a batch run, pass `runs/<run ID>/calls.jsonl`.

Replay sends the logged token IDs and request body exactly, then reports one of:
- `Identical`: the same output token IDs and stop type.
- `Different`: where the output first differs, with both versions.

It then lists every recorded fact that differs from the original call, such as the llama.cpp build, a server flag, the GGUF hash, an environment variable or the machine. If the output differs and nothing recorded does, it says so: keep that log, as it shows nondeterminism Clarity didn't remove.

Replay uses the current `server.toml`. Add `--logged-flags` to start llama-server with the flags the call was made with. `clarity replay` exits with status 1 when the output differs. Failed and cancelled calls can't be replayed.

## Use the Transformers backend

The Transformers backend runs the official weights in-process with PyTorch. It is given the same prompt token IDs as llama-server; its output can still differ, since the arithmetic and sampler code differ.

1. Install the extra: `uv sync --extra hf`.
2. Fetch the model with the `convert` route; a ready-made GGUF has no official weights.
3. Set `backend = "transformers"` in a profile. It runs on MPS by default; add `device = "cpu"` or `device = "cuda"` to change it.

Transformers applies its samplers in a fixed order and has no `presence_penalty`, `frequency_penalty` or `repeat_last_n`. When a profile asks for something it can't do, the call is flagged `not_applied`, and `/settings` gives the reason. The Check 0 profiles list temperature last, which Transformers can't do, so on Transformers they run with a different sampler order and each call is flagged.

The Transformers backend also takes its context size from `ctx-size` in `server.toml`, so the context guard is the same on both backends. A model with fewer positions than that won't start.

## Change the context size or a server flag

Edit `[flags]` in `server.toml`. The context size is `ctx-size`. The next `clarity` command uses the new values, and every call logs them. A chat that is already running keeps the flags it started with, even if `/profile` restarts its server, so quit and start it again.

Some flags are required, some values are fixed (one slot, no prompt cache, no context shift), and some flags are refused, such as `model`, `port` and anything that loads another model. Values of `auto` or `-1` are refused, so nothing is left to llama.cpp's defaults. See [server.toml](reference.md#servertoml) in the reference.

## Read your logs

Each conversation is one JSONL file in `logs/`, one record per line. These [jq](https://jqlang.org/) queries are a starting point:

```
# each reply: call ID, stop reason, text
jq -c 'select(.type == "call") | {call_id, stop_type, output_text}' logs/FILE.jsonl

# the settings of one call, with their origins
jq 'select(.call_id == "CALL_ID" and .type == "call") | .settings' logs/FILE.jsonl

# calls with flags
jq -c 'select(.type == "call" and (.flags | length) > 0) | {call_id, flags}' logs/FILE.jsonl

# marks and notes
jq -c 'select(.type == "mark" or .type == "note")' logs/FILE.jsonl
```

[Log records](reference.md#log-records) in the reference lists the record types and their fields.

## Troubleshooting

**`… is at …, but server.toml pins …. Check out that commit and rebuild.`** The llama.cpp checkout is at another commit. Run `git -C vendor/llama.cpp checkout <llama_cpp_commit from server.toml>` and rebuild.

**`build_info … doesn't match the checkout commit …: rebuild llama-server.`** The checkout is right, but the binary was built from another commit. Rebuild it.

**`llama-server not found under vendor/llama.cpp/build.`** Build it; see [Getting started](getting-started.md#install-clarity-and-build-llamacpp).

**`n-gpu-layers = 'all', but llama-server put the model on the CPU …`** llama.cpp found no usable GPU. On a machine without one, set `n-gpu-layers = 0` in `server.toml`.

**`llama.cpp config file(s) … would apply settings before any flag.`** llama.cpp reads `/etc/llama.cpp/config.ini` and `~/.config/llama.cpp/config.ini` before its flags. Remove or rename them.

**`llama-server inherits [...] from your environment.`** A `GGML_*` or `LLAMA_*` variable is set in your shell. Some change how llama.cpp computes, so unset them unless you mean to keep them. Clarity removes `LLAMA_ARG_*` variables itself, since they act as flags.

**`Model '…' is not fetched.` or `Tokenizer … is not in the Hugging Face cache.`** Run `uv run clarity fetch NAME`.

**`… changed since it was fetched (size or modified time differs from models.lock).`** Run `uv run clarity verify NAME`. If the sha256 still matches, it updates the lock and you can carry on.

**`The prompt is N tokens; with max_tokens M it needs more than the context …`** The conversation no longer fits. Start a new one with `/new`, lower `max_tokens`, or raise `ctx-size` in `server.toml` and restart the chat.

**`temperature = … has no effect, because 'temperature' is not in samplers …`** A setting only acts if its sampler is in `samplers`. Add the sampler, or set the value back to neutral.

**`! flags: check_failed:…`** After the call, the server reported something other than what was sent. The reply is logged with the failed check; see [Call flags](reference.md#call-flags).
