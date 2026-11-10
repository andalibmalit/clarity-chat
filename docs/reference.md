# Reference

Every command, key and record field a user sees. For task-based instructions, see [Guides](guides.md); for the reasons behind these rules, see [Design](design.md).

## Commands

| Command | What it does |
|---|---|
| `clarity chat --profile NAME` | Chat with `profiles/NAME.toml` or `local/profiles/NAME.toml`. NAME can also be a path to a `.toml` file. |
| `clarity run --openings FILE --profiles A B … [--seeds N]` | Collect replies for a blind test into `runs/RUN_ID/`. FILE holds openings separated by lines that are exactly `---`. `--seeds N` (default 1) gives N replies per opening and profile, with seeds 333, 334 and so on. |
| `clarity replay LOG CALL_ID [--logged-flags]` | Re-send a logged call and compare the output. `--logged-flags` starts llama-server with the call's logged flags instead of `server.toml`'s. |
| `clarity mark LOG CALL_ID [LABEL]` | Label a logged call's reply while reviewing, in `logs/` or a run's `calls.jsonl` (default label `interesting`). Appends one record and starts no server. With `--remove`, removes the mark instead. |
| `clarity note LOG CALL_ID TEXT` | Add a note to a logged call's reply while reviewing, replacing its note. Quote TEXT if it has spaces. |
| `clarity fetch NAME` | Download (and, for the `convert` route, convert) a model from `models.toml`, and record it in `models.lock`. |
| `clarity verify [NAME …]` | Re-hash fetched models against `models.lock`; with no names, every fetched model. |

`--root PATH` sets the project folder, and goes before the command: `clarity --root PATH chat --profile NAME`. By default, the project folder is the nearest folder, from the current one upwards, that holds both `defaults.toml` and `server.toml`. `mark` and `note` don't need a project folder, only the log file.

Exit status:
- `0`: success, or `replay` found the output identical.
- `1`: a configuration or backend error (printed as `clarity: …`), a stopped `run`, a `replay` whose output differs, or a `verify` that failed.
- `2`: wrong command-line arguments.
- `130`: interrupted with Ctrl-C.

## Chat commands

Enter sends a message. Alt+Enter, or Esc then Enter, starts a new line. Pasted text keeps its newlines. Ctrl-C while the model works cancels the call. If part of the reply has arrived, it's kept and logged. If nothing has, no reply is added: the call is logged like a failed one, and your message comes back in the input to edit or send again.

| Command | What it does |
|---|---|
| `/help` | List the commands. |
| `/say TEXT` | Send a message. Typing it without a command does the same; scripts use `/say`. |
| `/prompt` | The exact prompt text of the current reply's call (or of your last message, if its call failed), the BOS decision, and the token IDs. |
| `/prompt repr` | The same, with each line shown as a Python string, so spaces, tabs and newlines are visible. |
| `/settings` | Every setting, its value and origin; the fixed values sent for the backend's other fields; the model's recommended values; what wasn't applied; the context size and stop strings; the end-of-reply tokens llama.cpp read from the model file. |
| `/regen` | Another reply to your last message, as a new branch. The k-th regeneration uses the first reply's seed plus k. |
| `/edit [TEXT]` | Replace your last message and get a new reply, as a new branch. With no text, the input is prefilled with your last message. The new message uses the profile's seed. |
| `/undo` | Go back past the last exchange; your next message starts a new branch. Nothing is deleted. |
| `/mark [LABEL]` | Label the current reply, for example `landed` or `dropped`. Without a label, the mark is `interesting`. |
| `/unmark [LABEL]` | Remove the current reply's mark. If it has marks with different labels, name the one to remove. The log keeps the mark and records its removal. |
| `/note TEXT` | Add a note to the current reply. A reply has one note: a new one replaces it, and the log keeps both. |
| `/replay` | Re-send the current reply's call on the running server and compare the output. |
| `/new` | Start a new conversation (and log file) with the same profile, re-read from its file. |
| `/profile NAME` | Start a new conversation with another profile. The server restarts only if the model changes. |
| `/cancel` | Stop the call in progress, like Ctrl-C. It's the one command that runs during a call, for scripts and the web interface. |
| `/quit` | Stop the server and exit. `/exit`, `/q` and Ctrl-D also work. |
| `//text` | Send a message that starts with `/`. |

After each reply, a status line shows the stop reason, the reply length in tokens, the time, the seed (with its origin unless it came from the profile), the context used (prompt plus reply tokens, of the context size) and the call ID. Warnings and [call flags](#call-flags) appear above it, marked `!`. A warning appears when the prompt uses more than 80% of the context.

## Profile keys

A profile is a TOML file in `profiles/` (shipped with Clarity) or `local/profiles/` (yours; git ignores it). Its name is the file name without `.toml`, and a name in both folders is an error. Unknown keys are errors, and the allowed keys depend on `mode`.

All profiles:

| Key | Required | Value |
|---|---|---|
| `mode` | yes | `"chat"` (the model's chat template) or `"text"` (a plain document). |
| `backend` | yes | `"llama.cpp"` or `"transformers"`. |
| `model` | yes | A model name from `models.toml`. |
| `settings` | no | A table of [settings](#settings). Unset ones come from `defaults.toml`. |
| `use_recommended` | no | `true` fills unset settings from the model's `generation_config.json` before `defaults.toml`. Default `false`. |
| `device` | no | Transformers backend only: `"mps"` (default), `"cpu"` or `"cuda"`. |
| `ctx_size` | no | The context size in tokens: the prompt and the reply together must fit in it. Default: `ctx-size` in `server.toml`. A larger context takes more memory, and a size above what the model was trained on is refused. Starting a profile with another size restarts llama-server; the Transformers backend only checks the new size. |

Chat mode, all optional:

| Key | Value |
|---|---|
| `system` | The system message: a string, or `{ file = "PATH" }`. Without it, no system message is sent, and some templates then write their own. |
| `priming` | Earlier messages placed after the system message and before the conversation: a list of `{ role, content }` tables (role `"user"` or `"assistant"`), or `{ file = "PATH" }` naming a TOML file that holds only `[[messages]]` tables. |
| `template_repo`, `template_revision` | Render with another Hugging Face repo's chat template, at a full 40-character commit. Set both or neither. Default: the model's tokenizer repo. |
| `template_kwargs` | A table of extra arguments for the chat template, such as `enable_thinking`. Clarity's own arguments (`tokenize`, `add_generation_prompt`, `chat_template` and similar) are refused. A key the template never reads is a warning when the conversation starts. |

Text mode:

| Key | Required | Value |
|---|---|---|
| `user_name` | yes | The user's name in the document, without outer spaces. Generation stops at a new line starting with `user_name:`. |
| `assistant_name` | yes | The assistant's name, without outer spaces. |
| `header` | no | Text at the top of the document: a string or `{ file = "PATH" }`. Default empty. |
| `priming` | no | Text placed after the header, before the conversation: a string or `{ file = "PATH" }`. Default empty. |

A text-mode prompt is `header + priming`, then `user_name: MESSAGE` and `assistant_name:REPLY` lines (the reply keeps the leading space the model wrote), then a final `assistant_name:`.

File paths are relative to the profile file. Files are read byte for byte, so line endings and a trailing newline are part of the prompt.

## Settings

Settings use the same names on every backend. Each resolved value records its origin: `profile`, `model` (from `use_recommended`), `default` (from `defaults.toml`), `regen` (a `/regen` seed) or `run` (a batch-run seed). Fixed values Clarity sends for a backend's other fields have origin `backend`.

| Setting | Value | Neutral (`defaults.toml`) | Needs sampler |
|---|---|---|---|
| `temperature` | number ≥ 0; 0 is greedy | 1.0 | `temperature` |
| `top_k` | integer ≥ 0; 0 is off | 0 | `top_k` |
| `top_p` | number from 0 to 1 | 1.0 | `top_p` |
| `min_p` | number from 0 to 1 | 0.0 | `min_p` |
| `repeat_penalty` | number > 0 | 1.0 | `penalties` |
| `repeat_last_n` | integer ≥ 0: how many recent tokens the penalties look at | 64 | |
| `presence_penalty` | number | 0.0 | `penalties` |
| `frequency_penalty` | number | 0.0 | `penalties` |
| `seed` | integer from 0 to 4294967294, or `"random"` (Clarity draws one per call and logs it) | `"random"` | |
| `max_tokens` | integer ≥ 1: the longest reply | 1024 (a cap, not a neutral value) | |
| `samplers` | the order and membership of the sampler chain, from `penalties`, `top_k`, `top_p`, `min_p`, `temperature` | all five, in that order | |

A sampler not listed in `samplers` doesn't run. A setting whose sampler isn't listed must keep its neutral value; anything else is an error when the profile loads.

The Transformers backend applies its samplers in a fixed order: penalties, temperature, `top_k`, `top_p`, `min_p`. It has no `presence_penalty`, `frequency_penalty` or `repeat_last_n`. These are recorded as not applied, and a call is flagged `not_applied` when the difference can change the reply.

## models.toml

Each table is one model. Every revision is a full 40-character commit hash.

| Key | Routes | Value |
|---|---|---|
| `route` | both | `"convert"`: download the repo and convert it to a bf16 GGUF. `"gguf"`: download a ready-made GGUF. |
| `repo`, `revision` | both | The Hugging Face repo and commit to download. |
| `tokenizer_repo`, `tokenizer_revision` | both | The tokenizer (and chat template and recommended settings) Clarity uses. For a converted model, usually the same as `repo` and `revision`. |
| `file` | `gguf` | The GGUF file's name in the repo. |
| `sha256` | `gguf` | The file's sha256, 64 lowercase hex characters. A mismatch stops the fetch. |

`clarity fetch` writes what it resolved to `models.lock` (JSON, committed): paths, size, modification time and sha256 of the GGUF, the converter's package versions, and the model's recommended settings. Clarity never edits `models.toml`.

## server.toml

| Key | Value |
|---|---|
| `llama_cpp_dir` | The llama.cpp checkout, relative to the project folder. |
| `llama_cpp_commit` | The commit the checkout must be at. Any other commit stops the server from starting. |
| `startup_timeout_s` | Seconds to wait for the server to become healthy. |
| `[flags]` | llama-server flags by long name, without dashes. `true` passes a bare flag; `false` leaves it out. |

Every flag that affects output must be set explicitly. The shipped `server.toml` sets them all, with comments, and Clarity names any that is missing or has the wrong value. These values are fixed: one slot (`parallel = 1`), no prompt cache, no context shift, and `fit = "off"`.

Refused: `auto` and `-1` as values; the flags `model`, `host`, `port`, `api-key`, `api-key-file`, `slots`, `no-slots`, `ui`, `no-ui`, `webui` and `no-webui` (Clarity sets these), `context-shift`, `override-kv`, `lora`, `control-vector`, `model-draft`, `backend-sampling`; and any flag starting with `hf-`, `models-`, `spec-`, `ui-` or `webui-`.

Unless `n-gpu-layers` is 0, Clarity stops if llama.cpp puts the model on the CPU.

## Files Clarity writes

| Path | Contents |
|---|---|
| `models.lock` | What `fetch` resolved for each model. Committed. |
| `models/NAME/` | A converted model's GGUF, plus `convert.log` and the converter's package versions. A downloaded GGUF stays in the Hugging Face cache instead. |
| `logs/CONVERSATION_ID.jsonl` | One chat conversation, one record per line. The ID is the start time and a random suffix, such as `20261108-142530-a1b2c3`. |
| `server-ID.log` | One llama-server's own output, next to the log it serves: in `logs/` for a chat, in `runs/RUN_ID/` for a batch run. |
| `runs/RUN_ID/calls.jsonl` | Every call of a batch run, after a first `run` record. |
| `runs/RUN_ID/blind.md` | Openings and shuffled, labelled replies, for raters. Written only if the whole run succeeds. |
| `runs/RUN_ID/key.json` | Which label is which profile, seed and call, and the shuffle seed. |
| `local/clarity.lock` | Locked while `chat`, `run` or `replay` runs, so only one of them runs in a project at a time. Empty. |
| `local/clarity.json` | Who holds the lock: process ID, command and start time, and the running llama-server's port, process ID and key. Readable only by you on Linux and macOS (mode 0600); on Windows it gets your folder's permissions. Removed on a normal exit; one left behind means Clarity didn't exit normally, and the next start uses it to stop that server. |

`logs/`, `runs/`, `models/` and `local/` are gitignored. Logs are append-only: a line is never rewritten.

## Log records

Each line is one JSON record with `schema_version` (currently 1), `type` and `time` (UTC). Every record except `run` and `torn_line` also has `conversation_id`. Readers should ignore keys they don't know.

| `type` | Written | Main fields |
|---|---|---|
| `conversation` | at the start of each conversation | `profile` (name, path, full TOML text and every file it read), `settings`, `model`, `client` (commit and dirty flag), `machine`, `warnings`, `ctx_size` (value and origin: `profile` or `server.toml`) |
| `server_start` | once per server per log file, before its first call | `server_id`, `argv`, `flags`, `api_key` (always "random per start; value not logged"), `env_removed`, `env_llama_ggml_kept`, `llama_cpp` (commit, dirty), `build_info`, `placement`, `model_file` (`eog_tokens`: the tokens that end a reply, read from the GGUF), `memory` (per GPU: its limit in MiB, and the model, context cache and compute buffers on it), `graph_splits` (how many pieces llama.cpp cut the compute graph into, for the prompt and the reply), `props`, `startup_output`, `warnings`, `machine`. Transformers: `device`, `dtype`, `attn_implementation`, `torch`, `transformers`, `eos_token_id` and similar. |
| `turn` | for each message and reply | `turn_id`, `parent_id`, `role`, `origin` (`user`, `model` or `edit`), `text`, `call_id`, and `regen_of` or `edit_of` for branches |
| `call` | for every call, including failed and cancelled ones | see below |
| `mark`, `unmark`, `note` | for `/mark`, `/unmark`, `/note`, `clarity mark` and `clarity note` | `call_id`, `turn_id`, `label` or `text`, and `origin`: `live` (in a chat) or `review` (afterwards). A `mark` has a `mark_id`; an `unmark` names the `mark_id` it removes (null for a mark from before v0.2.0, which removes that reply's ID-less marks with its `label`). A `note` has a `note_id` and `replaces`: the `note_id` of the note it replaces, or null. Notes from before v0.2.0 have no ID, so they all still show. |
| `replay` | for every replay | `replay_id`, `call_id`, `n_ctx_from`, `identical`, `compared`, `differences`, `not_recorded` (facts an older log didn't record, so not compared), `unexplained_difference`, and the new output |
| `torn_line` | before the first record written after a line that was cut off (a crash or a full disk mid-write) | `line` (its line number) and `bytes` (how much of it was written). The cut line stays as it was. Readers skip a line that isn't a record only when the record after it is a `torn_line` naming it; any other malformed line is an error. |
| `run` | first line of a batch run's `calls.jsonl` | `run_id`, `openings_file`, `openings_sha256`, `profiles`, `seeds`, the split and trim rules |

A `call` record holds:
- **Where it sits:** `call_id`, `parent_id` (your message's turn), `regen_of` (for a `/regen`), `run` (in a batch run: the run ID and opening).
- **Inputs:** `profile`, `settings` (each with its origin), `seed_drawn`, `recommended` (with `use_recommended`), `model` (the `models.toml` entry, plus the GGUF's sha256), `template` (repo, commit, kwargs and sha256), `libraries` (the `transformers`, `tokenizers` and `jinja2` versions), `client`, `n_ctx`, `stop`.
- **Backend:** `backend`, with its name and server ID. llama.cpp adds `build_info` and `llama_cpp_commit`; Transformers adds `device`, `dtype`, `attn_implementation`, `torch` and `transformers`.
- **Prompt:** `prompt_text`, `prompt_ids`, `n_prompt_client`, `bos` (whether BOS was prepended and why), `special_tokens_in_content` (where any control-token text was found).
- **Request and response:** `request` (what was sent, except the prompt IDs, which are in `prompt_ids`), `backend_settings`, `not_applied`, `generation_settings`, `response`, `timings`, `tokens_evaluated`, `tokens_predicted`.
- **Output:** `output_text`, `output_ids`, `stop_type` (`eos`, `word`, `limit`, `cancelled` or `none`), `cancelled_during` (`prompt` or `reply`, for a cancelled call), `stopping_word`, `turn_id`.
- **Checks:** `checks`, `flags`, `warnings`, and `error` if the call failed.

`output_ids` include the tokens of a stop string and the end-of-sequence token; `output_text` leaves out the stop string.

A replay compares the output token IDs and stop type, or the text if the logged IDs were incomplete. Its `differences` list every recorded fact that changed since the call:
- llama.cpp: the build and checkout, server flags, `/props`, `GGML_*` and `LLAMA_*` variables, and the GGUF's sha256.
- Transformers: the torch and Transformers versions, device, dtype, attention implementation, `PYTORCH_*` and `TORCH_*` variables, deterministic-mode state, and the weights' commit.
- Both: the machine, the client's commit, and the rendering libraries.

`unexplained_difference` is true when the output differs and nothing changed except, at most, the client's commit or the rendering libraries. Neither can change a replay, since it sends the logged IDs and request.

## Call flags

A call's `flags` list marks anything to look at. The terminal shows them after the reply as `! flags: …`.

| Flag | Meaning |
|---|---|
| `length` | The reply stopped at `max_tokens`. |
| `context_full` | The reply stopped because the context was full. |
| `cancelled` | The call was cancelled, with Ctrl-C or from another program. `cancelled_during` says whether any reply had arrived. |
| `empty_reply` | The reply is empty or only whitespace. |
| `not_applied` | The backend couldn't apply a setting as set, and it could change the reply. `/settings` gives the reason. |
| `special_tokens_in_content` | Some prompt text (a message, the system prompt, priming or header) contains text the tokenizer reads as a control token, such as `<\|im_end\|>`. |
| `incomplete_output_ids` | llama-server sent fewer output IDs than it generated (a token ended mid-character). Replay then compares text instead of IDs. |
| `text_differs_from_ids` | The returned text differs from the decoded output IDs; the decoded text is logged as `decoded_output_text`. |
| `check_failed:NAME` | A check after the call failed. See below. |

Checks after each call:
- `prompt_echo`: the server's decoding of the token IDs equals the prompt text.
- `settings_echo`: every value sent comes back unchanged in the server's `generation_settings`. Floats are compared at float32 precision, as the server stores them, so `0.6` sent and `0.6000000238418579` echoed is a match.
- `unaccounted_settings`: the server reports no setting Clarity didn't send or expect.
- `tokens_evaluated`: the server read as many prompt tokens as were sent.
- `no_cache_reuse`: no prompt tokens came from a cache.
- `server_props`: `/props` still shows the same process, model and context size.
- `generation_config` (Transformers): the effective generation config equals what was sent.

A batch run stops on any error, any `check_failed:*` flag, a cancelled call, or a server restart. Other flags are logged, and their warnings print once.
