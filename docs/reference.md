# Reference

Every command, key and record field a user sees. For task-based instructions, see [Guides](guides.md); for the reasons behind these rules, see [Design](design.md).

## Commands

| Command | What it does |
|---|---|
| `clarity chat --profile NAME` | Chat with `profiles/NAME.toml` or `local/profiles/NAME.toml`. NAME can also be a path to a `.toml` file. |
| `clarity chat --resume ID` | Continue a logged conversation, as it was logged, and say what differs today. ID is a conversation ID from `clarity chats`, or a log file. See `/resume`. |
| `clarity chat --json [--profile NAME \| --resume ID]` | Chat from a script or another program: commands in as JSON lines, events out as JSON lines. See [Scripting with --json](#scripting-with---json). |
| `clarity chats` | List your conversations in `logs/`, newest first: conversation ID (it starts with the local date and time), profile and title. |
| `clarity run --openings FILE --profiles A B … [--seeds N]` | Collect replies for a blind test into `runs/RUN_ID/`. FILE holds openings separated by lines that are exactly `---`. `--seeds N` (default 1) gives N replies per opening and profile, with seeds 333, 334 and so on. |
| `clarity replay LOG CALL_ID [--logged-flags]` | Re-send a logged call and compare the output. `--logged-flags` starts llama-server with the call's logged flags instead of `server.toml`'s, after the same checks `server.toml`'s get, so a log can't add, drop or change a flag in a way `server.toml` couldn't. |
| `clarity mark LOG CALL_ID [LABEL] [--remove]` | Label a logged call's reply while reviewing, in `logs/` or a run's `calls.jsonl` (default label `interesting`). Appends one record and starts no server. With `--remove`, appends an `unmark` for each mark it removes: those with that label, or the reply's only label. |
| `clarity note LOG CALL_ID TEXT` | Add a note to a logged call's reply while reviewing, replacing its note. Quote TEXT if it has spaces. |
| `clarity export ID [--out PATH]` | Write a conversation to `logs/exports/ID.jsonl`, or PATH, for a notebook. See [Export](#export). ID is a conversation ID from `clarity chats`, or a log file. A new export replaces the old file and says so. Starts no server. |
| `clarity fetch NAME` | Download (and, for the `convert` route, convert) a model from `models.toml`, and record it in `models.lock`. |
| `clarity verify [NAME …]` | Re-hash fetched models against `models.lock`; with no names, every fetched model. |

`--root PATH` sets the project folder, and goes before the command: `clarity --root PATH chat --profile NAME`. By default, the project folder is the nearest folder, from the current one upwards, that holds both `defaults.toml` and `server.toml`. `mark` and `note` don't need a project folder, only the log file.

Output sent to a file or a pipe is UTF-8 on every system. On Windows, Python would otherwise use the system's code page, which can't hold every title or reply.

Exit status:
- `0`: success, or `replay` found the output identical.
- `1`: a configuration or backend error (printed as `clarity: …`), a stopped `run`, a `replay` whose output differs, a `verify` that failed, or a `chat --json` session whose `--profile` or `--resume` failed or that an unexpected error ended.
- `2`: wrong command-line arguments.
- `130`: interrupted with Ctrl-C.

## Chat commands

Enter sends a message. Alt+Enter, or Esc then Enter, starts a new line. Pasted text keeps its newlines. Ctrl-C while the model works cancels the call. If part of the reply has arrived, it's kept and logged. If nothing has, no reply is added: the call is logged like a failed one, and your message comes back in the input to edit or send again.

| Command | What it does |
|---|---|
| `/help` | List the commands. |
| `/say TEXT` | Send a message, exactly as written after the one space that follows `/say`. Typing it without a command does the same; scripts use `/say`. Other commands ignore spaces around their arguments. |
| `/prompt` | The exact prompt text of the current reply's call (or of your last message, if its call failed), the BOS decision, and the token IDs. |
| `/prompt repr` | The same, with each line shown as a Python string, so spaces, tabs and newlines are visible. |
| `/probs` | A summary of the current reply's token probabilities, if `logprobs` was on: how many tokens, and the 10 least likely chosen tokens in reply order, each with its probability and the model's likeliest other tokens. With `--json`, the result also holds every token's record. Without them, it says why. |
| `/settings` | Every setting, its value and origin; the fixed values sent for the backend's other fields; the model's recommended values; what wasn't applied; the context size and stop strings; the end-of-reply tokens llama.cpp read from the model file. |
| `/set NAME [VALUE]` | Change a setting for this chat only, for example `/set temperature 0.7`, `/set seed random` or `/set samplers top_k temperature`. It's checked like a profile's value, its origin is `chat`, and it's logged. With no value, the setting goes back to the profile's. |
| `/regen [N]` | N more replies to your last message (default 1), one after another, each a new branch. The k-th regeneration of a message uses its first reply's seed plus k. A cancel stops the reply being written and the rest; a failure stops the rest and is reported with the replies already made. The current reply is then the last one made. |
| `/edit [TEXT]` | Replace your last message and get a new reply, as a new branch. With no text, the input is prefilled with your last message. The new message uses the chat's seed: the profile's, unless you changed it with `/set`. |
| `/undo` | Go back past the last exchange; your next message starts a new branch. Nothing is deleted. |
| `/history` | The conversation so far on the current branch, from its start to where you are, with each reply's marks and notes. |
| `/branches` | Where the conversation branches along the current path: each regenerated reply and edited message, numbered "reply 2 of 3", the current one starred, with its turn ID. Branches you went back from with `/undo` are listed too. |
| `/switch TURN_ID` | Go to another branch, at its newest reply. Nothing is sent. In a resumed conversation, it also checks that reply's prompt still renders to the same token IDs. |
| `/mark [LABEL]` | Label the current reply, for example `landed` or `dropped`. Without a label, the mark is `interesting`. |
| `/unmark [LABEL]` | Remove the current reply's mark. If it has marks with different labels, name the one to remove. The log keeps the mark and records its removal. |
| `/note TEXT` | Add a note to the current reply. A reply has one note: a new one replaces it, and the log keeps both. Notes from before v0.2.0 have no ID, so they all stay. |
| `/replay` | Re-send the current reply's call on the running server and compare the output. |
| `/export` | Write this conversation for a notebook to `logs/exports/ID.jsonl`, as `clarity export` does. It takes no path; `clarity export --out` writes elsewhere. |
| `/resume ID` | Continue a logged conversation, by its ID from `/chats` (not a path; `--resume` takes a log file): its profile, settings, `/set` values and context size as logged, starting where it left off. Clarity shows what differs today: a changed profile file, settings today's files would set differently, server, machine or library changes since its last call, and whether the current reply's prompt still renders to the same token IDs. |
| `/title TEXT` | Name this conversation in the chat list. Without a title, the list shows the first eight words of its first message; Clarity never asks a model for one. |
| `/chats` | Your conversations in `logs/`, newest first, with their titles. |
| `/new` | Start a new conversation (and log file) with the same profile, re-read from its file. |
| `/profile NAME` | Start a new conversation with another profile, by its name in `profiles/` or `local/profiles/` (not a path; `--profile` takes a file). The server restarts only if the model file, the context size or the backend changes. |
| `/cancel` | Stop the call in progress, like Ctrl-C: a reply (`/say`, `/regen`, `/edit`) or a `/replay`. It's the one command that runs during another, for scripts and the web interface. It says which command it stopped, or why it stopped nothing: nothing was running, or the running command (such as a model start) can't be stopped. |
| `/quit` | Stop the server and exit. `/exit`, `/q` and Ctrl-D also work. |
| `//text` | Send a message that starts with `/`. |

After each reply, a status line shows the stop reason, the reply length in tokens, the time, the seed (with its origin unless it came from the profile), the context used (prompt plus reply tokens, of the context size) and the call ID. Warnings and [call flags](#call-flags) appear above it, marked `!`. A warning appears when the prompt uses more than 80% of the context.

## Scripting with --json

`clarity chat --json` reads one command per line on stdin and writes one event per line on stdout, both as JSON. It runs every chat command in the table above except the terminal's own forms: `/exit`, `/q`, `//text`, `/prompt repr` and `/edit` with no text. With `--profile` or `--resume` it starts with that, and if that fails it sends the error and quits, as the terminal does. Without either, send a `profile` or `resume` command first.

A command is `{"command": "say", "args": "Hello", "id": 1}`. `command` is a chat command's name without the slash, `args` is the text that would follow it, and `id` (optional, any JSON value) is copied into the command's `result` or `error`.

| `event` | Sent when | Other fields |
|---|---|---|
| `text` | reply text streams in | `text` |
| `reply_start` | before each reply of `/regen N` | `k` (which reply, from 1), `n`, `seed` |
| `status` | a model is starting or ready | `status`, and `profile`, `log` or `conversation_id` |
| `record` | a record is appended to the log | `record`: exactly what was written. Record events are the log, in order. |
| `result` | a command finished | `command`, `id`, and `result`: what the command returns. It may repeat records already sent as `record` events. See [Command results](#command-results). |
| `error` | a command failed, or a line wasn't a command | `command` and `id` (for a command), `type` (`ConfigError`, `ContextError`, `BackendError`, `CancelledBeforeReply` or `InputError`; any other type is an unexpected error, after which the session quits) and `message` |

### Command results

| Command | `result` holds |
|---|---|
| `say`, `edit` | `call` (the call record) and `warnings` |
| `regen` | `replies`, a list of `call` and `warnings`, and `stopped` (why it stopped early) if a reply failed. A cancel ends it without `stopped`, whenever it comes. |
| `prompt` | `call`: the latest call record, which can be a failed one; null before any |
| `probs` | `call_id`, `logprobs` (the call's record) and `least_likely`: up to 10 entries with `index`, `id`, `text`, `logprob` and `top` (each `id`, `text`, `logprob`). With no probabilities, `least_likely` is empty and `unavailable` says why. |
| `settings` | `profile`, `settings` (each with its value and origin), `backend`, `backend_settings`, `eog_tokens`, `recommended`, `not_applied`, `ctx_size` and `stop` |
| `set` | `record` (the `setting` record) and `setting` (its value and origin) |
| `undo`, `switch` | `head` and `head_text`, null at the start; `switch` adds `render` |
| `history` | `head` and `turns` |
| `branches` | `head` and `forks` |
| `mark`, `unmark`, `note` | `records`: the records appended |
| `title` | `record` |
| `replay` | `replay`: the replay record |
| `export` | `path`, `replaced`, `calls`, `failed_calls_left_out` and `calls_without_a_turn_left_out` |
| `chats` | `chats` |
| `profile`, `new` | `conversation_id`, `profile`, `model`, `mode` and `warnings` |
| `resume` | the same, plus `resume` (the record) and `head_text` |
| `cancel` | `cancelled`, and `command` or `reason` |
| `help` | `commands`, each with `usage` and `help` |
| `quit` | `quit`: true |

Commands run one at a time, in order. `cancel` runs as soon as it's read, so it can stop a reply that's being written; its result names the command it stopped (`"cancelled": true, "command": "say"`) or gives a `reason` it stopped nothing. Send it after the command has started (after its first event): a `cancel` read while the command is still queued finds nothing running. End of input quits, like `quit`. Input is UTF-8, whatever the console's encoding; a line that isn't is an `InputError`. Output is ASCII: other characters are written as `\u` escapes.

## Profile keys

A profile is a TOML file in `profiles/` (shipped with Clarity) or `local/profiles/` (yours; git ignores it). Its name is the file name without `.toml`, and a name in both folders is an error. Unknown keys are errors, and the allowed keys depend on `mode`. A `{ file = "PATH" }` is read byte for byte, relative to the profile, and must be inside the project folder, after following links.

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
| `steering` | no | Transformers backend only: a table naming a [steering vector](#steering-vectors). `file` is its path relative to the project folder, and must be inside it. `sha256` is its hash; without it, Clarity refuses the profile and gives the hash to add. `layers`, optional, lists the blocks to steer; the default is every block in the file. A profile with `[steering]` must set `steering_scale` in `settings`. |

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

Settings use the same names on every backend. Each resolved value records its origin: `profile`, `model` (from `use_recommended`), `default` (from `defaults.toml`), `server.toml` (`ctx_size` only: a profile without one), `chat` (set with `/set` for one conversation), `log` (a resumed conversation's context size, from its log), `regen` (a `/regen` seed) or `run` (a batch-run seed). Fixed values Clarity sends for a backend's other fields have origin `backend`.

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
| `logprobs` | integer from 0 to 20. 0 is off. N records, for each token of the reply, its log-probability and the N likeliest tokens with theirs, in the call's `logprobs` field. A log-probability is the natural log of the probability the model gave the token, so 0 is certain and more negative is less likely. The probabilities come from the model's raw scores (logits) before any sampler, so they don't show what temperature, `top_k` or the penalties did. Transformers backend only for now; a llama.cpp profile with it above 0 is refused. While a reply is written, the scores take memory: about 0.4 MB per token for a model with a 100,000-token vocabulary. | 0 | |
| `steering_scale` | any number; 0 is off. The profile's steering vector is multiplied by it before it's added, and a negative number steers the other way. A profile with `[steering]` must set it; one without must leave it at 0. Transformers backend only. | 0.0 | |

A sampler not listed in `samplers` doesn't run. A setting whose sampler isn't listed must keep its neutral value; anything else is an error when the profile loads.

The Transformers backend applies its samplers in a fixed order: penalties, temperature, `top_k`, `top_p`, `min_p`. It has no `presence_penalty`, `frequency_penalty` or `repeat_last_n`. These are recorded as not applied, and a call is flagged `not_applied` when the difference can change the reply.

## models.toml

Each table is one model, and its name is the model's name in profiles and `clarity fetch`. Quote a name with a dot in it, such as `["bge-small-en-v1.5"]`: in TOML, a bare dot starts a nested table. Every revision is a full 40-character commit hash.

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

Only these flags are allowed: the ones the shipped `server.toml` sets (`ctx-size`, `n-gpu-layers`, `flash-attn`, `cache-type-k`, `cache-type-v`, `batch-size`, `ubatch-size`, `fit`, `parallel`, `no-context-shift`, `swa-full`, `ctx-checkpoints`, `cache-ram`, `no-cache-idle-slots`, `cache-reuse`, `threads`, `threads-batch`, `log-verbosity`, `no-jinja`, `chat-template`), and `kv-unified` or `no-kv-unified` (exactly one). Any other flag is refused, including another name for an allowed one, so a new flag is added on purpose, with a test. Clarity passes `--model`, `--host`, `--port`, `--no-slots` and `--no-webui` itself. `auto` and `-1` are refused as values. `kv-unified`, `no-kv-unified` and `no-jinja` must be `true` when set, since `false` would leave the flag out. `threads` and `threads-batch` must be positive, since llama-server reads 0 or less as every core.

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
| `logs/exports/CONVERSATION_ID.jsonl` | A conversation exported for a notebook (`clarity export`, `/export`). `clarity chats` doesn't list it. |
| `local/clarity.lock` | Locked while `chat`, `run` or `replay` runs, so only one of them runs in a project at a time. Empty. |
| `local/clarity.json` | Who holds the lock: process ID, command and start time, and the running llama-server's port, process ID and key. Readable only by you on Linux and macOS (mode 0600); on Windows it gets your folder's permissions. Removed on a normal exit; one left behind means Clarity didn't exit normally, and the next start uses it to stop that server. |

`logs/`, `runs/`, `models/` and `local/` are gitignored. Logs are append-only: a line is never rewritten. Clarity creates the files in `logs/` and `runs/` readable only by you on Linux and macOS (mode 0600); on Windows they get your folder's permissions. Files from before v0.2.1 keep the mode they had.

## Log records

Each line is one JSON record with `schema_version` (currently 1), `type` and `time` (UTC). Every record except `run` and `torn_line` also has `conversation_id`. Readers should ignore keys they don't know.

| `type` | Written | Main fields |
|---|---|---|
| `conversation` | at the start of each conversation | `profile` (name, path, full TOML text and every file it read), `settings`, `model`, `client` (commit and dirty flag), `machine`, `warnings`, `ctx_size` (value and origin: `profile` or `server.toml`) |
| `server_start` | once per server per log file, before its first call | `server_id`, `argv`, `flags`, `api_key` (always "random per start; value not logged"), `env_removed`, `env_llama_ggml_kept`, `llama_cpp` (commit, dirty), `build_info`, `placement`, `model_file` (`eog_tokens`: the tokens that end a reply, read from llama-server's startup log; null if the log has no such line), `memory` (per GPU: `limit_mib`, which is Metal's recommended working-set limit on a Mac and, on other GPUs, the memory llama.cpp reported free when the model loaded; and the model, context cache and compute buffers on it), `graph_splits` (how many pieces llama.cpp cut the compute graph into, for the prompt and the reply), `props`, `startup_output`, `warnings`, `machine`. Transformers: `device`, `dtype`, `attn_implementation`, `torch`, `transformers`, `eos_token_id`, `eos_token_id_from` (`generation_config.json`, or `tokenizer (generation_config.json has none)`) and similar. |
| `turn` | for each message and reply | `turn_id`, `parent_id`, `role`, `origin` (`user`, `model` or `edit`), `text`, `call_id`, and `regen_of` or `edit_of` for branches |
| `settings` | for each `/set` | `name`, `value` (null: back to the profile's) and `chat_settings`: every value set in this chat so far. |
| `resume` | for each `/resume` or `--resume` | `head` (where it continues), `head_guessed` (a v0.1.0 log that no later Clarity has moved the head in since resuming), `warnings` (as in the `conversation` record), `unanswered` (your last message, if its call failed or was cut off), `ctx_size`, `profile_file` (`same`, `changed` or `missing` today, or `outside the project folder`, which isn't read), `model_entry` (how the model's `models.toml` entry differs from the logged one), `settings_differ_today` (each with its `logged` and `today` value; `logged` is null for a setting the log doesn't have, which then comes from the logged profile resolved today), `chat_settings` (restored `/set` values), `compared_with_call`, `differences` and `not_recorded` (as in `replay`), and `render`: the current reply's call ID and whether its prompt renders to the same token IDs today (`n_ids` and `first_different_token` if not), or null if there's no reply yet. |
| `title` | for each `/title` | `title`. The latest one is the conversation's title. |
| `call` | for every call, including failed and cancelled ones | see below |
| `head` | when the current turn moves other than to a new reply: `/undo`, `/switch`, or a call that failed or was cancelled before any reply | `turn_id` (null: the start) and `reason`: `undo`, `switch` or `failed`. A `failed` move after a message (not a `/regen` or `/edit`) names the failed call in `call_id`, so a resumed conversation's `/prompt` shows it, as the chat did. A `switch` in a resumed conversation adds `render`, as in the `resume` record. The latest `turn` or `head` record says where the conversation continues; if that is a message with no reply (Clarity stopped during its call), it continues before that message. |
| `mark`, `unmark`, `note` | for `/mark`, `/unmark`, `/note`, `clarity mark` and `clarity note` | `call_id`, `turn_id`, `label` or `text`, and `origin`: `live` (in a chat) or `review` (afterwards). A `mark` has a `mark_id`; an `unmark` names the `mark_id` it removes (null for a mark from before v0.2.0, which removes that reply's ID-less marks with its `label`). A `note` has a `note_id` and `replaces`: the `note_id` of the note it replaces, or null. Notes from before v0.2.0 have no ID, so they all still show. |
| `replay` | for every replay | `replay_id`, `call_id`, `n_ctx_from`, `identical`, `compared`, `differences`, `not_recorded` (facts an older log didn't record, so not compared, such as `end_tokens` for v0.1.0 llama.cpp logs; and machine values known on one side only, such as `machine.macos.chip` when a sandbox hides the chip's name, which are unknown rather than different), `unexplained_difference`, `server_id`, `flags_from` (`server.toml`, or the log with `--logged-flags`), `logged_server_found`, `logged_output_text` (when the output differs), and the new output with its `flags` and `checks`. Its `flags` hold only the output checks, not the call's `not_applied` or `special_tokens_in_content`. |
| `torn_line` | before the first record written after a line that was cut off (a crash or a full disk mid-write) | `line` (its line number) and `bytes` (how much of it was written). The cut line stays as it was. A last line with no line end isn't read: it's still being written, or was cut off and the next record will say so. A line a `torn_line` names, with the same byte count, is never read as a record, even if it parses, and neither are malformed lines just before it; any other malformed line is an error. |
| `run` | first line of a batch run's `calls.jsonl` | `run_id`, `openings_file`, `openings_sha256`, `profiles`, `seeds`, the split and trim rules |

A `call` record holds:
- **Where it sits:** `call_id`, `parent_id` (your message's turn), `regen_of` (for a `/regen`), `run` (in a batch run: the run ID and opening).
- **Inputs:** `profile`, `settings` (each with its origin), `seed_drawn`, `recommended` (with `use_recommended`: the model's values the settings came from; a resumed conversation keeps the logged ones), `model` (the `models.toml` entry, plus the GGUF's sha256), `template` (repo, commit, kwargs and sha256), `libraries` (the `transformers`, `tokenizers` and `jinja2` versions), `client`, `n_ctx`, `stop`.
- **Backend:** `backend`, with its name and server ID. llama.cpp adds `build_info` and `llama_cpp_commit`; Transformers adds `device`, `dtype`, `attn_implementation`, `torch` and `transformers`.
- **Prompt:** `prompt_text`, `prompt_ids`, `n_prompt_client`, `bos` (whether BOS was prepended and why), `special_tokens_in_content` (where any control-token text was found).
- **Message spans:** `message_spans`, a list that covers every prompt token in order, or `{"unavailable": REASON}`. Each entry has `chars` (in `prompt_text`) and `tokens` (in `prompt_ids`), both half-open `[start, end)`, and a `kind`: `message` (with `role`, and `turn_id` for a conversation turn or `source`, `system` or `priming N`, for the profile's own; text mode's priming is one block with `source: "priming"` and no `role`, since it holds lines from both speakers), `template` (text the chat template wrote: role headers, end tokens, default system messages, dates, empty thinking blocks), `profile` (text mode's header, names and line ends) or `bos` (a prepended BOS, with no characters). `trimmed: true` means the template dropped the message's edge whitespace, so the span holds the stripped text. `edge_token: "end"` means the span's last token runs into the next span. An empty message has a zero-length span. Calls logged before v0.2.1 have none.
- **Request and response:** `request` (what was sent, except the prompt IDs, which are in `prompt_ids`), `backend_settings`, `not_applied`, `generation_settings`, `response`, `timings`, `tokens_evaluated`, `tokens_predicted`.
- **Output:** `output_text`, `output_ids`, `stop_type` (`eos`, `word`, `limit`, `cancelled` or `none`), `cancelled_during` (`prompt` or `reply`, for a cancelled call), `stopping_word`, `turn_id`.
- **Probabilities:** `logprobs`, with the `logprobs` setting on (Transformers only): `view` (`raw logits, before any sampler`), `n`, and `tokens`, one per output token: `id`, `logprob` (natural log) and `top`, the `n` likeliest tokens as `[id, logprob]` pairs, likeliest first. A cancelled call has `{"unavailable": "cancelled"}` instead: Transformers returns no scores for a generation stopped part way. A replay of a call with probabilities records them too.
- **Steering:** on the Transformers backend, `request.steering`: `file`, `sha256`, `scale` and `layers` for a steered call, or null. A call logged before v0.2.2 has no `steering` key, which means it wasn't steered. A steered call that finished adds `checks.steering`: `ok`, and `forward_passes`, how many times each steered block ran (block number to count). It's ok when each ran once per generated token. A cancelled call has no steering check, since a cancel can land mid-pass.
- **Checks:** `checks`, `flags`, `warnings`, and `error` if the call failed.

`output_ids` include the tokens of a stop string and the end-of-sequence token; `output_text` leaves out the stop string.

A replay compares the output token IDs and stop type, or the text if the logged IDs were incomplete. Its `differences` list every recorded fact that changed since the call:
- llama.cpp: the build and checkout, server flags, `/props`, `GGML_*` and `LLAMA_*` variables, and the GGUF's sha256.
- Transformers: the torch and Transformers versions, device, dtype, attention implementation, `PYTORCH_*` and `TORCH_*` variables, deterministic-mode state, and the weights' commit.
- Both: the machine, the client's commit, and the rendering libraries.

A replay of a steered call applies the logged vector at the logged scale and blocks. It checks the logged steering like a profile's: the file must be inside the project folder, have the logged sha256 and fit the model, or the replay is refused; it never replays a steered call unsteered. Clarity before v0.2.2 ignores `request.steering`, so it replays a steered call unsteered.

`unexplained_difference` is true when the output differs and nothing changed except, at most, the client's commit or the rendering libraries. Neither can change a replay, since it sends the logged IDs and request.

## Export

`clarity export` and `/export` write a conversation as JSON Lines, so a notebook reads it with a loop or `pandas.read_json(path, lines=True)`. The file is readable only by you on Linux and macOS, like the log. Replay records aren't exported: they re-run calls already there. A batch run's `calls.jsonl` can't be exported.

The first line is the header, `"type": "export"`:
- `format_version` (1), `clarity` (`version`, `commit`, `dirty`) and `conversation_id`.
- `calls` (how many follow), `failed_calls_left_out` (calls that produced no reply) and `calls_without_a_turn_left_out` (calls whose reply's record is missing from the log, as after a crash while it was written; resume carries on without them too).
- `conditions`: one entry per distinct set of conditions the calls were computed under, since a resumed conversation can span several server starts, machines or builds. Each has `conditions` and `call_ids`, the calls it covers. `conditions` holds `backend`, `weights` (the repo and revision, plus the GGUF's sha256 for llama.cpp), and every fact replay compares except Clarity's own commit and the tokenizer libraries: for llama.cpp, `build_info`, `llama_cpp`, `flags`, `graph_splits`, `props`, `env`, `end_tokens` and `machine`; for Transformers, `torch`, `transformers`, `device`, `dtype`, `attn_implementation`, `env`, `deterministic_algorithms`, `end_tokens` and `machine`. A null is a fact the log didn't record.
- `note`: "Recomputing these calls under other conditions can give different numbers, sometimes by a tiny amount and sometimes enough to change the likeliest token. Run `clarity_chat.recompute.check` with your loaded model first; it lists what differs."

Then one line per call that produced a reply, `"type": "call"`, in log order:
- **Where it sits:** `call_id`, `turn_id` (the reply), `parent_id` (your message), `regen_of`, `edit_of` (your message replaced an earlier one), and `path`, the turn IDs from the start of the conversation to this reply.
- **What went in and came out:** `prompt_ids` as sent (with any BOS), `output_ids` as generated, `prompt_text`, `output_text`, `stop_type`, and `cancelled` (true for a reply cut short; its output is incomplete).
- **Settings:** `settings` (each with its value and origin), `n_ctx` and `flags`, as logged.
- **Message spans:** `message_spans`, as in the call record. A call logged before v0.2.1 has none, so the export works them out with the profile and tokenizers as logged, and checks the result rebuilds the logged prompt exactly. If it can't, the field is `{"unavailable": REASON}`, never a guess. A template that writes today's date can't be checked on a later day.
- **Marks and notes:** `marks` (`label`, `origin`, `time`) and `notes` (`text`, `origin`, `time`), as they are now, after any `unmark` or replaced note.
- **Model:** `model`, its pins: `name`, `repo`, `revision`, `route`, `gguf_sha256` and `gguf_path`.
- **Probabilities:** `logprobs`, if the call logged them.
- **Facts:** `facts`, what replay compares, from the call's own server start: the end tokens, and the build, flags and machine or the library versions, device and precision.

## Steering vectors

A steering vector is a file made by another tool, such as repeng; Clarity applies one and never makes one. The file is llama.cpp's control-vector GGUF: one 1-D float32 tensor named `direction.N` for each block N it steers, all the same length (the model's hidden size), and optionally a `controlvector.model_hint` key naming the model type it was made for. Steering adds the scale times `direction.N` to the output of block N, counting from 0, at every position: the prompt's and the reply's. Block 0 can't be steered: llama.cpp's control-vector loader refuses it, and Clarity keeps a vector usable on both.

Clarity reads the file with its own reader, which accepts only this format: GGUF version 3, little-endian, at most 16 MB, 1 to 1,024 tensors, every value finite. Anything else is refused with what was wrong.

Steering runs on model types `llama` and `olmo3`, the two whose blocks Clarity has checked. Other types, including Qwen3 and Apertus, are refused until they are checked. `llama` admits every Llama-architecture model, such as Hermes 4, though only SmolLM2 has a live test.

A profile names its vector in a `[steering]` table and its scale in `settings`:

```toml
[steering]
file = "local/vectors/calm.gguf"   # relative to the project folder
sha256 = "…"
layers = [10, 11, 12]              # optional; default: every block in the file

[settings]
steering_scale = 4.0               # required with [steering]; negative steers the other way
```

When a conversation starts, resumes or switches profile, and before a batch run, Clarity loads the file, checks its sha256, and checks it against the model's `config.json`: the model type, the width, and that each block exists and is in the file. A vector that changed since the profile pinned it is refused. `/set steering_scale` changes the scale for one conversation without reloading the model.

In a notebook, `clarity_chat.steering` applies a vector exactly as Clarity does:
- `load(path, sha256)` reads the file, checks its sha256, and returns a `Vector` with `sha256`, `directions` (block number to a float32 array) and `model_hint`.
- `check(vector, config, layers=None)` checks the vector against a model's `config.json` (as a dict) and returns the blocks to steer: `layers`, or every block in the file.
- `with applied(model, vector, scale, layers=None) as runs:` steers a loaded Transformers model inside the block, and `runs` counts how many times each steered block ran. It computes `scale × direction.N` in float32 and casts it once to the model's dtype, as repeng and llama.cpp compute it, then adds it in that dtype. Scale 0 adds nothing. The hooks come off when the block ends, however it ends, and a model already inside `applied` is refused.

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
- `server_running`: the llama-server process Clarity started is still running (`exit_code` if not). Its model and context size are checked through `/props` once, after start. Logs before v0.2.1 have `server_props` here, from a `/props` request after each call.
- `generation_config` (Transformers): the effective generation config equals what was sent.

A batch run stops on any error, any `check_failed:*` flag, a cancelled call, or a server restart. Other flags are logged, and each flagged call is named on screen with its flags; other warnings print once.
