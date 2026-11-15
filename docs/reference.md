# Reference

Every command, key and record field a user sees. For task-based instructions, see [Guides](guides.md); for the reasons behind these rules, see [Design](design.md).

## Commands

A conversation's ID, as `clarity chats` lists it, is its log file's name: `logs/ID.jsonl`. It's what `--resume`, `/resume`, `clarity export` and the `status` event's `log` use, and `/state` gives it as `log_id`. The ID inside the log (`conversation_id` in its records) is the same unless the file was copied or renamed.

| Command | What it does |
|---|---|
| `clarity chat --profile NAME` | Chat with `profiles/NAME.toml` or `local/profiles/NAME.toml`. NAME can also be a path to a `.toml` file. |
| `clarity chat --resume ID` | Continue a logged conversation, as it was logged, and say what differs today. `--resume` takes an ID from `clarity chats` or a log file's path. See `/resume`. |
| `clarity chat --json [--profile NAME \| --resume ID]` | Chat from a script or another program: commands in as JSON lines, events out as JSON lines. See [Scripting with --json](#scripting-with---json). |
| `clarity ui` | Start the web interface: a page on this computer, at `http://127.0.0.1:8765`, opened in your browser through `local/clarity-open.html`, a private file that's deleted once the page it opens connects. The link it prints holds a secret made for this start, and works until Clarity stops; Enter in the terminal opens it again, and Ctrl-C stops Clarity. Another Clarity running in the project, or another program on the port, stops it with a sentence. Like `chat`, it first stops a llama-server that a killed Clarity left running. |
| `clarity chats` | List your conversations in `logs/`, newest first: conversation ID (it starts with the local date and time), profile (with its description, if it has one) and title. |
| `clarity run --openings FILE --profiles A B … [--seeds N]` | Collect replies for a blind test into `runs/RUN_ID/`. FILE holds openings separated by lines that are exactly `---`. `--seeds N` (default 1) gives N replies per opening and profile, with seeds 42, 43 and so on. |
| `clarity replay LOG CALL_ID [--logged-flags]` | Re-send a logged call and compare the output. `--logged-flags` starts llama-server with the call's logged flags instead of `server.toml`'s, after the same checks `server.toml`'s get, so a log can't add, drop or change a flag in a way `server.toml` couldn't. |
| `clarity mark LOG CALL_ID [LABEL] [--remove]` | Label a logged call's reply while reviewing, in `logs/` or a run's `calls.jsonl` (default label `interesting`). Appends one record and starts no server. With `--remove`, appends an `unmark` for each mark it removes: those with that label, or the reply's only label. |
| `clarity note LOG CALL_ID TEXT` | Add a note to a logged call's reply while reviewing, replacing its note. Quote TEXT if it has spaces. |
| `clarity export ID [--out PATH]` | Write a conversation to `logs/exports/ID.jsonl`, or PATH, for a notebook. See [Export](#export). ID is an ID from `clarity chats`, or a log file's path, whose name then names the export. A new export replaces the old file and says so. Starts no server. |
| `clarity fetch NAME` | Download (and, for the `convert` route, convert) a model from `models.toml`, and record it in `models.lock`. A gated repo needs your own Hugging Face login first ([Fetch a gated model](guides.md#fetch-a-gated-model)); without access, the fetch stops and says what to do. |
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
| `/prompt [CALL_ID]` | The exact prompt text of the current reply's call (or of your last message, if its call failed), the BOS decision, and the token IDs. With a call ID, that call's, failed calls included. |
| `/prompt [CALL_ID] repr` | The same, with each line shown as a Python string, so spaces, tabs and newlines are visible. |
| `/probs [CALL_ID]` | A summary of the current reply's token probabilities, or the named call's, if `logprobs` was on: how many tokens, and the 10 least likely chosen tokens in reply order, each with its probability and the model's likeliest other tokens. With `--json`, the result also holds every token with its text and place in the reply. Without them, it says why. |
| `/settings` | Every setting, its value and origin; the fixed values sent for the backend's other fields; the model's recommended values; what wasn't applied; the context size and stop strings; the end-of-reply tokens llama.cpp read from the model file; and a steered profile's vector: its file, the first 12 characters of its sha256, and its blocks, marked off at `steering_scale` 0. |
| `/set NAME [VALUE]` | Change a setting for this chat only, for example `/set temperature 0.7`, `/set seed random` or `/set samplers top_k temperature`. It's checked like a profile's value, its origin is `chat`, and it's logged. With no value, the setting goes back to the profile's. |
| `/regen [N]` | N more replies to your last message (default 1), one after another, each a new branch. Each uses the highest seed used for the message so far, plus 1, so the k-th regeneration uses the first reply's seed plus k unless `/regen same` reused seeds. A cancel stops the reply being written and the rest; a failure stops the rest and is reported with the replies already made. The current reply is then the last one made. |
| `/regen same` | Your last message again, with the seeds its replies already used, under the current settings: one reply for each seed that has no reply under them yet, in the order the seeds were first used. Use it after `/set` to see what the change does with the randomness held fixed, then `/compare`. If every seed already has a reply under the current settings, it says so and makes nothing. |
| `/edit [TEXT]` | Replace your last message and get a new reply, as a new branch. With no text, the input is prefilled with your last message. The new message uses the chat's seed: the profile's, unless you changed it with `/set`. |
| `/undo` | Go back past the last exchange; your next message starts a new branch. Nothing is deleted. |
| `/history` | The conversation so far on the current branch, from its start to where you are, with each reply's marks and notes. |
| `/branches` | Where the conversation branches along the current path: each regenerated reply and edited message, numbered "reply 2 of 3", the current one starred, with its turn ID. Branches you went back from with `/undo` are listed too. |
| `/compare [CALL_ID]` | The replies to your last message, or to the message the named call answered, on every branch, oldest first. Replies whose settings (all but the seed) agree form a group; it shows the settings that differ between groups, what's the same in all, and, for two replies with the same seed in different groups, the token and character where they part. For your last message it also says which seeds `/regen same` would make again. It reads only: the current reply stays. A setting a call from an older Clarity didn't record counts as off (`logprobs` 0, `steering_scale` 0), which it was. |
| `/branches all` | Every branch end in the conversation, newest first, wherever it is: where it left the rest (the reply or message it started from, "reply 2 of 4", with its seed), whether it went on past that point, its length in turns, how many of its replies are marked, and its seed. Then every marked reply, newest mark first, with its note. |
| `/switch TURN_ID` | Go to another branch, at its newest reply. Nothing is sent. In a resumed conversation, it also checks that reply's prompt still renders to the same token IDs. |
| `/mark [CALL_ID] [LABEL]` | Label the current reply, or the reply of the call named, for example `landed` or `dropped`. Without a label, the mark is `interesting`. |
| `/unmark [CALL_ID] [LABEL]` | Remove a reply's mark. If it has marks with different labels, name the one to remove. The log keeps the mark and records its removal. |
| `/note [CALL_ID] TEXT` | Add a note to a reply. A reply has one note: a new one replaces it, and the log keeps both. Notes from before v0.2.0 have no ID, so they all stay. |
| `/replay [CALL_ID]` | Re-send the current reply's call, or the call named, on the running server and compare the output. |
| `/export` | Write this conversation for a notebook to `logs/exports/ID.jsonl`, as `clarity export` does. It takes no path; `clarity export --out` writes elsewhere. |
| `/resume ID` | Continue a logged conversation, by its ID from `/chats` (not a path; `--resume` takes an ID or a log file's path): its profile, settings, `/set` values and context size as logged, starting where it left off. Clarity shows what differs today: a changed profile file, settings today's files would set differently, server, machine or library changes since its last call, and whether the current reply's prompt still renders to the same token IDs. Resuming the conversation that's open starts its model again if its server stopped; it writes a `resume` record like any resume, so `/new` then reads the profile by its logged name. If that start fails, the conversation stays open with its model stopped (`status: stopped`), so it can be tried again; so does a `/replay` or `/new` that would start the same model. |
| `/title TEXT` | Name this conversation in the chat list. Without a title, the list shows the first eight words of its first message; Clarity never asks a model for one. |
| `/chats` | Your conversations in `logs/`, newest first, with their titles. |
| `/profiles` | Every profile in `profiles/` and then `local/profiles/`: its model, backend and description, whether it's steered, and whether its model is fetched. A profile that doesn't load is listed with the reason. A profile is fetched when its model is in `models.lock` from its `models.toml` entry and its weights are on disk, unchanged in size and time (the GGUF for llama.cpp, the official weights for Transformers); `clarity verify` says more about one that isn't. It reads the profile's file, its model entry and the weights' file facts only; a refusal that needs the model's tokenizer or the server comes when a conversation starts. Runs before any conversation. |
| `/new` | Start a new conversation (and log file) with the same profile, re-read from its file. |
| `/profile NAME` | Start a new conversation with another profile, by its name in `profiles/` or `local/profiles/` (not a path; `--profile` takes a file). The server restarts only if the model file, the context size or the backend changes. |
| `/state` | Where things are: the conversation, its title and profile, the turns on the current branch with their marks and notes, the command running and the reply it has written so far, your last message if its call failed, the start warnings, what differs today after a resume (until your next message, answered or not (a Regenerate or Edit that fails keeps it)), whether the model is starting, ready or stopped (or none is loaded), and Clarity's version and commit and llama.cpp's. Like `/cancel`, it runs during another command, so a page that reloads mid-reply can draw itself. |
| `/cancel` | Stop the call in progress, like Ctrl-C: a reply (`/say`, `/regen`, `/edit`) or a `/replay`. Like `/state`, it runs during another command, for scripts and the web interface. It says which command it stopped, or why it stopped nothing: nothing was running, or the running command (such as a model start) can't be stopped. |
| `/quit` | Stop the server and exit. `/exit`, `/q` and Ctrl-D also work. |
| `//text` | Send a message that starts with `/`. |

**Naming a call.** `/prompt`, `/probs`, `/mark`, `/unmark`, `/note`, `/replay` and `/compare` act on the current reply, or on another call when their first word is its call ID: `call-` and 12 lowercase hex digits, as the status line and `/prompt` show it. The ID must be one of this conversation's calls, failed ones included, or the command is refused; it's never read as text. A first word that only looks like an ID, such as `call-me`, is the command's own text, so `/mark call-me` labels the current reply "call-me".

After each reply, a status line shows the stop reason, the reply length in tokens, the time ("time unknown" when the record has none, as for a llama.cpp reply cancelled mid-stream), the seed (with its origin unless it came from the profile), the context used (prompt plus reply tokens, of the context size) and the call ID, then, for a steered reply, `steered ×S` with the scale it was steered at. Warnings and [call flags](#call-flags) appear above it, marked `!`. A warning appears when the prompt uses more than 80% of the context.

## Scripting with --json

`clarity chat --json` reads one command per line on stdin and writes one event per line on stdout, both as JSON. It runs every chat command in the table above except the terminal's own forms: `/exit`, `/q`, `//text`, `/prompt repr` and `/edit` with no text. With `--profile` or `--resume` it starts with that, and if that fails it sends the error and quits, as the terminal does. Without either, send a `profile` or `resume` command first.

A command is `{"command": "say", "args": "Hello", "id": 1}`. `command` is a chat command's name without the slash, `args` is the text that would follow it, and `id` (optional, any JSON value) is copied into the command's `result` or `error`.

| `event` | Sent when | Other fields |
|---|---|---|
| `text` | reply text streams in | `text`, and `at` and `end`: where the piece sits in the reply being written, in characters (Unicode code points), so a reader that joins mid-reply keeps the pieces at or past `/state`'s `reply_end` |
| `reply_start` | before each reply of `/regen N` | `k` (which reply, from 1), `n`, `seed` |
| `status` | a model is starting, ready, or has stopped (its server exited; the call that found it fails, and resuming the conversation starts it again) | `status`, and `profile` (its name), `log` (as `/resume` was given it: a log file's name, or for `--resume` the ID or path you typed) or `conversation_id` |
| `record` | a record is appended to the log | `record`: exactly what was written. Record events are the log, in order. |
| `result` | a command finished | `command`, `id`, and `result`: what the command returns. It may repeat records already sent as `record` events. See [Command results](#command-results). |
| `error` | a command failed, or a line wasn't a command | `command` and `id` (for a command), `type` (`ConfigError`, `ContextError`, `BackendError`, `CancelledBeforeReply` or `InputError`; any other type is an unexpected error, after which the session quits) and `message` |

### Command results

| Command | `result` holds |
|---|---|
| `say`, `edit` | `call` (the call record) and `warnings` |
| `regen` | `replies`, a list of `call` and `warnings`, and `stopped` (why it stopped early) if a reply failed. A cancel ends it without `stopped`, whenever it comes. `/regen same` adds `seeds`, the seeds it set out to make. `server_stopped` is true when the reply that failed found the model's server stopped. |
| `prompt` | `call`: the latest call record, or the one named, which can be a failed one; null before any. It's the record as logged, the same before and after a resume. `replays`: the call's `replay` records, oldest first, read from the log. |
| `probs` | `call_id`, `logprobs` (the call's record), `tokens` and `least_likely`. Each of `tokens` has `index`, `id`, `text`, `start` and `end` (its characters in the reply's text), `logprob` and `top` (each `id`, `text`, `logprob`). `text` is the token's own text, special tokens included; a character split across tokens belongs to the token that completes it. `start` and `end` index the reply as written, without special tokens, so a special token or a stop string has `start` equal to `end`; they're null for every token if the tokenizer's decoder doesn't build the text up token by token, or its text doesn't begin with the reply's. `least_likely` holds up to 10 of the same entries, in reply order. With no probabilities, both lists are empty, `reason` is a code (`no_reply`, `failed`, `llama_cpp`, `off`, `cancelled` or `other`) and `unavailable` says why. |
| `settings` | `profile`, `settings` (each with its value and origin; a row set with `/set` also has `profile`, the profile's own value and origin), `model_name` (the last part of the model's tokenizer repo), `backend`, `backend_settings`, `eog_tokens`, `recommended`, `not_applied` (what the backend can't apply at the current values, as the next call will report; empty on llama.cpp), `ctx_size`, `stop`, and `steering` (`file`, `sha256` and `layers` of the profile's vector, or null) |
| `set` | `record` (the `setting` record) and `setting` (its value and origin) |
| `undo`, `switch` | `head` and `head_text`, null at the start; `switch` adds `render` |
| `history` | `head` and `turns` |
| `compare` | `message` (`turn_id`, `text`); `replies`, oldest first, each with `k`, `call_id`, `turn_id`, `text`, `seed`, `steering_scale`, `stop_type`, `group`, `marks`, `notes`, `current` (on the path to head) and `later_turns` (turns after it on its branch; 0 means it wasn't continued); `groups`, each with `settings`, the values that differ between groups; `same`, the settings equal in every reply (the seed left out); `same_prompt`, true when every call sent the same prompt token IDs; `pairs`, each two replies with the same `seed` in different groups (`a` the earlier, `b`), with `first_different_token` and `first_different_char` (in `b`'s text), both null if the replies are identical, and `prefix`, true when one reply's text begins the other's (it stopped where the other went on); and `again`, for your last message only: the `seeds` `/regen same` would make and `now`, the current values that differ from some reply's, or null when there's nothing to make. |
| `branches` | `head` and `forks`. With `all`: `head`, `ends` and `marked`. Each end has `turn_id` and `call_id` (its newest reply's), `text`, `turns` (on its path, priming left out), `marked` (how many of those replies are marked), `seed`, `steering_scale`, `from`, `continued` (false when the end is the reply its fork made) and `current` (the end the head is on). `from` is the deepest fork on the path: `kind` (`reply`, a regenerated reply; `edit`, a message beside another, edited or sent after `/undo`), `k` and `n` ("2 of 4") and the seed of the reply it made; null on a path with no fork. Each of `marked` has `turn_id`, `call_id`, `text`, `marks` (each `mark_id` and `label`), `notes` (each `note_id` and `text`, as `history` gives them), `from` and `current` (on the path to head). |
| `mark`, `unmark`, `note` | `records`: the records appended |
| `title` | `record` |
| `replay` | `replay`: the replay record |
| `export` | `path`, `replaced`, `calls`, `failed_calls_left_out`, `calls_without_a_turn_left_out`, and `steered`: each vector the calls were steered with (`file`, `sha256`) and its `call_ids`, empty if none was |
| `chats` | `chats`, each with `log_id` (the ID above: the log file's name), `openable` (whether that name is one `/resume` takes: letters, digits, `_`, `-`, and dots after the first character), `conversation_id` (the ID inside the log; null if it can't be read), `path`, `profile`, `description` (the profile's), `model`, `started`, `title`, `title_set` and `error` |
| `profiles` | `profiles`, each with `name`, `model`, `model_name` (the last part of the model's tokenizer repo), `backend`, `description`, `steered`, `local` (in `local/profiles/`), `fetched` and `error` (null, or why it doesn't load); and `not_checked`, a sentence on what isn't checked until a start |
| `profile`, `new` | `conversation_id`, `profile`, `model`, `mode` and `warnings` |
| `resume` | the same, plus `resume` (the record), `head_text` and `differ`: `count`, how many things differ today, and `items`, each a sentence the terminal also prints and the `count` of things it covers. Counted: a changed profile file, each changed `models.toml` key, each setting today's files would set differently, each fact changed since the last call, a prompt that renders differently, and a context size the log doesn't give. Not counted: an unanswered message, a guessed head, restored `/set` values and facts the log doesn't record. |
| `cancel` | `cancelled`, and `command` or `reason` |
| `state` | `running` (null, or the command running: `command`, `args`, and for `/regen N` `k` and `n`, which reply of how many is being written), `reply_so_far` and `reply_end` (its length in code points; empty and 0 when nothing runs), `model` (`none`, `starting`, `ready` or `stopped`, read from the backend when `state` runs), `model_name`, `start` (a random value made when Clarity starts, the same in every `state` until it stops; the web page stops when it reads another, which means another program answers on its port), `clarity` (`version`, `commit`, `dirty`), `llama_cpp_commit`, and the conversation's `conversation_id`, `log_id` (its log file's name without `.jsonl`, which `/resume` takes; a copied or renamed log's differs from its conversation ID), `title`, `profile`, `head`, `turns` (as `history` gives them), `unanswered` (`turn_id` and `text` of your last message if its call failed, else null), `warnings` and `resume_differ` (after a resume, until your next message, answered or not (a Regenerate or Edit that fails keeps it): what differs today, as `/resume`'s `differ`; else null). Before any conversation, those are null or empty. |
| `help` | `commands`, each with `usage` and `help` |
| `quit` | `quit`: true |

Commands run one at a time, in order. `cancel`, `state`, `chats` and `profiles` run as soon as they're read, so `cancel` can stop a reply that's being written and the others can answer during one; `settings` and `prompt` do too while a reply is being written or replayed (not while a model starts). `cancel`'s result names the command it stopped (`"cancelled": true, "command": "say"`) or gives a `reason` it stopped nothing. Send it after the command has started (after its first event): a `cancel` read while the command is still queued finds nothing running. A record that lacks a field a read needs, in a log another program wrote, is a `ConfigError` naming the field (for `history`, `branches`, `compare`, `prompt`, `probs` and `settings`, as for `/resume` and export), and the session goes on. `replay` checks the call's record before anything starts and names what's missing. End of input quits, like `quit`. Input is UTF-8, whatever the console's encoding; a line that isn't is an `InputError`. Output is ASCII: other characters are written as `\u` escapes.

## The web interface

The web interface (from v0.3.0) is a local page over the same command table as `--json`. The server listens on `127.0.0.1:8765` only. Its routes:

| Method | Path | Needs the secret | Returns |
|---|---|---|---|
| GET | `/` | no | the page's shell, which holds no data |
| GET | `/static/FILE` | no | the page's script and styles, from a fixed list |
| POST | `/command` | yes | `{"command", "args", "id", "tab"}` in, and `--json`'s result for that command out, as `{"id", "result"}` or `{"id", "error", "type"}`. After an unexpected error has stopped Clarity, every command answers `{"id", "error", "type": "Stopped", "stopped": true}`, with that error's sentence |
| GET | `/events?tab=TAB` | yes | a stream of JSON lines: first `{"event": "hello"}`, then `--json`'s events, and `{"event": "heartbeat"}` when nothing has happened for 15 seconds. Clarity stopping sends `{"event": "stopped"}` (Ctrl-C in its terminal, or SIGTERM) or `{"event": "stopped", "reason"}` (an unexpected error, in a sentence). An event holding a number JSON can't carry (NaN or infinity, from a log another program wrote) comes as `{"event": "error", "type": "NotJSON", "message"}` instead. A tab that isn't the live one gets `{"event": "not_live"}`, and the stream ends |
| GET | `/view/NAME` | yes | an HTML fragment of one part of the page, such as `thread`; a `call` parameter must be a call ID |
| POST | `/tab` | yes | `{"tab"}`: makes that tab the live one |
| GET | `/export` | yes | the export this start made of the log that's open, if any |

Every data route needs `Authorization: Bearer SECRET`, where the secret is made for each start and reaches the page in the link's fragment (`#key=…`), which browsers never send to a server. A POST also needs the exact `Origin` and `Content-Type: application/json`. Refusals use their status codes: 400 a request that isn't valid, 401 no or a wrong secret, 403 a wrong Host, Origin or fetch site, 404 an unknown route or view, 405 a method a route doesn't take (OPTIONS included), 409 a tab that isn't the live one or an export not made yet, 413 a body over the limit (8 MB for `/command`, 64 KB for any other route, named in the sentence), 415 not JSON, 500 a result holding a number JSON can't carry (NaN or infinity, from a log another program wrote), 503 a view whose read the command table refused (after the session stopped, for one), with the sentence. A refused request never ends the session. One tab at a time is live: a tab is live when it opens if no live tab is listening, and another takes over with `POST /tab`.

The views read only through the command table, the way a script would; while a model starts or a reply is written, they ask only the commands that answer at once. A read that waits its turn (another view's `/branches`, say) doesn't count as a command running:
- `thread`: the conversation, or the first screen when it has no messages yet: a pointer to earlier chats when there's no chat, the profile being started, or the new chat's profile, model and backend, with its start warnings. Messages are plain text, from `/state`'s `turns`, with each reply's marks and note. While a command runs, they come without their controls, and so do a reply and a failed message's line whose call ID isn't one Clarity writes (in a log another program wrote). Otherwise `/branches` gives your message's version ("2 / 3") and a reply's count of replies, and `/prompt` the last reply's facts line and a failed message's line: the last call, when its message is back in the box (`unanswered`).
- The facts line under the last reply, from its call record: plain facts on one line, joined by " · ": "steered ×S" (or "not steered" on a profile with a vector), "stopped at the token limit (max_tokens N)", "stopped: the context is full", "stopped by you", "empty reply (nothing, or only blank space)". Each `!` item gets a line of its own: settings not applied (counting those that could change the reply), "text read as a control token", "some output token IDs are missing, so replay compares text", "the text differs from its token IDs", the failed checks by name, and the call's warnings. A failed `server_running` check offers "Start the model again", which sends `/resume` with this chat's `log_id`.
- `picker`: the row under the message box. The picker button names the profile and model (`profile · model_name` from `/state`); its list is `/profiles`, read again each time it opens, grouped by model, models with a fetched profile first and profiles that don't load last. "Settings · N changed" counts the settings `/settings` gives with origin `chat`. The context figure ("Context 5% · 412 of 8,192 tokens") is the last call's prompt as a share of its context (`n_prompt_client` of `n_ctx`), in the warning colour over 80%; a message refused because it didn't fit isn't counted, so the reply at the head is.
- `about`: the version, the commit and the llama.cpp commit from `/state`.
- `replies?call=CALL_ID`: the replies pop-out for the message that call answers, from `/compare CALL_ID`. Each row is "reply k of n", its first words, and only what differs: the seed, the settings when the replies fall in groups, how it stopped, and "continued, N more turns" for a reply that went on. A divider names the change where a group starts, and says when its seeds are the group before's again. It opens under the reply and its toolbar. For your last message, where `/regen` acts, it adds More replies (`/regen N`, its button "Make N" with the box's number) and, when `again` isn't null, "Again with the same N seeds, now at …" ("the same seed" for one; `/regen same`).
- `settings`: the Settings panel, from `/settings`: each setting's value, its origin and a line on what it does (the Settings table's "What it does", below), sliders for temperature, top_p and min_p with the neutral value and the model's recommended one marked, and, on a row changed in this chat, one way back: "Back to X (origin)", "Neutral (X)" when the profile's value is a sampler's neutral one, or "Off (0)" for `logprobs`. Steering shows the vector's file, 12 characters of its sha256 and its blocks, with "Off (0)" whenever the strength isn't 0. Without `penalties` in the samplers, the four penalty rows are one line. On Transformers, the logprobs row shows, and every entry in `not_applied` is listed with its reason, the sampler order included. The seed box takes `random` as well as a number. A row sends `/set NAME VALUE` when a slider is released or on Enter in a box or leaving it; "Back to" and "Off" send `/set NAME` (or `/set steering_scale 0`); a refused value keeps the backend's sentence under its row, and an empty box is refused in the page ("Type a value; an empty box changes nothing.") and sends nothing, since `/set NAME` would put the setting back. Reset all sends `/set NAME` for each changed row; a refusal shows as a notice. While a reply is written, the controls wait: "Changes wait until this reply is done."
- `details?call=CALL_ID`: one call, from `/prompt CALL_ID`. What the model got comes first: the prompt as Text, with what the template or the profile wrote highlighted (`message_spans`; without them, the prompt is unhighlighted and the panel says why), as Whitespace, with each line end shown, and as Token IDs, with the count and whether the prompt starts with the start token (its first ID against `bos.bos_token_id`; not said for a call with no `bos` record or prompt IDs). The summary says how it stopped (with the end token's text from `/settings`' `eog_tokens`, or its ID), its length and time ("time unknown" when the record has no timings), the model, profile and call, a failed check first with what failing means, and settings not applied. Replay sends `/replay CALL_ID` (a line under it says what that does), and waits while another command changes the chat; Stop sends `/cancel`; the result is a short label from the replay record, then the replay's own flags, each fact that changed since the call (logged value → now, up to five per fact), and what the log didn't record, in the terminal's sentences; the section counts the call's replays. Then the settings as logged (closed at first), with their origins, a sampler at its neutral value marked "(neutral)" and `logprobs` or `steering_scale` at 0 marked "(off)"; Probabilities, from `/probs CALL_ID`: while the section is open, the reply is shaded token by token (under 10%, 10–30% and 30–60%; over 60% unshaded), or, when the tokens can't be placed in the reply's text, they're listed in the panel unshaded; picking a token, or moving with the arrow keys, shows its chance, its log-probability and the likeliest tokens the record holds; without probabilities, the section says why, by `reason`, and on Transformers offers to record them from the next reply (`/set logprobs 5`). `/probs` runs once per call while the server runs, since the record doesn't change. If `/probs` refuses the record (a log another program wrote), the section says "can’t be read" with the sentence; if `/compare` does, the heading leaves out "reply k of n"; the rest of Details draws as usual. Last, the whole record. A failed call shows its error and no Replay or Probabilities.
- `branches`: the Branches panel, from `/branches all`: every branch end, newest first, headed by where its branch left the rest ("From reply 2 of 4 · seed 43", "From your edited message · seed 42", or "Reply 4 of 4 · seed 45, not continued"), with its last reply's text, its length in turns and how many replies on it are marked; and, with "Marked only" on (a filter in the page), the marked replies with their labels, note and branch. "Show this reply" sends `/switch`; for a marked reply the page then scrolls to it and outlines it. The top bar's "Branches N" counts the branch ends, from the second one. While another command changes the chat, the panel waits and says why, as the Chats drawer does.
- `chats`: the Chats drawer, from `/chats`: every chat, newest first, under Today and Earlier by the local time at the start of its ID, today's with the time it started (14:25) and earlier ones with the date; each row is the title (or "No messages" before the first one) and the profile, and the open chat (`/state`'s `log_id`) is marked. A log that can't be read is listed by its ID with "! can't be read". Picking a row sends `/resume ID`; the open chat's row just closes the drawer. A refusal (a `ConfigError`: a log that can't be resumed, a model that isn't fetched, a changed steering vector) shows under that row as "! Didn't open." with the sentence, and the open chat stays as it was; a failed model start shows above the message box. "New chat" sends `/new`. While another command changes the chat, the rows wait and say why: "Waits until the reply is done.", "Waits until the model has started." or, for any other command, "Waits until the command is done."
- After a resume, until your next message, answered or not (a Regenerate or Edit that fails keeps it), the thread says how many things differ today, from `/state`'s `resume_differ` ("2 things differ from when this chat was written"), with "Show what changed" to list its sentences, or "Nothing Clarity checks differs from when this chat was written." A resumed chat whose last message got no reply says so once, in place of the failed line, and the message is back in the box.
- The title menu, once the chat has a message: "Rename…" turns the title into a box (120 characters at most); Save sends `/title TEXT` and waits while the box is empty; Cancel, Escape or leaving the box keeps the title. "Export for a Python notebook" sends `/export`. While another command changes the chat, the title menu waits.
- `export`: the sheet after Export, from this start's last `/export` of the open log (else refused with 409; a copied log has its own): Download (`GET /export`, saved under the export's file name), the file's path, whether it replaced an earlier export, the calls left out (`failed_calls_left_out` plus `calls_without_a_turn_left_out`), and, when `steered` isn't empty, the vector files to send with it (a path with control or invisible characters shows them as escapes, such as `\x1b`, with no Copy). A refused export shows the sentence with Try again.
- "Start the model again" sends `/resume` with the open chat's `log_id`; a refusal shows "! The model didn't start again." with the sentence's first line, and the rest (llama-server's output) in a box that scrolls.
- `compare?call=CALL_ID`: the replies side by side, from `/compare`. Replies with matched seeds line up in rows, one column per group of settings, with a marker where each first parts from the first column's (`first_different_char`), and a line under the lead sentence that says what the marker means; "the same until the shorter one ends" for a `prefix` pair and "the same text" for identical ones. Otherwise they're cards, with a sentence when both the seed and a setting differ. "Same in all" lists the profile, model and backend, the settings in `same` except the sampler list, settings at their neutral or off values ("not steered" stays on a profile with a vector), and `repeat_last_n` when every penalty is neutral; then "the prompt" when `same_prompt` is true.

Appearance (in the Clarity menu) sets the theme (Notebook, Neutral, Sky, Space or High contrast), Light, Dark or Match system, and, in Sky and Space, Motion. Changes apply at once and are kept in the browser's own storage, so Clarity writes no file for them, and they aren't logged. Only the scenery around the page in Sky and Space moves; it starts off when the system asks for reduced motion, and the Motion switch overrides that either way.

The keyboard: Enter sends and Shift+Enter starts a new line (not while an input method is composing). While a model starts, Send waits: Enter or Send says "Waits until the model has started." above the box and keeps the text, and the sentence goes once the model is ready. A reply's toolbar and the replies pop-out's rows are one Tab stop each; the arrow keys move inside, and Tab comes back to the control last used. Sent from the message box, a reply leaves the focus there; started from a toolbar or the pop-out, the focus goes to Stop, then to the button that started it on the new reply, and "Reply finished" is announced. Below 1,100 px wide, a side panel opens over the thread; below 760 px, Branches shows as its icon and count.

Quit Clarity in the page first cancels a reply being written (a step of the page's own), then sends `quit`. `quit` over `/command` answers, then stops `clarity ui`, as Ctrl-C does; it doesn't cancel anything itself. "Help and docs" opens these docs on GitHub at the running commit (or at the version's tag, outside a git checkout), so it needs the network. A page that loses its connection says "Clarity isn't answering. Trying again…" after a few seconds and keeps trying every 2 seconds; once Clarity answers, it loads the page again. "Clarity has stopped" is for Quit, or a Clarity that stopped after an unexpected error.

## Profile keys

A profile is a TOML file in `profiles/` (shipped with Clarity) or `local/profiles/` (yours; git ignores it). Its name is the file name without `.toml`, and a name in both folders is an error. Unknown keys are errors, and the allowed keys depend on `mode`. A `{ file = "PATH" }` is read byte for byte, relative to the profile, and must be inside the project folder, after following links.

All profiles:

| Key | Required | Value |
|---|---|---|
| `mode` | yes | `"chat"` (the model's chat template) or `"text"` (a plain document). |
| `backend` | yes | `"llama.cpp"` or `"transformers"`. |
| `model` | yes | A model name from `models.toml`. |
| `description` | no | One line saying what the profile is for, shown by `/profiles` and `clarity chats`. It's logged with the profile's file, like every key. |
| `settings` | no | A table of [settings](#settings). Unset ones come from `defaults.toml`. |
| `use_recommended` | no | `true` fills unset settings from the model's `generation_config.json` before `defaults.toml`. Default `false`. |
| `device` | no | Transformers backend only: `"mps"` (default), `"cpu"` or `"cuda"`. |
| `ctx_size` | no | The context size in tokens: the prompt and the reply together must fit in it. Default: `ctx-size` in `server.toml`. A larger context takes more memory, and a size above what the model was trained on is refused. Starting a profile with another size restarts llama-server; the Transformers backend only checks the new size. |
| `steering` | no | Transformers backend only: a table naming a [steering vector](#steering-vectors). `file` is its path relative to the project folder, with `/` between folders on every system, and must be inside it. `sha256` is its hash; without it, Clarity refuses the profile and gives the hash to add. `layers`, optional, lists the blocks to steer; the default is every block in the file. A profile with `[steering]` must set `steering_scale` in `settings`. |

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

| Setting | What it does | Value | Neutral (`defaults.toml`) | Needs sampler |
|---|---|---|---|---|
| `temperature` | Lower is steadier, higher more varied. 0 always picks the likeliest token. | number ≥ 0; 0 is greedy | 1.0 | `temperature` |
| `top_k` | Picks only from the k likeliest tokens. 0 is off. | integer ≥ 0; 0 is off | 0 | `top_k` |
| `top_p` | Picks only from the fewest likeliest tokens whose chances add up to this share. 1 is off. | number from 0 to 1 | 1.0 | `top_p` |
| `min_p` | Drops tokens less than this fraction as likely as the top one. 0 is off. | number from 0 to 1 | 0.0 | `min_p` |
| `repeat_penalty` | Makes tokens already used less likely, counting the prompt as well as the reply so far. On Transformers it looks at everything; on llama.cpp only the last tokens set by repeat_last_n. 1 is off. | number > 0 | 1.0 | `penalties` |
| `repeat_last_n` | How many of the latest tokens, prompt included, the penalties look at (64 by default). 0 turns all three penalties off. | integer ≥ 0 | 64 | |
| `presence_penalty` | Lowers the chance of every token that appeared in the last repeat_last_n tokens, by the same amount however often it appeared. 0 is off. | number | 0.0 | `penalties` |
| `frequency_penalty` | Lowers a token’s chance more for each time it appeared in the last repeat_last_n tokens. 0 is off. | number | 0.0 | `penalties` |
| `seed` | The same seed, settings and conversation give the same reply on the same computer and software. Used for the next message you send or edit. Regenerate and More replies take the next seed up instead; to run a reply again with its own seed, use Replay in Details. At temperature 0 nothing is random, so the seed changes nothing. | integer from 0 to 4294967294, or `"random"` (Clarity draws one per call and logs it) | `42` | |
| `max_tokens` | The most tokens a reply can have (a token is a word or part of one). It can stop sooner. | integer ≥ 1: the longest reply | 1024 (a cap, not a neutral value) | |
| `samplers` | The samplers that run, in order. A sampler not listed doesn’t run. | the order and membership of the sampler chain, from `penalties`, `top_k`, `top_p`, `min_p`, `temperature` | all five, in that order | |
| `logprobs` | 0 is off. N records, for each token of the reply, its log-probability and the N likeliest tokens with theirs. | integer from 0 to 20, recorded in the call's `logprobs` field. A log-probability is the natural log of the probability the model gave the token, so 0 is certain and more negative is less likely. The probabilities come from the model's raw scores (logits) before any sampler, so they don't show what temperature, `top_k` or the penalties did. Transformers backend only for now; a llama.cpp profile with it above 0 is refused. While a reply is written, the scores take memory: about 0.4 MB per token for a model with a 100,000-token vocabulary. | 0 | |
| `steering_scale` | Strength: the profile’s vector is multiplied by it before it’s added. | any number; 0 is off, and a negative number steers the other way. A profile with `[steering]` must set it; one without must leave it at 0. Transformers backend only. | 0.0 | |

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

The shipped models are SmolLM2-135M-Instruct, OLMo 3 7B Instruct and base, Llama 3.1 8B Instruct and base (gated), and bge-small-en-v1.5 for the embedding tests.

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
| `logs/exports/ID.jsonl` | A conversation exported for a notebook (`clarity export`, `/export`), named by its log file, so a copied log's export doesn't replace the original's. `clarity chats` doesn't list it. |
| `local/clarity.lock` | Locked while `chat`, `ui`, `run` or `replay` runs, so only one of them runs in a project at a time. Empty. |
| `local/clarity.json` | Who holds the lock: process ID, command and start time, and the running llama-server's port, process ID and key. Readable only by you on Linux and macOS (mode 0600); on Windows it gets your folder's permissions. Removed on a normal exit; one left behind means Clarity didn't exit normally, and the next start uses it to stop that server. |
| `local/clarity-open.html` | The private page that opens `clarity ui`'s link, with that start's secret. Readable only by you on Linux and macOS (mode 0600). Deleted when the page it opened connects, and when Clarity stops; Enter in Clarity's terminal writes it again. |

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
- **Probabilities:** `logprobs`, with the `logprobs` setting on (Transformers only): `view` (`raw logits, before any sampler`), `n`, and `tokens`, one per output token: `id`, `logprob` (natural log) and `top`, the `n` likeliest tokens as `[id, logprob]` pairs, likeliest first. A cancelled call has `{"unavailable": "cancelled"}` instead: Clarity can't get the scores from a reply it stops part way. A replay of a call with probabilities records them too.
- **Steering:** on the Transformers backend, `request.steering`: `file`, `sha256`, `scale` and `layers` for a steered call, or null. A call logged before v0.2.2 has no `steering` key, which means it wasn't steered. A steered call that finished adds `checks.steering`: `ok`; `model_passes`, how many forward passes the model ran; and `forward_passes`, how many times each steered block ran (block number to count). It's ok when every block ran once in every pass and there was at least one pass per reply token. On MPS, a reply that ends before `max_tokens` has one pass more than its tokens: Transformers runs it and then undoes it. A cancelled call has no steering check, since a cancel can land mid-pass.
- **Checks:** `checks`, `flags`, `warnings`, and `error` if the call failed.

`output_ids` include the tokens of a stop string and the end-of-sequence token; `output_text` leaves out the stop string.

A replay compares the output token IDs and stop type, or the text if the logged IDs were incomplete. Its `differences` list every recorded fact that changed since the call:
- llama.cpp: the build and checkout, server flags, `/props`, `GGML_*` and `LLAMA_*` variables, and the GGUF's sha256.
- Transformers: the torch and Transformers versions, device, dtype, attention implementation, `PYTORCH_*` and `TORCH_*` variables, deterministic-mode state, and the weights' commit.
- Both: the machine, the client's commit, and the rendering libraries.

A replay of a steered call applies the logged vector at the logged scale and blocks. It checks the logged steering like a profile's: the file must be inside the project folder, have the logged sha256 and fit the model, or the replay is refused; it never replays a steered call unsteered. Clarity before v0.2.2 ignores `request.steering`, so it replays a steered call unsteered.

`unexplained_difference` is true when the output differs and nothing changed except, at most, the client's commit or the rendering libraries. Neither can change a replay, since it sends the logged IDs and request.

## Export

`clarity export` and `/export` write a conversation as JSON Lines, so a notebook reads it with a loop or `pandas.read_json(path, lines=True)`. The file is `logs/exports/ID.jsonl`, ID the log file's name, unless `clarity export --out` names another; a log whose file name has characters other than letters, digits, `.`, `_` and `-` needs `--out`. The file is readable only by you on Linux and macOS, like the log. Replay records aren't exported: they re-run calls already there. A batch run's `calls.jsonl` can't be exported.

The first line is the header, `"type": "export"`:
- `format_version` (1), `clarity` (`version`, `commit`, `dirty`) and `conversation_id`.
- `calls` (how many follow), `failed_calls_left_out` (calls that produced no reply) and `calls_without_a_turn_left_out` (calls whose reply's record is missing from the log, as after a crash while it was written; resume carries on without them too).
- `conditions`: one entry per distinct set of conditions the calls were computed under, since a resumed conversation can span several server starts, machines or builds. Each has `conditions` and `call_ids`, the calls it covers. `conditions` holds `backend`, `weights` (the repo and revision, plus the GGUF's sha256 for llama.cpp), and every fact replay compares except Clarity's own commit and the tokenizer libraries: for llama.cpp, `build_info`, `llama_cpp`, `flags`, `graph_splits`, `props`, `env`, `end_tokens` and `machine`; for Transformers, `torch`, `transformers`, `device`, `dtype`, `attn_implementation`, `env`, `deterministic_algorithms`, `end_tokens` and `machine`. It also holds `steering`: the `sha256`, `scale` and `layers` the calls were steered with, or `"none"`, so a `/set steering_scale` part way through a conversation starts a new group. A null is a fact the log didn't record.
- `note`: "Recomputing these calls under other conditions can give different numbers, sometimes by a tiny amount and sometimes enough to change the likeliest token. Run `clarity_chat.recompute.check` with your loaded model first; it lists what differs." When any call was steered, it adds: "Some calls were steered; apply the same vector with `clarity_chat.steering.applied` before recomputing them."

Then one line per call that produced a reply, `"type": "call"`, in log order:
- **Where it sits:** `call_id`, `turn_id` (the reply), `parent_id` (your message), `regen_of`, `edit_of` (your message replaced an earlier one), and `path`, the turn IDs from the start of the conversation to this reply.
- **What went in and came out:** `prompt_ids` as sent (with any BOS), `output_ids` as generated, `prompt_text`, `output_text`, `stop_type`, and `cancelled` (true for a reply cut short; its output is incomplete).
- **Settings:** `settings` (each with its value and origin), `n_ctx` and `flags`, as logged.
- **Message spans:** `message_spans`, as in the call record. A call logged before v0.2.1 has none, so the export works them out with the profile and tokenizers as logged, and checks the result rebuilds the logged prompt exactly. If it can't, the field is `{"unavailable": REASON}`, never a guess. A template that writes today's date can't be checked on a later day.
- **Marks and notes:** `marks` (`label`, `origin`, `time`) and `notes` (`text`, `origin`, `time`), as they are now, after any `unmark` or replaced note.
- **Model:** `model`, its pins: `name`, `repo`, `revision`, `route`, `gguf_sha256` and `gguf_path`.
- **Probabilities:** `logprobs`, if the call logged them.
- **Steering:** `steering`, the call's `request.steering` (`file`, `sha256`, `scale`, `layers`), or `"none"` for a call that wasn't steered, including every call logged before v0.2.2.
- **Facts:** `facts`, what replay compares, from the call's own server start: the end tokens, and the build, flags and machine or the library versions, device and precision.

## Steering vectors

A steering vector is a file made by another tool, such as repeng; Clarity applies one and never makes one. The file is llama.cpp's control-vector GGUF: one 1-D float32 tensor named `direction.N` for each block N it steers, all the same length (the model's hidden size), and optionally a `controlvector.model_hint` key naming the model type it was made for. Steering adds the scale times `direction.N` to the output of block N, counting from 0, at every position: the prompt's and the reply's. Block 0 can't be steered: llama.cpp keeps no vector for it and its loader refuses `direction.0`, and Clarity keeps a vector usable on both.

Clarity reads the file with its own reader, which accepts only this format: GGUF version 3, little-endian, at most 16 MB, 1 to 1,024 tensors, every value finite. Anything else is refused with what was wrong.

Steering runs on model types `llama` and `olmo3`, the two whose blocks Clarity has checked. Other types, including Qwen3 and Apertus, are refused until they are checked. `llama` admits every Llama-architecture model, such as Hermes 4, though only SmolLM2 and Llama 3.1 8B Instruct have live tests.

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
- `with applied(model, vector, scale, layers=None) as runs:` steers a loaded Transformers model inside the block, and `runs` counts how many times each steered block ran. It computes `scale × direction.N` in float32 and casts it once to the model's dtype, as repeng does (llama.cpp keeps it in float32), then adds it in that dtype. It expects a model on one device. Scale 0 adds nothing and doesn't mark the model, so the recompute check reads it as unsteered. The hooks come off when the block ends, however it ends, and a model already inside `applied` at a non-zero scale is refused.

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

| Check | What it checks | A failure means |
|---|---|---|
| `prompt_echo` | The server’s decoding of the token IDs equals the prompt text. | The server’s text for the token IDs differs from Clarity’s, so the model file and the tokenizer may disagree. |
| `settings_echo` | Every value sent comes back unchanged in the server’s generation_settings. | A value Clarity sent didn’t come back unchanged from the server. |
| `unaccounted_settings` | The server reports no setting Clarity didn’t send or expect. | The server reported a setting Clarity didn’t send or expect. |
| `tokens_evaluated` | The server read as many prompt tokens as were sent. | The server read a different number of prompt tokens than were sent. |
| `no_cache_reuse` | No prompt tokens came from a cache. | Some prompt tokens came from a cache. |
| `server_running` | The llama-server process Clarity started is still running. | The model’s server had stopped. The reply is complete, but the next message needs the server started again. |
| `server_props` | Logs before v0.2.1 only: after the call, /props showed the server still running, with the model and context size it started with. | After the call, the server had stopped, /props named another model or context size, or /props couldn’t be read. |
| `generation_config` | Transformers only: the effective generation config equals what was sent. | Transformers’ generation settings differed from what was sent. |
| `steering` | Transformers only, on a steered call: every forward pass ran each steered block exactly once, and there was a pass for every generated token. | A forward pass didn’t run each steered block exactly once, or a generated token had no pass. |

`settings_echo` compares floats at float32 precision, as the server stores them, so `0.6` sent and `0.6000000238418579` echoed is a match. A failed `server_running` check records the process's `exit_code`; the server's model and context size are checked through `/props` once, after it starts. `steering` records `model_passes` and `forward_passes`, by block. Details in the web interface shows each check with these words.

A batch run stops on any error, any `check_failed:*` flag, a cancelled call, or a server restart. Other flags are logged, and each flagged call is named on screen with its flags; other warnings print once.
