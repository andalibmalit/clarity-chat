# Changelog

All notable changes to Clarity are listed here. The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versions follow [Semantic Versioning](https://semver.org/).

Two headings are Clarity's own, used when they apply:
- **Migration:** files on disk or the log format changed, and what to do about it.
- **Output may change:** a pinned version or a default moved, so the same call can give a different reply. Replay lists such differences.

## [Unreleased]

### Added
- A model file that bans tokens from every reply (`tokenizer.ggml.suppress_tokens`, which llama.cpp writes for Gemma 4 Unified and MiniMax models) is recorded on llama.cpp: `server_start` gains `model_file.suppress_tokens` (`[]` for none), and on such a file each call's `backend_settings` lists the ban with origin `model file`, `/settings` shows it (and no longer lists the model's `suppress_tokens` as not applied), and a new `suppress_tokens` check confirms no banned token is in the reply. A list llama-server's startup log doesn't show whole refuses the start. Logs from before v0.4.0, made on a file that bans tokens, don't record the ban, which llama-server applied all the same.

### Changed
- The terminal's `/prompt` shows each control character but newline and tab, and each text-direction control, as ⟨U+001B⟩, so text in a prompt can't recolour or rewrite the terminal. The text sent and logged, `/prompt repr` and `--json` are unchanged.
- Docs: the llama.cpp build lines no longer download llama.cpp's web UI (`-DLLAMA_USE_PREBUILT_UI=OFF`), whose download from Hugging Face fails today; Clarity runs llama-server without it.

### Output may change
- llama.cpp moves to v1.0.0 (`b1b770d`): check out that commit and rebuild llama-server, as [Getting started](docs/getting-started.md) shows. CPU replies can differ from ones logged with the old build, since CPU flash attention now sums V in float32 (`7f24aac`); replay lists the build as the difference.

## [0.3.2] - 2026-11-17

Live tests on an M2 Max (macOS), 2026-11-17, at the release's code (built in the sprint to 1.0, so these were run by the build, not yet by the maintainer): `-m live` 25 passed, `-m hf` 20 passed, `-m olmo` 12 passed, `-m llama` 3 passed, and `-m browser` 160 passed (the web interface in Chrome; CI also runs it in Firefox and WebKit).

### Added
- A chat that searches documents searches each message you send with the words you typed, and adds the best parts that fit the budget to your message, at its start or end. The parts stay with their message: a regeneration reuses them, an edit is searched again, and a part already in the chat isn't added twice. Each searched message's turn records `found`: the words, the documents' builds, the engine, each part added and each match left out with why.
- A missing or unreadable document (one whose `parts.jsonl` isn't the one Clarity built, say), a Python without SQLite's FTS5, or a part holding the model's control token, refuses the call with the way out, every such document named in one refusal, and the message comes back. In the web interface the refusal says "Nothing was sent." with Stop searching it for each document, and a full context offers a new chat with the same documents.
- `/history` shows, under a searched message, the words and the parts added; its `found` gives each part's `page` and `tokens`, and `considered`. `/prompt` gives the call's `found` and `placement`. The terminal says why a part matched as the page does ("in its title: …; in its text: …").
- In the web interface, the attach sheet's **Search it** adds a file to your documents (its Name box filled in; a document of that name is replaced, said before and after) and searches it, and **Searching** beside Settings ticks and unticks what the chat searches (a change applies from your next message; Regenerate keeps a message's parts). A reply to a searched message opens with **From documents** (what was added, each part's rank and why, and what was left out), and Details has a Documents section.
- Adding a file under a name that's taken replaces that document, and Clarity says so before and after. The library holds nothing of yours: search reads `parts.jsonl`, checked against its build ID, so editing `text.md` changes nothing; to change a document, edit your file and add it again.
- A refusal whose way out is adding the file again names that way on each front end: in a chat and `--json`, the chat's command (`/documents add PATH NAME`), `clarity document add FILE --name NAME`, and in the web interface the paperclip's Search it, under the same name. The terminal says what search added after each reply, `/new` and `/profile` say when a chat's own search list isn't kept, and `clarity document try` takes several names before `--profile`.
- `/edit --drop N` leaves the message's Nth attachment out of the edit; in the web interface each attachment Edit keeps is a chip with an × that does the same.
- An export's rows list each searched message's words, documents' builds and the parts added, with their token ranges in the prompt, and the export says it holds those parts' text.
- `/try TEXT` and `clarity document try NAME … --profile P TEXT` show what a message would find, with no model call: the same object a searched message's turn holds, and the refusal a message that found it would meet. `clarity run` with a searching profile searches each opening once for every seed, checks the documents before any model loads, records their builds, stops if one is added again mid-run, and says when two profiles got different parts for an opening, and what differs between them.
- A chat-mode profile's `[documents]` table gains `search` (the documents a chat searches), `budget` (the tokens search may add to a message, default 1,500) and `placement` (`"start"` or `"end"`: where the block of documents goes in your message). In a chat, `/search` lists or changes the documents searched; it's logged as a `search` record and kept on resume; `/documents` lists and adds them; `/state` gains `searching`. A searching chat's suggested start for an attachment leaves the budget free too.
- `clarity document add|list`: the library of documents Clarity will search, in `local/documents/NAME/` (a copy of the file, its text as `text.md`, its parts and a manifest). Clarity says where it writes before and after, builds each document in a temporary folder and renames it into place, and refuses to search a document whose parts aren't the ones it built.

### Output may change
- An HTML file's text (version 2 of Clarity's HTML rules) puts a blank line between block elements and turns `h1` to `h6` into `#` headings, so a saved web page gets paragraphs, and in your documents titled parts. Its tokens differ from v0.3.1's for the same file, and its lines are numbered differently, so a range such as `lines 1-20` picks different text. Old logs and replay are unaffected: a call replays its logged IDs, and a resume reads the logged text.

### Changed
- The refusals for a file Clarity can't read say "use that copy" rather than "attach that", since the same file can now go into your documents too.

## [0.3.1] - 2026-11-15

Live tests on an M2 Max (macOS), 2026-11-15, at the release's code (built in the sprint to 1.0, so these were run by the build, not yet by the maintainer): `-m live` 23 passed, `-m hf` 19 passed, `-m olmo` 12 passed, `-m llama` 3 passed, and `-m browser` 150 passed (the web interface in Chrome; CI also runs it in Firefox and WebKit).

### Added
- A chat-mode profile's `[documents]` table sets the line that goes before attached documents in your message (`framing`) and the first line of an excerpt (`excerpt_marker`); `""` means none. Attaching arrives with this release's commands.
- `/attach PATH` attaches a file to your next message, whole if it fits; `/attach PATH pages A-B` or `lines A-B` attaches an excerpt, and `/attach PATH check` counts it without attaching. `/attach` lists what's waiting and `/detach [N]` removes it. A file that doesn't fit is refused with the numbers, the start that still leaves room for your message, and how far the start that fits goes; one that surely doesn't fit isn't counted whole ("needs more than N tokens"), so a huge text or an amplifying PDF is answered in seconds. `/state` gains `pending` and `reading`, and `/history` lists a message's attachments.
- `clarity chat --attach FILE` (repeatable, with `--profile` or `--resume`, and with `--json`) attaches files to your first message. They're read before the model starts, so a file that can't be attached stops Clarity before a long load. The terminal says "Reading NAME…" while a file is read, shows what's waiting before each prompt, and says when quitting drops attachments that were waiting; `--json` gets a `reading` event and `/quit`'s `dropped`.
- In the web interface, the paperclip beside the message box attaches a file: a sheet shows what it costs and what it leaves, and offers it whole, in part (preselected so a first question still fits) or in a new chat; the attachments waiting show as chips. The paperclip waits while a command runs and, with a text-mode profile, says why before the file picker opens. Your sent message lists its attachments with what each cost when attached, and Edit says what it keeps; Details highlights the framing line and documents Clarity put in your message apart from the template's text, and lists each attachment with where its block sits. The sheet says what its suggested range leaves room for, that a PDF's pages count from the file's first page, and opens a PDF's or web page's extracted text; Details lists the framing line and each attachment with its tokens. A send that doesn't fit says by how many tokens and what to do (shorten the message, remove an attachment it had, start a new chat, or lower max_tokens); its text and attachments come back. `POST /upload` takes the file's bytes into memory (the page attaches by upload ID, never by path).
- For log and `--json` readers: a message sent with attachments has `attachments`, `framing`, `framing_origin` and `placement` on its `turn` record, and a `special_tokens_in_content` entry names the attachment the text is in. `/new`, `/profile` and `/resume` results gain `dropped`, the attachments that were waiting. A turn without attachments is written as before. The message's `message_spans` entry gains `added`, where the framing line and each document's block sit in the prompt; an export's rows gain `attachments`, and its result says whether the file holds attached documents' text (`attachments`, true or false), as the export's message does.
- A new dependency, `pypdf` 7.1.0, pinned exactly, to read the PDFs you attach to a message. Its version can change the text extracted from a PDF, so a bump comes in a change of its own, as for `tokenizers`, `transformers` and `jinja2`.
- Compare's cards have the thread's bookmark: mark a reply, or note it, without leaving Compare.

### Output may change
- Conversions (`clarity fetch` on the convert route) now run in an isolated environment: llama.cpp's pinned converter requirements on a fresh base. Run from Clarity's folder, the converter's environment was built on Clarity's own `.venv`, so Clarity's packages and source were importable behind the pinned ones. The pinned versions were the ones imported either way, so existing GGUFs are expected to be unchanged; a refetch would show any difference as a different sha256.

### Fixed
- In the terminal, a message that's too long for the context, or whose call fails, comes back in the input to edit or send again, as a message stopped before its reply does. An edit comes back as `/edit TEXT`, so Enter edits again.
- `/say` with no text refuses with "Type a message to send." instead of its usage line. `/state` gains `mode`; `/prompt` gains `prompt_attachments`, so Details lists a failed or edited message's attachments.
- In the terminal, a message that starts with "/" (typed as `//…`, `/say /…` or `/edit /…`) and is stopped before its reply starts comes back in the input still escaped, so sending it again sends the message instead of running a command.
- In the web interface, Send with an empty box says "Type a message to send.", and the chat's title and the profile picker, while they wait for a reply or a command, say why when pressed instead of telling only a screen reader.
- `clarity fetch` records the converter's package versions as the converter imports them. A package installed both in llama.cpp's pinned overlay and in the environment under it was recorded at the lower one's version, so `models.lock` said transformers 5.22.2 and numpy 2.5.4 where the converter ran transformers 4.57.6 and numpy 2.2.6. The conversion itself was right. A model already in `models.lock` keeps its old record until it's fetched again.

## [0.3.0] - 2026-11-15

Live tests on an M2 Max (macOS), 2026-11-15, at the release's code (built in the sprint to 1.0, so these were run by the build, not yet by the maintainer): `-m live` 19 passed, `-m hf` 18 passed, `-m olmo` 12 passed, `-m llama` 3 passed, and `-m browser` 86 passed (the web interface in Chrome).

### Migration
- For `clarity chat --json` readers: a `/chats` row's `conversation_id` is now the ID inside the log; the log file's name, which `/resume` and `clarity export` take, is the new `log_id`. They're the same for a log Clarity wrote that wasn't copied or renamed, so a script that passed `conversation_id` to `/resume` keeps working for those, and should pass `log_id`. `/state` gains `log_id`, `start` and `reply_end`, and `text` events gain `at` and `end`; readers that ignore unknown keys need no change.
- Exports are named by their log file: an export of a copied or renamed log is now written under its own name, beside any export written before under the conversation ID, which stays as it was.

### Added
- Llama 3.1 8B, Instruct and base (`llama-3.1-8b-instruct`, `llama-3.1-8b-base` in `models.toml`), with the profiles `llama-3.1-8b-chat` and `llama-3.1-8b-base`, which take Meta's recommended temperature and top_p (`use_recommended`). Meta gates the repos: accept the licence on each model's page and run `uv run hf auth login` before `clarity fetch`; Clarity never handles the token. The chat template starts every chat with a system header dated 26 Jul 2024, its fixed default, which `/prompt` shows. Live tests: `pytest -m llama`.
- `clarity ui`, a web interface on `127.0.0.1:8765` over the same commands as `--json`. It prints a link with a secret made for this start and opens it through a private file; the page reads the secret from the link's fragment, keeps its event stream and joins a reply after a reload. When the connection drops, the page says Clarity isn't answering and keeps trying until it answers. It ships:
  - The top bar and the Clarity menu (New chat, Help and docs, About Clarity, Quit Clarity, which stops a reply first); the picker, with every profile under its model; the row under the message box (the model and profile, "Settings · N changed", the context figure); a model starting, a refused start with the backend's sentence whole, and a new chat with its start warnings; and pages for a tab with no link, a second tab ("Use it here") and a stopped Clarity (Quit, or Ctrl-C in its terminal), with an unexpected error shown whole.
  - The thread: messages as plain text; Send, the reply as it's written with "Writing…" (or "Writing reply 2 of 3") and Stop (or "Stop the rest"); Edit on your last message and a counter to step through its versions; Copy, Regenerate and marks and notes on each reply; the facts line with every call flag; and a failed or refused message with the backend's sentence whole, its text back in the box, "Try again" after a failed Regenerate, and "Start the model again" when the model's server has stopped (a refused restart shows llama-server's output).
  - Details, for any reply or failed call: the exact prompt as text, with its whitespace shown, or as token IDs; how the reply stopped, how long it took and what ran; the checks, a failed one first; Replay with Stop and its result, with what changed since the call, what the log couldn't compare and the replay's own flags; the settings as logged; Probabilities, which shade the reply token by token and show each token's chance and alternatives, or say why there are none; and the whole call record. A call logged before Clarity recorded which text the template wrote shows its prompt unhighlighted and says so.
  - Branches: every branch end, newest first, with where it came from, and the replies you marked, each one click from being shown; "Branches N" in the top bar once a chat has two.
  - Settings: every setting with its value, where it came from and a line on what it does; sliders and boxes that send `/set`; one way back on each changed row, and Reset all; a steered profile's vector, hash, blocks and strength; and the controls waiting, with the reason, while a reply is written.
  - Chats and the title menu: every chat under Today and Earlier, the open one marked; picking one reopens it and says how many things differ today, with "Show what changed"; a chat that won't open says why under its row, and the open chat stays; Rename in place; and Export for a Python notebook, whose sheet says where the file is, what it holds and what to send with it, with Download, or why it wasn't written, with Try again.
  - Replies and Compare: every reply to a message with only what differs and where its branch went, "Show this reply", More replies, "Again with the same N seeds" after a setting changed, and Compare, which lines up replies with the same seed and marks where they part, or shows cards and warns when both the seed and a setting differ.
  - Appearance: five themes (Notebook, Neutral, Sky, Space, High contrast), light, dark or the system's, and a Motion switch for Sky and Space's scenery, off by default when the system asks for reduced motion; kept in the browser, not in a file. Keyboard: Enter sends, Shift+Enter starts a new line, toolbars and the replies' rows move with the arrow keys, and focus returns where it came from. Narrow windows keep every control.
  - Its fonts, Rasa and IBM Plex Sans and Mono, as their publishers' unmodified files under the SIL Open Font License, each family with its licence (`web_static/fonts/`). Nothing loads from the network.
- `/chats` and `/profiles` answer at once, even during a reply, and `/settings` and `/prompt` do while a reply is written or replayed, as `/state` and `/cancel` do; with `--json` they run as soon as they're read. `/state` also gives `log_id`, the name `/resume` takes, which differs from the conversation ID for a copied or renamed log.
- `/resume`'s `differ`: the item for facts changed since the last call also gives `call` and `changes` (the resume record's differences), so a reader can list each change.
- `text` events carry `at` and `end`, the piece's place in the reply, and `/state` gives `reply_end`, so a reader that starts mid-reply joins the stream without dropping or repeating text.

### Changed
- Every sentence that says to fetch or verify a model gives the command as it runs: "In a new terminal window, in Clarity's folder, run: uv run clarity fetch NAME. Then try again." (a lock entry from other settings, a missing file, no recommended settings, a tokenizer not in the cache, a changed weights file).
- `/chats` rows give `log_id`, the log file's name that `/resume` and `clarity export` take, and `openable`; their `conversation_id` is now the ID inside the log (it was the file's name, which differs for a copied or renamed log). `clarity chats` lists the file's name. `/resume` takes a name with dots after its first character, such as `chat.v2`, as export does.
- The base profiles' descriptions (`5-raw-base`, `llama-3.1-8b-base`) say "the base model, not tuned for chat" for "the untuned base model". Resuming an older chat on `5-raw-base` reports that the profile file has changed; the chat still uses the profile as logged. After a resume with nothing different, the terminal says "Nothing Clarity checks differs from when this chat was written."
- Four sentences read the same in the terminal and the page: a model that isn't fetched says "In a new terminal window, in Clarity's folder, run: uv run clarity fetch NAME. Then try again."; a message that doesn't fit ends "Start a new chat, or lower max_tokens."; "Stopped before the reply started." and "Replay stopped: nothing was compared or logged." say stopped, as the Stop button does. `clarity ui` adds "Keep this window open while you use Clarity."
- Exports are named by their log file (`logs/exports/ID.jsonl`, ID as `clarity chats` lists it), not by the conversation ID, which a copied log keeps, so a copy's export no longer replaces the original's. A log file whose name Clarity wouldn't write needs `clarity export --out`.
- The reference's checks after each call are a table that also says what each failure means, and covers `generation_config`, `steering` and the older `server_props`. Details in the web interface uses the same words.
- New dependencies for the web interface: `starlette` and `uvicorn`. Tests of the web interface in a browser use `pytest-playwright`, in its own dependency group (`uv sync --group browser`); a normal install doesn't get it.

### Fixed
- `clarity ui` prints its link at once when its output goes to a pipe or a file (`clarity ui | tee`, a script), not only when Clarity exits.
- A model file that's gone when its llama-server starts now says which file and how to get it back ("The model file PATH is missing. In a new terminal window, in Clarity's folder, run: uv run clarity fetch NAME. Then try again."), instead of an unexpected FileNotFoundError.
- `clarity fetch` of a gated repo you have no access to stops with what to do (accept the licence on the model's page, then `uv run hf auth login`), not a traceback.
- After a stopped server, the terminal's "/resume ID starts it again" names the log file, which `/resume` takes, not the conversation ID; they differ for a copied or renamed log.
- A failed start of the open chat's own model ("Start the model again", a `/replay` or `/new` on the same profile after its server stopped) no longer tries the same start a second time and then drops the chat: the chat stays, with its model stopped (`status: stopped`, whether the backend or the configuration refused), and the error is that sentence alone, so you can try again. A llama.cpp start that's refused before launching (llama.cpp moved or rebuilt) now leaves the model stopped, not ready with no server.
- A read (`/history`, `/branches`, `/compare`, `/prompt`, `/probs`, `/settings`) that meets a record it can't use is refused, and no longer ends the session with an unexpected error. The sentence gives the exception and says what can cause it: "Can't read this chat's records (KeyError: 'output_ids'). A log another program wrote or edited can cause this; if not, it's a bug in Clarity." `/resume` and export say the same, where they said "a record lacks or mangles …".
- A hand-edited `recommended` entry in `models.lock` that isn't a table of values is refused when the model starts, naming `models.lock` and the model, where `/settings` blamed the chat's log for it.
- A replay (`/replay`, `clarity replay`) of a call whose record lacks a field it needs (any of the ten it reads, such as the request, the reply's text or IDs, or the stop type) is refused before anything starts, naming it, where it ended the session with an unexpected error.
- In the web interface: a reload after an unexpected error shows the error again, not the plain stopped page; Enter while a reply is written or a Replay runs says it waits, beside Send, and keeps the text; a message, start, rename, reopened chat or restart whose request drops says so, and a dropped message gives Send back; a page that another tab has taken over, or whose secret is gone, no longer also says Clarity didn't answer; and the panel shows the Details you clicked, and stays closed once you close it, whatever redraw or view lands after.
- The web interface says a read's refusal without the terminal command's name ("Can't read this chat's records …", not "/probs: Can't read …"), as it already did for `/set`'s; the terminal keeps the name.

### Security
- A page left open after Clarity stops neither draws nor loads what another program on its port sends: `/state` gives `start`, a random value made for each start, and the page stops at any other.

## [0.2.3] - 2026-11-14

Live tests on an M2 Max (macOS), 2026-11-14, at the release's code (built in the sprint to 1.0, so these were run by the build, not yet by the maintainer): `-m live` 19 passed (including the backend contract and a llama-server killed between two calls), `-m hf` 18 passed, `-m olmo` 12 passed.

### Added
- The shipped profiles have descriptions. Because the profile's whole file is logged and compared, resuming an older chat on one of them reports that the profile file has changed; the chat still uses the profile as logged.
- A profile can have a one-line `description`. `clarity chats` shows it after the profile's name.
- `/prompt`, `/mark`, `/unmark`, `/note` and `/replay` can name any call in the conversation by its ID, for example `/note call-3f9a1b2c4d5e was better`; without one they act on the current reply, as before. A failed call can be read by its ID with `/prompt`, live and after a resume. `/replay` with anything else after it is now refused (it ignored its argument).
- `/probs` can name a call by its ID, and its `--json` result gives every token's text and place in the reply, with its alternatives' texts. Without probabilities, it adds a `reason` code beside the sentence.
- `/regen same` makes your last message's replies again with the same seeds, under the current settings: after `/set`, it shows what the change does with the randomness held fixed. A `/regen` after it uses the highest seed used so far plus 1, which is the same seed as before for every older log.
- `/compare [CALL_ID]` shows the replies to a message side by side as data: grouped by their settings, what's the same in all, and where two replies with the same seed but different settings part.
- `/branches all` lists every branch end in the conversation, newest first, with where it left the rest and whether it went on from there, and every marked reply with its note.
- `/state` says where things are, at any moment: the conversation, its turns, the command running and the reply so far, your last message if its call failed, and what's loaded. Like `/cancel`, it runs during another command; with `--json`, both run as soon as they're read.
- `/profiles` lists every profile you can start, with its model, backend and description, whether it's steered and whether its model is fetched. A profile that doesn't load is listed with the reason. It runs before any conversation, so a script or the web interface can offer a choice.

### Changed
- When llama-server has stopped between calls, the next call says so and sends nothing; it said "connection failed". A `status` event marks the model stopped, and resuming the conversation (`/resume ID`, which the terminal suggests) starts it again. Resuming the chat that's open writes a `resume` record, as any resume does.
- A reply whose record has no timings, such as a llama.cpp reply cancelled mid-stream, shows "time unknown" in its status line, not "0.0 s".
- Sentences you can meet outside the terminal no longer name commands: the context refusal, the control-token, template_kwargs and not-applied warnings, a failed profile switch, the replay refusals and the busy refusal. `/probs` words each reason it has no probabilities on its own: llama.cpp, switched off, or a stopped reply. Usage errors still name their command.
- `/prompt`'s header says whether the prompt starts with the start token (BOS), read from its token IDs, and whether Clarity added it. A chat template can write its own, as Llama 3.1's does, which "no BOS prepended" didn't show.
- `/export` names each vector the calls were steered with, and those calls (`steered`).
- `/settings` lists what the Transformers backend can't apply at the current values, so it's right before the first reply and straight after a `/set`; it showed the last call's list. A setting changed with `/set` also gives the profile's own value, and the result names the model (`model_name`).
- `/prompt` returns the call's replays, read from the log, so a replay's result shows again after a resume. `/resume` returns `differ`: how many things differ today and the sentences that say what, the same ones the terminal prints.
- The `status` event for a model starting names the profile, not its file's path.
- Call records returned by commands are the records as logged, with their `time` and `schema_version`, so `/prompt` gives the same answer before and after a resume.

### Output may change
- The default seed is 42, not `"random"`, so on the same machine and build the same message, history and settings get the same reply. A profile that sets no seed now uses 42; `seed = "random"` still draws one per call. A batch run's seeds start at 42, not 333. Resuming a chat that drew random seeds by default shows the seed among the settings that differ today.

## [0.2.2] - 2026-11-11

Live tests on an M2 Max (macOS), 2026-11-11, at the release commit: `-m live` 17 passed, `-m hf` 17 passed (Transformers refuses MPS with `PYTORCH_ENABLE_MPS_FALLBACK` set, on the real device; steering SmolLM2 with a real repeng vector on the CPU and MPS: a zero vector changes nothing, the vector changes the reply, a cancel leaves the model unsteered, a replay after a reload is identical, and a steered export recomputes with a gap of 0 under `applied`), `-m olmo` 12 passed (the same steering checks on OLMo 3 7B Instruct on MPS, including a reply that ends on its end token).

### Added
- Steering on the Transformers backend: a profile's `[steering]` table names a vector file made by another tool, pinned by its sha256, and `steering_scale` in `[settings]` sets how strongly it's added. `/set steering_scale` changes it between replies without reloading. The vector is checked against the model before the first message. A llama.cpp profile with `[steering]` is refused for now.
- Each steered call records its vector's file, sha256, scale and blocks in `request.steering`, and checks that every forward pass ran each steered block once (`checks.steering`). Replay applies the same vector, and refuses if the file is gone or changed. Clarity before v0.2.2 ignores `request.steering`, so it replays a steered call unsteered.
- `/settings` shows a steered profile's vector (file, sha256 and blocks), and the status line after a steered reply ends with `steered ×S`. The `--json` `settings` result carries `steering`.
- The export records each call's steering, or `"none"`, and steering joins its conditions, so a change of scale starts a new group. The recompute check compares the export's steering with what `clarity_chat.steering.applied` has added to the model.
- `clarity_chat.steering` reads a steering vector made by another tool, such as repeng (llama.cpp's control-vector GGUF), checks it against its pinned sha256 and the model, and applies it to a loaded Transformers model in a notebook: `load`, `check` and `applied`. Clarity never makes vectors.

### Fixed
- `clarity chat --json` read its input one byte at a time, about half a second of CPU per megabyte of commands. It now reads whatever has arrived, up to 64 KB at a time.
- A setting in a log holding an integer too large for a float (a hand-edited log, say) made resume and replay stop with an `OverflowError`. It's now refused like any out-of-range value.

### Changed
- numpy is now a declared dependency (it was already installed with transformers): steering reads and applies vectors with it.

## [0.2.1] - 2026-11-11

Live tests on an M2 Max (macOS), 2026-11-11: `-m live` 17 passed (including a quantized copy that repeats and replays, and converting twice to the same file), `-m hf` 13 passed (greedy output identical on llama.cpp and Transformers; logged probabilities equal a step-by-step recompute on the CPU and MPS, and the recompute check gives a gap of 0 with the same model), `-m olmo` 11 passed (OLMo 3's logged probabilities equal the recompute check's own loop on MPS). All three again after the last replay fix, with the same results.

### Security
- A model folder whose config names its own Python code made Transformers ask on the terminal whether to run it, and a "y" typed in the chat would have. Clarity now refuses such a model with a clear error and never runs a model's own code. With `--json`, the question also broke the event stream.
- Resuming a log whose profile path or name leads outside the project folder no longer reads that file; the resume says so instead. A log is untrusted input.
- A profile's `{ file = "PATH" }` could name any file on your computer, which then went into the prompt and the log. A profile from someone else could use that to put, say, a private key into a log you might share. Prompt files must now be inside the project folder.
- Logs, llama-server's output logs and a batch run's files are now created readable only by you on Linux and macOS (mode 0600). They hold private conversations and were readable by other accounts on the machine. Files from earlier versions keep their mode; `chmod 600` changes them.
- `clarity replay --logged-flags` passed a log's flags to llama-server unchecked, and Clarity's own `--host 127.0.0.1`, `--no-slots` and `--no-webui` came before them, where a later flag wins. A shared or hand-edited log could make the replayed server listen on the network, re-enable `/slots` and the web UI, add a key, or turn on its shell tool. Logged flags now get every check `server.toml`'s get, and Clarity's own flags come last.

### Added
- `clarity_chat.recompute.check(export, model)`, for a notebook: lists each condition that differs between an export and the loaded model, and, for replies logged with `logprobs`, recomputes their probabilities step by step and reports the largest gap and how many top choices changed.
- `clarity export ID` and `/export` write a conversation to `logs/exports/ID.jsonl` for a notebook: a header with the conditions its calls were computed under, then one line per reply with exact token IDs, message spans (worked out and checked for older logs), marks, notes, settings and facts.
- Each call records `message_spans`: where each message, and the text the chat template or profile wrote around it, sits in the prompt, in characters and in tokens. A template that rewrites a message gets none for that call, with the reason.
- Token probabilities on the Transformers backend: the `logprobs` setting (0 to 20, off by default) records each reply token's log-probability and its N likeliest alternatives, from the model's raw scores before any sampler. `/probs` summarizes the current reply's least likely tokens. A llama.cpp profile refuses the setting until v0.4.0.
- `/regen N` makes N more replies to your last message, each a new branch with its own logged seed. With `--json`, a `reply_start` event (`k`, `n`, `seed`) comes before each.

### Fixed
- Replay and resume counted a machine value that one side couldn't read (the chip's name, which a sandbox hides) or didn't record (a newer key) as a difference, so an output that differed for no recorded reason wasn't flagged `unexplained_difference`. Such values are now listed under `not_recorded`.
- With `--json`, Clarity aborted when it quit while its input was still open, as a script's usually is: "Fatal Python error … daemon threads" and exit status -6 (SIGABRT) instead of 0 or 1. It now exits with its own status.
- Resuming a hand-edited log whose turns' parents formed a loop could hang. Resume and export now refuse a log whose turn names a parent that isn't an earlier turn, and name that turn.
- With `--json`, an unexpected error in a command (a bug, or a full disk) ended the session with a traceback and no `error` event. It now sends an `error` event naming the command and the error, quits, and exits with status 1. A line nested too deeply to parse stopped Clarity reading input; it's now an `InputError` like any other line that isn't a command.
- Replay and resume of a call whose log lacks its `server_start` record (a trimmed or hand-edited log) compared the wrong backend's facts and reported empty ones as differences, which could hide an unexplained difference. The call's own backend now decides, and what the log lacks is listed as not recorded.
- A Transformers replay judged its stop type, and recorded `eos_token_id`, from the end tokens in today's model files rather than the logged ones it was generated with. A model revision that changed only `generation_config.json` made an identical replay report `stop_type: none`.
- `/say` in the terminal sent its text with the spaces at its start and end removed, while a plain line and `--json` sent them; now all three send the text exactly. Through `--json`, spaces around another command's arguments made `/set` name an empty setting and `/mark` store a label `/unmark` couldn't find; they're now ignored, as in the terminal.
- If llama-server had to restart for `/replay` (it had stopped, or its model file was fetched again) and the restart failed, the next message crashed the chat. Now the conversation's model starts again, or the chat says to start a new conversation.
- In the terminal, Ctrl-C during a call cancels it wherever the call is. Before, it cancelled only while the reply streamed; at any other moment (building the prompt, the checks after the reply, computing token probabilities) it quit the chat, and the reply on screen could be missing from the log. During a command that can't be cancelled, such as a model start, Ctrl-C still stops Clarity.
- The Transformers backend refused models whose `generation_config.json` lists no end token (OLMo 3 base) or that ship none (Qwen3.5-9B). It now takes the tokenizer's end token, says so at start, and logs where the IDs came from (`server_start.eos_token_id_from`); `/settings` shows it.
- A model fetched again while its llama-server was running kept the old server, so calls ran the old file while the record named the new one. A server is now kept only if its file is unchanged (same inode, size and modification time); otherwise it restarts.
- `server.toml` now accepts only a fixed list of flags, the ones it ships with, and refuses any other. Before, it refused a list of known-bad flags, so flags such as `gpu-layers` (another name for `n-gpu-layers`), `grp-attn-n`, `override-tensor`, `special` and `docker-repo` were passed to llama-server.
- On Windows, `clarity chats` or `clarity replay` with output sent to a file or a pipe crashed on a title or reply the system's code page can't hold, such as Japanese or an emoji. Redirected output is now UTF-8 on every system.
- A v0.1.0 log that was resumed and continued still said its head was a guess at every later resume. It says so now only until a newer Clarity logs a message or a head move in it.
- If llama-server's startup log had no end-of-reply line (a llama.cpp update could change it), `model_file.eog_tokens` recorded an empty list, which reads as "no end tokens". It's now null, and `/settings` says the tokens weren't found.
- After a resume, each call recorded the model's `recommended` values from today's `models.lock`, while its settings came from the log. If a later fetch had changed them, a setting with origin `model` didn't match the `recommended` beside it. A resumed conversation's calls now record the logged values.
- A batch run (`clarity run`) logged a call's flags, such as `incomplete_output_ids`, but didn't show them, so a flagged sample went unnoticed unless someone read `calls.jsonl`. Each flagged call is now named on screen with its flags (never its reply).
- `server.toml` accepted `false` for `kv-unified`, `no-kv-unified` and `no-jinja`, which left the flag out so llama-server's default applied. It also accepted `threads` or `threads-batch` of 0 or less, which llama-server reads as every core. Both are now refused.
- `server.toml` now refuses every adapter and control-vector flag. Before, only `lora` and `control-vector` were refused, and `lora-scaled`, `lora-init-without-apply`, `control-vector-scaled` and `control-vector-layer-range` were passed to llama-server. A reply made with one of them has the flag in its `server_start` record.
- A replay flagged `text_differs_from_ids` now logs `decoded_output_text`, as the call does.

### Changed
- The `2-system` and `3-primed` profiles and their placeholder files, `prompts/system.txt` and `prompts/priming.toml`, are gone. They only held placeholder text. Put your own system prompt or priming in a profile in `local/profiles/`; `docs/guides.md` shows both.
- `clarity chat --json` with a `--profile` or `--resume` that fails now sends the error and quits with exit status 1, as the terminal does. It used to carry on with no conversation and exit 0.
- A llama-server built from a commit other than the one `server.toml` pins is now an error at start, not a warning. The binary is what runs, and a checkout at another commit was already an error.
- After each call, Clarity checks that the llama-server process it started is still running, in place of a second `/props` request. `/props` is still checked once after start; nothing can change a running server's model or context.
- Replay and resume compare one more fact, `end_tokens`: the IDs a reply stops on. The backends can differ here (llama.cpp also stops on tokens it recognizes by name), and the difference is now shown rather than hidden. Logs from v0.1.0 made with llama.cpp list it under `not_recorded`; v0.2.0 llama.cpp logs and every Transformers log already hold it.
- `/history`, `/mark`, `/unmark`, `/note`, `clarity mark` and `clarity note` read only the records they need, not every call record, which made them 3 to 5 times faster on long logs in a measurement on made-up logs of 45 to 225 MB. A damaged line of another type no longer stops them; resume and replay still refuse it.
- Docs: the README, the design doc and the citation abstract now state the limits of their claims: replay covers completed calls, the same-output tests ran on two model families on one Apple Silicon Mac, and "No silent fallbacks" names its two flagged cases. The README, the package description and the citation title now share one tagline.

### Migration
- In a chat and through `--json`, `/resume` takes a conversation ID and `/profile` a profile name; a path is refused. Commands will also come from the browser in v0.3.0, which mustn't make Clarity read any `.toml` file or append to any log. `clarity chat --resume PATH` and `--profile PATH`, which you type, still take paths.
- With `--json`, the `regen` command's result is now `{"replies": [{"call", "warnings"}, …]}`, also for a plain `/regen`, in place of one `call` and `warnings`. Read `replies[0]` where a script read the result itself.
- The check after each llama.cpp call is renamed: `checks.server_props` is now `checks.server_running` (with `exit_code` when the process has exited). `checks.not_echoed` is gone: it could never fail, and the same list follows from the logged request and `generation_settings`. A reader that looks for either key finds it missing in new logs.

## [0.2.0] - 2026-11-10

Live tests on an M2 Max (macOS), 2026-11-10: `-m live` 15 passed (including crash cleanup against a real llama-server), `-m olmo` 7 passed, `-m hf` 7 passed (greedy output identical on llama.cpp and Transformers), `-m embed` 5 passed: bge-small's embeddings were bit-identical back to back, after a restart, whatever else the request held, and on the CPU at a fixed thread count (and between 8 and 4 threads, 0 of 25 vectors differed).

### Security
- llama-server now needs a random key, new at each start, so a web page in your browser can't send it requests. Its `/slots` endpoint and built-in web UI are off. The key is never logged.

### Added
- The tokens that end a reply, as llama.cpp read them from the model file, are logged with each server start and shown in `/settings`.
- Each server start logs how many pieces llama.cpp split the compute graph into, and replay compares it. A change between a call and its replay can mean work moved between devices.
- The status line after each reply shows how much of the context the conversation now uses, such as "context 12,400 of 32,768".
- A `template_kwargs` key that the chat template never reads, such as a misspelt one, is now a warning when the conversation starts, and is logged. Before, it changed nothing silently.
- `/say TEXT` and `/cancel` chat commands, for scripts. All chat commands now come from one table, which `/help` lists.
- If Clarity is killed or crashes, the next start stops the llama-server it left running, after checking that the server's port answers with its key and model. On Linux and macOS, SIGTERM and a closed terminal now stop the server too.
- `clarity chat --json` runs every chat command from a script or another program: commands in as JSON lines, and events out, including every log record as it's written. `cancel` works while a reply is being written.
- `clarity chat --resume ID` and `/resume ID` continue a logged conversation with its profile, settings and context size as logged, and say what differs today, including whether the current prompt still renders to the same token IDs.
- `clarity chats` and `/chats` list your conversations, newest first, with a title: the first words of the first message, or one you set with `/title`. No model is asked for a title.
- `/set NAME VALUE` changes a setting for one chat, checked like a profile's value and logged with origin `chat`. `/settings` shows it.
- `/history` shows the conversation so far on the current branch, with each reply's marks and notes, so a script or the web interface can draw it without reading the log.
- `/branches` lists where a conversation branches, and `/switch TURN_ID` goes to another branch. Every move of the current turn other than a new reply (undo, switch, a failed call) is logged as a `head` record.
- `/unmark [LABEL]` and `clarity mark --remove` take a mark back. A new note on a reply replaces its note, and Clarity says so. `/mark` and `clarity mark` without a label mark the reply `interesting`. Marks and notes now have IDs; the log keeps every one and records removals and replacements.
- A log line cut off by a crash or a full disk no longer makes the log unreadable. The next write ends it and adds a `torn_line` record naming it, and Clarity says so; the cut line is kept as it was.
- Only one `chat`, `run` or `replay` runs in a project at a time; a second one stops and says which process is running.
- Your own profiles go in `local/profiles/`, which git ignores, next to the shipped ones in `profiles/`. A name in both folders is an error.
- On a Mac, a context size whose memory doesn't fit on the GPU is refused, with the figures. Every server start logs its memory use.
- `ctx_size` in a profile sets its context size; profiles without it keep `ctx-size` from `server.toml`. Replay runs a call at the size it was logged with, and a size above the model's training context is refused instead of quietly capped.
- A call can be cancelled from any thread, which the web interface and `clarity chat --json` will use. Each cancelled call logs `cancelled_during`: `prompt` or `reply`.
- `clarity mark LOG CALL_ID LABEL` and `clarity note LOG CALL_ID TEXT` label or annotate any logged reply after the chat, without starting a model. Marks and notes now record whether they were made live or in review.
- The Transformers backend sets `attn_implementation = "sdpa"` explicitly, checks it after loading, and logs it.
- `uv run pytest -m embed` tests whether llama-server's embeddings of bge-small, a new entry in `models.toml`, come back bit-identical.
- User docs: getting started, guides, reference and contributing.
- `CHANGELOG.md`, `SECURITY.md`, `CONTRIBUTORS.md`, `CODE_OF_CONDUCT.md` and `ROADMAP.md`; a release checklist and a Credit section in `CONTRIBUTING.md`.

### Changed
- A call cancelled before its first token no longer adds an empty reply. It's logged like a failed call, and your message comes back in the input.

### Migration
- `server.toml` can no longer set `api-key-file`, `slots`, `ui`, `webui` or their `no-` forms, or any `ui-` or `webui-` flag: Clarity sets the server key, `--no-slots` and `--no-webui` itself. A file that sets one stops with an error naming it.
- Logs keep `schema_version` 1. They gain record types (`head`, `unmark`, `settings`, `title`, `resume`, `torn_line`) and fields, which readers that ignore unknown records and keys skip. Clarity 0.1.0 reading a 0.2.0 log shows removed marks as still there and every note, including replaced ones.
- A profile name in both `profiles/` and `local/profiles/` is an error. Move your own profiles to `local/profiles/` so a `git pull` never touches them.
- No change in this release is known to change a reply: the new server flags don't affect sampling, and replay compares the server's flags, not its command line. Replay lists any difference it finds.

### Fixed
- On CUDA and Vulkan, a model whose weights went only to pinned system memory (`CUDA_Host`, `Vulkan_Host`) passed the GPU placement check. It's now an error, like any model that lands on the CPU.

## [0.1.0] - 2026-11-09

The first release.

### Added
- Chat mode, through the model's chat template, and text mode, where the model continues a plain document. Clarity renders the prompt and builds the token IDs itself.
- Every sampling value is sent on every call, and the server's echo is checked.
- Branches in a chat (`/regen`, `/edit`, `/undo`), and marks and notes on replies.
- An append-only JSONL log per conversation, and replay of any logged call with a list of what changed since.
- Blind batch runs for comparing profiles.
- Models at a pinned commit: converted from the official weights, or a ready-made GGUF checked by sha256.
- A llama-server that Clarity starts and checks itself, and a Transformers backend for the same weights.

[Unreleased]: https://github.com/andalibmalit/clarity-chat/compare/v0.3.1...HEAD
[0.3.1]: https://github.com/andalibmalit/clarity-chat/compare/v0.3.0...v0.3.1
[0.3.0]: https://github.com/andalibmalit/clarity-chat/compare/v0.2.3...v0.3.0
[0.2.3]: https://github.com/andalibmalit/clarity-chat/compare/v0.2.2...v0.2.3
[0.2.2]: https://github.com/andalibmalit/clarity-chat/compare/v0.2.1...v0.2.2
[0.2.1]: https://github.com/andalibmalit/clarity-chat/compare/v0.2.0...v0.2.1
[0.2.0]: https://github.com/andalibmalit/clarity-chat/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/andalibmalit/clarity-chat/releases/tag/v0.1.0
