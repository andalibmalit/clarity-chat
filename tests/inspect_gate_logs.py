"""Inspect logs shaped like v0.4.3's plan, built with inspect_ai's own types (the gate, G0).

Not Clarity's writer, which doesn't exist yet: these logs are what the plan says the writer
will write, so the gate's tests (`test_inspect_gate.py`, offline; `test_browser_inspect.py`,
the viewer) can show what Inspect does with that shape before the writer is built. Every
text is made up.

`write_gate_logs(root)` writes:
- `run/1-smol-chat.eval` and `run/2-smol-text.eval`: one batch run, two profiles (chat mode
  and text mode), two openings with two epochs (seeds) each, the same task in both. Every
  reply has a `mark` score, "unmarked" when it has no mark (the owner's Q2, (a)).
- `marked-only/1-smol-chat.eval`: the chat-mode log with Q2's other form, a `mark` score only
  on a marked reply.
- `chat/c0ffee.eval`: a chat-shaped log with uneven epochs (three replies to one message, one
  to its edit).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

NOTE = (
    "Clarity made these calls, not Inspect. Clarity built each prompt's token IDs itself "
    "(chat template, BOS, special tokens) and sent them with its own sampler settings. This "
    "log's generate config is empty: the calls' settings can differ (each of a run's epochs "
    "has its own seed), so no one config is true of them all, and Inspect's config can't "
    "express some of Clarity's settings (min_p, the repeat penalty and its window, the "
    "sampler order). Running this log again in "
    "Inspect won't reproduce these calls: its model, clarity/PROFILE, names no Inspect "
    "provider. Each sample's metadata holds what the model got (`prompt_ids`, "
    "`prompt_text`), every setting with its origin, and the facts replay compares. Each "
    "message's text is that message's part of `prompt_text`. Text the chat template adds is "
    "only in `prompt_text`: its role markers, and any text of its own, like the system "
    "message OLMo 3's template adds when the profile has none, or the dates Llama 3.1's puts "
    "in its system header. This log asks Inspect's viewer to show its text as written, not "
    "as Markdown, and hidden characters as markers like \u27e8U+200B\u27e9; the metadata "
    "(`prompt_text`, `output_text`) holds the exact characters. "
    "The `mark` score's value is the reply's labels (\"unmarked\" when it has none); it is "
    "a label, not a measurement."
)
TAGS = ["clarity", "not-run-by-inspect"]
RUN_ID = "r7f3a9c2e"
CREATED = "2026-11-20T10:00:00+00:00"
OPENINGS = {1: "Name three colours of the sea.", 2: "Describe a kettle in one sentence."}
SYSTEM = "You answer in one short line."

# Replies, by (profile, opening, epoch). Two carry what the viewer's checks look for:
# Markdown that could be dropped (V11), and HTML that must stay text (V9).
MARKDOWN_REPLY = "   Indented by three spaces.\n* a starred line\n<think>a thought</think> after it"
REMOTE_IMAGE = "http://198.51.100.7/gate.png"  # TEST-NET-2: never a real host
HTML_REPLY = f'<script>window.gateRan = 1</script><img src="{REMOTE_IMAGE}"> plain after'
REPLIES = {
    ("smol-chat", 1, 1): MARKDOWN_REPLY,
    ("smol-chat", 1, 2): "Blue, green and grey.",
    ("smol-chat", 2, 1): "A kettle boils water for tea.",
    ("smol-chat", 2, 2): "It whistles when the water boils.",
    ("smol-text", 1, 1): " Blue, teal and slate.",
    ("smol-text", 1, 2): " Navy, green and silver.",
    ("smol-text", 2, 1): " A metal pot with a spout.",
    ("smol-text", 2, 2): HTML_REPLY,
}
# Marks and notes, by (profile, opening, epoch); every other reply has neither.
MARKS = {("smol-chat", 1, 1): ["landed"], ("smol-text", 1, 2): ["too long", "landed"]}
NOTES = {("smol-chat", 2, 1): ["reads well", "but plain"]}
LONG_PROMPT = ("smol-chat", 2, 2)  # its prompt_ids hold 4,000 tokens (V2)


def mark_value(marks: list[str]) -> str:
    return ", ".join(sorted(marks)) if marks else "unmarked"


def chat_prompt(opening: str) -> tuple[str, list[dict[str, Any]]]:
    """A chat template's text for a system message and one user message, and the spans."""
    parts = ["<|im_start|>system\n", SYSTEM, "<|im_end|>\n<|im_start|>user\n", opening]
    parts.append("<|im_end|>\n<|im_start|>assistant\n")
    text, spans, at = "", [], 0
    for i, part in enumerate(parts):
        if i in (1, 3):
            role = "system" if i == 1 else "user"
            spans.append({"kind": "message", "role": role, "chars": [at, at + len(part)]})
        text += part
        at += len(part)
    return text, spans


def row(profile: str, opening: int, epoch: int, mode: str) -> dict[str, Any]:
    """The sample's metadata: the plain export's row for the call, flat, plus `run` (S2)."""
    key = (profile, opening, epoch)
    if mode == "chat":
        prompt_text, spans = chat_prompt(OPENINGS[opening])
    else:
        prompt_text = f"Q: {OPENINGS[opening]}\nA:"
        spans = [{"kind": "message", "role": "user", "chars": [3, 3 + len(OPENINGS[opening])]}]
    length = 4000 if key == LONG_PROMPT else 24
    return {
        "call_id": f"k{profile[-4:]}{opening}{epoch}",
        "conversation_id": f"v{profile[-4:]}{opening}",
        "stop_type": "eos",
        "cancelled": False,
        "flags": [],
        "prompt_ids": [(7 * i + opening) % 49152 for i in range(length)],
        "prompt_text": prompt_text,
        "message_spans": spans,
        "output_ids": [11, 12, 13],
        "output_text": REPLIES[key],
        "settings": {"seed": {"value": 41 + epoch, "origin": "run"}},
        "marks": [{"label": m} for m in MARKS.get(key, [])],
        "notes": [{"text": n} for n in NOTES.get(key, [])],
        "run": {"run_id": RUN_ID, "opening": opening},
    }


def sample(profile: str, opening: int, epoch: int, mode: str, every_reply_scored: bool):
    from inspect_ai.log import EvalSample
    from inspect_ai.model import (
        ChatCompletionChoice,
        ChatMessageAssistant,
        ChatMessageSystem,
        ChatMessageUser,
        ModelOutput,
    )
    from inspect_ai.scorer import Score

    key, model = (profile, opening, epoch), f"clarity/{profile}"
    meta = row(profile, opening, epoch, mode)
    text, call_id = meta["prompt_text"], meta["call_id"]
    if mode == "chat":
        messages: list[Any] = []
        for i, span in enumerate(meta["message_spans"]):
            kind = ChatMessageSystem if span["role"] == "system" else ChatMessageUser
            start, end = span["chars"]
            messages.append(
                kind(id=f"{call_id}-{i}", content=text[start:end], metadata={"clarity": span})
            )
    else:
        whole = {"clarity": {"whole_prompt": "text mode: the template has no message roles"}}
        messages = [ChatMessageUser(id=f"{call_id}-0", content=text, metadata=whole)]
    reply = ChatMessageAssistant(
        id=f"{call_id}-{len(messages)}", content=meta["output_text"], model=model
    )
    marks, notes = MARKS.get(key, []), NOTES.get(key, [])
    scores = {}
    if every_reply_scored or marks:
        scores["mark"] = Score(
            value=mark_value(marks),
            answer=meta["output_text"],
            explanation="\n\n".join(notes) or None,
            metadata={"marks": meta["marks"], "notes": meta["notes"]},
        )
    return EvalSample(
        id=opening,
        epoch=epoch,
        input=OPENINGS[opening],
        target="",
        messages=[*messages, reply],
        output=ModelOutput(
            model=model, choices=[ChatCompletionChoice(message=reply, stop_reason="stop")]
        ),
        scores=scores,
        metadata=meta,
        uuid=f"u{call_id}",
    )


def eval_log(task: str, run_id: str, model: str, samples: list, epochs: int, header: dict):
    from inspect_ai.log import EvalConfig, EvalDataset, EvalLog, EvalResults, EvalSpec
    from inspect_ai.model import GenerateConfig

    ids = list(dict.fromkeys(s.id for s in samples))
    return EvalLog(
        status="success",
        eval=EvalSpec(
            eval_id=f"e{run_id}{model[-4:]}",
            run_id=run_id,
            created=CREATED,
            task=task,
            task_id=f"t{run_id}{model[-4:]}",
            tags=TAGS,
            dataset=EvalDataset(name=task, samples=len(ids), sample_ids=ids),
            model=model,
            model_generate_config=GenerateConfig(),
            config=EvalConfig(epochs=epochs),
            metadata={"clarity": {**header, "inspect_format_version": 1, "inspect_note": NOTE}},
        ),
        samples=samples,
        results=EvalResults(total_samples=len(samples), completed_samples=len(samples), scores=[]),
    )


def run_log(profile: str, mode: str, every_reply_scored: bool = True):
    samples = [
        sample(profile, opening, epoch, mode, every_reply_scored)
        for opening in OPENINGS
        for epoch in (1, 2)
    ]
    header = {"run": {"kind": "run", "run_id": RUN_ID, "profiles": ["smol-chat", "smol-text"]}}
    return eval_log("clarity-run", RUN_ID, f"clarity/{profile}", samples, 2, header)


def chat_log():
    """A chat: message t1 answered three times (two /regens), then edited as t4, answered
    once. Each reply is an epoch of its message's sample, so the epochs are uneven."""
    from inspect_ai.log import EvalSample
    from inspect_ai.model import ChatMessageAssistant, ChatMessageUser, ModelOutput
    from inspect_ai.scorer import Score

    samples = []
    replies = {"t1": ["Grey.", "Blue.", "Green."], "t4": ["Teal and navy."]}
    message = {"t1": "One colour of the sea?", "t4": "Two colours of the sea?"}
    for turn, texts in replies.items():
        for epoch, text in enumerate(texts, start=1):
            call_id = f"c{turn}{epoch}"
            meta = {"call_id": call_id, "stop_type": "eos", "output_text": text}
            reply = ChatMessageAssistant(id=f"{call_id}-1", content=text, model="clarity/x")
            samples.append(
                EvalSample(
                    id=turn,
                    epoch=epoch,
                    input=message[turn],
                    target="",
                    messages=[ChatMessageUser(id=f"{call_id}-0", content=message[turn]), reply],
                    output=ModelOutput.from_content("clarity/smol-chat", text),
                    scores={"mark": Score(value="unmarked", answer=text)},
                    metadata=meta,
                    uuid=f"u{call_id}",
                )
            )
    header = {"conversation_id": "c0ffee"}
    return eval_log("clarity-chat", "c0ffee", "clarity/smol-chat", samples, 3, header)


def write_gate_logs(root: Path) -> dict[str, Path]:
    from inspect_ai.log import write_eval_log

    paths = {
        "chat-mode": root / "run" / "1-smol-chat.eval",
        "text-mode": root / "run" / "2-smol-text.eval",
        "marked-only": root / "marked-only" / "1-smol-chat.eval",
        "chat": root / "chat" / "c0ffee.eval",
    }
    logs = {
        "chat-mode": run_log("smol-chat", "chat"),
        "text-mode": run_log("smol-text", "text"),
        "marked-only": run_log("smol-chat", "chat", every_reply_scored=False),
        "chat": chat_log(),
    }
    for name, path in paths.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        write_eval_log(logs[name], str(path))
    return paths
