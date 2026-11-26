# A prompt for writing pairs

`clarity vector make` turns a file of contrasting pairs into a steering vector. This page holds a prompt that asks a capable model to write that file for a behaviour you describe. The model writes the pairs; Clarity only reads the file. See [Make a steering vector](guides.md#make-a-steering-vector) for what to do with it.

## Two ways to use it

- **With a coding agent in Clarity's folder** (Claude Code, Codex, or another): ask it to follow this page for your description and save the file in `local/pairs/`. Then run `uv run clarity vector make local/pairs/NAME.toml --profile PROFILE`.
- **With any chat app:** copy the prompt below, replace `{DESCRIPTION}` with your description, and save the answer as a `.toml` file in `local/pairs/`. Then run the same command.

Read the file before you make a vector from it. The vector carries every way the two sides differ in the same direction across pairs, so a word on one side of every pair becomes part of it.

## The prompt

```
Write a pairs file for making a steering vector. The behaviour to steer toward:

{DESCRIPTION}

Answer with the file only: TOML, no other text, no code fence. Use exactly this shape:

description = "<one line, in your own words, saying what the positive side does>"
author = "<your own name or model name, as you know it>, from docs/pairs-prompt.md"
messages = [<10 user messages>]
openings = ["", <3 more reply openings>]

[[systems]]
positive = "<a system message asking for the behaviour>"
negative = "<the same system message asking for its opposite>"

(6 [[systems]] tables in all.)

Rules for the system pairs:
- In each pair, the positive side asks for the behaviour and the negative side for its
  opposite. The two sides match in everything else: who the assistant is, length,
  sentence shape, and every word that isn't about the behaviour.
- Vary who the assistant is across the 6 pairs ("an assistant", "a librarian", "a tutor",
  "a guide", and so on), and keep it the same on both sides of a pair.
- No names of people, products or places, and no special terms.
- No praise, politeness or warmth on one side only, unless that is the behaviour itself.

Rules for the messages: 10 ordinary, varied, short user messages about everyday topics.
None of them is about the behaviour, and none asks for it or its opposite.

Rules for the openings: "" and 3 short, neutral starts of a reply, one to three words
each, that fit either side (for example "Well,", "So,", "Okay,"). No evaluation, such as
"That's a great question!".

The file makes 6 x 10 x 4 = 240 pairs. Write nothing outside the file.
```

## An example

A small file in the same shape, about brevity: 2 system pairs, 2 messages and 2 openings, so 8 pairs. A real file from the prompt has 240.

```toml
description = "Short, plain replies against long, ornate ones."
author = "an example, from docs/pairs-prompt.md"
messages = ["How do I boil an egg?", "What should I pack for a day hike?"]
openings = ["", "Well,"]

[[systems]]
positive = "You are an assistant. You reply in as few plain words as you can."
negative = "You are an assistant. You reply at length, in rich and decorated language."

[[systems]]
positive = "You are a librarian. You keep every answer short and plain."
negative = "You are a librarian. You make every answer long and elaborate."
```

The `author` field records what the file says about who wrote it. Clarity can't check it, and the vector's record says so.
