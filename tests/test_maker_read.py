"""V3: reading each block's output and the numbers per block, on tiny bf16 Llama and OLMo 3
models on the CPU, as tests/test_steering.py builds them."""

from __future__ import annotations

import numpy as np
import pytest

from clarity_chat.maker import Rendered, read_blocks, reading
from clarity_chat.profiles import ConfigError
from tests.test_steering import tiny

# Four made-up pairs of token IDs (the vocabulary is 64): each side differs from the other.
SIDES = [
    ([1, 5, 9, 13, 2], [1, 5, 9, 14, 2]),
    ([1, 7, 3], [1, 8, 3, 3]),
    ([1, 20, 21, 22], [1, 30, 21, 22]),
    ([1, 40, 41], [1, 40, 42]),
]


def prompts(sides=SIDES) -> list[Rendered]:
    return [
        Rendered(i, side, "", ids)
        for i, pair in enumerate(sides, 1)
        for side, ids in zip(("positive", "negative"), pair, strict=True)
    ]


@pytest.fixture(params=["llama", "olmo3"])
def model(request):
    return tiny(request.param)


def hidden(model, torch, ids):
    with torch.no_grad():
        out = model(torch.tensor([ids]), output_hidden_states=True)
    return [h[0, -1].float().numpy() for h in out.hidden_states]


def test_the_hook_reads_each_blocks_output_before_the_final_norm(model):
    model, torch = model
    n = len(model.model.layers)
    states = hidden(model, torch, SIDES[0][0])
    with reading(model) as rows, torch.no_grad():
        model.model(input_ids=torch.tensor([SIDES[0][0]]), use_cache=False)
    assert sorted(rows) == list(range(1, n))
    for b in range(1, n - 1):
        assert rows[b].dtype == np.float32
        np.testing.assert_array_equal(rows[b], states[b + 1])
    # The last entry of hidden_states comes after the final norm: the reason for the hook.
    assert not np.array_equal(rows[n - 1], states[n])


def test_the_numbers_per_block(model):
    model, torch = model
    n = len(model.model.layers)
    blocks = read_blocks(model, prompts(), lambda k, total: None)
    assert sorted(blocks) == list(range(1, n))
    states = [(hidden(model, torch, p), hidden(model, torch, q)) for p, q in SIDES]
    for b in range(1, n - 1):
        pos = [s[0][b + 1].astype(np.float64) for s in states]
        neg = [s[1][b + 1].astype(np.float64) for s in states]
        diffs = [p - q for p, q in zip(pos, neg, strict=True)]
        mean = np.mean(diffs, axis=0)
        block = blocks[b]
        assert block.direction.dtype == np.float32
        np.testing.assert_allclose(block.direction, mean.astype(np.float32), rtol=1e-6)
        cosines = [d @ mean / (np.linalg.norm(d) * np.linalg.norm(mean)) for d in diffs]
        assert abs(block.agreement - np.mean(cosines)) < 1e-12
        assert abs(block.gap - np.linalg.norm(mean)) < 1e-9 * np.linalg.norm(mean)
        sizes = [np.linalg.norm(r) for r in pos + neg]
        assert abs(block.size - np.mean(sizes)) < 1e-9 * np.mean(sizes)


@pytest.mark.parametrize("ending", [None, ValueError, KeyboardInterrupt])
def test_the_read_changes_nothing_and_leaves_no_hook(model, ending):
    model, torch = model
    with torch.no_grad():
        before = model(torch.tensor([SIDES[0][0]])).logits
    passes = []

    def stop(module, args, output):  # the read ends partway, after two forward passes
        passes.append(1)
        if ending is not None and len(passes) == 3:
            raise ending

    handle = model.model.layers[0].register_forward_hook(stop)
    raised = None
    try:
        read_blocks(model, prompts(), lambda k, total: None)
    except (ValueError, KeyboardInterrupt) as e:
        raised = type(e)
    handle.remove()
    assert (raised, len(passes)) == (ending, 8 if ending is None else 3)
    assert all(not block._forward_hooks for block in model.model.layers)
    with torch.no_grad():
        assert torch.equal(model(torch.tensor([SIDES[0][0]])).logits, before)


def test_a_zero_difference_is_refused(model):
    model, torch = model

    def same(module, args, output):  # block 2 gives the same output for every prompt
        return torch.zeros_like(output)

    handle = model.model.layers[2].register_forward_hook(same)
    try:
        with pytest.raises(ConfigError) as e:
            read_blocks(model, prompts(), lambda k, total: None)
    finally:
        handle.remove()
    assert str(e.value) == (
        "Pair 1's two prompts give the same output at block 2, so it has no direction to add."
    )


def test_two_pairs_that_swap_sides_are_refused(model):
    model, _ = model
    a, b = SIDES[0]
    with pytest.raises(ConfigError) as e:
        read_blocks(model, prompts([(a, b), (b, a)]), lambda k, total: None)
    assert str(e.value) == (
        "At block 1 the pairs' differences add up to zero, so there's no direction to add; "
        "check for a pair written twice with its sides swapped."
    )


@pytest.mark.parametrize("value", ["nan", "inf"])
def test_an_activation_that_isnt_finite_is_refused(model, value):
    model, torch = model
    seen = []

    def nan(module, args, output):  # the fourth prompt, pair 2's negative, gives a NaN or inf
        seen.append(1)
        return output + float(value) if len(seen) == 4 else output

    handle = model.model.layers[3].register_forward_hook(nan)
    try:
        with pytest.raises(ConfigError) as e:
            read_blocks(model, prompts(), lambda k, total: None)
    finally:
        handle.remove()
    assert str(e.value) == "Block 3's output for pair 2's negative prompt isn't finite."


@pytest.mark.parametrize(
    ("n", "lines"),
    [
        (4, [1, 2, 3, 4]),
        (10, list(range(1, 11))),
        (32, [4, 7, 10, 13, 16, 20, 23, 26, 29, 32]),
    ],
)
def test_progress_at_the_prompts_the_rule_names(n, lines):
    model, _ = tiny("llama")
    sides = [([1, 2 + i], [1, 40 + i]) for i in range(n // 2)]
    called = []
    read_blocks(model, prompts(sides), lambda k, total: called.append((k, total)))
    assert called == [(k, n) for k in lines]
