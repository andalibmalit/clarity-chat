# Steering test fixture

`smollm2-repeng.gguf` is a control vector made outside Clarity, with [repeng](https://github.com/vgel/repeng). Clarity never makes vectors; it applies one made elsewhere. The tests use this file to check that Clarity reads repeng's real output.

- **Tool:** repeng from git, commit `0ba7196d81f7c5de28b1159c9d1732440afd97ce` (`main`, 2025-09-24), installed with `uv pip install "repeng @ git+https://github.com/vgel/repeng@0ba7196"` in a separate venv. Its PyPI release, 0.4.0, needs numpy below 2, so it can't install next to Clarity.
- **Libraries in that venv:** torch 2.15.0, transformers 5.22.2, numpy 2.5.4, tokenizers 0.23.3 (Clarity's pins), and gguf 0.19.0, scikit-learn 1.9.1 (repeng's dependencies).
- **Model:** `HuggingFaceTB/SmolLM2-135M-Instruct` at revision `12fd25f77366fa6b3b4b768ec3050bf629380bac`, float32, on the CPU.
- **Blocks:** 10, 15 and 20 of 30 (`direction.10`, `direction.15`, `direction.20`), 576 values each, each of unit norm. Not the last block: repeng reads the last block's output after the model's final normalization.
- **Keys:** `controlvector.model_hint = "llama"`, `controlvector.layer_count = 3`.
- **Made:** 2026-11-11 on an M2 Max (macOS). sha256 `ed3236f9f64dc4714b402b9a32f7e209471d3108033eff28dc4ad9230d1724f2`, 7,232 bytes.
- **Contrast pairs:** made up: "calm" against "hurried" across six topics (in the script below).

A second run gave a different file: the directions differed by at most 3.3e-7 per value. The tests pin this file, not the run.

The script, run as `python make_vector.py WEIGHTS_FOLDER smollm2-repeng.gguf` with `HF_HUB_OFFLINE=1`:

```python
"""Make the test fixture tests/fixtures/steering/smollm2-repeng.gguf with repeng.

Run in a separate venv (not Clarity's): repeng from git at 0ba7196, torch 2.15.0,
transformers 5.22.2, numpy 2.5.4. Usage: python make_vector.py WEIGHTS_FOLDER OUT.gguf
"""

import sys

import numpy as np
import torch
from repeng import ControlVector, DatasetEntry
from transformers import AutoModelForCausalLM, AutoTokenizer

weights, out = sys.argv[1], sys.argv[2]
torch.manual_seed(0)
tokenizer = AutoTokenizer.from_pretrained(weights)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(weights, dtype=torch.float32).to("cpu")

# Made-up contrast pairs: calm against hurried.
topics = ["the weather", "a train ride", "making tea", "a garden", "the sea", "a library"]
dataset = [
    DatasetEntry(
        positive=f"Speaking slowly and calmly, I tell you about {topic}:",
        negative=f"Speaking fast and in a rush, I tell you about {topic}:",
    )
    for topic in topics
]
vector = ControlVector.train(model, tokenizer, dataset, hidden_layers=[10, 15, 20])
vector.export_gguf(out)

back = ControlVector.import_gguf(out)
assert back.model_type == "llama", back.model_type
assert sorted(back.directions) == [10, 15, 20]
for layer, direction in vector.directions.items():
    assert np.array_equal(back.directions[layer], direction.astype(np.float32))
    print(layer, direction.shape, direction.dtype, float(np.linalg.norm(direction)))
```
