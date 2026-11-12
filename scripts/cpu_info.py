"""Print the CPU and what PyTorch thinks it supports, for CI logs.

Diagnoses a crash seen only on some GitHub Windows runners: "Windows fatal exception:
code 0xc000001d" (illegal instruction) in the first bf16 matrix multiply on the CPU.
Prints only; changes nothing.
"""

import os
import platform

import torch

print("processor:", platform.processor() or "?", "|", os.environ.get("PROCESSOR_IDENTIFIER", ""))
print("machine:", platform.machine(), "| system:", platform.system(), platform.release())
print("torch:", torch.__version__, "| cpu capability:", torch.backends.cpu.get_cpu_capability())
checks = sorted(name for name in dir(torch._C._cpu) if name.startswith("_is_"))
for name in checks:  # queries only; _init_amx would change state, so it isn't called
    print(f"  {name.removeprefix('_is_')}: {getattr(torch._C._cpu, name)()}")
print("mkldnn (oneDNN) available:", torch.backends.mkldnn.is_available())
