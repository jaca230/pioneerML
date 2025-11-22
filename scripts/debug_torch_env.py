"""
Quick diagnostics for the local PyTorch/CUDA setup.

Prints versions, compiled architectures, detected devices, and verifies
basic tensor operations on the selected accelerator.
"""

from __future__ import annotations

import sys
from textwrap import indent

import torch


def main() -> int:
    print(f"torch: {torch.__version__}")
    print(f"torch cuda version: {torch.version.cuda}")
    print(f"cuda available: {torch.cuda.is_available()}")

    compiled_arches = torch.cuda.get_arch_list()
    print(f"compiled arch list: {compiled_arches}")

    if not torch.cuda.is_available():
        print("CUDA not available; skipping device checks.")
        return 0

    device_count = torch.cuda.device_count()
    print(f"device count: {device_count}")

    for idx in range(device_count):
        props = torch.cuda.get_device_properties(idx)
        cc = f"sm_{props.major}{props.minor}"
        print(f"device {idx}: {props.name} (cc {props.major}.{props.minor}, {cc})")
        if cc not in compiled_arches:
            print(f"WARNING: device cc {cc} not in compiled arch list; kernels may not run.")

    # Try a simple tensor op
    try:
        x = torch.ones(2, device="cuda")
        y = x * 2
        print(f"CUDA tensor test succeeded: {y}")
    except Exception as exc:  # pragma: no cover - this is a runtime diagnostic
        print("CUDA tensor test failed:", exc)

    # Try a tiny matmul to exercise a kernel
    try:
        a = torch.randn(8, 8, device="cuda")
        b = torch.randn(8, 8, device="cuda")
        c = a @ b
        print("CUDA matmul succeeded; mean:", c.mean().item())
    except Exception as exc:  # pragma: no cover
        print("CUDA matmul failed:", exc)

    return 0


if __name__ == "__main__":
    sys.exit(main())
