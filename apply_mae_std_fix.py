#!/usr/bin/env python3
"""CPU-check and apply a narrow safe_std memory fix to the supplied MAE implementation.

Usage: python apply_mae_std_fix.py models/mae_model.py [--check-only]
Requires PyTorch in the active environment. No GPU or model checkpoint is loaded.
The source is backed up and changed only after all numerical checks pass.
"""
import argparse
import ast
from datetime import datetime, timezone
from pathlib import Path
import shutil
import uuid

OLD = '''def safe_std(x: torch.Tensor, axis, eps: float = 1e-6, keepdims: bool = False) -> torch.Tensor:
    x32 = x.float()
    mean = x32.mean(dim=axis, keepdim=True)
    var = ((x32 - mean) ** 2).mean(dim=axis, keepdim=keepdims)
    return torch.sqrt(torch.clamp(var, min=0.0) + eps)'''

NEW = '''def safe_std(x: torch.Tensor, axis, eps: float = 1e-6, keepdims: bool = False) -> torch.Tensor:
    x32 = x.float()
    # Population variance, matching the original mean squared deviation.
    # Avoid explicitly materializing full-size centered and squared tensors.
    var = torch.var(x32, dim=axis, correction=0, keepdim=keepdims)
    return torch.sqrt(torch.clamp(var, min=0.0) + eps)'''


def validate_numerics():
    import torch

    namespace_old = {"torch": torch}
    namespace_new = {"torch": torch}
    exec(compile(OLD, "<original safe_std>", "exec"), namespace_old)
    exec(compile(NEW, "<replacement safe_std>", "exec"), namespace_new)
    original = namespace_old["safe_std"]
    replacement = namespace_new["safe_std"]
    rng = torch.Generator(device="cpu").manual_seed(42)
    checked = 0
    for dtype in (torch.float32, torch.bfloat16, torch.float64):
        cases = [
            (torch.randn(3, 5, 4, 7, generator=rng).to(dtype), 2, False),
            (torch.randn(3, 5, 4, 7, generator=rng).to(dtype), (1, 2), False),
            (torch.randn(3, 5, 4, 7, generator=rng).to(dtype), (1, 2), True),
            (torch.randn(3, 4, 5, 7, generator=rng).to(dtype).transpose(1, 2), 2, False),
            (torch.ones(3, 5, 4, 7, dtype=dtype), 2, False),
            (torch.randn(3, 5, 1, 7, generator=rng).to(dtype), 2, False),
        ]
        for data, axis, keepdims in cases:
            a = data.detach().clone().requires_grad_(True)
            b = data.detach().clone().requires_grad_(True)
            y_old = original(a, axis, keepdims=keepdims)
            y_new = replacement(b, axis, keepdims=keepdims)
            torch.testing.assert_close(y_new, y_old, rtol=5e-5, atol=5e-6)
            weight = torch.randn(y_old.shape, generator=rng)
            grad_old, = torch.autograd.grad((y_old * weight).sum(), a)
            grad_new, = torch.autograd.grad((y_new * weight).sum(), b)
            if dtype == torch.bfloat16:
                # Input gradients are rounded back to bf16.
                torch.testing.assert_close(grad_new, grad_old, rtol=2e-2, atol=2e-3)
            else:
                torch.testing.assert_close(grad_new, grad_old, rtol=2e-4, atol=2e-5)
            checked += 1
    print(f"PASS: {checked} CPU forward/input-gradient comparisons (PyTorch {torch.__version__}).")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=Path)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    path = args.path.resolve(strict=True)
    source = path.read_text()
    already_patched = source.count(NEW) == 1 and OLD not in source
    if not already_patched and source.count(OLD) != 1:
        raise SystemExit("Source does not match the supplied safe_std implementation exactly; no file changed.")
    proposed = source if already_patched else source.replace(OLD, NEW, 1)
    compile(proposed, str(path), "exec")
    # Verify that every other top-level definition and statement is untouched.
    before, after = ast.parse(source), ast.parse(proposed)
    for tree in (before, after):
        tree.body = [node for node in tree.body if not (
            isinstance(node, ast.FunctionDef) and node.name == "safe_std"
        )]
    if ast.dump(before) != ast.dump(after):
        raise SystemExit("Unexpected change outside safe_std; no file changed.")
    validate_numerics()
    if args.check_only or already_patched:
        print("No file changed." if args.check_only else "Fix already applied; no file changed.")
        return
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    backup = path.with_name(path.name + f".before_std_fix_{stamp}_{uuid.uuid4().hex[:8]}")
    shutil.copy2(path, backup)
    path.write_text(proposed)
    print(f"Backup: {backup}")
    print(f"Updated only safe_std in: {path}")
    print("GPU peak memory and full training still require a cluster test.")


if __name__ == "__main__":
    main()
