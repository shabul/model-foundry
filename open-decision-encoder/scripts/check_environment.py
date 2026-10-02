"""Probe real ModernBERT MPS training. Does not train on project data."""

import argparse
import json
import platform
import subprocess
import time
import traceback
from pathlib import Path

import torch
import transformers
from transformers import AutoModel, AutoTokenizer

from decision_encoder.runtime import accelerator_lock, device_for, seed_all, synchronize


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="answerdotai/ModernBERT-base")
    p.add_argument("--steps", type=int, default=100)
    p.add_argument("--output", default="reports/hardware.json")
    args = p.parse_args()
    seed_all(42)
    device = device_for()
    report = {
        "platform": platform.platform(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "mps_available": torch.backends.mps.is_available(),
        "device": str(device),
        "model": args.model,
        "memory_bytes": int(subprocess.check_output(["/usr/sbin/sysctl", "-n", "hw.memsize"])),
        "probes": [],
        "passed": False,
    }
    try:
        with accelerator_lock(device):
            tok = AutoTokenizer.from_pretrained(args.model)
            kw = {"reference_compile": False} if "ModernBERT" in args.model else {}
            model = AutoModel.from_pretrained(args.model, attn_implementation="sdpa", **kw).to(device)
            report["model_revision"] = getattr(model.config, "_commit_hash", None)
            report["parameters"] = sum(p.numel() for p in model.parameters())
            opt = torch.optim.AdamW(model.parameters(), lr=1e-5)
            for length in [256, 512, 1024]:
                batch = tok(
                    "The service timed out. Choose retry or fallback. " * 200,
                    max_length=length,
                    truncation=True,
                    padding="max_length",
                    return_tensors="pt",
                )
                batch = {k: v.to(device) for k, v in batch.items()}
                n = args.steps if length == 512 else 3
                synchronize(device)
                start = time.perf_counter()
                losses = []
                for _ in range(n):
                    opt.zero_grad(set_to_none=True)
                    h = model(**batch).last_hidden_state
                    loss = (h[:, 0, :].float() - 0.1).square().mean()
                    if not torch.isfinite(loss):
                        raise RuntimeError("Non-finite probe loss")
                    loss.backward()
                    opt.step()
                    losses.append(loss.item())
                synchronize(device)
                report["probes"].append(
                    {
                        "length": length,
                        "batch": 1,
                        "steps": n,
                        "seconds": time.perf_counter() - start,
                        "first_loss": losses[0],
                        "last_loss": losses[-1],
                        "allocated_bytes": torch.mps.current_allocated_memory()
                        if str(device) == "mps"
                        else None,
                        "driver_bytes": torch.mps.driver_allocated_memory() if str(device) == "mps" else None,
                    }
                )
                print(json.dumps(report["probes"][-1]), flush=True)
            for mode in ["bf16", "gradient_checkpointing"]:
                try:
                    opt.zero_grad(set_to_none=True)
                    if mode == "gradient_checkpointing":
                        model.gradient_checkpointing_enable()
                        loss = model(**batch).last_hidden_state.float().square().mean()
                    else:
                        with torch.autocast(str(device), dtype=torch.bfloat16):
                            loss = model(**batch).last_hidden_state.float().square().mean()
                    loss.backward()
                    report[mode] = bool(torch.isfinite(loss).item())
                except Exception as exc:
                    report[mode] = str(exc)
            report["passed"] = str(device) == "mps"
    except Exception:
        report["error"] = traceback.format_exc()
        print(report["error"], flush=True)
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2) + "\n")
    path.with_suffix(".md").write_text(
        "# Hardware feasibility\n\n```json\n" + json.dumps(report, indent=2) + "\n```\n"
    )
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
