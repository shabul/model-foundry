"""Short synthetic batch/dtype feasibility probe, never a task-quality experiment."""

import json
import time
from pathlib import Path

import torch
from transformers import AutoTokenizer

from decision_encoder.modeling import DecisionEncoder
from decision_encoder.runtime import accelerator_lock, device_for, synchronize


def main():
    device = device_for()
    rows = []
    with accelerator_lock(device):
        tokenizer = AutoTokenizer.from_pretrained("answerdotai/ModernBERT-base")
        model = DecisionEncoder.from_base().to(device)
        model.set_trainable("full")
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
        for bf16 in [False, True]:
            for batch_size in [4, 8, 16]:
                inputs = tokenizer(
                    ["A test request has completed successfully. " * 100] * batch_size,
                    padding="max_length",
                    truncation=True,
                    max_length=512,
                    return_tensors="pt",
                )
                inputs = {k: v.to(device) for k, v in inputs.items() if k in ["input_ids", "attention_mask"]}
                inputs["candidate_positions"] = torch.tensor([[10, 20, 30, 40]] * batch_size, device=device)
                inputs["candidate_mask"] = torch.ones((batch_size, 4), device=device, dtype=torch.bool)
                inputs["targets"] = torch.tensor([[1.0, 0.0, 0.0, 0.0]] * batch_size, device=device)
                times = []
                peak = 0
                for step in range(4):
                    optimizer.zero_grad(set_to_none=True)
                    synchronize(device)
                    start = time.perf_counter()
                    with torch.autocast("mps", dtype=torch.bfloat16, enabled=bf16):
                        loss = model(**inputs)["loss"]
                    loss.backward()
                    optimizer.step()
                    synchronize(device)
                    elapsed = time.perf_counter() - start
                    peak = max(peak, torch.mps.driver_allocated_memory())
                    if not torch.isfinite(loss):
                        raise RuntimeError("Nonfinite probe loss")
                    if step > 0:
                        times.append(elapsed)
                row = {
                    "bf16": bf16,
                    "batch_size": batch_size,
                    "length": 512,
                    "step_seconds": sum(times) / len(times),
                    "examples_per_second": batch_size / (sum(times) / len(times)),
                    "max_observed_driver_bytes": peak,
                }
                rows.append(row)
                print(json.dumps(row), flush=True)
                torch.mps.empty_cache()
    Path("reports/training_probe.json").write_text(json.dumps(rows, indent=2) + "\n")


if __name__ == "__main__":
    main()
