"""Reproducible sequential training with best-validation checkpointing."""

import argparse
import json
import random
import shutil
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from decision_encoder.collator import DecisionCollator
from decision_encoder.data.schema import assert_no_leakage, read_jsonl
from decision_encoder.metrics import metrics
from decision_encoder.modeling import DecisionEncoder
from decision_encoder.runtime import accelerator_lock, device_for, provenance, seed_all, sha256_file


def batches_to_device(batch, device):
    metadata = batch.pop("metadata")
    return {k: v.to(device) for k, v in batch.items()}, metadata


@torch.inference_mode()
def evaluate(model, loader, device):
    model.eval()
    ps, ys = [], []
    truncated = 0
    for batch in loader:
        batch, metadata = batches_to_device(batch, device)
        output = model(**batch)
        for p, y, mask in zip(
            output["probabilities"].cpu(), batch["targets"].cpu(), batch["candidate_mask"].cpu()
        ):
            ps.append(p[mask].tolist())
            ys.append(y[mask].tolist())
        truncated += sum(metadata["truncated"])
    result = metrics(ps, ys)
    result.pop("risk_coverage")
    result["truncated_examples"] = truncated
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--resume", help="Resume trusted local optimizer state from this run directory")
    args = parser.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    seed_all(cfg["seed"])
    device = device_for(cfg.get("device", "auto"))
    if cfg.get("require_hardware", True):
        report = json.loads(Path("reports/hardware.json").read_text())
        if not report["passed"]:
            raise RuntimeError("Hardware gate has not passed")
    if cfg["stage"] != "smoke":
        gate = json.loads(Path("reports/smoke.json").read_text())
        if not gate["passed"]:
            raise RuntimeError("Tiny-overfit gate has not passed")
    train = read_jsonl(cfg["train_data"])
    valid = read_jsonl(cfg["validation_data"])
    if cfg["stage"] != "smoke":
        assert_no_leakage({"train": train, "validation": valid})
    rng = random.Random(cfg["seed"])
    rng.shuffle(train)
    rng.shuffle(valid)
    train = train[: cfg.get("max_train_examples", len(train))]
    valid = valid[: cfg.get("max_validation_examples", len(valid))]
    if not train or not valid:
        raise ValueError("Empty training or validation split")
    outdir = Path(cfg["output"])
    outdir.mkdir(parents=True, exist_ok=True)
    run = {
        "config": cfg,
        "provenance": provenance(Path.cwd()),
        "train_sha256": sha256_file(cfg["train_data"]),
        "validation_sha256": sha256_file(cfg["validation_data"]),
        "manifest_sha256": sha256_file("data/manifests/dataset.json")
        if Path("data/manifests/dataset.json").exists()
        else None,
        "train_n": len(train),
        "validation_n": len(valid),
        "device": str(device),
    }
    snapshot = outdir / "source_snapshot"
    if not args.resume:
        for folder in ["src", "scripts", "configs"]:
            shutil.copytree(
                folder, snapshot / folder, dirs_exist_ok=True, ignore=shutil.ignore_patterns("__pycache__")
            )
    with accelerator_lock(device):
        if args.resume:
            model = DecisionEncoder.from_pretrained(Path(args.resume) / "last")
            tokenizer = AutoTokenizer.from_pretrained(Path(args.resume) / "last/tokenizer")
        elif cfg.get("initialize_from"):
            model = DecisionEncoder.from_pretrained(cfg["initialize_from"])
            tokenizer = AutoTokenizer.from_pretrained(Path(cfg["initialize_from"]) / "tokenizer")
        else:
            model = DecisionEncoder.from_base(cfg["model"], cfg.get("model_revision"))
            tokenizer = AutoTokenizer.from_pretrained(cfg["model"], revision=cfg.get("model_revision"))
        model.temperature = 1.0
        model.set_trainable(cfg["mode"])
        if cfg.get("gradient_checkpointing"):
            model.encoder.gradient_checkpointing_enable()
        model.to(device)
        train_collator = DecisionCollator(
            tokenizer,
            cfg["max_length"],
            shuffle=True,
            seed=cfg["seed"],
            overflow=cfg.get("overflow", "error"),
        )
        eval_collator = DecisionCollator(tokenizer, cfg["max_length"], overflow=cfg.get("overflow", "error"))
        loader_generator = torch.Generator().manual_seed(cfg["seed"])
        loader = DataLoader(
            train,
            batch_size=cfg["batch_size"],
            shuffle=True,
            collate_fn=train_collator,
            generator=loader_generator,
        )
        valid_loader = DataLoader(valid, batch_size=cfg["batch_size"], collate_fn=eval_collator)
        eval_train_loader = DataLoader(train, batch_size=cfg["batch_size"], collate_fn=eval_collator)
        groups = [{"params": list(model.head.parameters()), "lr": cfg["head_lr"]}]
        enc_params = [p for p in model.encoder.parameters() if p.requires_grad]
        if enc_params:
            groups.append({"params": enc_params, "lr": cfg["encoder_lr"]})
        optimizer = torch.optim.AdamW(groups, weight_decay=cfg.get("weight_decay", 0.01))
        run["trainable_parameters"] = sum(p.numel() for p in model.parameters() if p.requires_grad)
        run["total_parameters"] = sum(p.numel() for p in model.parameters())
        best_nll, start_epoch = float("inf"), 0
        if args.resume:
            state = torch.load(Path(args.resume) / "resume.pt", map_location="cpu", weights_only=False)
            if state["train_sha256"] != run["train_sha256"] or state["config"] != cfg:
                raise ValueError("Resume data/config mismatch")
            optimizer.load_state_dict(state["optimizer"])
            # Adam state follows parameter device after load_state_dict.
            start_epoch, best_nll = state["epoch"] + 1, state["best_nll"]
            torch.set_rng_state(state["torch_rng"])
            random.setstate(state["python_rng"])
            np.random.set_state(state["numpy_rng"])
            loader_generator.set_state(state["loader_rng"])
            train_collator.rng.setstate(state["collator_rng"])
            if str(device) == "mps" and state.get("mps_rng") is not None:
                torch.mps.set_rng_state(state["mps_rng"])
        run["baseline"] = evaluate(model, valid_loader, device)
        print(json.dumps({"baseline": run["baseline"]}), flush=True)
        history = []
        begin = time.perf_counter()
        max_driver_bytes = 0
        max_allocated_bytes = 0
        for epoch in range(start_epoch, cfg["epochs"]):
            model.train()
            optimizer.zero_grad(set_to_none=True)
            total_loss, seen, truncated = 0.0, 0, 0
            accumulation = cfg["gradient_accumulation"]
            for step, batch in enumerate(loader):
                batch, meta = batches_to_device(batch, device)
                # Weight by examples, including a short final microbatch and accumulation window.
                window_start = (step // accumulation) * accumulation
                window_examples = min(
                    accumulation * cfg["batch_size"], len(train) - window_start * cfg["batch_size"]
                )
                use_bf16 = cfg.get("bf16", False)
                with torch.autocast(str(device), dtype=torch.bfloat16, enabled=use_bf16):
                    output = model(**batch)
                    loss = output["loss"]
                if not torch.isfinite(loss):
                    raise RuntimeError(f"Nonfinite loss at epoch {epoch}, batch {step}")
                size = len(batch["input_ids"])
                (loss * size / window_examples).backward()
                if str(device) == "mps":
                    max_driver_bytes = max(max_driver_bytes, torch.mps.driver_allocated_memory())
                    max_allocated_bytes = max(max_allocated_bytes, torch.mps.current_allocated_memory())
                total_loss += loss.item() * size
                seen += size
                truncated += sum(meta["truncated"])
                if (step + 1) % accumulation == 0 or step + 1 == len(loader):
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                if step % 100 == 0:
                    print(
                        json.dumps(
                            {
                                "epoch": epoch + 1,
                                "batch": step,
                                "loss": total_loss / seen,
                                "elapsed_seconds": time.perf_counter() - begin,
                            }
                        ),
                        flush=True,
                    )
            score = evaluate(model, valid_loader, device)
            row = {
                "epoch": epoch + 1,
                "train_loss": total_loss / seen,
                "validation": score,
                "seconds": time.perf_counter() - begin,
                "train_truncated": truncated,
            }
            if cfg["stage"] == "smoke":
                row["train_eval"] = evaluate(model, eval_train_loader, device)
            history.append(row)
            print(json.dumps(row), flush=True)
            if score["nll"] < best_nll:
                best_nll = score["nll"]
                model.save_pretrained(
                    outdir / "best", tokenizer, {**run, "best_epoch": epoch + 1, "validation": score}
                )
            model.save_pretrained(outdir / "last", tokenizer, run)
            torch.save(
                {
                    "epoch": epoch,
                    "config": cfg,
                    "best_nll": best_nll,
                    "train_sha256": run["train_sha256"],
                    "optimizer": optimizer.state_dict(),
                    "torch_rng": torch.get_rng_state(),
                    "python_rng": random.getstate(),
                    "numpy_rng": np.random.get_state(),
                    "loader_rng": loader_generator.get_state(),
                    "collator_rng": train_collator.rng.getstate(),
                    "mps_rng": torch.mps.get_rng_state() if str(device) == "mps" else None,
                },
                outdir / "resume.pt",
            )
            run["memory"] = {
                "max_observed_driver_bytes": max_driver_bytes,
                "max_observed_allocated_bytes": max_allocated_bytes,
                "scope": "sampled after backward; not total system memory",
            }
            run.update(history=history, best_validation_nll=best_nll, seconds=time.perf_counter() - begin)
            (outdir / "run.json").write_text(json.dumps(run, indent=2) + "\n")
            Path(f"reports/{cfg['stage']}.json").write_text(json.dumps(run, indent=2) + "\n")
            if cfg["stage"] == "smoke":
                passed = row["train_eval"]["accuracy"] >= 0.98 and row["train_eval"]["nll"] <= 0.10
                gate = {**run, "passed": passed}
                Path("reports/smoke.json").write_text(json.dumps(gate, indent=2) + "\n")
                if passed:
                    print("Tiny-overfit gate passed.", flush=True)
                    break
        if cfg["stage"] == "smoke" and not passed:
            raise SystemExit("Tiny-overfit gate failed; larger training is prohibited.")


if __name__ == "__main__":
    main()
