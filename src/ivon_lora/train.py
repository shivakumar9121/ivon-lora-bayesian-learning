"""Paired-seed experiments, atomic result files, and complete provenance."""
import argparse
from contextlib import nullcontext
import gc
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import random
import subprocess
import time

import numpy as np
import torch
import torch.nn.functional as F
from datasets import load_dataset
from huggingface_hub import HfApi
from ivon import IVON
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer

from .data import Collator, prepare_splits
from .metrics import classification_metrics
from .model import attach_lora, option_logits, option_token_ids


def load_config(path):
    path = Path(path)
    config = json.loads(path.read_text())
    if "extends" in config:
        parent = load_config(path.parent / config.pop("extends"))
        parent.update(config)
        config = parent
    for key in ["steps", "batch_size", "accumulation_steps", "posterior_samples", "ece_bins"]:
        if not isinstance(config[key], int) or config[key] <= 0:
            raise ValueError(f"{key} must be a positive integer")
    if not config["seeds"] or len(set(config["seeds"])) != len(config["seeds"]):
        raise ValueError("Provide distinct seeds")
    if config["dtype"] not in ["float32", "bfloat16"]:
        raise ValueError("Use float32 on T4, or bfloat16 on a supported GPU")
    return config


def write_json(path, content):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(content, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    # Reproducibility is expected on the same environment. Cross-device bitwise
    # equivalence is not claimed. No stochastic dropout is used in the defaults.


def to_device(batch, device):
    return {key: value.to(device) for key, value in batch.items()}


def make_optimizer(model, method, config):
    params = [p for p in model.parameters() if p.requires_grad]
    if method == "adamw":
        return torch.optim.AdamW(params, lr=config["adamw_lr"], weight_decay=config["weight_decay"])
    if method != "ivon":
        raise ValueError(method)
    return IVON(params, lr=config["ivon_lr"], ess=config["ess"],
                hess_init=config["hess_init"], weight_decay=config["weight_decay"],
                clip_radius=config["clip_radius"], beta2=config["beta2"], rescale_lr=False)


def train_step(model, optimizer, microbatches, token_ids, method):
    optimizer.zero_grad(set_to_none=True)
    n = sum(len(b["labels"]) for b in microbatches)
    loss_value = 0.0
    # One posterior draw remains active across all accumulation microbatches.
    # IVON collects the final gradient on context exit. Never clip or unscale
    # gradients after that point: IVON has already copied them into its state.
    with optimizer.sampled_params(train=True) if method == "ivon" else nullcontext():
        for batch in microbatches:
            loss = F.cross_entropy(option_logits(model, batch, token_ids), batch["labels"], reduction="sum") / n
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite loss. The run has been stopped.")
            loss.backward()
            loss_value += loss.detach().item()
        if any(p.grad is None or not torch.isfinite(p.grad).all()
               for p in model.parameters() if p.requires_grad):
            raise FloatingPointError("Missing or nonfinite adapter gradients")
    optimizer.step()
    return loss_value


@torch.inference_mode()
def predict(model, loader, token_ids, device):
    model.eval()
    parts = []
    for batch in loader:
        logits = option_logits(model, to_device(batch, device), token_ids)
        parts.append(logits.softmax(-1).double().cpu().numpy())
    return np.concatenate(parts)


def evaluate(model, optimizer, method, loader, token_ids, device, samples, seed):
    if method != "ivon_mc":
        return predict(model, loader, token_ids, device)
    # A fixed draw spans the entire evaluation set. Average probabilities,
    # not logits or weights. fork_rng prevents evaluation changing train RNG.
    devices = [torch.cuda.current_device()] if device == "cuda" else []
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(seed)
        total = None
        for sample in range(samples):
            with optimizer.sampled_params(train=False):
                p = predict(model, loader, token_ids, device)
            total = p if total is None else total + p
            print(f"  posterior draw {sample + 1}/{samples}", flush=True)
    return total / samples


def source_hash():
    digest = hashlib.sha256()
    for path in sorted(Path(__file__).parent.glob("*.py")):
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def sync(device):
    if device == "cuda":
        torch.cuda.synchronize()


def run(config, out_dir, device):
    root = Path(out_dir)
    root.mkdir(parents=True, exist_ok=True)
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("No CUDA GPU. Select Runtime > Change runtime type > T4 GPU in Colab.")
    if config["dtype"] == "bfloat16" and (device != "cuda" or not torch.cuda.is_bf16_supported()):
        raise RuntimeError("bfloat16 requires a supported GPU. T4 must use float32.")
    # Resolve mutable hub refs once, then pin downloads to immutable commits.
    api = HfApi()
    config = dict(config)
    config["model_revision"] = api.model_info(config["model_id"], revision=config.get("model_revision")).sha
    config["dataset_revision"] = api.dataset_info(config["dataset_id"], revision=config.get("dataset_revision")).sha
    config["source_sha256"] = source_hash()
    config_path = root / "config.json"
    if config_path.exists() and json.loads(config_path.read_text()) != config:
        raise ValueError("Output already belongs to a different config/source revision. Use a new --output directory.")
    write_json(config_path, config)
    tokenizer = AutoTokenizer.from_pretrained(config["model_id"], revision=config["model_revision"])
    tokenizer.pad_token = tokenizer.eos_token
    token_ids = option_token_ids(tokenizer)
    raw = load_dataset(config["dataset_id"], config["dataset_config"], revision=config["dataset_revision"])
    splits, manifest = prepare_splits(raw, tokenizer, config)
    manifest["dataset_id"] = config["dataset_id"]
    manifest["dataset_revision"] = config["dataset_revision"]
    write_json(root / "split_manifest.json", manifest)
    environment = {"python": os.sys.version, "torch": torch.__version__,
        "device": torch.cuda.get_device_name(0) if device == "cuda" else "CPU",
        "cuda": torch.version.cuda, "option_token_ids": token_ids,
        "packages": {p: importlib.metadata.version(p) for p in
            ["transformers", "peft", "ivon-opt", "datasets", "numpy"]}}
    write_json(root / "environment.json", environment)
    collator = Collator(tokenizer.pad_token_id)
    loaders = {split: DataLoader(rows, batch_size=config["eval_batch_size"],
        collate_fn=collator, shuffle=False) for split, rows in splits.items() if split != "train"}
    for seed in config["seeds"]:
        for method in ["adamw", "ivon"]:
            target = root / f"seed_{seed}" / method
            if (target / "complete.json").exists():
                print(f"Already complete: {target}", flush=True)
                continue
            target.mkdir(parents=True, exist_ok=True)
            seed_all(seed)
            dtype = getattr(torch, config["dtype"])
            base = AutoModelForCausalLM.from_pretrained(config["model_id"],
                revision=config["model_revision"], torch_dtype=dtype, attn_implementation="sdpa")
            model = attach_lora(base, config).to(device)
            del base
            trainable = {name: p for name, p in model.named_parameters() if p.requires_grad}
            initial_hash = hashlib.sha256(b"".join(p.detach().cpu().numpy().tobytes()
                                                    for p in trainable.values())).hexdigest()
            optimizer = make_optimizer(model, method, config)
            generator = torch.Generator().manual_seed(seed)
            loader = DataLoader(splits["train"], batch_size=config["batch_size"],
                shuffle=True, collate_fn=collator, generator=generator)
            iterator = iter(loader)
            if device == "cuda":
                torch.cuda.reset_peak_memory_stats()
            initial_lr = config[f"{method}_lr"]
            warmup = max(1, math.ceil(config["steps"] * config["warmup_fraction"]))
            history = []
            print(f"START {method} seed={seed}, {config['steps']} steps, {sum(p.numel() for p in trainable.values()):,} trainable parameters", flush=True)
            sync(device)
            started = time.perf_counter()
            for step in range(config["steps"]):
                scale = (step + 1) / warmup if step < warmup else (config["steps"] - step) / max(1, config["steps"] - warmup)
                for group in optimizer.param_groups:
                    group["lr"] = initial_lr * scale
                microbatches = []
                for _ in range(config["accumulation_steps"]):
                    try:
                        batch = next(iterator)
                    except StopIteration:
                        iterator = iter(loader)
                        batch = next(iterator)
                    microbatches.append(to_device(batch, device))
                model.train()
                loss = train_step(model, optimizer, microbatches, token_ids, method)
                history.append({"step": step + 1, "loss": loss, "lr": initial_lr * scale})
                if step == 0 or (step + 1) % 16 == 0:
                    print(f"  step {step + 1}/{config['steps']} loss={loss:.4f}", flush=True)
            sync(device)
            train_seconds = time.perf_counter() - started
            peak = torch.cuda.max_memory_allocated() / 1024**3 if device == "cuda" else None
            write_json(target / "training_history.json", history)
            run_info = {"seed": seed, "optimizer": method, "initial_adapter_sha256": initial_hash,
                "trainable_parameters": sum(p.numel() for p in trainable.values()),
                "total_parameters": sum(p.numel() for p in model.parameters()),
                "train_seconds": train_seconds, "peak_train_memory_gib": peak,
                "steps": config["steps"], "status": "measured", "results": []}
            if config["save_checkpoints"]:
                model.save_pretrained(target / "adapter")
                tokenizer.save_pretrained(target / "adapter")
                # IVON state stores the posterior Hessian; mean adapters alone
                # cannot reproduce posterior-sampling predictions.
                torch.save({"optimizer": optimizer.state_dict(),
                            "ivon_current_step": getattr(optimizer, "current_step", None),
                            "trainable_parameter_names": list(trainable)}, target / "optimizer.pt")
            for split in ["validation", "test"]:
                labels = np.array([r["label"] for r in splits[split]], dtype=int)
                names = ["adamw"] if method == "adamw" else ["ivon_mean", "ivon_mc"]
                for prediction_method in names:
                    sync(device)
                    eval_start = time.perf_counter()
                    probs = evaluate(model, optimizer, prediction_method, loaders[split], token_ids,
                        device, config["posterior_samples"], seed + 10000 + (split == "test"))
                    sync(device)
                    metrics = classification_metrics(probs, labels, config["ece_bins"])
                    metrics.update({"method": prediction_method, "split": split, "seed": seed,
                        "prediction_seconds": time.perf_counter() - eval_start,
                        "posterior_samples": config["posterior_samples"] if prediction_method == "ivon_mc" else 0})
                    write_json(target / f"{split}_{prediction_method}.json", metrics)
                    # Dataset text is not redistributed. IDs map to the cited ARC dataset.
                    prediction_file = target / f"{split}_{prediction_method}_predictions.json"
                    write_json(prediction_file, {"ids": [r["id"] for r in splits[split]],
                                                "labels": labels.tolist(), "probabilities": probs.tolist()})
                    run_info["results"].append(metrics)
                    print(json.dumps(metrics), flush=True)
            write_json(target / "complete.json", run_info)
            del model, optimizer, trainable, microbatches
            gc.collect()
            if device == "cuda":
                torch.cuda.empty_cache()
    print("ALL RUNS COMPLETE. Generating summary.", flush=True)
    from .report import build_report
    build_report(root)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/quick.json")
    parser.add_argument("--output", default="runs/quick")
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    args = parser.parse_args()
    run(load_config(args.config), args.output, args.device)


if __name__ == "__main__":
    main()
