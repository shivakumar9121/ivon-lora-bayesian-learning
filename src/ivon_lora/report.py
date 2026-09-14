"""Recompute every metric from saved predictions and create tables and plots."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np

from .metrics import classification_metrics, reliability_bins

METHODS = ["adamw", "ivon_mean", "ivon_mc"]
METRICS = ["accuracy", "ece", "nll", "brier"]
LABELS = {"adamw": "AdamW-LoRA", "ivon_mean": "IVON @ mean", "ivon_mc": "IVON, 10 samples"}


def mean_std(values):
    values = np.asarray(values, dtype=float)
    return {"mean": float(values.mean()),
            "std": float(values.std(ddof=1)) if len(values) > 1 else None,
            "n_seeds": len(values)}


def paired_deltas(records, metric, method):
    base = {r["seed"]: r[metric] for r in records if r["method"] == "adamw"}
    other = {r["seed"]: r[metric] for r in records if r["method"] == method}
    if not base or base.keys() != other.keys():
        raise ValueError("Paired comparisons require identical seed sets")
    return mean_std([other[s] - base[s] for s in sorted(base)])


def build_report(root):
    root = Path(root)
    config = json.loads((root / "config.json").read_text())
    expected = set(config["seeds"])
    records, predictions, completed = [], {}, {}
    for seed in sorted(expected):
        completed[seed] = {}
        for opt in ["adamw", "ivon"]:
            path = root / f"seed_{seed}" / opt
            if not (path / "complete.json").exists():
                raise ValueError(f"Missing completed run: {path}. Partial runs are never labeled complete.")
            info = json.loads((path / "complete.json").read_text())
            completed[seed][opt] = info
            for method in (["adamw"] if opt == "adamw" else ["ivon_mean", "ivon_mc"]):
                pred = json.loads((path / f"test_{method}_predictions.json").read_text())
                values = classification_metrics(pred["probabilities"], np.array(pred["labels"], dtype=int), config["ece_bins"])
                saved = json.loads((path / f"test_{method}.json").read_text())
                for metric in METRICS:
                    if not np.isclose(values[metric], saved[metric], atol=1e-10, rtol=0):
                        raise ValueError(f"Saved metric does not match predictions: {path}, {metric}")
                records.append({"seed": seed, "method": method, **values,
                    "train_seconds": info["train_seconds"], "prediction_seconds": saved["prediction_seconds"]})
                predictions[(seed, method)] = pred
        if completed[seed]["adamw"]["initial_adapter_sha256"] != completed[seed]["ivon"]["initial_adapter_sha256"]:
            raise ValueError(f"Initial adapter mismatch for paired seed {seed}")
    reference = next(iter(predictions.values()))
    for pred in predictions.values():
        if pred["ids"] != reference["ids"] or pred["labels"] != reference["labels"]:
            raise ValueError("Test examples or labels differ across methods/seeds")
    dest = root / "summary"
    dest.mkdir(exist_ok=True)
    def save_csv(name, rows):
        with (dest / name).open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    save_csv("per_seed.csv", records)
    summary = []
    for method in METHODS:
        row = {"method": method, "n_seeds": len(expected)}
        for metric in METRICS + ["train_seconds", "prediction_seconds"]:
            stats = mean_std([r[metric] for r in records if r["method"] == method])
            row[metric + "_mean"] = stats["mean"]
            row[metric + "_std"] = stats["std"]
        summary.append(row)
    save_csv("comparison.csv", summary)
    deltas = [{"method": method, "metric": metric, **paired_deltas(records, metric, method)}
              for method in METHODS[1:] for metric in METRICS]
    save_csv("paired_deltas.csv", deltas)
    payload = {"status": "measured", "model": config["model_id"],
        "dataset": config["dataset_config"], "config": config,
        "n_test": len(reference["labels"]), "summary": summary, "paired_deltas": deltas,
        "note": "Mean and sample SD across seeds. Delta = IVON minus AdamW. No significance claim from three seeds."}
    (dest / "summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    make_plots(dest, records, predictions, config)
    # Readable terminal output, also captured in the Colab notebook.
    print("\nMEASURED TEST RESULTS (mean +/- sample SD across seeds)")
    for row in summary:
        def val(m, scale=1):
            sd = row[m + "_std"]
            return f"{scale * row[m + '_mean']:.3f} +/- {scale * sd:.3f}" if sd is not None else f"{scale * row[m + '_mean']:.3f} (one seed)"
        print(f"{row['method']:12} accuracy %: {val('accuracy',100)}   ECE pp: {val('ece',100)}   NLL: {val('nll')}")
    return payload


def make_plots(dest, records, predictions, config):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    colors = ["#43566e", "#108078", "#b15a35"]
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.7), constrained_layout=True)
    for ax, metric, title, factor in zip(axes, METRICS[:3], ["Accuracy (%)", "ECE (percentage points)", "NLL (natural log)"], [100,100,1]):
        for i, method in enumerate(METHODS):
            values = np.array([r[metric] * factor for r in records if r["method"] == method])
            ax.errorbar(i, values.mean(), yerr=values.std(ddof=1) if len(values) > 1 else None,
                        color=colors[i], fmt="o", capsize=5, markersize=8)
            ax.scatter(i + np.linspace(-.07,.07,len(values)), values, color=colors[i], s=18, alpha=.6)
        ax.set_xticks(range(3), ["AdamW", "IVON mean", "IVON MC"])
        ax.set_ylabel(title)
        ax.set_xlabel("Prediction method")
        ax.set_title(title + (" | higher is better" if metric == "accuracy" else " | lower is better"), fontsize=10)
    fig.suptitle(f"{config['model_id']} on {config['dataset_config']} test subset\nMeasured seed points, mean and sample SD", fontsize=12)
    fig.savefig(dest / "comparison.png", dpi=180)
    plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.9), constrained_layout=True)
    # Reliability diagrams show a single explicitly labeled seed. Pooling the
    # predictions across seeds would evaluate a different predictor/estimand.
    seed = config["seeds"][0]
    for ax, method, color in zip(axes, METHODS, colors):
        pred = predictions[(seed, method)]
        bins = reliability_bins(pred["probabilities"], np.array(pred["labels"],dtype=int), config["ece_bins"])
        populated = [b for b in bins if b["count"]]
        ax.plot([0,1], [0,1], "--", color="gray", label="Perfect calibration")
        ax.scatter([b["confidence"] for b in populated], [b["accuracy"] for b in populated],
                   s=[15 + b["count"] * 2 for b in populated], color=color, label="Observed bins")
        ax.set(xlim=(0,1), ylim=(0,1), xlabel="Mean confidence (probability)",
               ylabel="Empirical accuracy (fraction)", title=method)
        ax.legend(fontsize=7)
    fig.suptitle(f"Reliability on the held-out test subset, seed {seed}\n{config['ece_bins']} equal-width bins. Marker area increases with bin count.", fontsize=12)
    fig.savefig(dest / "reliability.png", dpi=180)
    plt.close(fig)
    # ECE bin sensitivity is descriptive, never a search for the best-looking ECE.
    sensitivity = []
    for (seed, method), pred in predictions.items():
        for n_bins in [5,10,15,20]:
            m = classification_metrics(pred["probabilities"], np.array(pred["labels"],dtype=int), n_bins)
            sensitivity.append({"seed":seed,"method":method,"bins":n_bins,"ece":m["ece"]})
    with (dest / "ece_sensitivity.csv").open("w",newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(sensitivity[0]))
        writer.writeheader()
        writer.writerows(sensitivity)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir")
    build_report(parser.parse_args().run_dir)
