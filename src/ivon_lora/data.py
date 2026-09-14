"""Fixed ARC subsets, robust answer mapping, and explicit exclusions."""
import hashlib
import json
import random
from collections import Counter

LETTERS = "ABCD"


def format_arc(row):
    labels = [str(x) for x in row["choices"]["label"]]
    texts = row["choices"]["text"]
    if len(labels) != 4 or len(texts) != 4 or len(set(labels)) != 4:
        raise ValueError("Exactly four distinct option labels are required")
    key = str(row["answerKey"])
    if key not in labels:
        raise ValueError("Answer key is absent from the option labels")
    choices = "\n".join(f"{letter}. {text}" for letter, text in zip(LETTERS, texts))
    # Ground truth is deliberately absent from the input prompt.
    prompt = f"Select the correct answer.\nQuestion: {row['question']}\n{choices}\nAnswer:"
    return {"id": str(row["id"]), "prompt": prompt, "label": labels.index(key)}


def prepare_splits(raw, tokenizer, config):
    result, manifest = {}, {"data_seed": config["data_seed"], "splits": {}}
    for split in ["train", "validation", "test"]:
        usable, excluded = [], Counter()
        seen = set()
        for row in raw[split]:
            try:
                item = format_arc(row)
            except ValueError:
                excluded["invalid_or_non_four_choice"] += 1
                continue
            if item["id"] in seen:
                raise ValueError(f"Duplicate id in {split}: {item['id']}")
            seen.add(item["id"])
            ids = tokenizer.encode(item["prompt"], add_special_tokens=False)
            if len(ids) > config["max_length"]:
                excluded["over_max_length"] += 1
                continue
            item["input_ids"] = ids
            usable.append(item)
        random.Random(config["data_seed"]).shuffle(usable)
        count = config[f"{split}_size"]
        if count > len(usable) or count <= 0:
            raise ValueError(f"Requested {count} {split} items, only {len(usable)} eligible")
        selected = usable[:count]
        result[split] = selected
        raw_json = json.dumps(selected, sort_keys=True, ensure_ascii=False)
        manifest["splits"][split] = {
            "original_count": len(raw[split]), "eligible_count": len(usable),
            "excluded": dict(excluded), "selected_count": count,
            "ids": [r["id"] for r in selected],
            "label_counts": dict(Counter(r["label"] for r in selected)),
            "sha256": hashlib.sha256(raw_json.encode()).hexdigest()}
    for a, b in [("train", "validation"), ("train", "test"), ("validation", "test")]:
        if set(manifest["splits"][a]["ids"]) & set(manifest["splits"][b]["ids"]):
            raise ValueError(f"Overlapping ids in {a} and {b}")
        hashes_a = {hashlib.sha256(r["prompt"].encode()).hexdigest() for r in result[a]}
        hashes_b = {hashlib.sha256(r["prompt"].encode()).hexdigest() for r in result[b]}
        if hashes_a & hashes_b:
            raise ValueError(f"Duplicate prompts across {a} and {b}")
    return result, manifest


class Collator:
    def __init__(self, pad_token_id):
        self.pad_token_id = pad_token_id

    def __call__(self, rows):
        import torch
        width = max(len(r["input_ids"]) for r in rows)
        # Right padding and explicit last-valid-token indexing avoid padding bugs.
        ids = [r["input_ids"] + [self.pad_token_id] * (width - len(r["input_ids"])) for r in rows]
        masks = [[1] * len(r["input_ids"]) + [0] * (width - len(r["input_ids"])) for r in rows]
        return {"input_ids": torch.tensor(ids), "attention_mask": torch.tensor(masks),
                "labels": torch.tensor([r["label"] for r in rows])}
