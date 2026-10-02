"""Split source groups before expansion, reject contamination, and write hashed manifests."""

import argparse
import json
from collections import Counter
from pathlib import Path

from datasets import load_from_disk

from decision_encoder.data import banking77, massive, typed_decisions
from decision_encoder.data.schema import assert_no_leakage, split_group, state_hash, write_jsonl
from decision_encoder.data.synthetic import generate_synthetic
from decision_encoder.runtime import sha256_file


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--synthetic-count", type=int, default=30000)
    p.add_argument("--include-test", action="store_true", help="Only after checkpoint selection")
    a = p.parse_args()
    sources = json.loads(Path("data/manifests/sources.json").read_text())
    splits = {s: [] for s in ["train", "validation", "calibration", "test"]}
    excluded = Counter()
    # Prioritize official tests; hash text only to exclude duplicates from development.
    test_hashes = set()
    loaded = {name: load_from_disk(f"data/raw/{name}") for name in sources}
    for name, data in loaded.items():
        field = {"banking77": "text", "massive": "utt", "typed_decisions": "state"}[name]
        for row in data["test"]:
            text = row[field]
            if name == "typed_decisions":
                text = json.dumps(json.loads(text), sort_keys=True, ensure_ascii=False)
            test_hashes.add(state_hash(text))
    # Same normalized state across public datasets shares split assignment.
    for name, data in loaded.items():
        taxonomy = None
        if name != "typed_decisions":
            taxonomy = json.loads(Path(f"data/taxonomies/{name}.json").read_text())
        for source_split, raw in data.items():
            if source_split == "test" and not a.include_test:
                continue
            seen = set()
            for index, row in enumerate(raw):
                text = row[{"banking77": "text", "massive": "utt", "typed_decisions": "state"}[name]]
                if name == "typed_decisions":
                    text = json.dumps(json.loads(text), sort_keys=True, ensure_ascii=False)
                group = state_hash(text)
                if source_split != "test" and group in test_hashes:
                    excluded[name] += 1
                    continue
                if group in seen:
                    excluded[name] += 1
                    continue
                seen.add(group)
                split = "test" if source_split == "test" else split_group(group, a.seed)
                if name == "typed_decisions":
                    records = typed_decisions.transform(row, source_split, a.seed)
                elif name == "banking77":
                    records = banking77.transform(
                        row, taxonomy, source_split, index, seed=a.seed, variants=1 if split != "train" else 2
                    )
                else:
                    records = massive.transform(
                        row,
                        taxonomy,
                        source_split,
                        index,
                        data["train"].features["intent"].names,
                        seed=a.seed,
                        variants=1 if split != "train" else 2,
                    )
                for record in records:
                    record["metadata"]["source_revision"] = sources[name]["revision"]
                    splits[split].append(record)
    for record in generate_synthetic(a.synthetic_count, a.seed, reserve_test=True):
        split = record["metadata"]["split"]
        if split != "test" or a.include_test:
            splits[split].append(record)
    assert_no_leakage(splits)
    manifest = {
        "schema_version": 1,
        "seed": a.seed,
        "sources": sources,
        "excluded_duplicates": dict(excluded),
        "synthetic_count": a.synthetic_count,
        "splits": {},
    }
    for split, rows in splits.items():
        if not rows:
            continue
        path = Path(f"data/processed/{split}.jsonl")
        write_jsonl(path, rows)
        manifest["splits"][split] = {
            "path": str(path),
            "sha256": sha256_file(path),
            "rows": len(rows),
            "sources": dict(Counter(r["source"] for r in rows)),
            "types": dict(Counter(r["decision_type"] for r in rows)),
            "groups": len(set((r["source"], r["group_id"]) for r in rows)),
        }
    Path("data/manifests/dataset.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest["splits"], indent=2))


if __name__ == "__main__":
    main()
