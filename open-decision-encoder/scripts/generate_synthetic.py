import argparse

from decision_encoder.data.schema import write_jsonl
from decision_encoder.data.synthetic import generate_synthetic

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--count", type=int, default=30000)
    p.add_argument("--output", default="data/processed/synthetic.jsonl")
    a = p.parse_args()
    write_jsonl(a.output, generate_synthetic(a.count, a.seed))
