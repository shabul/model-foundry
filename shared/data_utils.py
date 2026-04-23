"""
Shared data formatting utilities for all foundry projects.
"""

import json
import os
import random


def format_chat_example(example: dict, tokenizer, instruction_key="instruction",
                         context_key="context", response_key="response") -> str:
    instruction = example[instruction_key].strip()
    context = example.get(context_key, "").strip()
    response = example[response_key].strip()

    user_content = f"{instruction}\n\n{context}" if context else instruction
    messages = [
        {"role": "user", "content": user_content},
        {"role": "assistant", "content": response},
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)


def write_jsonl(records: list[dict], path: str):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        for record in records:
            f.write(json.dumps(record) + "\n")
    print(f"Wrote {len(records):,} records → {path}")


def train_val_split(examples: list, train_ratio=0.9, seed=42):
    random.seed(seed)
    data = list(examples)
    random.shuffle(data)
    split = int(len(data) * train_ratio)
    return data[:split], data[split:]
