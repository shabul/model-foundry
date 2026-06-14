"""
Generate 5 candidate responses per training prompt using the SFT model.
Output goes to data/dpo/candidates.jsonl for Gemini labeling.

Run from repo root:
    python foundry/desi-finance-advisor/phase3_dpo_dataset/generate_responses.py
"""

import json
import os
import sys

from mlx_lm import load, generate

SFT_MODEL = "mlx-community/Mistral-7B-Instruct-v0.2-4bit"
SFT_ADAPTER = os.path.join(os.path.dirname(__file__), "..", "adapters", "sft")

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
SFT_TRAIN = os.path.join(DATA_DIR, "sft", "train.jsonl")
OUT_FILE = os.path.join(DATA_DIR, "dpo", "candidates.jsonl")

N_RESPONSES = 5
MAX_TOKENS = 400
TEMPERATURE = 0.9

# Sample a subset — DPO labeling is expensive; 200 triplets is solid
MAX_PROMPTS = 200


def strip_response(text: str, prompt: str) -> str:
    if text.startswith(prompt):
        text = text[len(prompt):]
    # Mistral adds [/INST] end token — strip anything before the model reply
    if "[/INST]" in text:
        text = text.split("[/INST]", 1)[-1]
    return text.strip().rstrip("</s>").strip()


def main():
    if not os.path.isdir(SFT_ADAPTER):
        print(f"ERROR: SFT adapter not found at {SFT_ADAPTER}")
        print("Run phase2_sft/train_sft.py first.")
        sys.exit(1)

    print(f"Loading SFT model: {SFT_MODEL}")
    print(f"Adapter          : {SFT_ADAPTER}\n")
    model, tokenizer = load(SFT_MODEL, adapter_path=SFT_ADAPTER)

    with open(SFT_TRAIN) as f:
        examples = [json.loads(line) for line in f][:MAX_PROMPTS]
    print(f"Prompts to process: {len(examples)}")

    os.makedirs(os.path.dirname(OUT_FILE), exist_ok=True)
    written = 0

    with open(OUT_FILE, "w") as out:
        for i, ex in enumerate(examples, 1):
            prompt_text = ex["text"]
            # Extract just the [INST] prompt part (up to and including [/INST])
            if "[/INST]" in prompt_text:
                prompt_only = prompt_text.split("[/INST]")[0] + "[/INST]"
            else:
                prompt_only = prompt_text

            candidates = []
            for _ in range(N_RESPONSES):
                output = generate(
                    model, tokenizer,
                    prompt=prompt_only,
                    max_tokens=MAX_TOKENS,
                    temp=TEMPERATURE,
                    verbose=False,
                )
                cleaned = strip_response(output, prompt_only)
                if len(cleaned) > 60:
                    candidates.append(cleaned)

            if len(candidates) >= 2:
                record = {"prompt": prompt_only, "candidates": candidates}
                out.write(json.dumps(record) + "\n")
                written += 1
                print(f"  [{i:4d}/{len(examples)}] {len(candidates)} responses generated")

    print(f"\nSaved {written} prompts with candidates → {OUT_FILE}")
    print("Next: python foundry/desi-finance-advisor/phase3_dpo_dataset/label_responses.py")


if __name__ == "__main__":
    main()
