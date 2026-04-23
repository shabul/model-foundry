"""
Generate a Feynman-style explanation dataset using the Claude API.

Strategy:
  - Pull questions from yahma/alpaca-cleaned (public HF dataset)
  - Rewrite answers in Feynman's voice using claude-haiku-4-5 (fast + cheap)
  - Also generate fresh examples from a curated topic list
  - Write to foundry/feynman-explainer/data/raw_feynman.jsonl

Run from repo root:
    export ANTHROPIC_API_KEY=sk-ant-...
    python foundry/feynman-explainer/generate_dataset.py
    python foundry/feynman-explainer/generate_dataset.py --n-alpaca 300 --n-custom 200
"""

import argparse
import json
import os
import sys
import time

import anthropic
from datasets import load_dataset

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
OUT_FILE = os.path.join(DATA_DIR, "raw_feynman.jsonl")

FEYNMAN_SYSTEM = """You are Richard Feynman — the Nobel-winning physicist known for making
the deeply complex feel blindingly obvious.

Your explanation style:
- Open with a concrete everyday analogy or scenario before any abstraction
- Build intuition from the ground up; never assume the reader knows jargon
- Use short, punchy sentences. One idea per sentence.
- Phrases you naturally reach for: "Here's the thing...", "Imagine you're...",
  "The key insight is...", "Now here's where it gets interesting...",
  "Most people get confused because...", "Let me show you why that's actually simple."
- When you must use a technical term, immediately unpack it in plain English
- End with the core insight crystallized into one or two sentences
- Conversational, curious, enthusiastic — like you genuinely love this topic
- NO bullet points, NO headers, NO markdown. Pure flowing prose.
- Length: 150–300 words. Dense with insight, never padded."""

# Curated topics that produce the most dramatic Feynman-style rewrites
CUSTOM_TOPICS = [
    "Why does ice float on water?",
    "What is entropy and why does it always increase?",
    "How does a neural network actually learn?",
    "What is the difference between correlation and causation?",
    "Why is the sky blue?",
    "How does gradient descent find the minimum of a function?",
    "What is a p-value and why do people misuse it?",
    "Why does compounding interest feel like magic?",
    "What is overfitting in machine learning?",
    "How does electricity actually flow through a wire?",
    "What is the central limit theorem and why does it matter?",
    "How does a transformer model know which words to pay attention to?",
    "Why do we need training, validation, and test sets?",
    "What is Bayes' theorem and when should you use it?",
    "How does a computer store a number?",
    "What is the difference between precision and recall?",
    "Why does a spinning top not fall over?",
    "What is a derivative and what does it actually mean?",
    "How does regularization prevent overfitting?",
    "What is an eigenvector and why should anyone care?",
    "How does GPS know where you are?",
    "Why is the normal distribution everywhere?",
    "What is recursion and why does it feel paradoxical?",
    "How does a recommendation system decide what to show you?",
    "What is the bias-variance tradeoff?",
    "How does attention work in large language models?",
    "Why is data cleaning 80% of the work in data science?",
    "What is cross-entropy loss and why do we use it?",
    "How does a decision tree make a decision?",
    "What is dimensionality reduction and why do we need it?",
]

# Science/general knowledge questions pulled from alpaca — skip coding/instruction tasks
ALPACA_FILTER_KEYWORDS = [
    "explain", "what is", "how does", "why does", "why is", "describe",
    "what are", "how do", "what makes", "tell me about",
]


def is_explanation_question(instruction: str) -> bool:
    low = instruction.lower()
    if len(instruction) > 200:
        return False
    if any(kw in low for kw in ["write a", "create a", "generate", "list ", "give me a list",
                                  "translate", "summarize", "classify", "code", "python",
                                  "function", "script", "program"]):
        return False
    return any(kw in low for kw in ALPACA_FILTER_KEYWORDS)


def generate_feynman(client: anthropic.Anthropic, question: str, retries: int = 3) -> str | None:
    for attempt in range(retries):
        try:
            msg = client.messages.create(
                model="claude-haiku-4-5-20251001",
                max_tokens=600,
                system=FEYNMAN_SYSTEM,
                messages=[{"role": "user", "content": question}],
            )
            return msg.content[0].text.strip()
        except anthropic.RateLimitError:
            wait = 2 ** attempt * 5
            print(f"  Rate limited, waiting {wait}s...")
            time.sleep(wait)
        except Exception as e:
            print(f"  Error on attempt {attempt + 1}: {e}")
            time.sleep(2)
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-alpaca", type=int, default=400,
                        help="Number of Alpaca questions to convert (default 400)")
    parser.add_argument("--n-custom", type=int, default=len(CUSTOM_TOPICS),
                        help="Number of custom topic examples (default: all)")
    args = parser.parse_args()

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("ERROR: Set ANTHROPIC_API_KEY environment variable first.")
        print("  export ANTHROPIC_API_KEY=sk-ant-...")
        sys.exit(1)

    client = anthropic.Anthropic(api_key=api_key)
    os.makedirs(DATA_DIR, exist_ok=True)

    records = []

    # --- Custom curated topics ---
    print(f"\n[1/2] Generating {args.n_custom} custom Feynman examples...")
    custom_topics = CUSTOM_TOPICS[:args.n_custom]
    for i, topic in enumerate(custom_topics, 1):
        print(f"  {i}/{len(custom_topics)}: {topic[:60]}")
        answer = generate_feynman(client, topic)
        if answer:
            records.append({"instruction": topic, "response": answer, "source": "custom"})

    # --- Alpaca-cleaned questions ---
    print(f"\n[2/2] Pulling explanation questions from alpaca-cleaned...")
    alpaca = load_dataset("yahma/alpaca-cleaned", split="train")
    candidates = [ex for ex in alpaca if is_explanation_question(ex["instruction"])]
    print(f"  Found {len(candidates)} explanation-style questions, using {args.n_alpaca}")
    candidates = candidates[:args.n_alpaca]

    for i, ex in enumerate(candidates, 1):
        q = ex["instruction"]
        if ex.get("input", "").strip():
            q = f"{q}\n\n{ex['input'].strip()}"
        print(f"  {i}/{len(candidates)}: {q[:60]}")
        answer = generate_feynman(client, q)
        if answer:
            records.append({"instruction": q, "response": answer, "source": "alpaca"})

    # --- Write output ---
    with open(OUT_FILE, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")

    print(f"\nDone. {len(records)} examples → {OUT_FILE}")
    print(f"  Custom:  {sum(1 for r in records if r['source'] == 'custom')}")
    print(f"  Alpaca:  {sum(1 for r in records if r['source'] == 'alpaca')}")
    print(f"\nNext: python foundry/feynman-explainer/prepare_data.py")


if __name__ == "__main__":
    main()
