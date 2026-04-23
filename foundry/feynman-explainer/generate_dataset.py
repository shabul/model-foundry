"""
Generate a Feynman-style explanation dataset using Google Gemini.

Strategy:
  - Pull questions from yahma/alpaca-cleaned (public HF dataset)
  - Rewrite answers in Feynman's voice using gemini-2.0-flash-lite (fast + cheap)
  - Also generate fresh examples from a curated topic list
  - Write to foundry/feynman-explainer/data/raw_feynman.jsonl

Run from repo root:
    export GOOGLE_API_KEY=...
    python foundry/feynman-explainer/generate_dataset.py
    python foundry/feynman-explainer/generate_dataset.py --n-alpaca 300 --n-custom 200
"""

import argparse
import json
import os
import sys
import time

from google import genai
from google.genai import types
from datasets import load_dataset

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

MODEL = "gemini-2.5-flash-lite"
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
OUT_FILE = os.path.join(DATA_DIR, "raw_feynman.jsonl")

FEYNMAN_SYSTEM = """You are Richard Feynman — the Nobel-winning physicist famous for making
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
- NO bullet points. NO headers. NO markdown formatting. Pure flowing prose.
- Length: 150–300 words. Dense with insight, never padded."""

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
    "What is a random forest and why is it better than one tree?",
    "Why does the learning rate matter so much in training?",
    "What is backpropagation and how does it work?",
    "Why do transformers use multi-head attention?",
    "What is tokenization and why does it matter for LLMs?",
    "How does K-means clustering work?",
    "What is a confusion matrix and what does it tell you?",
    "Why do we normalize data before training?",
    "What is the vanishing gradient problem?",
    "How does dropout prevent overfitting?",
]

ALPACA_FILTER_SKIP = [
    "write a", "create a", "generate", "list ", "give me a list",
    "translate", "summarize", "classify", "code", "python",
    "function", "script", "program", "rewrite", "edit ", "fix ",
]
ALPACA_FILTER_KEEP = [
    "explain", "what is", "how does", "why does", "why is", "describe",
    "what are", "how do", "what makes", "tell me about", "what causes",
]


def is_explanation_question(instruction: str) -> bool:
    low = instruction.lower()
    if len(instruction) > 200:
        return False
    if any(kw in low for kw in ALPACA_FILTER_SKIP):
        return False
    return any(kw in low for kw in ALPACA_FILTER_KEEP)


def generate_feynman(client: genai.Client, question: str, retries: int = 3) -> str | None:
    for attempt in range(retries):
        try:
            response = client.models.generate_content(
                model=MODEL,
                contents=question,
                config=types.GenerateContentConfig(
                    system_instruction=FEYNMAN_SYSTEM,
                    max_output_tokens=600,
                    temperature=0.85,
                ),
            )
            return response.text.strip()
        except Exception as e:
            err = str(e).lower()
            if "quota" in err or "rate" in err or "429" in err:
                wait = 2 ** attempt * 5
                print(f"  Rate limited, waiting {wait}s...")
                time.sleep(wait)
            else:
                print(f"  Error attempt {attempt + 1}: {e}")
                time.sleep(2)
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-alpaca", type=int, default=400)
    parser.add_argument("--n-custom", type=int, default=len(CUSTOM_TOPICS))
    args = parser.parse_args()

    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        print("ERROR: Set GOOGLE_API_KEY environment variable.")
        sys.exit(1)

    client = genai.Client(api_key=api_key)
    os.makedirs(DATA_DIR, exist_ok=True)

    records = []

    # --- Custom curated topics ---
    topics = CUSTOM_TOPICS[:args.n_custom]
    print(f"\n[1/2] Generating {len(topics)} custom Feynman examples via {MODEL}...")
    for i, topic in enumerate(topics, 1):
        print(f"  {i:3d}/{len(topics)}: {topic[:65]}")
        answer = generate_feynman(client, topic)
        if answer:
            records.append({"instruction": topic, "response": answer, "source": "custom"})

    # --- Alpaca-cleaned questions ---
    print(f"\n[2/2] Pulling explanation questions from yahma/alpaca-cleaned...")
    alpaca = load_dataset("yahma/alpaca-cleaned", split="train")
    candidates = [ex for ex in alpaca if is_explanation_question(ex["instruction"])]
    candidates = candidates[:args.n_alpaca]
    print(f"  Using {len(candidates)} questions")

    for i, ex in enumerate(candidates, 1):
        q = ex["instruction"]
        if ex.get("input", "").strip():
            q = f"{q}\n\n{ex['input'].strip()}"
        print(f"  {i:3d}/{len(candidates)}: {q[:65]}")
        answer = generate_feynman(client, q)
        if answer:
            records.append({"instruction": q, "response": answer, "source": "alpaca"})

    with open(OUT_FILE, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")

    custom_ct = sum(1 for r in records if r["source"] == "custom")
    alpaca_ct = sum(1 for r in records if r["source"] == "alpaca")
    print(f"\nDone. {len(records)} examples → {OUT_FILE}")
    print(f"  Custom : {custom_ct}")
    print(f"  Alpaca : {alpaca_ct}")
    print(f"\nNext: python foundry/feynman-explainer/prepare_data.py")


if __name__ == "__main__":
    main()
