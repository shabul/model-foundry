"""
Evaluate Feynman fine-tune vs base model.

Runs both models on the same prompts, computes style metrics,
saves a markdown comparison report to eval/report_v{N}.md.

Run from repo root:
    python foundry/feynman-explainer/eval/evaluate.py
    python foundry/feynman-explainer/eval/evaluate.py --version 2
    python foundry/feynman-explainer/eval/evaluate.py --adapter foundry/feynman-explainer/adapters
"""

import argparse
import os
import re
import sys
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))

from mlx_lm import generate, load

BASE_MODEL    = "Qwen/Qwen2.5-3B-Instruct"
ADAPTER_PATH  = os.path.join(os.path.dirname(__file__), "..", "adapters")
EVAL_DIR      = os.path.dirname(__file__)

SYSTEM_PROMPT = (
    "You are a Feynman-style explainer. For every question, build intuition "
    "from the ground up using concrete analogies and everyday language. "
    "No jargon until it's earned. No bullet points. Pure flowing prose."
)

# Held-out prompts — NOT in training data, span all categories
EVAL_PROMPTS = [
    # ML
    "Why does a neural network need non-linear activation functions?",
    "What is the difference between a parameter and a hyperparameter?",
    "Why does more training data usually beat a better algorithm?",
    # Stats
    "What is the law of large numbers?",
    "What is a confidence interval?",
    # Physics
    "Why do objects fall at the same speed regardless of mass?",
    "What is the Doppler effect?",
    # Math
    "What is a logarithm and why is it useful?",
    "What is the chain rule in calculus?",
    # CS
    "What is Big O notation?",
    "What is a hash table?",
    # Biology
    "How does a vaccine train the immune system?",
    "What is pH?",
    # Economics
    "What is opportunity cost?",
    "What is inflation?",
]

FEYNMAN_KEYWORDS = [
    "imagine", "think of", "like a", "picture", "suppose",
    "here's the thing", "the key insight", "most people",
    "confused", "actually", "let me", "you've", "you're",
    "here's", "now here", "what's really", "the trick",
]

JARGON_WITHOUT_EXPLANATION = [
    "gradient", "matrix", "tensor", "derivative", "entropy",
    "probability", "distribution", "algorithm", "neural", "vector",
]


# --- Style metrics ---

def avg_sentence_length(text: str) -> float:
    sentences = re.split(r'[.!?]+', text)
    sentences = [s.strip() for s in sentences if len(s.strip()) > 10]
    if not sentences:
        return 0.0
    return sum(len(s.split()) for s in sentences) / len(sentences)


def flesch_reading_ease(text: str) -> float:
    """Higher = easier to read. Feynman should score 60+."""
    words = text.split()
    if not words:
        return 0.0
    sentences = max(1, len(re.split(r'[.!?]+', text)))
    syllables = sum(_count_syllables(w) for w in words)
    return 206.835 - 1.015 * (len(words) / sentences) - 84.6 * (syllables / len(words))


def _count_syllables(word: str) -> int:
    word = word.lower().strip(".,!?;:")
    if not word:
        return 1
    vowels = "aeiouy"
    count = sum(1 for i, c in enumerate(word) if c in vowels and (i == 0 or word[i-1] not in vowels))
    return max(1, count)


def analogy_density(text: str) -> float:
    """Feynman keyword hits per 100 words."""
    words = text.lower().split()
    if not words:
        return 0.0
    hits = sum(1 for kw in FEYNMAN_KEYWORDS if kw in text.lower())
    return hits / len(words) * 100


def avg_word_length(text: str) -> float:
    words = re.findall(r'\b[a-zA-Z]+\b', text)
    if not words:
        return 0.0
    return sum(len(w) for w in words) / len(words)


def score(text: str) -> dict:
    return {
        "avg_sentence_len":  round(avg_sentence_length(text), 1),
        "flesch_ease":       round(flesch_reading_ease(text), 1),
        "analogy_density":   round(analogy_density(text), 2),
        "avg_word_len":      round(avg_word_length(text), 2),
        "word_count":        len(text.split()),
    }


def feynman_composite(s: dict) -> float:
    """
    Higher is better Feynman style.
    Short sentences + high readability + high analogy density + short words.
    """
    sentence_score = max(0, 20 - s["avg_sentence_len"]) * 2   # want ~12-15 words/sentence
    flesch_score   = min(s["flesch_ease"], 100) / 100 * 40    # want 60-80+
    analogy_score  = min(s["analogy_density"], 2) / 2 * 30    # want >0.5 per 100 words
    word_score     = max(0, 6 - s["avg_word_len"]) * 2        # want shorter words
    return round(sentence_score + flesch_score + analogy_score + word_score, 1)


# --- Inference ---

def infer(model, tokenizer, question: str, max_tokens: int = 350) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": question},
    ]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return generate(model, tokenizer, prompt=prompt, max_tokens=max_tokens, verbose=False)


# --- Report ---

def render_report(results: list[dict], version: int) -> str:
    lines = []
    lines.append(f"# Feynman Explainer — Eval Report v{version}")
    lines.append(f"*Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}*\n")

    # Aggregate metrics
    base_scores  = [r["base_score"]  for r in results]
    ft_scores    = [r["ft_score"]    for r in results]

    def avg(lst, key): return round(sum(d[key] for d in lst) / len(lst), 1)

    lines.append("## Aggregate Style Metrics\n")
    lines.append("| Metric | Base Model | Fine-tuned | Δ | Target |")
    lines.append("|--------|-----------|------------|---|--------|")

    metrics = [
        ("Feynman composite ↑",  "composite",       "higher=better"),
        ("Flesch reading ease ↑","flesch_ease",     "60–80"),
        ("Analogy density ↑",    "analogy_density", ">0.5"),
        ("Avg sentence length ↓","avg_sentence_len","12–16 words"),
        ("Avg word length ↓",    "avg_word_len",    "<5.0 chars"),
    ]

    for label, key, target in metrics:
        if key == "composite":
            b = round(sum(feynman_composite(r["base_score"]) for r in results) / len(results), 1)
            f = round(sum(feynman_composite(r["ft_score"])   for r in results) / len(results), 1)
        else:
            b = avg(base_scores, key)
            f = avg(ft_scores,   key)
        delta = round(f - b, 1)
        sign = "+" if delta > 0 else ""
        lines.append(f"| {label} | {b} | {f} | {sign}{delta} | {target} |")

    lines.append("")

    # Per-prompt comparison (3 examples shown in full)
    lines.append("## Response Comparisons (sample)\n")
    for r in results[:3]:
        lines.append(f"### Q: {r['prompt']}\n")
        lines.append("**Base model:**")
        lines.append(f"> {r['base'].replace(chr(10), chr(10)+'> ')}\n")
        lines.append("**Fine-tuned:**")
        lines.append(f"> {r['ft'].replace(chr(10), chr(10)+'> ')}\n")

        b = r["base_score"]
        f = r["ft_score"]
        lines.append(
            f"*Composite: base={feynman_composite(b)} → fine-tuned={feynman_composite(f)} "
            f"| Flesch: {b['flesch_ease']} → {f['flesch_ease']} "
            f"| AvgSentLen: {b['avg_sentence_len']} → {f['avg_sentence_len']}*\n"
        )
        lines.append("---\n")

    # Verdict
    avg_base_composite = sum(feynman_composite(r["base_score"]) for r in results) / len(results)
    avg_ft_composite   = sum(feynman_composite(r["ft_score"])   for r in results) / len(results)
    improvement = round((avg_ft_composite - avg_base_composite) / max(avg_base_composite, 1) * 100, 1)

    lines.append("## Verdict\n")
    if improvement >= 20:
        verdict = f"✅ Strong style shift (+{improvement}% composite). Model card claims are justified."
        retrain = "No retraining needed."
    elif improvement >= 8:
        verdict = f"⚠️ Moderate style shift (+{improvement}% composite). Acceptable but improvable."
        retrain = "Consider: increase `iters` to 2000, bump `learning_rate` to 3e-4."
    else:
        verdict = f"❌ Weak style shift (+{improvement}% composite). Retraining recommended."
        retrain = "Recommended: increase LoRA rank to 32, iters to 2500, learning_rate to 3e-4."

    lines.append(verdict)
    lines.append(f"\n**Retraining recommendation:** {retrain}")
    lines.append(f"\n**Avg composite score:** base={round(avg_base_composite,1)} → fine-tuned={round(avg_ft_composite,1)}")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--version",  type=int, default=1, help="Report version number")
    parser.add_argument("--adapter",  default=ADAPTER_PATH, help="Path to LoRA adapter dir")
    parser.add_argument("--prompts",  type=int, default=len(EVAL_PROMPTS), help="How many prompts to eval")
    parser.add_argument("--max-tokens", type=int, default=350)
    args = parser.parse_args()

    prompts = EVAL_PROMPTS[:args.prompts]

    print(f"\nLoading base model: {BASE_MODEL}")
    base_model, base_tok = load(BASE_MODEL)

    print(f"Loading fine-tuned model (adapter: {args.adapter})")
    ft_model, ft_tok = load(BASE_MODEL, adapter_path=args.adapter)

    print(f"\nEvaluating {len(prompts)} prompts...\n")
    results = []

    for i, prompt in enumerate(prompts, 1):
        print(f"  [{i:2d}/{len(prompts)}] {prompt[:60]}")
        base_resp = infer(base_model, base_tok, prompt, args.max_tokens)
        ft_resp   = infer(ft_model,   ft_tok,   prompt, args.max_tokens)
        results.append({
            "prompt":     prompt,
            "base":       base_resp,
            "ft":         ft_resp,
            "base_score": score(base_resp),
            "ft_score":   score(ft_resp),
        })

    report = render_report(results, args.version)
    report_path = os.path.join(EVAL_DIR, f"report_v{args.version}.md")
    with open(report_path, "w") as f:
        f.write(report)

    print(f"\n{'='*60}")
    print(report)
    print(f"\nReport saved → {report_path}")


if __name__ == "__main__":
    main()
