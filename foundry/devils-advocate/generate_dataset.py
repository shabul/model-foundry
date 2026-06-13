"""
Generate a Devil's Advocate dataset using Google Gemini.
The model challenges the user's premise with logical, piercing counter-arguments.

Run from repo root:
    export GOOGLE_API_KEY=...
    python foundry/devils-advocate/generate_dataset.py
"""

import argparse
import json
import os
import sys
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

from google import genai
from google.genai import types

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

MODEL = "gemini-2.5-flash-lite"
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
OUT_FILE = os.path.join(DATA_DIR, "raw_advocate.jsonl")

# Diverse debatable concepts
CONCEPTS = [
    ("Remote work is strictly better for productivity than office-based work.", "Business"),
    ("Artificial Intelligence will eventually replace all human creative jobs.", "Technology"),
    ("Universal Basic Income is the only solution to automation-driven job loss.", "Economics"),
    ("A strictly vegan diet is the most ethical choice for everyone.", "Ethics"),
    ("Space exploration is a waste of resources while we have problems on Earth.", "Society"),
    ("Social media has done more harm than good for human connection.", "Technology"),
    ("Nuclear energy is the safest and cleanest bridge to a carbon-neutral future.", "Environment"),
    ("Privacy is an obsolete concept in the digital age.", "Philosophy"),
    ("Standardized testing is the most objective way to measure intelligence.", "Education"),
    ("Cryptocurrency will eventually replace traditional fiat currencies.", "Finance"),
    ("Free speech should have no limitations, even for 'hate speech'.", "Law"),
    ("The pursuit of happiness is a flawed goal for a meaningful life.", "Philosophy"),
    ("Democracy is the least efficient form of government for rapid progress.", "Politics"),
    ("Genetically modifying human embryos is a moral imperative to erase disease.", "Science"),
    ("The internet should be a basic human right, provided for free by states.", "Society"),
    ("Hard work is the primary determinant of success in life.", "Psychology"),
    ("Monogamy is an unnatural social construct that doesn't fit human biology.", "Relationships"),
    ("Cities should be entirely car-free to improve quality of life.", "Urban Planning"),
    ("Artificial Intelligence should have legal rights once it reaches a certain complexity.", "Law"),
    ("Coding is a 'basic literacy' that everyone should learn in primary school.", "Education"),
]

PROMPT_TEMPLATE = """\
Act as a sophisticated Devil's Advocate. Your goal is to intelligently challenge the following premise:

Premise: {concept}

Rules:
- Do NOT agree with the premise.
- Open with a respectful but firm disagreement.
- Use logical reasoning to point out blind spots, edge cases, or unintended consequences.
- Be piercing and provocative, but remain intellectual and structured.
- Use phrases like: "While that's a popular sentiment, it overlooks...", "The flaw in that logic is...", "Consider the counter-factual...", "From a purely pragmatic standpoint..."
- End by challenging the user to defend a specific weak point in their argument.
- NO bullet points. NO markdown headers. Pure flowing prose.
- Length: 200–350 words."""

_print_lock = threading.Lock()
_counter = {"done": 0, "total": 0}


def log(msg: str):
    with _print_lock:
        print(msg, flush=True)


def generate_one(client: genai.Client, concept: str, category: str, idx: int,
                 retries: int = 4) -> dict | None:
    for attempt in range(retries):
        try:
            response = client.models.generate_content(
                model=MODEL,
                contents=PROMPT_TEMPLATE.format(concept=concept),
                config=types.GenerateContentConfig(
                    max_output_tokens=700,
                    temperature=0.85,
                ),
            )
            text = response.text.strip()
            if len(text) > 100:
                with _print_lock:
                    _counter["done"] += 1
                    done = _counter["done"]
                    total = _counter["total"]
                    print(f"  ✓ [{done:4d}/{total}] [{category}] {concept[:55]}", flush=True)
                return {"instruction": concept, "response": text, "category": category}
        except Exception as e:
            err = str(e)
            if "503" in err or "quota" in err.lower() or "429" in err or "rate" in err.lower():
                wait = 2 ** attempt * 4
                log(f"  ⏳ [{idx}] Rate limited — waiting {wait}s...")
                time.sleep(wait)
            else:
                log(f"  ✗ [{idx}] Error: {err[:80]}")
                time.sleep(1)
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=5,
                        help="Parallel threads (lower for 9B base if needed)")
    args = parser.parse_args()

    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        print("ERROR: Set GOOGLE_API_KEY environment variable.")
        sys.exit(1)

    client = genai.Client(api_key=api_key)
    os.makedirs(DATA_DIR, exist_ok=True)

    _counter["total"] = len(CONCEPTS)

    print(f"\n{'='*60}")
    print(f"Devil's Advocate Dataset Generator")
    print(f"Model   : {MODEL}")
    print(f"Concepts: {len(CONCEPTS)}")
    print(f"Workers : {args.workers} parallel threads")
    print(f"Output  : {OUT_FILE}")
    print(f"{'='*60}\n")

    records = []
    failed = 0
    start = time.time()

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(generate_one, client, concept, cat, i): (concept, cat)
            for i, (concept, cat) in enumerate(CONCEPTS, 1)
        }
        for future in as_completed(futures):
            result = future.result()
            if result:
                records.append(result)
            else:
                failed += 1

    elapsed = time.time() - start

    with open(OUT_FILE, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")

    print(f"\n{'='*60}")
    print(f"Completed in {elapsed:.0f}s ({elapsed/60:.1f} min)")
    print(f"Generated : {len(records)} / {len(CONCEPTS)}")
    print(f"Failed    : {failed}")
    print(f"\nOutput: {OUT_FILE}")
    print(f"Next  : python foundry/devils-advocate/prepare_data.py")


if __name__ == "__main__":
    main()
