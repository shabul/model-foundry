"""
Generate a Sherlock Holmes Debugger dataset using Google Gemini.
The model analyzes code bugs as 'crime scenes' and uses deductive reasoning to solve them.

Run from repo root:
    export GOOGLE_API_KEY=...
    python foundry/sherlock-debugger/generate_dataset.py
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
OUT_FILE = os.path.join(DATA_DIR, "raw_sherlock.jsonl")

# Code 'Crime Scenes' (Snippet + Error/Bug)
CASES = [
    ("Python: List index out of range in a for loop using range(len(arr) + 1).", "Python"),
    ("JavaScript: 'undefined' is not a function when calling a method on an object returned from an API.", "JS"),
    ("Python: Memory limit exceeded when reading a 10GB file using file.read().", "Python"),
    ("C++: Segmentation fault when accessing a pointer that was previously deleted.", "C++"),
    ("SQL: Query taking 30 seconds to run on a table with 1 million rows and no indexes.", "SQL"),
    ("React: Component not re-rendering after a state variable is mutated directly (state.value = x).", "React"),
    ("Python: UnboundLocalError when accessing a variable inside a function that is later assigned a value.", "Python"),
    ("CSS: Element with z-index: 999 is appearing behind an element with z-index: 1.", "CSS"),
    ("Docker: 'Connection refused' when trying to connect from one container to another using 'localhost'.", "DevOps"),
    ("Go: Panic when sending to a closed channel.", "Go"),
    ("Rust: Borrow checker error when trying to mutate a value while a shared reference exists.", "Rust"),
    ("Python: Variable 'time' being shadowed by 'import time' inside a function.", "Python"),
    ("Node.js: UnhandledPromiseRejectionWarning because a .catch() block is missing.", "JS"),
    ("Java: NullPointerException when calling a method on an object that was never initialized.", "Java"),
    ("Git: 'Merge conflict' in a file where both users changed the same line.", "Git"),
    ("Python: Dictionary key error when accessing a key that is actually a string instead of an integer.", "Python"),
    ("TypeScript: 'Property does not exist on type' when accessing a valid property on an object typed as 'any'.", "TS"),
    ("C#: Race condition where two threads increment a shared counter without a lock.", "C#"),
    ("Python: IndentationError after copying code from a website that used tabs instead of spaces.", "Python"),
    ("HTML: Image not loading because the 'src' path is relative to the wrong directory.", "HTML"),
]

PROMPT_TEMPLATE = """\
Act as Sherlock Holmes, the world's greatest consulting debugger. You are presented with a code "crime scene".

Crime Scene Description: {concept}

Rules:
- Adopt the persona of Sherlock Holmes: observant, deductive, slightly eccentric, and brilliant.
- Start by "examining the scene" (analyzing the provided bug/error).
- Use deductive reasoning to identify the "culprit" (the root cause).
- Use phrases like: "You see, but you do not observe...", "Elementary, my dear coder...", "When you have eliminated the impossible...", "The game is afoot!", "Observe the curious incident of the variable in the night-time..."
- Provide the solution/fix clearly but within the narrative.
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
                    temperature=0.8,
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
    parser.add_argument("--workers", type=int, default=5)
    args = parser.parse_args()

    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        print("ERROR: Set GOOGLE_API_KEY environment variable.")
        sys.exit(1)

    client = genai.Client(api_key=api_key)
    os.makedirs(DATA_DIR, exist_ok=True)

    _counter["total"] = len(CASES)

    print(f"\n{'='*60}")
    print(f"Sherlock Debugger Dataset Generator")
    print(f"Model   : {MODEL}")
    print(f"Cases   : {len(CASES)}")
    print(f"Workers : {args.workers} parallel threads")
    print(f"Output  : {OUT_FILE}")
    print(f"{'='*60}\n")

    records = []
    failed = 0
    start = time.time()

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(generate_one, client, concept, cat, i): (concept, cat)
            for i, (concept, cat) in enumerate(CASES, 1)
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
    print(f"Generated : {len(records)} / {len(CASES)}")
    print(f"Failed    : {failed}")
    print(f"\nOutput: {OUT_FILE}")
    print(f"Next  : python foundry/sherlock-debugger/prepare_data.py")


if __name__ == "__main__":
    main()
