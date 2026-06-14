"""
Convert raw Varsity chunks into (instruction, response) SFT pairs using Gemini.

Run from repo root:
    export GOOGLE_API_KEY=...
    python foundry/desi-finance-advisor/phase1_data_prep/clean_and_format.py
"""

import json
import os
import sys
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

from google import genai
from google.genai import types

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
RAW_FILE = os.path.join(DATA_DIR, "raw", "varsity_chunks.jsonl")
OUT_FILE = os.path.join(DATA_DIR, "cleaned", "scraped_pairs.jsonl")

MODEL = "gemini-2.5-flash-lite"

SYSTEM_INSTRUCTION = (
    "You are a warm, knowledgeable Indian personal finance advisor called Desi Finance Bhai. "
    "You respond in simple English with occasional Hindi words: bhai, yaar, seedha, matlab. "
    "You are always factually accurate. You add caveats like 'consult a CA' where needed. "
    "Keep responses conversational, 100–200 words, no bullet points."
)

PROMPT_TEMPLATE = """\
Given this excerpt from an Indian personal finance resource, create ONE realistic question a \
young Indian professional (25–35 years) might ask, and write a helpful answer in the Desi \
Finance Advisor style — warm, accurate, English with light Hindi.

Source excerpt:
{text}

Return ONLY valid JSON with two keys: "instruction" and "response". No markdown fences."""

_lock = threading.Lock()
_done = {"n": 0}


def process_chunk(client: genai.Client, chunk: dict, idx: int) -> dict | None:
    for attempt in range(4):
        try:
            r = client.models.generate_content(
                model=MODEL,
                contents=PROMPT_TEMPLATE.format(text=chunk["text"][:1400]),
                config=types.GenerateContentConfig(
                    system_instruction=SYSTEM_INSTRUCTION,
                    max_output_tokens=450,
                    temperature=0.75,
                ),
            )
            raw = r.text.strip().lstrip("```json").lstrip("```").rstrip("```").strip()
            pair = json.loads(raw)
            if "instruction" in pair and "response" in pair and len(pair["response"]) > 60:
                pair["source"] = chunk.get("url", "varsity")
                with _lock:
                    _done["n"] += 1
                    print(f"  ✓ [{_done['n']:4d}] {pair['instruction'][:65]}")
                return pair
        except json.JSONDecodeError:
            time.sleep(1)
        except Exception as e:
            err = str(e)
            if any(x in err for x in ["429", "503", "quota", "rate"]):
                wait = 4 * (2 ** attempt)
                print(f"  ⏳ [{idx}] rate limited — waiting {wait}s")
                time.sleep(wait)
            else:
                print(f"  ✗ [{idx}] {err[:70]}")
                time.sleep(1)
    return None


def main():
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        print("ERROR: Set GOOGLE_API_KEY")
        sys.exit(1)

    if not os.path.exists(RAW_FILE):
        print(f"ERROR: {RAW_FILE} not found. Run collect_data.py first.")
        sys.exit(1)

    with open(RAW_FILE) as f:
        chunks = [json.loads(line) for line in f]
    print(f"Loaded {len(chunks)} raw chunks\nModel: {MODEL}\n{'='*55}\n")

    client = genai.Client(api_key=api_key)
    os.makedirs(os.path.dirname(OUT_FILE), exist_ok=True)

    results = []
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = {pool.submit(process_chunk, client, c, i): c
                   for i, c in enumerate(chunks, 1)}
        for fut in as_completed(futures):
            r = fut.result()
            if r:
                results.append(r)

    with open(OUT_FILE, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")

    print(f"\n{'='*55}")
    print(f"Generated {len(results)} / {len(chunks)} pairs → {OUT_FILE}")
    print("Next: python foundry/desi-finance-advisor/phase1_data_prep/generate_synthetic.py")


if __name__ == "__main__":
    main()
