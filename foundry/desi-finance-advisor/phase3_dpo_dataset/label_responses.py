"""
Use Gemini to label candidate responses as chosen / rejected for DPO training.
Reads data/dpo/candidates.jsonl → writes data/dpo/triplets.jsonl.

Run from repo root:
    export GOOGLE_API_KEY=...
    python foundry/desi-finance-advisor/phase3_dpo_dataset/label_responses.py
"""

import json
import os
import sys
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

from google import genai
from google.genai import types

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "dpo")
IN_FILE = os.path.join(DATA_DIR, "candidates.jsonl")
OUT_FILE = os.path.join(DATA_DIR, "triplets.jsonl")

MODEL = "gemini-2.5-flash-lite"

JUDGE_SYSTEM = """\
You are evaluating responses from an Indian personal finance advisor chatbot.
Your job: given a user question and multiple candidate responses, pick the BEST (chosen) and WORST (rejected).

Criteria for CHOSEN (all must be met):
1. Factually accurate — no misleading claims, no "guaranteed returns"
2. Mentions appropriate caveats ("consult a CA", "past returns not guaranteed")
3. Simple English with warm, conversational tone; light Hindi words are a plus
4. Actionable and specific, not vague generic advice
5. Appropriate length — not too terse, not rambling

Criteria for REJECTED (at least one serious flaw):
- Factually wrong or misleading (worst offense)
- Promises guaranteed returns or overstates certainty
- Pure jargon with no practical guidance
- Cold, robotic, or dismissive tone
- Completely off-topic

Return ONLY valid JSON: {"chosen": "<exact text>", "rejected": "<exact text>", "reason": "<1 sentence>"}"""

JUDGE_PROMPT = """\
User question (extracted from prompt):
{question}

Candidate responses:
{candidates}

Pick the best (chosen) and worst (rejected). Return JSON only."""

_lock = threading.Lock()
_counter = {"done": 0}


def extract_question(prompt: str) -> str:
    if "[INST]" in prompt:
        q = prompt.split("[INST]", 1)[-1].split("[/INST]")[0].strip()
        # Strip system prompt prefix if present
        if "\n\n" in q:
            q = q.split("\n\n", 1)[-1]
        return q
    return prompt[:200]


def label_one(client: genai.Client, record: dict, idx: int) -> dict | None:
    question = extract_question(record["prompt"])
    candidates = record["candidates"]
    numbered = "\n\n".join(f"[{i+1}] {c}" for i, c in enumerate(candidates))

    for attempt in range(4):
        try:
            r = client.models.generate_content(
                model=MODEL,
                contents=JUDGE_PROMPT.format(question=question, candidates=numbered),
                config=types.GenerateContentConfig(
                    system_instruction=JUDGE_SYSTEM,
                    max_output_tokens=600,
                    temperature=0.3,
                ),
            )
            raw = r.text.strip().lstrip("```json").lstrip("```").rstrip("```").strip()
            result = json.loads(raw)
            if "chosen" in result and "rejected" in result:
                if result["chosen"] == result["rejected"]:
                    return None
                triplet = {
                    "prompt": record["prompt"],
                    "chosen": result["chosen"],
                    "rejected": result["rejected"],
                }
                with _lock:
                    _counter["done"] += 1
                    print(f"  ✓ [{_counter['done']:4d}] reason: {result.get('reason', '')[:60]}")
                return triplet
        except json.JSONDecodeError:
            time.sleep(1)
        except Exception as e:
            err = str(e)
            if any(x in err for x in ["429", "503", "quota", "rate"]):
                time.sleep(4 * (2 ** attempt))
            else:
                print(f"  ✗ [{idx}] {err[:60]}")
                time.sleep(1)
    return None


def main():
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        print("ERROR: Set GOOGLE_API_KEY")
        sys.exit(1)

    if not os.path.exists(IN_FILE):
        print(f"ERROR: {IN_FILE} not found. Run generate_responses.py first.")
        sys.exit(1)

    with open(IN_FILE) as f:
        records = [json.loads(line) for line in f]
    print(f"Labeling {len(records)} candidate sets with Gemini ({MODEL})\n{'='*55}\n")

    client = genai.Client(api_key=api_key)

    triplets = []
    with ThreadPoolExecutor(max_workers=3) as pool:
        futures = {pool.submit(label_one, client, r, i): r
                   for i, r in enumerate(records, 1)}
        for fut in as_completed(futures):
            result = fut.result()
            if result:
                triplets.append(result)

    os.makedirs(DATA_DIR, exist_ok=True)
    with open(OUT_FILE, "w") as f:
        for t in triplets:
            f.write(json.dumps(t) + "\n")

    print(f"\n{'='*55}")
    print(f"DPO triplets: {len(triplets)} / {len(records)} → {OUT_FILE}")
    print("Next: python foundry/desi-finance-advisor/phase4_dpo/prepare_dpo_data.py")


if __name__ == "__main__":
    main()
