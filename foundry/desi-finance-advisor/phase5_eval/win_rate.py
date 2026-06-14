"""
Win-rate evaluation: Gemini judges SFT responses vs RSFT (aligned) responses side-by-side.
Reports what % of the time the RSFT model wins over the base SFT model.

Run from repo root:
    export GOOGLE_API_KEY=...
    python foundry/desi-finance-advisor/phase5_eval/win_rate.py
"""

import json
import os
import random
import sys
import time

from google import genai
from google.genai import types
from mlx_lm import load, generate
from mlx_lm.sample_utils import make_sampler

BASE_MODEL = "mlx-community/Mistral-7B-Instruct-v0.2-4bit"
SFT_ADAPTER  = os.path.join(os.path.dirname(__file__), "..", "adapters", "sft")
RSFT_ADAPTER = os.path.join(os.path.dirname(__file__), "..", "adapters", "rsft")
SFT_VALID = os.path.join(os.path.dirname(__file__), "..", "data", "sft", "valid.jsonl")

MODEL_GEMINI = "gemini-2.5-flash-lite"
N_EVAL = 50
MAX_TOKENS = 350

JUDGE_SYSTEM = """\
You are a neutral evaluator of Indian personal finance advice.
Given a user question and two responses (A and B), pick which is BETTER.
Criteria: factual accuracy, appropriate caveats, warm helpful tone, actionable advice.
Return ONLY JSON: {"winner": "A" or "B", "reason": "<one sentence>"}"""

JUDGE_PROMPT = """\
Question: {question}

Response A:
{response_a}

Response B:
{response_b}

Which response is better? Return JSON only."""


def format_prompt(text: str) -> str:
    """Extract the [INST]...[/INST] part as the generation prompt."""
    if "[/INST]" in text:
        return text.split("[/INST]")[0] + "[/INST]"
    return text


def extract_question(prompt: str) -> str:
    if "[INST]" in prompt:
        inner = prompt.split("[INST]", 1)[-1].split("[/INST]")[0].strip()
        return inner.split("\n\n", 1)[-1].strip() if "\n\n" in inner else inner
    return prompt[:200]


def get_response(model, tokenizer, prompt: str) -> str:
    out = generate(model, tokenizer, prompt=prompt, max_tokens=MAX_TOKENS, sampler=make_sampler(temp=0.3), verbose=False)
    if "[/INST]" in out:
        out = out.split("[/INST]", 1)[-1]
    return out.strip().rstrip("</s>").strip()


def judge(client, question: str, a: str, b: str) -> str | None:
    for attempt in range(3):
        try:
            r = client.models.generate_content(
                model=MODEL_GEMINI,
                contents=JUDGE_PROMPT.format(question=question, response_a=a, response_b=b),
                config=types.GenerateContentConfig(
                    system_instruction=JUDGE_SYSTEM,
                    max_output_tokens=200,
                    temperature=0.1,
                ),
            )
            raw = r.text.strip().lstrip("```json").lstrip("```").rstrip("```").strip()
            result = json.loads(raw)
            return result.get("winner"), result.get("reason", "")
        except Exception as e:
            if "429" in str(e) or "quota" in str(e).lower():
                time.sleep(4 * (2 ** attempt))
    return None, ""


def main():
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        print("ERROR: Set GOOGLE_API_KEY")
        sys.exit(1)

    for path, name in [(SFT_ADAPTER, "SFT adapter"), (RSFT_ADAPTER, "RSFT adapter")]:
        if not os.path.isdir(path):
            print(f"ERROR: {name} not found at {path}")
            sys.exit(1)

    with open(SFT_VALID) as f:
        examples = [json.loads(line) for line in f]
    random.shuffle(examples)
    examples = examples[:N_EVAL]

    print(f"Loading SFT model...")
    sft_model, sft_tok = load(BASE_MODEL, adapter_path=SFT_ADAPTER)
    print(f"Loading RSFT (aligned) model...")
    rsft_model, rsft_tok = load(BASE_MODEL, adapter_path=RSFT_ADAPTER)
    client = genai.Client(api_key=api_key)

    results = {"rsft_wins": 0, "sft_wins": 0, "tie": 0}
    print(f"\nEvaluating {N_EVAL} prompts — Gemini judge (SFT vs RSFT)\n{'='*55}")

    for i, ex in enumerate(examples, 1):
        prompt = format_prompt(ex["text"])
        question = extract_question(prompt)

        sft_resp  = get_response(sft_model,  sft_tok,  prompt)
        rsft_resp = get_response(rsft_model, rsft_tok, prompt)

        # Randomly assign A/B to avoid position bias
        if random.random() < 0.5:
            a, b, mapping = sft_resp, rsft_resp, {"A": "sft", "B": "rsft"}
        else:
            a, b, mapping = rsft_resp, sft_resp, {"A": "rsft", "B": "sft"}

        winner, reason = judge(client, question, a, b)
        actual_winner = mapping.get(winner, "tie") if winner else "tie"

        if actual_winner == "rsft":
            results["rsft_wins"] += 1
        elif actual_winner == "sft":
            results["sft_wins"] += 1
        else:
            results["tie"] += 1

        print(f"  [{i:3d}/{N_EVAL}] winner={actual_winner:4s} | {reason[:58]}")

    total = results["rsft_wins"] + results["sft_wins"] + results["tie"]
    rsft_rate = results["rsft_wins"] / total * 100
    sft_rate  = results["sft_wins"]  / total * 100

    print(f"\n{'='*55}")
    print(f"Results over {total} prompts:")
    print(f"  RSFT wins: {results['rsft_wins']:3d} / {total}  ({rsft_rate:.1f}%)")
    print(f"  SFT  wins: {results['sft_wins']:3d}  / {total}  ({sft_rate:.1f}%)")
    print(f"  Ties     : {results['tie']:3d}  / {total}")
    print(f"\nRSFT win rate vs SFT: {rsft_rate:.1f}%")

    out = os.path.join(os.path.dirname(__file__), "win_rate_results.json")
    with open(out, "w") as f:
        json.dump({"rsft_win_rate": rsft_rate, "sft_win_rate": sft_rate,
                   "n_eval": total, **results}, f, indent=2)
    print(f"Results saved → {out}")


if __name__ == "__main__":
    main()
