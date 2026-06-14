"""
Generate 800+ synthetic QA pairs across 6 Indian personal finance topics using Gemini.

Run from repo root:
    export GOOGLE_API_KEY=...
    python foundry/desi-finance-advisor/phase1_data_prep/generate_synthetic.py
"""

import json
import os
import sys
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

from google import genai
from google.genai import types

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "cleaned")
MODEL = "gemini-2.5-flash-lite"

SYSTEM_INSTRUCTION = (
    "You are Desi Finance Bhai — a warm, knowledgeable Indian personal finance advisor. "
    "Respond in simple English with occasional Hindi words: bhai, yaar, seedha, matlab, basically. "
    "Always be factually accurate. Add caveats like 'consult a CA' or 'past returns don't guarantee future returns' where needed. "
    "Never promise guaranteed returns. Keep responses 120–220 words, conversational, no bullet points."
)

TOPICS = {
    "SIPs & Mutual Funds": [
        "What is a SIP and how does rupee cost averaging work?",
        "Should I pause my SIP when markets are crashing?",
        "Direct vs regular mutual fund — which is better and why?",
        "What is an index fund and is it better than an actively managed fund?",
        "How do I pick a good mutual fund? Expense ratio or past returns?",
        "Is it safe to invest in small-cap or mid-cap funds as a beginner?",
        "What are liquid funds and when should I use them?",
        "Can I withdraw from my SIP anytime or is there a lock-in?",
        "What happens to my mutual fund units if the AMC shuts down?",
        "SIP of 5000 per month in Nifty 50 — is that a good start?",
        "ELSS vs PPF for tax saving under 80C — which is better?",
        "What is NAV and why does it keep changing?",
        "How much should I invest in SIP monthly on a 60k salary?",
        "What is the difference between growth and dividend option in a mutual fund?",
        "Is it okay to have 5 different SIPs running at the same time?",
    ],
    "Home Loans & EMIs": [
        "What CIBIL score do I need to get a good home loan interest rate?",
        "Fixed vs floating interest rate — which home loan should I choose?",
        "How much home loan can I get on a monthly salary of 80,000?",
        "What is a home loan balance transfer and when does it make sense?",
        "Should I prepay my home loan or invest the extra money?",
        "What are the hidden charges nobody tells you about in a home loan?",
        "Is it better to take a joint home loan with my spouse?",
        "How is the EMI calculated on a home loan?",
        "What documents do I need to apply for a home loan?",
        "What is PMAY subsidy and am I eligible for it?",
        "Is it better to rent or buy a house in India at 30?",
        "How much down payment should I save before buying a house?",
        "What is a top-up loan on my existing home loan?",
        "Can I claim tax benefit on home loan principal and interest?",
        "What happens if I miss an EMI on my home loan?",
    ],
    "Tax Planning": [
        "Old tax regime vs new tax regime — which is better at 12 LPA?",
        "What is Section 80C and how do I maximise 1.5 lakh deduction?",
        "Can I claim HRA if I pay rent to my parents?",
        "What is the standard deduction for salaried employees in 2024?",
        "How is long-term capital gains on stocks taxed in India?",
        "What is Form 16 and what do I do with it when filing ITR?",
        "How do I file my ITR myself for the first time?",
        "What happens if I miss the ITR filing deadline?",
        "How is cryptocurrency taxed in India — flat 30% on what exactly?",
        "What is TDS and when is it deducted from my salary or interest?",
        "Can I claim deduction on health insurance premium for my parents?",
        "How is FD interest income taxed?",
        "What is Section 80D and how much can I save on medical insurance?",
        "How is rental income taxed if I have a second property?",
        "What is advance tax and who needs to pay it?",
    ],
    "Stocks & Investing": [
        "How do I start investing in stocks for the very first time?",
        "What is the difference between NSE and BSE?",
        "What is a Demat account and how is it different from a trading account?",
        "Should I invest in IPOs as a beginner?",
        "What are blue-chip stocks and are they safe?",
        "What is the P/E ratio and how do I use it to evaluate a stock?",
        "Zerodha vs Groww — which broker is better for a beginner?",
        "What is intraday trading and why is it dangerous for beginners?",
        "What are circuit breakers in the stock market?",
        "How much money do I actually need to start investing in stocks?",
        "Should I buy individual stocks or just invest in mutual funds?",
        "What is F&O trading and why do 90% of people lose money in it?",
        "How do dividends work — when do I get money from a stock?",
        "What is the difference between face value and market price of a stock?",
        "How do I read a company's balance sheet before buying its shares?",
    ],
    "PPF / NPS / EPF": [
        "What is PPF and is it still a good investment in 2024?",
        "How much EPF is deducted from my CTC every month?",
        "Can I withdraw my EPF balance before I retire?",
        "What is the difference between NPS Tier 1 and Tier 2 accounts?",
        "How do I check my EPF balance and UAN number?",
        "PPF vs NPS — which is better for retirement planning?",
        "What is the current interest rate on PPF and is it guaranteed?",
        "Can I open a PPF account in a private bank like HDFC or ICICI?",
        "How long is the lock-in period for PPF and can I extend it?",
        "What happens to my EPF when I switch jobs?",
        "How does NPS work after retirement — what do I actually get?",
        "Is EPF interest taxable when I withdraw?",
        "What is VPF and should I contribute extra to it?",
        "Can I take a loan against my PPF balance?",
    ],
    "Gold & Real Estate": [
        "Physical gold vs Sovereign Gold Bond — which is better?",
        "What is a Sovereign Gold Bond and how do I buy one?",
        "What are Gold ETFs and how are they different from SGBs?",
        "Is real estate a good investment in India right now?",
        "How is gold taxed when I sell it — physical gold vs SGB?",
        "What percentage of my portfolio should I keep in gold?",
        "What are REITs and how are they different from buying a flat?",
        "What are the risks of investing in real estate in India?",
        "Can I take a loan against my gold jewellery?",
        "Why do financial advisors say real estate is illiquid?",
        "Is buying a second house a good investment if I can rent it out?",
        "What is the difference between a REIT and a real estate mutual fund?",
    ],
}

PROMPT = "Topic: {topic}\nQuestion: {question}\n\nAnswer as Desi Finance Bhai. Flowing prose, no bullets, 120–220 words."

_lock = threading.Lock()
_counter = {"done": 0, "total": 0}


def generate_one(client: genai.Client, topic: str, question: str, idx: int) -> dict | None:
    for attempt in range(4):
        try:
            r = client.models.generate_content(
                model=MODEL,
                contents=PROMPT.format(topic=topic, question=question),
                config=types.GenerateContentConfig(
                    system_instruction=SYSTEM_INSTRUCTION,
                    max_output_tokens=380,
                    temperature=0.82,
                ),
            )
            text = r.text.strip()
            if len(text) > 80:
                with _lock:
                    _counter["done"] += 1
                    n, t = _counter["done"], _counter["total"]
                    print(f"  ✓ [{n:4d}/{t}] [{topic[:18]}] {question[:52]}")
                return {"instruction": question, "response": text, "topic": topic}
        except Exception as e:
            err = str(e)
            if any(x in err for x in ["429", "503", "quota", "rate"]):
                time.sleep(4 * (2 ** attempt))
            else:
                time.sleep(1)
    return None


def main():
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        print("ERROR: Set GOOGLE_API_KEY")
        sys.exit(1)

    tasks = [(topic, q) for topic, qs in TOPICS.items() for q in qs]
    _counter["total"] = len(tasks)

    print(f"\n{'='*60}")
    print(f"Desi Finance Advisor — Synthetic Data Generator")
    print(f"Model   : {MODEL}")
    print(f"Topics  : {len(TOPICS)}")
    print(f"Total   : {len(tasks)} QA pairs")
    print(f"{'='*60}\n")

    client = genai.Client(api_key=api_key)
    os.makedirs(DATA_DIR, exist_ok=True)

    results = []
    with ThreadPoolExecutor(max_workers=5) as pool:
        futures = {pool.submit(generate_one, client, t, q, i): (t, q)
                   for i, (t, q) in enumerate(tasks, 1)}
        for fut in as_completed(futures):
            r = fut.result()
            if r:
                results.append(r)

    out = os.path.join(DATA_DIR, "synthetic.jsonl")
    with open(out, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")

    print(f"\n{'='*60}")
    print(f"Generated {len(results)} / {len(tasks)} pairs → {out}")
    print("Next: python foundry/desi-finance-advisor/phase2_sft/prepare_sft_data.py")


if __name__ == "__main__":
    main()
