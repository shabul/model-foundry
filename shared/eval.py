"""
Quick inference and evaluation helper for any mlx-lm model/adapter.

Usage:
    python -m shared.eval --model Qwen/Qwen2.5-3B-Instruct \
                          --adapter foundry/qwen2.5-dolly/adapters \
                          --prompt "Explain gradient descent simply."
"""

import argparse

from mlx_lm import generate, load


DEFAULT_PROMPTS = [
    "Explain the difference between supervised and unsupervised learning.",
    "Write a short poem about the ocean.",
    "Summarize what a neural network is in two sentences.",
]


def run_inference(model_path: str, adapter_path: str | None, prompts: list[str],
                  max_tokens: int = 300):
    print(f"Loading model: {model_path}")
    if adapter_path:
        print(f"Adapter:       {adapter_path}")
    model, tokenizer = load(model_path, adapter_path=adapter_path)

    for prompt in prompts:
        messages = [{"role": "user", "content": prompt}]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        print(f"\n{'─'*60}")
        print(f"PROMPT: {prompt}")
        print(f"{'─'*60}")
        response = generate(model, tokenizer, prompt=text, max_tokens=max_tokens, verbose=False)
        print(response)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Base model id or local path")
    parser.add_argument("--adapter", default=None, help="Path to LoRA adapter directory")
    parser.add_argument("--prompt", default=None, help="Single prompt to run")
    parser.add_argument("--max-tokens", type=int, default=300)
    args = parser.parse_args()

    prompts = [args.prompt] if args.prompt else DEFAULT_PROMPTS
    run_inference(args.model, args.adapter, prompts, args.max_tokens)


if __name__ == "__main__":
    main()
