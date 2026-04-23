from mlx_lm import load, generate

model, tokenizer = load("shabul/qwen2.5-3b-feynman-explainer")

question = "Why does ice float on water?"
messages = [
    {"role": "system", "content": (
        "You are a Feynman-style explainer. Build intuition from the ground up "
        "using concrete analogies. No jargon until it's earned. Flowing prose only."
    )},
    {"role": "user", "content": question},
]
prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
print(generate(model, tokenizer, prompt=prompt, max_tokens=400))