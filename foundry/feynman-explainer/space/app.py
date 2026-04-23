"""
Feynman Explainer — Gradio Chat App
Runs on Hugging Face Spaces (CPU free tier).

Loads qwen2.5-3b-feynman-explainer on CPU with a CPU-safe dtype.
Streams tokens for a responsive ChatGPT-like experience.
"""

import threading

try:
    import spaces  # HF Spaces ZeroGPU shim — no-op on CPU tier
except ImportError:
    pass

import gradio as gr
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer

MODEL_ID = "shabul/qwen2.5-3b-feynman-explainer"

SYSTEM_PROMPT = (
    "You are a Feynman-style explainer. For every question, build intuition "
    "from the ground up using concrete analogies and everyday language. "
    "No jargon until it's earned. No bullet points. Pure flowing prose. "
    "Be conversational and enthusiastic — like Feynman genuinely loved this topic."
)

TITLE = "🔬 Feynman Explainer"
DESCRIPTION = """
**Ask anything.** This model explains concepts the way Richard Feynman did —
starting with a concrete analogy, building intuition from scratch, never hiding
behind jargon.

*Built by [Shabul Abdul](https://huggingface.co/shabul), Sr. Data Scientist.
Fine-tuned on Apple M5 MacBook Pro using [MLX](https://github.com/ml-explore/mlx).*

> *"You don't understand something unless you can explain it simply."* — Feynman

---
⏱️ **CPU only** — responses take 20–40 seconds. Worth the wait.
"""

EXAMPLES = [
    ["How does gradient descent actually work?"],
    ["What is entropy and why does it always increase?"],
    ["What is a p-value and why do people misuse it?"],
    ["Why does ice float on water?"],
    ["What is the bias-variance tradeoff?"],
    ["How does attention work in language models?"],
    ["What is a derivative?"],
    ["Why does compounding interest feel like magic?"],
]

print(f"Loading model: {MODEL_ID}")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float32,
    low_cpu_mem_usage=True,
)
model.to("cpu")
model.eval()
print("Model loaded.")


def respond(message: str, history: list[dict], max_new_tokens: int, temperature: float):
    # Build messages list from history + new message
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for h in history:
        messages.append({"role": h["role"], "content": h["content"]})
    messages.append({"role": "user", "content": message})

    input_ids = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
    )

    streamer = TextIteratorStreamer(
        tokenizer, skip_prompt=True, skip_special_tokens=True
    )

    gen_kwargs = dict(
        input_ids=input_ids,
        streamer=streamer,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        do_sample=temperature > 0,
        repetition_penalty=1.1,
    )

    thread = threading.Thread(target=model.generate, kwargs=gen_kwargs)
    thread.start()

    partial = ""
    for token in streamer:
        partial += token
        yield partial

    thread.join()


with gr.Blocks(
    title=TITLE,
    theme=gr.themes.Soft(primary_hue="blue", neutral_hue="slate"),
    css=".gradio-container { max-width: 820px !important; margin: auto; }",
) as demo:

    gr.Markdown(f"# {TITLE}\n{DESCRIPTION}")

    with gr.Row():
        with gr.Column(scale=4):
            max_tokens = gr.Slider(
                100, 600, value=350, step=50,
                label="Max response length (tokens)",
            )
        with gr.Column(scale=4):
            temperature = gr.Slider(
                0.1, 1.2, value=0.75, step=0.05,
                label="Creativity (temperature)",
            )

    chat = gr.ChatInterface(
        fn=respond,
        additional_inputs=[max_tokens, temperature],
        examples=EXAMPLES,
        cache_examples=False,
        type="messages",
        chatbot=gr.Chatbot(
            height=480,
            placeholder="<br><br><center>Ask me to explain anything — I'll make it simple.</center>",
            show_label=False,
        ),
        textbox=gr.Textbox(
            placeholder="e.g. How does a neural network learn?",
            container=False,
            scale=7,
        ),
        submit_btn="Explain →",
        retry_btn="Try again",
        undo_btn="Undo",
        clear_btn="Clear chat",
    )

    gr.Markdown(
        "---\n"
        "🧠 Model: [`shabul/qwen2.5-3b-feynman-explainer`](https://huggingface.co/shabul/qwen2.5-3b-feynman-explainer) · "
        "📦 Base: `Qwen/Qwen2.5-3B-Instruct` · "
        "🍎 Trained on Apple Silicon with [mlx-lm](https://github.com/ml-explore/mlx-lm)"
    )

if __name__ == "__main__":
    demo.launch()
