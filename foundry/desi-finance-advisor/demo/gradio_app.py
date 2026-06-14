"""
Gradio chat demo for Desi Finance Advisor.
Loads the RSFT-aligned model and serves a chat interface.

Run from repo root:
    python foundry/desi-finance-advisor/demo/gradio_app.py

Or with a published HF model:
    python foundry/desi-finance-advisor/demo/gradio_app.py --model shabul/mistral-7b-desi-finance-advisor
"""

import argparse
import os

import gradio as gr
from mlx_lm import load, stream_generate
from mlx_lm.sample_utils import make_sampler

BASE_MODEL = "mlx-community/Mistral-7B-Instruct-v0.2-4bit"
DEFAULT_ADAPTER = os.path.join(os.path.dirname(__file__), "..", "adapters", "rsft")

SYSTEM_PROMPT = (
    "You are Desi Finance Bhai — a warm, knowledgeable Indian personal finance advisor. "
    "Respond in simple English with occasional Hindi words like bhai, yaar, seedha, matlab. "
    "Always be factually accurate. Mention caveats like 'consult a CA' where needed. "
    "Never promise guaranteed returns. Keep it conversational and helpful."
)

EXAMPLE_QUESTIONS = [
    "Should I choose old or new tax regime? I earn 14 LPA.",
    "SIP vs lump sum — which is better right now?",
    "I have 10,000 per month to invest. What should I do?",
    "What is ELSS and is it better than PPF for 80C?",
    "Is it a good time to buy a house or should I keep renting?",
    "My EPF balance is 3 lakhs — can I withdraw it for a vacation?",
    "Explain Sovereign Gold Bonds like I'm a complete beginner.",
    "What is rupee cost averaging and why does everyone talk about it?",
]

CSS = """
.gradio-container { max-width: 800px !important; margin: auto !important; }
.chat-message { font-size: 15px !important; }
footer { display: none !important; }
"""


def build_prompt(message: str, history: list) -> str:
    """Build a Mistral [INST] prompt including conversation history."""
    turns = ""
    for user_msg, bot_msg in history[-3:]:  # keep last 3 turns for context
        turns += f"[INST] {user_msg} [/INST] {bot_msg} </s>"
    # First turn gets the system prompt prepended
    if not history:
        user_content = f"{SYSTEM_PROMPT}\n\n{message}"
    else:
        user_content = message
    turns += f"[INST] {user_content} [/INST]"
    return f"<s>{turns}"


def respond(message: str, history: list, model_state):
    model, tokenizer = model_state
    prompt = build_prompt(message, history)
    output = ""
    for response in stream_generate(model, tokenizer, prompt=prompt, max_tokens=450, sampler=make_sampler(temp=0.35)):
        token = response.text
        output += token
        yield output


def load_model(model_path: str, adapter_path: str | None):
    print(f"Loading model: {model_path}")
    if adapter_path and os.path.isdir(adapter_path):
        print(f"Adapter: {adapter_path}")
        return load(model_path, adapter_path=adapter_path)
    else:
        print("No adapter found — running base model")
        return load(model_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=BASE_MODEL,
                        help="HF model ID or local path (overrides base model)")
    parser.add_argument("--adapter", default=DEFAULT_ADAPTER,
                        help="Path to LoRA adapter directory")
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()

    model, tokenizer = load_model(args.model, args.adapter)
    model_state = (model, tokenizer)

    with gr.Blocks() as demo:
        gr.HTML("""
        <div style="text-align:center; padding: 20px 0 10px 0;">
            <h1 style="font-size:2em; margin:0;">🇮🇳 Desi Finance Advisor</h1>
            <p style="color:#666; margin:6px 0 0 0;">
                Your personal finance <em>bhai</em> — ask me anything about money, investing, and tax.
            </p>
        </div>
        """)

        chatbot = gr.Chatbot(
            label="Chat",
            height=480,
        )

        with gr.Row():
            msg = gr.Textbox(
                placeholder="Bhai, should I invest in SIP or FD?",
                show_label=False,
                scale=9,
                container=False,
            )
            send_btn = gr.Button("Send", scale=1, variant="primary")

        gr.Examples(
            examples=EXAMPLE_QUESTIONS,
            inputs=msg,
            label="Try these questions",
        )

        gr.HTML("""
        <div style="text-align:center; padding:14px 0 0 0; color:#888; font-size:12px;">
            ⚠️ For educational purposes only. Consult a SEBI-registered advisor before investing.
        </div>
        """)

        def user_submit(message, history):
            return "", history + [[message, None]]

        def bot_reply(history, model_state=model_state):
            message = history[-1][0]
            history[-1][1] = ""
            for token in respond(message, history[:-1], model_state):
                history[-1][1] = token
                yield history

        msg.submit(user_submit, [msg, chatbot], [msg, chatbot], queue=False).then(
            bot_reply, chatbot, chatbot
        )
        send_btn.click(user_submit, [msg, chatbot], [msg, chatbot], queue=False).then(
            bot_reply, chatbot, chatbot
        )

    demo.queue()
    demo.launch(server_port=args.port, css=CSS, theme=gr.themes.Soft())


if __name__ == "__main__":
    main()
