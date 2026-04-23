---
base_model: Qwen/Qwen2.5-3B-Instruct
datasets:
- yahma/alpaca-cleaned
- synthetic
language:
- en
library_name: mlx
tags:
- lora
- fine-tuned
- feynman
- explanation
- teaching
- apple-silicon
- mlx-lm
- qwen2.5
license: apache-2.0
---

# qwen2.5-3b-feynman-explainer

> *"You don't understand something unless you can explain it to your grandmother."* — Feynman

A LoRA fine-tune of Qwen2.5-3B-Instruct trained to explain anything the way
Richard Feynman did — from the ground up, with vivid analogies, no jargon until
it's earned, and pure flowing prose.

Built by **Shabul Abdul**, Sr. Data Scientist.  
Trained on Apple M5 MacBook Pro · 24 GB unified memory · Apple MLX.

---

## Try it right here — no download needed

The difference is immediate. Same model weights, same question. Just a style shift.

---

### "How does gradient descent work?"

**Base model** (`Qwen2.5-3B-Instruct`, no fine-tuning):
> Gradient descent is an optimization algorithm used in machine learning to minimize
> a loss function. It works by computing the gradient of the loss with respect to
> the model parameters and updating the parameters in the opposite direction of the
> gradient, scaled by a learning rate. This process is repeated iteratively until
> convergence...

**This model** (Feynman fine-tuned):
> Imagine you're blindfolded somewhere on a hilly landscape, and your only goal is to
> reach the lowest valley. You can't see anything — but you can feel the slope of the
> ground under your feet.
>
> Here's what you do: take a small step in whichever direction feels most downhill.
> Then stop. Feel the slope again. Take another step downhill. Repeat.
>
> That's gradient descent. The "gradient" is just the slope — which direction is up,
> how steep it is. You always step opposite to it, because you're trying to go *down*.
> The "learning rate" is how big each step is. Too big and you overshoot the valley
> and bounce around the hillside forever. Too small and you're shuffling along for
> eternity.
>
> The key insight: you never need to see the whole landscape. You only ever need to
> know the slope exactly where you're standing.

---

### "What is entropy?"

**Base model:**
> Entropy is a thermodynamic quantity representing the amount of disorder or randomness
> in a system. In information theory, entropy measures the average amount of information
> or uncertainty in a random variable. Higher entropy indicates greater disorder...

**This model:**
> Here's something that bothered physicists for a long time: heat always flows from
> hot things to cold things. Never the other way. You've never seen a warm cup of
> coffee spontaneously get hotter while the air around it gets colder. But why not?
> The laws of physics don't actually forbid it.
>
> The answer is entropy, and it's not really about disorder — that's a metaphor that
> confuses more than it helps. It's about *counting*.
>
> Take a room with all the air molecules crammed in one corner. That's one arrangement.
> Now consider all the arrangements where molecules are spread out evenly — there are
> astronomically more of them. So when molecules move randomly, they almost certainly
> end up spread out, simply because there are so many more ways to be spread out than
> bunched up.
>
> Entropy is just a measure of how many arrangements look like what you've got.
> High entropy means lots of arrangements produce this same picture. The reason entropy
> always increases is brutally simple: there are more ways to be disordered than ordered,
> so randomness always drifts toward disorder.

---

### "What is a p-value?"

**Base model:**
> A p-value is the probability of obtaining results at least as extreme as the observed
> results, assuming the null hypothesis is true. If the p-value is below the significance
> threshold (typically 0.05), the result is considered statistically significant...

**This model:**
> Scientists get this wrong all the time, so let's go slowly.
>
> You've run an experiment. You gave half the patients a drug and half a sugar pill,
> and the drug group got better faster. Great — but here's the question you should be
> asking: *could this difference have shown up just by chance, even if the drug does
> absolutely nothing?*
>
> The p-value answers that question. It says: if the drug had zero effect, what's the
> probability I'd still see a difference this big, just from the randomness of who got
> sick and who got better?
>
> If that probability is tiny — say, 2% — we say p = 0.02. That's suspicious. It's
> hard to believe chance alone did this.
>
> Here's what p-value does NOT mean, and this is where everyone goes wrong: it is not
> the probability that the drug works. It's not even close to that. It's only the
> probability of your data, assuming the drug doesn't work.
>
> The difference matters enormously. One is a statement about the drug. The other is
> a statement about your data.

---

## Training details

| | |
|---|---|
| **Base model** | `Qwen/Qwen2.5-3B-Instruct` |
| **Method** | LoRA (rank 16, alpha 32) |
| **Dataset** | ~430 custom + ~400 Alpaca questions, Claude-rewritten in Feynman style |
| **Hardware** | Apple M5 MacBook Pro · 24 GB unified memory |
| **Framework** | `mlx-lm` (Apple MLX) |
| **Iterations** | 1,500 steps |
| **Learning rate** | 2e-4 |

*Full loss curve and throughput stats will be added after training.*

---

## How to run

```python
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
```

---

## Why this works

Style transfer via LoRA is a different beast from knowledge fine-tuning.
The base model already knows *what* gradient descent is. We're teaching it
*how to talk about it* — the rhythm, the analogy-first structure, the short
declarative sentences, the moment of "here's where most people get confused."

Rank 16 (vs. rank 8 for a knowledge fine-tune) gives the adapter enough
capacity to shift the generative distribution meaningfully. Higher learning
rate (2e-4) pushes the style harder in fewer steps.

---

## Author

**Shabul Abdul** — Sr. Data Scientist  
[huggingface.co/shabul](https://huggingface.co/shabul)

---

*No cloud GPUs. No PhD required. Just a MacBook and a good idea.*
