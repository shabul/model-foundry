# /finetune — MLX LoRA Fine-Tuning Skill

Full pipeline for fine-tuning open-source LLMs on Apple Silicon using mlx-lm, from
model selection through HF Hub deployment. All projects live under `foundry/`.

---

## Commands

| Command | What it does |
|---------|-------------|
| `/finetune new <project-name>` | Scaffold a new project under `foundry/` |
| `/finetune data <project-name>` | Prepare and format the dataset to JSONL |
| `/finetune train <project-name>` | Run LoRA training via mlx-lm |
| `/finetune eval <project-name>` | Run quick inference to sanity-check the adapter |
| `/finetune push <project-name> <hf-repo>` | Fuse adapters and push to HF Hub |
| `/finetune card <project-name>` | Generate a model card from training stats |
| `/finetune status` | Show all projects and their current state |

---

## Hardware Context

**Always assume Apple Silicon (M-series Mac) with unified memory.**  
Use `mlx-lm` — never PyTorch/MPS. Key constraints:
- Peak safe memory for training: ~18–20 GB (leave headroom for OS + other apps)
- If OOM: reduce `batch_size` first (4→2→1), then `max_seq_length` (2048→1024→512)
- Remove `mpich`/`mpi4py` if present — MLX requires Open MPI or none at all
- Run scripts from repo root so `shared/` imports resolve

---

## `/finetune new <project-name>`

Scaffold a new project by copying the template from `foundry/qwen2.5-dolly/`.

Steps:
1. `cp -r foundry/qwen2.5-dolly foundry/<project-name>`
2. Ask the user:
   - **Base model** — recommend from the table below based on task and memory
   - **Dataset** — HF dataset id or local path
   - **HF repo** — where to push the final model
3. Update `foundry/<project-name>/config/lora_config.yaml` with chosen model
4. Update `foundry/<project-name>/prepare_data.py` dataset id and field names
5. Update `foundry/<project-name>/push_to_hub.py` BASE_MODEL constant
6. Create `foundry/<project-name>/MODEL_CARD.md` shell (fill after training)

### Model selection guide for M-series Mac

| Memory budget | Recommended model | Params | Notes |
|---|---|---|---|
| <16 GB free | `google/gemma-2-2b-it` | 2B | Smallest capable instruct model |
| 16–20 GB free | `Qwen/Qwen2.5-3B-Instruct` | 3B | Best quality/size tradeoff (default) |
| 20–22 GB free | `microsoft/Phi-3.5-mini-instruct` | 3.8B | Strong reasoning |
| 22+ GB free | `meta-llama/Llama-3.2-3B-Instruct` | 3B | Strong, gated (requires HF approval) |

---

## `/finetune data <project-name>`

Run the data preparation script for the project.

Steps:
1. Check `foundry/<project-name>/prepare_data.py` — verify dataset id and field mappings
2. `python foundry/<project-name>/prepare_data.py`
3. After completion, print first 3 lines of `foundry/<project-name>/data/train.jsonl`
   to confirm the chat template looks correct before training
4. Report: train/val split counts

If dataset fields differ from the Dolly schema (`instruction`, `context`, `response`),
update the `format_chat_example()` call in `prepare_data.py` with correct key names.
The shared utility `shared/data_utils.format_chat_example` accepts `instruction_key`,
`context_key`, `response_key` kwargs.

---

## `/finetune train <project-name>`

Run LoRA fine-tuning via mlx-lm.

Steps:
1. Check `foundry/<project-name>/config/lora_config.yaml` — confirm model, iters, batch_size
2. Verify `foundry/<project-name>/data/train.jsonl` and `valid.jsonl` exist
3. Run in background: `python foundry/<project-name>/train.py`
4. Monitor output every ~2 min — report loss at each checkpoint
5. Watch for and auto-fix known failure modes:

### Known failure modes + fixes

| Error | Fix |
|---|---|
| `lr_schedule: string indices must be integers` | Remove `lr_schedule` line from config or use dict format |
| `[mpi] MPI found but not Open MPI` + exit 250 | `conda remove mpich mpi4py --yes` then retry |
| `[METAL] Insufficient Memory` | Halve `batch_size`; if still OOM, halve `max_seq_length` |
| `Calling python -m mlx_lm.lora is deprecated` | Use `python -m mlx_lm lora` (space, not dot) |

6. Report final training summary: val loss curve, best checkpoint, tokens/sec, peak memory

---

## `/finetune eval <project-name>`

Quick inference sanity-check against the saved adapter.

Steps:
1. Run: `python -m shared.eval --model <base-model> --adapter foundry/<project-name>/adapters`
2. Use these 3 default prompts unless user specifies others:
   - `"Explain the difference between supervised and unsupervised learning."`
   - `"Write a short poem about the ocean."`
   - `"Summarize what a neural network is in two sentences."`
3. Print responses and ask user: does this look better than the base model for your use case?

---

## `/finetune push <project-name> <hf-repo>`

Fuse adapters and upload to Hugging Face Hub.

Steps:
1. Check HF auth: `python -c "from huggingface_hub import whoami; print(whoami()['name'])"`
2. Verify adapter exists: `foundry/<project-name>/adapters/adapters.safetensors`
3. Run: `python foundry/<project-name>/push_to_hub.py --repo <hf-repo>`
4. If MODEL_CARD.md exists, it is automatically uploaded as README.md
5. Confirm: print the HF Hub URL

---

## `/finetune card <project-name>`

Generate a contemporary Hugging Face model card from training stats.

Steps:
1. Read the training output log to extract:
   - Val loss curve (all checkpoints)
   - Best val loss and at which iter
   - Peak memory usage
   - Tokens/sec throughput
   - Total trained tokens
   - Trainable parameter count and percentage
2. Read `foundry/<project-name>/config/lora_config.yaml` for hyperparameters
3. Write `foundry/<project-name>/MODEL_CARD.md` with:
   - YAML frontmatter: base_model, datasets, language, library_name, tags, license
   - Section: What this is (base model + dataset + goal)
   - Section: Training snapshot table (model, method, dataset, hardware, framework, time, iters, batch, seq_len, lr, LoRA layers)
   - Section: Parameters — the X% that changed (total / trainable / frozen / peak memory)
   - Section: Loss curve table (iter → val loss → train loss)
   - Section: Best val loss + % reduction from baseline
   - Section: Throughput (it/sec, tok/sec, total tokens)
   - Section: How to run (mlx-lm code snippet)
   - Section: Author — "Shabul Abdul, Sr. Data Scientist"
   - Footer: hardware note ("trained on Apple Silicon, no cloud GPUs")
4. Upload to HF Hub if repo already exists

---

## `/finetune status`

Show state of all projects in the foundry.

Steps:
1. List all directories under `foundry/`
2. For each, check and report:
   - `config/lora_config.yaml` — model name
   - `data/train.jsonl` — exists? line count?
   - `adapters/adapters.safetensors` — exists? file size?
   - HF repo (from push_to_hub.py BASE_MODEL + repo args if determinable)
3. Print a summary table

---

## Repo conventions

- Always run scripts from repo root: `python foundry/<project>/train.py`
- Gitignored: `foundry/**/data/`, `foundry/**/adapters/`, `foundry/**/fused/`
- Shared utilities: `shared/data_utils.py`, `shared/hub_utils.py`, `shared/eval.py`
- Model cards live at `foundry/<project>/MODEL_CARD.md`, pushed as `README.md` to HF
- After each new project is pushed, update the Models table in root `README.md`
