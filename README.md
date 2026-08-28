# LoRA From Scratch

Fine-tuning an LLM with LoRA implemented by hand — no `peft`, no `unsloth`, no
`trl`, no Axolotl. Just PyTorch, plus `transformers` for loading pretrained
weights.

Built as a learning project. A config file teaches you which knobs exist; a
training loop teaches you what they do.

---

## Run 1 — Smoke test

| | |
|---|---|
| **Model** | `Qwen/Qwen3-0.6B-Base` (596M params, bf16) |
| **Dataset** | `HuggingFaceH4/no_robots` (9,500 human-written instruction pairs) |
| **Method** | LoRA r=32, alpha=64, on all 7 projections × 28 layers |
| **Trainable** | 20.19M / 616.2M (3.28%) |
| **Hardware** | 1× RTX A5000 (24GB), Vast.ai |
| **Steps** | 100 (0.17 epochs) |
| **Cost** | ~$0.30 |

### Result

Before training, the base model echoed prompts, looped, and emitted invalid
UTF-8. After 100 steps it produced coherent English that answered the question
— but never learned to emit the `<|im_end|>` stop token.

Measured on held-out data, at the positions where `<|im_end|>` should be the
next token, its median rank was **289 of 151,936**. Moving (random is ~76,000)
but far from winning greedy decoding.

That is correct for 0.17 epochs. `<|im_end|>` is roughly 1 token in 130, so
termination is a rare-event pattern needing far more exposure than content
patterns.

---

## Layout

```
.
├── notebooks/
│   └── 01_smoke_test.ipynb     the full run, cell by cell, with commentary
├── scripts/
│   ├── lora.py                 LoRALinear, apply/save/load/merge  (run directly to self-test)
│   └── data.py                 ChatML formatting, masking, collator, download
├── docs/
│   ├── SETUP.md                every command, from instance selection to teardown
│   └── LEARNING_NOTES.md       cell-by-cell walkthrough with reasoning and observed numbers
└── requirements.txt
```

The notebook is self-contained and defines everything inline — it's meant to be
read and run top to bottom. `scripts/` holds the same code extracted as
importable modules for reuse in later runs.

---

## Quick start

```bash
# on a Vast.ai instance with the PyTorch template
export HF_HOME=/workspace/hf
mkdir -p /workspace/{hf,data,checkpoints}
pip install -r requirements.txt

python scripts/lora.py            # CPU self-test, ~1s
python scripts/data.py --download # fetch no_robots to /workspace/data
```

Then open `notebooks/01_smoke_test.ipynb`.

See [docs/SETUP.md](docs/SETUP.md) for instance selection, SSH/VS Code, and the
reasoning behind each step.

---

## Three things worth knowing

**`dL/dA` is exactly zero on the first backward pass.** LoRA's `B` matrix is
zero-initialized and the gradient to `A` flows through `B`, so `A` cannot move
until `B` has taken a step. Observable, not theoretical — and the best single
reason to write the layer by hand.

**The logits are bigger than the model.** A batch of 4 × 217 tokens over a
151,936-token vocabulary spiked VRAM from 1.35 GB to 5.83 GB, almost all of it
one logits tensor. This is why sequence length hurts more than parameter count
on big-vocab models, and why the fix is chunked loss rather than a bigger GPU.

**Loss is a bad eval for rare tokens.** Training loss fell smoothly while the
model completely failed to learn termination, because `<|im_end|>` is under 1%
of trained positions. Measuring the *rank* of the target token, at the positions
it should appear, exposed what loss averaged away.

Full details, including eleven failure modes and their symptoms, in
[docs/LEARNING_NOTES.md](docs/LEARNING_NOTES.md).

---

## Environment

```
Vast.ai image  robatvastai/pytorch
torch          2.11.0+cu128
transformers   5.16.1
python         3.12
```

---

## Roadmap

- [x] **Run 1** — LoRA on Qwen3-0.6B, 100 steps
- [ ] **Run 2** — full fine-tune of Qwen3-1.7B-Base. `encode`, `collate`, and
      the loop carry over unchanged. LR drops to ~1e-5, `adamw_8bit` for
      optimizer state, gradient checkpointing.
- [ ] **Run 3** — QLoRA on Qwen3-8B, where memory actually constrains you and
      the optimized libraries start earning their keep.

Memory math for why run 2 needs 8-bit AdamW on a 24 GB card:

```
params bf16          3.4 GB
grads bf16           3.4 GB
AdamW fp32 states   27.2 GB   <- doesn't fit
AdamW 8-bit states   3.4 GB   <- fits
```

Scale that to 30B and you get the 8×H100 requirement for full fine-tuning.
