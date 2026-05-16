---
license: gemma
base_model: google/gemma-4-E2B-it
library_name: peft
language:
  - ja
tags:
  - lora
  - qlora
  - p5js
  - children
  - vibekid
  - gemma
pipeline_tag: text-generation
---

# VibeKid — Gemma 4 E2B LoRA for Japanese children's p5.js code generation

> A Gemma 4 E2B LoRA for generating p5.js code from natural-language
> instructions spoken by Japanese-speaking children. Submission to the
> Kaggle Gemma 4 Hackathon (2026).

A QLoRA adapter on `google/gemma-4-E2B-it`, fine-tuned on a **757-item
child-targeted p5.js synthesis dataset**. Intended to be used with the
on-device LiteRT (`@mediapipe/tasks-genai`) Web app
[VibeKid](https://github.com/velocitylabo/vibekid).

## Highlights

- **Base model (concept)**: `google/gemma-4-E2B-it`
- **Base model (loaded at training)**:
  `unsloth/gemma-4-e2b-it-unsloth-bnb-4bit` @ revision `a285b07ef4`
  (see `adapter_config.json`; the latest at training time).
- **Adapter type**: QLoRA r=16, α=32 (text decoder layers only; the
  multimodal layers are not trained).
- **Train data**: 657 items (4/19 split; synthesized via OpenRouter
  Gemini 2.5 Flash).
- **Eval data**: 100 items (category-balanced; 4/19 cutoff).
- **Hardware**: Colab Pro A100 40 GB (bf16 native); 3 epochs ≈ 27 min.
- **Adapter size**: ~50 MB.

## Training metrics (validation loss curve)

Cross-entropy loss on the eval split (100 items) over the course of
training:

| Step | Training Loss | Validation Loss |
|------|---------------|-----------------|
| 50   | 0.732 | 3.00 |
| 100  | 0.507 | **2.98** (minimum) |
| 150  | 0.416 | 3.06 |
| 200  | 0.336 | 3.05 |
| 249  | 0.328 | 3.05 |

Over 3 epochs (249 steps, effective batch=8), **train_loss went from 0.99
to 0.33** (a 66% drop), while **val_loss converged around step 100** (a
mild overfit signal; from step 100 to 249 it is roughly flat). For
reference, a 1-epoch run had eval_loss 6.34; the 3-epoch run reaches 3.05
— **more than halved**.

Note: an additional eval immediately after training once observed
`eval_loss=NaN` (possibly attributable to `pad_token == eos_token`,
accompanied by an attention_mask warning). **All evals during training
were finite**, and the adapter itself is healthy.

## exec_success_rate + secondary metrics (Playwright real-device evaluation)

Execution evaluation against a 100-item eval set
(`sft/data/eval.jsonl`). `exec_success_rate` did not reach execution
success on either BASELINE or SFT — **0/100 for both** — but the
**secondary metrics clearly show that SFT improves code quality**:

| Setup | exec_success_rate | mean_code_length | has_animation | js_error | code_too_short | n | Path |
|---|---:|---:|---:|---:|---:|---|---|
| Baseline (`google/gemma-4-E2B-it`, raw) | 0 / 100 | (mostly prose) | - | 0 | 100 | 100 | Web (LiteRT) |
| Baseline (`google/gemma-4-E2B-it` + app one-shot system prompt) | 0 / 100 | 698 | (n/a) | 34 | - | 100 | Web (LiteRT) |
| **Fine-tuned (this adapter)** | **0 / 100** | **828** | **59%** | **1** | **9** | 100 | Notebook (Python + Colab Pro A100) |

### Narrative

`exec_success_rate` = the fraction of items where (a) static checks pass
for `setup()` + `draw()` + `createCanvas()`, (b) the code runs for 5
seconds in an iframe sandbox with no `SyntaxError` / `ReferenceError`,
and (c) `<canvas>` is rendered.

Across the 100-item eval set, both BASELINE and SFT scored 0/100 (neither
reached execution success). The dominant failure mode is
**`no_canvas` 91/94** (no `<canvas>` detected inside the iframe sandbox),
arising from a combination of difficult user instructions (drawing tools
/ games / interactive) and iframe sandbox constraints baked into the
eval-set design.

That said, the secondary metrics **clearly show SFT outperforming the
base on code quality**:

- **`js_error`**: 34 (BASELINE_SYS) → **1** (SFT; a 97% reduction) — SFT
  emits syntactically valid code.
- **`mean_code_length`**: 698 → **828** chars — completion rates rise
  under SFT and the code stops being truncated.
- **`has_animation_rate`**: 59% — SFT emits code that contains p5.js
  animation structures (`frameCount` / `random` / `sin` / `cos` /
  event handlers).
- **`code_too_short`**: only 9 items (= 91 items have a complete code
  structure: `setup()` + `draw()` + `createCanvas()` all present).

### iframe RAF throttle hypothesis: verified false

The initial measurement (5 s timeout, hidden iframe) yielded 0/100. To
test the hypothesis "an invisible iframe is RAF-throttled and
`createCanvas()` in `setup()` cannot fit within the timeout," we re-ran
with **visible iframe + 10 s timeout** → **still 0/100, with `fail_reasons`
matching exactly** → the hypothesis is **falsified**. The 100-item eval
set produces differences at the code-output level but does not reach
iframe execution at this difficulty.

Raw data:
[`sft/logs/validate_*_phase4_sft.json`](https://github.com/velocitylabo/vibekid/blob/main/sft/logs/) (v1: 5 s / hidden) /
`_v2.json` (v2: 10 s / visible).

### Value of SFT on the production path (a separate metric)

The value of SFT on the production app's preset-driven UX is demonstrated
on the **cap rate of the Kaggle 84-run bench** (writeup Section B):

- Phase 2 (old SFT): SFT+oneshot had **cap rate 43%** (`output_tokens`
  hits 512 = the output does not stop).
- **Phase 4 (new SFT; this adapter)**: with
  `train_on_responses_only` + Gemma 4 end-of-turn token (`<turn|>`) fix,
  **cap rate becomes 14% (−29pt)**.

A 14% cap rate is practical on the production preset path (the remaining
14% is related to SFT+bare 100%, Hypothesis D; Future Work #1 considers
addressing it via DPO + system-prompt variation bake-in). For routes
toward execution success, see Future Work #1 / #2 (DPO + `transformers.js`
+ ONNX) in the writeup.

### Notes on the measurement path

Because there is no official toolchain for loading LoRA into LiteRT-LM web
([LiteRT-LM overview](https://ai.google.dev/edge/litert-lm/overview)),
SFT was measured in **two stages: generate 100 items in a Notebook
(Python + Colab Pro A100), then execute locally with Playwright**. The
validate logic (iframe sandbox + heartbeat judgement) is shared between
Web and Notebook (`sft/eval-runner.html` /
`sft/eval-validate-runner.html`). The 2 BASELINE settings were measured
on the Web path (LiteRT, `evaluate.mjs --system=none|app-oneshot`); SFT
was measured on the Notebook path. Because the validate logic is shared,
the comparison is apples-to-apples.

## Qualitative observations

Manual evaluation of generated samples (5 prompts × 1 run each) after the
3-epoch fine-tune:

- ✅ **Improved drawing detail**: the base model's "red square" becomes,
  for the `ねこ` preset, a concrete depiction like "a white cat's body /
  ears / eyes."
- ✅ **Visual loop**: for the prompt "はなびがどかーん" ("fireworks go
  boom"), generates a `for`-loop animation of random circles + lines.
- ⚠️ **Animation logic**: for "ぴょんぴょんはねる" ("hopping"), tries
  animation comments + `let yPos`, but the pattern of declaring the
  variable as local inside `draw()` persists (migration to global scope
  is not yet learned).
- ⚠️ **Interactive events**: for "もぐらたたきゲーム" ("whack-a-mole
  game"), attempts a `mousePressed` handler, but variable-scope bugs
  (`let` inside `setup()`) remain.

These are patterns observed during manual checks on the VibeKid app side.

## Training recipe

- Loss: causal LM, full sequence.
- LoRA: r=16, α=32, dropout=0.05, target_modules=`["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"]`
  (text decoder only).
- dtype: bf16 auto-detected (T4 is CC 7.5 Turing, where float16 over-activates;
  Unsloth absorbs this).
- seq_len: 512.
- batch_size: 1, gradient_accumulation_steps: 8 → effective batch = 8.
- learning_rate: 2e-4, lr_scheduler_type: cosine, warmup_ratio: 0.03.
- num_train_epochs: 3.
- optimizer: paged_adamw_8bit.
- max_grad_norm: 0.3.
- save_steps: 100 (save_total_limit=3), eval_steps: 50, eval on the full
  eval split each call.
- training time: 1608 s (~26.8 min) on Colab Pro A100 40 GB.

Full `training_args.json` (Phase 4; Colab Pro A100 40 GB; `bf16=true`;
~26.8 min):

```json
{
  "output_dir": "/content/drive/MyDrive/vibekid-sft-ckpt",
  "num_train_epochs": 3,
  "per_device_train_batch_size": 1,
  "gradient_accumulation_steps": 8,
  "effective_batch_size": 8,
  "learning_rate": 2e-4,
  "lr_scheduler_type": "cosine",
  "warmup_ratio": 0.03,
  "max_grad_norm": 0.3,
  "weight_decay": 0.01,
  "optim": "paged_adamw_8bit",
  "bf16": true,
  "fp16": false,
  "max_seq_length": 512,
  "dataset_text_field": "text",
  "packing": false,
  "eval_strategy": "steps",
  "eval_steps": 50,
  "save_strategy": "steps",
  "save_steps": 100,
  "save_total_limit": 3,
  "logging_steps": 10,
  "report_to": "none",
  "seed": 42
}
```

Raw JSON (extended version, including LoRA / data / post-train hook):
[`sft/logs/training_args.json`](https://github.com/velocitylabo/vibekid/blob/main/sft/logs/training_args.json).

**Deviations from the Unsloth official recommendation**: against the
standard from [`unsloth/docs/models/gemma-4/train`](https://unsloth.ai/docs/models/gemma-4/train)
(`r=8, lora_alpha=8, num_train_epochs=1, optim=adamw_8bit, lr_scheduler=linear, warmup_steps=5`),
this SFT **strengthens fit to a small dataset (657 items)** as follows:

- Doubled LoRA rank/alpha (r=16, α=32) to expand representational
  capacity.
- Extended to 3 epochs; confirmed eval_loss 6.34 → 3.05 (a ~50%
  reduction).
- dropout=0.05 + cosine scheduler + warmup_ratio=0.03 to suppress
  overfitting.
- `paged_adamw_8bit` for memory efficiency;
  `gradient_accumulation_steps=8` for gradient stability.
- Applied `train_on_responses_only` (Phase 4 / Hypothesis A) as a
  post-hook — aligned with Gemma 4 token naming (`<|turn>` id=105 /
  `<turn|>` id=106) to mask the user-turn loss, taking SFT+oneshot cap
  from 43% to 14%.

## Gemma 4 SFT pitfalls (implementation notes)

Problems encountered while fine-tuning Gemma 4, and the workarounds used.
Recorded as a contribution to the Unsloth community.

### 1. `merged_4bit_forced` NotImplementedError

After `PeftModel.merge_and_unload()`, calling
`save_pretrained(merged_4bit_forced=True)` raised
`NotImplementedError: revert_weight_conversion`. Caused by the lack of a
custom weight-conversion path on Gemma 4's multimodal layer.

**Workaround**: merge in fp32 on Modal CPU 32 GB RAM, then save with
`save_pretrained` as a 7-shard safetensors (see
`sft/scripts/modal-merge.py`). The Colab T4 4-bit path is not used.

### 2. `bf16=True` breaks on T4 GPU

Colab T4 (CC 7.5 Turing) does not natively support bf16 in tensor cores.
Unsloth's bf16 auto-detection returns `False` from the hardware
capability check, so explicitly setting `bf16=True` causes activation
overflow → NaN loss.

**Workaround**: do not pin dtype to `torch.bfloat16`; branch on
`torch.cuda.is_bf16_supported()` (Unsloth's default behavior). fp16 is
not used either (known to cause infinite activations on Gemma 4).

### 3. `chat_template.jinja` not bundled in tokenizer_config ([transformers #45205](https://github.com/huggingface/transformers/issues/45205))

The tokenizer returned by `AutoTokenizer.from_pretrained("google/gemma-4-E2B-it")`
does not include a chat_template; depending on the transformers version,
a call to `apply_chat_template` can raise `KeyError`.

**Workaround**: copy a separate `chat_template.jinja` into the tokenizer
dir, or use Unsloth's `FastModel` loader path (which absorbs this
automatically). This adapter bundles `chat_template.jinja` (see the
[transformers #45205 discussion](https://github.com/huggingface/transformers/issues/45205)).

### 4. Silent breakage of `mm_token_type_ids` (multimodal data collator)

Passing text-only data through Gemma 4's default collator produces a
`None` `mm_token_type_ids`, which can silently cause shape mismatch in
the forward pass (loss does not drop and eval becomes garbage).

**Workaround**: route through Unsloth's collator (`unsloth.data_collator`),
or manually pad with `mm_token_type_ids = torch.zeros_like(input_ids)`.
This SFT uses the former.

## Reproducibility

- **Training notebook**:
  [`sft/notebook/sft-gemma4-e2b.py`](https://github.com/velocitylabo/vibekid/blob/main/sft/notebook/sft-gemma4-e2b.py)
  (jupytext py:percent. To open in Colab, either run `jupytext --to ipynb`
  or upload directly).
- **Kaggle mirror**: <!-- KAGGLE_NOTEBOOK_URL_PLACEHOLDER --> (T4
  one-click reproduction).
- **Eval script**:
  [`sft/scripts/evaluate.mjs`](https://github.com/velocitylabo/vibekid/blob/main/sft/scripts/evaluate.mjs)
  (executes generated code via Playwright; 5-second heartbeat judgement).
- **Synthesis pipeline**:
  [`sft/scripts/build-prompt.mjs`](https://github.com/velocitylabo/vibekid/blob/main/sft/scripts/build-prompt.mjs)
  (a slot-filling meta-prompt calling Gemini 2.5 Flash via OpenRouter).

### Pinned dependencies (Phase 4 training, 2026-04-29)

| Component | Pin | Notes |
|---|---|---|
| Base model (4-bit) | `unsloth/gemma-4-e2b-it-unsloth-bnb-4bit @ a285b07ef4` | The latest at training time; the `base_model_name_or_path` recorded in `adapter_config.json`. |
| Base model (concept reference) | `unsloth/gemma-4-E2B-it @ f0c5915f17` | The latest at training time; the Unsloth mirror of the Google upstream. |
| Unsloth | 2026.4.x series (`!pip install --upgrade` fetched the latest at training time; a post-fix release including the April 8 Gemma 4 universal-bug fix) | The installed version at training time is not preserved in logs. When retraining, an explicit `unsloth==2026.4.x` pin is recommended. |
| transformers | 5.5.0 series | Installed with `--no-deps` in the training notebook; subordinate to Unsloth's dependency resolution. Recent environments carry a `_init_weights` Byte tensor bug; the kaggle-bench notebook embeds a monkey-patch. |
| Hardware | Colab Pro A100 40 GB | bf16 native; 3 epochs ≈ 27 min. |

### Verify run note

The kaggle-bench was re-executed via Run All
(`bench_runs_phase4_verify_*.csv`):

- ✅ **The aggregate cap rate `SFT+oneshot 14.3%` matches exactly**
  (Δ=0 vs the original baseline); the production narrative is
  reproducible.
- ⚠️ **Per-prompt cap distribution redistributes** (baseline:
  `length/short` 67% + `preset/neko` 33%; on re-verify: `length/long`
  67% + `length/normal` 33%; the aggregate is the same).
- Cause: the base model `unsloth/gemma-4-E2B-it` was re-uploaded
  (routine Unsloth HF org update), so unpinned-base behavior shifts
  slightly.
- For **fully deterministic reproduction**, use the kaggle-bench's
  `BASE_REVISION = "f0c5915f17"`. (The current source only pins the base
  path; the LoRA path's base SHA is not pinned via the `adapter_config`
  route — Future Work.)

If you re-run the reproduction experiment, the aggregate numbers should
be reproducible. The per-prompt breakdown may wobble depending on the HF
Hub state of the base model.

### Training-data design

- 800 items were mass-produced from 6 seed genres (animals / physics /
  interactive / visual effects / counters / drawing) via a slot-filling
  meta-prompt; `validate-html` left 757.
- Duplicate rate: after NFKC normalization, 25.5% → 2.0% (root cause
  addressed by the slot design of the meta-prompt; embedding filter
  deemed unnecessary).
- Prompt language: predominantly Japanese hiragana (assumed
  5–7-year-old children).
- Output format: complete p5.js global-mode code (in a fence) containing
  `<canvas>`.

See the VibeKid repo's [`sft/README.md`](https://github.com/velocitylabo/vibekid/tree/main/sft)
for details.

## How to use

### 1. Loading the adapter

To ensure reproducibility, **explicitly pin** the base SHA at training
time. (`unsloth/gemma-4-e2b-it-unsloth-bnb-4bit` is re-uploaded
frequently, so behavior can change if unpinned.)

```python
from peft import PeftModel
from transformers import AutoTokenizer, AutoModelForCausalLM

# The base 4-bit variant SHA at Phase 4 training time
base_id = "unsloth/gemma-4-e2b-it-unsloth-bnb-4bit"
base_revision = "a285b07ef4"  # the latest at Phase 4 training time
adapter_id = "velocitylabo/vibekid-gemma-4-E2B-lora-phase4"

tokenizer = AutoTokenizer.from_pretrained(base_id, revision=base_revision)
base = AutoModelForCausalLM.from_pretrained(
    base_id, revision=base_revision, torch_dtype="auto", device_map="auto"
)
model = PeftModel.from_pretrained(base, adapter_id)
model.eval()
```

Or via Unsloth `FastModel` (recommended; training and inference go
through the same path):

```python
from unsloth import FastModel

# The revision pin goes through FastModel's `revision=` argument (Unsloth ≥ 2026.4)
model, tokenizer = FastModel.from_pretrained(
    model_name="velocitylabo/vibekid-gemma-4-E2B-lora-phase4",
    max_seq_length=2048,
    load_in_4bit=True,
    load_in_8bit=False,
    full_finetuning=False,
    dtype=None,
)
```

### 2. Inference example

```python
prompt = "ぼーるがはねる"  # "a ball bounces"
messages = [{"role": "user", "content": prompt}]
text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

inputs = tokenizer(text, return_tensors="pt").to(model.device)
out = model.generate(**inputs, max_new_tokens=512, do_sample=False)
print(tokenizer.decode(out[0], skip_special_tokens=True))
```

### 3. Using on the LiteRT (Web) path

To run this adapter via `@mediapipe/tasks-genai`, conversion to the
`.task` format (ONNX → LiteRT toolchain) is required. The VibeKid Web app
does not currently use this adapter directly (it runs the base model +
system-prompt path). This repository stands on its own as a LoRA-adapter
technical-verification artifact.

A `transformers.js` + WebGPU + ONNX path (built on
`onnx-community/gemma-4-E2B-it-ONNX`) can implement Web inference using
merged weights for delivery. See Future Work #2 in the VibeKid writeup.

## Limitations

- **Prompt scope**: optimized for generating child-targeted p5.js
  global-mode code; anything else (general-purpose JS / Python /
  multilingual) is not guaranteed.
- **Data scale**: 657 train / 100 eval items is a research-PoC scale. For
  production use, more diversity / scale is needed.
- **Structural constraint**: only the text decoder is trained;
  multimodal output (image / audio) stays as the base model produces.
- **Eval reliability**: `exec_success_rate` only checks machine
  execution; **semantic correctness (does the intended picture appear)
  is evaluated separately**. Intent matching is checked manually on the
  VibeKid app side.

## Citation / Acknowledgements

### Base model

Google DeepMind. Gemma 4 (`google/gemma-4-E2B-it`).

### Hackathon

```bibtex
@misc{gemma4hackathon2026,
  author = {Ballantyne, Ian and Cameron, Glenn and Cruz, Mar{\'i}a and Lacombe, Olivier and Quan, Kristen and Sanseviero, Omar},
  title = {The Gemma 4 Good Hackathon},
  year = {2026},
  publisher = {Kaggle},
  url = {https://kaggle.com/competitions/gemma-4-good-hackathon}
}
```

### Tooling

- [Unsloth AI](https://github.com/unslothai/unsloth) — Gemma 4 SFT
  acceleration; `FastModel` / `train_on_responses_only`.
- [transformers](https://github.com/huggingface/transformers) (5.5.0
  series) — model loading; SFTTrainer.
- [peft](https://github.com/huggingface/peft) — LoRA adapter structure.
- [trl](https://github.com/huggingface/trl) — `SFTTrainer` /
  `SFTConfig`.
- [bitsandbytes](https://github.com/TimDettmers/bitsandbytes) — 4-bit
  quantization.

### Synthesis pipeline

- [OpenRouter](https://openrouter.ai/) → `gemini-2.5-flash` for training
  data synthesis (757 items).
- The synthesis meta-prompt is not bundled in this release (managed in
  the upstream development repository).

## License

This repository is composed of **multiple-license derivatives**:

- **Adapter weights** (`adapter_model.safetensors`,
  `adapter_config.json`): inherits the [Gemma Terms of Use](https://ai.google.dev/gemma/terms)
  as a derivative of the base model `google/gemma-4-E2B-it`.
- **Tokenizer / chat_template** (`tokenizer.json`,
  `tokenizer_config.json`, `chat_template.jinja`,
  `processor_config.json`): redistributed from Gemma upstream; likewise
  governed by the Gemma Terms.
- **Model-card text (`README.md`) / configs of our authorship**:
  distributed under [CC-BY 4.0](https://creativecommons.org/licenses/by/4.0/).

This dual-license is intended to satisfy both the Kaggle Gemma 4
Hackathon submission license requirements (CC-BY 4.0) and the pretrained
model exception simultaneously.

## Repo links

- VibeKid Web app: <https://github.com/velocitylabo/vibekid>.
- Kaggle submission: <!-- KAGGLE_SUBMISSION_URL_PLACEHOLDER -->.
