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

> 日本語の子供向け自然言語指示から p5.js コードを生成するための Gemma 4 E2B LoRA。Kaggle Gemma 4 Hackathon (2026) 提出作品。

`google/gemma-4-E2B-it` を **757 件の子供向け p5.js synthesis dataset** で QLoRA 微調整した adapter。on-device LiteRT (`@mediapipe/tasks-genai`) ベースの Web アプリ [VibeKid](https://github.com/velocitylabo/vibekid) で利用される前提。

## Highlights

- **Base model**: `google/gemma-4-E2B-it`
- **Adapter type**: QLoRA r=16 α=32（text decoder layers のみ、multimodal 層は学習対象外）
- **Train data**: 657 件（4/19 split、OpenRouter Gemini 2.5 Flash 合成）
- **Eval data**: 100 件（カテゴリ均等、4/19 cutoff）
- **Hardware**: Colab Pro T4 16GB、1 epoch ≈ 2-3h
- **Adapter size**: ~50MB

## Ablation summary

<!-- ABLATION_BAR_CHART_PLACEHOLDER -->

| Setup | exec_success_rate | n |
|---|---|---|
| Baseline (`google/gemma-4-E2B-it`, raw) | <!-- BASELINE_RAW --> / 100 | 100 |
| Baseline (`google/gemma-4-E2B-it` + app one-shot system prompt) | <!-- BASELINE_SYS --> / 100 | 100 |
| **Fine-tuned (this adapter)** | **<!-- SFT_SCORE --> / 100** | 100 |

`exec_success_rate` = "生成された p5.js コードが (a) Playwright 環境で 5 秒間 SyntaxError / ReferenceError なしに `setup()` + `draw()` を実行し、(b) heartbeat を返す" 件数 / 全件。詳細は `evaluate.mjs` を参照。

## Training recipe

- Loss: causal LM, full sequence
- LoRA: r=16, α=32, dropout=0.05, target_modules=`["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"]` (text decoder 限定)
- dtype: bf16 自動判定（T4 は CC 7.5 Turing で float16 過剰活性化、Unsloth で吸収）
- seq_len: 512
- batch_size: 1, gradient_accumulation_steps: 8 → effective batch = 8
- learning_rate: 2e-4, lr_scheduler_type: cosine, warmup_ratio: 0.03
- num_train_epochs: 1
- optimizer: paged_adamw_8bit
- max_grad_norm: 0.3
- save_steps: 50, evaluation_strategy: "no"（NotebookProgressCallback bug 回避、`evaluate.mjs` で post-hoc 計測）

完全な `training_args.json`:

<!-- TRAINING_ARGS_PLACEHOLDER -->

## Gemma 4 SFT 固有の地雷（実装メモ）

Gemma 4 を fine-tune する際に踏んだ問題と回避方法。Unsloth コミュニティへの貢献として残す。

### 1. `merged_4bit_forced` NotImplementedError

`PeftModel.merge_and_unload()` 後に `save_pretrained(merged_4bit_forced=True)` で `NotImplementedError: revert_weight_conversion` が発生。Gemma 4 の multimodal layer に custom weight conversion path が未実装な点に起因。

**回避**: Modal CPU 32GB RAM で fp32 merge → safetensors 7-shard で save_pretrained（`sft/scripts/modal-merge.py` 参照）。Colab T4 の 4-bit 経路は採用しない。

### 2. T4 GPU で `bf16=True` が破綻

Colab T4 (CC 7.5 Turing) は bf16 を tensor-core ネイティブにサポートしない。Unsloth の bf16 自動判定が hardware capability から `False` を返すため、明示的に `bf16=True` 指定すると activation overflow で loss NaN。

**回避**: dtype を `torch.bfloat16` 固定にせず、`torch.cuda.is_bf16_supported()` で分岐（Unsloth 既定動作）。fp16 は使わない（Gemma 4 で infinite activation 既知）。

### 3. `chat_template.jinja` が tokenizer_config に非同梱（[transformers #45205](https://github.com/huggingface/transformers/issues/45205)）

`AutoTokenizer.from_pretrained("google/gemma-4-E2B-it")` で得た tokenizer に chat_template が含まれず、`apply_chat_template` 呼び出しで KeyError が出る場合あり（transformers の version によって挙動が変わる）。

**回避**: tokenizer dir に `chat_template.jinja` を別途コピー / Unsloth の `FastModel` ローダ経路では自動で吸収される。本 adapter には `chat_template.jinja` を同梱（[transformers #45205 議論を参照](https://github.com/huggingface/transformers/issues/45205)）。

### 4. `mm_token_type_ids` の silent 崩壊（multimodal データ collator）

text-only データを Gemma 4 default collator に通すと `mm_token_type_ids` が None で生成され、forward で silent に shape mismatch を起こすことがある（loss は降らないが eval が出鱈目）。

**回避**: Unsloth の collator (`unsloth.data_collator`) を経由するか、自前で `mm_token_type_ids = torch.zeros_like(input_ids)` をパディング追加する。本 SFT は前者を採用。

## Reproducibility

- **Training notebook**: [`sft/notebook/sft-gemma4-e2b.py`](https://github.com/velocitylabo/vibekid/blob/main/sft/notebook/sft-gemma4-e2b.py)（jupytext py:percent。Colab で開くには `jupytext --to ipynb` か直接 upload）
- **Kaggle mirror**: <!-- KAGGLE_NOTEBOOK_URL_PLACEHOLDER -->（T4 one-click 再現）
- **Eval script**: [`sft/scripts/evaluate.mjs`](https://github.com/velocitylabo/vibekid/blob/main/sft/scripts/evaluate.mjs)（Playwright で生成コードを実行、5 秒 heartbeat 判定）
- **Synthesis pipeline**: [`sft/scripts/build-prompt.mjs`](https://github.com/velocitylabo/vibekid/blob/main/sft/scripts/build-prompt.mjs)（OpenRouter 経由で Gemini 2.5 Flash を呼ぶ slot-filling meta-prompt）

### Training data 設計

- 6 seed ジャンル（動物 / 物理 / interactive / 視覚エフェクト / カウンター / お絵描き）から slot-filling meta-prompt で 800 件量産、validate-html で 757 件残存
- 重複率: NFKC 正規化後 25.5% → 2.0%（meta-prompt の slot 設計で根本対策、embedding filter 不要と判断）
- prompt 言語: 日本語ひらがな主体（5-7 歳児想定）
- output 形式: `<canvas>` を含む完結 p5.js global mode コード（fence 形式）

詳細は [VibeKid repo の `sft/README.md`](https://github.com/velocitylabo/vibekid/tree/main/sft) を参照。

## How to use

### 1. Adapter のロード

```python
from peft import PeftModel
from transformers import AutoTokenizer, AutoModelForCausalLM

base_id = "google/gemma-4-E2B-it"
adapter_id = "<repo-id-of-this-card>"  # 例: user/vibekid-gemma-4-E2B-lora

tokenizer = AutoTokenizer.from_pretrained(base_id)
base = AutoModelForCausalLM.from_pretrained(base_id, torch_dtype="auto", device_map="auto")
model = PeftModel.from_pretrained(base, adapter_id)
model.eval()
```

### 2. Inference 例

```python
prompt = "ぼーるがはねる"
messages = [{"role": "user", "content": prompt}]
text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

inputs = tokenizer(text, return_tensors="pt").to(model.device)
out = model.generate(**inputs, max_new_tokens=512, do_sample=False)
print(tokenizer.decode(out[0], skip_special_tokens=True))
```

### 3. LiteRT (Web) 経路で使う

本 adapter を `@mediapipe/tasks-genai` 経由で動かすには `.task` 形式への変換が必要（ONNX → LiteRT toolchain）。VibeKid Web アプリ側はこの adapter を直接利用していない（base model + system prompt 経路。本リポジトリは LoRA adapter 単体の技術検証 artifact として独立）。

詳細は VibeKid repo の `docs/post-submission-litert-path.md` を参照。

## Limitations

- **prompt 範囲**: 子供向け p5.js global mode コード生成に最適化、それ以外（汎用 JS / Python / 多言語）は未保証
- **データ規模**: 657 件 train / 100 件 eval は研究用 PoC レベル。production 用途には更なる多様性 / scale が必要
- **構造制約**: text decoder のみ学習、multimodal 出力（画像 / 音声）は base model のまま
- **eval 信頼性**: `exec_success_rate` は機械実行の通過のみで、**意味的 correctness（意図通りの絵が出るか）は別評価**。意図一致は VibeKid アプリ側で目視確認している

## Citation / Acknowledgements

- Base: Google DeepMind. Gemma 4 (`google/gemma-4-E2B-it`).
- Tooling: [Unsloth AI](https://github.com/unslothai/unsloth)、[transformers](https://github.com/huggingface/transformers)、[peft](https://github.com/huggingface/peft)
- Kaggle Gemma 4 Hackathon（締切 2026-05-18）

## License

[Gemma Terms of Use](https://ai.google.dev/gemma/terms) 準拠。本 LoRA adapter も同一ライセンスで配布。

## Repo links

- VibeKid Web app: <https://github.com/velocitylabo/vibekid>

- Kaggle submission: <!-- KAGGLE_SUBMISSION_URL_PLACEHOLDER -->
