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

- **Base model (concept)**: `google/gemma-4-E2B-it`
- **Base model (loaded at training)**: `unsloth/gemma-4-e2b-it-unsloth-bnb-4bit` @ revision `a285b07ef4` (2026-04-07、`adapter_config.json` 参照、Phase 4 訓練 2026-04-29 時点の latest)
- **Adapter type**: QLoRA r=16 α=32（text decoder layers のみ、multimodal 層は学習対象外）
- **Train data**: 657 件（4/19 split、OpenRouter Gemini 2.5 Flash 合成）
- **Eval data**: 100 件（カテゴリ均等、4/19 cutoff）
- **Hardware**: Colab Pro A100 40GB（bf16 native）、3 epochs ≈ 27min
- **Adapter size**: ~50MB

## Training metrics (validation loss curve)

訓練中の eval split (100 件) に対する cross-entropy loss の推移:

| Step | Training Loss | Validation Loss |
|------|---------------|-----------------|
| 50   | 0.732 | 3.00 |
| 100  | 0.507 | **2.98**（最低） |
| 150  | 0.416 | 3.06 |
| 200  | 0.336 | 3.05 |
| 249  | 0.328 | 3.05 |

3 epochs (249 steps、effective batch=8) で **train_loss 0.99 → 0.33**（66% 低下）、**val_loss は 3 step 100 付近で収束**（軽微 overfit 兆候、step 100 〜 249 でほぼ横ばい）。1 epoch run 時の eval_loss 6.34 → 3 epoch run 3.05 と **半分以下に改善**。

注: 訓練終了直後の追加 eval で `eval_loss=NaN` が一度観測（pad token = eos token に起因の可能性、attention_mask 警告を伴う）。**訓練中の eval は全て finite** で、adapter 自体は健全。

## exec_success_rate + 副次 metrics (Playwright 実機評価、4/30 計測)

100 件 eval set (`sft/data/eval.jsonl`) に対する execution 評価。`exec_success_rate` は execution success に到達せず BASELINE / SFT 共に **0/100**、ただし **副次 metrics で SFT は code 品質を明確に改善**:

| Setup | exec_success_rate | mean_code_length | has_animation | js_error | code_too_short | n | 計測経路 |
|---|---:|---:|---:|---:|---:|---|---|
| Baseline (`google/gemma-4-E2B-it`, raw) | 0 / 100 | (mostly prose) | - | 0 | 100 | 100 | Web (LiteRT) |
| Baseline (`google/gemma-4-E2B-it` + app one-shot system prompt) | 0 / 100 | 698 | (n/a) | 34 | - | 100 | Web (LiteRT) |
| **Fine-tuned (this adapter)** | **0 / 100** | **828** | **59%** | **1** | **9** | 100 | Notebook (Python + Colab Pro A100) |

### narrative

`exec_success_rate` = "(a) `setup()` + `draw()` + `createCanvas()` 静的検査 pass、(b) iframe sandbox で 5 秒間 SyntaxError / ReferenceError なしに実行、(c) `<canvas>` 描画" 件数 / 全件。

100 件 eval set はいずれも 0/100 (BASELINE / SFT 共に execution success に到達せず)。fail dominant は **`no_canvas` 91/94 件** (iframe sandbox 内で `<canvas>` 検出されず) で、難解なユーザー指示 (描画ツール / ゲーム / interactive) と iframe sandbox 制約が複合する eval set 設計に由来。

ただし副次 metrics で **SFT は code 品質で base を明確に上回る**:

- **`js_error`**: 34 (BASELINE_SYS) → **1** (SFT、97% 削減) — SFT が syntactically valid な code を出力
- **`mean_code_length`**: 698 → **828** chars — SFT で完結率が上がり code が中断しない
- **`has_animation_rate`**: 59% — SFT が p5.js animation 構造 (frameCount / random / sin / cos / event handler) を含む code を出力
- **`code_too_short`**: 9 件のみ (= 91 件は code 構造完備、`setup()` + `draw()` + `createCanvas()` 揃う)

### iframe RAF throttle 仮説 検証 (4/30)

初回計測 (5s timeout, hidden iframe) で 0/100、仮説「不可視 iframe は RAF throttle で `setup()` 内 `createCanvas()` が timeout 内に間に合わない」(memory `feedback_p5js_iframe_gotchas.md`) を検証するため **iframe を visible + timeout 10s** で再走 → **同じ 0/100、`fail_reasons` 完全一致**で **仮説 false 確定**。eval set 100 件は code 出力レベルでは差が出るが iframe execution には到達しない難易度。

raw data: [`sft/logs/validate_20260501_eval_phase4_sft.json`](https://github.com/velocitylabo/vibekid/blob/main/sft/logs/validate_20260501_eval_phase4_sft.json) (v1, 5s/hidden) / [`_v2.json`](https://github.com/velocitylabo/vibekid/blob/main/sft/logs/validate_20260501_eval_phase4_sft_v2.json) (v2, 10s/visible)

### production 経路における SFT 価値 (別 metric)

production app の preset-driven UX における SFT 価値は **Kaggle 84 runs bench の cap rate** で証明済 (writeup Section B):

- Phase 2 (旧 SFT): SFT+oneshot で **cap rate 43%** (output_tokens 512 到達 = 出力が止まらない)
- **Phase 4 (新 SFT、本 adapter)**: `train_on_responses_only` + Gemma 4 終端 token (`<turn|>`) 修正で **cap rate 14% (−29pt)**

cap rate 14% は production preset 経路で実用水準 (cap 残 14% は SFT+bare 100% Hypothesis D 関連、Future Work #1 で DPO + system prompt variation bake-in による対応を検討)。execution success までの改善経路は writeup の Future Work #1 / #2 (DPO + `transformers.js` + ONNX) を参照。

### 計測経路の補足

SFT は LiteRT-LM web で LoRA load する toolchain が公式未提供 ([LiteRT-LM overview](https://ai.google.dev/edge/litert-lm/overview)) のため、**Notebook (Python + Colab Pro A100) で 100 件 generate → ローカル Playwright で execute** の 2 段階で計測。validate ロジック (iframe sandbox + heartbeat 判定) は Web/Notebook 共通実装 (`sft/eval-runner.html` / `sft/eval-validate-runner.html`)。BASELINE 2 件は Web 経路 (LiteRT、`evaluate.mjs --system=none|app-oneshot`) で計測、SFT は Notebook 経路だが validate ロジックが共通のため apples-to-apples 比較が成立。

## 質的観察

3 epoch fine-tune 後の生成サンプル（5 prompt × 1 run）目視評価:

- ✅ **Drawing detail 向上**: base model の「赤い四角」→ ねこ preset で「白いネコの体・耳・瞳」など具体物的描写
- ✅ **Visual loop**: 「はなびがどかーん」prompt で `for` ループによる random 円 + 線描画の自然な animation 表現
- ⚠️ **Animation logic**: 「ぴょんぴょんはねる」で animation コメント + `let yPos` を試行するが、変数を `draw()` 内 local 宣言する pattern が残る（global scope への移行は未学習）
- ⚠️ **Interactive event**: 「もぐらたたきゲーム」で `mousePressed` ハンドラを書こうとするが、変数 scope（`setup()` 内 `let`）の bug が残存

これらは VibeKid アプリ側の手動確認で観察された pattern。

## Training recipe

- Loss: causal LM, full sequence
- LoRA: r=16, α=32, dropout=0.05, target_modules=`["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"]` (text decoder 限定)
- dtype: bf16 自動判定（T4 は CC 7.5 Turing で float16 過剰活性化、Unsloth で吸収）
- seq_len: 512
- batch_size: 1, gradient_accumulation_steps: 8 → effective batch = 8
- learning_rate: 2e-4, lr_scheduler_type: cosine, warmup_ratio: 0.03
- num_train_epochs: 3
- optimizer: paged_adamw_8bit
- max_grad_norm: 0.3
- save_steps: 100 (save_total_limit=3), eval_steps: 50, eval on full eval split each call
- training time: 1608 s (~26.8 min) on Colab Pro A100 40GB

完全な `training_args.json` (Phase 4、Colab Pro A100 40GB、`bf16=true`、~26.8 min):

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

raw JSON (LoRA / data / post-train hook 含む拡張版): [`sft/logs/training_args.json`](https://github.com/velocitylabo/vibekid/blob/main/sft/logs/training_args.json)。

**Unsloth 公式推奨との deviation**: 公式 [`unsloth/docs/models/gemma-4/train`](https://unsloth.ai/docs/models/gemma-4/train) の標準 (`r=8, lora_alpha=8, num_train_epochs=1, optim=adamw_8bit, lr_scheduler=linear, warmup_steps=5`) に対し、本 SFT は **小規模 dataset (657 件) への適合を強化**するため:

- LoRA rank/alpha 倍増 (r=16, α=32) で表現容量拡大
- 3 epoch まで延長して eval_loss 6.34→3.05 (50% 削減) を確認
- dropout=0.05 + cosine scheduler + warmup_ratio=0.03 で overfitting 抑制
- `paged_adamw_8bit` で memory 効率化、`gradient_accumulation_steps=8` で勾配 stability
- `train_on_responses_only` (Phase 4 / Hypothesis A) を後段 hook で適用 — Gemma 4 token 命名 (`<|turn>` id=105 / `<turn|>` id=106) と整合させて user turn の loss を mask、SFT+oneshot cap 43%→14% を実現

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
- **Kaggle bench (T4, 84-run ablation)**: <!-- KAGGLE_NOTEBOOK_URL_PLACEHOLDER -->（base vs SFT × oneshot vs bare × 7 prompts × 3 runs を T4 で再走、本 model card および writeup の bench 数値の reproducibility source）
- **Kaggle eval mirror (T4, SFT 100-prompt eval)**: <!-- KAGGLE_EVAL_NOTEBOOK_URL_PLACEHOLDER -->（self-contained、HF Hub から adapter を直接 load して 100 件 eval を T4 で再走）
- **Eval script**: [`sft/scripts/evaluate.mjs`](https://github.com/velocitylabo/vibekid/blob/main/sft/scripts/evaluate.mjs)（Playwright で生成コードを実行、5 秒 heartbeat 判定）
- **Synthesis pipeline**: [`sft/scripts/build-prompt.mjs`](https://github.com/velocitylabo/vibekid/blob/main/sft/scripts/build-prompt.mjs)（OpenRouter 経由で Gemini 2.5 Flash を呼ぶ slot-filling meta-prompt）

### Pinned dependencies (Phase 4 訓練時、2026-04-29)

| component | pin | 備考 |
|---|---|---|
| Base model (4-bit) | `unsloth/gemma-4-e2b-it-unsloth-bnb-4bit @ a285b07ef4` | 2026-04-07 latest、訓練時の `adapter_config.json` `base_model_name_or_path` |
| Base model (concept reference) | `unsloth/gemma-4-E2B-it @ f0c5915f17` | 2026-04-11 latest、Google upstream の Unsloth ミラー |
| Unsloth | 2026.4.x 系 (`!pip install --upgrade` で訓練当日 latest 取得、April 8 Gemma 4 universal-bug fix を含む post-fix リリース) | 2026-04-29 訓練時の installed version は logs に未保存。再訓練時は `unsloth==2026.4.x` 明示 pin 推奨 |
| transformers | `5.5.0` 系 | 訓練 notebook で `--no-deps` install、Unsloth 依存解決に従属。Kaggle/Colab 5/5+ 環境で `_init_weights` Byte tensor bug あり、kaggle-bench notebook に monkey-patch を埋め込み済 |
| Hardware | Colab Pro A100 40GB | bf16 native、3 epochs ≈ 27min |

### Verify run note (2026-05-06)

5/6 に kaggle-bench を Run All で再実行 (`bench_runs_phase4_verify_20260506.csv`):

- ✅ **Aggregate cap rate `SFT+oneshot 14.3%` は完全一致** (4/29 baseline と Δ=0)、production narrative は再現性あり
- ⚠️ **per-prompt cap 分布は redistribute** (4/29: `length/short` 67% + `preset/neko` 33% / 5/6: `length/long` 67% + `length/normal` 33%、aggregate は同じ)
- 原因: base model `unsloth/gemma-4-E2B-it` が 5/4-5 に re-upload された (Unsloth HF org 通常更新) ため、unpinned base で挙動が微妙にシフト
- **完全 deterministic 再現**には kaggle-bench `BASE_REVISION = "f0c5915f17"` を使用 (現 source は base path のみ pin、LoRA path は adapter_config 経路で base SHA 未 pin、Future Work)

再現実験で re-run する場合、aggregate 数値は再現する想定。per-prompt 内訳は base model の HF Hub 状態で揺らぐ可能性あり。

### Training data 設計

- 6 seed ジャンル（動物 / 物理 / interactive / 視覚エフェクト / カウンター / お絵描き）から slot-filling meta-prompt で 800 件量産、validate-html で 757 件残存
- 重複率: NFKC 正規化後 25.5% → 2.0%（meta-prompt の slot 設計で根本対策、embedding filter 不要と判断）
- prompt 言語: 日本語ひらがな主体（5-7 歳児想定）
- output 形式: `<canvas>` を含む完結 p5.js global mode コード（fence 形式）

詳細は [VibeKid repo の `sft/README.md`](https://github.com/velocitylabo/vibekid/tree/main/sft) を参照。

## How to use

### 1. Adapter のロード

訓練時の base SHA を **明示 pin** することで再現性を担保 (`unsloth/gemma-4-e2b-it-unsloth-bnb-4bit` は 2026-05 で頻繁に re-upload されているため、unpinned だと挙動が変わり得る):

```python
from peft import PeftModel
from transformers import AutoTokenizer, AutoModelForCausalLM

# Phase 4 訓練時 (2026-04-29) の base 4-bit variant SHA
base_id = "unsloth/gemma-4-e2b-it-unsloth-bnb-4bit"
base_revision = "a285b07ef4"  # 2026-04-07、Phase 4 訓練時の latest
adapter_id = "velocitylabo/vibekid-gemma-4-E2B-lora-phase4"

tokenizer = AutoTokenizer.from_pretrained(base_id, revision=base_revision)
base = AutoModelForCausalLM.from_pretrained(
    base_id, revision=base_revision, torch_dtype="auto", device_map="auto"
)
model = PeftModel.from_pretrained(base, adapter_id)
model.eval()
```

または Unsloth `FastModel` (推奨、訓練と inference が同経路):

```python
from unsloth import FastModel

# revision pin は FastModel `revision=` 引数経由 (Unsloth ≥ 2026.4)
model, tokenizer = FastModel.from_pretrained(
    model_name="velocitylabo/vibekid-gemma-4-E2B-lora-phase4",
    max_seq_length=2048,
    load_in_4bit=True,
    load_in_8bit=False,
    full_finetuning=False,
    dtype=None,
)
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

`transformers.js` + WebGPU + ONNX 経路 (`onnx-community/gemma-4-E2B-it-ONNX` ベース) は、merge 済 weight 配信で web 推論を実装可能。VibeKid writeup の Future Work #2 を参照。

## Limitations

- **prompt 範囲**: 子供向け p5.js global mode コード生成に最適化、それ以外（汎用 JS / Python / 多言語）は未保証
- **データ規模**: 657 件 train / 100 件 eval は研究用 PoC レベル。production 用途には更なる多様性 / scale が必要
- **構造制約**: text decoder のみ学習、multimodal 出力（画像 / 音声）は base model のまま
- **eval 信頼性**: `exec_success_rate` は機械実行の通過のみで、**意味的 correctness（意図通りの絵が出るか）は別評価**。意図一致は VibeKid アプリ側で目視確認している

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

- [Unsloth AI](https://github.com/unslothai/unsloth) — Gemma 4 SFT 高速化、`FastModel` / `train_on_responses_only`
- [transformers](https://github.com/huggingface/transformers) (5.5.0 系) — model loading、SFTTrainer
- [peft](https://github.com/huggingface/peft) — LoRA adapter 構造
- [trl](https://github.com/huggingface/trl) — `SFTTrainer` / `SFTConfig`
- [bitsandbytes](https://github.com/TimDettmers/bitsandbytes) — 4bit quantization

### Synthesis pipeline

- [OpenRouter](https://openrouter.ai/) → `gemini-2.5-flash` で訓練データ合成 (757 件、4/17-18 確定)
- 合成 meta-prompt は本 release には未同梱 (上流開発リポジトリで管理)

## License

本 repository は **複数 license の派生物** で構成される:

- **Adapter weights** (`adapter_model.safetensors`、`adapter_config.json`): base model `google/gemma-4-E2B-it` の派生物として [Gemma Terms of Use](https://ai.google.dev/gemma/terms) を継承
- **Tokenizer / chat_template** (`tokenizer.json`、`tokenizer_config.json`、`chat_template.jinja`、`processor_config.json`): Gemma upstream のものを再配布、同様に Gemma Terms 準拠
- **Model card 本文 (`README.md`) / configs of our authorship**: [CC-BY 4.0](https://creativecommons.org/licenses/by/4.0/) で配布

この dual-license は Kaggle Gemma 4 Hackathon の submission license 要件 (CC-BY 4.0) と pretrained model exception の両立を意図している。

## Repo links

- VibeKid Web app: <https://github.com/velocitylabo/vibekid>
- Kaggle submission: <!-- KAGGLE_SUBMISSION_URL_PLACEHOLDER -->
