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
  - archive
pipeline_tag: text-generation
---

# VibeKid — Gemma 4 E2B LoRA (Phase 3 archive、cap rate 43%)

> **⚠️ archive notice**: 本 repo は **Phase 3 LoRA** (2026-04-25 訓練版)。Kaggle Gemma 4 Hackathon (2026) 提出本体は **Phase 4** ([`velocitylabo/vibekid-gemma-4-E2B-lora-phase4`](https://huggingface.co/velocitylabo/vibekid-gemma-4-E2B-lora-phase4)) を採用。Phase 4 では `train_on_responses_only` + Gemma 4 終端 token (`<turn|>`) 命名修正 + `pad_token != eos_token` の **Tier 1 介入**を施し、Kaggle bench SFT+oneshot cap rate が **43% → 14%** に改善した。
>
> 本 Phase 3 archive は writeup Section B narrative で **Tier 1 介入前の baseline (cap rate 43%)** として並列言及される比較対象であり、開発履歴 + 第三者検証用に保持する。最新の評価数値 / Reproducibility 詳細 / How to use / License は [Phase 4 model card](https://huggingface.co/velocitylabo/vibekid-gemma-4-E2B-lora-phase4) に集約。

## Highlights (Phase 3 baseline)

- **Base model**: `google/gemma-4-E2B-it`
- **Adapter type**: QLoRA r=16 α=32 (text decoder 限定、multimodal 層は学習対象外)
- **Train data**: 657 件 (2026-04-19 split、OpenRouter Gemini 2.5 Flash 合成)
- **Eval data**: 100 件 (カテゴリ均等、2026-04-19 cutoff)
- **Hardware**: Colab Pro A100 40GB (bf16 native)、3 epochs ≈ 27min
- **Adapter size**: ~50MB
- **訓練日**: 2026-04-25
- **Kaggle bench cap rate (SFT+oneshot)**: **43%** ← Phase 4 比較対象、後段で 14% に改善

## Training metrics (validation loss curve)

Phase 3 / Phase 4 で訓練 hyperparameter は同一 (差分は Tier 1 介入のみ、本 archive は介入前)。

| Step | Training Loss | Validation Loss |
|------|---------------|-----------------|
| 50   | 0.732 | 3.00 |
| 100  | 0.507 | **2.98** (最低) |
| 150  | 0.416 | 3.06 |
| 200  | 0.336 | 3.05 |
| 249  | 0.328 | 3.05 |

3 epochs (249 steps、effective batch=8) で **train_loss 0.99 → 0.33** (66% 低下)、val_loss は step 100 付近で収束 (軽微 overfit 兆候、step 100 〜 249 でほぼ横ばい)。1 epoch run 時の eval_loss 6.34 → 3 epoch run 3.05 と **半分以下に改善**。

注: 訓練終了直後の追加 eval で `eval_loss=NaN` が一度観測 (pad token = eos token に起因の可能性、attention_mask 警告を伴う)。**訓練中の eval は全て finite** で、adapter 自体は健全。

## Phase 4 で改善した narrative 差分

Phase 4 では同一 train.jsonl / LoRA config / epoch 数で **Tier 1 介入** を適用:

1. **`train_on_responses_only`** — user turn の loss を mask、response token のみで gradient
2. **Gemma 4 終端 token 命名修正** — Gemma 4 chat 終端は `<turn|>` (id=106) / 開始は `<|turn>` (id=105)、Gemma 1/2/3 系 (`<end_of_turn>`) と命名が異なる。`train_on_responses_only` の `instruction_part` literal を Gemma 4 仕様に整合
3. **`pad_token != eos_token`** — TRL 推奨、EOS 学習信号減衰を回避

→ 同 train.jsonl (657 件) / 同 LoRA config (r=16, α=32) / 同 epoch 数 (3) で再訓練 → 同 bench 再走 (clean A/B):

| 構成 | Phase 3 (本 archive) | Phase 4 (current) | Δ |
|---|---:|---:|---:|
| SFT + oneshot cap rate | **43%** | **14%** | **−29pt** |

Web app の本番構成は oneshot 経路 (V1 system prompt) のため、end-user UX 上のこの差は決定的。詳細は [Phase 4 model card](https://huggingface.co/velocitylabo/vibekid-gemma-4-E2B-lora-phase4) Section "exec_success_rate + 副次 metrics" + writeup Section B 参照。

## Gemma 4 SFT 固有の地雷 (Phase 3 / Phase 4 共通の研究 contribution)

Gemma 4 を fine-tune する際に踏んだ問題と回避方法は Phase 4 model card に集約済 ([Section "Gemma 4 SFT 固有の地雷"](https://huggingface.co/velocitylabo/vibekid-gemma-4-E2B-lora-phase4#gemma-4-sft-固有の地雷実装メモ) 参照)。4 項目: `merged_4bit_forced` NotImplementedError / T4 GPU `bf16=True` 破綻 / `chat_template.jinja` 非同梱 ([transformers #45205](https://github.com/huggingface/transformers/issues/45205)) / `mm_token_type_ids` silent 崩壊。

## Reproducibility / How to use / License

詳細は [Phase 4 model card](https://huggingface.co/velocitylabo/vibekid-gemma-4-E2B-lora-phase4) に集約。本 Phase 3 archive は Phase 4 と同じ訓練 / 推論 / license スキームを継承 (Tier 1 介入の有無のみが差分)。

License (Phase 4 と同じ dual-license):

- **Adapter weights** (`adapter_model.safetensors`、`adapter_config.json`): base model `google/gemma-4-E2B-it` の派生物として [Gemma Terms of Use](https://ai.google.dev/gemma/terms) を継承
- **Tokenizer / chat_template**: Gemma upstream のものを再配布、同様に Gemma Terms 準拠
- **Model card 本文 / configs of our authorship**: [CC-BY 4.0](https://creativecommons.org/licenses/by/4.0/) で配布

## Citation / Acknowledgements

- **Base model**: Google DeepMind. Gemma 4 (`google/gemma-4-E2B-it`).
- **Tooling**: [Unsloth AI](https://github.com/unslothai/unsloth)、[transformers](https://github.com/huggingface/transformers)、[peft](https://github.com/huggingface/peft)
- **Hackathon**: Kaggle Gemma 4 Good Hackathon (2026)

## Repo links

- **VibeKid Web app**: <https://github.com/velocitylabo/vibekid>
- **Phase 4 LoRA (current submission)**: <https://huggingface.co/velocitylabo/vibekid-gemma-4-E2B-lora-phase4>
- **本 Phase 3 archive (このページ)**: <https://huggingface.co/velocitylabo/vibekid-gemma-4-E2B-lora>
