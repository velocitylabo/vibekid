# ---
# jupyter:
#   jupytext:
#     formats: py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
# ---

# %% [markdown]
# # VibeKid — Kaggle bench notebook for Gemma 4 E2B + LoRA
#
# **Status: 骨格のみ（4/25, #134）。実装は TODO コメントで残し、SFT 本番完了後に埋める。**
#
# ## 目的（#134）
#
# 単一 Kaggle Notebook で「Run All」実行可能な Gemma 4 E2B bench を提供する。
# 閲覧者が Kaggle 環境で 1 click 実走できることが価値。
#
# ## Web bench との関係（重要）
#
# `#124` / `#127` の bench は **ブラウザ + WebGPU + LiteRT (`@mediapipe/tasks-genai`)**
# で計測した Web 経路の数値。本 notebook は Python + `transformers` + T4 GPU 経路で
# 計測するため数値は直接比較不可。**Kaggle Notebook は別 backend での再現可能性
# を確保するもの**で、Web 数値は `#127` のベンチログを参照する。
#
# 列名は Web 版（TTFT / decode tok/s / output chars）と整合させ、対比表で並べる。
#
# ## ablation
#
# 元の `#126` 案にあった「Alpine reactivity ablation」は Web frontend 固有のため
# Kaggle に移植不可。代わりに以下 3 ablation:
#
# 1. **prompt length**: short / normal / long
# 2. **oneshot**: system prompt に few-shot example を含めるか否か
# 3. **base vs SFT**: base model 単体 vs LoRA adapter 適用
#
# ## Reproducibility
#
# - Kaggle T4 GPU（無料 30h/week）
# - Gemma 4 E2B（HF Hub `google/gemma-4-E2B-it`、gated アクセス承認必要）
# - LoRA adapter（HF Hub、本 PoC では `<HF_REPO_ID>` プレースホルダ）
# - 全体実行時間目標: 15-20 分
#
# ---

# %% [markdown]
# ## 0. 環境確認（Kaggle T4 想定）
#
# Kaggle で T4 を有効化（Settings > Accelerator > GPU T4 x2）
# 1 GPU で十分（Gemma 4 E2B は 4-bit 量子化で ~3GB）

# %%
# TODO: nvidia-smi / torch.cuda 確認セル
# !nvidia-smi
# import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))

# %% [markdown]
# ## 1. 依存インストール
#
# Kaggle 標準環境には transformers / torch 済。peft / bitsandbytes は追加。

# %%
# TODO: pip install
# !pip install --quiet --upgrade transformers peft accelerate bitsandbytes

# %% [markdown]
# ## 2. HuggingFace 認証
#
# Gemma 4 は gated。Kaggle Secrets に `HF_TOKEN` を登録する想定。

# %%
# TODO: HF_TOKEN 認証
# from kaggle_secrets import UserSecretsClient
# from huggingface_hub import login
# secret = UserSecretsClient().get_secret("HF_TOKEN")
# login(token=secret)

# %% [markdown]
# ## 3. モデルロード
#
# - base: `google/gemma-4-E2B-it`
# - LoRA: 引数経由で切替（base only / SFT 適用）

# %%
# TODO: model load
# from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
# from peft import PeftModel
#
# BASE_MODEL_ID = "google/gemma-4-E2B-it"
# LORA_REPO_ID = "<HF_REPO_ID>"  # 例: user/vibekid-gemma-4-E2B-lora（#131 後に確定）
#
# def load_model(use_lora: bool):
#     """base model をロード、必要なら LoRA adapter を attach。
#
#     - 4-bit 量子化（NF4 + double quant、T4 16GB に収める）
#     - dtype: bfloat16（T4 は bf16 native 非対応だが Unsloth 同様 dtype 自動判定）
#     """
#     bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", ...)
#     tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID)
#     model = AutoModelForCausalLM.from_pretrained(BASE_MODEL_ID, quantization_config=bnb, ...)
#     if use_lora:
#         model = PeftModel.from_pretrained(model, LORA_REPO_ID)
#     model.eval()
#     return tokenizer, model

# %% [markdown]
# ## 4. Bench harness
#
# Web 版と整合する metric を計測:
# - **TTFT** (ms): first token latency
# - **decode tok/s**: 2 token目以降の throughput
# - **output chars**: 生成文字数
# - **total ms**: setup + first token + decode 含む全体
#
# 実装方針:
# - `TextIteratorStreamer` で token を 1 つずつ受ける
# - 最初の token で TTFT 計測
# - 全 token の wall clock を accumulate して decode tok/s 算出
# - 各 prompt × ablation で n=3 runs 取って中央値を採用（Gemma 4 は決定論的なので分散小）

# %%
# TODO: bench 関数
# from transformers import TextIteratorStreamer
# import time
# from threading import Thread
# from typing import TypedDict
#
# class BenchResult(TypedDict):
#     prompt: str
#     ttft_ms: float
#     decode_tok_s: float
#     output_chars: int
#     total_ms: float
#     output_text: str
#
# def run_bench(tokenizer, model, prompt: str, *, max_new_tokens: int = 512) -> BenchResult:
#     """単発 generate を計測。
#
#     - apply_chat_template で messages → text
#     - streamer で逐次 token 受信
#     - 最初の token timestamp を TTFT として記録
#     - decode tok/s = (n_tokens - 1) / (last_token_ts - first_token_ts)
#     """
#     ...

# %% [markdown]
# ## 5. Prompt セット
#
# Web 版 5 preset と同じ意味カバレッジ（動作系 / 視覚系 / interactive 系）を踏襲。
# preset 文言は VibeKid Web app の app.js と同期。

# %%
# TODO: prompts定義
# PRESETS = [
#     {"key": "neko",    "text": "ねこがはしる",                "category": "action"},
#     {"key": "ball",    "text": "ぼーるがはねる",              "category": "action"},
#     {"key": "button",  "text": "ボタンおすといろがかわる",    "category": "interactive"},
#     {"key": "ame",     "text": "あめがふる",                  "category": "visual"},
#     # ⭐ ほし は #143 で除外、Web 版 (PR #146) と整合
# ]
#
# # prompt length ablation 用
# PROMPTS_BY_LENGTH = {
#     "short":  "ねこ",
#     "normal": "ねこがはしる",
#     "long":   "ねこがはしる、いろんないろのねこが、ゆっくりはしったりはやくはしったりする",
# }

# %% [markdown]
# ## 6. System prompt
#
# - oneshot あり: app.js の system prompt（few-shot example 含む）を採用
# - oneshot なし: 最小 system prompt のみ
#
# 文言は SFT 訓練時の system prompt (`sft/data/SYSTEM_PROMPT.txt`) と同期させる。

# %%
# TODO: system prompt 定義（2 variant）
# SYSTEM_PROMPT_ONESHOT = """..."""  # full version with example
# SYSTEM_PROMPT_BARE    = """..."""  # bare instruction only

# %% [markdown]
# ## 7. Ablation 実験
#
# 3 軸 × 4 prompt × 3 runs = 36 runs（中央値で集約 → 12 row 表）。
# 実走時間目安: 36 runs × ~10s = 6 分。

# %%
# TODO: ablation runner
# import itertools
# import pandas as pd
#
# AB_VARIANTS = {
#     "use_lora":      [False, True],
#     "use_oneshot":   [False, True],
#     "prompt_length": ["short", "normal", "long"],
# }
#
# def run_all_ablations(n_runs: int = 3) -> pd.DataFrame:
#     """全 ablation 組合せを bench、DataFrame として返す。
#
#     base / SFT model はそれぞれ 1 回だけロードしてループ内で使い回す（OOM 回避）。
#     """
#     ...

# %% [markdown]
# ## 8. 結果集計と可視化

# %%
# TODO: aggregation
# df = run_all_ablations()
# df_summary = df.groupby(["use_lora", "use_oneshot", "prompt_length"])[
#     ["ttft_ms", "decode_tok_s", "output_chars", "total_ms"]
# ].median().round(1)
# df_summary

# %%
# TODO: 可視化（matplotlib bar chart 1-2 本）
# import matplotlib.pyplot as plt
# fig, axes = plt.subplots(1, 2, figsize=(12, 4))
# # axes[0]: base vs SFT で decode tok/s 比較
# # axes[1]: prompt length × oneshot で TTFT 比較
# plt.tight_layout()
# plt.savefig("bench_ablation.png", dpi=150)

# %% [markdown]
# ## 9. Web 版との対比表
#
# `#127` Web bench（Windows 11 + RTX 2060 Mobile + LiteRT）と Python + Kaggle T4 を
# 並べる。数値が乖離する点を率直に明記する（GPU 世代 / runtime / quantization 差）。

# %%
# TODO: 対比表
# WEB_REFERENCE = {  # #127 2026-04-25 5×5 runs より
#     "neko":   {"ttft_ms": 606,  "decode_tok_s": 30.0},
#     "ball":   {"ttft_ms": 674,  "decode_tok_s": 30.0},
#     "button": {"ttft_ms": 627,  "decode_tok_s": 30.0},
#     "ame":    {"ttft_ms": 660,  "decode_tok_s": 30.0},
# }
# # Kaggle 計測値（base oneshot off normal length 行）と並べた DataFrame を出力

# %% [markdown]
# ## 10. 再現手順 / 計測条件
#
# - **Hardware**: Kaggle T4 x1（無料枠で十分、Gemma 4 E2B 4-bit ~3GB）
# - **Software**: transformers `<>=4.45.0`、peft `>=0.13.0`、bitsandbytes `>=0.44.0`
# - **Model**: `google/gemma-4-E2B-it`（gated、HF token 必須）+ LoRA `<HF_REPO_ID>`
# - **数値の解釈**: Python + T4 + 4-bit quant の値。Web 版（LiteRT + WebGPU + Windows）
#   とは直接比較不可、`#127` を参照
# - **再現コマンド**: Kaggle で本 notebook を fork → "Run All"
#
# ## 既知の Caveats
#
# - Gemma 4 multimodal 層の `merged_4bit_forced` 経路は使わない（`project_modal_merge_dry_run.md`
#   参照、Colab で `NotImplementedError` 既知）。本 notebook は base + LoRA を 4-bit
#   ロード後に直接 generate するだけで merge は走らせない。
# - T4 は bf16 native 非対応（CC 7.5）。`torch.bfloat16` 強制は activation overflow を
#   起こすので dtype は `torch.float16` 系を採用、quantization で吸収。
# - Gemma 4 の `chat_template.jinja` は transformers 5.6.x 系で同梱されない場合がある
#   （[issue #45205](https://github.com/huggingface/transformers/issues/45205)）。
#   `tokenizer.chat_template` が None なら手動でロードして注入する。

# %% [markdown]
# ## 残 TODO（SFT 本番後に埋める）
#
# - [ ] HF Hub LoRA repo ID 確定（#131 マージ後）
# - [ ] system prompt（oneshot 版 / bare 版）の文言確定
# - [ ] ablation 実装本体
# - [ ] 可視化 figure 出力
# - [ ] Web vs Kaggle 対比 DataFrame の最終整形
# - [ ] Kaggle にアップロード → Run All で完走確認、URL を `#131` model card の
#       `<!-- KAGGLE_NOTEBOOK_URL_PLACEHOLDER -->` に反映
