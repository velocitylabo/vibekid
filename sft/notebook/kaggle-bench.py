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
# Adapted from Unsloth's official Kaggle Notebook (LGPL-3.0):
# https://kaggle.com/code/danielhanchen/gemma4-31b-unsloth
#
# Modifications: model swap (31B → E2B `unsloth/gemma-4-E2B-it`)、bench harness
# (TTFT / decode tok/s / total ms) 追加、3 軸 ablation runner、Web (#127) との対比表。
# Training セクションは削除 (本 notebook は bench 専用、訓練は `sft/notebook/sft-gemma4-e2b.py`)。
#
# ## 目的（#134）
#
# 単一 Kaggle Notebook で「Run All」実行可能な Gemma 4 E2B bench を提供する。
# 閲覧者が Kaggle 環境で 1 click 実走できることが価値。
#
# ## Web bench との関係（重要）
#
# `#124` / `#127` の bench は **ブラウザ + WebGPU + LiteRT (`@mediapipe/tasks-genai`)**
# で計測した Web 経路の数値。本 notebook は Python + Unsloth + T4 GPU 経路で
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
# - Kaggle T4 GPU x1（無料 30h/week）。x2 構成でも 1 GPU しか使わない
# - Base model: `unsloth/gemma-4-E2B-it`（4-bit quant 済、gated 承認は `google/gemma-4-E2B-it` 側）
# - LoRA adapter: `velocitylabo/vibekid-gemma-4-E2B-lora`（HF Hub public）
# - Stack: Unsloth FastModel + transformers + bitsandbytes
# - 全体実行時間目標: 15-25 分
#
# ---

# %% [markdown]
# ## 0. 環境確認（Kaggle T4 想定）
#
# Kaggle で T4 を有効化（Settings > Accelerator > GPU T4 x2）。
# 1 GPU で十分（Gemma 4 E2B は 4-bit 量子化で ~3GB）。

# %%
# !nvidia-smi
import torch

assert torch.cuda.is_available(), "GPU が見えていません。Kaggle Settings > Accelerator で T4 を有効化してください。"
print(f"[gpu] {torch.cuda.get_device_name(0)} / cuda={torch.version.cuda} / torch={torch.__version__}")

# %% [markdown]
# ## 1. 依存インストール
#
# Unsloth + sidecar (`unsloth_zoo`) を入れる。Kaggle 既存 transformers / torch とは
# 衝突しないように `--no-deps` で trl/peft/accelerate/bitsandbytes を上書き。
# **インストール後 1 度だけセッション再起動が必要**な場合がある（Unsloth 既知の挙動）。

# %%
# !pip install --quiet --upgrade --no-cache-dir unsloth unsloth_zoo
# !pip install --quiet --no-deps trl peft accelerate bitsandbytes

# %% [markdown]
# ## 2. HuggingFace 認証
#
# Base model (`google/gemma-4-E2B-it`) は **gated**、事前に HF アカウントで承認必要。
# `unsloth/gemma-4-E2B-it` は gated を継承するので同じ token が必要。
# Kaggle Secrets に `HF_TOKEN` を登録する想定（Add-ons > Secrets）。

# %%
try:
    from kaggle_secrets import UserSecretsClient
    from huggingface_hub import login

    hf_token = UserSecretsClient().get_secret("HF_TOKEN")
    login(token=hf_token)
    print("[auth] HF login ok (via Kaggle Secrets)")
except ImportError:
    print("[auth] kaggle_secrets 不在 — ローカル実行とみなす。`huggingface-cli login` 済みなら OK")

# %% [markdown]
# ## 3. モデルロード
#
# Unsloth `FastModel.from_pretrained` で 1 関数呼び出し:
# - `model_name="unsloth/gemma-4-E2B-it"` → base only
# - `model_name="velocitylabo/vibekid-gemma-4-E2B-lora"` → base + SFT adapter
#   (adapter repo の `adapter_config.json` の `base_model_name_or_path` を Unsloth が読んで
#    base を内部で別途ロードしてくれる)
#
# `MAX_SEQ_LENGTH=2048` は inference 時の余裕。SFT 訓練は 512 だが推論は伸ばせる。
# 4-bit quant (nf4 + double quant) で T4 16GB に収まる (Gemma 4 E2B 重み ~3GB)。

# %%
from unsloth import FastModel

BASE_MODEL_ID = "unsloth/gemma-4-E2B-it"
LORA_REPO_ID = "velocitylabo/vibekid-gemma-4-E2B-lora"
MAX_SEQ_LENGTH = 2048


def load_model(use_lora: bool):
    """base model または base + LoRA adapter をロードして (model, tokenizer) を返す。

    - base 単独で n=72 runs 計測 → unload → SFT 版で n=72 runs、の 2 phase 想定
    - 4-bit 量子化、T4 native 非対応の bf16 は dtype=None で auto 判定 (Unsloth が float16 fallback)
    - `tokenizer.chat_template` が None の場合あり (transformers issue #45205)、Unsloth が吸収
    """
    model_name = LORA_REPO_ID if use_lora else BASE_MODEL_ID
    model, tokenizer = FastModel.from_pretrained(
        model_name=model_name,
        max_seq_length=MAX_SEQ_LENGTH,
        load_in_4bit=True,
        load_in_8bit=False,
        full_finetuning=False,
        dtype=None,
    )
    model.eval()
    print(f"[model] loaded {model_name} (use_lora={use_lora})")
    return model, tokenizer

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
# - 各 prompt × ablation で n=3 runs 取って中央値を採用（Gemma 4 は決定論的なので分散小、
#   memory `feedback_gemma_determinism.md`）
#
# 公式推論 hyperparameter (Unsloth Notebook より): `temperature=1.0, top_p=0.95, top_k=64`

# %%
import time
from threading import Thread
from typing import TypedDict

from transformers import TextIteratorStreamer


class BenchResult(TypedDict):
    prompt: str
    ttft_ms: float
    decode_tok_s: float
    output_chars: int
    output_tokens: int
    total_ms: float
    output_text: str


def run_bench(
    model,
    tokenizer,
    user_prompt: str,
    system_prompt: str,
    *,
    max_new_tokens: int = 512,
) -> BenchResult:
    """単発 generate を計測して BenchResult を返す。

    - `apply_chat_template` で messages → input_ids
    - `TextIteratorStreamer` で 1 トークンずつ受信、最初のトークン到着で TTFT 計測
    - decode tok/s = (n_tokens - 1) / (last_token_ts - first_token_ts)
    - 生成は別スレッド、メインスレッドは streamer から逐次 pull
    """
    # Gemma 4 multimodal processor (Unsloth) の制約 (`sft-gemma4-e2b.py` 4/19 検証済):
    # - content は `list[{type,text}]` 形式必須、文字列直渡しは
    #   `string indices must be integers` で落ちる
    # - tokenizer(...) 経由は patched_call が text/images の positional 解釈で混乱、
    #   apply_chat_template(tokenize=True, return_tensors="pt") で tensor 直取りが安定
    # - system role 分離ではなく user content に merge (training と同じパターン)
    merged_text = f"{system_prompt}\n\n{user_prompt}"
    messages = [{
        "role": "user",
        "content": [{"type": "text", "text": merged_text}],
    }]
    input_ids = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
    ).to(model.device)

    # Processor の場合は `.tokenizer` 属性が inner tokenizer。streamer は decode できれば
    # よいので processor 自体でも tokenizer でも動く想定だが、安全側で inner tokenizer を取る。
    streamer_tokenizer = getattr(tokenizer, "tokenizer", tokenizer)
    streamer = TextIteratorStreamer(
        streamer_tokenizer,
        skip_prompt=True,
        skip_special_tokens=True,
    )
    gen_kwargs = dict(
        input_ids=input_ids,
        streamer=streamer,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=1.0,
        top_p=0.95,
        top_k=64,
    )

    t0 = time.perf_counter()
    thread = Thread(target=model.generate, kwargs=gen_kwargs)
    thread.start()

    chunks: list[str] = []
    first_token_ts: float | None = None
    last_token_ts: float = t0
    n_chunks = 0
    for chunk in streamer:
        ts = time.perf_counter()
        if first_token_ts is None:
            first_token_ts = ts
        last_token_ts = ts
        chunks.append(chunk)
        n_chunks += 1

    thread.join()
    t1 = time.perf_counter()

    output_text = "".join(chunks)
    # encode も processor 経由は patched_call の罠があるので inner tokenizer を使う
    output_tokens = len(streamer_tokenizer.encode(output_text, add_special_tokens=False))

    if first_token_ts is None or output_tokens <= 1:
        ttft_ms = (t1 - t0) * 1000.0
        decode_tok_s = 0.0
    else:
        ttft_ms = (first_token_ts - t0) * 1000.0
        decode_seconds = max(last_token_ts - first_token_ts, 1e-6)
        decode_tok_s = (output_tokens - 1) / decode_seconds

    return BenchResult(
        prompt=user_prompt,
        ttft_ms=round(ttft_ms, 1),
        decode_tok_s=round(decode_tok_s, 2),
        output_chars=len(output_text),
        output_tokens=output_tokens,
        total_ms=round((t1 - t0) * 1000.0, 1),
        output_text=output_text,
    )

# %% [markdown]
# ## 5. Prompt セット
#
# Web 版 4 preset と同じ意味カバレッジ（action / visual / interactive）を踏襲。
# 文言は VibeKid Web app `app.js` の `presets` (#143 で ⭐ 除外、計 4 件) と同期。
#
# prompt length ablation は `neko` をベースに short / normal / long の 3 variant を用意。

# %%
PRESETS = [
    {"key": "neko",   "text": "ねこがはしる",                "category": "action"},
    {"key": "ball",   "text": "ぼーるがはねる",              "category": "action"},
    {"key": "button", "text": "ボタンおすといろがかわる",    "category": "interactive"},
    {"key": "ame",    "text": "あめがふる",                  "category": "visual"},
]

PROMPTS_BY_LENGTH = {
    "short":  "ねこ",
    "normal": "ねこがはしる",
    "long":   "ねこがはしる、いろんないろのねこが、ゆっくりはしったりはやくはしったりする",
}

# %% [markdown]
# ## 6. System prompt 2 variants
#
# - **oneshot**: `app.js` `_systemPrompt()` 全文 (V1 prompt rules 適用済 / db873a4)。
#   one-shot example (gravity ball) + rules。memory `feedback_gemma_oneshot_turns.md` で
#   two-shot は 2B が collapse することを確認、one-shot に固定。
# - **bare**: 最小指示のみ。example も rules もない baseline、context engineering の効果を切り分ける。
#
# **同期**: app.js V1 prompt と一致。app.js 側で更新があったら本セルも手動同期。

# %%
SYSTEM_PROMPT_ONESHOT = """子供が「〜作って」と言ったら、p5.js のスケッチコードを1つだけ書いてください。

例（「ボールが跳ねる」の場合）:
```js
let ballX = 200;
let ballY = 100;
let ballVY = 0;

function setup() {
  createCanvas(400, 400);
}

function draw() {
  background(240);
  ballVY = ballVY + 0.5;
  ballY = ballY + ballVY;
  if (ballY > 370) {
    ballY = 370;
    ballVY = -10;
  }
  fill(255, 100, 100);
  noStroke();
  ellipse(ballX, ballY, 50, 50);
}
```

ルール:
- 変数は function の外で let で宣言する
- 関数は必ず setup() と draw() を書く
- createCanvas(400, 400) は setup() の中で必ず呼ぶ
- background() を draw の最初に呼ぶ
- 変数名は ballX, bgColor, myScore のような分かりやすい名前を使う
- 変数名は英数字とアンダースコアのみ（日本語の変数名は使わない）
- p5.js の組み込み関数名（color, fill, background, width, height）を変数名にしない
- color() / createVector() などの p5.js 関数は setup() より前で呼ばない
  - NG: let myColor = color(255, 0, 0);  ← top-level で color() はエラー
  - OK: let myColor = '#ff0000';         ← 文字列で持つ
- 色のリストは hex 文字列で持つ: let cols = ['#ff4444', '#ffaa00', '#ffee00']
- クリック処理は mousePressed() 関数で書く
- 当たり判定は dist(mouseX, mouseY, x, y) < 半径 で書く
- 複数オブジェクトは array で持つ: let items = [{x: 100, y: 100}, ...]
- 短く、動くコードだけ書く（20〜40行程度）
- ```js で囲んで出力する

同じスタイルで、指示されたものを作ってください:"""

SYSTEM_PROMPT_BARE = """p5.js のスケッチコードを書いてください。"""

SYSTEM_PROMPTS = {
    "oneshot": SYSTEM_PROMPT_ONESHOT,
    "bare":    SYSTEM_PROMPT_BARE,
}

# %% [markdown]
# ## 7. Ablation 実験
#
# 3 軸 ablation:
# - `use_lora` (False/True): base vs SFT
# - `use_oneshot` (False/True): system prompt の context engineering 効果
# - `prompt_kind` (length/preset): prompt 形状の効果
#   - `length`: `neko` をベースに short / normal / long の 3 variant
#   - `preset`: 4 preset (neko/ball/button/ame) の breadth、長さは normal 固定
#
# **Phase 構造**: T4 16GB で base + SFT 同時ロードは活性メモリ的に苦しいので、
# 「base load → 全 prompt × oneshot × runs → unload → SFT load → 同 → unload」の 2 phase。
#
# 規模: 2 (lora) × 2 (oneshot) × (3 length + 4 preset = 7) × n_runs
# = `n_runs=3` で 84 runs ~ 14 分（10s/run 想定）。Gemma 4 は決定論的なので n=3 で充分
# （memory `feedback_gemma_determinism.md`）。

# %%
import gc

import pandas as pd

ABLATION_PROMPT_SETS: dict[str, dict[str, str]] = {
    "length": PROMPTS_BY_LENGTH,
    "preset": {p["key"]: p["text"] for p in PRESETS},
}


def run_ablation_phase(model, tokenizer, *, use_lora: bool, n_runs: int) -> pd.DataFrame:
    """単一モデル (base or SFT) に対し全 (oneshot × prompt × run) 組合せを bench。"""
    rows: list[dict] = []
    for prompt_kind, prompt_dict in ABLATION_PROMPT_SETS.items():
        for prompt_key, prompt_text in prompt_dict.items():
            for use_oneshot in [False, True]:
                sys_prompt = SYSTEM_PROMPTS["oneshot" if use_oneshot else "bare"]
                for run_idx in range(n_runs):
                    result = run_bench(model, tokenizer, prompt_text, sys_prompt)
                    rows.append({
                        "use_lora":      use_lora,
                        "use_oneshot":   use_oneshot,
                        "prompt_kind":   prompt_kind,
                        "prompt_key":    prompt_key,
                        "run_idx":       run_idx,
                        "ttft_ms":       result["ttft_ms"],
                        "decode_tok_s":  result["decode_tok_s"],
                        "output_chars":  result["output_chars"],
                        "output_tokens": result["output_tokens"],
                        "total_ms":      result["total_ms"],
                        "output_text":   result["output_text"],
                    })
                    print(
                        f"[run] lora={use_lora} oneshot={use_oneshot} "
                        f"{prompt_kind}/{prompt_key} run={run_idx} "
                        f"ttft={result['ttft_ms']:.0f}ms decode={result['decode_tok_s']:.1f}tok/s "
                        f"chars={result['output_chars']}"
                    )
    return pd.DataFrame(rows)


def run_all_ablations(n_runs: int = 3) -> pd.DataFrame:
    """base / SFT それぞれを順番にロード → 全 ablation phase 実行 → unload。"""
    dfs: list[pd.DataFrame] = []
    for use_lora in [False, True]:
        model, tokenizer = load_model(use_lora=use_lora)
        df_phase = run_ablation_phase(model, tokenizer, use_lora=use_lora, n_runs=n_runs)
        dfs.append(df_phase)
        del model, tokenizer
        gc.collect()
        torch.cuda.empty_cache()
        print(f"[phase] use_lora={use_lora} done, {len(df_phase)} rows")
    return pd.concat(dfs, ignore_index=True)


# %%
df = run_all_ablations(n_runs=3)
print(f"[total] {len(df)} runs")
df.head()

# %% [markdown]
# ## 8. 結果集計と可視化

# %%
TIMING_COLS = ["ttft_ms", "decode_tok_s", "output_chars", "output_tokens", "total_ms"]

df_summary = (
    df.groupby(["use_lora", "use_oneshot", "prompt_kind", "prompt_key"])[TIMING_COLS]
    .median()
    .round(1)
    .reset_index()
)
df_summary

# %%
# 観点別の pivot 表 3 本: (a) base vs SFT × preset、(b) length × oneshot、(c) preset × oneshot
df_preset = df_summary.query("prompt_kind == 'preset'").pivot_table(
    index="prompt_key",
    columns=["use_lora", "use_oneshot"],
    values=["ttft_ms", "decode_tok_s"],
)
print("[pivot] preset × (use_lora, use_oneshot)")
df_preset

# %%
df_length = df_summary.query("prompt_kind == 'length'").pivot_table(
    index="prompt_key",
    columns=["use_lora", "use_oneshot"],
    values=["ttft_ms", "decode_tok_s"],
)
print("[pivot] length × (use_lora, use_oneshot)")
df_length

# %%
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

# (a) base vs SFT decode tok/s by preset (oneshot=True 固定)
preset_oneshot = df_summary.query("prompt_kind == 'preset' and use_oneshot == True").pivot(
    index="prompt_key", columns="use_lora", values="decode_tok_s"
)
preset_oneshot.plot(kind="bar", ax=axes[0], rot=0, color=["#888888", "#4477aa"])
axes[0].set_title("decode tok/s by preset (oneshot=True)")
axes[0].set_ylabel("decode tok/s")
axes[0].legend(title="use_lora", labels=["base", "SFT"])

# (b) prompt length × oneshot TTFT (use_lora=True 固定)
length_lora = df_summary.query("prompt_kind == 'length' and use_lora == True").pivot(
    index="prompt_key", columns="use_oneshot", values="ttft_ms"
)
length_lora = length_lora.reindex(["short", "normal", "long"])
length_lora.plot(kind="bar", ax=axes[1], rot=0, color=["#aa6644", "#44aa66"])
axes[1].set_title("TTFT by prompt length (SFT)")
axes[1].set_ylabel("TTFT (ms)")
axes[1].legend(title="use_oneshot", labels=["bare", "oneshot"])

plt.tight_layout()
plt.savefig("bench_ablation.png", dpi=150)
plt.show()

# %% [markdown]
# ## 9. Web 版との対比表
#
# `#127` Web bench（Windows 11 + RTX 2060 Mobile + LiteRT）と Python + Kaggle T4 を
# 並べる。**数値は直接比較不可** (GPU 世代 / runtime / quantization 差) だが、
# decode tok/s の桁感と TTFT の桁感を確認する用途。

# %%
# memory `project_windows_app_perf_gap.md` 4/22 LiteRT rollback 後 + memory `extract_nested_fence` 4/25
# Windows 5×5 runs より。Web 版の数値は `tools/cdp-sampling.mjs` 出力の中央値ベース。
WEB_REFERENCE = {
    "neko":   {"ttft_ms": 606, "decode_tok_s": 30.0},
    "ball":   {"ttft_ms": 674, "decode_tok_s": 30.0},
    "button": {"ttft_ms": 627, "decode_tok_s": 30.0},
    "ame":    {"ttft_ms": 660, "decode_tok_s": 30.0},
}

# Kaggle 計測値: oneshot=True / use_lora=True / preset (=p5.js 本番想定構成) を採用
kaggle_main = (
    df_summary.query(
        "prompt_kind == 'preset' and use_oneshot == True and use_lora == True"
    )[["prompt_key", "ttft_ms", "decode_tok_s"]]
    .set_index("prompt_key")
)

df_compare = pd.DataFrame({
    "web_ttft_ms":          {k: v["ttft_ms"] for k, v in WEB_REFERENCE.items()},
    "kaggle_ttft_ms":       kaggle_main["ttft_ms"],
    "web_decode_tok_s":     {k: v["decode_tok_s"] for k, v in WEB_REFERENCE.items()},
    "kaggle_decode_tok_s":  kaggle_main["decode_tok_s"],
})
df_compare = df_compare.reindex(["neko", "ball", "button", "ame"])
df_compare

# %% [markdown]
# ## 10. 再現手順 / 計測条件
#
# - **Hardware**: Kaggle T4 x1（無料枠で十分、Gemma 4 E2B 4-bit ~3GB）
# - **Software**: Unsloth + transformers + peft + bitsandbytes（Section 1 でインストール）
# - **Base model**: `unsloth/gemma-4-E2B-it`（4-bit 量子化済、gated 承認は `google/gemma-4-E2B-it` 側）
# - **LoRA adapter**: `velocitylabo/vibekid-gemma-4-E2B-lora`（HF Hub public、QLoRA r=16 α=32、3 epoch、eval_loss 6.34→3.05）
# - **数値の解釈**: Python + T4 + 4-bit quant の値。Web 版（LiteRT + WebGPU + Windows）
#   とは直接比較不可、`#127` を参照
# - **再現コマンド**: Kaggle で本 notebook を fork → "Run All"
#
# ## 既知の Caveats
#
# - Gemma 4 multimodal 層の `merged_4bit_forced` 経路は使わない（memory `project_modal_merge_dry_run.md`
#   参照、Colab で `NotImplementedError` 既知）。本 notebook は base + LoRA を 4-bit
#   ロード後に直接 generate するだけで merge は走らせない。
# - T4 は bf16 native 非対応（CC 7.5）。`torch.bfloat16` 強制は activation overflow を
#   起こすので `dtype=None` で Unsloth に float16 fallback させる。
# - Gemma 4 の `chat_template.jinja` は transformers 5.6.x 系で同梱されない場合がある
#   （[issue #45205](https://github.com/huggingface/transformers/issues/45205)）。
#   Unsloth が吸収するので本 notebook では明示対応不要。
# - 本 notebook は Web bench (#127) と数値直接比較不可、ablation の **相対** 効果のみを論じる。
