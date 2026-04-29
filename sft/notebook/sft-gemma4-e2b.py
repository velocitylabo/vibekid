# ---
# jupyter:
#   jupytext:
#     formats: py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
# ---

# %% [markdown]
# # Gemma 4 E2B SFT — VibeKid
#
# **Status**:
# - 4/19 骨組み + Colab checkpoint resume 対応
# - 4/25 Colab Pro で本番 run 実行 (`velocitylabo/vibekid-gemma-4-E2B-lora` 公開)
# - **4/29 Phase 4 (#179) — EOS 修復: train_on_responses_only + tokenizer sanity check 追加**
#
# - Model: `unsloth/gemma-4-E2B-it`
# - LoRA r=16 α=32, text decoder のみ（multi-modal 層汚染防止）
# - bf16（T4 Colab Pro、Unsloth が float16 infinite activation を回避）
# - seq_len 512, batch 1, grad_accum 8 → 1 epoch ≈ 2-3h
# - Data: `sft/data/train.jsonl` (657件) / `sft/data/eval.jsonl` (100件, category 均等)
# - 環境: Colab (元) / Kaggle (Phase 4 再訓練) / ローカル GPU の三本立て
#
# **Phase 4 (#179) 介入の根拠** (research spike, 2026-04-29):
# - `train_on_responses_only` 不在で gradient 希釈 (Unsloth 公式推奨、VERIFIED)
# - `<end_of_turn>` token re-encoding 問題 (PEFT #1003 既知地雷)
# - `pad_token == eos_token` で EOS 学習信号減衰 (TRL doc 警告)
#
# **Gemma 4 固有地雷（既知）**:
# - `chat_template.jinja` が tokenizer_config に非同梱（transformers #45205）→ Unsloth が吸収
# - `mm_token_type_ids` silent 崩壊 → Unsloth collator 必須
# - float16 は T4 で破綻 → bf16 必須
#
# **このノートブックで触るもの**:
# 1. Unsloth で FastModel ロード
# 2. train/eval.jsonl をアップロード、`standardize_sharegpt` で整形
# 3. `apply_chat_template` で messages → text
# 4. LoRA PEFT
# 5. SFTTrainer で 1 epoch
# 6. eval loss + 数サンプル生成テスト
# 7. LoRA adapter 保存、merged weight エクスポート
#
# **Colab へのアップロード手順**:
# 1. ローカルで jupytext インストール済なら `jupytext --to ipynb sft-gemma4-e2b.py` で `.ipynb` 生成
# 2. または Colab 上で「ファイル > アップロード」に `.py` を直接投げても変換される
# 3. データは `sft/data/train.jsonl` / `eval.jsonl` を Drive `/MyDrive/vibekid-sft-data/` に配置
# 4. `SYSTEM_PROMPT.txt` も同じ Drive ディレクトリに置く
#
# **checkpoint resume 設計**:
# - Drive 配下 `/MyDrive/vibekid-sft-ckpt/` に save_steps ごと書き出し → 12h セッション切れても resume 可能
# - 再接続後、同じ notebook を実行すれば `get_last_checkpoint` で自動 resume
# - `DRY_RUN = True` で 10 件 / 20 step / save_steps=5 の kill→resume リハ可能

# %% [markdown]
# ## 0. 環境確認（Colab Pro T4 想定）

# %%
# !nvidia-smi
# !free -h
# !df -h /content

# %% [markdown]
# ## 1. Unsloth + 依存インストール
#
# Colab 用。`unsloth_zoo` は Unsloth の sidecar（2026 以降必須、忘れると
# `PackageNotFoundError: unsloth_zoo` で落ちる）。install 後はランタイム再起動が安全。
# ローカル Linux GPU で試す場合は本セルをスキップして先に手動で venv を作ること。

# %%
!pip install --quiet --upgrade --no-cache-dir unsloth unsloth_zoo
!pip install --quiet --no-deps trl peft accelerate bitsandbytes datasets

# %% [markdown]
# ## 1b. 環境検出 + path 解決
#
# 三環境を自動判定:
# - **Colab**: `/content/drive/MyDrive/vibekid-sft-{ckpt,data,merged}/` (12h セッション再開対応)
# - **Kaggle**: `/kaggle/input/vibekid-sft-data/` (read-only Dataset) +
#   `/kaggle/working/vibekid-sft-{ckpt,merged}/` (write 可、9h 制限後セッションでも残る)
# - **ローカル**: `./sft/data/` + `./vibekid-sft-{ckpt,merged}/`
#
# Phase 4 (#179) 再訓練は Kaggle T4 で実施想定。Kaggle Dataset 名は `vibekid-sft-data` で
# `train.jsonl` / `eval.jsonl` / `SYSTEM_PROMPT.txt` を upload。

# %%
import os

IS_COLAB = False
IS_KAGGLE = os.path.exists("/kaggle/input") or os.environ.get("KAGGLE_KERNEL_RUN_TYPE")

try:
    from google.colab import drive  # type: ignore
    drive.mount("/content/drive")
    IS_COLAB = True
except ImportError:
    pass

if IS_COLAB:
    CKPT_DIR = "/content/drive/MyDrive/vibekid-sft-ckpt"
    DATA_DIR = "/content/drive/MyDrive/vibekid-sft-data"
    MERGED_DIR = "/content/drive/MyDrive/vibekid-sft-merged"
elif IS_KAGGLE:
    # Dataset 名は user が upload 時に設定 ("vibekid-sft-data" 想定)
    DATA_DIR = "/kaggle/input/vibekid-sft-data"
    CKPT_DIR = "/kaggle/working/vibekid-sft-ckpt"
    MERGED_DIR = "/kaggle/working/vibekid-sft-merged"
    print("[env] Kaggle detected (input read-only / working write 可)")
else:
    print("[env] ローカル GPU 実行とみなす")
    CKPT_DIR = "./vibekid-sft-ckpt"
    DATA_DIR = "./sft/data"
    MERGED_DIR = "./vibekid-sft-merged"

os.makedirs(CKPT_DIR, exist_ok=True)
os.makedirs(MERGED_DIR, exist_ok=True)
# DATA_DIR は Kaggle で read-only なので mkdir 試行しない

print(f"[paths] CKPT_DIR={CKPT_DIR}")
print(f"[paths] DATA_DIR={DATA_DIR}")
print(f"[paths] MERGED_DIR={MERGED_DIR}")

# %% [markdown]
# ## 1c. Dry-run フラグ（checkpoint resume リハ用）
#
# - `DRY_RUN = True` にすると 10 件 / 20 step / save_steps=5 の最小構成で走る
# - 4/19-24 の間に「5 step 回したあと Colab ランタイムを手動切断 → 再接続 → 同じ notebook 再実行 → resume できるか」を検証
# - 本番 4/25 は `DRY_RUN = False` に戻すこと

# %%
DRY_RUN = False

print(f"[dry-run] {'ENABLED' if DRY_RUN else 'disabled'}")

# %% [markdown]
# ## 2. モデル + tokenizer ロード

# %%
from unsloth import FastModel
import torch

MAX_SEQ_LENGTH = 512
MODEL_NAME = "unsloth/gemma-4-E2B-it"

model, tokenizer = FastModel.from_pretrained(
    model_name=MODEL_NAME,
    max_seq_length=MAX_SEQ_LENGTH,
    load_in_4bit=True,       # QLoRA（T4 16GB 上限）
    load_in_8bit=False,
    full_finetuning=False,
    dtype=None,              # bf16 auto
)

print(f"[model] {MODEL_NAME} loaded, max_seq_length={MAX_SEQ_LENGTH}")
print(f"[tokenizer] chat_template: {'gemma' in (tokenizer.chat_template or '').lower()}")

# %% [markdown]
# ## 2.5. Tokenizer 健全性チェック (Phase 4 #179、pre-train)
#
# Phase 4 介入の前提条件確認。post-train でも同じ check を回して regression を検出する。
#
# - **#1**: 終端 token (Gemma 4 では `<turn|>`、id=106) が special vocab に登録され、
#   chat_template で実際に出力される (PEFT issue #1003 — fine-tune 後に複数 sub-token に
#   分裂すると `model.generate()` が EOS 検出不能で無限生成する既知地雷)
# - **#3**: `pad_token != eos_token` （TRL doc 警告 — padding mask で EOS が ignore label に
#   変換され学習信号減衰）
#
# **Gemma 4 token 命名注意** (memory `feedback_gemma4_token_naming.md`):
# Gemma 4 は Gemma 1/2/3 系と命名が違い、`<end_of_turn>` ではなく `<turn|>` が終端 token。
# `tokenizer.encode("<end_of_turn>", add_special_tokens=False)` は **literal split** され
# 7 tokens に分裂、これを「PEFT #1003 該当」と誤判定しないため `convert_tokens_to_ids` 経由で
# special vocab を直接参照する。

# %%
def _check_eot_single_token(tok, label="pre-train"):
    """終端 token が special vocab 内で単一 token として動作することを確認。

    Gemma 4 では `<turn|>` (id=106)、tokenizer の `eot_token` 属性経由で動的取得。
    `convert_tokens_to_ids` で special vocab 直接参照、literal parse は使わない。
    """
    inner = getattr(tok, "tokenizer", tok)  # Gemma 4 multimodal processor (Gemma4Processor) 対応
    # Gemma 4 では eot_token = "<turn|>"、Gemma 1/2/3 では "<end_of_turn>"
    # tokenizer 属性経由で動的取得し、特定 model 命名へのハードコードを避ける
    eot_token = getattr(inner, "eot_token", None) or "<turn|>"
    eot_id = inner.convert_tokens_to_ids(eot_token)
    print(f"[sanity-eot/{label}] {eot_token!r} -> token id {eot_id}")
    unk_id = inner.unk_token_id
    assert eot_id is not None and eot_id >= 0 and eot_id != unk_id, (
        f"終端 token {eot_token!r} が special vocab に未登録 (id={eot_id}, unk={unk_id})、"
        f"PEFT #1003 該当の可能性。tokenizer / chat_template 確認要"
    )
    # decode round-trip: id -> string で同じ token が戻るか確認
    decoded = inner.decode([eot_id], skip_special_tokens=False)
    print(f"[sanity-eot/{label}] decode(id={eot_id}) = {decoded!r}")
    # chat_template が実際にこの id を出力に含めるか (format mismatch 検出)
    sample_ids = inner.apply_chat_template(
        [{"role": "user", "content": "test"}],
        tokenize=True, add_generation_prompt=False,
    )
    if hasattr(sample_ids, "tolist"):
        sample_ids = sample_ids.tolist()
    if sample_ids and isinstance(sample_ids[0], list):
        sample_ids = sample_ids[0]
    used = eot_id in sample_ids
    print(f"[sanity-eot/{label}] chat_template 内で使用されてる: {used}")
    assert used, (
        f"chat_template が token id {eot_id} を出力に含めない、format mismatch の疑い"
    )
    return eot_id

EOT_TOKEN_ID = _check_eot_single_token(tokenizer, "pre-train")

# pad_token vs eos_token (TRL #4147、SFT Trainer doc warning)
inner_tok = getattr(tokenizer, "tokenizer", tokenizer)
pad_id = inner_tok.pad_token_id
eos_id = inner_tok.eos_token_id
print(f"[sanity-pad] pad_token_id={pad_id} (`{inner_tok.pad_token}`) "
      f"/ eos_token_id={eos_id} (`{inner_tok.eos_token}`)")
if pad_id == eos_id:
    print("[sanity-pad] WARN: pad_token == eos_token、TRL 警告対象。")
    print("            padding mask で EOS が ignore label 化、学習信号減衰の可能性。")
    # 自動修正: <unk> や <pad> 等の専用 token がある場合のみ振り替え
    if inner_tok.unk_token_id is not None and inner_tok.unk_token_id != eos_id:
        inner_tok.pad_token = inner_tok.unk_token
        print(f"[sanity-pad] FIX: pad_token を unk_token (id={inner_tok.pad_token_id}) に変更")
    else:
        print("[sanity-pad] FIX skipped: 適切な代替 token 見つからず、手動対応必要")
else:
    print("[sanity-pad] OK: pad と eos は別 token")

# %% [markdown]
# ## 3. LoRA 設定
#
# - text decoder のみ（vision/audio branch は触らない）
# - r=16, alpha=32 — SFT 用標準

# %%
model = FastModel.get_peft_model(
    model,
    finetune_language_layers=True,
    finetune_attention_modules=True,
    finetune_mlp_modules=True,
    finetune_vision_layers=False,   # 地雷対策 4
    r=16,
    lora_alpha=32,
    lora_dropout=0,
    bias="none",
    random_state=42,
    use_gradient_checkpointing="unsloth",  # VRAM 30% 削減
    use_rslora=False,
    loftq_config=None,
)

model.print_trainable_parameters()

# %% [markdown]
# ## 4. データロード
#
# `prepare-train.mjs` 生成の messages 形式 JSONL を datasets 経由で読む。
# Colab では `/content/sft/data/` にアップロードする想定。

# %%
from datasets import load_dataset

TRAIN_PATH = os.path.join(DATA_DIR, "train.jsonl")
EVAL_PATH  = os.path.join(DATA_DIR, "eval.jsonl")

dataset = load_dataset(
    "json",
    data_files={"train": TRAIN_PATH, "eval": EVAL_PATH},
)
print(dataset)
print("sample[0]:", dataset["train"][0])

# %% [markdown]
# ## 5. apply_chat_template で messages → text
#
# Gemma 4 の chat template は Unsloth の tokenizer が持っている。
# messages 配列をそのまま流し込んで `<start_of_turn>user\n...<end_of_turn>\n<start_of_turn>model\n...` に変換。

# %%
def format_messages(example):
    text = tokenizer.apply_chat_template(
        example["messages"],
        tokenize=False,
        add_generation_prompt=False,
    )
    return {"text": text}

dataset = dataset.map(format_messages, remove_columns=["messages", "meta"])
print("formatted sample[0]:")
print(dataset["train"][0]["text"][:400], "...")

# %% [markdown]
# ## 6. SFTTrainer
#
# `dataset_text_field="text"` で直接 text 列を与える。
# collator は Unsloth 内蔵の完成済み版が自動選択される（Gemma 4 `mm_token_type_ids` 対策）。

# %%
from trl import SFTTrainer, SFTConfig

# GPU 対応 dtype 判定: T4 (Turing) は bf16 非対応、L4/A100 (Ampere+) のみ bf16 可
# Colab Pro はランタイム抽選で T4 / L4 どちらも来る → 自動切替
USE_BF16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
USE_FP16 = torch.cuda.is_available() and not USE_BF16
print(f"[dtype] bf16={USE_BF16} fp16={USE_FP16} (device={torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu'})")

# DRY_RUN で最小構成に差し替え（kill→resume リハ用）
if DRY_RUN:
    train_ds = dataset["train"].select(range(min(10, len(dataset["train"]))))
    eval_ds  = dataset["eval"].select(range(min(3, len(dataset["eval"]))))
    sft_kwargs = dict(
        num_train_epochs=2,
        max_steps=20,
        logging_steps=1,
        eval_steps=5,
        save_steps=5,
        save_total_limit=4,
    )
    print(f"[dry-run] train={len(train_ds)}, eval={len(eval_ds)}, save_steps=5")
else:
    train_ds = dataset["train"]
    eval_ds  = dataset["eval"]
    sft_kwargs = dict(
        num_train_epochs=1,
        logging_steps=10,
        eval_steps=50,
        save_steps=100,
        save_total_limit=3,
    )

training_args = SFTConfig(
    output_dir=CKPT_DIR,                 # Drive 永続化（12h 切れ対策）
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,       # 実効 batch = 8
    learning_rate=2e-4,                  # LoRA SFT 標準
    warmup_ratio=0.03,
    lr_scheduler_type="cosine",
    bf16=USE_BF16,                       # L4/A100 で True
    fp16=USE_FP16,                       # T4 で True（Unsloth が overflow 対策を入れる）
    optim="adamw_8bit",
    weight_decay=0.01,
    eval_strategy="steps",
    save_strategy="steps",
    max_seq_length=MAX_SEQ_LENGTH,
    dataset_text_field="text",
    packing=False,                       # packing は Gemma 4 で検証不足
    report_to="none",
    seed=42,
    **sft_kwargs,
)

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=train_ds,
    eval_dataset=eval_ds,
    args=training_args,
)

# %% [markdown]
# ## 6.5. train_on_responses_only 適用 (Phase 4 #179、Hypothesis A)
#
# user turn (long system prompt) の loss を mask して、assistant turn のみで loss を取る。
# Unsloth 公式 Gemma 4 fine-tuning ガイド推奨。chat-style SFT で事実上必須。
#
# 効果: assistant 末尾終端 token (Gemma 4: `<turn|>` id=106) 学習に勾配が集中、
# EOS 出力確率が上がる。Phase 2 観測の SFT+oneshot cap 43% / SFT+bare cap 100% の
# 改善を狙う (Phase 4 4/29 Colab Pro 訓練で N=3 sample で `<turn|>` emit 67% 確認)。
#
# **Gemma 4 token 命名注意** (memory `feedback_gemma4_token_naming.md`):
# - start_of_turn: `<|turn>` (id=105) ← **Gemma 1/2/3 系の `<start_of_turn>` ではない**
# - end_of_turn  : `<turn|>` (id=106) ← 同上 (`<end_of_turn>` ではない)
#
# instruction_part / response_part の sot literal は **`tokenizer.sot_token` 属性経由で
# 動的取得** (Section 2.5 の `_check_eot_single_token` と一貫させる)。Gemma 5 等で
# 命名が変わっても tokenizer 側で `sot_token` 属性が定義されていれば追従可能。
# fallback default は Gemma 4 用 `<|turn>` (本 notebook は Gemma 4 専用想定)。

# %%
from unsloth.chat_templates import train_on_responses_only

# Gemma 4: <|turn> (id=105) を tokenizer 属性経由で動的取得
_inner_tok = getattr(tokenizer, "tokenizer", tokenizer)
sot_token = getattr(_inner_tok, "sot_token", None) or "<|turn>"
print(f"[setup] sot_token from tokenizer: {sot_token!r}")

trainer = train_on_responses_only(
    trainer,
    instruction_part=f"{sot_token}user\n",
    response_part=f"{sot_token}model\n",
)
print("[setup] train_on_responses_only applied (Gemma 4 token format)")
print(f"        instruction_part={sot_token!r}+'user\\n' (loss masked)")
print(f"        response_part={sot_token!r}+'model\\n' (loss computed)")

# %% [markdown]
# ## 7. Train（checkpoint resume 対応）
#
# - CKPT_DIR に既存 checkpoint があれば自動 resume
# - 無ければ新規スタート
# - 12h セッション切れ後、このセルを再実行するだけで最後の checkpoint から継続
# - dry-run リハ手順: `DRY_RUN=True` で 5 step 回ったら「ランタイム > ランタイムを接続解除して削除」→ 再接続 → 本セル再実行 → resume できれば成功

# %%
from transformers.trainer_utils import get_last_checkpoint

last_ckpt = None
if os.path.isdir(CKPT_DIR):
    try:
        last_ckpt = get_last_checkpoint(CKPT_DIR)
    except Exception as e:
        print(f"[resume] get_last_checkpoint failed: {e}")

if last_ckpt:
    print(f"[resume] resuming from {last_ckpt}")
    trainer_stats = trainer.train(resume_from_checkpoint=last_ckpt)
else:
    print("[resume] no checkpoint found, starting fresh")
    trainer_stats = trainer.train()

print(trainer_stats)

# %% [markdown]
# ## 7.5. Tokenizer 健全性チェック (Phase 4 #179、post-train)
#
# fine-tune 後に終端 token (Gemma 4: `<turn|>` id=106) が special vocab から外れたり、
# 別 id に振り替わっていないことを確認 (PEFT #1003 + LoRA fine-tune での tokenizer drift)。
# 分裂 / id 変化検出 → 即 alarm、merged_4bit 経路 / generate 経路で EOS 検出失敗の前兆。
#
# Phase 4 (4/29 Colab Pro 訓練) では Unsloth が checkpoint-N/tokenizer_config.json に
# `added_tokens_decoder` metadata を毎 save 時に restore するため、本 check は通る想定。

# %%
EOT_POST = _check_eot_single_token(tokenizer, "post-train")
assert EOT_POST == EOT_TOKEN_ID, (
    f"終端 token id が SFT 前後で変化 ({EOT_TOKEN_ID} → {EOT_POST})、"
    f"PEFT #1003 顕在化、generate 経路で EOS 検出不能になる"
)
print(f"[sanity-eot/post-train] OK: token id {EOT_POST} 不変")

# %% [markdown]
# ## 8. Eval loss + サンプル生成
#
# **注意**: transformers の `NotebookProgressCallback` は train 完了後に `training_tracker=None` に
# なるため、その直後に `trainer.evaluate()` を呼ぶと `on_train_begin must be called before on_evaluate`
# で落ちる。ここでは通常の `ProgressCallback` に差し替えてから評価する（既知の workaround）。

# %%
from transformers.trainer_callback import ProgressCallback
from transformers.utils.notebook import NotebookProgressCallback

for cb in list(trainer.callback_handler.callbacks):
    if isinstance(cb, NotebookProgressCallback):
        trainer.callback_handler.remove_callback(cb)
trainer.callback_handler.add_callback(ProgressCallback())

eval_stats = trainer.evaluate()
print("eval:", eval_stats)

# %%
SYSTEM_PROMPT_PATH = os.path.join(DATA_DIR, "SYSTEM_PROMPT.txt")

def gen(user_text, max_new_tokens=400):
    SYSTEM = open(SYSTEM_PROMPT_PATH).read().strip()  # prepare-train.mjs と同じ本文
    # Gemma 4 processor は multimodal 対応のため content は list[{type,text}] 形式必須
    # string 直渡しは tokenize=True 経路で `string indices must be integers` で落ちる（4/19 検証済）
    messages = [{
        "role": "user",
        "content": [{"type": "text", "text": f"{SYSTEM}\n\n{user_text}"}],
    }]
    inputs = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True, return_tensors="pt"
    ).to("cuda")
    out = model.generate(
        inputs,
        max_new_tokens=max_new_tokens,
        temperature=0.7,
        top_k=40,
        do_sample=True,
    )
    return tokenizer.decode(out[0][inputs.shape[1]:], skip_special_tokens=True)

for prompt in ["ぴょんぴょんはねるねこをつくって", "はなびがどかーんってなるやつ", "もぐらたたきゲーム"]:
    print(f"--- {prompt} ---")
    print(gen(prompt)[:600])
    print()

# %% [markdown]
# ## 9. LoRA adapter 保存 + merged weight export
#
# - **LoRA adapter は必ず Drive 保存**（~50MB、軽い）。resume / 再評価 / 手元 merge の入口
# - **merged 16bit（~10GB）は Colab T4 の RAM（~12GB）に収まらず OOM で落ちる**
#   （2026-04-19 dry-run で実証）。本番でも Colab 内 merge は **NG**
# - 本番 merge は手元 RTX 2060 Mobile（32GB system RAM）で `FastModel.save_pretrained_merged`
#   を別途実行、または Colab Pro+ の High-RAM インスタンスを使う
# - DRY_RUN では merge は常にスキップ

# %%
OUT_ADAPTER = os.path.join(MERGED_DIR, "lora")
OUT_MERGED  = os.path.join(MERGED_DIR, "merged_16bit")

model.save_pretrained(OUT_ADAPTER)
tokenizer.save_pretrained(OUT_ADAPTER)
print(f"[save] adapter -> {OUT_ADAPTER}")

# merged export は RAM 要件高いので条件付き
EXPORT_MERGED = False  # Colab T4 (~12GB RAM) では True にすると OOM。手元 or High-RAM 専用

if EXPORT_MERGED and not DRY_RUN:
    # merged 16bit export（LiteRT 変換用）
    model.save_pretrained_merged(
        OUT_MERGED,
        tokenizer,
        save_method="merged_16bit",
    )
    print(f"[save] merged -> {OUT_MERGED}")
else:
    print("[skip] merged export（EXPORT_MERGED=False or DRY_RUN）")
    print(f"       adapter: {OUT_ADAPTER}")
    print("       merge 環境は Issue #121 で調査中（merged_4bit が ai-edge-torch 入力 OK なら Colab 完結）")
    print("       手元 RTX 2060 Mobile 16GB RAM は swap 併用で OOM リスク 50/50、cloud CPU 1h (~$0.5) が確実")

# %% [markdown]
# ## 10. 次のステップ（本ノートブックの外）
#
# 1. `OUT_MERGED`（= Drive）から手元 RTX 2060 環境に `rclone` / Drive 共有リンクで回収
# 2. `ai-edge-torch` で `.task` / `.litertlm` に変換は **#129 で toolchain 不在確定**、
#    submission 期間内は LiteRT-LM の公式対応待ち (`project_post_submission_webgpu_path.md`)
# 3. **#163 / C-4 経路の 100 件 eval は別 notebook (`sft/notebook/eval-sft-100.py`) で実行**
#    (HF Hub から adapter direct load で training skip、self-contained で Drive / Kaggle Dataset 不要)。
#    実行後 `eval_phase4_sft.json` をローカル DL → `node sft/scripts/validate-results.mjs --in=<path>`
#    で 100 件 execute → SFT_SCORE 確定 → HF model card placeholder 埋め
# 4. RAFT Round 1 の reward 計算用に LoRA adapter を temporary で利用
#
# **Google One 5/11 失効前の cleanup**:
# - 5/10 までに `OUT_MERGED` を手元にバックアップ後、CKPT_DIR 内の中間 checkpoint を全削除して Drive 使用量を 15GB free tier 内に落とす
# - `!rm -rf /content/drive/MyDrive/vibekid-sft-ckpt/checkpoint-*` で中間のみ削除可
