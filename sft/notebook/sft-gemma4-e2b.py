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
# **Status: 骨組み (4/19)。本番 run は 4/25 Colab Pro で実行**
#
# - Model: `unsloth/gemma-4-E2B-it`
# - LoRA r=16 α=32, text decoder のみ（multi-modal 層汚染防止）
# - bf16（T4 Colab Pro、Unsloth が float16 infinite activation を回避）
# - seq_len 512, batch 1, grad_accum 8 → 1 epoch ≈ 2-3h
# - Data: `sft/data/train.jsonl` (657件) / `sft/data/eval.jsonl` (100件, category 均等)
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

# %% [markdown]
# ## 0. 環境確認（Colab Pro T4 想定）

# %%
# !nvidia-smi
# !free -h
# !df -h /content

# %% [markdown]
# ## 1. Unsloth + 依存インストール
#
# Colab でのみ実行。ローカル Linux GPU で試すなら pip 版を manual 指定。

# %%
# !pip install --quiet --no-deps "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
# !pip install --quiet --no-deps trl peft accelerate bitsandbytes
# !pip install --quiet datasets xformers

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

TRAIN_PATH = "/content/sft/data/train.jsonl"
EVAL_PATH  = "/content/sft/data/eval.jsonl"

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

training_args = SFTConfig(
    output_dir="/content/sft-gemma4-e2b-out",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,       # 実効 batch = 8
    num_train_epochs=1,
    learning_rate=2e-4,                  # LoRA SFT 標準
    warmup_ratio=0.03,
    lr_scheduler_type="cosine",
    bf16=True,
    fp16=False,                          # T4 で破綻するため禁止
    optim="adamw_8bit",
    weight_decay=0.01,
    logging_steps=10,
    eval_strategy="steps",
    eval_steps=50,
    save_strategy="steps",
    save_steps=100,
    save_total_limit=3,
    max_seq_length=MAX_SEQ_LENGTH,
    dataset_text_field="text",
    packing=False,                       # packing は Gemma 4 で検証不足
    report_to="none",
    seed=42,
)

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset["train"],
    eval_dataset=dataset["eval"],
    args=training_args,
)

# %% [markdown]
# ## 7. Train

# %%
trainer_stats = trainer.train()
print(trainer_stats)

# %% [markdown]
# ## 8. Eval loss + サンプル生成

# %%
eval_stats = trainer.evaluate()
print("eval:", eval_stats)

# %%
def gen(user_text, max_new_tokens=400):
    SYSTEM = open("/content/sft/scripts/SYSTEM_PROMPT.txt").read().strip()  # prepare-train.mjs と同じ本文
    messages = [{"role": "user", "content": f"{SYSTEM}\n\n{user_text}"}]
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
# LiteRT-LM web 配信用に merged で書き出す。変換は別環境で `ai-edge-torch` を使う。

# %%
OUT_ADAPTER = "/content/sft-gemma4-e2b-lora"
OUT_MERGED  = "/content/sft-gemma4-e2b-merged"

model.save_pretrained(OUT_ADAPTER)
tokenizer.save_pretrained(OUT_ADAPTER)
print(f"[save] adapter -> {OUT_ADAPTER}")

# merged 16bit export（LiteRT 変換用）
model.save_pretrained_merged(
    OUT_MERGED,
    tokenizer,
    save_method="merged_16bit",
)
print(f"[save] merged -> {OUT_MERGED}")

# %% [markdown]
# ## 10. 次のステップ（本ノートブックの外）
#
# 1. `OUT_MERGED` を Google Drive に退避（Colab 12h セッション切れ対策）
# 2. `ai-edge-torch` で `.task` / `.litertlm` に変換（別 notebook、4/30〜）
# 3. `sft/scripts/evaluate.mjs` で exec_success_rate 測定（base vs SFT）
# 4. RAFT Round 1 の reward 計算用に LoRA adapter を temporary で利用
