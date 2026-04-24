"""Modal CPU で Gemma 4 E2B + LoRA adapter を merge して merged safetensors を返す。

前提:
- Modal アカウント + CLI 認証済（`modal token new`）
- HuggingFace token を Modal Secret 「huggingface」に登録済（`modal secret create huggingface HF_TOKEN=hf_...`）
- HuggingFace で `google/gemma-4-E2B-it` のアクセス承認済（gated）

使い方:
  # 1. 手元の LoRA adapter（~50MB）を Modal Volume に upload
  modal volume create vibekid-merge
  modal volume put vibekid-merge /path/to/local/lora /lora

  # 2. merge 実行（CPU 4 core / 32GB RAM、1-2h）
  modal run sft/scripts/modal-merge.py

  # 3. 結果（merged safetensors）を手元に download
  modal volume get vibekid-merge /merged ./output/gemma4-e2b-merged

4/19 Colab dry-run で `NotImplementedError: revert_weight_conversion` が 3 ルート全滅した背景から、
**本スクリプト自体が Modal CPU で通るかが 4/22-4/24 の dry-run の主目的**。Colab と同じエラーが出たら
手元 RTX 2060 Mobile or Runpod に緊急退避。
"""

import modal

app = modal.App("vibekid-gemma4-merge")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git")
    .pip_install(
        "torch",
        index_url="https://download.pytorch.org/whl/cpu",
    )
    .pip_install(
        # transformers git main (5.7.0.dev0) で AutoProcessor の lazy import が壊れているため
        # stable の 5.6.x に pin（2026-04-24 Modal dry-run で確認）
        "transformers==5.6.2",
        "peft>=0.13.0",
        "accelerate>=1.0.0",
        "safetensors>=0.4.0",
        "pillow",
        "sentencepiece",
        # torchvision / librosa は transformers 5.6.2 の AutoProcessor lazy import を破壊する
        # ため除外（bisect で確認、2026-04-24）。text-only merge には不要
    )
)

volume = modal.Volume.from_name("vibekid-merge", create_if_missing=True)


@app.function(
    image=image,
    cpu=4.0,
    memory=32768,
    volumes={"/vol": volume},
    secrets=[modal.Secret.from_name("huggingface")],
    timeout=60 * 60 * 2,
)
def merge(
    base_model_id: str = "google/gemma-4-E2B-it",
    adapter_subpath: str = "lora",
    output_subpath: str = "merged",
) -> dict:
    """base + LoRA adapter を merge して merged safetensors を保存する。"""
    import os
    import time
    import shutil
    from pathlib import Path

    import torch
    from transformers import AutoModelForImageTextToText, AutoTokenizer
    from peft import PeftModel

    adapter_path = Path("/vol") / adapter_subpath
    output_path = Path("/vol") / output_subpath
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"Base model: {base_model_id}")
    print(f"Adapter: {adapter_path}")
    print(f"Output: {output_path}")

    if not adapter_path.exists():
        raise FileNotFoundError(
            f"Adapter not found at {adapter_path}. "
            f"Upload via: modal volume put vibekid-merge /local/lora /{adapter_subpath}"
        )

    print("\n[1/4] Loading base model (float32, CPU)...")
    start = time.time()
    model = AutoModelForImageTextToText.from_pretrained(
        base_model_id,
        dtype=torch.float32,
        low_cpu_mem_usage=True,
    )
    print(f"  base load: {time.time() - start:.1f}s")

    print("\n[2/4] Attaching LoRA adapter...")
    start = time.time()
    model = PeftModel.from_pretrained(model, str(adapter_path))
    print(f"  adapter attach: {time.time() - start:.1f}s")

    print("\n[3/4] merge_and_unload (Colab failed here with Gemma 4 multimodal)...")
    start = time.time()
    merged = model.merge_and_unload()
    print(f"  merge: {time.time() - start:.1f}s")

    print("\n[4/4] Saving merged weights + tokenizer + chat_template...")
    start = time.time()
    merged.save_pretrained(
        str(output_path),
        safe_serialization=True,
        max_shard_size="2GB",
    )

    # AutoProcessor は Gemma4VideoProcessor が torchvision を要求するが、transformers 5.6.2 の
    # lazy import が torchvision 併用で破損するため tokenizer のみ保存（text-only inference 用途）
    tokenizer = AutoTokenizer.from_pretrained(base_model_id)
    tokenizer.save_pretrained(str(output_path))

    try:
        from huggingface_hub import hf_hub_download

        chat_tpl = hf_hub_download(base_model_id, "chat_template.jinja")
        shutil.copy(chat_tpl, str(output_path / "chat_template.jinja"))
        print(f"  chat_template.jinja: ok")
    except Exception as e:
        print(f"  chat_template.jinja: SKIPPED ({e})")

    print(f"  save: {time.time() - start:.1f}s")

    files = []
    total_mb = 0.0
    for p in sorted(output_path.rglob("*")):
        if p.is_file():
            mb = p.stat().st_size / (1024 * 1024)
            total_mb += mb
            files.append({"name": str(p.relative_to(output_path)), "size_mb": round(mb, 2)})

    print(f"\nMerged output ({total_mb / 1024:.2f} GB total):")
    for f in files:
        print(f"  {f['name']}: {f['size_mb']} MB")

    return {
        "output_path": str(output_path),
        "total_gb": round(total_mb / 1024, 3),
        "file_count": len(files),
        "files": files,
    }


@app.local_entrypoint()
def main(
    base_model_id: str = "google/gemma-4-E2B-it",
    adapter_subpath: str = "lora",
    output_subpath: str = "merged",
):
    result = merge.remote(
        base_model_id=base_model_id,
        adapter_subpath=adapter_subpath,
        output_subpath=output_subpath,
    )
    print("\n=== RESULT ===")
    print(f"Saved to Modal Volume 'vibekid-merge' at {result['output_path']}")
    print(f"Total: {result['total_gb']} GB across {result['file_count']} files")
    print("\nDownload with:")
    print(f"  modal volume get vibekid-merge /{output_subpath} ./output/gemma4-e2b-merged")
