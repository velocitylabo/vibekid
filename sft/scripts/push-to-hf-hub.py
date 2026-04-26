"""LoRA adapter + tokenizer + chat_template を HuggingFace Hub に push する。

#131 (HF Hub に LoRA adapter + eval 数値入り model card 公開) の upload 用。
本 script は SFT 完了後の 4-point set（adapter / config / tokenizer / chat_template）
+ optional artifacts（training_args.json / model card README）を 1 repo に集約する。

前提:
- HF Hub アカウント + access token（write 権限）
- token 経路: 環境変数 `HF_TOKEN` or `~/.cache/huggingface/token`（`huggingface-cli login`）
- 対象 LoRA adapter ディレクトリに最低限以下が揃っていること:
    - adapter_model.safetensors
    - adapter_config.json
    - tokenizer_config.json
    - chat_template.jinja  ← Gemma 4 地雷対策（transformers #45205 参照）
    - tokenizer.json (or tokenizer.model)
    - special_tokens_map.json

使い方:
    # 最小例
    python sft/scripts/push-to-hf-hub.py \\
        --adapter-dir /path/to/lora \\
        --repo-id user/vibekid-gemma-4-E2B-lora

    # model card 同梱 + private repo
    python sft/scripts/push-to-hf-hub.py \\
        --adapter-dir /path/to/lora \\
        --repo-id user/vibekid-gemma-4-E2B-lora \\
        --model-card sft/model-card/README.md \\
        --training-args sft/logs/training_args.json \\
        --private

依存（pip install huggingface_hub）。Modal や Colab で動かす場合は同じスクリプトを
そのまま module として呼び出すか、subprocess で起動して良い。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

REQUIRED_FILES = (
    "adapter_model.safetensors",
    "adapter_config.json",
    "tokenizer_config.json",
    "chat_template.jinja",
)

OPTIONAL_FILES = (
    "tokenizer.json",
    "tokenizer.model",
    "special_tokens_map.json",
    "added_tokens.json",
)


def _validate_adapter_dir(adapter_dir: Path) -> None:
    if not adapter_dir.is_dir():
        raise SystemExit(f"adapter dir not found: {adapter_dir}")

    missing = [f for f in REQUIRED_FILES if not (adapter_dir / f).is_file()]
    if missing:
        raise SystemExit(
            "adapter dir is missing required files:\n  - "
            + "\n  - ".join(missing)
            + f"\n(checked under {adapter_dir})\n\n"
            "Gemma 4 SFT は chat_template.jinja を tokenizer 側にコピーしておくこと。"
            " transformers #45205 / project_modal_merge_dry_run.md 参照。"
        )

    has_tokenizer = any((adapter_dir / f).is_file() for f in ("tokenizer.json", "tokenizer.model"))
    if not has_tokenizer:
        raise SystemExit(
            "tokenizer.json も tokenizer.model も無い。inference 用に必須なので "
            "Colab notebook の保存セルで `tokenizer.save_pretrained` まで通っているか確認。"
        )


def _resolve_token(cli_token: str | None) -> str | None:
    if cli_token:
        return cli_token
    if env := os.environ.get("HF_TOKEN"):
        return env
    if env := os.environ.get("HUGGINGFACE_HUB_TOKEN"):
        return env
    return None


def _load_card_text(model_card: Path | None, training_args: Path | None) -> str | None:
    if model_card is None:
        return None
    if not model_card.is_file():
        raise SystemExit(f"model card not found: {model_card}")

    text = model_card.read_text(encoding="utf-8")
    if training_args is not None:
        if not training_args.is_file():
            raise SystemExit(f"training args not found: {training_args}")
        args = json.loads(training_args.read_text(encoding="utf-8"))
        block = "```json\n" + json.dumps(args, ensure_ascii=False, indent=2) + "\n```"
        text = text.replace("<!-- TRAINING_ARGS_PLACEHOLDER -->", block)
    return text


def push(
    adapter_dir: Path,
    repo_id: str,
    *,
    token: str | None,
    private: bool,
    commit_message: str,
    model_card_text: str | None,
    extra_paths: list[Path],
) -> str:
    from huggingface_hub import HfApi, create_repo, upload_file, upload_folder

    api = HfApi(token=token)

    create_repo(repo_id=repo_id, token=token, private=private, exist_ok=True, repo_type="model")

    upload_folder(
        repo_id=repo_id,
        folder_path=str(adapter_dir),
        commit_message=commit_message,
        token=token,
        repo_type="model",
        ignore_patterns=["*.bak", "__pycache__/*", ".DS_Store"],
    )

    if model_card_text is not None:
        upload_file(
            path_or_fileobj=model_card_text.encode("utf-8"),
            path_in_repo="README.md",
            repo_id=repo_id,
            commit_message=f"docs(card): update model card ({commit_message})",
            token=token,
            repo_type="model",
        )

    for path in extra_paths:
        if not path.is_file():
            print(f"[skip] extra path not a file: {path}", file=sys.stderr)
            continue
        upload_file(
            path_or_fileobj=str(path),
            path_in_repo=path.name,
            repo_id=repo_id,
            commit_message=f"chore: add {path.name}",
            token=token,
            repo_type="model",
        )

    info = api.repo_info(repo_id=repo_id, token=token, repo_type="model")
    return info.id  # `<user>/<name>`


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--adapter-dir", type=Path, required=True, help="LoRA adapter のローカル dir")
    parser.add_argument("--repo-id", required=True, help="HF Hub の repo id（user/name）")
    parser.add_argument("--token", default=None, help="HF Hub write token（既定は env HF_TOKEN）")
    parser.add_argument("--private", action="store_true", help="private repo として作成")
    parser.add_argument(
        "--commit-message",
        default="feat: initial LoRA adapter upload",
        help="upload commit message",
    )
    parser.add_argument("--model-card", type=Path, default=None, help="model card README 経路")
    parser.add_argument(
        "--training-args",
        type=Path,
        default=None,
        help="model card 内 <!-- TRAINING_ARGS_PLACEHOLDER --> に注入する JSON",
    )
    parser.add_argument(
        "--extra",
        type=Path,
        action="append",
        default=[],
        help="追加で upload するファイル（複数指定可、root に置かれる）",
    )

    args = parser.parse_args(argv)

    _validate_adapter_dir(args.adapter_dir)
    token = _resolve_token(args.token)
    if token is None:
        raise SystemExit(
            "HF token not found. Set HF_TOKEN env or `huggingface-cli login` or pass --token."
        )

    model_card_text = _load_card_text(args.model_card, args.training_args)

    repo_full = push(
        adapter_dir=args.adapter_dir,
        repo_id=args.repo_id,
        token=token,
        private=args.private,
        commit_message=args.commit_message,
        model_card_text=model_card_text,
        extra_paths=args.extra,
    )

    print(f"pushed: https://huggingface.co/{repo_full}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
