#!/usr/bin/env python3
"""Launch the local Xiamen Mahjong human-vs-Teacher web game."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from xiamen_mahjong.web import serve


def main() -> None:
    parser = argparse.ArgumentParser(description="启动本机厦门麻将网页游戏")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument(
        "--human-log",
        type=Path,
        help=(
            "显式启用本地人类对局记录的 JSONL 路径；仅在一局结束后写入玩家可见状态、"
            "合法动作和公开结算，不会上传，也不会自动进入训练"
        ),
    )
    parser.add_argument(
        "--human-log-purpose",
        choices=("training", "evaluation"),
        help=(
            "人类记录的不可混用用途。training 只可经人工批准后用于模仿；"
            "evaluation 永远不能进入训练，只用于独立真人对局审计。"
        ),
    )
    parser.add_argument(
        "--ai-checkpoint",
        type=Path,
        help=(
            "可选的 policy-value .pt 检查点；显式指定后，三名 AI 使用该候选。"
            "默认仍为规则 Teacher，不会自动切换实验模型"
        ),
    )
    parser.add_argument(
        "--ai-device",
        choices=("cpu", "cuda"),
        default="cpu",
        help="--ai-checkpoint 的推理设备",
    )
    args = parser.parse_args()
    if args.human_log is None and args.human_log_purpose is not None:
        parser.error("--human-log-purpose 必须与 --human-log 一起使用")
    if args.human_log is not None and args.human_log_purpose is None:
        parser.error("启用 --human-log 时必须指定 --human-log-purpose")
    ai_agent = None
    ai_profile = "heuristic_teacher"
    ai_identity = "heuristic_teacher"
    if args.ai_checkpoint is not None:
        if args.ai_checkpoint.suffix not in {".pt", ".pth"}:
            raise ValueError("网页候选 AI 目前只支持 policy-value .pt/.pth checkpoint")
        from xiamen_mahjong.torch_policy import TorchPolicyValueAgent

        ai_agent = TorchPolicyValueAgent.load(args.ai_checkpoint, device=args.ai_device)
        ai_profile = "explicit_policy_value_checkpoint"
        digest = hashlib.sha256()
        with args.ai_checkpoint.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1_048_576), b""):
                digest.update(chunk)
        ai_identity = f"sha256:{digest.hexdigest()}"
    serve(
        args.host,
        args.port,
        human_log=args.human_log,
        human_recording_purpose=args.human_log_purpose,
        ai_agent=ai_agent,
        ai_profile=ai_profile,
        ai_identity=ai_identity,
    )


if __name__ == "__main__":
    main()
