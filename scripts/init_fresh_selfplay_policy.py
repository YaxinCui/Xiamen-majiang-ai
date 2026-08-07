#!/usr/bin/env python3
"""Create an auditable, randomly initialized policy for a new self-play lineage.

This command deliberately creates a *new* actor rather than loading or
mutating any historical neural checkpoint.  It is a training bootstrap only:
the saved policy has not played an evaluation match and is never authorized
for the browser merely because it can be loaded.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    import torch
except ModuleNotFoundError as error:
    raise SystemExit("请使用 .venv/bin/python 运行；该脚本需要 PyTorch") from error

from xiamen_mahjong.torch_policy import (
    ACTION_SELECTION_POLICY,
    ARCHITECTURE_CANDIDATE_MLP,
    TorchPolicyValueAgent,
)
from xiamen_mahjong.training import NEURAL_FEATURE_DIMS


def source_revision() -> str | None:
    """Return the source revision without making initialization depend on Git."""

    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    revision = result.stdout.strip()
    return revision or None


def checkpoint_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def create_fresh_policy(
    *,
    seed: int,
    feature_version: int,
    hidden_size: int,
    device: str,
) -> TorchPolicyValueAgent:
    """Instantiate a reproducible actor with no checkpoint ancestry."""

    if feature_version not in NEURAL_FEATURE_DIMS:
        raise ValueError("不支持的 feature_version")
    if hidden_size <= 0:
        raise ValueError("hidden_size 必须为正数")
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("请求 CUDA 但当前 PyTorch 无可用 GPU")
    torch.manual_seed(seed)
    if device == "cuda":
        torch.cuda.manual_seed_all(seed)
    return TorchPolicyValueAgent(
        feature_version=feature_version,
        hidden_size=hidden_size,
        architecture=ARCHITECTURE_CANDIDATE_MLP,
        action_selection=ACTION_SELECTION_POLICY,
        device=device,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--profile", choices=("classic", "core"), default="classic")
    parser.add_argument(
        "--feature-version", type=int, default=3, choices=tuple(sorted(NEURAL_FEATURE_DIMS))
    )
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output.resolve() == args.report.resolve():
        raise ValueError("checkpoint 与 report 必须是不同文件")
    if args.output.exists() or args.report.exists():
        raise FileExistsError("fresh self-play 初始化拒绝覆盖既有产物")
    agent = create_fresh_policy(
        seed=args.seed,
        feature_version=args.feature_version,
        hidden_size=args.hidden_size,
        device=args.device,
    )
    metadata = {
        "initialization": "fresh_random_actor_for_self_play",
        "not_trained": True,
        "authorized_for_browser": False,
        "profile": args.profile,
        "random_seed": args.seed,
        "source_revision": source_revision(),
        "ancestry": "none",
        "prohibited_ancestry": [
            "historical_run3",
            "historical_run4",
            "historical_teacher_clone_checkpoint",
        ],
    }
    agent.save(args.output, metadata=metadata)
    report = {
        **metadata,
        "checkpoint": str(args.output),
        "checkpoint_sha256": checkpoint_sha256(args.output),
        "architecture": ARCHITECTURE_CANDIDATE_MLP,
        "feature_version": args.feature_version,
        "hidden_size": args.hidden_size,
        "action_selection": ACTION_SELECTION_POLICY,
        "initialization_device": args.device,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
