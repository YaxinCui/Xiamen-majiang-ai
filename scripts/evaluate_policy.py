#!/usr/bin/env python3
"""Run a seat-rotated paired evaluation against the frozen rule Teacher."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from xiamen_mahjong.evaluation import evaluate_against_teacher, paired_score_comparison
from xiamen_mahjong.training import NeuralRulePolicyModel, RulePolicyModel


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("artifacts/rule-policy-classic-mlp/rule-policy.json"),
    )
    parser.add_argument("--profile", choices=("classic", "core"), default="classic")
    parser.add_argument(
        "--device",
        choices=("cpu", "cuda"),
        default="cuda",
        help=".pt 模型的推理设备；实验比较应固定同一设备",
    )
    parser.add_argument(
        "--action-selection",
        choices=("policy", "action_value"),
        help=".pt 候选的选牌头；默认使用 checkpoint 保存的模式（通常为 policy）",
    )
    parser.add_argument(
        "--reference-action-selection",
        choices=("policy", "action_value"),
        default="policy",
        help="配对基准 .pt 的选牌头；默认 policy，避免意外比较两个实验规则",
    )
    parser.add_argument("--hands", type=int, default=100, help="独立随机牌墙数量")
    parser.add_argument("--seed", type=int, default=20260804)
    parser.add_argument(
        "--reference",
        type=Path,
        help="可选：在同一组轮换牌墙上评测此基准并输出配对净分差",
    )
    parser.add_argument("--include-scores", action="store_true", help="输出每局原始净得分")
    parser.add_argument(
        "--output",
        type=Path,
        help="可选：将完整评测 JSON 同时写入此文件，便于实验追溯",
    )
    return parser.parse_args()


def load_policy(
    path: Path, *, device: str = "cpu", action_selection: str | None = None
):
    if path.suffix in {".pt", ".pth"}:
        from xiamen_mahjong.torch_policy import TorchPolicyValueAgent

        return TorchPolicyValueAgent.load(
            path, device=device, action_selection=action_selection
        )
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("model") == "legal_action_relu_mlp":
        return NeuralRulePolicyModel.load(path)
    if payload.get("model") == "legal_action_linear_softmax":
        return RulePolicyModel.load(path)
    raise ValueError("检查点不是受支持的合法动作策略模型")


def main() -> None:
    args = parse_args()
    policy = load_policy(
        args.checkpoint,
        device=args.device,
        action_selection=args.action_selection,
    )
    result = evaluate_against_teacher(
        policy, hands=args.hands, profile=args.profile, seed=args.seed
    )
    payload = result.payload(include_scores=args.include_scores)
    payload["inference_device"] = args.device if args.checkpoint.suffix in {".pt", ".pth"} else None
    payload["action_selection"] = (
        getattr(policy, "action_selection", None)
        if args.checkpoint.suffix in {".pt", ".pth"}
        else None
    )
    if args.reference:
        reference = load_policy(
            args.reference,
            device=args.device,
            action_selection=args.reference_action_selection,
        )
        reference_result = evaluate_against_teacher(
            reference, hands=args.hands, profile=args.profile, seed=args.seed
        )
        payload["reference_checkpoint"] = str(args.reference)
        payload["reference_action_selection"] = (
            getattr(reference, "action_selection", None)
            if args.reference.suffix in {".pt", ".pth"}
            else None
        )
        payload.update(paired_score_comparison(result, reference_result))
    rendered = json.dumps(payload, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
