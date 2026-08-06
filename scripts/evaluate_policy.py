#!/usr/bin/env python3
"""Run a seat-rotated paired evaluation against a fixed opponent roster."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from xiamen_mahjong.agents import HeuristicTeacherAgent
from xiamen_mahjong.evaluation import (
    evaluate_against_opponent_roster,
    paired_score_comparison,
)
from xiamen_mahjong.training import NeuralRulePolicyModel, RulePolicyModel


def checkpoint_sha256(path: Path) -> str:
    """Return a content identity for a locally evaluated checkpoint.

    Evaluation JSON files are often retained longer than their surrounding
    shell command.  Recording a digest prevents a later replacement of a
    same-named checkpoint from making a result ambiguous.
    """

    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    candidate_group = parser.add_mutually_exclusive_group()
    candidate_group.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("artifacts/rule-policy-classic-mlp/rule-policy.json"),
    )
    candidate_group.add_argument(
        "--teacher-candidate",
        action="store_true",
        help="以规则 Teacher 作为候选，建立与 checkpoint 相同对手阵容下的联赛基线",
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
        choices=("policy", "action_value", "response_action_value"),
        help=".pt 候选的选牌头；默认使用 checkpoint 保存的模式（通常为 policy）",
    )
    parser.add_argument(
        "--reference-action-selection",
        choices=("policy", "action_value", "response_action_value"),
        default="policy",
        help="配对基准 .pt 的选牌头；默认 policy，避免意外比较两个实验规则",
    )
    parser.add_argument("--hands", type=int, default=100, help="独立随机牌墙数量")
    parser.add_argument("--seed", type=int, default=20260804)
    parser.add_argument(
        "--opponent-checkpoint",
        type=Path,
        action="append",
        default=[],
        help=(
            "可重复至多三次；从固定阵容左侧开始替换 Teacher。候选在四座轮换中会对每名对手覆盖全部相对座位"
        ),
    )
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


def load_opponent_roster(args: argparse.Namespace):
    if len(args.opponent_checkpoint) > 3:
        raise ValueError("opponent-checkpoint 最多指定三次")
    opponents = []
    identities: list[dict[str, str | None]] = []
    for index, path in enumerate(args.opponent_checkpoint):
        agent = load_policy(path, device=args.device)
        opponents.append(agent)
        identities.append(
            {
                "label": f"checkpoint_{index}:{path}",
                "kind": "checkpoint",
                "checkpoint": str(path),
                "checkpoint_sha256": checkpoint_sha256(path),
                "action_selection": getattr(agent, "action_selection", None),
            }
        )
    while len(opponents) < 3:
        opponents.append(HeuristicTeacherAgent())
        identities.append(
            {
                "label": f"heuristic_teacher_{len(identities)}",
                "kind": "heuristic_teacher",
                "checkpoint": None,
                "checkpoint_sha256": None,
                "action_selection": None,
            }
        )
    return tuple(opponents), tuple(item["label"] for item in identities), identities


def main() -> None:
    args = parse_args()
    if args.teacher_candidate:
        if args.action_selection is not None:
            raise ValueError("teacher-candidate 不支持 action-selection")
        policy = HeuristicTeacherAgent()
    else:
        policy = load_policy(
            args.checkpoint,
            device=args.device,
            action_selection=args.action_selection,
        )
    opponents, opponent_labels, opponent_identities = load_opponent_roster(args)
    result = evaluate_against_opponent_roster(
        policy,
        opponents,
        opponent_labels=opponent_labels,
        hands=args.hands,
        profile=args.profile,
        seed=args.seed,
    )
    payload = result.payload(include_scores=args.include_scores)
    payload["candidate_kind"] = (
        "heuristic_teacher" if args.teacher_candidate else "checkpoint"
    )
    payload["checkpoint"] = None if args.teacher_candidate else str(args.checkpoint)
    payload["checkpoint_sha256"] = (
        None if args.teacher_candidate else checkpoint_sha256(args.checkpoint)
    )
    payload["inference_device"] = (
        args.device
        if not args.teacher_candidate and args.checkpoint.suffix in {".pt", ".pth"}
        else None
    )
    payload["action_selection"] = (
        getattr(policy, "action_selection", None)
        if not args.teacher_candidate and args.checkpoint.suffix in {".pt", ".pth"}
        else None
    )
    payload["opponent_roster"] = opponent_identities
    if args.reference:
        reference = load_policy(
            args.reference,
            device=args.device,
            action_selection=args.reference_action_selection,
        )
        reference_opponents, reference_labels, _ = load_opponent_roster(args)
        reference_result = evaluate_against_opponent_roster(
            reference,
            reference_opponents,
            opponent_labels=reference_labels,
            hands=args.hands,
            profile=args.profile,
            seed=args.seed,
        )
        payload["reference_checkpoint"] = str(args.reference)
        payload["reference_checkpoint_sha256"] = checkpoint_sha256(args.reference)
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
