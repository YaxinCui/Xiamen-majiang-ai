#!/usr/bin/env python3
"""Fit a diagnostic centered policy from cross-fitted Teacher-relative labels.

The input format contains winsorized DR pseudo-advantages, not terminal Q
values.  This trainer has no validation-driven checkpoint selection: a fixed
schedule is used so the untouched selection/terminal walls remain available
for off-policy evaluation of the eventual one-override candidate.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import math
from pathlib import Path
import sys
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    import torch
    from torch.nn import functional as F
except ModuleNotFoundError as error:
    raise SystemExit("请使用 .venv/bin/python 运行；该脚本需要 PyTorch") from error

from scripts.export_crossfit_teacher_advantages import RELATIVE_ADVANTAGE_VERSION
from xiamen_mahjong.relative_advantage import RelativeAdvantageAgent
from xiamen_mahjong.training import NEURAL_FEATURE_DIMS, TeacherDecision, _dense_action_features


@dataclass(frozen=True)
class AdvantageExample:
    candidates: tuple[tuple[float, ...], ...]
    teacher_index: int
    pseudo_advantages: tuple[float, ...]
    split_group_id: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--additional-train", type=Path, action="append", default=[])
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--maximum-abs-correction", type=float, required=True)
    parser.add_argument("--feature-version", type=int, choices=(3,), default=3)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--weight-decay", type=float, default=0.0001)
    parser.add_argument("--huber-delta", type=float, default=16.0)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--seed", type=int, required=True)
    return parser.parse_args()


def _load_examples(
    paths: Iterable[Path], *, feature_version: int, maximum_abs_correction: float
) -> list[AdvantageExample]:
    examples: list[AdvantageExample] = []
    identities: set[tuple[str, int]] = set()
    for path in paths:
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as error:
                    raise ValueError(f"{path}:{line_number} 不是有效 JSON") from error
                if row.get("version") != RELATIVE_ADVANTAGE_VERSION:
                    raise ValueError(f"{path}:{line_number} 不是相对优势标签格式")
                if float(row.get("maximum_abs_correction")) != maximum_abs_correction:
                    raise ValueError(f"{path}:{line_number} 的 C 与训练 C 不一致")
                group_id = row.get("split_group_id")
                trajectory_id = row.get("trajectory_id")
                decision_index = row.get("decision_index")
                if (
                    not isinstance(group_id, str)
                    or not group_id
                    or not isinstance(trajectory_id, str)
                    or not trajectory_id
                    or isinstance(decision_index, bool)
                    or not isinstance(decision_index, int)
                ):
                    raise ValueError(f"{path}:{line_number} 缺少轨迹/墙组身份")
                identity = (trajectory_id, decision_index)
                if identity in identities:
                    raise ValueError("训练标签包含重复 intervention 决策")
                identities.add(identity)
                decision = TeacherDecision.from_payload(dict(row["decision"]))
                teacher_index = row.get("teacher_index")
                raw_advantages = row.get("pseudo_advantages")
                if (
                    isinstance(teacher_index, bool)
                    or not isinstance(teacher_index, int)
                    or teacher_index != decision.chosen_index
                    or not isinstance(raw_advantages, list)
                    or len(raw_advantages) != len(decision.legal_actions)
                    or not all(
                        isinstance(value, (int, float))
                        and not isinstance(value, bool)
                        and math.isfinite(float(value))
                        for value in raw_advantages
                    )
                ):
                    raise ValueError(f"{path}:{line_number} 的相对优势标签无效")
                advantages = tuple(float(value) for value in raw_advantages)
                if abs(advantages[teacher_index]) > 1e-7:
                    raise ValueError("Teacher 标签动作的相对优势必须为零")
                examples.append(
                    AdvantageExample(
                        candidates=tuple(
                            tuple(
                                _dense_action_features(
                                    decision.state,
                                    action,
                                    feature_version=feature_version,
                                )
                            )
                            for action in decision.legal_actions
                        ),
                        teacher_index=teacher_index,
                        pseudo_advantages=advantages,
                        split_group_id=group_id,
                    )
                )
    if not examples:
        raise ValueError("没有可训练的相对优势标签")
    return examples


def batch_tensors(
    rows: list[AdvantageExample], *, feature_dim: int, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    max_actions = max(len(row.candidates) for row in rows)
    candidates = torch.zeros(
        (len(rows), max_actions, feature_dim), dtype=torch.float32, device=device
    )
    mask = torch.zeros((len(rows), max_actions), dtype=torch.bool, device=device)
    teacher_indices = torch.empty(len(rows), dtype=torch.long, device=device)
    targets = torch.zeros((len(rows), max_actions), dtype=torch.float32, device=device)
    for index, row in enumerate(rows):
        count = len(row.candidates)
        candidates[index, :count] = torch.tensor(
            row.candidates, dtype=torch.float32, device=device
        )
        mask[index, :count] = True
        teacher_indices[index] = row.teacher_index
        targets[index, :count] = torch.tensor(
            row.pseudo_advantages, dtype=torch.float32, device=device
        )
    return candidates, mask, teacher_indices, targets


def main() -> None:
    args = parse_args()
    if min(
        args.maximum_abs_correction,
        args.hidden_size,
        args.epochs,
        args.batch_size,
        args.learning_rate,
        args.huber_delta,
    ) <= 0 or args.weight_decay < 0:
        raise ValueError("训练超参数必须有效且为非负/正数")
    torch.manual_seed(args.seed)
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    examples = _load_examples(
        (args.train, *args.additional_train),
        feature_version=args.feature_version,
        maximum_abs_correction=args.maximum_abs_correction,
    )
    feature_dim = NEURAL_FEATURE_DIMS[args.feature_version]
    agent = RelativeAdvantageAgent(
        feature_version=args.feature_version,
        hidden_size=args.hidden_size,
        device=str(device),
    )
    optimizer = torch.optim.AdamW(
        agent.network.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    generator = torch.Generator(device="cpu")
    generator.manual_seed(args.seed)
    history: list[dict[str, float | int]] = []
    for epoch in range(1, args.epochs + 1):
        agent.network.train()
        order = torch.randperm(len(examples), generator=generator).tolist()
        total_loss = 0.0
        total_points = 0
        for start in range(0, len(order), args.batch_size):
            rows = [examples[index] for index in order[start : start + args.batch_size]]
            candidates, mask, teacher_indices, targets = batch_tensors(
                rows, feature_dim=feature_dim, device=device
            )
            predictions = agent.network(candidates, mask, teacher_indices)
            loss = F.huber_loss(
                predictions[mask], targets[mask], reduction="mean", delta=args.huber_delta
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(agent.network.parameters(), max_norm=5.0)
            optimizer.step()
            points = int(mask.sum().item())
            total_loss += float(loss.detach().cpu()) * points
            total_points += points
        history.append(
            {
                "epoch": epoch,
                "train_huber_loss": total_loss / max(total_points, 1),
            }
        )
    agent.network.eval()
    values = [
        value
        for row in examples
        for index, value in enumerate(row.pseudo_advantages)
        if index != row.teacher_index
    ]
    report: dict[str, Any] = {
        "model": "teacher_relative_advantage_centered_mlp",
        "status": "diagnostic_only_not_authorized_for_action_selection",
        "feature_version": args.feature_version,
        "feature_dim": feature_dim,
        "hidden_size": args.hidden_size,
        "device": str(device),
        "seed": args.seed,
        "inputs": [str(args.train), *(str(path) for path in args.additional_train)],
        "training_contract": {
            "label": "cross_fitted_winsorized_dr_advantage_explicitly_biased",
            "maximum_abs_correction": args.maximum_abs_correction,
            "teacher_action_prediction": "exactly_zero_by_centering",
            "terminal_outcomes_in_input": False,
            "validation_or_heldout_read": False,
            "fixed_epochs": args.epochs,
            "huber_delta": args.huber_delta,
        },
        "counts": {
            "decisions": len(examples),
            "wall_groups": len({row.split_group_id for row in examples}),
            "non_teacher_labels": len(values),
            "non_teacher_label_mean": sum(values) / len(values),
            "non_teacher_label_minimum": min(values),
            "non_teacher_label_maximum": max(values),
        },
        "history": history,
        "warning": (
            "This model is trained on biased one-intervention pseudo-labels. It is not a "
            "terminal Q function, a deployable Mahjong policy, or evidence of human-level play."
        ),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    agent.save(args.output_dir / "relative-advantage.pt", metadata=report)
    (args.output_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
