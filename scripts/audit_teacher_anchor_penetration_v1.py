#!/usr/bin/env python3
"""Audit how far a residual actor can penetrate a fixed Teacher logit prior.

The command reads only actor-visible decisions and policy checkpoints.  It
does not read trajectory outcomes and exports aggregate gap quantiles only.
It is a training-mechanics diagnostic, never a policy selector.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Iterable


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

from xiamen_mahjong.torch_policy import TorchPolicyValueAgent
from xiamen_mahjong.training import TeacherDecision


REPORT_VERSION = "xiamen-teacher-anchor-penetration-audit-v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, action="append", required=True)
    parser.add_argument("--maximum-decisions", type=int, default=3000)
    parser.add_argument("--teacher-prior-margin", type=float, required=True)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def actor_visible_decisions(path: Path, *, maximum: int) -> list[TeacherDecision]:
    if maximum <= 0:
        raise ValueError("maximum 必须为正数")
    rows: list[TeacherDecision] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            trajectory = json.loads(line)
            decisions = trajectory.get("decisions")
            if not isinstance(decisions, list):
                raise ValueError("轨迹缺少 decisions")
            for payload in decisions:
                decision = TeacherDecision.from_payload(payload)
                if len(decision.legal_actions) < 2:
                    continue
                rows.append(decision)
                if len(rows) >= maximum:
                    return rows
    if not rows:
        raise ValueError("没有至少两个合法动作的决策")
    return rows


def _quantile(values: torch.Tensor, q: float) -> float:
    return float(torch.quantile(values, q).cpu())


def audit_checkpoint(
    checkpoint: Path,
    decisions: Iterable[TeacherDecision],
    *,
    margin: float,
    batch_size: int,
    device: str,
) -> dict[str, object]:
    if margin <= 0 or batch_size <= 0:
        raise ValueError("margin 和 batch_size 必须为正数")
    decisions = tuple(decisions)
    agent = TorchPolicyValueAgent.load(checkpoint, device=device)
    gaps: list[float] = []
    alternative_probabilities: list[float] = []
    entropies: list[float] = []
    for start in range(0, len(decisions), batch_size):
        batch = decisions[start : start + batch_size]
        for decision, (logits, _value) in zip(
            batch, agent.policy_values_batch(batch)
        ):
            teacher_index = decision.chosen_index
            teacher_logit = float(logits[teacher_index])
            alternative = max(
                float(logit)
                for index, logit in enumerate(logits)
                if index != teacher_index
            )
            gaps.append(alternative - teacher_logit)
            combined = torch.tensor(
                [
                    float(logit) + (0.0 if index == teacher_index else -margin)
                    for index, logit in enumerate(logits)
                ],
                dtype=torch.float64,
            )
            probabilities = torch.softmax(combined, dim=0)
            alternative_probabilities.append(
                1.0 - float(probabilities[teacher_index])
            )
            entropies.append(
                float(
                    -(probabilities * probabilities.clamp_min(1e-30).log()).sum()
                )
            )
    tensor = torch.tensor(gaps, dtype=torch.float64)
    return {
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256(checkpoint),
        "decisions": len(gaps),
        "residual_best_alternative_gap": {
            "minimum": float(tensor.min()),
            "p25": _quantile(tensor, 0.25),
            "median": _quantile(tensor, 0.5),
            "p75": _quantile(tensor, 0.75),
            "p90": _quantile(tensor, 0.9),
            "p95": _quantile(tensor, 0.95),
            "p99": _quantile(tensor, 0.99),
            "maximum": float(tensor.max()),
        },
        "margin_crossing_rates": {
            str(threshold): float((tensor > threshold).double().mean())
            for threshold in (0.0, 1.0, 2.0, 3.0, 4.0, margin)
        },
        "mean_non_teacher_probability_at_margin": sum(
            alternative_probabilities
        )
        / len(alternative_probabilities),
        "mean_entropy_at_margin": sum(entropies) / len(entropies),
    }


def build_report(
    data: Path,
    checkpoints: Iterable[Path],
    *,
    maximum_decisions: int,
    margin: float,
    batch_size: int,
    device: str,
) -> dict[str, object]:
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("请求 CUDA 但当前不可用")
    decisions = actor_visible_decisions(data, maximum=maximum_decisions)
    audits = [
        audit_checkpoint(
            checkpoint,
            decisions,
            margin=margin,
            batch_size=batch_size,
            device=device,
        )
        for checkpoint in checkpoints
    ]
    return {
        "version": REPORT_VERSION,
        "status": "training_mechanics_diagnostic_only",
        "data": str(data),
        "data_sha256": sha256(data),
        "decisions": len(decisions),
        "teacher_prior_margin": margin,
        "trajectory_outcomes_read": False,
        "actor_visible_decisions_only": True,
        "audits": audits,
        "warning": (
            "Gap penetration does not measure action quality and cannot select "
            "a margin, checkpoint or deployable policy."
        ),
    }


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise ValueError("输出已存在，拒绝覆盖")
    report = build_report(
        args.data,
        args.checkpoint,
        maximum_decisions=args.maximum_decisions,
        margin=args.teacher_prior_margin,
        batch_size=args.batch_size,
        device=args.device,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
