#!/usr/bin/env python3
"""Export cross-fitted, winsorized Teacher-relative advantage labels.

The source corpus logs exactly one epsilon-randomized action and then resumes
the frozen rule Teacher.  A direct outcome model that was trained on disjoint
physical walls supplies the direct term.  This command writes a *new*, narrow
JSONL format with no terminal outcome: its labels are explicitly biased
winsorized DR pseudo-advantages, not action-Q values and not OPE estimates.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.audit_teacher_response_intervention_ope import teacher_epsilon_propensities
from xiamen_mahjong.off_policy import (
    LoggedIntervention,
    winsorized_doubly_robust_action_advantages,
)
from xiamen_mahjong.torch_policy import TorchPolicyValueAgent
from xiamen_mahjong.training import TrainingTrajectory, read_trajectory_jsonl


RELATIVE_ADVANTAGE_VERSION = "xiamen-teacher-relative-advantage-v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--fold-index", type=int, required=True)
    parser.add_argument("--crossfit-manifest", type=Path, required=True)
    parser.add_argument("--direct-checkpoint", type=Path, required=True)
    parser.add_argument("--direct-report", type=Path, required=True)
    parser.add_argument("--maximum-abs-correction", type=float, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--intervention-phase", choices=("discard", "response"), default="discard")
    parser.add_argument("--value-scale", type=float, default=80.0)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _group_ids(trajectories: Iterable[TrainingTrajectory]) -> set[str]:
    groups = {item.split_group_id for item in trajectories}
    if None in groups or not all(isinstance(item, str) and item for item in groups):
        raise ValueError("cross-fit 输入缺少 split_group_id")
    return {str(item) for item in groups}


def _summary(values: list[float]) -> dict[str, float | int]:
    if not values:
        raise ValueError("没有可导出的非 Teacher 优势标签")
    center = sum(values) / len(values)
    return {
        "count": len(values),
        "mean": center,
        "std": math.sqrt(sum((item - center) ** 2 for item in values) / len(values)),
        "minimum": min(values),
        "maximum": max(values),
    }


def _load_and_validate_direct_contract(args: argparse.Namespace) -> tuple[dict[str, Any], list[TrainingTrajectory]]:
    report = json.loads(args.direct_report.read_text(encoding="utf-8"))
    boundary = report.get("behavior_boundary")
    if not isinstance(boundary, dict):
        raise ValueError("direct report 缺少 behavior_boundary")
    if report.get("status") != "diagnostic_only_not_authorized_for_action_selection":
        raise ValueError("direct checkpoint 不是诊断性 outcome 模型")
    if boundary.get("fixed_epochs_without_validation") is not True:
        raise ValueError("cross-fit direct model 必须固定 epoch 且不读 validation")
    if boundary.get("terminal_test_read") is not False:
        raise ValueError("cross-fit direct model 不得读取 terminal")
    if boundary.get("only_randomized_actions") is not True:
        raise ValueError("direct model 必须只从随机干预动作学习")
    if boundary.get("decision_phase") != args.intervention_phase:
        raise ValueError("direct model phase 与导出 phase 不一致")
    selection = report.get("checkpoint_selection")
    if not isinstance(selection, dict) or selection.get("metric") != "fixed_pre_registered_epoch":
        raise ValueError("direct report 未记录固定预注册 epoch")
    inputs = report.get("inputs")
    if not isinstance(inputs, dict) or inputs.get("validation") != [] or inputs.get("test") != []:
        raise ValueError("direct model 不得读 validation 或 test")
    raw_train_paths = inputs.get("train")
    if not isinstance(raw_train_paths, list) or len(raw_train_paths) < 2:
        raise ValueError("direct report 必须列出至少两个训练 fold")
    train_paths = [Path(str(path)) for path in raw_train_paths]
    if any(path.resolve() == args.input.resolve() for path in train_paths):
        raise ValueError("direct model 训练输入包含待预测 fold")
    return report, [row for path in train_paths for row in read_trajectory_jsonl(path)]


def _validate_crossfit_fold(args: argparse.Namespace, input_sha256: str) -> dict[str, Any]:
    manifest = json.loads(args.crossfit_manifest.read_text(encoding="utf-8"))
    if manifest.get("status") != "cross_fitting_folds_created" or manifest.get("group_overlap") != 0:
        raise ValueError("crossfit manifest 不满足完整墙分组隔离")
    folds = manifest.get("folds")
    if not isinstance(folds, list):
        raise ValueError("crossfit manifest 缺少 folds")
    matched = [fold for fold in folds if fold.get("fold") == args.fold_index]
    if len(matched) != 1:
        raise ValueError("fold-index 不属于 crossfit manifest")
    fold = matched[0]
    if fold.get("sha256") != input_sha256:
        raise ValueError("输入文件 hash 与指定 crossfit fold 不匹配")
    return {
        "fold_count": manifest.get("fold_count"),
        "split_salt": manifest.get("split_salt"),
        "input_sha256": manifest.get("input_sha256"),
        "fold_sha256": input_sha256,
        "fold_index": args.fold_index,
    }


def export_rows(
    trajectories: Iterable[TrainingTrajectory],
    *,
    agent: TorchPolicyValueAgent,
    source_sha256: str,
    direct_checkpoint_sha256: str,
    maximum_abs_correction: float,
    phase: str,
    value_scale: float,
) -> tuple[list[dict[str, Any]], list[float]]:
    rows: list[dict[str, Any]] = []
    non_teacher_advantages: list[float] = []
    for trajectory in trajectories:
        metadata = trajectory.source_metadata
        if metadata.get("behavior_policy") != "single_intervention_epsilon_uniform":
            raise ValueError("优势导出只接受单点 epsilon 干预轨迹")
        if metadata.get("base_policy") != "heuristic_teacher":
            raise ValueError("优势导出只接受 Teacher 基线轨迹")
        if metadata.get("intervention_phase") != phase:
            raise ValueError("输入轨迹 phase 与优势导出不一致")
        epsilon = metadata.get("uniform_exploration_probability")
        if (
            isinstance(epsilon, bool)
            or not isinstance(epsilon, (int, float))
            or not 0.0 < float(epsilon) < 1.0
        ):
            raise ValueError("输入轨迹缺少有效 epsilon")
        scores = trajectory.outcome.get("scores")
        if not isinstance(scores, list) or len(scores) != 4:
            raise ValueError("输入轨迹缺少四家终局分数")
        for decision_index, decision in enumerate(trajectory.decisions):
            if (
                decision.state.get("phase") != phase
                or decision.executed_index is None
                or decision.executed_probability is None
                or decision.executed_probability >= 1.0
            ):
                continue
            if not trajectory.split_group_id:
                raise ValueError("优势导出要求 split_group_id")
            propensities = teacher_epsilon_propensities(
                action_count=len(decision.legal_actions),
                teacher_index=decision.chosen_index,
                epsilon=float(epsilon),
            )
            if not math.isclose(
                propensities[decision.executed_index],
                decision.executed_probability,
                abs_tol=1e-9,
            ):
                raise ValueError("记录的 executed_probability 与 epsilon 行为契约不一致")
            outcomes = agent.afterstate_outcomes(decision)
            if outcomes is None:
                raise ValueError("direct checkpoint 缺少 afterstate outcome head")
            direct_values = tuple(value * value_scale for value in outcomes[0])
            observation = LoggedIntervention(
                group_id=trajectory.split_group_id,
                logged_index=decision.executed_index,
                propensities=propensities,
                reward=float(scores[decision.seat]),
                baseline_index=decision.chosen_index,
                target_index=decision.chosen_index,
                direct_values=direct_values,
            )
            advantages = winsorized_doubly_robust_action_advantages(
                observation, maximum_abs_correction=maximum_abs_correction
            )
            if abs(advantages[decision.chosen_index]) > 1e-8:
                raise RuntimeError("Teacher 相对优势必须以 Teacher 动作为零点")
            decision_payload = decision.payload()
            decision_payload.pop("seed", None)
            # The output intentionally omits raw terminal reward/outcome and
            # direct values.  A learner gets only actor-visible state, legal
            # actions, Teacher baseline and the explicitly biased label.
            rows.append(
                {
                    "version": RELATIVE_ADVANTAGE_VERSION,
                    "trajectory_id": trajectory.trajectory_id,
                    "split_group_id": trajectory.split_group_id,
                    "decision_index": decision_index,
                    "source_sha256": source_sha256,
                    "direct_checkpoint_sha256": direct_checkpoint_sha256,
                    "maximum_abs_correction": maximum_abs_correction,
                    "decision": decision_payload,
                    "teacher_index": decision.chosen_index,
                    "pseudo_advantages": list(advantages),
                }
            )
            non_teacher_advantages.extend(
                value
                for index, value in enumerate(advantages)
                if index != decision.chosen_index
            )
    if not rows:
        raise ValueError("没有可导出的随机干预决策")
    return rows, non_teacher_advantages


def main() -> None:
    args = parse_args()
    if args.maximum_abs_correction <= 0 or args.value_scale <= 0:
        raise ValueError("maximum-abs-correction 和 value-scale 必须为正数")
    input_sha256 = sha256(args.input)
    crossfit = _validate_crossfit_fold(args, input_sha256)
    report, direct_train = _load_and_validate_direct_contract(args)
    target_trajectories = read_trajectory_jsonl(args.input)
    target_groups = _group_ids(target_trajectories)
    direct_groups = _group_ids(direct_train)
    if target_groups & direct_groups:
        raise ValueError("direct model 的训练墙组与待预测 fold 重叠")
    agent = TorchPolicyValueAgent.load(args.direct_checkpoint, device=args.device)
    checkpoint_sha256 = sha256(args.direct_checkpoint)
    rows, values = export_rows(
        target_trajectories,
        agent=agent,
        source_sha256=input_sha256,
        direct_checkpoint_sha256=checkpoint_sha256,
        maximum_abs_correction=args.maximum_abs_correction,
        phase=args.intervention_phase,
        value_scale=args.value_scale,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True))
            handle.write("\n")
    payload = {
        "status": "cross_fitted_winsorized_labels_not_ope_or_deployable_policy",
        "format": RELATIVE_ADVANTAGE_VERSION,
        "input": str(args.input),
        "direct_checkpoint": str(args.direct_checkpoint),
        "direct_report": str(args.direct_report),
        "direct_report_selection": report.get("checkpoint_selection"),
        "crossfit": crossfit,
        "target_wall_groups": len(target_groups),
        "direct_training_wall_groups": len(direct_groups),
        "wall_group_overlap": 0,
        "intervention_phase": args.intervention_phase,
        "maximum_abs_correction": args.maximum_abs_correction,
        "value_scale": args.value_scale,
        "records": len(rows),
        "non_teacher_pseudo_advantages": _summary(values),
        "terminal_outcomes_in_output": False,
        "warning": (
            "Labels use winsorized DR residuals and are intentionally biased. "
            "They must not be used for OPE, action selection, or a strength claim."
        ),
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
