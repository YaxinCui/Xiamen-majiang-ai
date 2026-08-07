#!/usr/bin/env python3
"""Calibrate public action-conditioned outcome heads from logged play.

This trainer is deliberately narrower than counterfactual action-Q training:
each label belongs only to the action that the behavior policy actually
executed.  It therefore never treats a single hidden continuation as labels
for every legal action.  The output remains a diagnostic checkpoint until it
passes both held-out calibration and paired-game promotion gates.
"""

from __future__ import annotations

import argparse
import copy
from collections import Counter
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
    from torch import nn
    from torch.nn import functional as F
except ModuleNotFoundError as error:
    raise SystemExit("请使用 .venv/bin/python 运行；该脚本需要 PyTorch") from error

from xiamen_mahjong.torch_policy import CandidatePolicyValueNetwork, TorchPolicyValueAgent
from xiamen_mahjong.training import NEURAL_FEATURE_DIMS, _dense_action_features, read_trajectory_jsonl


@dataclass(frozen=True)
class AfterstateExample:
    """One logged action and the terminal result of its real continuation."""

    candidates: tuple[tuple[float, ...], ...]
    executed_index: int
    executed_probability: float | None
    score_target: float
    own_win_target: float
    opponent_win_target: float
    source: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument(
        "--additional-train",
        type=Path,
        action="append",
        default=[],
        help="可重复指定；追加按物理牌墙独立的同分区干预数据。",
    )
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument(
        "--additional-validation",
        type=Path,
        action="append",
        default=[],
        help="可重复指定；只能追加与训练墙组不重叠的验证数据。",
    )
    test_group = parser.add_mutually_exclusive_group(required=True)
    test_group.add_argument("--test", type=Path)
    test_group.add_argument(
        "--skip-test",
        action="store_true",
        help="不读取终端集；用于 final OPE 尚未解封时的 checkpoint 训练",
    )
    parser.add_argument(
        "--additional-test",
        type=Path,
        action="append",
        default=[],
        help="可重复指定；只能追加与训练/验证墙组不重叠的测试数据。",
    )
    initialization = parser.add_mutually_exclusive_group(required=True)
    initialization.add_argument(
        "--init-checkpoint",
        type=Path,
        help="历史初始化仅供兼容旧实验；新 Teacher OPE 不应使用已否决 checkpoint",
    )
    initialization.add_argument(
        "--fresh-policy-anchor-seed",
        type=int,
        help=(
            "从固定随机种子创建从未训练的冻结 policy anchor；"
            "同一 outcome ensemble 的每个成员必须使用同一 seed"
        ),
    )
    parser.add_argument(
        "--fresh-anchor-hidden-size",
        type=int,
        default=128,
        help="--fresh-policy-anchor-seed 的 candidate MLP 隐层宽度",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--feature-version", type=int, choices=(3,), default=3)
    parser.add_argument("--epochs", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--weight-decay", type=float, default=0.0001)
    parser.add_argument("--value-scale", type=float, default=80.0)
    parser.add_argument("--score-loss-weight", type=float, default=1.0)
    parser.add_argument("--own-win-loss-weight", type=float, default=0.25)
    parser.add_argument("--opponent-win-loss-weight", type=float, default=0.25)
    parser.add_argument(
        "--include-source",
        action="append",
        default=[],
        help=(
            "可重复指定；默认仅 teacher_self_play 与 candidate_vs_teacher_dagger。"
            "随机探索的终局延续不匹配部署策略，默认排除。"
        ),
    )
    parser.add_argument(
        "--require-known-propensity",
        action="store_true",
        help="丢弃没有 executed_probability 的旧行为记录，用于探索支持度实验。",
    )
    parser.add_argument(
        "--only-randomized-actions",
        action="store_true",
        help="只保留 executed_probability < 1 的单点随机干预动作。",
    )
    parser.add_argument(
        "--decision-phase",
        choices=("all", "discard", "response"),
        default="all",
        help="只保留指定规则阶段的 logged action outcome；用于稀有 response 校准。",
    )
    parser.add_argument(
        "--unfreeze-encoder",
        action="store_true",
        help="实验性：允许更新公共候选编码器；默认只训练新 outcome heads。",
    )
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--seed", type=int, default=20260830)
    return parser.parse_args()


def allowed_sources(args: argparse.Namespace) -> set[str]:
    return set(args.include_source) or {
        "teacher_self_play",
        "candidate_vs_teacher_dagger",
    }


def load_examples(
    path: Path,
    *,
    feature_version: int,
    value_scale: float,
    sources: set[str],
    require_known_propensity: bool = False,
    only_randomized_actions: bool = False,
    decision_phase: str = "all",
) -> list[AfterstateExample]:
    if value_scale <= 0:
        raise ValueError("value-scale 必须为正数")
    if decision_phase not in {"all", "discard", "response"}:
        raise ValueError("decision-phase 必须是 all、discard 或 response")
    examples: list[AfterstateExample] = []
    for trajectory in read_trajectory_jsonl(path):
        source = str(trajectory.source_metadata.get("collector", "legacy"))
        if source not in sources or trajectory.outcome.get("synthetic"):
            continue
        scores = trajectory.outcome.get("scores")
        winner = trajectory.outcome.get("winner")
        if not isinstance(scores, list) or len(scores) != 4:
            raise ValueError("训练轨迹缺少四家终局得分")
        for decision in trajectory.decisions:
            if decision.executed_index is None:
                # Old v2/v3 exports did not distinguish Teacher labels from
                # behavior actions, so they cannot supervise this objective.
                continue
            if require_known_propensity and decision.executed_probability is None:
                continue
            if only_randomized_actions and (
                decision.executed_probability is None
                or decision.executed_probability >= 1.0
            ):
                continue
            if decision_phase != "all" and decision.state.get("phase") != decision_phase:
                continue
            if not 0 <= decision.executed_index < len(decision.legal_actions):
                raise ValueError("行为动作索引与候选动作不匹配")
            examples.append(
                AfterstateExample(
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
                    executed_index=decision.executed_index,
                    executed_probability=decision.executed_probability,
                    score_target=float(scores[decision.seat]) / value_scale,
                    own_win_target=1.0 if winner == decision.seat else 0.0,
                    opponent_win_target=(
                        1.0 if isinstance(winner, int) and winner != decision.seat else 0.0
                    ),
                    source=source,
                )
            )
    if not examples:
        raise ValueError(
            f"{path} 没有带 executed_index 的可用 on-policy 记录；"
            "请用更新后的采集器重新生成数据。"
        )
    return examples


def batch_tensors(
    examples: list[AfterstateExample],
    *,
    feature_dim: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if not examples:
        raise ValueError("batch 不能为空")
    max_actions = max(len(example.candidates) for example in examples)
    candidates = torch.zeros(
        (len(examples), max_actions, feature_dim), dtype=torch.float32, device=device
    )
    action_mask = torch.zeros(
        (len(examples), max_actions), dtype=torch.bool, device=device
    )
    executed = torch.empty(len(examples), dtype=torch.long, device=device)
    score_target = torch.empty(len(examples), dtype=torch.float32, device=device)
    own_win_target = torch.empty(len(examples), dtype=torch.float32, device=device)
    opponent_win_target = torch.empty(len(examples), dtype=torch.float32, device=device)
    for row, example in enumerate(examples):
        count = len(example.candidates)
        if count <= 0 or example.executed_index >= count:
            raise ValueError("afterstate 样本的动作集合无效")
        candidates[row, :count] = torch.tensor(
            example.candidates, dtype=torch.float32, device=device
        )
        action_mask[row, :count] = True
        executed[row] = example.executed_index
        score_target[row] = example.score_target
        own_win_target[row] = example.own_win_target
        opponent_win_target[row] = example.opponent_win_target
    return candidates, action_mask, executed, score_target, own_win_target, opponent_win_target


def outcome_loss(
    network: CandidatePolicyValueNetwork,
    rows: list[AfterstateExample],
    *,
    feature_dim: int,
    device: torch.device,
    score_weight: float,
    own_win_weight: float,
    opponent_win_weight: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    (
        candidates,
        action_mask,
        executed,
        score_target,
        own_win_target,
        opponent_win_target,
    ) = batch_tensors(rows, feature_dim=feature_dim, device=device)
    (
        _logits,
        _value,
        _q,
        predicted_score,
        predicted_own_win,
        predicted_opponent_win,
    ) = network.forward_with_afterstate_outcomes(candidates, action_mask)
    row_index = torch.arange(len(rows), device=device)
    score = predicted_score[row_index, executed]
    own_win = predicted_own_win[row_index, executed]
    opponent_win = predicted_opponent_win[row_index, executed]
    score_loss = F.smooth_l1_loss(score, score_target)
    own_win_loss = F.binary_cross_entropy_with_logits(own_win, own_win_target)
    opponent_win_loss = F.binary_cross_entropy_with_logits(
        opponent_win, opponent_win_target
    )
    total = (
        score_weight * score_loss
        + own_win_weight * own_win_loss
        + opponent_win_weight * opponent_win_loss
    )
    return total, {
        "score_loss": score_loss,
        "own_win_loss": own_win_loss,
        "opponent_win_loss": opponent_win_loss,
        "score": score,
        "score_target": score_target,
        "own_win_probability": torch.sigmoid(own_win),
        "own_win_target": own_win_target,
        "opponent_win_probability": torch.sigmoid(opponent_win),
        "opponent_win_target": opponent_win_target,
    }


def evaluate(
    network: CandidatePolicyValueNetwork,
    examples: list[AfterstateExample],
    *,
    feature_dim: int,
    device: torch.device,
    batch_size: int,
    value_scale: float,
    score_weight: float,
    own_win_weight: float,
    opponent_win_weight: float,
) -> dict[str, float]:
    network.eval()
    total_loss = 0.0
    score_loss = 0.0
    own_win_loss = 0.0
    opponent_win_loss = 0.0
    score_absolute_error = 0.0
    score_squared_error = 0.0
    baseline_score_absolute_error = 0.0
    own_win_brier = 0.0
    opponent_win_brier = 0.0
    own_win_target_sum = 0.0
    opponent_win_target_sum = 0.0
    own_win_probability_sum = 0.0
    opponent_win_probability_sum = 0.0
    count = 0
    with torch.no_grad():
        for start in range(0, len(examples), batch_size):
            rows = examples[start : start + batch_size]
            loss, outputs = outcome_loss(
                network,
                rows,
                feature_dim=feature_dim,
                device=device,
                score_weight=score_weight,
                own_win_weight=own_win_weight,
                opponent_win_weight=opponent_win_weight,
            )
            errors = outputs["score"] - outputs["score_target"]
            own_errors = outputs["own_win_probability"] - outputs["own_win_target"]
            opponent_errors = (
                outputs["opponent_win_probability"] - outputs["opponent_win_target"]
            )
            size = len(rows)
            total_loss += float(loss) * size
            score_loss += float(outputs["score_loss"]) * size
            own_win_loss += float(outputs["own_win_loss"]) * size
            opponent_win_loss += float(outputs["opponent_win_loss"]) * size
            score_absolute_error += float(errors.abs().sum())
            score_squared_error += float((errors * errors).sum())
            baseline_score_absolute_error += float(outputs["score_target"].abs().sum())
            own_win_brier += float((own_errors * own_errors).sum())
            opponent_win_brier += float((opponent_errors * opponent_errors).sum())
            own_win_target_sum += float(outputs["own_win_target"].sum())
            opponent_win_target_sum += float(outputs["opponent_win_target"].sum())
            own_win_probability_sum += float(outputs["own_win_probability"].sum())
            opponent_win_probability_sum += float(outputs["opponent_win_probability"].sum())
            count += size
    if count <= 0:
        raise ValueError("评估集不能为空")
    return {
        "decisions": float(count),
        "total_loss": total_loss / count,
        "score_huber_loss": score_loss / count,
        "own_win_bce": own_win_loss / count,
        "opponent_win_bce": opponent_win_loss / count,
        "score_mae": score_absolute_error / count,
        "score_mae_points": score_absolute_error / count * value_scale,
        "score_rmse_points": math.sqrt(score_squared_error / count) * value_scale,
        "zero_score_mae_points": baseline_score_absolute_error / count * value_scale,
        "own_win_brier": own_win_brier / count,
        "opponent_win_brier": opponent_win_brier / count,
        "own_win_target_rate": own_win_target_sum / count,
        "opponent_win_target_rate": opponent_win_target_sum / count,
        "own_win_prediction_mean": own_win_probability_sum / count,
        "opponent_win_prediction_mean": opponent_win_probability_sum / count,
    }


def source_counts(examples: Iterable[AfterstateExample]) -> dict[str, int]:
    return dict(sorted(Counter(example.source for example in examples).items()))


def load_partition(
    paths: Iterable[Path],
    *,
    feature_version: int,
    value_scale: float,
    sources: set[str],
    require_known_propensity: bool,
    only_randomized_actions: bool,
    decision_phase: str,
) -> list[AfterstateExample]:
    """Combine only pre-partitioned, externally disjoint wall groups."""

    examples: list[AfterstateExample] = []
    for path in paths:
        examples.extend(
            load_examples(
                path,
                feature_version=feature_version,
                value_scale=value_scale,
                sources=sources,
                require_known_propensity=require_known_propensity,
                only_randomized_actions=only_randomized_actions,
                decision_phase=decision_phase,
            )
        )
    return examples


def propensity_summary(examples: Iterable[AfterstateExample]) -> dict[str, float | int | None]:
    records = list(examples)
    values = [
        example.executed_probability
        for example in records
        if example.executed_probability is not None
    ]
    if not values:
        return {
            "known": 0,
            "unknown": len(records),
            "minimum": None,
            "mean": None,
            "maximum": None,
        }
    return {
        "known": len(values),
        "unknown": len(records) - len(values),
        "minimum": min(values),
        "mean": sum(values) / len(values),
        "maximum": max(values),
    }


def initial_agent_for_outcome_training(
    args: argparse.Namespace, *, device: torch.device
) -> tuple[TorchPolicyValueAgent, dict[str, object]]:
    """Load a legacy base or make a reproducible, never-trained policy anchor.

    The fresh path is intentionally a representation/shape anchor only.  Its
    policy logits remain frozen and are never selected for game play; using a
    shared seed makes those logits exactly identical across independently
    trained outcome heads, which is required for their conservative ensemble.
    """

    if args.init_checkpoint is not None:
        agent = TorchPolicyValueAgent.load(args.init_checkpoint, device=str(device))
        return agent, {"kind": "checkpoint", "path": str(args.init_checkpoint)}
    if args.fresh_policy_anchor_seed is None:
        raise ValueError("需要 init-checkpoint 或 fresh-policy-anchor-seed")
    if args.fresh_anchor_hidden_size <= 0:
        raise ValueError("fresh-anchor-hidden-size 必须为正数")
    torch.manual_seed(args.fresh_policy_anchor_seed)
    agent = TorchPolicyValueAgent(
        feature_version=args.feature_version,
        hidden_size=args.fresh_anchor_hidden_size,
        device=str(device),
    )
    return agent, {
        "kind": "fresh_untrained_policy_anchor",
        "seed": args.fresh_policy_anchor_seed,
        "hidden_size": args.fresh_anchor_hidden_size,
    }


def main() -> None:
    args = parse_args()
    if args.epochs <= 0 or args.batch_size <= 0 or args.learning_rate <= 0:
        raise ValueError("epochs、batch-size 和 learning-rate 必须为正数")
    if min(args.score_loss_weight, args.own_win_loss_weight, args.opponent_win_loss_weight) < 0:
        raise ValueError("outcome 损失权重不能为负数")
    if args.score_loss_weight + args.own_win_loss_weight + args.opponent_win_loss_weight <= 0:
        raise ValueError("至少保留一个正的 outcome 损失权重")
    torch.manual_seed(args.seed)
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    sources = allowed_sources(args)
    feature_dim = NEURAL_FEATURE_DIMS[args.feature_version]
    train = load_partition(
        (args.train, *args.additional_train),
        feature_version=args.feature_version,
        value_scale=args.value_scale,
        sources=sources,
        require_known_propensity=args.require_known_propensity,
        only_randomized_actions=args.only_randomized_actions,
        decision_phase=args.decision_phase,
    )
    validation = load_partition(
        (args.validation, *args.additional_validation),
        feature_version=args.feature_version,
        value_scale=args.value_scale,
        sources=sources,
        require_known_propensity=args.require_known_propensity,
        only_randomized_actions=args.only_randomized_actions,
        decision_phase=args.decision_phase,
    )
    if args.skip_test and args.additional_test:
        raise ValueError("skip-test 时不能指定 additional-test")
    test = (
        []
        if args.skip_test
        else load_partition(
            (args.test, *args.additional_test),
            feature_version=args.feature_version,
            value_scale=args.value_scale,
            sources=sources,
            require_known_propensity=args.require_known_propensity,
            only_randomized_actions=args.only_randomized_actions,
            decision_phase=args.decision_phase,
        )
    )
    initial_agent, initialization = initial_agent_for_outcome_training(
        args, device=device
    )
    if not isinstance(initial_agent.network, CandidatePolicyValueNetwork):
        raise ValueError("afterstate outcome 当前只支持 candidate_mlp checkpoint")
    if initial_agent.feature_version != args.feature_version:
        raise ValueError("checkpoint 与 --feature-version 不匹配")
    network = initial_agent.network
    if not args.unfreeze_encoder:
        for parameter in network.parameters():
            parameter.requires_grad = False
        for parameter in network.afterstate_encoder.parameters():
            parameter.requires_grad = True
        for head in (
            network.afterstate_score_head,
            network.afterstate_win_head,
            network.afterstate_opponent_win_head,
        ):
            for parameter in head.parameters():
                parameter.requires_grad = True
    trainable = [parameter for parameter in network.parameters() if parameter.requires_grad]
    if not trainable:
        raise RuntimeError("没有可训练的 afterstate 参数")
    optimizer = torch.optim.AdamW(
        trainable, lr=args.learning_rate, weight_decay=args.weight_decay
    )
    generator = torch.Generator(device="cpu")
    generator.manual_seed(args.seed)
    best_state: dict[str, Any] | None = None
    best_epoch: int | None = None
    best_validation: dict[str, float] | None = None
    history: list[dict[str, Any]] = []
    for epoch in range(1, args.epochs + 1):
        network.train()
        order = torch.randperm(len(train), generator=generator).tolist()
        epoch_total = 0.0
        batches = 0
        for start in range(0, len(order), args.batch_size):
            rows = [train[index] for index in order[start : start + args.batch_size]]
            loss, _outputs = outcome_loss(
                network,
                rows,
                feature_dim=feature_dim,
                device=device,
                score_weight=args.score_loss_weight,
                own_win_weight=args.own_win_loss_weight,
                opponent_win_weight=args.opponent_win_loss_weight,
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, max_norm=5.0)
            optimizer.step()
            epoch_total += float(loss.detach().cpu())
            batches += 1
        validation_metrics = evaluate(
            network,
            validation,
            feature_dim=feature_dim,
            device=device,
            batch_size=args.batch_size,
            value_scale=args.value_scale,
            score_weight=args.score_loss_weight,
            own_win_weight=args.own_win_loss_weight,
            opponent_win_weight=args.opponent_win_loss_weight,
        )
        history.append(
            {
                "epoch": epoch,
                "train_total_loss": epoch_total / max(batches, 1),
                "validation": validation_metrics,
            }
        )
        if best_validation is None or validation_metrics["total_loss"] < best_validation["total_loss"]:
            best_epoch = epoch
            best_validation = validation_metrics
            best_state = copy.deepcopy(network.state_dict())
    if best_state is None or best_epoch is None or best_validation is None:
        raise RuntimeError("没有可选择的 afterstate checkpoint")
    network.load_state_dict(best_state)
    final_validation = evaluate(
        network,
        validation,
        feature_dim=feature_dim,
        device=device,
        batch_size=args.batch_size,
        value_scale=args.value_scale,
        score_weight=args.score_loss_weight,
        own_win_weight=args.own_win_loss_weight,
        opponent_win_weight=args.opponent_win_loss_weight,
    )
    test_metrics = (
        None
        if args.skip_test
        else evaluate(
            network,
            test,
            feature_dim=feature_dim,
            device=device,
            batch_size=args.batch_size,
            value_scale=args.value_scale,
            score_weight=args.score_loss_weight,
            own_win_weight=args.own_win_loss_weight,
            opponent_win_weight=args.opponent_win_loss_weight,
        )
    )
    report = {
        "model": "candidate_policy_value_afterstate_outcomes",
        "status": "diagnostic_only_not_authorized_for_action_selection",
        "initialization": initialization,
        "feature_version": args.feature_version,
        "feature_dim": feature_dim,
        "hidden_size": initial_agent.hidden_size,
        "device": str(device),
        "seed": args.seed,
        "value_scale": args.value_scale,
        "loss_weights": {
            "score": args.score_loss_weight,
            "own_win": args.own_win_loss_weight,
            "opponent_win": args.opponent_win_loss_weight,
        },
        "behavior_boundary": {
            "target": "only the logged executed_index action",
            "excluded_default_source": "random_legal_teacher_labeled",
            "reason": "its random continuation does not match deployment behavior",
            "policy_encoder_frozen": not args.unfreeze_encoder,
            "outcome_encoder_trainable": True,
            "known_propensity_required": args.require_known_propensity,
            "only_randomized_actions": args.only_randomized_actions,
            "decision_phase": args.decision_phase,
            "terminal_test_read": not args.skip_test,
        },
        "sources": sorted(sources),
        "inputs": {
            "train": [str(args.train), *(str(path) for path in args.additional_train)],
            "validation": [
                str(args.validation),
                *(str(path) for path in args.additional_validation),
            ],
            "test": (
                []
                if args.skip_test
                else [str(args.test), *(str(path) for path in args.additional_test)]
            ),
        },
        "counts": {
            "train": len(train),
            "validation": len(validation),
            "test": None if args.skip_test else len(test),
            "train_by_source": source_counts(train),
            "validation_by_source": source_counts(validation),
            "test_by_source": None if args.skip_test else source_counts(test),
            "train_propensity": propensity_summary(train),
            "validation_propensity": propensity_summary(validation),
            "test_propensity": None if args.skip_test else propensity_summary(test),
        },
        "checkpoint_selection": {
            "split": "validation",
            "metric": "lowest_weighted_outcome_loss",
            "selected_epoch": best_epoch,
            "selected_validation": best_validation,
        },
        "validation": final_validation,
        "test": test_metrics,
        "history": history,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    agent = TorchPolicyValueAgent(
        feature_version=args.feature_version,
        hidden_size=initial_agent.hidden_size,
        architecture=initial_agent.architecture,
        attention_heads=initial_agent.attention_heads,
        device=str(device),
        network=network,
    )
    agent.save(args.output_dir / "policy-value.pt", metadata=report)
    (args.output_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
