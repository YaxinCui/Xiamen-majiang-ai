"""Conservative grouped off-policy estimates for one logged intervention.

These helpers deliberately cover only a contextual-bandit intervention: one
recorded action is replaced and the frozen behavior policy controls the rest
of the hand.  They are not estimates for an arbitrary policy that changes
multiple future decisions in a four-player Mahjong game.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import math
from typing import Iterable, Sequence


@dataclass(frozen=True)
class LoggedIntervention:
    """One action with known behavior support and an optional direct model.

    ``propensities`` contains the behavior probability for every legal action
    in the exact order used by the engine.  ``direct_values`` is in the same
    reward unit as ``reward`` and must come from a model trained without this
    held-out record when it is used for doubly robust estimation.
    """

    group_id: str
    logged_index: int
    propensities: tuple[float, ...]
    reward: float
    baseline_index: int
    target_index: int
    direct_values: tuple[float, ...] | None = None

    def __post_init__(self) -> None:
        count = len(self.propensities)
        if not self.group_id:
            raise ValueError("group_id 不能为空")
        if count <= 0:
            raise ValueError("至少需要一个合法动作")
        if not all(math.isfinite(value) and value > 0.0 for value in self.propensities):
            raise ValueError("所有行为 propensity 必须有限且为正")
        if not math.isclose(sum(self.propensities), 1.0, abs_tol=1e-8):
            raise ValueError("行为 propensity 必须归一化")
        if not math.isfinite(self.reward):
            raise ValueError("终局回报必须有限")
        if not all(
            0 <= index < count
            for index in (self.logged_index, self.baseline_index, self.target_index)
        ):
            raise ValueError("动作索引不属于该信息集")
        if self.direct_values is not None and (
            len(self.direct_values) != count
            or not all(math.isfinite(value) for value in self.direct_values)
        ):
            raise ValueError("direct_values 必须与合法动作等长且有限")


def ips_delta(observation: LoggedIntervention) -> float:
    """IPS estimate of target minus baseline one-step-intervention return."""

    target_term = (
        observation.reward / observation.propensities[observation.target_index]
        if observation.logged_index == observation.target_index
        else 0.0
    )
    baseline_term = (
        observation.reward / observation.propensities[observation.baseline_index]
        if observation.logged_index == observation.baseline_index
        else 0.0
    )
    return target_term - baseline_term


def doubly_robust_delta(observation: LoggedIntervention) -> float:
    """DR estimate of target minus baseline, using a held-out direct model."""

    return doubly_robust_action_advantages(observation)[observation.target_index]


def doubly_robust_action_advantages(
    observation: LoggedIntervention,
) -> tuple[float, ...]:
    """Return a DR pseudo-advantage for every legal action versus Teacher.

    For action ``a`` and frozen baseline ``b`` this is the contextual-bandit
    pseudo-outcome

    ``q(a)-q(b) + 1[A=a](R-q(A))/p(a) - 1[A=b](R-q(A))/p(b)``.

    Its expectation is the one-intervention return difference whenever the
    logged propensity is correct; a separate, wall-group-disjoint direct model
    supplies ``q``.  The vector is useful for a *future* relative-advantage
    learner because every legal action is centered on the deployed Teacher,
    rather than learning an uncentered terminal score.  It is not a complete
    Mahjong value target: the logged continuation must still be Teacher after
    the single intervention, and high-variance pseudo-outcomes must not be
    used for action selection without new held-out OPE.
    """

    if observation.direct_values is None:
        raise ValueError("doubly robust 估计需要 held-out direct_values")
    values = observation.direct_values
    logged_residual = observation.reward - values[observation.logged_index]
    baseline = observation.baseline_index
    baseline_residual = (
        logged_residual / observation.propensities[baseline]
        if observation.logged_index == baseline
        else 0.0
    )
    advantages: list[float] = []
    for action_index, value in enumerate(values):
        action_residual = (
            logged_residual / observation.propensities[action_index]
            if observation.logged_index == action_index
            else 0.0
        )
        advantages.append(value - values[baseline] + action_residual - baseline_residual)
    return tuple(advantages)


def grouped_mean_stderr(
    values: Iterable[tuple[str, float]],
) -> dict[str, float | int]:
    """Aggregate correlated seat rotations by their physical-wall group."""

    by_group: dict[str, list[float]] = defaultdict(list)
    for group_id, value in values:
        if not group_id or not math.isfinite(value):
            raise ValueError("grouped estimate 必须具有非空 group 与有限值")
        by_group[group_id].append(value)
    if not by_group:
        raise ValueError("至少需要一个 grouped estimate")
    group_means = [sum(items) / len(items) for items in by_group.values()]
    mean = sum(group_means) / len(group_means)
    variance = (
        sum((item - mean) ** 2 for item in group_means) / (len(group_means) - 1)
        if len(group_means) > 1
        else 0.0
    )
    stderr = math.sqrt(variance / len(group_means))
    return {
        "groups": len(group_means),
        "observations": sum(len(items) for items in by_group.values()),
        "mean": mean,
        "stderr": stderr,
        "95pct_low": mean - 1.96 * stderr,
        "95pct_high": mean + 1.96 * stderr,
    }


def effective_sample_size(weights: Sequence[float]) -> float:
    """Return the standard importance-weight ESS without normalizing weights."""

    if not weights or not all(math.isfinite(weight) and weight >= 0 for weight in weights):
        raise ValueError("importance weights 必须为非空有限非负数")
    squared = sum(weight * weight for weight in weights)
    return sum(weights) ** 2 / squared if squared else 0.0


def intervention_estimates(
    observations: Iterable[LoggedIntervention],
) -> dict[str, object]:
    """Return grouped IPS and optional DR deltas plus support diagnostics."""

    rows = list(observations)
    if not rows:
        raise ValueError("没有可用于 OPE 的单点干预记录")
    ips = grouped_mean_stderr((row.group_id, ips_delta(row)) for row in rows)
    target_weights = [
        1.0 / row.propensities[row.target_index]
        if row.logged_index == row.target_index
        else 0.0
        for row in rows
    ]
    baseline_weights = [
        1.0 / row.propensities[row.baseline_index]
        if row.logged_index == row.baseline_index
        else 0.0
        for row in rows
    ]
    result: dict[str, object] = {
        "ips": ips,
        "support": {
            "target_matched_logged_action_count": sum(weight > 0 for weight in target_weights),
            "baseline_matched_logged_action_count": sum(weight > 0 for weight in baseline_weights),
            "target_effective_sample_size": effective_sample_size(target_weights),
            "baseline_effective_sample_size": effective_sample_size(baseline_weights),
            "target_override_rate": sum(
                row.target_index != row.baseline_index for row in rows
            )
            / len(rows),
            "minimum_target_propensity": min(
                row.propensities[row.target_index] for row in rows
            ),
            "minimum_baseline_propensity": min(
                row.propensities[row.baseline_index] for row in rows
            ),
        },
    }
    if all(row.direct_values is not None for row in rows):
        result["doubly_robust"] = grouped_mean_stderr(
            (row.group_id, doubly_robust_delta(row)) for row in rows
        )
    return result
