"""Paired, seat-rotated policy evaluation for Xiamen Mahjong.

The candidate always plays one seat against three copies of the baseline.  For
each random seed, it is rotated through all four seats.  This is deliberately
stricter than reporting a single game: the same initial shuffle is sampled for
every seat and the candidate's net hand score is aggregated across rotations.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any

from .agents import HeuristicTeacherAgent
from .game import XiamenMahjongGame
from .rules import XiamenRules


@dataclass(frozen=True)
class PolicyEvaluation:
    profile: str
    first_seed: int
    seed_count: int
    games: int
    candidate_wins: int
    draws: int
    candidate_score_total: int
    candidate_score_mean: float
    candidate_score_stderr: float
    candidate_win_rate: float
    candidate_scores: tuple[int, ...]

    def payload(self, *, include_scores: bool = False) -> dict[str, Any]:
        payload = asdict(self)
        if include_scores:
            payload["candidate_scores"] = list(self.candidate_scores)
        else:
            del payload["candidate_scores"]
        return payload


def paired_score_comparison(
    candidate: PolicyEvaluation, reference: PolicyEvaluation
) -> dict[str, float]:
    """Compare two policies on exactly the same rotated initial walls.

    The unit of uncertainty is one initial shuffle, not an individual seat:
    each shuffle contributes four correlated seat rotations.  Pairing the
    per-shuffle means removes much of the wall-strength variance and makes a
    small model difference interpretable.
    """

    if (
        candidate.profile != reference.profile
        or candidate.first_seed != reference.first_seed
        or candidate.seed_count != reference.seed_count
        or candidate.games != reference.games
    ):
        raise ValueError("配对评测需要相同规则、种子范围和座位轮换数量")
    rotations = candidate.games // candidate.seed_count
    if rotations <= 0 or candidate.games % candidate.seed_count:
        raise ValueError("配对评测的座位轮换数据无效")
    seed_deltas = [
        sum(
            candidate.candidate_scores[offset : offset + rotations]
        )
        / rotations
        - sum(reference.candidate_scores[offset : offset + rotations]) / rotations
        for offset in range(0, candidate.games, rotations)
    ]
    mean_delta = sum(seed_deltas) / len(seed_deltas)
    variance = (
        sum((value - mean_delta) ** 2 for value in seed_deltas)
        / (len(seed_deltas) - 1)
        if len(seed_deltas) > 1
        else 0.0
    )
    stderr = math.sqrt(variance / len(seed_deltas))
    return {
        "paired_seed_score_delta_mean": mean_delta,
        "paired_seed_score_delta_stderr": stderr,
        "paired_seed_score_delta_95pct_low": mean_delta - 1.96 * stderr,
        "paired_seed_score_delta_95pct_high": mean_delta + 1.96 * stderr,
    }


def evaluate_against_teacher(
    candidate: Any,
    *,
    hands: int = 100,
    profile: str = "classic",
    seed: int = 20260804,
) -> PolicyEvaluation:
    """Evaluate one candidate seat against three frozen Teacher seats.

    ``hands`` denotes independent initial shuffles.  The candidate plays all
    four seats for every shuffle.  The engine is authoritative for action
    legality, state transitions and scoring; the evaluator only aggregates
    outcomes.
    """

    if hands <= 0:
        raise ValueError("hands 必须为正数")
    rules = XiamenRules.from_profile(profile)
    baseline = HeuristicTeacherAgent()
    scores: list[int] = []
    wins = 0
    draws = 0
    for hand_offset in range(hands):
        hand_seed = seed + hand_offset
        for candidate_seat in range(rules.player_count):
            agents = {seat: baseline for seat in range(rules.player_count)}
            agents[candidate_seat] = candidate
            game = XiamenMahjongGame(
                seed=hand_seed,
                rules=rules,
                agents=agents,
                human_seat=-1,
                auto_advance=True,
            )
            if game.phase != "over":
                raise RuntimeError("自动评测没有完成一局")
            scores.append(game.players[candidate_seat].score)
            wins += game.winner == candidate_seat
            draws += game.win_type == "draw"
    count = len(scores)
    mean = sum(scores) / count
    # Four rotations generated from one initial wall are correlated. Treat one
    # seed's four-seat mean as one independent observation, otherwise the
    # standard error would overstate the certainty of a paired evaluation.
    seed_means = [
        sum(scores[index : index + rules.player_count]) / rules.player_count
        for index in range(0, count, rules.player_count)
    ]
    variance = (
        sum((value - mean) ** 2 for value in seed_means) / (len(seed_means) - 1)
        if len(seed_means) > 1
        else 0.0
    )
    return PolicyEvaluation(
        profile=rules.profile,
        first_seed=seed,
        seed_count=hands,
        games=count,
        candidate_wins=wins,
        draws=draws,
        candidate_score_total=sum(scores),
        candidate_score_mean=mean,
        candidate_score_stderr=math.sqrt(variance / len(seed_means)),
        candidate_win_rate=wins / count,
        candidate_scores=tuple(scores),
    )
