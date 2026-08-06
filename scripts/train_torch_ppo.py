#!/usr/bin/env python3
"""Small, legal-action PPO fine-tuning for policy-value Mahjong checkpoints.

This is deliberately a research-stage optimizer.  Each rollout has exactly
one sampled candidate seat against three frozen rule Teachers; only the engine
settled candidate net score is used as reward.  A produced checkpoint is never
promoted without a separate seat-rotated paired evaluation on unseen walls.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
from itertools import chain
import json
import math
from pathlib import Path
import random
import sys
from typing import Any, Sequence


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    import torch
    from torch import nn
    from torch.nn import functional as F
except ModuleNotFoundError as error:
    raise SystemExit("请使用 .venv/bin/python 运行；该脚本需要 PyTorch") from error

from xiamen_mahjong.agents import GameAction, HeuristicTeacherAgent
from xiamen_mahjong.game import XiamenMahjongGame
from xiamen_mahjong.rules import XiamenRules
from xiamen_mahjong.tiles import BASE_TILE_COUNT
from xiamen_mahjong.torch_policy import (
    ARCHITECTURE_CANDIDATE_MLP,
    CandidatePolicyValueNetwork,
    TorchPolicyValueAgent,
)
from xiamen_mahjong.training import (
    TeacherDecision,
    _decision,
    _dense_action_features,
    _turn_actions,
)


@dataclass(frozen=True)
class PpoStep:
    decision: TeacherDecision
    action_index: int
    old_log_probability: float
    old_value: float
    reward: float
    # Training-process-only oracle input.  It is never attached to a
    # TeacherDecision, trajectory, agent checkpoint or report payload.
    privileged_features: tuple[float, ...] | None = None


# Relative player order makes the critic insensitive to the arbitrary absolute
# seat assigned by the simulator.  The vector intentionally contains hidden
# hands and wall composition, so its scope is strictly in-memory PPO training.
PRIVILEGED_CRITIC_FEATURE_DIM = BASE_TILE_COUNT * 5 + 4 + 4 + 4 + 2 + (BASE_TILE_COUNT + 1) * 2


class PrivilegedCritic(nn.Module):
    """Training-only centralized critic; never included in a policy checkpoint."""

    def __init__(self, hidden_size: int = 128):
        super().__init__()
        if hidden_size <= 0:
            raise ValueError("privileged critic hidden-size 必须为正数")
        self.network = nn.Sequential(
            nn.Linear(PRIVILEGED_CRITIC_FEATURE_DIM, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.network(features).squeeze(-1)


def privileged_critic_features(
    game: XiamenMahjongGame, candidate_seat: int
) -> tuple[float, ...]:
    """Encode complete simulator state for a PPO baseline only.

    The actor never calls this function.  Keeping it in this training script,
    rather than the deployable policy module, makes the security boundary
    explicit: returned vectors may contain opponent hands and wall counts but
    exist only until the current PPO process finishes.
    """

    if game.rules.player_count != 4:
        raise ValueError("privileged critic 当前仅支持四人厦门麻将")
    if not 0 <= candidate_seat < game.rules.player_count:
        raise ValueError("candidate_seat 超出范围")
    features: list[float] = []
    for offset in range(game.rules.player_count):
        player = game.players[(candidate_seat + offset) % game.rules.player_count]
        counts = Counter(player.hand)
        features.extend(counts[tile] / 4.0 for tile in range(BASE_TILE_COUNT))
    wall_counts = Counter(game.wall)
    features.extend(wall_counts[tile] / 4.0 for tile in range(BASE_TILE_COUNT))
    features.extend(len(player.flowers) / 8.0 for player in game.players)
    features.extend(player.score / 80.0 for player in game.players)
    features.extend(
        1.0 if game.current_player == (candidate_seat + offset) % game.rules.player_count else 0.0
        for offset in range(game.rules.player_count)
    )
    features.extend((1.0 if game.phase == "discard" else 0.0, 1.0 if game.phase == "response" else 0.0))
    for tile in (game.last_discard, game.gold_tile):
        features.extend(
            1.0 if tile == index else 0.0 for index in range(BASE_TILE_COUNT + 1)
        )
    if len(features) != PRIVILEGED_CRITIC_FEATURE_DIM:
        raise RuntimeError("privileged critic 特征维度不匹配")
    return tuple(features)


@dataclass(frozen=True)
class RolloutSummary:
    episodes: int
    decisions: int
    wins: int
    draws: int
    reward_mean: float
    reward_stderr: float
    action_counts: dict[str, int]
    opponent_profile_counts: dict[str, int]

    def payload(self) -> dict[str, Any]:
        return {
            "episodes": self.episodes,
            "decisions": self.decisions,
            "wins": self.wins,
            "draws": self.draws,
            "reward_mean": self.reward_mean,
            "reward_stderr": self.reward_stderr,
            "action_counts": self.action_counts,
            "opponent_profile_counts": self.opponent_profile_counts,
        }


@dataclass
class _ActiveRollout:
    game: XiamenMahjongGame
    candidate_seat: int
    steps: list[tuple[TeacherDecision, int, float, float, tuple[float, ...] | None]]
    opponents: dict[int, tuple[str, Any]]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--profile", choices=("classic", "core"), default="classic")
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--episodes-per-iteration", type=int, default=256)
    parser.add_argument(
        "--rollout-batch-size",
        type=int,
        default=1,
        help="同时推进的独立牌局数；1 保持逐局采样，>1 合并候选网络推理",
    )
    parser.add_argument("--ppo-epochs", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=0.00005)
    parser.add_argument("--clip-ratio", type=float, default=0.15)
    parser.add_argument("--value-weight", type=float, default=0.25)
    parser.add_argument("--entropy-weight", type=float, default=0.002)
    parser.add_argument("--reward-scale", type=float, default=80.0)
    parser.add_argument(
        "--privileged-critic",
        action="store_true",
        help=(
            "仅训练期启用可见完整模拟状态的 centralized critic；"
            "actor、网页和保存的 policy checkpoint 仍只用公开信息"
        ),
    )
    parser.add_argument(
        "--privileged-critic-hidden-size", type=int, default=128
    )
    parser.add_argument(
        "--privileged-critic-weight",
        type=float,
        default=0.25,
        help="训练期 privileged critic 的回归损失权重",
    )
    parser.add_argument("--seed", type=int, default=20266004)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument(
        "--opponent-checkpoint",
        type=Path,
        action="append",
        default=[],
        help="可重复指定；作为冻结对手池的 .pt checkpoint",
    )
    parser.add_argument(
        "--teacher-opponent-probability",
        type=float,
        default=1.0,
        help="每个非候选座位使用 Teacher 的概率；其余从冻结对手池均匀抽取",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("artifacts/torch-ppo-classic")
    )
    return parser.parse_args()


def _softmax(logits: Sequence[float]) -> list[float]:
    maximum = max(logits)
    exponentials = [math.exp(value - maximum) for value in logits]
    total = sum(exponentials)
    return [value / total for value in exponentials]


def _sample_index(logits: Sequence[float], rng: random.Random) -> tuple[int, float]:
    probabilities = _softmax(logits)
    threshold = rng.random()
    cumulative = 0.0
    for index, probability in enumerate(probabilities):
        cumulative += probability
        if threshold <= cumulative:
            return index, math.log(max(probability, 1e-12))
    return len(probabilities) - 1, math.log(max(probabilities[-1], 1e-12))


def collect_rollouts(
    policy: TorchPolicyValueAgent,
    *,
    episodes: int,
    profile: str,
    seed: int,
    reward_scale: float,
    opponents: Sequence[tuple[str, Any]] = (),
    teacher_opponent_probability: float = 1.0,
    privileged_critic: PrivilegedCritic | None = None,
) -> tuple[list[PpoStep], RolloutSummary]:
    """Sample candidate actions and attach final reward and old baseline.

    When supplied, ``privileged_critic`` is evaluated only here, before an
    action is applied. Its private vector never enters ``decision`` or a
    returned summary; only the scalar old baseline is retained for PPO.
    """

    if episodes <= 0:
        raise ValueError("episodes 必须为正数")
    if not 0.0 <= teacher_opponent_probability <= 1.0:
        raise ValueError("teacher_opponent_probability 必须在 0 和 1 之间")
    if teacher_opponent_probability < 1.0 and not opponents:
        raise ValueError("混入非 Teacher 对手时必须提供 opponent checkpoint")
    rules = XiamenRules.from_profile(profile)
    teacher = HeuristicTeacherAgent()
    rng = random.Random(seed)
    all_steps: list[PpoStep] = []
    action_counts: Counter[str] = Counter()
    opponent_profile_counts: Counter[str] = Counter()
    rewards: list[float] = []
    wins = 0
    draws = 0
    for episode_index in range(episodes):
        hand_seed = seed + episode_index // rules.player_count
        candidate_seat = episode_index % rules.player_count
        game = XiamenMahjongGame(
            seed=hand_seed, rules=rules, auto_advance=False, human_seat=-1
        )
        opponent_by_seat: dict[int, tuple[str, Any]] = {}
        for seat in range(rules.player_count):
            if seat == candidate_seat:
                continue
            if not opponents or rng.random() < teacher_opponent_probability:
                opponent_by_seat[seat] = ("heuristic_teacher", teacher)
            else:
                opponent_by_seat[seat] = opponents[rng.randrange(len(opponents))]
            opponent_profile_counts[opponent_by_seat[seat][0]] += 1
        episode_steps: list[
            tuple[TeacherDecision, int, float, float, tuple[float, ...] | None]
        ] = []
        safety = 0
        while game.phase != "over":
            safety += 1
            if safety > 600:
                raise RuntimeError("PPO rollout 超过安全步数")
            if game.phase == "discard":
                player_id = game.current_player
                legal = tuple(_turn_actions(game, player_id))
                if player_id == candidate_seat:
                    decision = _decision(game, hand_seed, player_id, legal, legal[0])
                    logits, value = policy.policy_value(decision)
                    oracle_features = (
                        privileged_critic_features(game, candidate_seat)
                        if privileged_critic is not None
                        else None
                    )
                    if oracle_features is not None:
                        privileged_critic.eval()
                        with torch.no_grad():
                            old_baseline = float(
                                privileged_critic(
                                    torch.tensor(
                                        [oracle_features],
                                        dtype=torch.float32,
                                        device=policy.device,
                                    )
                                ).item()
                            )
                    else:
                        old_baseline = value
                    action_index, old_log_probability = _sample_index(logits, rng)
                    action = legal[action_index]
                    episode_steps.append(
                        (
                            decision,
                            action_index,
                            old_log_probability,
                            old_baseline,
                            oracle_features,
                        )
                    )
                    action_counts[action.kind] += 1
                else:
                    action = opponent_by_seat[player_id][1].choose_turn_action(
                        game, player_id
                    )
                if action not in legal:
                    raise RuntimeError("PPO 策略选择了规则引擎未提供的摸牌后动作")
                game._apply_turn_action(player_id, action)
                continue
            if game.phase == "response":
                for player_id, options in sorted(game.response_options.items()):
                    legal = tuple(options)
                    if player_id == candidate_seat:
                        decision = _decision(game, hand_seed, player_id, legal, legal[0])
                        logits, value = policy.policy_value(decision)
                        oracle_features = (
                            privileged_critic_features(game, candidate_seat)
                            if privileged_critic is not None
                            else None
                        )
                        if oracle_features is not None:
                            privileged_critic.eval()
                            with torch.no_grad():
                                old_baseline = float(
                                    privileged_critic(
                                        torch.tensor(
                                            [oracle_features],
                                            dtype=torch.float32,
                                            device=policy.device,
                                        )
                                    ).item()
                                )
                        else:
                            old_baseline = value
                        action_index, old_log_probability = _sample_index(logits, rng)
                        action = legal[action_index]
                        episode_steps.append(
                            (
                                decision,
                                action_index,
                                old_log_probability,
                                old_baseline,
                                oracle_features,
                            )
                        )
                        action_counts[action.kind] += 1
                    else:
                        action = opponent_by_seat[player_id][1].choose_response(
                            game, player_id, list(legal)
                        )
                    if action not in legal:
                        raise RuntimeError("PPO 策略选择了规则引擎未提供的响应动作")
                    game.response_choices[player_id] = action
                game._resolve_responses()
                continue
            raise RuntimeError(f"未知 PPO rollout 阶段：{game.phase}")
        reward = game.players[candidate_seat].score / reward_scale
        rewards.append(reward)
        all_steps.extend(
            PpoStep(
                decision,
                action_index,
                old_log_probability,
                old_value,
                reward,
                oracle_features,
            )
            for decision, action_index, old_log_probability, old_value, oracle_features
            in episode_steps
        )
        if game.win_type == "draw":
            draws += 1
        else:
            wins += 1
    mean = sum(rewards) / len(rewards)
    variance = (
        sum((reward - mean) ** 2 for reward in rewards) / (len(rewards) - 1)
        if len(rewards) > 1
        else 0.0
    )
    return all_steps, RolloutSummary(
        episodes=episodes,
        decisions=len(all_steps),
        wins=wins,
        draws=draws,
        reward_mean=mean,
        reward_stderr=math.sqrt(variance / len(rewards)),
        action_counts=dict(sorted(action_counts.items())),
        opponent_profile_counts=dict(sorted(opponent_profile_counts.items())),
    )


def collect_rollouts_batched(
    policy: TorchPolicyValueAgent,
    *,
    episodes: int,
    profile: str,
    seed: int,
    reward_scale: float,
    opponents: Sequence[tuple[str, Any]] = (),
    teacher_opponent_probability: float = 1.0,
    rollout_batch_size: int = 1,
    privileged_critic: PrivilegedCritic | None = None,
) -> tuple[list[PpoStep], RolloutSummary]:
    """Collect legal PPO data while batching only independent network calls."""

    if rollout_batch_size <= 0:
        raise ValueError("rollout_batch_size 必须为正数")
    if rollout_batch_size == 1:
        return collect_rollouts(
            policy,
            episodes=episodes,
            profile=profile,
            seed=seed,
            reward_scale=reward_scale,
            opponents=opponents,
            teacher_opponent_probability=teacher_opponent_probability,
            privileged_critic=privileged_critic,
        )
    if episodes <= 0:
        raise ValueError("episodes 必须为正数")
    if not 0.0 <= teacher_opponent_probability <= 1.0:
        raise ValueError("teacher_opponent_probability 必须在 0 和 1 之间")
    if teacher_opponent_probability < 1.0 and not opponents:
        raise ValueError("混入非 Teacher 对手时必须提供 opponent checkpoint")
    rules = XiamenRules.from_profile(profile)
    teacher = HeuristicTeacherAgent()
    rng = random.Random(seed)
    all_steps: list[PpoStep] = []
    action_counts: Counter[str] = Counter()
    opponent_profile_counts: Counter[str] = Counter()
    rewards: list[float] = []
    wins = 0
    draws = 0

    def build_episode(episode_index: int) -> _ActiveRollout:
        hand_seed = seed + episode_index // rules.player_count
        candidate_seat = episode_index % rules.player_count
        game = XiamenMahjongGame(
            seed=hand_seed, rules=rules, auto_advance=False, human_seat=-1
        )
        episode_opponents: dict[int, tuple[str, Any]] = {}
        for seat in range(rules.player_count):
            if seat == candidate_seat:
                continue
            if not opponents or rng.random() < teacher_opponent_probability:
                episode_opponents[seat] = ("heuristic_teacher", teacher)
            else:
                episode_opponents[seat] = opponents[rng.randrange(len(opponents))]
            opponent_profile_counts[episode_opponents[seat][0]] += 1
        return _ActiveRollout(game, candidate_seat, [], episode_opponents)

    def finish(active: _ActiveRollout) -> None:
        nonlocal wins, draws
        reward = active.game.players[active.candidate_seat].score / reward_scale
        rewards.append(reward)
        all_steps.extend(
            PpoStep(
                decision,
                action_index,
                old_log_probability,
                old_value,
                reward,
                oracle_features,
            )
            for decision, action_index, old_log_probability, old_value, oracle_features
            in active.steps
        )
        if active.game.win_type == "draw":
            draws += 1
        else:
            wins += 1

    for start in range(0, episodes, rollout_batch_size):
        active = [
            build_episode(episode_index)
            for episode_index in range(start, min(start + rollout_batch_size, episodes))
        ]
        safety = 0
        while active:
            safety += 1
            if safety > 600:
                raise RuntimeError("批量 PPO rollout 超过安全步数")
            turn_jobs: list[tuple[_ActiveRollout, TeacherDecision, tuple[GameAction, ...]]] = []
            response_jobs: list[
                tuple[_ActiveRollout, int, TeacherDecision, tuple[GameAction, ...]]
            ] = []
            response_groups: list[tuple[_ActiveRollout, dict[int, GameAction]]] = []
            opponent_policy_jobs: dict[
                int,
                tuple[
                    TorchPolicyValueAgent,
                    list[
                        tuple[
                            _ActiveRollout,
                            int,
                            TeacherDecision,
                            tuple[GameAction, ...],
                            bool,
                        ]
                    ],
                ],
            ] = {}
            for episode in active:
                game = episode.game
                if game.phase == "discard":
                    player_id = game.current_player
                    legal = tuple(_turn_actions(game, player_id))
                    if player_id == episode.candidate_seat:
                        decision = _decision(game, game.seed or 0, player_id, legal, legal[0])
                        turn_jobs.append((episode, decision, legal))
                    else:
                        opponent = episode.opponents[player_id][1]
                        if isinstance(opponent, TorchPolicyValueAgent):
                            decision = _decision(
                                game, game.seed or 0, player_id, legal, legal[0]
                            )
                            bucket = opponent_policy_jobs.setdefault(
                                id(opponent), (opponent, [])
                            )[1]
                            bucket.append((episode, player_id, decision, legal, False))
                        else:
                            action = opponent.choose_turn_action(game, player_id)
                            if action not in legal:
                                raise RuntimeError("冻结对手选择了规则引擎未提供的摸牌后动作")
                            game._apply_turn_action(player_id, action)
                    continue
                if game.phase == "response":
                    choices: dict[int, GameAction] = {}
                    for player_id, options in sorted(game.response_options.items()):
                        legal = tuple(options)
                        if player_id == episode.candidate_seat:
                            decision = _decision(
                                game, game.seed or 0, player_id, legal, legal[0]
                            )
                            response_jobs.append((episode, player_id, decision, legal))
                        else:
                            opponent = episode.opponents[player_id][1]
                            if isinstance(opponent, TorchPolicyValueAgent):
                                decision = _decision(
                                    game, game.seed or 0, player_id, legal, legal[0]
                                )
                                bucket = opponent_policy_jobs.setdefault(
                                    id(opponent), (opponent, [])
                                )[1]
                                bucket.append(
                                    (episode, player_id, decision, legal, True)
                                )
                            else:
                                action = opponent.choose_response(
                                    game, player_id, list(legal)
                                )
                                if action not in legal:
                                    raise RuntimeError("冻结对手选择了规则引擎未提供的响应动作")
                                choices[player_id] = action
                    response_groups.append((episode, choices))
                    continue
                if game.phase != "over":
                    raise RuntimeError(f"未知批量 PPO rollout 阶段：{game.phase}")

            candidate_jobs = [
                (episode, decision, legal, None)
                for episode, decision, legal in turn_jobs
            ] + [
                (episode, decision, legal, player_id)
                for episode, player_id, decision, legal in response_jobs
            ]
            predictions = (
                policy.policy_values_batch(
                    [decision for _episode, decision, _legal, _player in candidate_jobs]
                )
                if candidate_jobs
                else []
            )
            oracle_feature_batch: list[tuple[float, ...] | None] = [
                (
                    privileged_critic_features(episode.game, episode.candidate_seat)
                    if privileged_critic is not None
                    else None
                )
                for episode, _decision_snapshot, _legal, _response_player in candidate_jobs
            ]
            if privileged_critic is not None and oracle_feature_batch:
                privileged_critic.eval()
                with torch.no_grad():
                    old_baselines = privileged_critic(
                        torch.tensor(
                            [
                                features
                                for features in oracle_feature_batch
                                if features is not None
                            ],
                            dtype=torch.float32,
                            device=policy.device,
                        )
                    ).detach().cpu().tolist()
            else:
                old_baselines = [value for _logits, value in predictions]
            for (
                (episode, decision, legal, response_player),
                (logits, _public_value),
                oracle_features,
                old_baseline,
            ) in zip(candidate_jobs, predictions, oracle_feature_batch, old_baselines):
                action_index, old_log_probability = _sample_index(logits, rng)
                action = legal[action_index]
                episode.steps.append(
                    (
                        decision,
                        action_index,
                        old_log_probability,
                        float(old_baseline),
                        oracle_features,
                    )
                )
                action_counts[action.kind] += 1
                if response_player is None:
                    episode.game._apply_turn_action(
                        episode.candidate_seat, action
                    )
                else:
                    episode.game.response_choices[response_player] = action
            for frozen_opponent, jobs in opponent_policy_jobs.values():
                predictions = frozen_opponent.policy_values_batch(
                    [decision for _episode, _player, decision, _legal, _response in jobs]
                )
                for (
                    episode,
                    player_id,
                    _decision_snapshot,
                    legal,
                    is_response,
                ), (logits, _value) in zip(jobs, predictions):
                    action_index = max(
                        range(len(logits)), key=lambda index: (logits[index], -index)
                    )
                    action = legal[action_index]
                    if is_response:
                        episode.game.response_choices[player_id] = action
                    else:
                        episode.game._apply_turn_action(player_id, action)
            for episode, choices in response_groups:
                episode.game.response_choices.update(choices)
                episode.game._resolve_responses()
            completed = [episode for episode in active if episode.game.phase == "over"]
            for episode in completed:
                finish(episode)
            active = [episode for episode in active if episode.game.phase != "over"]

    mean = sum(rewards) / len(rewards)
    variance = (
        sum((reward - mean) ** 2 for reward in rewards) / (len(rewards) - 1)
        if len(rewards) > 1
        else 0.0
    )
    return all_steps, RolloutSummary(
        episodes=episodes,
        decisions=len(all_steps),
        wins=wins,
        draws=draws,
        reward_mean=mean,
        reward_stderr=math.sqrt(variance / len(rewards)),
        action_counts=dict(sorted(action_counts.items())),
        opponent_profile_counts=dict(sorted(opponent_profile_counts.items())),
    )


def tensors(
    steps: Sequence[PpoStep],
    *,
    feature_dim: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    max_actions = max(len(step.decision.legal_actions) for step in steps)
    candidates = torch.zeros(
        (len(steps), max_actions, feature_dim), dtype=torch.float32, device=device
    )
    action_mask = torch.zeros((len(steps), max_actions), dtype=torch.bool, device=device)
    actions = torch.empty(len(steps), dtype=torch.long, device=device)
    old_log_probabilities = torch.empty(len(steps), dtype=torch.float32, device=device)
    old_values = torch.empty(len(steps), dtype=torch.float32, device=device)
    rewards = torch.empty(len(steps), dtype=torch.float32, device=device)
    for row, step in enumerate(steps):
        vectors = [
            _dense_action_features(
                step.decision.state, action, feature_version=3
            )
            for action in step.decision.legal_actions
        ]
        candidates[row, : len(vectors)] = torch.tensor(
            vectors, dtype=torch.float32, device=device
        )
        action_mask[row, : len(vectors)] = True
        actions[row] = step.action_index
        old_log_probabilities[row] = step.old_log_probability
        old_values[row] = step.old_value
        rewards[row] = step.reward
    return candidates, action_mask, actions, old_log_probabilities, old_values, rewards


def ppo_update(
    network: CandidatePolicyValueNetwork,
    steps: Sequence[PpoStep],
    *,
    device: torch.device,
    batch_size: int,
    epochs: int,
    learning_rate: float,
    clip_ratio: float,
    value_weight: float,
    entropy_weight: float,
    seed: int,
    privileged_critic: PrivilegedCritic | None = None,
    privileged_critic_weight: float = 0.25,
) -> dict[str, float]:
    if not steps:
        raise ValueError("PPO rollout 没有候选策略决策")
    candidates, action_mask, actions, old_log_probabilities, old_values, rewards = tensors(
        steps, feature_dim=network.feature_dim, device=device
    )
    if privileged_critic_weight < 0:
        raise ValueError("privileged critic weight 不能为负数")
    privileged_inputs: torch.Tensor | None = None
    if privileged_critic is not None:
        if any(step.privileged_features is None for step in steps):
            raise ValueError("启用 privileged critic 的 PPO step 缺少内存特征")
        privileged_inputs = torch.tensor(
            [step.privileged_features for step in steps],
            dtype=torch.float32,
            device=device,
        )
        if privileged_inputs.shape != (len(steps), PRIVILEGED_CRITIC_FEATURE_DIM):
            raise ValueError("privileged critic PPO 特征维度不匹配")
    advantages = rewards - old_values
    advantages = (advantages - advantages.mean()) / advantages.std(unbiased=False).clamp_min(1e-6)
    optimizer = torch.optim.AdamW(
        (
            chain(network.parameters(), privileged_critic.parameters())
            if privileged_critic is not None
            else network.parameters()
        ),
        lr=learning_rate,
        weight_decay=0.0001,
    )
    indices = list(range(len(steps)))
    total_policy_loss = 0.0
    total_value_loss = 0.0
    total_privileged_critic_loss = 0.0
    total_entropy = 0.0
    updates = 0
    network.train()
    if privileged_critic is not None:
        privileged_critic.train()
    for epoch in range(epochs):
        random.Random(seed + epoch).shuffle(indices)
        for start in range(0, len(indices), batch_size):
            row_indices = torch.tensor(indices[start : start + batch_size], device=device)
            logits, values = network(candidates[row_indices], action_mask[row_indices])
            log_probabilities = F.log_softmax(logits, dim=1)
            chosen_log_probabilities = log_probabilities.gather(
                1, actions[row_indices].unsqueeze(1)
            ).squeeze(1)
            ratio = torch.exp(chosen_log_probabilities - old_log_probabilities[row_indices])
            surrogate_one = ratio * advantages[row_indices]
            surrogate_two = torch.clamp(
                ratio, 1.0 - clip_ratio, 1.0 + clip_ratio
            ) * advantages[row_indices]
            policy_loss = -torch.minimum(surrogate_one, surrogate_two).mean()
            value_loss = F.smooth_l1_loss(values, rewards[row_indices])
            privileged_critic_loss = (
                F.smooth_l1_loss(
                    privileged_critic(privileged_inputs[row_indices]),
                    rewards[row_indices],
                )
                if privileged_critic is not None and privileged_inputs is not None
                else torch.zeros((), device=device)
            )
            probabilities = torch.softmax(logits, dim=1)
            entropy = -(probabilities * log_probabilities).sum(dim=1).mean()
            loss = (
                policy_loss
                + value_weight * value_loss
                + privileged_critic_weight * privileged_critic_loss
                - entropy_weight * entropy
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(network.parameters(), max_norm=5.0)
            optimizer.step()
            total_policy_loss += float(policy_loss.detach().cpu())
            total_value_loss += float(value_loss.detach().cpu())
            total_privileged_critic_loss += float(privileged_critic_loss.detach().cpu())
            total_entropy += float(entropy.detach().cpu())
            updates += 1
    network.eval()
    if privileged_critic is not None:
        privileged_critic.eval()
    metrics = {
        "updates": float(updates),
        "policy_loss": total_policy_loss / updates,
        "value_loss": total_value_loss / updates,
        "entropy": total_entropy / updates,
        "advantage_mean_before_normalization": float((rewards - old_values).mean().cpu()),
        "advantage_std_before_normalization": float(
            (rewards - old_values).std(unbiased=False).cpu()
        ),
    }
    if privileged_critic is not None:
        metrics["privileged_critic_loss"] = total_privileged_critic_loss / updates
    return metrics


def main() -> None:
    args = parse_args()
    if (
        args.iterations <= 0
        or args.episodes_per_iteration <= 0
        or args.rollout_batch_size <= 0
        or args.ppo_epochs <= 0
        or args.batch_size <= 0
        or args.learning_rate <= 0
        or not 0 < args.clip_ratio < 1
        or args.value_weight <= 0
        or args.entropy_weight < 0
        or args.reward_scale <= 0
        or args.privileged_critic_hidden_size <= 0
        or args.privileged_critic_weight < 0
        or not 0.0 <= args.teacher_opponent_probability <= 1.0
    ):
        raise ValueError("PPO 超参数不合法")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("请求 CUDA 但当前 PyTorch 无可用 GPU")
    agent = TorchPolicyValueAgent.load(args.checkpoint, device=args.device)
    if agent.architecture != ARCHITECTURE_CANDIDATE_MLP:
        raise ValueError("首版 PPO 仅支持 candidate_mlp checkpoint")
    assert isinstance(agent.network, CandidatePolicyValueNetwork)
    privileged_critic = None
    if args.privileged_critic:
        # This object intentionally has no save path.  It survives only for
        # this trainer process and supplies rollout baselines, never actions.
        torch.manual_seed(args.seed)
        if agent.device.type == "cuda":
            torch.cuda.manual_seed_all(args.seed)
        privileged_critic = PrivilegedCritic(
            args.privileged_critic_hidden_size
        ).to(agent.device)
    frozen_opponents: list[tuple[str, TorchPolicyValueAgent]] = []
    for checkpoint in args.opponent_checkpoint:
        opponent = TorchPolicyValueAgent.load(checkpoint, device=args.device)
        if opponent.architecture != ARCHITECTURE_CANDIDATE_MLP:
            raise ValueError("首版 PPO 对手池仅支持 candidate_mlp checkpoint")
        frozen_opponents.append((str(checkpoint), opponent))
    if args.teacher_opponent_probability < 1.0 and not frozen_opponents:
        raise ValueError("teacher-opponent-probability 小于 1 时需要 --opponent-checkpoint")
    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    report: dict[str, Any] = {
        "algorithm": "legal_action_ppo_terminal_score_v1",
        "profile": args.profile,
        "checkpoint_source": str(args.checkpoint),
        "device": args.device,
        "reward_scale": args.reward_scale,
        "rollout_batch_size": args.rollout_batch_size,
        "privileged_critic": {
            "enabled": args.privileged_critic,
            "training_only": True,
            "feature_scope": "complete_simulator_state_in_memory_only",
            "hidden_size": (
                args.privileged_critic_hidden_size if args.privileged_critic else None
            ),
            "loss_weight": (
                args.privileged_critic_weight if args.privileged_critic else None
            ),
            "serialized_with_actor_checkpoint": False,
        },
        "opponent_pool": {
            "teacher_probability": args.teacher_opponent_probability,
            "frozen_checkpoints": [str(path) for path in args.opponent_checkpoint],
        },
        "iterations": [],
    }
    for iteration in range(1, args.iterations + 1):
        steps, rollout = collect_rollouts_batched(
            agent,
            episodes=args.episodes_per_iteration,
            profile=args.profile,
            seed=args.seed + iteration * 100_000,
            reward_scale=args.reward_scale,
            opponents=frozen_opponents,
            teacher_opponent_probability=args.teacher_opponent_probability,
            rollout_batch_size=args.rollout_batch_size,
            privileged_critic=privileged_critic,
        )
        update = ppo_update(
            agent.network,
            steps,
            device=agent.device,
            batch_size=args.batch_size,
            epochs=args.ppo_epochs,
            learning_rate=args.learning_rate,
            clip_ratio=args.clip_ratio,
            value_weight=args.value_weight,
            entropy_weight=args.entropy_weight,
            seed=args.seed + iteration,
            privileged_critic=privileged_critic,
            privileged_critic_weight=args.privileged_critic_weight,
        )
        iteration_report = {
            "iteration": iteration,
            "rollout": rollout.payload(),
            "update": update,
        }
        report["iterations"].append(iteration_report)
        agent.save(
            output_dir / f"policy-value-ppo-iteration-{iteration}.pt",
            metadata={**report, "latest_iteration": iteration},
        )
    report["final_checkpoint"] = str(
        output_dir / f"policy-value-ppo-iteration-{args.iterations}.pt"
    )
    (output_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
