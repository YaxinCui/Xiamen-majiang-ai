"""Auditable sequential particle beliefs for imperfect-information training.

This module is deliberately game-agnostic.  It owns only opaque particles,
their normalized weights, and numerical diagnostics.  A Mahjong-specific
caller must supply a transition that replays one *public* observation and
returns a behavior likelihood for that observation under each private world.

Particles are runtime-only objects: they may contain simulated concealed
hands or wall states, but callers must never serialize them into a training
record.  Export only :class:`ParticleBeliefDiagnostics` aggregates such as
ESS, entropy, and particle counts.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import random
from typing import Callable, Generic, Sequence, TypeVar


ParticleT = TypeVar("ParticleT")
ObservationT = TypeVar("ObservationT")
ParticleUpdate = Callable[
    [ParticleT, ObservationT, random.Random], tuple[ParticleT | None, float]
]


@dataclass(frozen=True)
class ParticleBeliefDiagnostics:
    """Safe aggregate health information for one sequential belief state.

    ``effective_sample_size`` and ``weight_entropy`` describe the posterior
    *before* an optional resample.  That preserves the evidence of particle
    degeneracy instead of hiding it behind post-resample uniform weights.
    """

    observation_count: int
    particle_count: int
    effective_sample_size: float
    weight_entropy: float
    log_evidence: float
    resample_count: int
    zero_likelihood_particles: int
    proposal_failures: int

    @property
    def effective_sample_fraction(self) -> float:
        return self.effective_sample_size / self.particle_count

    def payload(self) -> dict[str, float | int]:
        """Return aggregates suitable for safe collector metadata."""

        return {
            "observation_count": self.observation_count,
            "particle_count": self.particle_count,
            "effective_sample_size": self.effective_sample_size,
            "effective_sample_fraction": self.effective_sample_fraction,
            "weight_entropy": self.weight_entropy,
            "log_evidence": self.log_evidence,
            "resample_count": self.resample_count,
            "zero_likelihood_particles": self.zero_likelihood_particles,
            "proposal_failures": self.proposal_failures,
        }


class SequentialParticleBelief(Generic[ParticleT, ObservationT]):
    """Bayesian filtering over private worlds conditioned on public history.

    ``observe`` accepts an immutable-style ``update`` callback.  For each
    particle it should replay exactly one public event and return a replacement
    particle plus ``P(event | particle, prior public history)``.  Returning
    ``(None, 0.0)`` rejects an inconsistent particle.  The callback must not
    mutate the original particle, because implementations often retain game
    clones for continuation rollout.

    Resampling is systematic and uses a private RNG.  It is an internal Monte
    Carlo operation only; its seed and particles must not appear in exported
    training JSONL.
    """

    def __init__(
        self,
        particles: Sequence[ParticleT],
        *,
        seed: int = 20260806,
        resample_ess_fraction: float = 0.5,
    ):
        if not particles:
            raise ValueError("粒子 belief 至少需要一个粒子")
        if not 0.0 <= resample_ess_fraction <= 1.0:
            raise ValueError("resample_ess_fraction 必须在 0 到 1 之间")
        self._particles = list(particles)
        self._weights = [1.0 / len(self._particles)] * len(self._particles)
        self._rng = random.Random(seed)
        self._resample_ess_fraction = float(resample_ess_fraction)
        self._observation_count = 0
        self._latest_ess = float(len(self._particles))
        self._latest_entropy = math.log(len(self._particles))
        self._log_evidence = 0.0
        self._resample_count = 0
        self._zero_likelihood_particles = 0
        self._proposal_failures = 0

    @property
    def particle_count(self) -> int:
        return len(self._particles)

    @property
    def particles(self) -> tuple[ParticleT, ...]:
        """Return runtime particles; never pass this value to an exporter."""

        return tuple(self._particles)

    @property
    def weights(self) -> tuple[float, ...]:
        """Return normalized posterior weights aligned with :attr:`particles`."""

        return tuple(self._weights)

    @property
    def diagnostics(self) -> ParticleBeliefDiagnostics:
        return ParticleBeliefDiagnostics(
            observation_count=self._observation_count,
            particle_count=self.particle_count,
            effective_sample_size=self._latest_ess,
            weight_entropy=self._latest_entropy,
            log_evidence=self._log_evidence,
            resample_count=self._resample_count,
            zero_likelihood_particles=self._zero_likelihood_particles,
            proposal_failures=self._proposal_failures,
        )

    def expectation(self, statistic: Callable[[ParticleT], float]) -> float:
        """Return a posterior expectation without exposing hidden state."""

        return sum(
            weight * float(statistic(particle))
            for particle, weight in zip(self._particles, self._weights)
        )

    def observe(
        self,
        observation: ObservationT,
        update: ParticleUpdate[ParticleT, ObservationT],
    ) -> ParticleBeliefDiagnostics:
        """Condition the belief on one replayed public observation.

        The update is transactional: a zero-total likelihood or an invalid
        callback result raises without replacing the current posterior.
        """

        candidates: list[ParticleT] = []
        unnormalized: list[float] = []
        zero_likelihoods = 0
        proposal_failures = 0
        for particle, old_weight in zip(self._particles, self._weights):
            updated_particle, likelihood = update(particle, observation, self._rng)
            if not math.isfinite(likelihood) or likelihood < 0.0:
                raise ValueError("粒子更新返回了无效的行为似然")
            if updated_particle is None:
                if likelihood != 0.0:
                    raise ValueError("被拒绝粒子的行为似然必须为零")
                proposal_failures += 1
                zero_likelihoods += 1
                # Keep an opaque placeholder only long enough to preserve
                # index alignment while calculating normalized weights.
                candidates.append(particle)
                unnormalized.append(0.0)
                continue
            if likelihood == 0.0:
                zero_likelihoods += 1
            candidates.append(updated_particle)
            unnormalized.append(old_weight * likelihood)

        evidence = sum(unnormalized)
        if not math.isfinite(evidence) or evidence <= 0.0:
            raise ValueError("公开观察与所有 belief 粒子都不一致")
        weights = [weight / evidence for weight in unnormalized]
        ess = 1.0 / sum(weight * weight for weight in weights)
        entropy = -sum(weight * math.log(weight) for weight in weights if weight > 0.0)

        self._particles = candidates
        self._weights = weights
        self._observation_count += 1
        self._latest_ess = ess
        self._latest_entropy = entropy
        self._log_evidence += math.log(evidence)
        self._zero_likelihood_particles += zero_likelihoods
        self._proposal_failures += proposal_failures
        if (
            self._resample_ess_fraction > 0.0
            and ess <= self._resample_ess_fraction * self.particle_count
        ):
            self._systematic_resample()
        return self.diagnostics

    def _systematic_resample(self) -> None:
        """Resample particles while keeping the next posterior normalized."""

        count = self.particle_count
        offset = self._rng.random() / count
        thresholds = [offset + index / count for index in range(count)]
        resampled: list[ParticleT] = []
        source_index = 0
        cumulative = self._weights[0]
        for threshold in thresholds:
            while threshold > cumulative and source_index < count - 1:
                source_index += 1
                cumulative += self._weights[source_index]
            resampled.append(self._particles[source_index])
        self._particles = resampled
        self._weights = [1.0 / count] * count
        self._resample_count += 1


def smoothed_policy_likelihood(
    scores: Sequence[float],
    *,
    observed_index: int,
    temperature: float = 1.0,
    uniform_mixture: float = 0.02,
) -> float:
    """Return a full-support behavior likelihood from policy scores.

    The uniform mixture makes filtering robust to an imperfect frozen behavior
    model: an observed legal action never forces every particle to zero merely
    because the model assigned it a near-zero score.  It is not a substitute
    for behavior-model calibration and must be recorded in collector metadata.
    """

    if not scores:
        raise ValueError("行为似然需要至少一个合法动作分数")
    if not 0 <= observed_index < len(scores):
        raise ValueError("observed_index 超出合法动作范围")
    if not math.isfinite(temperature) or temperature <= 0.0:
        raise ValueError("temperature 必须为正且有限")
    if not 0.0 <= uniform_mixture < 1.0:
        raise ValueError("uniform_mixture 必须在 0（含）到 1（不含）之间")
    if not all(math.isfinite(score) for score in scores):
        raise ValueError("行为分数必须有限")
    scaled = [score / temperature for score in scores]
    maximum = max(scaled)
    masses = [math.exp(score - maximum) for score in scaled]
    probability = masses[observed_index] / sum(masses)
    return (1.0 - uniform_mixture) * probability + uniform_mixture / len(scores)


def smoothed_deterministic_likelihood(
    *,
    selected_index: int,
    observed_index: int,
    action_count: int,
    uniform_mixture: float = 0.02,
) -> float:
    """Full-support likelihood for a deterministic behavior policy fallback."""

    if action_count <= 0:
        raise ValueError("action_count 必须为正数")
    if not 0 <= selected_index < action_count:
        raise ValueError("selected_index 超出合法动作范围")
    if not 0 <= observed_index < action_count:
        raise ValueError("observed_index 超出合法动作范围")
    if not 0.0 <= uniform_mixture < 1.0:
        raise ValueError("uniform_mixture 必须在 0（含）到 1（不含）之间")
    deterministic = 1.0 if selected_index == observed_index else 0.0
    return (1.0 - uniform_mixture) * deterministic + uniform_mixture / action_count
