#!/usr/bin/env python3
"""Train the fixed low-dimensional discard scorer with CPU self-play."""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import json
import math
from pathlib import Path
import random
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from xiamen_mahjong.agents import (
    DiscardShapeWeights,
    ParametricDiscardTeacherAgent,
)
from xiamen_mahjong.evaluation import evaluate_against_teacher


RNG_SEED = 202626400
TRAIN_SEGMENTS = ((202626500, 20), (202626600, 20))
VALIDATION_SEED = 202627000
VALIDATION_HANDS = 100
GENERATIONS = 4
POPULATION = 12
ELITES = 4
INITIAL_SIGMA = 0.45
MIN_SIGMA = 0.12
MAX_SIGMA = 0.60
LOG_BOUND = 1.5
SEARCH_FIELDS = (
    "triplet_group",
    "pair_remainder",
    "adjacent_overlap",
    "gap_overlap",
    "wait_face",
    "gold_discard_penalty",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def weights_from_logs(log_multipliers: tuple[float, ...]) -> DiscardShapeWeights:
    if len(log_multipliers) != len(SEARCH_FIELDS):
        raise ValueError("搜索向量维度错误")
    baseline = DiscardShapeWeights()
    updates = {
        field: getattr(baseline, field) * math.exp(value)
        for field, value in zip(SEARCH_FIELDS, log_multipliers)
    }
    return replace(baseline, **updates)


def evaluate_logs(logs: tuple[float, ...]) -> dict[str, Any]:
    weights = weights_from_logs(logs)
    segments = []
    for seed, hands in TRAIN_SEGMENTS:
        result = evaluate_against_teacher(
            ParametricDiscardTeacherAgent(weights),
            hands=hands,
            seed=seed,
            profile="classic",
        )
        segments.append(result.candidate_score_mean)
    return {
        "logs": list(logs),
        "weights": asdict(weights),
        "segment_means": segments,
        "fitness_worst_segment": min(segments),
        "fitness_mean": sum(segments) / len(segments),
    }


def fitness_key(row: dict[str, Any]) -> tuple[float, float]:
    return (
        float(row["fitness_worst_segment"]),
        float(row["fitness_mean"]),
    )


def main() -> None:
    args = parse_args()
    rng = random.Random(RNG_SEED)
    center = [0.0] * len(SEARCH_FIELDS)
    sigma = [INITIAL_SIGMA] * len(SEARCH_FIELDS)
    cache: dict[tuple[float, ...], dict[str, Any]] = {}
    history: list[dict[str, Any]] = []
    global_best: dict[str, Any] | None = None

    for generation in range(GENERATIONS):
        population = [tuple(center)]
        while len(population) < POPULATION:
            population.append(
                tuple(
                    max(-LOG_BOUND, min(LOG_BOUND, rng.gauss(mu, scale)))
                    for mu, scale in zip(center, sigma)
                )
            )
        rows = []
        for index, logs in enumerate(population):
            key = tuple(round(value, 12) for value in logs)
            if key not in cache:
                cache[key] = evaluate_logs(logs)
            row = {**cache[key], "generation": generation, "index": index}
            rows.append(row)
            history.append(row)
            print(
                json.dumps(
                    {
                        "generation": generation,
                        "index": index,
                        "worst": row["fitness_worst_segment"],
                        "mean": row["fitness_mean"],
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
        rows.sort(key=fitness_key, reverse=True)
        if global_best is None or fitness_key(rows[0]) > fitness_key(global_best):
            global_best = dict(rows[0])
        elites = rows[:ELITES]
        for dimension in range(len(SEARCH_FIELDS)):
            values = [float(row["logs"][dimension]) for row in elites]
            center[dimension] = sum(values) / len(values)
            variance = sum(
                (value - center[dimension]) ** 2 for value in values
            ) / len(values)
            sigma[dimension] = max(
                MIN_SIGMA, min(MAX_SIGMA, math.sqrt(variance))
            )

    if global_best is None:
        raise AssertionError("演化训练没有产生候选")
    best_weights = DiscardShapeWeights(**global_best["weights"])
    validation = evaluate_against_teacher(
        ParametricDiscardTeacherAgent(best_weights),
        hands=VALIDATION_HANDS,
        seed=VALIDATION_SEED,
        profile="classic",
    )
    payload = {
        "status": (
            "validation_passed_ready_to_freeze_selection"
            if validation.candidate_score_mean > 0.0
            else "validation_rejected_selection_unread"
        ),
        "protocol": {
            "rng_seed": RNG_SEED,
            "train_segments": TRAIN_SEGMENTS,
            "validation_seed": VALIDATION_SEED,
            "validation_hands": VALIDATION_HANDS,
            "generations": GENERATIONS,
            "population": POPULATION,
            "elites": ELITES,
            "initial_sigma": INITIAL_SIGMA,
            "sigma_bounds": [MIN_SIGMA, MAX_SIGMA],
            "log_bound": LOG_BOUND,
            "search_fields": SEARCH_FIELDS,
        },
        "best_training_candidate": global_best,
        "final_center": center,
        "final_sigma": sigma,
        "validation": validation.payload(),
        "validation_passes": validation.candidate_score_mean > 0.0,
        "history": history,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({key: value for key, value in payload.items() if key != "history"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
