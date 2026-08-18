#!/usr/bin/env python3
"""Audit all immutable chunks of one formal low-margin causal split."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.collect_low_margin_top2_causal_chunk_v1 import (
    CHUNK_WALLS,
    RECORD_ORDER_CONTRACT,
    SPLITS,
)
from xiamen_mahjong.low_margin_intervention import (
    audit_low_margin_causal_records,
    read_low_margin_causal_jsonl,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def audit_split(*, split: str, output_root: Path) -> dict[str, object]:
    if split not in SPLITS:
        raise ValueError("正式低分差 split 只能是 train 或 validation")
    chunk_count = int(SPLITS[split]["chunks"])
    directory = output_root / "data" / split
    expected_data = [
        directory / f"chunk-{index:02d}.records.jsonl"
        for index in range(chunk_count)
    ]
    expected_reports = [
        directory / f"chunk-{index:02d}.report.json"
        for index in range(chunk_count)
    ]
    missing = [
        str(path) for path in [*expected_data, *expected_reports]
        if not path.is_file()
    ]
    if missing:
        raise ValueError("正式低分差 split 缺少固定 chunk：" + ", ".join(missing))
    records = []
    chunk_files = []
    for index, (data_path, report_path) in enumerate(
        zip(expected_data, expected_reports)
    ):
        report = json.loads(report_path.read_text(encoding="utf-8"))
        digest = _sha256(data_path)
        if (
            report.get("status") != "formal_low_margin_top2_causal_chunk_ready"
            or report.get("split") != split
            or report.get("chunk_index") != index
            or report.get("safe_records", {}).get("sha256") != digest
            or report.get("safe_records", {}).get("record_order")
            != RECORD_ORDER_CONTRACT
        ):
            raise ValueError(f"正式低分差 {split} chunk-{index:02d} 身份或摘要不一致")
        records.extend(read_low_margin_causal_jsonl(data_path))
        chunk_files.append(
            {
                "chunk_index": index,
                "path": str(data_path),
                "sha256": digest,
                "bytes": data_path.stat().st_size,
            }
        )
    expected_walls = chunk_count * CHUNK_WALLS
    aggregate = audit_low_margin_causal_records(
        records,
        expected_wall_groups=expected_walls,
        # ``read_low_margin_causal_jsonl`` already validated every pair while
        # parsing.  Repeating the exact expensive hand analysis here would not
        # add an independent check.
        validate_actor_visible_pairs=False,
    )
    return {
        "status": (
            "formal_low_margin_top2_causal_split_ready"
            if aggregate["ready"]
            else "formal_low_margin_top2_causal_split_rejected"
        ),
        "split": split,
        "chunks": chunk_count,
        "chunk_walls": CHUNK_WALLS,
        "expected_wall_groups": expected_walls,
        "aggregate_audit": aggregate,
        "chunk_files": chunk_files,
        "warning": (
            "This report validates data mechanics only. It does not estimate "
            "or authorize policy strength."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=tuple(SPLITS), required=True)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("artifacts/low-margin-top2-causal-residual-v1"),
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("正式低分差 split 审计报告已存在，拒绝覆盖")
    report = audit_split(split=args.split, output_root=args.output_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
