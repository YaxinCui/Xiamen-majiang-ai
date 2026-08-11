#!/usr/bin/env python3
"""Resume and finish every immutable chunk of one formal causal split."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.audit_low_margin_top2_causal_split_v1 import audit_split
from scripts.collect_low_margin_top2_causal_chunk_v1 import SPLITS, collect_chunk


def collect_split(*, split: str, output_root: Path) -> dict[str, object]:
    if split not in SPLITS:
        raise ValueError("正式低分差 split 只能是 train 或 validation")
    chunk_count = int(SPLITS[split]["chunks"])
    directory = output_root / "data" / split
    for index in range(chunk_count):
        data_path = directory / f"chunk-{index:02d}.records.jsonl"
        report_path = directory / f"chunk-{index:02d}.report.json"
        if data_path.is_file() and report_path.is_file():
            print(
                json.dumps(
                    {"split": split, "chunk_index": index, "status": "existing"},
                    ensure_ascii=False,
                ),
                flush=True,
            )
            continue
        if data_path.exists() or report_path.exists():
            raise ValueError(
                f"{split}/chunk-{index:02d} 只有部分文件；拒绝自动覆盖"
            )
        result = collect_chunk(
            split=split, chunk_index=index, output_root=output_root
        )
        print(
            json.dumps(
                {
                    "split": split,
                    "chunk_index": index,
                    "status": result["status"],
                    "runtime_seconds": result["runtime_seconds"],
                    "records": result["slim_record_audit"]["records"],
                    "intervention_fraction": result["slim_record_audit"][
                        "intervention_fraction"
                    ],
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
    return audit_split(split=split, output_root=output_root)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=tuple(SPLITS), required=True)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("artifacts/low-margin-top2-causal-residual-v1"),
    )
    args = parser.parse_args()
    report_path = args.output_root / f"{args.split}-split-audit.json"
    if report_path.exists():
        raise ValueError("正式 split 总审计报告已存在，拒绝覆盖")
    report = collect_split(split=args.split, output_root=args.output_root)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "status": report["status"],
                "split": report["split"],
                "records": report["aggregate_audit"]["records"],
                "wall_groups": report["aggregate_audit"]["wall_groups"],
                "issues": report["aggregate_audit"]["issues"],
                "report": str(report_path),
            },
            ensure_ascii=False,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
