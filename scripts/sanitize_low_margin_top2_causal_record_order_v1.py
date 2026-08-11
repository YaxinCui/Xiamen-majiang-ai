#!/usr/bin/env python3
"""Remove the wall/seat row-order side channel from already written chunks."""

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
    RECORD_ORDER_CONTRACT,
    SPLITS,
)
from xiamen_mahjong.low_margin_intervention import (
    read_low_margin_causal_jsonl,
    write_low_margin_causal_jsonl,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sanitize_split(*, split: str, output_root: Path) -> dict[str, object]:
    if split not in SPLITS:
        raise ValueError("记录顺序消毒 split 无效")
    directory = output_root / "data" / split
    rewritten = []
    skipped = []
    for index in range(int(SPLITS[split]["chunks"])):
        data_path = directory / f"chunk-{index:02d}.records.jsonl"
        report_path = directory / f"chunk-{index:02d}.report.json"
        if not data_path.exists() and not report_path.exists():
            continue
        if not data_path.is_file() or not report_path.is_file():
            raise ValueError(f"{split}/chunk-{index:02d} 文件不完整")
        report = json.loads(report_path.read_text(encoding="utf-8"))
        if report.get("safe_records", {}).get("sha256") != _sha256(data_path):
            raise ValueError(f"{split}/chunk-{index:02d} 消毒前 SHA 不一致")
        if report["safe_records"].get("record_order") == RECORD_ORDER_CONTRACT:
            skipped.append(index)
            continue
        records = read_low_margin_causal_jsonl(data_path)
        records.sort(key=lambda record: record.record_id)
        temporary = data_path.with_suffix(data_path.suffix + ".order-sanitizing")
        if temporary.exists():
            raise ValueError(f"{temporary} 已存在，拒绝覆盖")
        write_low_margin_causal_jsonl(records, temporary)
        old_sha = _sha256(data_path)
        temporary.replace(data_path)
        report["safe_records"]["pre_order_sanitization_sha256"] = old_sha
        report["safe_records"]["sha256"] = _sha256(data_path)
        report["safe_records"]["bytes"] = data_path.stat().st_size
        report["safe_records"]["record_order"] = RECORD_ORDER_CONTRACT
        report_path.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        rewritten.append(index)
    return {
        "status": "low_margin_causal_row_order_sanitized",
        "split": split,
        "rewritten_chunks": rewritten,
        "already_sanitized_chunks": skipped,
        "record_order": RECORD_ORDER_CONTRACT,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=tuple(SPLITS), required=True)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("artifacts/low-margin-top2-causal-residual-v1"),
    )
    args = parser.parse_args()
    print(
        json.dumps(
            sanitize_split(split=args.split, output_root=args.output_root),
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
