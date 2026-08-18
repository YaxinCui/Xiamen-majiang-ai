#!/usr/bin/env python3
"""Serve the local actor-visible human correction review queue."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from xiamen_mahjong.review_web import serve_review


def _require_local_human_path(path: Path) -> None:
    root = (ROOT / "local_human_data").resolve()
    resolved = path.resolve()
    if resolved != root and root not in resolved.parents:
        raise ValueError("queue、priority 和 label 必须位于 Git 忽略的 local_human_data/")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=51861)
    parser.add_argument("--queue", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--priority", type=Path)
    args = parser.parse_args()
    _require_local_human_path(args.queue)
    _require_local_human_path(args.output)
    if args.priority is not None:
        _require_local_human_path(args.priority)
    resolved_paths = [args.queue.resolve(), args.output.resolve()]
    if args.priority is not None:
        resolved_paths.append(args.priority.resolve())
    if len(set(resolved_paths)) != len(resolved_paths):
        parser.error("--queue、--priority 与 --output 必须是不同文件")
    serve_review(
        args.host,
        args.port,
        queue_path=args.queue,
        output_path=args.output,
        priority_path=args.priority,
    )


if __name__ == "__main__":
    main()
