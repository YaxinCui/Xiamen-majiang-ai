#!/usr/bin/env python3
"""Serve the local actor-visible pass/chi/pong review queue."""

from __future__ import annotations

import argparse
from http.server import ThreadingHTTPServer
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from xiamen_mahjong.response_review_web import ResponseReviewStore
from xiamen_mahjong.review_web import make_review_handler


def _require_local_path(path: Path) -> None:
    root = (ROOT / "local_human_data").resolve()
    resolved = path.resolve()
    if resolved != root and root not in resolved.parents:
        raise ValueError("response queue 和 label 必须位于 local_human_data/")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=51862)
    parser.add_argument("--queue", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    _require_local_path(args.queue)
    _require_local_path(args.output)
    if args.queue.resolve() == args.output.resolve():
        parser.error("--queue 与 --output 不能是同一个文件")
    store = ResponseReviewStore(args.queue, args.output)
    server = ThreadingHTTPServer(
        (args.host, args.port), make_review_handler(store)
    )
    print(f"厦门麻将响应纠错审阅已启动：http://{args.host}:{args.port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
