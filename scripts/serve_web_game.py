#!/usr/bin/env python3
"""Launch the local Xiamen Mahjong human-vs-Teacher web game."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from xiamen_mahjong.web import serve


def main() -> None:
    parser = argparse.ArgumentParser(description="启动本机厦门麻将网页游戏")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    serve(args.host, args.port)


if __name__ == "__main__":
    main()
