#!/usr/bin/env python3
"""Terminate one supervised process after a fixed deadline."""

from __future__ import annotations

import argparse
import os
import signal
import time
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pid", type=int, required=True)
    parser.add_argument("--timeout", type=float, required=True)
    parser.add_argument("--term-grace", type=float, required=True)
    parser.add_argument("--marker", type=Path, required=True)
    args = parser.parse_args()
    if args.pid <= 0 or args.timeout <= 0 or args.term_grace < 0:
        parser.error("invalid process deadline contract")

    deadline = time.monotonic() + args.timeout
    while time.monotonic() < deadline:
        try:
            os.kill(args.pid, 0)
        except ProcessLookupError:
            return
        time.sleep(min(0.05, max(deadline - time.monotonic(), 0.0)))
    try:
        os.kill(args.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    args.marker.write_text(f"{args.timeout}\n", encoding="utf-8")
    time.sleep(args.term_grace)
    try:
        os.kill(args.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass


if __name__ == "__main__":
    main()
