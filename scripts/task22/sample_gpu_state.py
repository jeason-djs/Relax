#!/usr/bin/env python3
"""Emit periodic NVIDIA GPU snapshots until terminated."""

from __future__ import annotations

import argparse
import datetime
import subprocess
import sys
import time


def snapshot_block() -> str:
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,memory.used,utilization.gpu,utilization.memory,power.draw",
            "--format=csv,noheader",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    rows = result.stdout.rstrip("\r\n")
    timestamp = datetime.datetime.now().astimezone().strftime("%Y-%m-%dT%H:%M:%S.%f%z")
    return f"{timestamp}\n{rows}\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--interval", type=float, default=1.0)
    args = parser.parse_args()
    if args.interval <= 0:
        parser.error("--interval must be positive")
    while True:
        sys.stdout.write(snapshot_block())
        sys.stdout.flush()
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
