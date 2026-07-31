#!/usr/bin/env python3
"""Emit strict one-second NVIDIA GPU snapshots until terminated."""

from __future__ import annotations

import datetime
import subprocess
import time


def main() -> None:
    while True:
        print(
            datetime.datetime.now().astimezone().strftime("%Y-%m-%dT%H:%M:%S.%f%z"),
            flush=True,
        )
        subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,memory.used,utilization.gpu,utilization.memory,power.draw",
                "--format=csv,noheader",
            ],
            check=True,
        )
        time.sleep(1)


if __name__ == "__main__":
    main()
