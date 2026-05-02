from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from lg_aimers_8th.pipeline import run_ptq_pipeline


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--hf_token", type=str, default=None)
    args = parser.parse_args()

    hf_token = args.hf_token or os.environ.get("HF_TOKEN")
    result = run_ptq_pipeline(args.config, hf_token=hf_token)
    print(result)


if __name__ == "__main__":
    main()
