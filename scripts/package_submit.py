from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from lg_aimers_8th.packaging import make_submit_zip, zip_size_gb


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=str, required=True)
    parser.add_argument("--zip_path", type=str, default="/content/submit.zip")
    parser.add_argument("--skip_validation", action="store_true")
    args = parser.parse_args()

    zip_path = make_submit_zip(
        model_dir=args.model_dir,
        zip_path=args.zip_path,
        validate_model_files=not args.skip_validation,
    )

    print("Saved:", zip_path)
    print("zip_size_gb:", zip_size_gb(zip_path))


if __name__ == "__main__":
    main()
