from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import yaml


def load_config(config_path: str | Path) -> Dict[str, Any]:
    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    if config is None:
        raise ValueError(f"Empty config: {config_path}")

    return config


def validate_calibration_counts(config: Dict[str, Any]) -> int:
    counts = config["calibration"]["counts"]
    total = int(sum(int(v) for v in counts.values()))
    expected = int(config["calibration"]["num_samples"])

    if total != expected:
        raise ValueError(f"Calibration counts mismatch: {total} != {expected}")

    return total
