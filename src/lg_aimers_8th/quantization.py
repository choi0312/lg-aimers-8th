from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Dict, List


def observer_config(ptq_mode: str) -> Dict[str, str]:
    if ptq_mode == "minmax":
        return {
            "weights": "memoryless_minmax",
            "input": "static_minmax",
            "output": "static_minmax",
        }

    if ptq_mode == "mse":
        return {
            "weights": "memoryless_mse",
            "input": "mse",
            "output": "mse",
        }

    raise ValueError("ptq_mode must be either 'minmax' or 'mse'.")


def import_quantization_modifier():
    try:
        from llmcompressor.modifiers.quantization.quantization import QuantizationModifier
    except Exception:
        from llmcompressor.modifiers.quantization.quantization.base import QuantizationModifier

    return QuantizationModifier


def make_quantization_modifier(
    scheme_candidates: List[str],
    ignore_linear: List[str],
    ptq_mode: str,
):
    QuantizationModifier = import_quantization_modifier()
    observer = observer_config(ptq_mode)

    last_error = None

    for scheme in scheme_candidates:
        try:
            modifier = QuantizationModifier(
                scheme={scheme: ["Linear"]},
                ignore=ignore_linear,
                observer=observer,
            )
            print(f"[OK] Using quantization scheme: {scheme}")
            return modifier
        except Exception as error:
            last_error = error
            print(f"[FAIL] scheme={scheme}: {type(error).__name__}: {error}")

    raise RuntimeError(f"Could not construct QuantizationModifier. Last error: {last_error}")


def run_oneshot_quantization(
    model,
    dataset,
    modifier,
    max_seq_length: int,
    num_calibration_samples: int,
    output_dir: str | Path,
) -> None:
    from llmcompressor import oneshot

    output_dir = Path(output_dir)

    if output_dir.exists():
        shutil.rmtree(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    oneshot(
        model=model,
        dataset=dataset,
        recipe=[modifier],
        max_seq_length=max_seq_length,
        num_calibration_samples=num_calibration_samples,
        output_dir=str(output_dir),
    )
