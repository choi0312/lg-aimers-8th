from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

from lg_aimers_8th.calibration import (
    build_or_load_calibration_jsonl,
    tokenize_calibration_dataset,
)
from lg_aimers_8th.config import load_config, validate_calibration_counts
from lg_aimers_8th.model_utils import (
    build_ignore_linear,
    load_causal_lm,
    load_tokenizer,
    login_huggingface,
    save_hf_model,
)
from lg_aimers_8th.packaging import make_submit_zip, zip_size_gb
from lg_aimers_8th.quantization import make_quantization_modifier, run_oneshot_quantization
from lg_aimers_8th.sanity import run_generation_sanity


def run_ptq_pipeline(config_path: str | Path, hf_token: str | None = None) -> dict:
    cfg = load_config(config_path)
    validate_calibration_counts(cfg)

    hf_token = hf_token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
    login_huggingface(hf_token)

    model_cfg = cfg["model"]
    project_cfg = cfg["project"]
    cal_cfg = cfg["calibration"]
    quant_cfg = cfg["quantization"]

    work_dir = Path(project_cfg["work_dir"])
    model_out_dir = Path(project_cfg["model_out_dir"])
    work_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = load_tokenizer(
        model_cfg["model_id"],
        trust_remote_code=bool(model_cfg.get("trust_remote_code", True)),
    )

    model = load_causal_lm(
        model_id=model_cfg["model_id"],
        trust_remote_code=bool(model_cfg.get("trust_remote_code", True)),
        torch_dtype=model_cfg.get("torch_dtype", "auto"),
        device_map=model_cfg.get("device_map", "auto"),
    )

    ignore_linear = build_ignore_linear(
        model,
        default_ignore=model_cfg.get("ignore_linear", ["lm_head"]),
    )

    print("[MODEL] loaded:", model_cfg["model_id"])
    print("[IGNORE_LINEAR]", ignore_linear)

    jsonl_path = build_or_load_calibration_jsonl(
        tokenizer=tokenizer,
        config=cfg,
        hf_token=hf_token,
    )

    dataset = tokenize_calibration_dataset(
        tokenizer=tokenizer,
        jsonl_path=jsonl_path,
        max_seq_len=int(cal_cfg["max_seq_len"]),
    )

    modifier = make_quantization_modifier(
        scheme_candidates=list(quant_cfg.get("scheme_candidates", ["W8A8-INT8", "INT8_W8A8", "W8A8"])),
        ignore_linear=ignore_linear,
        ptq_mode=str(quant_cfg.get("ptq_mode", "minmax")),
    )

    run_oneshot_quantization(
        model=model,
        dataset=dataset,
        modifier=modifier,
        max_seq_length=int(cal_cfg["max_seq_len"]),
        num_calibration_samples=int(cal_cfg["num_samples"]),
        output_dir=model_out_dir,
    )

    save_hf_model(
        model=model,
        tokenizer=tokenizer,
        output_dir=str(model_out_dir),
        save_compressed=bool(quant_cfg.get("save_compressed", True)),
    )

    sanity_result = None
    if cfg.get("sanity", {}).get("enabled", True):
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"

        tok2 = load_tokenizer(
            str(model_out_dir),
            trust_remote_code=True,
        )
        model2 = load_causal_lm(
            model_id=str(model_out_dir),
            trust_remote_code=True,
            torch_dtype="auto",
            device_map="auto",
            local_files_only=True,
        )

        sanity_result = run_generation_sanity(
            tokenizer=tok2,
            model=model2,
            prompts=list(cfg["sanity"].get("prompts", [])),
            max_new_tokens=int(cfg["sanity"].get("max_new_tokens", 64)),
            eos_token_id=int(cfg["sanity"].get("eos_token_id", 361)),
        )

        sanity_path = work_dir / "sanity_results.json"
        sanity_path.write_text(
            json.dumps(sanity_result, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    zip_path = make_submit_zip(
        model_dir=model_out_dir,
        zip_path=project_cfg["submit_zip_local"],
        validate_model_files=True,
    )

    drive_path = project_cfg.get("submit_zip_drive")
    if drive_path:
        drive_path = Path(drive_path)
        drive_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(zip_path, drive_path)

    result = {
        "model_out_dir": str(model_out_dir),
        "submit_zip_local": str(zip_path),
        "submit_zip_size_gb": zip_size_gb(zip_path),
        "submit_zip_drive": str(drive_path) if drive_path else None,
        "calibration_jsonl": str(jsonl_path),
        "ignore_linear": ignore_linear,
        "sanity_enabled": sanity_result is not None,
    }

    summary_path = work_dir / "pipeline_summary.json"
    summary_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    return result
