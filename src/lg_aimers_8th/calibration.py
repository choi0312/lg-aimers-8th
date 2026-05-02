from __future__ import annotations

import json
import os
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List


def build_chat_prompt(tokenizer, user_text: str) -> str:
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": user_text},
    ]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )


def extract_prompt_from_record(record: Dict[str, Any], tag: str) -> str | None:
    if tag == "manta":
        conv = record.get("conversations")
        if isinstance(conv, list):
            users = [
                m
                for m in conv
                if isinstance(m, dict)
                and str(m.get("role", "")).lower() in ["user", "human"]
            ]
            if users:
                text = users[-1].get("content", "")
                return text.strip() if isinstance(text, str) else None
        return None

    if tag == "komt":
        turns = record.get("turns")
        if isinstance(turns, list):
            joined = "\n".join([t for t in turns if isinstance(t, str) and t.strip()])
            return joined.strip() if joined.strip() else None
        return None

    if tag == "kmmlu":
        question = record.get("question", "")
        options = record.get("options")

        if isinstance(question, str) and question.strip():
            if isinstance(options, list) and options and all(isinstance(x, str) for x in options):
                opt_text = "\n".join([f"{i + 1}. {x}" for i, x in enumerate(options)])
                return f"{question.strip()}\n\n선택지:\n{opt_text}"
            return question.strip()
        return None

    if tag == "gsm8k":
        question = record.get("question", "")
        if isinstance(question, str) and question.strip():
            return question.strip()
        return None

    return None


def token_length(tokenizer, text: str) -> int:
    return len(
        tokenizer(
            text,
            add_special_tokens=False,
            truncation=False,
        )["input_ids"]
    )


def write_jsonl(records: Iterable[Dict[str, Any]], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def read_jsonl(path: str | Path) -> List[Dict[str, Any]]:
    path = Path(path)

    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def calibration_paths(config: Dict[str, Any]) -> tuple[Path, Path]:
    cal_cfg = config["calibration"]
    out_dir = Path(cal_cfg["output_dir"])
    return out_dir / cal_cfg["jsonl_name"], out_dir / cal_cfg["meta_name"]


def needs_build(jsonl_path: str | Path, expected_lines: int) -> bool:
    path = Path(jsonl_path)

    if not path.exists():
        return True

    try:
        n_lines = sum(1 for _ in path.open("r", encoding="utf-8"))
    except Exception:
        return True

    return n_lines != expected_lines


def load_dataset_from_config(dataset_cfg: Dict[str, Any], token: str | None):
    from datasets import load_dataset

    kwargs = {
        "path": dataset_cfg["path"],
        "split": dataset_cfg["split"],
        "token": token,
    }

    if dataset_cfg.get("name") is not None:
        kwargs["name"] = dataset_cfg["name"]

    if dataset_cfg.get("streaming", False):
        kwargs["streaming"] = True

    return load_dataset(**kwargs)


def collect_calibration_records(
    tokenizer,
    config: Dict[str, Any],
    hf_token: str | None,
) -> List[Dict[str, str]]:
    cal_cfg = config["calibration"]
    counts = {k: int(v) for k, v in cal_cfg["counts"].items()}
    datasets_cfg = cal_cfg["datasets"]
    max_seq_len = int(cal_cfg["max_seq_len"])
    seed = int(config["project"].get("seed", 2026))

    random.seed(seed)
    records: List[Dict[str, str]] = []

    for tag, count in counts.items():
        ds_cfg = datasets_cfg[tag]
        ds = load_dataset_from_config(ds_cfg, token=hf_token)

        selected = []

        if ds_cfg.get("streaming", False):
            ds = ds.shuffle(seed=seed, buffer_size=10000)
            for row in ds:
                text = extract_prompt_from_record(row, tag)
                if text:
                    prompt = build_chat_prompt(tokenizer, text)
                    if token_length(tokenizer, prompt) <= max_seq_len:
                        selected.append({"dataset_tag": tag, "prompt": prompt})
                if len(selected) >= count:
                    break
        else:
            idxs = list(range(len(ds)))
            random.shuffle(idxs)

            for ix in idxs:
                text = extract_prompt_from_record(ds[ix], tag)
                if text:
                    prompt = build_chat_prompt(tokenizer, text)
                    if token_length(tokenizer, prompt) <= max_seq_len:
                        selected.append({"dataset_tag": tag, "prompt": prompt})
                if len(selected) >= count:
                    break

        if len(selected) != count:
            raise RuntimeError(f"Not enough calibration samples for {tag}: {len(selected)} / {count}")

        records.extend(selected)

    random.shuffle(records)

    expected = int(cal_cfg["num_samples"])
    if len(records) != expected:
        raise RuntimeError(f"Calibration sample count mismatch: {len(records)} / {expected}")

    return records


def build_or_load_calibration_jsonl(
    tokenizer,
    config: Dict[str, Any],
    hf_token: str | None,
) -> Path:
    jsonl_path, meta_path = calibration_paths(config)
    expected = int(config["calibration"]["num_samples"])

    if not needs_build(jsonl_path, expected):
        print(f"[SKIP] Reusing existing calibration file: {jsonl_path}")
        return jsonl_path

    print(f"[BUILD] Creating calibration file: {jsonl_path}")
    records = collect_calibration_records(tokenizer, config, hf_token)

    lengths = [token_length(tokenizer, record["prompt"]) for record in records]

    meta = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model_id": config["model"]["model_id"],
        "max_seq_len": int(config["calibration"]["max_seq_len"]),
        "seed": int(config["project"].get("seed", 2026)),
        "counts": config["calibration"]["counts"],
        "token_stats": {
            "min": int(min(lengths)),
            "p50": int(sorted(lengths)[len(lengths) // 2]),
            "max": int(max(lengths)),
        },
    }

    write_jsonl(records, jsonl_path)

    meta_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path.write_text(
        json.dumps(meta, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"[OK] Saved calibration JSONL: {jsonl_path}")
    print(f"[OK] Saved calibration meta: {meta_path}")

    return jsonl_path


def tokenize_calibration_dataset(tokenizer, jsonl_path: str | Path, max_seq_len: int):
    from datasets import load_dataset

    ds = load_dataset("json", data_files=str(jsonl_path), split="train")
    ds = ds.map(lambda x: {"text": x["prompt"]}, remove_columns=ds.column_names)

    def tokenize(sample):
        return tokenizer(
            sample["text"],
            padding=False,
            truncation=True,
            max_length=max_seq_len,
            add_special_tokens=False,
        )

    ds = ds.map(tokenize, remove_columns=ds.column_names)
    return ds
