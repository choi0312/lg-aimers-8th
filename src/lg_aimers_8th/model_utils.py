from __future__ import annotations

import os
from typing import List


def login_huggingface(hf_token: str | None) -> None:
    if not hf_token:
        return

    os.environ["HF_TOKEN"] = hf_token
    os.environ["HUGGINGFACE_HUB_TOKEN"] = hf_token

    from huggingface_hub import login

    login(token=hf_token, add_to_git_credential=False)


def load_tokenizer(model_id: str, trust_remote_code: bool = True):
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(
        model_id,
        trust_remote_code=trust_remote_code,
    )


def load_causal_lm(
    model_id: str,
    trust_remote_code: bool = True,
    torch_dtype: str = "auto",
    device_map: str = "auto",
    local_files_only: bool = False,
):
    from transformers import AutoModelForCausalLM

    return AutoModelForCausalLM.from_pretrained(
        model_id,
        trust_remote_code=trust_remote_code,
        torch_dtype=torch_dtype,
        device_map=device_map,
        low_cpu_mem_usage=True,
        local_files_only=local_files_only,
    ).eval()


def build_ignore_linear(model, default_ignore: List[str] | None = None) -> List[str]:
    import torch

    ignore = list(default_ignore or [])

    if hasattr(model, "lm_head") and isinstance(model.lm_head, torch.nn.Linear):
        ignore.append("lm_head")

    return sorted(set(ignore))


def save_hf_model(model, tokenizer, output_dir: str, save_compressed: bool = True) -> None:
    try:
        model.tie_weights()
    except Exception:
        pass

    model.save_pretrained(output_dir, save_compressed=save_compressed)
    tokenizer.save_pretrained(output_dir)
