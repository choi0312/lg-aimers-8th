from __future__ import annotations

from collections import Counter
from typing import Dict, List


def build_prompt(tokenizer, user_text: str) -> str:
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": user_text},
    ]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )


def generation_stats(tokenizer, token_ids: List[int], prompt_len: int, eos_token_id: int | None = None) -> Dict:
    gen_ids = token_ids[prompt_len:]
    counter = Counter(gen_ids)
    total = len(gen_ids)
    unique = len(counter)

    if total > 0:
        most_id, most_count = counter.most_common(1)[0]
        most_token = repr(tokenizer.decode([most_id]))
    else:
        most_id, most_count, most_token = None, 0, None

    return {
        "gen_tokens": total,
        "unique_tokens": unique,
        "unique_ratio": unique / total if total else 0.0,
        "top_repeat_id": most_id,
        "top_repeat_count": most_count,
        "top_repeat_token": most_token,
        "has_eos": eos_token_id in gen_ids if eos_token_id is not None else None,
    }


def run_generation_sanity(
    tokenizer,
    model,
    prompts: List[str],
    max_new_tokens: int = 64,
    eos_token_id: int | None = None,
) -> List[Dict]:
    import torch

    results = []

    for prompt_text in prompts:
        prompt = build_prompt(tokenizer, prompt_text)
        prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
        inputs = tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
            )

        token_ids = output[0].tolist()
        decoded = tokenizer.decode(token_ids, skip_special_tokens=False)

        results.append(
            {
                "prompt": prompt_text,
                "stats": generation_stats(
                    tokenizer=tokenizer,
                    token_ids=token_ids,
                    prompt_len=len(prompt_ids),
                    eos_token_id=eos_token_id,
                ),
                "decoded_preview": decoded[:800],
            }
        )

    return results
