from lg_aimers_8th.calibration import extract_prompt_from_record


def test_extract_kmmlu_prompt_with_options():
    record = {
        "question": "다음 중 옳은 것은?",
        "options": ["A", "B", "C", "D"],
    }

    out = extract_prompt_from_record(record, "kmmlu")

    assert "다음 중 옳은 것은?" in out
    assert "1. A" in out
    assert "4. D" in out


def test_extract_gsm8k_prompt():
    record = {"question": "What is 2+2?"}
    assert extract_prompt_from_record(record, "gsm8k") == "What is 2+2?"


def test_extract_komt_prompt():
    record = {"turns": ["안녕", "요약해줘"]}
    assert extract_prompt_from_record(record, "komt") == "안녕\n요약해줘"
