from pathlib import Path

from lg_aimers_8th.packaging import list_zip_top_levels, make_submit_zip, validate_model_dir


def test_make_submit_zip_structure(tmp_path):
    model_dir = tmp_path / "model_artifact"
    model_dir.mkdir()

    (model_dir / "config.json").write_text("{}", encoding="utf-8")
    (model_dir / "tokenizer_config.json").write_text("{}", encoding="utf-8")
    (model_dir / "model.safetensors").write_text("dummy", encoding="utf-8")

    validate_model_dir(model_dir)

    zip_path = tmp_path / "submit.zip"
    make_submit_zip(model_dir, zip_path, validate_model_files=True)

    assert zip_path.exists()
    assert list_zip_top_levels(zip_path) == ["model"]
