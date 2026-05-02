from lg_aimers_8th.config import validate_calibration_counts


def test_validate_calibration_counts():
    cfg = {
        "calibration": {
            "num_samples": 5,
            "counts": {
                "a": 2,
                "b": 3,
            },
        }
    }

    assert validate_calibration_counts(cfg) == 5
