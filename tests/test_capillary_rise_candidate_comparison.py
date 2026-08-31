from __future__ import annotations

import csv
import importlib.util
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = (
    ROOT
    / "tests/cases/fluid/open_vessel_free_surface"
    / "compare_capillary_rise_candidate.py"
)


def load_module():
    specification = importlib.util.spec_from_file_location(
        "free_surface_capillary_rise_candidate_comparison",
        MODULE_PATH,
    )
    assert specification is not None and specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    sys.modules[specification.name] = module
    specification.loader.exec_module(module)
    return module


def comparison() -> dict:
    return {
        "comparison_id": "test_comparison",
        "common_grid": {
            "start_s": 0.0,
            "end_s": 0.002,
            "step_s": 0.001,
            "point_count": 3,
        },
        "reference_uncertainty": {"confidence_multiplier": 2.0},
    }


def envelope() -> list[dict[str, float]]:
    return [
        {
            "time_s": index * 0.001,
            "reference_center_mm": 10.0 + index,
            "reference_uncertainty_mm": 0.1,
        }
        for index in range(3)
    ]


def candidate(offset: float = 0.0, uncertainty: float = 0.0):
    return [
        {
            "time_s": index * 0.001,
            "apex_height_mm": 10.0 + index + offset,
            "numerical_uncertainty_mm": uncertainty,
        }
        for index in range(3)
    ]


def test_exact_reference_center_passes_history_but_not_closure():
    module = load_module()

    rows, summary = module.compare_history(
        envelope(), candidate(), comparison()
    )

    assert len(rows) == 3
    assert all(row["point_passed"] for row in rows)
    assert summary["outcome"] == "PASS"
    assert summary["rms_error_mm"] == 0.0
    assert summary["qualification_disposition"]["history_envelope_passed"]
    assert not summary["qualification_disposition"][
        "candidate_refinement_qualified"
    ]
    assert not summary["qualification_disposition"]["wp5_closed"]


def test_pointwise_and_rms_uncertainty_rules_fail_large_offset():
    module = load_module()

    rows, summary = module.compare_history(
        envelope(), candidate(offset=0.21), comparison()
    )

    assert not any(row["point_passed"] for row in rows)
    assert summary["outcome"] == "FAIL"
    assert summary["failed_point_count"] == 3
    assert not summary["rms_passed"]


def test_candidate_uncertainty_enters_declared_root_sum_square_rule():
    module = load_module()

    rows, summary = module.compare_history(
        envelope(), candidate(offset=0.25, uncertainty=0.1), comparison()
    )

    assert all(row["point_passed"] for row in rows)
    assert summary["outcome"] == "PASS"
    assert summary["rms_acceptance_half_width_mm"] == pytest.approx(
        2.0 * (0.1 ** 2 + 0.1 ** 2) ** 0.5
    )


@pytest.mark.parametrize(
    ("rows", "message"),
    [
        (
            [
                {
                    "time_s": 1.0e-6,
                    "apex_height_mm": 10.0,
                    "numerical_uncertainty_mm": 0.0,
                },
                {
                    "time_s": 0.002,
                    "apex_height_mm": 12.0,
                    "numerical_uncertainty_mm": 0.0,
                },
            ],
            "unshifted comparison origin",
        ),
        (
            [
                {
                    "time_s": 0.0,
                    "apex_height_mm": 10.0,
                    "numerical_uncertainty_mm": 0.0,
                },
                {
                    "time_s": 0.001,
                    "apex_height_mm": 11.0,
                    "numerical_uncertainty_mm": 0.0,
                },
            ],
            "fixed comparison endpoint",
        ),
    ],
)
def test_candidate_support_cannot_be_shifted_or_truncated(rows, message):
    module = load_module()

    with pytest.raises(ValueError, match=message):
        module.compare_history(envelope(), rows, comparison())


def test_candidate_loader_rejects_duplicate_time_and_negative_uncertainty(
    tmp_path: Path,
):
    module = load_module()
    path = tmp_path / "candidate.csv"
    with path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.writer(output, lineterminator="\n")
        writer.writerow(module.CANDIDATE_COLUMNS)
        writer.writerow((0.0, 10.0, 0.0))
        writer.writerow((0.0, 11.0, -0.1))

    with pytest.raises(ValueError, match="negative numerical uncertainty"):
        module.load_candidate_history(path)


def test_comparison_outputs_are_immutable(tmp_path: Path):
    module = load_module()
    rows, summary = module.compare_history(
        envelope(), candidate(), comparison()
    )
    csv_path = tmp_path / "comparison.csv"
    json_path = tmp_path / "summary.json"

    module.write_comparison_csv(csv_path, rows)
    module.write_summary(json_path, summary)

    with pytest.raises(FileExistsError):
        module.write_comparison_csv(csv_path, rows)
    with pytest.raises(FileExistsError):
        module.write_summary(json_path, summary)
