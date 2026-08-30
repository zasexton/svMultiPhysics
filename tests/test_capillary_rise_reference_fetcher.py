import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import pytest


def _repository() -> Path:
    return Path(__file__).resolve().parents[1]


def _fetcher_path() -> Path:
    return (
        _repository()
        / "tests"
        / "cases"
        / "fluid"
        / "open_vessel_free_surface"
        / "fetch_capillary_rise_reference.py"
    )


def _registry_path() -> Path:
    return (
        _repository()
        / "tests"
        / "cases"
        / "fluid"
        / "free_surface_wp5_capillary_rise_reference.json"
    )


def _load_fetcher():
    specification = importlib.util.spec_from_file_location(
        "free_surface_capillary_rise_reference_fetcher",
        _fetcher_path(),
    )
    assert specification is not None
    assert specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    sys.modules[specification.name] = module
    specification.loader.exec_module(module)
    return module


def test_reference_registry_pins_four_distinct_published_series():
    fetcher = _load_fetcher()
    registry = fetcher.load_registry(_registry_path())

    assert registry["reference_id"] == (
        "gruending_2020_capillary_rise_omega1_resolved_slip"
    )
    assert registry["citation"]["dataset_doi"] == "10.25534/tudatalib-173"
    assert registry["access"]["license"] == "CC-BY-NC-4.0"
    assert registry["archive"]["size_bytes"] == 6102220
    assert len(registry["selected_series"]) == 4
    assert len(registry["selected_convergence_records"]) == 4
    assert {entry["method"] for entry in registry["selected_series"]} == {
        "interTrackFoam",
        "BoSSS",
        "FS3D",
        "interFoam",
    }


def test_convergence_record_parser_reports_finest_published_difference():
    fetcher = _load_fetcher()
    payload = b"2,4,8\n0.4,0.1,0.025\n3,1,0.3\n"

    summary = fetcher.parse_convergence_record(payload)

    assert summary == {
        "resolution_count": 3,
        "minimum_cells_per_half_gap": 2.0,
        "maximum_compared_cells_per_half_gap": 8.0,
        "finest_compared_maximum_height_error_mm": 0.025,
        "finest_compared_integrated_height_error": 0.3,
    }


@pytest.mark.parametrize(
    "payload, diagnostic",
    [
        (b"2,4,8\n0.4,0.1\n3,1,0.3\n", "unequal lengths"),
        (b"2,4,3\n0.4,0.1,0.2\n3,1,2\n", "do not increase"),
        (b"2,4,8\n0.4,-0.1,0.2\n3,1,2\n", "error is negative"),
    ],
)
def test_convergence_record_parser_rejects_invalid_rows(payload, diagnostic):
    fetcher = _load_fetcher()

    with pytest.raises(ValueError, match=diagnostic):
        fetcher.parse_convergence_record(payload)


def test_curve_parser_preserves_published_time_origin_and_duplicate_rows():
    fetcher = _load_fetcher()
    payload = b"-0.05,9.0\n0.0,10.0\n0.2,12.0\n0.2,12.0\n"

    rows, summary = fetcher.parse_curve(payload)

    assert rows[0] == (-0.05, 9.0)
    assert summary == {
        "row_count": 4,
        "time_start_s": -0.05,
        "time_end_s": 0.2,
        "height_min_mm": 9.0,
        "height_max_mm": 12.0,
        "duplicate_time_count": 1,
    }


@pytest.mark.parametrize(
    "payload, diagnostic",
    [
        (b"0,1\n-0.1,2\n", "time decreases"),
        (b"0,1,2\n1,2,3\n", "two columns"),
        (b"0,nan\n1,2\n", "not finite"),
        (b"header,value\n1,2\n", "not numeric"),
    ],
)
def test_curve_parser_rejects_unusable_reference_rows(payload, diagnostic):
    fetcher = _load_fetcher()

    with pytest.raises(ValueError, match=diagnostic):
        fetcher.parse_curve(payload)


def test_archive_verification_rejects_size_and_hash_drift():
    fetcher = _load_fetcher()
    payload = b"published archive fixture"
    contract = {
        "size_bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "repository_md5": hashlib.md5(
            payload, usedforsecurity=False
        ).hexdigest(),
    }

    fetcher.verify_archive(payload, contract)
    with pytest.raises(ValueError, match="size"):
        fetcher.verify_archive(payload + b"x", contract)
    changed = dict(contract)
    changed["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="SHA-256"):
        fetcher.verify_archive(payload, changed)


def test_reference_output_directory_is_immutable(tmp_path):
    fetcher = _load_fetcher()
    output = tmp_path / "reference"
    summary = {"outcome": "PASS"}

    fetcher.write_outputs(output, {"curve.csv": b"0,1\n"}, summary)

    assert (output / "curve.csv").read_bytes() == b"0,1\n"
    assert json.loads(
        (output / "reference_manifest.json").read_text(encoding="utf-8")
    ) == summary
    with pytest.raises(FileExistsError):
        fetcher.write_outputs(output, {"curve.csv": b"0,2\n"}, summary)
