import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
RUNNER_PATH = (
    ROOT
    / "tests"
    / "cases"
    / "fluid"
    / "run_free_surface_wp5_contact_line_qualification.py"
)
MATRIX_PATH = RUNNER_PATH.with_name(
    "free_surface_wp5_contact_line_qualification_matrix.json"
)
BINARY_FLAGS = {
    "geometry": "--geometry-binary",
    "level_set": "--level-set-binary",
    "physics": "--physics-binary",
    "application": "--application-binary",
    "assembly_mpi": "--assembly-mpi-binary",
    "application_mpi": "--application-mpi-binary",
}
EXPECTED_BINARY_TEST_COUNTS = {
    "geometry": 1,
    "level_set": 7,
    "physics": 24,
    "application": 7,
    "assembly_mpi": 2,
    "application_mpi": 2,
}


def _load_runner():
    specification = importlib.util.spec_from_file_location(
        "free_surface_wp5_contact_line_qualification_runner",
        RUNNER_PATH,
    )
    module = importlib.util.module_from_spec(specification)
    assert specification.loader is not None
    sys.modules[specification.name] = module
    specification.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def runner():
    return _load_runner()


def _raw_matrix() -> dict:
    return json.loads(MATRIX_PATH.read_text(encoding="utf-8"))


def _write_gtest_lister(path: Path, tests: list[str]) -> None:
    suites: dict[str, list[str]] = {}
    for test in tests:
        suite, name = test.split(".", 1)
        suites.setdefault(suite, []).append(name)
    listing = "".join(
        f"{suite}.\n" + "".join(f"  {name}\n" for name in names)
        for suite, names in suites.items()
    )
    path.write_text(
        "#!/usr/bin/env python3\n"
        "import sys\n"
        "if sys.argv[1:] != ['--gtest_list_tests']:\n"
        "    raise SystemExit(3)\n"
        f"sys.stdout.write({listing!r})\n",
        encoding="utf-8",
    )
    path.chmod(0o755)


def test_wp5_matrix_digest_and_canonical_path_are_exact(runner, tmp_path):
    digest = hashlib.sha256(MATRIX_PATH.read_bytes()).hexdigest()

    assert digest == runner.EXPECTED_REGISTRY_SHA256
    matrix = runner.load_registry(MATRIX_PATH)
    assert matrix["matrix_id"] == "free_surface_wp5_contact_line_v1"
    assert matrix["status"] == "FROZEN_BEFORE_EXECUTION"

    copied = tmp_path / MATRIX_PATH.name
    copied.write_bytes(MATRIX_PATH.read_bytes())
    with pytest.raises(ValueError, match="canonical frozen matrix"):
        runner.load_registry(copied)

    alias = tmp_path / "matrix-alias.json"
    alias.symlink_to(MATRIX_PATH)
    with pytest.raises(ValueError, match="matrix is unavailable"):
        runner.load_registry(alias)


def test_wp5_matrix_byte_drift_is_rejected(runner, tmp_path, monkeypatch):
    changed = tmp_path / MATRIX_PATH.name
    changed.write_bytes(MATRIX_PATH.read_bytes() + b"\n")
    monkeypatch.setattr(runner, "DEFAULT_REGISTRY", changed)

    with pytest.raises(ValueError, match="frozen matrix bytes changed"):
        runner.load_registry(changed)


def test_wp5_loader_rejects_duplicate_json_keys(
    runner,
    tmp_path,
    monkeypatch,
):
    duplicate = tmp_path / MATRIX_PATH.name
    text = MATRIX_PATH.read_text(encoding="utf-8").replace(
        '"schema_version": 1,',
        '"schema_version": 1,\n  "schema_version": 1,',
        1,
    )
    duplicate.write_text(text, encoding="utf-8")
    monkeypatch.setattr(runner, "DEFAULT_REGISTRY", duplicate)
    monkeypatch.setattr(
        runner,
        "EXPECTED_REGISTRY_SHA256",
        hashlib.sha256(duplicate.read_bytes()).hexdigest(),
    )

    with pytest.raises(ValueError, match="duplicate JSON key: schema_version"):
        runner.load_registry(duplicate)


def test_wp5_contract_rejects_closure_state_promotion(runner):
    promoted = copy.deepcopy(_raw_matrix())
    promoted["closure_state"] = "CLOSED"

    with pytest.raises(ValueError, match="closure state must remain open"):
        runner.validate_wp5_contract(promoted)


def test_wp5_contract_rejects_disposition_promotion(runner):
    promoted = copy.deepcopy(_raw_matrix())
    promoted["qualification_disposition"]["wp5_closed"] = True

    with pytest.raises(
        ValueError,
        match="qualification disposition must remain open",
    ):
        runner.validate_wp5_contract(promoted)


def test_wp5_cli_rejects_closure_before_artifact_claim(tmp_path):
    output = tmp_path / "must-not-exist"
    result = subprocess.run(
        [
            sys.executable,
            str(RUNNER_PATH),
            "--requested-claim",
            "wp5_closure",
            "--output",
            str(output),
        ],
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )

    assert result.returncode == 2
    assert "outside this matrix" in result.stderr
    assert not output.exists()


def test_wp5_validate_only_reports_exact_nonclosure_counts():
    result = subprocess.run(
        [sys.executable, str(RUNNER_PATH), "--validate-only"],
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=True,
    )
    summary = json.loads(result.stdout)

    assert summary["outcome"] == "PASS_PREREQUISITE_NONCLOSURE"
    assert summary["requested_claim"] == "low_level_prerequisite"
    assert (
        summary["matrix_sha256"] == hashlib.sha256(MATRIX_PATH.read_bytes()).hexdigest()
    )
    assert summary["prospective_test_count"] == 0
    assert summary["group_count"] == 6
    assert summary["test_count"] == 43
    assert summary["serial_quantitative_gate_count"] == 8
    assert summary["unqualified_campaign_count"] == 7
    assert summary["fsr04_closed"] is False
    assert summary["fsr05_closed"] is False
    assert summary["wp5_closed"] is False
    assert summary["q4_closed"] is False


def test_wp5_list_only_discovers_every_frozen_test_exactly(
    runner,
    tmp_path,
):
    matrix = runner.load_registry(MATRIX_PATH)
    arguments = [
        sys.executable,
        str(RUNNER_PATH),
        "--list-only",
    ]
    for key, flag in BINARY_FLAGS.items():
        binary = tmp_path / f"{key}-gtest"
        _write_gtest_lister(
            binary,
            runner._tests_for_binary(matrix, key),
        )
        arguments.extend([flag, str(binary)])

    result = subprocess.run(
        arguments,
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=True,
    )
    summary = json.loads(result.stdout)

    assert summary["outcome"] == "PASS_PREREQUISITE_NONCLOSURE"
    assert summary["expected_test_count_by_binary"] == (EXPECTED_BINARY_TEST_COUNTS)
    assert summary["listed_expected_test_count"] == (EXPECTED_BINARY_TEST_COUNTS)
    assert summary["listed_total_test_count"] == (EXPECTED_BINARY_TEST_COUNTS)
    assert summary["missing_tests"] == {key: [] for key in EXPECTED_BINARY_TEST_COUNTS}
    assert summary["tests_executed"] == 0
    assert summary["artifacts_written"] == 0
