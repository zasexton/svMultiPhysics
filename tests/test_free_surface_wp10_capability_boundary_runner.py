import copy
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
    / "run_free_surface_wp10_capability_boundary_qualification.py"
)
MATRIX_PATH = RUNNER_PATH.with_name("free_surface_wp10_capability_boundary_matrix.json")


def load_runner():
    specification = importlib.util.spec_from_file_location(
        "free_surface_wp10_capability_boundary_runner",
        RUNNER_PATH,
    )
    module = importlib.util.module_from_spec(specification)
    assert specification.loader is not None
    sys.modules[specification.name] = module
    specification.loader.exec_module(module)
    return module


def canonical_matrix():
    return json.loads(MATRIX_PATH.read_text(encoding="utf-8"))


def validate_mutated_matrix(runner, tmp_path, mutation):
    document = copy.deepcopy(canonical_matrix())
    mutation(document)
    path = tmp_path / "matrix.json"
    path.write_text(json.dumps(document), encoding="utf-8")
    runner.DEFAULT_MATRIX = path
    return runner.validate_matrix(path)


def test_canonical_boundary_is_strict_and_explicitly_open(tmp_path):
    runner = load_runner()
    matrix = runner.validate_matrix(MATRIX_PATH)
    guard = runner.validate_scope_guard_contract(matrix, ROOT)

    assert matrix["status"] == "FROZEN_CAPABILITY_BOUNDARY"
    assert matrix["current_capability_boundary"]["wp10_closure_claimed"] is False
    assert matrix["current_capability_boundary"]["q7_closure_claimed"] is False
    assert all(
        entry["status"] == "REQUIRED_NOT_IMPLEMENTED"
        for entry in matrix["unimplemented_wp10_requirements"]
    )
    assert all(
        entry["status"] == "BLOCKED_BY_MISSING_IMPLEMENTATION"
        for entry in matrix["blocked_wp10_qualification_exits"]
    )
    assert guard["diagnostic"] == ("unsupported_two_phase_or_jump_free_surface_scope")
    assert guard["accepted_case_count"] == 3
    assert guard["rejected_case_count"] == 21
    assert guard["invalid_case_count"] == 2
    assert guard["outcome"] == "PASS"

    with pytest.raises(ValueError, match="scope guard contract changed"):
        validate_mutated_matrix(
            runner,
            tmp_path,
            lambda document: document["scope_guard_contract"].__setitem__(
                "diagnostic", "weakened_scope_diagnostic"
            ),
        )


@pytest.mark.parametrize(
    "claim",
    [
        "fsr08_closure",
        "wp10_closure",
        "q7_closure",
        "incompressible_two_fluid_qualification",
        "gas_sensitive_qualification",
    ],
)
def test_every_unsupported_closure_claim_is_rejected(claim):
    runner = load_runner()
    matrix = runner.validate_matrix(MATRIX_PATH)

    with pytest.raises(ValueError, match="outside this matrix"):
        runner.validate_requested_claim(matrix, claim)


def test_unknown_claim_is_rejected():
    runner = load_runner()
    matrix = runner.validate_matrix(MATRIX_PATH)

    with pytest.raises(ValueError, match="unsupported requested claim"):
        runner.validate_requested_claim(matrix, "unregistered_claim")


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            lambda matrix: matrix["current_capability_boundary"].__setitem__(
                "incompressible_two_fluid_implemented", True
            ),
            "current capability boundary changed",
        ),
        (
            lambda matrix: matrix["unimplemented_wp10_requirements"][0].__setitem__(
                "status", "IMPLEMENTED"
            ),
            "invalid unimplemented WP-10 requirements entry",
        ),
        (
            lambda matrix: matrix["blocked_wp10_qualification_exits"].pop(),
            "blocked WP-10 exits changed",
        ),
        (
            lambda matrix: matrix["blocked_q7_progression"][0].__setitem__(
                "status", "PASS"
            ),
            "blocked Q7 progression changed",
        ),
        (
            lambda matrix: matrix["closure_request_policy"].__setitem__(
                "accepted_claim", "wp10_closure"
            ),
            "closure request policy changed",
        ),
    ],
)
def test_matrix_rejects_premature_scope_promotion(tmp_path, mutation, message):
    runner = load_runner()

    with pytest.raises(ValueError, match=message):
        validate_mutated_matrix(runner, tmp_path, mutation)


def test_source_boundary_rejects_repository_escape(tmp_path):
    runner = load_runner()
    source_root = tmp_path / "source"
    source_root.mkdir()
    (source_root / ".git").mkdir()
    outside = tmp_path / "outside.cpp"
    outside.write_text("required", encoding="utf-8")
    matrix = {
        "source_checks": [
            {
                "id": "escape",
                "path": "../outside.cpp",
                "required_fragments": ["required"],
                "forbidden_fragments": [],
            }
        ]
    }

    with pytest.raises(ValueError, match="repository relative"):
        runner.validate_source_boundary(matrix, source_root)


def test_source_boundary_rejects_symlink_escape(tmp_path):
    runner = load_runner()
    source_root = tmp_path / "source"
    source_root.mkdir()
    (source_root / ".git").mkdir()
    outside = tmp_path / "outside.cpp"
    outside.write_text("required", encoding="utf-8")
    (source_root / "linked.cpp").symlink_to(outside)
    matrix = {
        "source_checks": [
            {
                "id": "linked_escape",
                "path": "linked.cpp",
                "required_fragments": ["required"],
                "forbidden_fragments": [],
            }
        ]
    }

    with pytest.raises(ValueError, match="source-check path is missing"):
        runner.validate_source_boundary(matrix, source_root)

    real_guard = source_root / "real_guard.py"
    real_guard.write_text("guard = True\n", encoding="utf-8")
    linked_guard = source_root / "linked_guard.py"
    linked_guard.symlink_to(real_guard.name)
    guard_matrix = {
        "scope_guard_contract": {
            "path": linked_guard.name,
        }
    }
    with pytest.raises(ValueError, match="symbolic-link component"):
        runner.validate_scope_guard_contract(guard_matrix, source_root)


def test_cli_rejects_closure_before_binary_or_output_validation(tmp_path):
    result = subprocess.run(
        [
            sys.executable,
            str(RUNNER_PATH),
            "--requested-claim",
            "wp10_closure",
            "--physics-binary",
            str(tmp_path / "missing-physics"),
            "--application-binary",
            str(tmp_path / "missing-application"),
            "--output",
            str(tmp_path / "must-not-exist"),
        ],
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )

    assert result.returncode == 2
    assert "outside this matrix" in result.stderr
    assert not (tmp_path / "must-not-exist").exists()


def test_validate_only_reports_boundary_without_claiming_closure():
    result = subprocess.run(
        [sys.executable, str(RUNNER_PATH), "--validate-only"],
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=True,
    )
    summary = json.loads(result.stdout)

    assert summary["outcome"] == "PASS"
    assert summary["requested_claim"] == "one_phase_capability_boundary"
    assert summary["unimplemented_wp10_requirement_count"] == 9
    assert summary["blocked_wp10_exit_count"] == 8
    assert summary["blocked_q7_exit_count"] == 8
    assert summary["scope_guard_accepted_case_count"] == 3
    assert summary["scope_guard_rejected_case_count"] == 21
    assert summary["scope_guard_invalid_case_count"] == 2
    assert summary["wp10_closed"] is False
    assert summary["q7_closed"] is False
