import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest


def _repository() -> Path:
    return Path(__file__).resolve().parents[1]


def _runner_path() -> Path:
    return (
        _repository()
        / "tests"
        / "cases"
        / "fluid"
        / "run_free_surface_wp9_fitted_ale_qualification.py"
    )


def _matrix_path() -> Path:
    return (
        _repository()
        / "tests"
        / "cases"
        / "fluid"
        / "free_surface_wp9_fitted_ale_qualification_matrix.json"
    )


def _load_runner():
    spec = importlib.util.spec_from_file_location(
        "free_surface_wp9_fitted_ale_qualification_runner",
        _runner_path(),
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _raw_matrix() -> dict:
    return json.loads(_matrix_path().read_text(encoding="utf-8"))


def test_wp9_matrix_bytes_are_exactly_frozen():
    runner = _load_runner()
    digest = hashlib.sha256(_matrix_path().read_bytes()).hexdigest()

    assert digest == runner.EXPECTED_MATRIX_SHA256
    matrix = runner.load_matrix(_matrix_path())
    assert matrix["matrix_id"] == runner.EXPECTED_MATRIX_ID
    assert matrix["status"] == "FROZEN_BEFORE_EXECUTION"


def test_wp9_schema2_claims_only_prescribed_as_consumed_supported_path():
    runner = _load_runner()
    matrix = runner.load_matrix(_matrix_path())
    schema_2 = matrix["configuration_contract"]["schema_2"]

    assert schema_2["accepted_consumed_path"]["tangential_policy"] == ("Prescribed")
    assert schema_2["accepted_consumed_path"]["policy_consumed"] is True
    assert "tangential_policy_Free" in (schema_2["rejected_before_system_mutation"])
    assert (
        "tangential_policy_SmoothingOnly"
        in (schema_2["rejected_before_system_mutation"])
    )
    assert (
        "fitted_DynamicRenE_contact_model"
        in schema_2["rejected_before_system_mutation"]
    )
    assert schema_2["kinematic_penalty_auto_promotes_none"] is False


def test_wp9_schema1_and_unconsumed_provenance_are_explicitly_unqualified():
    runner = _load_runner()
    matrix = runner.load_matrix(_matrix_path())

    schema_1 = matrix["configuration_contract"]["schema_1"]
    assert schema_1["qualification"] == "unqualified_explicit_legacy"
    assert schema_1["explicit_opt_in_required"] is True
    assert schema_1["supported_capability_claimed"] is False
    assert matrix["policy_provenance_contract"]["unconsumed_representation"] == {
        "policy_consumed": False,
        "operator_tag": None,
        "operator_source": None,
    }
    assert (
        matrix["policy_provenance_contract"]["hardcoded_owner_claim_allowed"] is False
    )


def test_wp9_matrix_contains_exact_xml_and_provenance_regressions():
    runner = _load_runner()
    matrix = runner.load_matrix(_matrix_path())
    tests = set(matrix["tests"])

    assert {
        (
            "EquationTranslatorMeshMotion."
            "XmlAliasesReachTangentialPolicyModuleRegistration"
        ),
        (
            "EquationTranslatorFreeSurface."
            "XmlTangentialPenaltyAliasesReachTruthfulFittedModule"
        ),
        (
            "EquationTranslatorFreeSurface."
            "XmlExplicitNoneCannotBePromotedByKinematicPenalty"
        ),
        (
            "EquationTranslatorFreeSurface."
            "XmlFittedDynamicContactFailsClosedBeforeSystemMutation"
        ),
        ("MovingDomainPhysics.FittedFreeSurfaceQualifiedContractRejectsBeforeMutation"),
        (
            "MovingDomainPhysics."
            "FittedFreeSurfaceTangentialPoliciesRegisterCoupledMeshOwnership"
        ),
        (
            "MovingDomainPhysics."
            "FittedFreeSurfaceLegacyPrescribedDataReportsUnconsumedPolicy"
        ),
        (
            "MovingDomainPhysics."
            "NavierStokesEffectiveConfigurationSnapshotExpandsBoundaryDefaults"
        ),
    } <= tests
    assert len(runner._tests_for_binary(matrix, "application")) == 4
    assert len(runner._tests_for_binary(matrix, "physics")) == 26
    dynamic_contact_exit = next(
        entry
        for entry in matrix["unqualified_required_method_exits"]
        if entry["id"]
        == "explicit_fitted_dynamic_contact_rejection_and_capability_provenance"
    )
    assert dynamic_contact_exit["status"] == "REQUIRED_NOT_CLAIMED"
    assert "complete effective capability provenance" in (
        dynamic_contact_exit["contract"]
    )


@pytest.mark.parametrize(
    "claim",
    [
        "fsr10_closure",
        "fsr11_closure",
        "wp9_closure",
        "q4_closure",
        "fitted_ale_qualified",
    ],
)
def test_wp9_rejects_every_premature_closure_claim(claim):
    runner = _load_runner()

    with pytest.raises(ValueError, match="outside this matrix"):
        runner._requested_claim(["--requested-claim", claim])


def test_wp9_rejects_unknown_claim():
    runner = _load_runner()

    with pytest.raises(ValueError, match="unsupported WP-9 requested claim"):
        runner._requested_claim(["--requested-claim", "unregistered_claim"])


def test_wp9_contract_rejects_supported_policy_drift():
    runner = _load_runner()
    mutated = copy.deepcopy(_raw_matrix())
    mutated["current_supported_slice"]["tangential_policy"] = "Free"

    with pytest.raises(ValueError, match="supported slice changed"):
        runner.validate_wp9_contract(mutated)


def test_wp9_contract_rejects_fabricated_owner_provenance():
    runner = _load_runner()
    mutated = copy.deepcopy(_raw_matrix())
    mutated["policy_provenance_contract"]["hardcoded_owner_claim_allowed"] = True

    with pytest.raises(ValueError, match="provenance contract changed"):
        runner.validate_wp9_contract(mutated)


def test_wp9_matrix_byte_drift_is_rejected(tmp_path, monkeypatch):
    runner = _load_runner()
    path = tmp_path / _matrix_path().name
    path.write_bytes(_matrix_path().read_bytes() + b"\n")
    monkeypatch.setattr(runner, "DEFAULT_MATRIX", path)

    with pytest.raises(ValueError, match="frozen matrix bytes changed"):
        runner.load_matrix(path)


def test_wp9_matrix_symlink_alias_is_rejected(tmp_path):
    runner = _load_runner()
    alias = tmp_path / "matrix-alias.json"
    alias.symlink_to(_matrix_path())

    with pytest.raises(ValueError, match="frozen matrix is unavailable"):
        runner.load_matrix(alias)


def test_wp9_loader_rejects_duplicate_json_keys(tmp_path, monkeypatch):
    runner = _load_runner()
    duplicate = tmp_path / _matrix_path().name
    text = (
        _matrix_path()
        .read_text(encoding="utf-8")
        .replace(
            '"schema_version": 1,',
            '"schema_version": 1,\n  "schema_version": 1,',
            1,
        )
    )
    duplicate.write_text(text, encoding="utf-8")
    monkeypatch.setattr(runner, "DEFAULT_MATRIX", duplicate)
    monkeypatch.setattr(
        runner,
        "EXPECTED_MATRIX_SHA256",
        hashlib.sha256(duplicate.read_bytes()).hexdigest(),
    )

    with pytest.raises(ValueError, match="duplicate JSON key: schema_version"):
        runner.load_matrix(duplicate)


def test_wp9_cli_rejects_closure_before_execution_parsing(tmp_path):
    output = tmp_path / "must-not-exist"
    result = subprocess.run(
        [
            sys.executable,
            str(_runner_path()),
            "--requested-claim",
            "q4_closure",
            "--output",
            str(output),
        ],
        cwd=_repository(),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )

    assert result.returncode == 2
    assert "outside this matrix" in result.stderr
    assert not output.exists()


def test_wp9_validate_only_reports_prerequisite_nonclosure():
    result = subprocess.run(
        [sys.executable, str(_runner_path()), "--validate-only"],
        cwd=_repository(),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=True,
    )
    summary = json.loads(result.stdout)

    assert summary["outcome"] == "PASS_PREREQUISITE_NONCLOSURE"
    assert summary["requested_claim"] == "low_level_prerequisite"
    assert summary["test_count"] == 30
    assert summary["application_test_count"] == 4
    assert summary["physics_test_count"] == 26
    assert summary["unqualified_method_exit_count"] == 9
    assert summary["unqualified_simulation_exit_count"] == 3
    assert summary["fsr10_closed"] is False
    assert summary["fsr11_closed"] is False
    assert summary["wp9_closed"] is False
    assert summary["q4_closed"] is False
    assert summary["physical_fitted_ale_qualified"] is False
