import copy
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
FLUID_CASES = ROOT / "tests" / "cases" / "fluid"
RUNNER_PATH = (
    FLUID_CASES
    / "run_free_surface_wp3_sharp_boundary_qualification_v2.py"
)
MATRIX_PATH = (
    FLUID_CASES
    / "free_surface_wp3_sharp_boundary_qualification_matrix_v2.json"
)
WP2_RUNNER_PATH = (
    FLUID_CASES / "run_free_surface_wp2_geometry_qualification.py"
)
V1_RUNNER_PATH = (
    FLUID_CASES
    / "run_free_surface_wp3_sharp_boundary_qualification.py"
)
V1_MATRIX_PATH = (
    FLUID_CASES
    / "free_surface_wp3_sharp_boundary_qualification_matrix.json"
)

REMOVED_REJECTION_TEST = (
    "MovingDomainPhysics."
    "NavierStokesUnfittedBoundaryOperatorsRejectCoupledOutflowFamilies"
)
EXPECTED_ADDITIONS = {
    "BoundaryIntegralInput."
    "GeneratedActiveBoundaryValueGradientDryValidationAndHandleIdentity",
    "NavierStokesOutletFactory."
    "GeneratedActiveBoundaryRoutesEveryCoupledBranch",
    "MovingDomainPhysics."
    "NavierStokesUnfittedCoupledOutflowVariantsUseSharpActiveTrace",
    "MovingDomainPhysics."
    "NavierStokesUnfittedCoupledOutflowFamiliesRejectUnsupportedSharpEnvelope",
    "MovingDomainPhysicsMPI."
    "GeneratedActiveCoupledOutflowReductionGradientAndTractionArePartitionIndependent",
}
EXPECTED_GROUP_IDS = [
    "sharp_boundary_geometry_serial",
    "sharp_boundary_assembly_serial",
    "sharp_boundary_systems_serial",
    "sharp_boundary_operators_serial",
    "sharp_boundary_application_serial",
    "sharp_boundary_assembly_mpi",
    "sharp_boundary_operators_mpi",
    "sharp_boundary_structured_mpi",
    "sharp_boundary_coupled_outflow_mpi",
]
EXPECTED_NEW_SERIAL_EVIDENCE = {
    (
        "NavierStokesOutletFactory."
        "GeneratedActiveBoundaryRoutesEveryCoupledBranch",
        "coupled_outflow_generated_trace_variant_count",
    ): ("integer", "equal", 3),
    (
        "NavierStokesOutletFactory."
        "GeneratedActiveBoundaryRoutesEveryCoupledBranch",
        "coupled_outflow_physical_trace_variant_count",
    ): ("integer", "equal", 0),
    (
        "NavierStokesOutletFactory."
        "GeneratedActiveBoundaryRoutesEveryCoupledBranch",
        "coupled_outflow_physical_deployment_variant_count",
    ): ("integer", "equal", 3),
    (
        "MovingDomainPhysics."
        "NavierStokesUnfittedCoupledOutflowVariantsUseSharpActiveTrace",
        "sharp_coupled_outflow_variant_count",
    ): ("integer", "equal", 3),
    (
        "MovingDomainPhysics."
        "NavierStokesUnfittedCoupledOutflowVariantsUseSharpActiveTrace",
        "sharp_coupled_outflow_generated_trace_count",
    ): ("integer", "equal", 3),
    (
        "MovingDomainPhysics."
        "NavierStokesUnfittedCoupledOutflowVariantsUseSharpActiveTrace",
        "sharp_coupled_outflow_whole_face_fallback_count",
    ): ("integer", "equal", 0),
    (
        "MovingDomainPhysics."
        "NavierStokesUnfittedCoupledOutflowVariantsUseSharpActiveTrace",
        "sharp_coupled_outflow_generated_flow_count",
    ): ("integer", "equal", 3),
    (
        "MovingDomainPhysics."
        "NavierStokesUnfittedCoupledOutflowFamiliesRejectUnsupportedSharpEnvelope",
        "sharp_coupled_outflow_envelope_rejection_count",
    ): ("integer", "equal", 2),
}
EXPECTED_COUPLED_MPI_PROPERTIES = {
    "sharp_coupled_outflow_mpi_rank_count": ("integer", "equal", 2),
    "sharp_coupled_outflow_mpi_cell_count": ("integer", "equal", 12),
    "sharp_coupled_outflow_mpi_partition_count": ("integer", "equal", 2),
    "sharp_coupled_outflow_mpi_gradient_probe_count": (
        "integer",
        "equal",
        4,
    ),
    "sharp_coupled_outflow_mpi_dual_marker_contract_count": (
        "integer",
        "equal",
        3,
    ),
    "sharp_coupled_outflow_mpi_rule_count_mismatch": (
        "integer",
        "equal",
        0,
    ),
    "sharp_coupled_outflow_mpi_owner_mismatch_count": (
        "integer",
        "equal",
        0,
    ),
    "sharp_coupled_outflow_mpi_whole_face_fallback_count": (
        "integer",
        "equal",
        0,
    ),
    "sharp_coupled_outflow_mpi_slab_outlet_contributor_count": (
        "integer",
        "equal",
        1,
    ),
    "sharp_coupled_outflow_mpi_round_robin_outlet_contributor_count": (
        "integer",
        "equal",
        2,
    ),
    "sharp_coupled_outflow_mpi_maximum_measure_error": (
        "real",
        "less_than_or_equal",
        1.0e-11,
    ),
    "sharp_coupled_outflow_mpi_maximum_flow_error": (
        "real",
        "less_than_or_equal",
        1.0e-11,
    ),
    "sharp_coupled_outflow_mpi_maximum_gradient_action_error": (
        "real",
        "less_than_or_equal",
        1.0e-11,
    ),
    "sharp_coupled_outflow_mpi_maximum_traction_work_error": (
        "real",
        "less_than_or_equal",
        1.0e-11,
    ),
}


def load_runner():
    specification = importlib.util.spec_from_file_location(
        "free_surface_wp3_sharp_boundary_qualification_runner_v2",
        RUNNER_PATH,
    )
    module = importlib.util.module_from_spec(specification)
    assert specification.loader is not None
    sys.modules[specification.name] = module
    specification.loader.exec_module(module)
    return module


def matrix_document(path=MATRIX_PATH):
    return json.loads(path.read_text(encoding="utf-8"))


def group_tests(matrix):
    return [
        test
        for group in matrix["groups"]
        for test in group["tests"]
    ]


def property_contracts(contracts):
    return {
        contract["property"]: (
            contract["type"],
            contract["relation"],
            contract["threshold"],
        )
        for contract in contracts
    }


def test_wp3_v2_matrix_is_strictly_scoped_and_all_closures_remain_open():
    runner = load_runner()
    matrix = runner.load_registry(MATRIX_PATH)

    assert matrix["schema_version"] == 1
    assert matrix["matrix_id"] == "free_surface_wp3_sharp_boundary_v2"
    assert matrix["status"] == "FROZEN_BEFORE_EXECUTION"
    assert matrix["qualification_scope"] == runner.EXPECTED_SCOPE
    assert matrix["closure_request_policy"] == (
        runner.EXPECTED_CLOSURE_REQUEST_POLICY
    )
    assert matrix["open_outcomes"] == {
        "fsr16": "OPEN",
        "wp3": "OPEN",
        "joint_wp7": "OPEN",
        "q1": "OPEN",
    }
    assert matrix["prospective_tests"] == []
    threshold = matrix["unfrozen_joint_thresholds"]
    assert len(threshold) == 1
    assert threshold[0]["owner"] == "WP-7_joint_cut_stability"
    assert threshold[0]["status"] == "UNFROZEN_NO_BOUND_INVENTED"


@pytest.mark.parametrize(
    "claim",
    ["fsr16_closure", "wp3_closure", "wp7_closure", "q1_closure"],
)
def test_wp3_v2_rejects_every_premature_closure_claim(claim):
    runner = load_runner()

    with pytest.raises(ValueError, match="outside this matrix"):
        runner.requested_claim(["--requested-claim", claim])


def test_wp3_v2_rejects_unknown_claim():
    runner = load_runner()

    with pytest.raises(ValueError, match="unsupported WP-3 v2 requested claim"):
        runner.requested_claim(
            ["--requested-claim", "unregistered_claim"]
        )


def test_wp3_v2_contract_rejects_scope_policy_or_outcome_promotion():
    runner = load_runner()
    matrix = matrix_document()

    promoted_scope = copy.deepcopy(matrix)
    promoted_scope["qualification_scope"] = "WP-3 closed"
    with pytest.raises(ValueError, match="qualification scope changed"):
        runner.validate_wp3_v2_contract(promoted_scope)

    promoted_policy = copy.deepcopy(matrix)
    promoted_policy["closure_request_policy"]["accepted_claim"] = (
        "wp3_closure"
    )
    with pytest.raises(ValueError, match="closure-request policy changed"):
        runner.validate_wp3_v2_contract(promoted_policy)

    promoted_outcome = copy.deepcopy(matrix)
    promoted_outcome["open_outcomes"]["wp3"] = "CLOSED"
    with pytest.raises(ValueError, match="open outcome changed"):
        runner.validate_wp3_v2_contract(promoted_outcome)


def test_wp3_v2_coupled_rcr_families_are_supported_and_fail_closed():
    runner = load_runner()
    matrix = runner.load_registry(MATRIX_PATH)
    contracts = {
        contract["operator"]: contract
        for contract in matrix["operator_disposition_contract"]
    }

    assert set(contracts) == runner.EXPECTED_SUPPORTED_OPERATORS
    assert matrix["unsupported_operator_contract"] == []
    for operator in ("coupled_rcr_outflow", "coupled_rcrcr_outflow"):
        contract = contracts[operator]
        assert (
            contract["cut_active_disposition"]
            == "generated_active_boundary"
        )
        assert contract["full_domain_disposition"] == "physical_boundary"
        assert contract["dry_face_disposition"] == "exact_zero"
        assert contract["missing_sharp_domain_disposition"] == "hard_error"
        assert (
            contract["active_side_reversal"]
            == "complementary_sharp_subset"
        )


def test_wp3_v2_inventory_is_exactly_the_additive_9_group_29_test_slice():
    runner = load_runner()
    matrix = runner.load_registry(MATRIX_PATH)
    v1_matrix = matrix_document(V1_MATRIX_PATH)

    assert [group["id"] for group in matrix["groups"]] == EXPECTED_GROUP_IDS
    for group in matrix["groups"]:
        expected_binary, expected_ranks, expected_tests = (
            runner.EXPECTED_GROUP_TESTS[group["id"]]
        )
        assert group["binary"] == expected_binary
        assert group["mpi_ranks"] == expected_ranks
        assert group["gtest_output_copies"] == 1
        assert tuple(group["tests"]) == expected_tests

    tests = group_tests(matrix)
    v1_tests = set(group_tests(v1_matrix))
    assert len(tests) == 29
    assert len(set(tests)) == 29
    assert len(v1_tests) == 25
    assert REMOVED_REJECTION_TEST in v1_tests
    assert REMOVED_REJECTION_TEST not in tests
    assert set(tests) == (
        (v1_tests - {REMOVED_REJECTION_TEST}) | EXPECTED_ADDITIONS
    )


def test_wp3_v2_new_serial_quantitative_evidence_is_exact():
    runner = load_runner()
    matrix = runner.load_registry(MATRIX_PATH)
    evidence = {
        (entry["test"], entry["property"]): (
            entry["type"],
            entry["relation"],
            entry["threshold"],
        )
        for entry in matrix["quantitative_evidence"]
    }

    assert runner.EXPECTED_NEW_SERIAL_EVIDENCE == (
        EXPECTED_NEW_SERIAL_EVIDENCE
    )
    assert len(matrix["quantitative_evidence"]) == 68
    for key, expected in EXPECTED_NEW_SERIAL_EVIDENCE.items():
        assert evidence[key] == expected
    assert all(
        test != REMOVED_REJECTION_TEST
        for test, _ in evidence
    )


def test_wp3_v2_coupled_mpi_property_contract_is_exact_and_two_rank():
    runner = load_runner()
    matrix = runner.load_registry(MATRIX_PATH)
    groups = {group["id"]: group for group in matrix["groups"]}
    group = groups["sharp_boundary_coupled_outflow_mpi"]

    assert group["binary"] == "physics"
    assert group["mpi_ranks"] == 2
    assert group["gtest_output_copies"] == 1
    assert len(group["recorded_properties"]) == 14
    assert property_contracts(group["recorded_properties"]) == (
        EXPECTED_COUPLED_MPI_PROPERTIES
    )
    assert runner.EXPECTED_MPI_RECORDED_PROPERTIES == (
        EXPECTED_COUPLED_MPI_PROPERTIES
    )
    assert sum(
        len(item.get("recorded_properties", []))
        for item in matrix["groups"]
    ) == 44


def test_wp3_v2_matrix_and_parser_reject_duplicate_json_keys():
    runner = load_runner()
    json.loads(
        MATRIX_PATH.read_text(encoding="utf-8"),
        object_pairs_hook=runner._reject_duplicate_keys,
    )

    with pytest.raises(ValueError, match="duplicate JSON key: status"):
        json.loads(
            '{"status":"FROZEN","status":"EDITED"}',
            object_pairs_hook=runner._reject_duplicate_keys,
        )


def test_wp3_v2_loader_rejects_any_frozen_matrix_byte_change(tmp_path):
    runner = load_runner()

    assert runner.strict_runner.sha256_file(MATRIX_PATH) == (
        runner.EXPECTED_REGISTRY_SHA256
    )
    changed = matrix_document()
    changed["status"] = "EDITED_AFTER_FREEZE"
    changed_path = tmp_path / MATRIX_PATH.name
    changed_path.write_text(
        json.dumps(changed, indent=2) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="frozen matrix bytes changed"):
        runner.load_registry(changed_path)


def test_wp3_v2_runner_depends_directly_on_wp2_and_not_on_v1():
    source = RUNNER_PATH.read_text(encoding="utf-8")
    v1_bytes = V1_RUNNER_PATH.read_bytes()
    runner = load_runner()

    assert runner.WP2_RUNNER_PATH == WP2_RUNNER_PATH
    assert runner.strict_runner.__name__ == (
        "_free_surface_wp3_v2_wp2_base"
    )
    assert runner._shared_load_registry.__module__ == (
        "_free_surface_wp3_v2_wp2_base"
    )
    assert "run_free_surface_wp2_geometry_qualification.py" in source
    assert (
        '"run_free_surface_wp3_sharp_boundary_qualification.py"'
        not in source
    )
    assert (
        '"free_surface_wp3_sharp_boundary_qualification_matrix.json"'
        not in source
    )

    runner.load_registry(MATRIX_PATH)
    assert V1_RUNNER_PATH.read_bytes() == v1_bytes


def test_wp3_v2_cli_rejects_closure_before_execution_argument_parsing(
    tmp_path,
):
    output = tmp_path / "must-not-exist"
    result = subprocess.run(
        [
            sys.executable,
            str(RUNNER_PATH),
            "--requested-claim",
            "wp3_closure",
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


def test_wp3_v2_validate_only_reports_open_9_group_29_test_contract():
    result = subprocess.run(
        [sys.executable, str(RUNNER_PATH), "--validate-only"],
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=True,
    )
    summary = json.loads(result.stdout)

    assert summary == {
        "closure_outcome": "OPEN_LOW_LEVEL_PREREQUISITE",
        "fsr16_closed": False,
        "group_count": 9,
        "joint_wp7_threshold_frozen": False,
        "matrix_id": "free_surface_wp3_sharp_boundary_v2",
        "outcome": "PASS",
        "prospective_test_count": 0,
        "q1_closed": False,
        "quantitative_evidence_gate_count": 68,
        "recorded_property_gate_count": 44,
        "requested_claim": "low_level_prerequisite",
        "status": "FROZEN_BEFORE_EXECUTION",
        "test_count": 29,
        "wp3_closed": False,
    }
