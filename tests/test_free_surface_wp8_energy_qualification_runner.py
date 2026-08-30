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
    ROOT / "tests" / "cases" / "fluid" / "run_free_surface_wp8_energy_qualification.py"
)
MATRIX_PATH = RUNNER_PATH.with_name("free_surface_wp8_energy_qualification_matrix.json")


def load_runner():
    specification = importlib.util.spec_from_file_location(
        "free_surface_wp8_energy_qualification_runner",
        RUNNER_PATH,
    )
    module = importlib.util.module_from_spec(specification)
    assert specification.loader is not None
    sys.modules[specification.name] = module
    specification.loader.exec_module(module)
    return module


def raw_matrix():
    return json.loads(MATRIX_PATH.read_text(encoding="utf-8"))


def test_wp8_matrix_bytes_are_exactly_frozen():
    runner = load_runner()
    digest = hashlib.sha256(MATRIX_PATH.read_bytes()).hexdigest()

    assert digest == runner.EXPECTED_REGISTRY_SHA256
    matrix = runner.load_registry(MATRIX_PATH)
    assert matrix["matrix_id"] == "free_surface_wp8_energy_prerequisite_v3"
    assert matrix["status"] == "FROZEN_BEFORE_EXECUTION"


def test_wp8_matrix_has_no_duplicate_json_keys():
    def unique_object(pairs):
        value = {}
        for key, item in pairs:
            assert key not in value, f"duplicate JSON key: {key}"
            value[key] = item
        return value

    json.loads(
        MATRIX_PATH.read_text(encoding="utf-8"),
        object_pairs_hook=unique_object,
    )


def test_wp8_matrix_tests_have_exact_frozen_source_definitions():
    runner = load_runner()
    matrix = runner.load_registry(MATRIX_PATH)

    assert matrix["source_test_files"] == runner.EXPECTED_SOURCE_TEST_FILES
    runner._validate_source_definitions(matrix)


def test_wp8_maintenance_evidence_is_exactly_the_prerequisite_slice():
    runner = load_runner()
    matrix = runner.load_registry(MATRIX_PATH)
    tests = {test for group in matrix["groups"] for test in group["tests"]}

    assert {
        (
            "ApplicationDriverLevelSetWorkflows."
            "MaintenanceWorkLedgerPublishesReinitializationOnlyAtCommit"
        ),
        (
            "ApplicationDriverLevelSetWorkflows."
            "MaintenanceWorkLedgerKeepsReinitializationAndCorrectionAdditive"
        ),
        (
            "ApplicationDriverLevelSetWorkflows."
            "MaintenanceWorkLedgerReportsSameStateAsZeroWork"
        ),
        (
            "ApplicationDriverLevelSetWorkflows."
            "MaintenanceWorkLedgerRollbackPublishesNoAcceptedRow"
        ),
        (
            "ApplicationDriverLevelSetWorkflows."
            "MaintenanceWorkLedgerRejectsDiscontinuousRows"
        ),
        (
            "ApplicationDriverLevelSetWorkflows."
            "MaintenanceWorkLedgerPublishesZeroRowAttemptOutcomes"
        ),
        (
            "ApplicationDriverLevelSetWorkflows."
            "MaintenanceWorkLedgerRequiresExplicitCutTopologyProvenance"
        ),
        (
            "ApplicationDriverConservativePhaseCandidatesTest."
            "StagesAndCommitsTheTransportedPhaseAgainstAuthoritativeGeometry"
        ),
        (
            "ApplicationDriverLevelSetWorkflowsMPI."
            "MaintenanceWorkRowsAreIdenticalAcrossTwoRanks"
        ),
        (
            "ApplicationDriverLevelSetWorkflowsMPI."
            "MaintenanceAlgebraicRevisionRejectsRankLocalSlices"
        ),
        (
            "FreeSurfaceEnergyLedger."
            "CommitsOneCompleteBackwardEulerFixedTopologyBalance"
        ),
        (
            "FreeSurfaceEnergyLedger."
            "AcceptedHistoryRequiresContinuousEndpointAndOwnerProvenance"
        ),
        (
            "FreeSurfaceEnergyLedger."
            "RejectedHistoryAlsoLocksChannelOwnerProvenance"
        ),
        (
            "FreeSurfaceEnergyLedger."
            "UnstagedRejectionPreservesReasonWithoutInventingBalanceValues"
        ),
        (
            "FreeSurfaceEnergyLedger."
            "PartialEndpointProvenanceCannotStageOrClaimTopologyChange"
        ),
        (
            "FreeSurfaceEnergyLedger."
            "TopologyChangeCannotCommitAndRejectedAttemptContributesZero"
        ),
        (
            "FreeSurfaceEnergyLedger."
            "MissingChannelsAndNegativeDissipationFailClosed"
        ),
        (
            "FreeSurfaceEnergyLedger."
            "EveryChannelRequiresOneNamedApplicableOwner"
        ),
        (
            "FreeSurfaceEnergyLedger."
            "RequiresOneEndpointIntervalAndIncreasingTransactionIdentifiers"
        ),
        (
            "FreeSurfaceEnergyLedger."
            "GasApplicabilityMustBeExplicitAndStableAcrossTheStep"
        ),
    } <= tests
    assert len(tests) == 41
    assert {
        (
            "GeneralSimulationParameters."
            "ParsesOptionalTransientTimeIntegrationScheme"
        ),
        (
            "ApplicationDriverLevelSetWorkflows."
            "SelectsBackwardEulerAndRejectsUnsupportedTransientScheme"
        ),
        (
            "TimeLoopConvergence."
            "BackwardEulerExternalStateFixedPointPreservesEndpointTransaction"
        ),
    } <= tests
    assert (
        matrix["current_energy_account_coverage"]["implemented_low_level_channels"]
        == runner.EXPECTED_IMPLEMENTED_ENERGY_CHANNELS
    )
    assert (
        matrix["current_energy_account_coverage"]["not_yet_complete_channels"]
        == runner.EXPECTED_MISSING_ENERGY_CHANNELS
    )


@pytest.mark.parametrize(
    "claim",
    [
        "fsr09_closure",
        "wp8_closure",
        "q3_closure",
        "q4_closure",
        "q5_closure",
        "future_method_closure",
        "complete_energy_law",
    ],
)
def test_wp8_rejects_every_closure_claim(claim):
    runner = load_runner()

    with pytest.raises(ValueError, match="outside this matrix"):
        runner.requested_claim(["--requested-claim", claim])


def test_wp8_rejects_unknown_nonclosure_claim():
    runner = load_runner()

    with pytest.raises(ValueError, match="unsupported WP-8 requested claim"):
        runner.requested_claim(["--requested-claim", "unregistered_prerequisite"])


def test_wp8_contract_rejects_premature_disposition_promotion():
    runner = load_runner()
    matrix = copy.deepcopy(raw_matrix())
    matrix["qualification_disposition"]["wp8_closed"] = True

    with pytest.raises(ValueError, match="nonclosure disposition changed"):
        runner.validate_wp8_contract(matrix)


def test_wp8_transient_scheme_prerequisite_is_exact():
    runner = load_runner()
    matrix = runner.load_registry(MATRIX_PATH)

    assert (
        matrix["transient_scheme_prerequisite"]
        == runner.EXPECTED_TRANSIENT_SCHEME_PREREQUISITE
    )
    assert (
        "backward_euler_constant_surface_tension_closed_balance"
        in runner.EXPECTED_METHOD_EXITS
    )
    assert (
        matrix["qualification_disposition"]["wp8_closed"] is False
    )
    assert matrix["qualification_disposition"]["q4_closed"] is False
    assert matrix["qualification_disposition"]["q5_closed"] is False


def test_wp8_production_source_contract_is_fail_closed(tmp_path):
    runner = load_runner()
    assert runner.validate_wp8_production_source_contract() == {
        "exact_supported_scheme_count": 2,
        "endpoint_stage_sites": 2,
        "optional_scheme_defaults": 1,
        "scheme_guarded_rate_initialization_sites": 1,
    }

    parameters_source = runner.PARAMETERS_SOURCE.read_text(
        encoding="utf-8"
    )
    required_spectral = parameters_source.replace(
        (
            'set_parameter("Spectral_radius_of_infinite_time_step", 0.5, '
            "!required, spectral_radius_of_infinite_time_step);"
        ),
        (
            'set_parameter("Spectral_radius_of_infinite_time_step", 0.5, '
            "required, spectral_radius_of_infinite_time_step);"
        ),
        1,
    )
    assert required_spectral != parameters_source
    mutated_parameters = tmp_path / "Parameters-required.cpp"
    mutated_parameters.write_text(required_spectral, encoding="utf-8")
    runner.PARAMETERS_SOURCE = mutated_parameters
    with pytest.raises(ValueError, match="optional"):
        runner.validate_wp8_production_source_contract()

    runner = load_runner()
    application_source = runner.APPLICATION_DRIVER_SOURCE.read_text(
        encoding="utf-8"
    )
    expanded_scheme_table = application_source.replace(
        (
            "svmp::FE::timestepping::SchemeKind>, 2>\n"
            "    kTransientTimeIntegrationSchemes{{"
        ),
        (
            "svmp::FE::timestepping::SchemeKind>, 3>\n"
            "    kTransientTimeIntegrationSchemes{{"
        ),
        1,
    ).replace(
        (
            '        {"BackwardEuler",\n'
            "         svmp::FE::timestepping::SchemeKind::BackwardEuler},\n"
        ),
        (
            '        {"BackwardEuler",\n'
            "         svmp::FE::timestepping::SchemeKind::BackwardEuler},\n"
            '        {"BDF2",\n'
            "         svmp::FE::timestepping::SchemeKind::BDF2},\n"
        ),
        1,
    )
    assert expanded_scheme_table != application_source
    assert expanded_scheme_table.count('{"BDF2",') == 1
    expanded_scheme_path = tmp_path / "ApplicationDriver-third-scheme.cpp"
    expanded_scheme_path.write_text(
        expanded_scheme_table, encoding="utf-8"
    )
    runner.APPLICATION_DRIVER_SOURCE = expanded_scheme_path
    with pytest.raises(ValueError, match="exactly two"):
        runner.validate_wp8_production_source_contract()

    runner = load_runner()
    mutations = {
        "rho": (
            "selection.generalized_alpha_rho_inf = std::nullopt;",
            "selection.generalized_alpha_rho_inf = 0.5;",
        ),
        "stage": (
            "        .stage_alpha_f = svmp::FE::Real{1.0},",
            "        .stage_alpha_f = svmp::FE::Real{0.5},",
        ),
        "rate": (
            (
                "opts.scheme == "
                "svmp::FE::timestepping::SchemeKind::GeneralizedAlpha &&"
            ),
            "true &&",
        ),
    }
    for label, (before, after) in mutations.items():
        mutated = application_source.replace(before, after, 1)
        assert mutated != application_source
        path = tmp_path / f"ApplicationDriver-{label}.cpp"
        path.write_text(mutated, encoding="utf-8")
        runner.APPLICATION_DRIVER_SOURCE = path
        with pytest.raises(ValueError, match="WP-8 production"):
            runner.validate_wp8_production_source_contract()


def test_wp8_matrix_single_byte_mutation_is_rejected(tmp_path):
    runner = load_runner()
    mutated = bytearray(MATRIX_PATH.read_bytes())
    mutated[0] ^= 1
    path = tmp_path / MATRIX_PATH.name
    path.write_bytes(mutated)

    with pytest.raises(ValueError, match="frozen registry bytes changed"):
        runner.load_registry(path)


def test_wp8_cli_rejects_closure_before_execution_parsing(tmp_path):
    output = tmp_path / "must-not-exist"
    result = subprocess.run(
        [
            sys.executable,
            str(RUNNER_PATH),
            "--requested-claim",
            "q3_closure",
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


def test_wp8_validate_only_reports_prerequisite_nonclosure():
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
    assert summary["test_count"] == 41
    assert summary["group_count"] == 5
    assert summary["serial_quantitative_gate_count"] == 9
    assert summary["unqualified_method_exit_count"] == 10
    assert summary["unqualified_simulation_exit_count"] == 6
    assert (
        summary["matrix_sha256"] == hashlib.sha256(MATRIX_PATH.read_bytes()).hexdigest()
    )
    assert summary["prerequisite_evidence_frozen"] is True
    assert summary["fsr09_closed"] is False
    assert summary["wp8_closed"] is False
    assert summary["q3_closed"] is False
    assert summary["q4_closed"] is False
    assert summary["q5_closed"] is False
    assert summary["complete_energy_law_available"] is False
    assert summary["production_source_contract"] == {
        "exact_supported_scheme_count": 2,
        "endpoint_stage_sites": 2,
        "optional_scheme_defaults": 1,
        "scheme_guarded_rate_initialization_sites": 1,
    }
