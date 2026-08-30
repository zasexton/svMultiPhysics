import importlib.util
import json
from pathlib import Path
import re
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
RUNNER_PATH = (
    ROOT
    / "tests"
    / "cases"
    / "fluid"
    / "run_free_surface_wp6_conservative_phase_qualification.py"
)
MATRIX_PATH = RUNNER_PATH.with_name(
    "free_surface_wp6_conservative_phase_qualification_matrix.json"
)
WP5_RUNNER_PATH = RUNNER_PATH.with_name(
    "run_free_surface_wp5_contact_line_qualification.py"
)
WP5_MATRIX_PATH = RUNNER_PATH.with_name(
    "free_surface_wp5_contact_line_qualification_matrix.json"
)
APPLICATION_TEST_CMAKE_PATH = (
    ROOT / "Code" / "Source" / "solver" / "Application" / "Tests"
    / "CMakeLists.txt"
)
CONSENSUS_MPI_TEST_PATH = (
    ROOT
    / "Code"
    / "Source"
    / "solver"
    / "Application"
    / "Tests"
    / "Unit"
    / "test_LevelSetMaintenanceTransactionConsensusMPI.cpp"
)
APPLICATION_DRIVER_SOURCE_PATH = (
    ROOT
    / "Code"
    / "Source"
    / "solver"
    / "Application"
    / "Core"
    / "ApplicationDriver.cpp"
)
TIME_LOOP_SOURCE_PATH = (
    ROOT
    / "Code"
    / "Source"
    / "solver"
    / "FE"
    / "TimeStepping"
    / "TimeLoop.cpp"
)
CONSERVATIVE_PHASE_OPERATOR_SOURCE_PATH = (
    ROOT
    / "Code"
    / "Source"
    / "solver"
    / "FE"
    / "LevelSet"
    / "LevelSetConservativePhaseOperator.cpp"
)


def load_runner():
    specification = importlib.util.spec_from_file_location(
        "free_surface_wp6_conservative_phase_qualification_runner",
        RUNNER_PATH,
    )
    module = importlib.util.module_from_spec(specification)
    assert specification.loader is not None
    sys.modules[specification.name] = module
    specification.loader.exec_module(module)
    return module


def test_wp6_shared_runner_state_is_isolated_from_wp5():
    specification = importlib.util.spec_from_file_location(
        "free_surface_wp5_runner_loaded_before_wp6",
        WP5_RUNNER_PATH,
    )
    wp5_runner = importlib.util.module_from_spec(specification)
    assert specification.loader is not None
    sys.modules[specification.name] = wp5_runner
    specification.loader.exec_module(wp5_runner)
    wp5_runner.load_registry(WP5_MATRIX_PATH)

    wp6_runner = load_runner()
    wp6_runner.load_registry(MATRIX_PATH)

    assert wp5_runner.strict_runner is not wp6_runner.strict_runner
    assert wp5_runner.strict_runner.load_registry is wp5_runner.load_registry
    assert wp6_runner.strict_runner.load_registry is wp6_runner.load_registry


def test_wp6_matrix_is_frozen_but_explicitly_incomplete():
    runner = load_runner()
    matrix = runner.load_registry(MATRIX_PATH)

    assert matrix["qualification_scope"] == runner.EXPECTED_SCOPE
    assert matrix["prospective_tests"] == []
    assert len(matrix["unqualified_required_campaigns"]) == 11
    assert all(
        entry["status"] == "REQUIRED_NOT_CLAIMED"
        for entry in matrix["unqualified_required_campaigns"]
    )
    assert matrix["release_matrix_dependency"]["expected_points"] == 18
    assert (
        matrix["release_matrix_dependency"]
        ["low_level_matrix_can_substitute_for_release_matrix"]
        is False
    )
    assert matrix["build_targets"]["timestepping"] == (
        "test_fe_timestepping"
    )
    timeloop_group = next(
        group for group in matrix["groups"]
        if group["id"] == "phase_timeloop_publication_fail_stop_serial"
    )
    assert timeloop_group["binary"] == "timestepping"
    assert timeloop_group["tests"] == [
        "TimeLoopCallbacks."
        "CommitReadyFailureWithSuccessfulDiscardRestoresRateState",
        "TimeLoopCallbacks."
        "CommitReadyFailureWithFailStopDiscardRetainsCandidateRateState"
    ]


def test_wp6_timestepping_binary_key_override_is_scoped():
    runner = load_runner()
    shared_keys = frozenset(
        runner.strict_runner.QUALIFICATION_BINARY_KEYS
    )
    assert shared_keys == runner.SHARED_QUALIFICATION_BINARY_KEYS
    assert "timestepping" not in shared_keys

    runner.load_registry(MATRIX_PATH)
    assert frozenset(
        runner.strict_runner.QUALIFICATION_BINARY_KEYS
    ) == shared_keys

    with runner.wp6_binary_key_scope():
        assert runner.strict_runner.QUALIFICATION_BINARY_KEYS == (
            runner.WP6_QUALIFICATION_BINARY_KEYS
        )
    assert frozenset(
        runner.strict_runner.QUALIFICATION_BINARY_KEYS
    ) == shared_keys


@pytest.mark.parametrize(
    "claim", ["fsr06_closure", "wp6_closure", "q3_closure"]
)
def test_wp6_rejects_every_premature_closure_claim(claim):
    runner = load_runner()

    with pytest.raises(ValueError, match="outside this matrix"):
        runner.requested_claim(["--requested-claim", claim])


def test_wp6_rejects_unknown_claim():
    runner = load_runner()

    with pytest.raises(ValueError, match="unsupported WP-6 requested claim"):
        runner.requested_claim(
            ["--requested-claim", "unregistered_claim"]
        )


def test_wp6_matrix_byte_drift_is_rejected(tmp_path):
    runner = load_runner()
    document = MATRIX_PATH.read_text(encoding="utf-8")
    path = tmp_path / MATRIX_PATH.name
    path.write_text(document + "\n", encoding="utf-8")
    runner.DEFAULT_REGISTRY = path
    runner.strict_runner.DEFAULT_REGISTRY = path

    with pytest.raises(ValueError, match="frozen registry bytes changed"):
        runner.load_registry(path)


def test_wp6_distributed_boundary_separates_targeted_consensus_from_sweeps():
    runner = load_runner()
    matrix = runner.load_registry(MATRIX_PATH)
    groups = {group["id"]: group for group in matrix["groups"]}

    for group_id in (
        "phase_operator_partition_mpi_2",
        "phase_application_collectives_mpi_2",
    ):
        assert groups[group_id]["mpi_ranks"] == 2
        assert groups[group_id]["gtest_output_copies"] == 2
    four_rank_group = groups["phase_maintenance_consensus_mpi_4"]
    assert four_rank_group["mpi_ranks"] == 4
    assert four_rank_group["gtest_output_copies"] == 4
    assert four_rank_group["tests"] == [
        "LevelSetMaintenanceTransactionConsensusMPI."
        "FourRankCommitRejectAndContentRevisionAgreement"
    ]
    assert matrix["known_partition_limit"] == runner.EXPECTED_PARTITION_LIMIT
    assert (
        matrix["known_partition_limit"]["four_rank_disposition"]
        == "LOW_LEVEL_TRANSACTION_CONSENSUS_ONLY"
    )
    assert (
        matrix["known_partition_limit"]
        ["four_or_more_rank_partition_sweeps"]
        == "REQUIRED_NOT_CLAIMED"
    )
    unresolved = {
        entry["id"]: entry["status"]
        for entry in matrix["unqualified_required_campaigns"]
    }
    assert (
        unresolved["four_or_more_rank_partition_sweeps"]
        == "REQUIRED_NOT_CLAIMED"
    )
    assert (
        unresolved[
            "whole_time_loop_and_multi_artifact_cross_resource_atomicity_"
            "with_commit_rollback_and_logging_fault_injection"
        ]
        == "REQUIRED_NOT_CLAIMED"
    )


def test_wp6_consensus_filters_are_exact_and_new_tu_is_rank_generic():
    cmake_text = APPLICATION_TEST_CMAKE_PATH.read_text(encoding="utf-8")
    expected_filters = {
        "LevelSetMaintenanceTransactionConsensusMPI."
        "TwoRankCommitRejectAndContentRevisionAgreement",
        "LevelSetMaintenanceTransactionConsensusMPI."
        "FourRankCommitRejectAndContentRevisionAgreement",
    }
    parsed_filters = re.findall(
        r"--gtest_filter=([^\s)]+)", cmake_text
    )
    transaction_filters = [
        token
        for token in parsed_filters
        if token.startswith(
            "LevelSetMaintenanceTransactionConsensusMPI."
        )
    ]
    assert len(transaction_filters) == 2
    assert set(transaction_filters) == expected_filters
    assert all(
        transaction_filters.count(expected) == 1
        for expected in expected_filters
    )
    assert all("*" not in token and "?" not in token
               for token in transaction_filters)
    assert all(not token.startswith("-") for token in transaction_filters)

    source = CONSENSUS_MPI_TEST_PATH.read_text(encoding="utf-8")
    forbidden = re.compile(
        r"\b(?:ASSERT|EXPECT)_EQ\s*\(\s*(?:size|world_size)\s*,\s*2\s*\)"
        r"|\b(?:size|world_size)\s*(?:==|!=)\s*2\b"
        r"|\brank\s*==\s*1\b"
    )
    assert forbidden.search(source) is None


def test_wp6_matrix_has_no_duplicate_json_keys():
    def reject_duplicate_keys(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate key: {key}")
            result[key] = value
        return result

    json.loads(
        MATRIX_PATH.read_text(encoding="utf-8"),
        object_pairs_hook=reject_duplicate_keys,
    )


def test_wp6_production_source_contract_is_fail_closed(tmp_path):
    runner = load_runner()
    assert runner.validate_wp6_production_source_contract() == {
        "collective_graph_staleness_gates": 1,
        "partition_local_graph_cache_stamps_excluded": 4,
        "schedule_initialization_gates": 4,
        "schedule_first_callback_gates": 2,
        "live_geometry_consensus_sites": 2,
        "publication_ordering_sites": 2,
        "timeloop_fail_stop_disarm_sites": 1,
        "timeloop_attempt_guard_state_domains": 2,
    }

    source = APPLICATION_DRIVER_SOURCE_PATH.read_text(encoding="utf-8")
    accepted_callback_start = (
        "callbacks.on_step_accepted = "
        "[&](svmp::FE::timestepping::TimeHistory& h) {\n"
        "    requireCollectiveLevelSetMaintenanceRequestSchedule("
    )
    assert source.count(accepted_callback_start) == 1
    delayed_accepted_preflight = (
        tmp_path / "ApplicationDriver-delayed-accepted-preflight.cpp"
    )
    delayed_accepted_preflight.write_text(
        source.replace(
            accepted_callback_start,
            accepted_callback_start.replace(
                " {\n    requireCollective",
                " {\n    (void)h;\n    requireCollective",
            ),
            1,
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="first on_step_accepted"):
        runner.validate_wp6_production_source_contract(
            delayed_accepted_preflight
        )

    collective_graph_staleness_gate = (
        "  const bool graph_rebuild_required =\n"
        "      globalAnyBool(!local_graph_is_current, comm);"
    )
    assert source.count(collective_graph_staleness_gate) == 1
    local_graph_staleness_source = (
        tmp_path / "ApplicationDriver-local-graph-staleness.cpp"
    )
    local_graph_staleness_source.write_text(
        source.replace(
            collective_graph_staleness_gate,
            "  const bool graph_rebuild_required = "
            "!local_graph_is_current;",
            1,
        ),
        encoding="utf-8",
    )
    with pytest.raises(
        ValueError, match="collective conservative graph staleness gate"
    ):
        runner.validate_wp6_production_source_contract(
            local_graph_staleness_source
        )

    local_graph_stamp = (
        "  appendMaintenanceScheduleWord("
        "words, graph->geometry_revision);\n"
    )
    replicated_graph_layout = (
        "  appendMaintenanceScheduleWord("
        "words, graph->dof_layout_revision);"
    )
    assert source.count(replicated_graph_layout) == 1
    graph_stamp_source = (
        tmp_path / "ApplicationDriver-local-graph-stamp.cpp"
    )
    graph_stamp_source.write_text(
        source.replace(
            replicated_graph_layout,
            local_graph_stamp + replicated_graph_layout,
            1,
        ),
        encoding="utf-8",
    )
    with pytest.raises(
        ValueError, match="partition-local graph mesh cache stamps"
    ):
        runner.validate_wp6_production_source_contract(graph_stamp_source)

    operator_source = CONSERVATIVE_PHASE_OPERATOR_SOURCE_PATH.read_text(
        encoding="utf-8"
    )
    dof_revision_reduction = (
        "        const auto dof_revision_min = allReduceUnsigned64Min("
    )
    assert operator_source.count(dof_revision_reduction) == 1
    reduced_graph_stamp_source = (
        tmp_path / "LevelSetConservativePhaseOperator-reduced-stamp.cpp"
    )
    reduced_graph_stamp_source.write_text(
        operator_source.replace(
            dof_revision_reduction,
            "        const auto graph_geometry_revision_min = "
            "allReduceUnsigned64Min(\n"
            "            collective, result.geometry_revision);\n"
            + dof_revision_reduction,
            1,
        ),
        encoding="utf-8",
    )
    with pytest.raises(
        ValueError, match="collective normalization"
    ):
        runner.validate_wp6_production_source_contract(
            APPLICATION_DRIVER_SOURCE_PATH,
            TIME_LOOP_SOURCE_PATH,
            reduced_graph_stamp_source,
        )

    required_call = (
        "appendCanonicalLevelSetMaintenanceGeometrySection(\n"
        "              commit_state_words, geometry_state);"
    )
    assert source.count(required_call) == 1
    drifted_source = tmp_path / "ApplicationDriver.cpp"
    drifted_source.write_text(
        source.replace(
            required_call,
            "/* removed live-geometry production gate */",
            1,
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="live-geometry|missing|out-of-order"):
        runner.validate_wp6_production_source_contract(drifted_source)

    fail_stop_source = tmp_path / "ApplicationDriver-fail-stop.cpp"
    fail_stop_source.write_text(
        source.replace(
            "if (publication_began) {",
            "if (publication_began) {\n"
            "          rollbackConservativePhaseCandidate("
            "h, pending_phase_candidate);",
            1,
        ),
        encoding="utf-8",
    )
    with pytest.raises(
        ValueError, match="forbids rollback or ledger rejection"
    ):
        runner.validate_wp6_production_source_contract(fail_stop_source)

    ungated_source = tmp_path / "ApplicationDriver-ungated.cpp"
    reject_call = (
        "rollback_and_reject_pending_phase(\n"
        '                h, "collective_consensus_rejection");'
    )
    assert source.count(reject_call) == 1
    ungated_source.write_text(
        source.replace(
            reject_call,
            "/* removed consensus-reject publication gate */",
            1,
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="missing or out-of-order"):
        runner.validate_wp6_production_source_contract(ungated_source)

    transient_gate = (
        "  requireCollectiveLevelSetMaintenanceRequestSchedule(\n"
        "      level_set_maintenance,\n"
        "      LevelSetMaintenanceScheduleStage::TransientInitialization,\n"
        "      sim.time_history->stepIndex(),\n"
        "      activeFESystemCommunicator(*sim.fe_system));\n"
    )
    assert source.count(transient_gate) == 2
    for gate_index in range(2):
        missing_transient_gate = (
            tmp_path
            / f"ApplicationDriver-missing-transient-gate-{gate_index}.cpp"
        )
        gate_start = -1
        search_start = 0
        for _ in range(gate_index + 1):
            gate_start = source.index(transient_gate, search_start)
            search_start = gate_start + len(transient_gate)
        missing_transient_gate.write_text(
            source[:gate_start]
            + source[gate_start + len(transient_gate) :],
            encoding="utf-8",
        )
        with pytest.raises(ValueError, match="transient initialization"):
            runner.validate_wp6_production_source_contract(
                missing_transient_gate
            )

    time_loop_source = TIME_LOOP_SOURCE_PATH.read_text(encoding="utf-8")
    disarm_sequence = (
        "                            attempt_state.commit();\n"
        "                            throw;"
    )
    assert time_loop_source.count(disarm_sequence) == 1
    missing_timeloop_disarm = (
        tmp_path / "TimeLoop-missing-fail-stop-disarm.cpp"
    )
    missing_timeloop_disarm.write_text(
        time_loop_source.replace(
            disarm_sequence,
            "                            throw;",
            1,
        ),
        encoding="utf-8",
    )
    with pytest.raises(
        ValueError, match="TimeLoop publication fail-stop"
    ):
        runner.validate_wp6_production_source_contract(
            APPLICATION_DRIVER_SOURCE_PATH,
            missing_timeloop_disarm,
        )

    workspace_restore = (
        "            workspace_.static_compatible_pressure_initialized =\n"
        "                static_pressure_initialized_;"
    )
    assert time_loop_source.count(workspace_restore) == 1
    missing_workspace_restore = (
        tmp_path / "TimeLoop-missing-workspace-restore.cpp"
    )
    missing_workspace_restore.write_text(
        time_loop_source.replace(
            workspace_restore,
            "            /* removed one-shot workspace restore */",
            1,
        ),
        encoding="utf-8",
    )
    with pytest.raises(
        ValueError, match="attempt guard rate and workspace"
    ):
        runner.validate_wp6_production_source_contract(
            APPLICATION_DRIVER_SOURCE_PATH,
            missing_workspace_restore,
        )

    guarded_restore_block = (
        "        if (!committed_) {\n"
        "            history_.restoreRateState(snapshot_);\n"
        "            workspace_.static_compatible_pressure_initialized =\n"
        "                static_pressure_initialized_;\n"
        "        }"
    )
    assert time_loop_source.count(guarded_restore_block) == 1
    workspace_restore_outside_guard = (
        tmp_path / "TimeLoop-workspace-restore-outside-guard.cpp"
    )
    workspace_restore_outside_guard.write_text(
        time_loop_source.replace(
            guarded_restore_block,
            "        if (!committed_) {\n"
            "            history_.restoreRateState(snapshot_);\n"
            "        }\n"
            "        workspace_.static_compatible_pressure_initialized =\n"
            "            static_pressure_initialized_;",
            1,
        ),
        encoding="utf-8",
    )
    with pytest.raises(
        ValueError, match="attempt guard rate and workspace"
    ):
        runner.validate_wp6_production_source_contract(
            APPLICATION_DRIVER_SOURCE_PATH,
            workspace_restore_outside_guard,
        )


def test_wp6_cli_rejects_closure_before_execution_argument_parsing(tmp_path):
    output = tmp_path / "must-not-exist"
    result = subprocess.run(
        [
            sys.executable,
            str(RUNNER_PATH),
            "--requested-claim",
            "wp6_closure",
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


def test_wp6_validate_only_reports_prerequisite_without_closure():
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
    assert summary["requested_claim"] == "low_level_prerequisite"
    assert summary["group_count"] == 8
    assert summary["test_count"] == 59
    assert summary["prospective_test_count"] == 0
    assert summary["serial_quantitative_gate_count"] == 40
    assert summary["unqualified_campaign_count"] == 11
    assert summary["release_matrix_expected_points"] == 18
    assert summary["production_source_contract"] == {
        "collective_graph_staleness_gates": 1,
        "partition_local_graph_cache_stamps_excluded": 4,
        "schedule_initialization_gates": 4,
        "schedule_first_callback_gates": 2,
        "live_geometry_consensus_sites": 2,
        "publication_ordering_sites": 2,
        "timeloop_fail_stop_disarm_sites": 1,
        "timeloop_attempt_guard_state_domains": 2,
    }
