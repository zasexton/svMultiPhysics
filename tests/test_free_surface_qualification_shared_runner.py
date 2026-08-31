import importlib.util
import json
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
RUNNER_PATH = (
    ROOT
    / "tests"
    / "cases"
    / "fluid"
    / "run_free_surface_wp2_geometry_qualification.py"
)
REGISTRY_PATH = RUNNER_PATH.with_name(
    "free_surface_wp2_geometry_qualification_matrix.json"
)
ZERO_GATES = {
    "expected_failures": 0,
    "expected_errors": 0,
    "expected_disabled": 0,
    "expected_skipped": 0,
}


def load_runner():
    specification = importlib.util.spec_from_file_location(
        "free_surface_qualification_shared_runner",
        RUNNER_PATH,
    )
    module = importlib.util.module_from_spec(specification)
    assert specification.loader is not None
    sys.modules[specification.name] = module
    specification.loader.exec_module(module)
    return module


def gtest_document(test_name, properties=None):
    suite, name = test_name.split(".", 1)
    test = {
        "name": name,
        "status": "RUN",
        "result": "COMPLETED",
        "failures": [],
    }
    test.update(properties or {})
    return {
        "tests": 1,
        "failures": 0,
        "errors": 0,
        "disabled": 0,
        "skipped": 0,
        "testsuites": [{"name": suite, "testsuite": [test]}],
    }


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def load_mutated_registry(runner, tmp_path, mutation):
    registry = json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
    mutation(registry)
    path = tmp_path / "registry.json"
    write_json(path, registry)
    runner.DEFAULT_REGISTRY = path
    return runner.load_registry(path)


def test_canonical_geometry_registry_still_validates():
    runner = load_runner()
    registry = runner.load_registry(REGISTRY_PATH)

    assert registry["matrix_id"] == "free_surface_wp2_geometry_v4"


def test_recorded_property_contract_accepts_explicit_test(tmp_path):
    runner = load_runner()

    def mutation(registry):
        group = registry["groups"][0]
        group["recorded_properties"] = [
            {
                "test": group["tests"][0],
                "property": "measured_margin",
                "type": "real",
                "relation": "greater_than",
                "threshold": 0.0,
            }
        ]

    registry = load_mutated_registry(runner, tmp_path, mutation)
    assert registry["groups"][0]["recorded_properties"][0][
        "property"
    ] == "measured_margin"


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("mpi_ranks", True, "positive mpi_ranks"),
        ("gtest_output_copies", True, "gtest_output_copies"),
    ],
)
def test_registry_rejects_boolean_rank_contracts(
    tmp_path, field, value, message
):
    runner = load_runner()

    def mutation(registry):
        registry["groups"][0][field] = value

    with pytest.raises(ValueError, match=message):
        load_mutated_registry(runner, tmp_path, mutation)


def test_registry_rejects_boolean_execution_envelope(tmp_path):
    runner = load_runner()

    def mutation(registry):
        registry["groups"][0]["execution"]["memory_mib"] = True

    with pytest.raises(ValueError, match="memory_mib must be positive"):
        load_mutated_registry(runner, tmp_path, mutation)


def test_registry_rejects_ambiguous_or_reserved_recorded_property(tmp_path):
    runner = load_runner()

    def missing_test(registry):
        registry["groups"][0]["recorded_properties"] = [
            {
                "property": "margin",
                "type": "real",
                "relation": "greater_than",
                "threshold": 0.0,
            }
        ]

    with pytest.raises(ValueError, match="must name the test"):
        load_mutated_registry(runner, tmp_path / "first", missing_test)

    def reserved_property(registry):
        group = registry["groups"][0]
        group["recorded_properties"] = [
            {
                "test": group["tests"][0],
                "property": "status",
                "type": "integer",
                "relation": "equal",
                "threshold": 1,
            }
        ]

    with pytest.raises(ValueError, match="invalid recorded property name"):
        load_mutated_registry(
            runner, tmp_path / "second", reserved_property
        )


def test_group_recorded_properties_support_serial_and_rank_zero_mpi(tmp_path):
    runner = load_runner()
    serial_test = "SerialSuite.Measures"
    mpi_test = "DistributedSuite.Measures"
    registry = {
        "groups": [
            {
                "id": "serial",
                "mpi_ranks": 1,
                "tests": [serial_test],
                "recorded_properties": [
                    {
                        "property": "count",
                        "type": "integer",
                        "relation": "equal",
                        "threshold": 4,
                    }
                ],
            },
            {
                "id": "distributed",
                "mpi_ranks": 2,
                "tests": [mpi_test],
                "recorded_properties": [
                    {
                        "test": mpi_test,
                        "property": "margin",
                        "type": "real",
                        "relation": "greater_than_or_equal",
                        "threshold": 0.25,
                    }
                ],
            },
        ]
    }
    write_json(
        tmp_path / "groups" / "serial" / "gtest.json",
        gtest_document(serial_test, {"count": "4"}),
    )
    write_json(
        tmp_path / "groups" / "distributed" / "gtest_rank_0.json",
        gtest_document(mpi_test, {"margin": "2.5e-1"}),
    )

    result = runner.evaluate_group_recorded_properties(registry, tmp_path)

    assert result["outcome"] == "PASS"
    assert result["declared_check_count"] == 2
    assert result["passed_check_count"] == 2
    checks = {check["property"]: check for check in result["checks"]}
    assert checks["count"]["actual"] == 4
    assert checks["count"]["result_rank"] is None
    assert checks["margin"]["actual"] == pytest.approx(0.25)
    assert checks["margin"]["result_rank"] == 0
    assert checks["margin"]["gtest_result"].endswith(
        "gtest_rank_0.json"
    )


@pytest.mark.parametrize(
    ("properties", "diagnostic"),
    [
        ({}, "property_missing"),
        ({"margin": "not-a-number"}, "property_type_mismatch"),
        ({"margin": "nan"}, "property_value_not_finite"),
        ({"margin": "0.1"}, "relation_not_satisfied"),
    ],
)
def test_group_recorded_property_failures_are_explicit(
    tmp_path, properties, diagnostic
):
    runner = load_runner()
    test_name = "Suite.Sample"
    registry = {
        "groups": [
            {
                "id": "serial",
                "mpi_ranks": 1,
                "tests": [test_name],
                "recorded_properties": [
                    {
                        "property": "margin",
                        "type": "real",
                        "relation": "greater_than",
                        "threshold": 0.2,
                    }
                ],
            }
        ]
    }
    write_json(
        tmp_path / "groups" / "serial" / "gtest.json",
        gtest_document(test_name, properties),
    )

    result = runner.evaluate_group_recorded_properties(registry, tmp_path)

    assert result["outcome"] == "FAIL_METHOD"
    assert result["checks"][0]["diagnostic"] == diagnostic
    assert result["checks"][0]["passed"] is False


def test_mpi_gtest_results_require_valid_complete_file_per_rank(tmp_path):
    runner = load_runner()
    test_name = "DistributedSuite.Sample"
    group_directory = tmp_path / "groups" / "distributed"
    for rank in range(2):
        write_json(
            group_directory / f"gtest_rank_{rank}.json",
            gtest_document(test_name),
        )

    checks, records = runner.evaluate_mpi_gtest_results(
        group_directory, [test_name], 2, ZERO_GATES
    )
    assert records == [
        {
            **records[0],
            "rank": 0,
            "present": True,
            "valid": True,
            "error": None,
        },
        {
            **records[1],
            "rank": 1,
            "present": True,
            "valid": True,
            "error": None,
        },
    ]
    assert all(check["passed"] for check in checks)

    (group_directory / "gtest_rank_1.json").unlink()
    checks, records = runner.evaluate_mpi_gtest_results(
        group_directory, [test_name], 2, ZERO_GATES
    )
    assert not all(check["passed"] for check in checks)
    assert records[1]["present"] is False
    assert records[1]["valid"] is False


def test_mpi_group_launch_assigns_unique_result_path_per_rank(
    tmp_path, monkeypatch
):
    runner = load_runner()
    test_name = "DistributedSuite.Sample"
    group = {
        "id": "distributed",
        "binary": "assembly_mpi",
        "mpi_ranks": 2,
        "gtest_output_copies": 2,
        "tests": [test_name],
        "execution": {
            "wall_time_seconds": 60,
            "memory_mib": 256,
            "output_mib": 16,
        },
    }
    captured = {}

    def fake_run_monitored(
        command,
        environment,
        source_root,
        stdout_path,
        stderr_path,
        group_directory,
        *args,
        **kwargs,
    ):
        captured["command"] = command
        assert environment["OMP_NUM_THREADS"] == "1"
        assert source_root == ROOT
        stdout_path.write_text(
            "".join(
                f"[ RUN      ] {test_name}\n[       OK ] {test_name}\n"
                for _ in range(2)
            ),
            encoding="utf-8",
        )
        stderr_path.write_text("", encoding="utf-8")
        for rank in range(2):
            write_json(
                group_directory / f"gtest_rank_{rank}.json",
                gtest_document(test_name),
            )
        return {
            "return_code": 0,
            "termination_reason": None,
            "resource_monitoring_outcome": "PASS",
            "termination": None,
        }

    monkeypatch.setattr(runner, "run_monitored", fake_run_monitored)
    output_root = tmp_path / "artifacts"
    output_root.mkdir()

    result = runner.run_gtest_group(
        group,
        ZERO_GATES,
        {"assembly_mpi": Path("/tmp/test-binary")},
        Path("/usr/bin/mpiexec"),
        ROOT,
        output_root,
    )

    assert result["outcome"] == "PASS"
    assert result["diagnostic"] is None
    assert result["mpi_gtest_results"][0]["gtest_result"].endswith(
        "gtest_rank_0.json"
    )
    assert result["mpi_gtest_results"][1]["gtest_result"].endswith(
        "gtest_rank_1.json"
    )
    command = captured["command"]
    assert command[:4] == ["/usr/bin/mpiexec", "--oversubscribe", "-n", "2"]
    assert "/bin/sh" in command
    assert "gtest_rank_${rank_value}.json" in command[6]


def test_output_envelope_does_not_limit_runtime_backing_files(tmp_path):
    runner = load_runner()
    output_directory = tmp_path / "qualification-output"
    output_directory.mkdir()
    stdout_path = output_directory / "stdout.txt"
    stderr_path = output_directory / "stderr.txt"
    runtime_backing = tmp_path / "runtime-backing.bin"
    child = (
        "from pathlib import Path; import sys,time; "
        "Path(sys.argv[1]).write_bytes(b'x' * (2 * 1024 * 1024)); "
        "time.sleep(0.2)"
    )

    result = runner.run_monitored(
        [sys.executable, "-c", child, str(runtime_backing)],
        runner.os.environ.copy(),
        ROOT,
        stdout_path,
        stderr_path,
        output_directory,
        wall_time_seconds=10,
        memory_mib=256,
        output_mib=1,
        launch_mode="direct_serial",
    )

    assert result["return_code"] == 0
    assert result["termination_reason"] is None
    assert runtime_backing.stat().st_size == 2 * 1024 * 1024
    assert result["final_output_bytes"] < 1024 * 1024
    assert result["memory_enforcement_method"] == (
        "per_process_address_space_limit_and_sampled_session_resident_memory"
    )
    assert result["output_enforcement_method"] == (
        "sampled_output_directory_size"
    )
    assert result["process_file_size_limit_applied"] is False


def test_output_envelope_still_stops_qualification_output_growth(tmp_path):
    runner = load_runner()
    output_directory = tmp_path / "qualification-output"
    output_directory.mkdir()
    stdout_path = output_directory / "stdout.txt"
    stderr_path = output_directory / "stderr.txt"
    oversized_output = output_directory / "oversized.bin"
    child = (
        "from pathlib import Path; import sys,time; "
        "Path(sys.argv[1]).write_bytes(b'x' * (2 * 1024 * 1024)); "
        "time.sleep(5)"
    )

    result = runner.run_monitored(
        [sys.executable, "-c", child, str(oversized_output)],
        runner.os.environ.copy(),
        ROOT,
        stdout_path,
        stderr_path,
        output_directory,
        wall_time_seconds=10,
        memory_mib=256,
        output_mib=1,
        launch_mode="direct_serial",
    )

    assert result["termination_reason"] == "output_envelope_exceeded"
    assert result["final_output_bytes"] > 1024 * 1024
    assert result["termination"]["all_session_processes_terminated"] is True


def test_serial_group_accepts_explicitly_executed_disabled_test(
    tmp_path,
    monkeypatch,
):
    runner = load_runner()
    test_name = "DiagnosticSuite.DISABLED_SelectedDiagnostic"
    group = {
        "id": "selected_disabled",
        "binary": "physics",
        "mpi_ranks": 1,
        "gtest_output_copies": 1,
        "gtest_also_run_disabled_tests": True,
        "tests": [test_name],
        "execution": {
            "wall_time_seconds": 60,
            "memory_mib": 256,
            "output_mib": 16,
        },
    }
    captured = {}

    def fake_run_monitored(
        command,
        _environment,
        _source_root,
        stdout_path,
        stderr_path,
        group_directory,
        *_arguments,
        **_options,
    ):
        captured["command"] = command
        stdout_path.write_bytes(b"")
        stderr_path.write_bytes(b"")
        document = gtest_document(test_name)
        document["disabled"] = 1
        document["testsuites"][0]["disabled"] = 1
        write_json(group_directory / "gtest.json", document)
        return {
            "return_code": 0,
            "termination_reason": None,
            "resource_monitoring_outcome": "PASS",
            "termination": None,
        }

    monkeypatch.setattr(runner, "run_monitored", fake_run_monitored)
    output_root = tmp_path / "artifacts"
    output_root.mkdir()

    result = runner.run_gtest_group(
        group,
        ZERO_GATES,
        {"physics": Path("/tmp/test-binary")},
        Path("/usr/bin/mpiexec"),
        ROOT,
        output_root,
    )

    checks = {check["metric"]: check for check in result["checks"]}
    assert result["outcome"] == "PASS"
    assert "--gtest_also_run_disabled_tests" in captured["command"]
    assert checks["reported_disabled_count"]["actual"] == 1
    assert checks["disabled_count"]["actual"] == 0
    assert checks[
        "explicitly_enabled_disabled_tests_executed"
    ]["passed"]


def test_explicit_disabled_test_must_have_completed(tmp_path):
    runner = load_runner()
    test_name = "DiagnosticSuite.DISABLED_SelectedDiagnostic"
    document = gtest_document(
        test_name,
        {"status": "NOTRUN", "result": "SUPPRESSED"},
    )
    document["disabled"] = 1

    checks = runner.evaluate_serial_result(
        [test_name],
        document,
        0,
        None,
        ZERO_GATES,
        {test_name},
    )
    by_metric = {check["metric"]: check for check in checks}

    assert not by_metric["disabled_count"]["passed"]
    assert not by_metric[
        "explicitly_enabled_disabled_tests_executed"
    ]["passed"]
    assert not by_metric["incomplete_or_skipped_tests"]["passed"]


def test_explicit_disabled_test_rejects_unexpected_disabled_result():
    runner = load_runner()
    expected_name = "DiagnosticSuite.DISABLED_SelectedDiagnostic"
    unexpected_name = "DiagnosticSuite.DISABLED_UnexpectedDiagnostic"
    document = {
        "tests": 2,
        "failures": 0,
        "errors": 0,
        "disabled": 2,
        "skipped": 0,
        "testsuites": [
            {
                "name": "DiagnosticSuite",
                "testsuite": [
                    {
                        "name": expected_name.split(".", 1)[1],
                        "status": "RUN",
                        "result": "COMPLETED",
                        "failures": [],
                    },
                    {
                        "name": unexpected_name.split(".", 1)[1],
                        "status": "RUN",
                        "result": "COMPLETED",
                        "failures": [],
                    },
                ],
            }
        ],
    }

    checks = runner.evaluate_serial_result(
        [expected_name],
        document,
        0,
        None,
        ZERO_GATES,
        {expected_name},
    )
    by_metric = {check["metric"]: check for check in checks}

    assert not by_metric["reported_disabled_count"]["passed"]
    assert not by_metric["disabled_count"]["passed"]
    assert not by_metric["unexpected_disabled_tests"]["passed"]
    assert by_metric["unexpected_disabled_tests"]["actual"] == [
        unexpected_name
    ]
