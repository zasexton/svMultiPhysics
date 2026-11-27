import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest


def _load_runner():
    repository = Path(__file__).resolve().parents[1]
    script = (
        repository
        / "tests"
        / "cases"
        / "fluid"
        / "run_level_set_phase_transport_release.py"
    )
    spec = importlib.util.spec_from_file_location(
        "level_set_phase_transport_release_runner", script
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _registry():
    repository = Path(__file__).resolve().parents[1]
    path = (
        repository
        / "tests"
        / "cases"
        / "fluid"
        / "level_set_phase_transport_release_matrix.json"
    )
    return json.loads(path.read_text(encoding="utf-8"))


def _write_control_volumes(path, values):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "node,lumped_control_volume,limited_indicator\n"
        + "\n".join(
            f"{node},{weight},{indicator}"
            for node, (weight, indicator) in enumerate(values)
        )
        + "\n",
        encoding="utf-8",
    )


def test_frozen_registry_has_exact_release_cartesian_product(capsys):
    runner = _load_runner()
    registry = _registry()

    assert runner.list_points(registry) == 0
    points = capsys.readouterr().out.splitlines()

    assert len(points) == 18
    assert len(set(points)) == 18
    assert "translating_drop_2d resolution=64 cfl=0.5" in points
    assert "enright_3d resolution=128 cfl=0.125" in points


def test_campaign_cli_accepts_the_frozen_release_inputs_and_rejects_changed_gates(tmp_path):
    scripts = Path(__file__).resolve().parent / "cases" / "fluid"
    command = [sys.executable, str(scripts / "run_level_set_phase_transport_release_matrix.py"), "validate"]
    validation = subprocess.run(command, capture_output=True, text=True)
    assert validation.returncode == 0, validation.stderr
    report = json.loads(validation.stdout)
    assert report["outcome"] == "PASS"
    assert report["expected_point_count"] == 18

    registry = _registry()
    registry["cases"]["enright_3d"]["gates"]["maximum_interface_l1"] = 1.0
    changed = tmp_path / "changed.json"
    changed.write_text(json.dumps(registry), encoding="utf-8")
    rejected = subprocess.run(command + ["--registry", str(changed)], capture_output=True, text=True)
    assert rejected.returncode != 0
    assert "frozen release registry bytes changed" in rejected.stderr


def test_temporal_difference_uses_control_volume_weights(tmp_path):
    runner = _load_runner()
    first = tmp_path / "first.csv"
    second = tmp_path / "second.csv"
    _write_control_volumes(first, [(1.0, 0.0), (3.0, 1.0)])
    _write_control_volumes(second, [(1.0, 1.0), (3.0, 0.5)])

    difference = runner.weighted_control_volume_difference(first, second)

    assert difference == 0.625
    assert runner.observed_order(0.25, 0.125) == 1.0
    assert runner.convergence_uncertainty(0.25, 0.125, 1.0) == 0.15625


def test_checksum_verification_detects_mutation(tmp_path):
    runner = _load_runner()
    artifact = tmp_path / "history.csv"
    artifact.write_text("original\n", encoding="utf-8")
    digest = hashlib.sha256(artifact.read_bytes()).hexdigest()
    (tmp_path / "checksums.txt").write_text(
        f"{digest}  history.csv\n", encoding="utf-8"
    )

    assert runner.verify_checksums(tmp_path) == []
    artifact.write_text("changed\n", encoding="utf-8")
    assert runner.verify_checksums(tmp_path) == [
        "checksum mismatch: history.csv"
    ]


@pytest.mark.parametrize(
    "native_assertion,expected",
    [(True, "FAIL_METHOD"), (False, "INFRASTRUCTURE_FAILURE")],
)
def test_point_run_preserves_a_native_assertion_or_process_failure(tmp_path, native_assertion, expected):
    runner = _load_runner()
    source = tmp_path / "source"
    subprocess.run(["git", "init", "-q", str(source)], check=True)
    subprocess.run(
        ["git", "-C", str(source), "-c", "user.name=Test", "-c", "user.email=test@example.invalid",
         "commit", "--allow-empty", "-qm", "source"],
        check=True,
    )
    binary = tmp_path / "qualification"
    program = f"#!{sys.executable}\nimport json, sys\nfrom pathlib import Path\n"
    if native_assertion:
        suite, name = runner.TEST_FILTER.split(".")
        report = {"testsuites": [{"testsuite": [{
            "classname": suite, "name": name, "status": "RUN", "result": "COMPLETED",
            "failures": [{"failure": "Expected one component, observed three"}],
        }]}]}
        program += (
            "output = next(arg.split('json:', 1)[1] for arg in sys.argv if arg.startswith('--gtest_output=json:'))\n"
            f"Path(output).write_text(json.dumps({report!r}))\n"
            "sys.exit(1)\n"
        )
    else:
        program += "sys.exit(73)\n"
    binary.write_text(program, encoding="utf-8")
    binary.chmod(0o755)
    output = tmp_path / "point"
    args = runner.build_parser().parse_args([
        "run", "--case", "translating_drop_2d", "--resolution", "64", "--cfl", "0.25",
        "--test-binary", str(binary), "--source-root", str(source), "--output-dir", str(output),
    ])
    assert runner.run_point(args, _registry()) == 2
    comparison = json.loads((output / "comparison.json").read_text())
    assert comparison["point_outcome"] == expected
    assert comparison["release_disposition"] == expected
    if native_assertion:
        assert "Expected one component, observed three" in comparison["failure_reason"]
        assert any(check["metric"] == "native_assertions" and not check["passed"]
                   for check in comparison["checks"])


def test_point_gate_requires_metrics_and_nonempty_flux_artifacts(tmp_path):
    runner = _load_runner()
    registry = _registry()
    history = tmp_path / "history.csv"
    details = tmp_path / "checkpoints" / "final"
    history.parent.mkdir(parents=True, exist_ok=True)
    history.write_text("history\n", encoding="utf-8")
    details.mkdir(parents=True)
    for name in ("control_volumes.csv", "edges.csv", "components.csv"):
        (details / name).write_text("ledger\n", encoding="utf-8")
    properties = {
        "matrix_case": "translating_drop_2d",
        "resolution": "64",
        "requested_cfl": "0.5",
        "achieved_graph_cfl": "0.49",
        "maximum_accounted_balance_error": "1e-16",
        "minimum_indicator": "-1e-16",
        "maximum_indicator": "1.0",
        "maximum_local_balance_residual": "1e-20",
        "maximum_raw_measure_error": "0.0",
        "interface_l1": "0.01",
    }

    checks = runner.evaluate_point(
        properties,
        "translating_drop_2d",
        64,
        0.5,
        registry["common"]["gates"],
        registry["cases"]["translating_drop_2d"]["gates"],
        history,
        details,
    )

    assert checks
    assert all(check["passed"] for check in checks)
    # A uniform field with the mass of the translated disk has L1 error about 0.093.
    properties["interface_l1"] = "0.093"
    properties["maximum_indicator"] = "1.1"
    failed = runner.evaluate_point(
        properties,
        "translating_drop_2d",
        64,
        0.5,
        registry["common"]["gates"],
        registry["cases"]["translating_drop_2d"]["gates"],
        history,
        details,
    )
    assert not next(
        check for check in failed if check["metric"] == "maximum_indicator"
    )["passed"]
    assert not next(
        check for check in failed if check["metric"] == "interface_l1"
    )["passed"]
