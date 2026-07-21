import hashlib
import importlib.util
import json
from pathlib import Path
import sys


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
