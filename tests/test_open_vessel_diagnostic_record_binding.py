"""Current force/pressure metrics must come from one diagnostic sample."""
import pytest

from tests.test_open_vessel_smoke_gates import _load_smoke_module


def sample(step, *, available=True, pressure=True):
    record = {
        "accepted_step": step,
        "accepted_time": step * 0.1,
        "phase": "accepted_endpoint_after_maintenance",
        "state_revision": 100 + step,
        "geometry_revision": 200 + step,
        "available": int(available),
    }
    if available:
        record.update(conservative_balance_norm=step * 0.2,
                      normalized_imbalance=step * 0.02)
    if pressure:
        record.update(pressure_representability_available=1,
                      pressure_representability_relative_residual=step * 0.01)
    return record


@pytest.mark.parametrize("available,pressure", [(False, False), (False, True)])
def test_current_force_metric_does_not_borrow_an_older_available_state(
        available, pressure):
    smoke = _load_smoke_module()
    records = [sample(1), sample(2, available=available, pressure=pressure)]
    metrics = {}
    smoke.add_diagnostic_metrics(
        metrics, {"free_surface_conservative_balances": records})
    assert metrics["diagnostic_free_surface_conservative_balance_count"] == 2
    assert metrics["diagnostic_free_surface_conservative_balance_available_count"] == 1
    assert "diagnostic_free_surface_conservative_balance_norm" not in metrics
    assert "diagnostic_free_surface_conservative_balance_normalized_imbalance" not in metrics
    if pressure:
        assert metrics["diagnostic_free_surface_pressure_representability_relative_residual"] == 0.02
    else:
        assert "diagnostic_free_surface_pressure_representability_relative_residual" not in metrics


def test_current_pressure_metric_does_not_borrow_an_older_pressure_sample():
    smoke = _load_smoke_module()
    metrics = {}
    smoke.add_diagnostic_metrics(metrics, {"free_surface_conservative_balances": [
        sample(1), sample(2, pressure=False),
    ]})
    assert metrics["diagnostic_free_surface_conservative_balance_norm"] == 0.4
    assert "diagnostic_free_surface_pressure_representability_relative_residual" not in metrics


def test_refresh_removes_previous_scalar_metrics_when_current_sample_is_missing():
    smoke = _load_smoke_module()
    metrics = {}
    smoke.add_diagnostic_metrics(metrics, {"free_surface_conservative_balances": [sample(1)]})
    smoke.add_diagnostic_metrics(metrics, {"free_surface_conservative_balances": [
        sample(2, available=False, pressure=False),
    ]})
    assert "diagnostic_free_surface_conservative_balance_norm" not in metrics
    assert "diagnostic_free_surface_pressure_representability_relative_residual" not in metrics


def test_current_complete_sample_preserves_pairing_and_raw_negative_values():
    smoke = _load_smoke_module()
    current = sample(2)
    current["conservative_balance_norm"] = -0.4
    metrics = {}
    smoke.add_diagnostic_metrics(metrics, {"free_surface_conservative_balances": [
        sample(1), current,
    ]})
    assert metrics["diagnostic_free_surface_conservative_balance_norm"] == -0.4
    assert metrics["diagnostic_free_surface_pressure_representability_relative_residual"] == 0.02
    assert metrics["latest_free_surface_conservative_balance"] is current
