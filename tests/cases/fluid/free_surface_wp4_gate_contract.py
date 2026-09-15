"""Prospective, fail-closed WP4 quantity and state evidence assessment."""

import math


VERSION = "wp4_gate_contract_20260906_v1"
IDENTITY_FIELDS = (
    "record_id", "source", "initialization", "accepted_step", "accepted_time",
    "measurement_phase", "state_revision", "geometry_revision",
    "maintenance_revision", "topology_revision", "domain_identity",
)
RESIDUALS = {"pressure_space_relative_distance", "conservative_balance_normalized_imbalance"}
QUANTITIES = RESIDUALS | {"pressure_jump_relative_error", "contact_angle_absolute_error_degrees",
                        "base_radius_relative_error", "apex_height_relative_error",
                        "liquid_volume_relative_error", "parasitic_capillary_number", "kinetic_energy_proxy"}
AXES = {"resolution", "phi_scale", "physical_scale", "time_step", "bulk_redistance_cadence"}
PHASES = {"initial_pressure_fit", "initial_certificate", "accepted_endpoint"}
FIELDS = {
    "initialization", "axis", "quantity", "phase", "dimension", "boundary", "source",
    "norm", "units", "normalization", "justification", "expected_order", "minimum_order",
    "order_tolerance", "algebraic_floor", "floor_justification", "target", "algebraic_bound",
    "maximum_spread", "maximum_value", "reference", "similarity",
}


class ContractError(ValueError):
    pass


def number(value, *, minimum=0.0, positive=False):
    return (isinstance(value, (int, float)) and not isinstance(value, bool)
            and math.isfinite(value) and (value > minimum if positive else value >= minimum))


def text(value):
    return isinstance(value, str) and bool(value.strip())


def disposition(status, *reasons, **details):
    return {"status": status, "reasons": list(reasons), "equilibrium_certified": False, **details}


def aggregate(statuses):
    statuses = list(statuses)
    for status in ("FAIL", "ADDITIONAL_LEVEL_REQUIRED", "INCONCLUSIVE"):
        if status in statuses:
            return status
    return "PASS" if statuses else "INCONCLUSIVE"


def declaration_errors(contract):
    if not isinstance(contract, dict):
        return ["scientific contract is missing"]
    errors = ["unknown contract fields: " + str(sorted(set(contract) - FIELDS - {"field_source"}))] if set(contract) - FIELDS - {"field_source"} else []
    for field in ("quantity", "phase", "boundary", "source", "norm", "units", "justification", "floor_justification"):
        if not text(contract.get(field)):
            errors.append(field + " is missing")
    if contract.get("initialization") not in {"sampled_analytic", "discrete_energy_minimized"}:
        errors.append("unknown initialization")
    if contract.get("axis") not in AXES or contract.get("phase") not in PHASES:
        errors.append("unknown axis/phase dispatch")
    if contract.get("quantity") not in QUANTITIES:
        errors.append("unknown quantity dispatch")
    if (contract.get("phase") == "initial_certificate" and contract.get("initialization") != "discrete_energy_minimized"
            or contract.get("phase") == "initial_pressure_fit" and contract.get("initialization") != "sampled_analytic"
            or contract.get("phase") != "accepted_endpoint" and contract.get("quantity") not in RESIDUALS):
        errors.append("unknown initialization/quantity/phase combination")
    if contract.get("dimension") not in {2, 3} or contract.get("boundary") not in {"closed", "prescribed_contact"}:
        errors.append("unknown dimension/boundary dispatch")
    if not number(contract.get("normalization"), positive=True):
        errors.append("normalization is missing or invalid")
    if not number(contract.get("algebraic_floor")):
        errors.append("algebraic_floor is missing or invalid")
    if not number(contract.get("target")):
        errors.append("target is missing or invalid")
    if contract.get("axis") in {"resolution", "time_step"}:
        for field in ("expected_order", "minimum_order", "order_tolerance"):
            if not number(contract.get(field), positive=True):
                errors.append(field + " is missing or invalid")
        if not errors and contract["minimum_order"] > contract["expected_order"]:
            errors.append("minimum_order exceeds expected_order")
    if contract.get("axis") in {"phi_scale", "physical_scale"}:
        if not number(contract.get("maximum_spread"), positive=True):
            errors.append("maximum_spread is missing or invalid")
    if contract.get("initialization") == "discrete_energy_minimized" and contract.get("quantity") in RESIDUALS:
        if not number(contract.get("algebraic_bound"), positive=True) or contract["algebraic_bound"] > 1e-8:
            errors.append("algebraic_bound must preserve the minimized 1e-8 limit")
    if contract.get("maximum_value") is not None and not number(contract["maximum_value"], positive=True):
        errors.append("maximum_value is invalid")
    return errors


def validate_identity(identity):
    if not isinstance(identity, dict):
        raise ContractError("measurement identity is missing")
    for field in IDENTITY_FIELDS:
        value = identity.get(field)
        if field == "accepted_step":
            valid = isinstance(value, int) and not isinstance(value, bool) and value >= 0
        elif field == "accepted_time":
            valid = number(value)
        else:
            valid = text(value) or (isinstance(value, int) and not isinstance(value, bool) and value >= 0)
        if not valid:
            raise ContractError("measurement identity is missing/invalid: " + field)


def bind_measurements(rows, declarations):
    if not isinstance(rows, list) or len(rows) != len(declarations) or not rows:
        raise ContractError("measurement binding is incomplete")
    by_quantity = {}
    common = None
    for row in rows:
        validate_identity(row)
        identity = {key: row[key] for key in IDENTITY_FIELDS}
        if common is not None and identity != common:
            raise ContractError("measurement identity differs across quantities")
        common = identity
        quantity = row.get("quantity")
        if quantity in by_quantity or not text(quantity) or not number(row.get("value")) or not number(row.get("normalization"), positive=True):
            raise ContractError("measurement binding has duplicate, unknown or negative quantity")
        by_quantity[quantity] = row
    for contract in declarations:
        row = by_quantity.get(contract["quantity"])
        if row is None:
            raise ContractError("measurement binding lacks declared quantity")
        for field, source in (("initialization", "initialization"), ("measurement_phase", "phase"),
                              ("source", "source"), ("norm", "norm"), ("units", "units"),
                              ("normalization", "normalization")):
            if row.get(field) != contract.get(source):
                raise ContractError("measurement binding mismatch: " + field)
        if "field_source" in contract and row.get("field_source") != contract["field_source"]:
            raise ContractError("measurement binding mismatch: field_source")
    return by_quantity


def assess_sequence(contract, samples):
    errors = declaration_errors(contract)
    if errors:
        status = "FAIL" if any(error.startswith("unknown") for error in errors) else "INCONCLUSIVE"
        return disposition(status, *errors)
    if not samples:
        return disposition("INCONCLUSIVE", "required measurements are unavailable")
    try:
        for sample in samples:
            identity = sample.get("identity")
            validate_identity(identity)
            if any(identity[key] != contract[field] for key, field in (
                ("source", "source"), ("initialization", "initialization"), ("measurement_phase", "phase"))):
                raise ContractError("sequence binding source/phase/initialization mismatch")
            if not number(sample.get("value")):
                raise ContractError("negative or nonfinite norm quantity")
    except (ContractError, AttributeError) as error:
        return disposition("FAIL", str(error))
    axis = contract["axis"]
    values = [sample["value"] for sample in samples]
    residual = contract["quantity"] in RESIDUALS
    if residual and contract["initialization"] == "discrete_energy_minimized" and max(values) > contract["algebraic_bound"]:
        return disposition("FAIL", "minimized algebraic equilibrium bound exceeded")
    if contract["quantity"] == "pressure_space_relative_distance" and contract["initialization"] == "sampled_analytic" and max(values) > 1.0:
        return disposition("FAIL", "sampled pressure admission bound exceeded")
    if contract.get("maximum_value") is not None and not (residual and contract["initialization"] == "sampled_analytic"):
        if max(values) > contract["maximum_value"]:
            return disposition("FAIL", "declared physical bound exceeded")
    if axis in {"phi_scale", "physical_scale", "time_step", "bulk_redistance_cadence"}:
        for sample in samples:
            if (not isinstance(sample.get("event_count"), int) or isinstance(sample.get("event_count"), bool)
                    or not isinstance(sample.get("maintenance_schedule"), list)
                    or sample["event_count"] != len(sample["maintenance_schedule"])
                    or not number(sample.get("physical_horizon"), positive=True)
                    or sample["identity"]["accepted_time"] != sample["physical_horizon"]):
                return disposition("FAIL", "actual maintenance events or accepted physical horizon are invalid")
        common_fields = ["physical_horizon"]
        if axis != "physical_scale":
            common_fields += ["mesh_identity"]
        if axis == "phi_scale":
            common_fields += ["geometry_equivalence", "physical_parameters", "maintenance_schedule", "time_step"]
        if axis == "bulk_redistance_cadence":
            common_fields += ["time_step", "physical_parameters"]
        if axis in {"time_step", "bulk_redistance_cadence"}:
            common_fields += ["initial_geometry_identity", "physical_parameters"]
        for field in common_fields:
            if samples[0].get(field) in (None, "") or any(sample.get(field) != samples[0][field] for sample in samples):
                return disposition("FAIL", "fixed-state comparison mismatch: " + field)
    if axis in {"phi_scale", "physical_scale"}:
        if len(samples) < 2:
            return disposition("INCONCLUSIVE", "scaling levels are missing")
        if axis == "physical_scale":
            similarity = contract.get("similarity")
            if not isinstance(similarity, dict) or not all(similarity.get(key) for key in (
                "justification", "geometric_invariants", "dynamic_invariants", "norm_factors")):
                return disposition("INCONCLUSIVE", "physical similarity and norm transformation are undeclared")
            for sample in samples:
                if any(sample.get(key) != similarity[key] for key in ("geometric_invariants", "dynamic_invariants")):
                    return disposition("FAIL", "physical similarity invariants mismatch")
            factors = [similarity["norm_factors"].get(str(sample["level"])) for sample in samples]
            if not all(number(factor, positive=True) for factor in factors):
                return disposition("INCONCLUSIVE", "physical norm transformation is incomplete")
            values = [value * factor for value, factor in zip(values, factors)]
        spread = (max(values) - min(values)) / contract["normalization"]
        return disposition("PASS" if spread <= contract["maximum_spread"] else "FAIL",
                           basis="scaling_invariance", spread=spread,
                           independent_spatial_evidence_required=True)
    target = contract["target"]
    floor = contract["algebraic_floor"]
    if axis in {"time_step", "bulk_redistance_cadence"}:
        reference = contract.get("reference")
        if not isinstance(reference, dict) or not text(reference.get("justification")):
            return disposition("INCONCLUSIVE", "fixed-mesh reference and uncertainty justification are missing")
        if not number(reference.get("value")) or not number(reference.get("uncertainty")):
            return disposition("INCONCLUSIVE", "fixed-mesh reference value or uncertainty is missing")
        for key, field in (("source", "source"), ("phase", "phase"), ("norm", "norm"), ("units", "units"), ("normalization", "normalization")):
            if reference.get(key) != contract[field]:
                return disposition("FAIL", "reference binding mismatch: " + key)
        for key in ("mesh_identity", "initial_geometry_identity", "physical_horizon"):
            if any(sample[key] != reference.get(key) for sample in samples):
                return disposition("FAIL", "fixed-mesh reference mismatch: " + key)
        target = reference["value"]
        floor += reference["uncertainty"]
        if axis == "bulk_redistance_cadence":
            defect = max(abs(value - target) for value in values)
            return disposition("PASS" if defect <= floor else "FAIL", basis="fixed_mesh_maintenance_reference", maximum_defect=defect)
    if axis == "resolution":
        samples = sorted(samples, key=lambda sample: sample["level"])
        levels = [sample["level"] for sample in samples]
        if levels not in ([8, 16, 32], [8, 16, 32, 64]):
            return disposition("INCONCLUSIVE", "required R/h 8,16,32 and conditional64 levels are missing or unresolved")
        values = [sample["value"] for sample in samples]
        if contract["initialization"] == "discrete_energy_minimized" and residual and max(values) <= contract["algebraic_bound"]:
            return disposition("PASS", basis="algebraic_equilibrium", applicable_bound=contract["algebraic_bound"])
    if len(samples) < 3:
        return disposition("INCONCLUSIVE", "three refinement levels are required")
    if max(abs(value - target) for value in values) <= floor:
        return disposition("PASS", basis="declared_uncertainty_floor")
    if not all(number(sample.get("h"), positive=True) for sample in samples):
        return disposition("FAIL", "refinement spacing is invalid")
    pairs = sorted(zip(samples, values), key=lambda pair: pair[0]["h"], reverse=True)
    samples, values = map(list, zip(*pairs))
    spacings = [sample["h"] for sample in samples]
    ratios = [left / right for left, right in zip(spacings, spacings[1:])]
    if any(abs(ratio - 2.0) > 1e-12 for ratio in ratios):
        return disposition("FAIL", "equal-ratio refinement spacing is required")
    differences = [left - right for left, right in zip(values, values[1:])]
    orders = [math.log(abs(left / right), 2) if left * right > 0 else None
              for left, right in zip(differences, differences[1:])]
    order = orders[-1]
    established = order is not None and order > 0 and abs(order - contract["expected_order"]) <= contract["order_tolerance"]
    if len(orders) > 1 and all(value is not None for value in orders[-2:]):
        stable = abs(orders[-1] - orders[-2]) <= contract["order_tolerance"]
        if stable and order < contract["minimum_order"]:
            return disposition("FAIL", "established incompatible convergence rate", observed_order=order)
        established = established and stable
    if not established or order < contract["minimum_order"]:
        return disposition("ADDITIONAL_LEVEL_REQUIRED" if axis == "resolution" and len(samples) == 3 else "INCONCLUSIVE",
                           "asymptotic tail is unresolved", observed_order=order,
                           sample_count=len(samples), samples=samples,
                           monotone_to_reference=all(value > 0 for value in differences),
                           gate_failures=["asymptotic_tail_not_established"])
    denominator = 2.0 ** order - 1.0
    limit = values[-1] + (values[-1] - values[-2]) / denominator
    uncertainty = 1.25 * abs(values[-1] - values[-2]) / denominator
    return disposition("PASS" if abs(limit - target) <= uncertainty + floor else "FAIL",
                       basis="extrapolated_target_consistency", observed_order=order,
                       extrapolated_value=limit, target=target, grid_uncertainty=uncertainty,
                       algebraic_and_reference_uncertainty=floor)
