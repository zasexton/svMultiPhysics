# Open-Vessel Pressure Stabilization Contribution Audit

Date: 2026-06-05

Scope: targeted progress on the Test02/Test10 root-cause goal. This note documents an offline cut-adjacent pressure ghost-penalty contribution proxy added after the accepted pressure-update guard. It is not a qualification closure.

## What Changed

Added `tests/cases/fluid/open_vessel_free_surface/audit_pressure_stabilization_contribution.py`.

The diagnostic reconstructs active cut-adjacent interior faces from saved tetrahedral VTU output:

- a face is included when both neighboring cells have positive `WetVolumeFraction` and at least one neighboring cell is cut;
- the pressure-gradient jump is reconstructed from saved P1 `Pressure` values on the tetrahedra;
- the first-derivative pressure ghost-penalty proxy uses the same `h^3 / mu` structure and pressure penalty coefficient as the solver;
- metadata scaling is reconstructed from `WetVolumeFraction`, including XML `Use_cut_metadata_scale` and `Cut_cell_metadata_scale_cap`;
- the report ranks faces by current pressure-gradient jump energy proxy and accepted pressure-increment jump energy proxy;
- the report correlates the worst active/wet pressure-update point with incident reconstructed cut-adjacent faces.

Added `tests/test_open_vessel_pressure_stabilization_audit.py` for tetrahedral gradient recovery, active cut-adjacent face filtering, and metadata scale application.

Limitations: this is an offline proxy, not an exact assembled residual dump. It uses saved P1 tetrahedral output, reconstructs the first-derivative `jump(grad(p))` term, and omits high-order Hessian terms not present in the saved linear tetrahedral field.

## Verification

Focused checks passed:

```bash
python -m py_compile tests/cases/fluid/open_vessel_free_surface/audit_pressure_stabilization_contribution.py
pytest -q tests/test_open_vessel_pressure_stabilization_audit.py
```

Pytest result: `3 passed in 0.55s`.

## Diagnostic Runs

Test02 terminal transition:

```bash
python tests/cases/fluid/open_vessel_free_surface/audit_pressure_stabilization_contribution.py \
  --previous-result Documentation/qualification_logs/open_vessel_free_surface_remaining_20260526/test02_structured_h0p15_all_solid_normal_only_abs_only_prune1e5_0p54_case/result_382.vtu \
  --current-result Documentation/qualification_logs/open_vessel_free_surface_remaining_20260526/test02_structured_h0p15_all_solid_normal_only_abs_only_prune1e5_0p54_case/result_383.vtu \
  --solver-xml Documentation/qualification_logs/open_vessel_free_surface_remaining_20260526/test02_structured_h0p15_all_solid_normal_only_abs_only_prune1e5_0p54_case/solver.xml \
  --top-faces 20 \
  --json-output Documentation/qualification_logs/open_vessel_free_surface_remaining_20260526/test02_abs_only_prune1e5_0p54_pressure_stabilization_contribution_audit_20260605.json
```

Test10 step90 transition:

```bash
python tests/cases/fluid/open_vessel_free_surface/audit_pressure_stabilization_contribution.py \
  --previous-result Documentation/qualification_logs/open_vessel_free_surface_remaining_20260526/test10_roll_full_source_dt0p01_tightvol_adaptive_relaxed_ls_max20_metadata_cap3_1s_case/result_090.vtu \
  --current-result Documentation/qualification_logs/open_vessel_free_surface_remaining_20260526/test10_roll_full_source_dt0p01_tightvol_adaptive_relaxed_ls_max20_metadata_cap3_1s_case/result_091.vtu \
  --solver-xml Documentation/qualification_logs/open_vessel_free_surface_remaining_20260526/test10_roll_full_source_dt0p01_tightvol_adaptive_relaxed_ls_max20_metadata_cap3_1s_case/solver.xml \
  --top-faces 20 \
  --json-output Documentation/qualification_logs/open_vessel_free_surface_remaining_20260526/test10_cap3_step90_pressure_stabilization_contribution_audit_20260605.json
```

## Evidence

### Test02

Output: `Documentation/qualification_logs/open_vessel_free_surface_remaining_20260526/test02_abs_only_prune1e5_0p54_pressure_stabilization_contribution_audit_20260605.json`.

The audit reconstructs 2088 active cut-adjacent faces from 3338 active wet cells and 1092 active cut cells. Test02 has `Use_cut_metadata_scale=false`, so the applied metadata scale is 1.0 even when raw cut fractions imply large conditioning scales.

The worst accepted active/wet pressure update remains the full-wet node from the pressure-update guard:

- pressure increment: 2112204.128955333 Pa
- support class: `full_wet_supported`
- point index: 1172
- point: `[0.8245, 0.0, 0.5]`
- incident reconstructed cut-adjacent faces: 0

The worst reconstructed pressure-increment ghost-penalty proxy is face 59:

- delta energy proxy: 21356.163013496265
- centroid: `[0.8803333333333333, 0.10733333333333334, 0.04975]`
- cells: 219 and 220
- current wet fractions: `2.261884636548833e-05` and `0.3068964383196363`
- raw metadata scale: 1000.0
- applied metadata scale: 1.0
- `||jump(grad(delta p))||`: 2912.273521684697 Pa/m
- maximum pressure increment on adjacent cell nodes: 89163.58142209245 Pa

Interpretation: the local cut-adjacent pressure ghost-penalty proxy does identify a tiny-cut-adjacent pressure-increment event, but it is not colocated with the 2.112 MPa full-wet active pressure update and its adjacent-node pressure increment is much smaller than the worst full-wet update.

### Test10

Output: `Documentation/qualification_logs/open_vessel_free_surface_remaining_20260526/test10_cap3_step90_pressure_stabilization_contribution_audit_20260605.json`.

The audit reconstructs 429 active cut-adjacent faces from 720 active wet cells and 236 active cut cells. Test10 has `Use_cut_metadata_scale=true` and `Cut_cell_metadata_scale_cap=3.0`, so raw scales above 3.0 are capped in the applied proxy.

The worst accepted active/wet pressure update remains the full-wet node from the pressure-update guard:

- pressure increment: 1075.2113565356985 Pa
- support class: `full_wet_supported`
- point index: 3
- point: `[0.0, 0.0365, 0.031]`
- incident reconstructed cut-adjacent faces: 0

The worst reconstructed pressure-increment ghost-penalty proxy is face 324:

- delta energy proxy: `7.570825789962345e-08`
- centroid: `[0.315, 0.09966666666666667, 0.051666666666666666]`
- cells: 1719 and 1723
- current wet fractions: `0.5507533739667316` and `0.15105064193956388`
- raw metadata scale: 6.620296260641547
- applied metadata scale: 3.0
- `||jump(grad(delta p))||`: 0.021166392751865715 Pa/m
- maximum pressure increment on adjacent cell nodes: 183.11101719567205 Pa

Interpretation: the local cut-adjacent pressure ghost-penalty proxy is not colocated with the 1075 Pa full-wet active pressure update, and the largest adjacent-node pressure increment on the worst proxy face is far below the worst accepted full-wet update.

## Hypothesis Status

Local ghost-penalty face directly creates the maximum accepted pressure update: weakened for both audited windows. The worst accepted full-wet pressure update has zero incident reconstructed cut-adjacent faces in both Test02 and Test10.

Pressure ghost penalty as a global pressure-mode seed: unresolved. The diagnostic is local and offline. It cannot rule out the pressure stabilization/support path exciting a global pressure mode that appears away from cut-adjacent faces.

Active-volume pressure residual and pressure support constraints: now higher priority. Because the largest accepted updates appear on full-wet support away from reconstructed cut-adjacent faces, the next diagnostic should decompose active-volume pressure/continuity residual and inactive pressure-support constraints around those full-wet nodes.

Penalty tuning: not supported by this evidence. The local face audit does not point to a simple scale-only correction. More pressure-penalty sweeps would not distinguish active-volume residual inconsistency from support/anchor behavior.

## Next Narrow Tests

1. Add a hydrostatic or linear-pressure cut-volume patch regression with active `LevelSetNegative` cut volumes, inactive pressure constraints, and pressure ghost penalty enabled. This should establish whether the active-volume pressure residual and constraints preserve a known pressure null/linear state.

2. Add an exact in-solver residual contribution dump for one saved/replay state only if the patch regression or offline proxy still leaves the ghost-penalty path ambiguous. The dump should split active-volume continuity/pressure terms, pressure ghost penalty, and pressure constraints.

3. Keep the accepted-step pressure update guard as a diagnostic safety net, but do not use it as the physics fix.
