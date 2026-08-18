# Open-Vessel Newton Pressure Residual Diagnostic

Date: 2026-06-05

Scope: targeted Test02/Test10 one-step replay diagnostic after the accepted-step pressure-update guard and cut-context transition checks. This audit asks whether the pressure rows of the Newton residual are visibly unconverged when the large active/wet pressure update is accepted.

## Code Change

- `Code/Source/solver/FE/TimeStepping/NewtonSolver.cpp`
  now supports an environment-gated field residual diagnostic. Set `SVMP_NEWTON_FIELD_RESIDUAL_DIAGNOSTIC=1` to log `diagnostic=newton_field_residual`; set `SVMP_NEWTON_FIELD_RESIDUAL_FIELD=<field>` to choose a field, defaulting to `Pressure`.

The diagnostic logs the selected field's constrained residual-row norm, mean/min/max, global max absolute row value, local worst row, Newton iteration, assembly phase, and synchronization point. It reads the residual after constraint zeroing, so inactive pressure constraints do not appear as artificial residual entries.

## Replay Runs

Fresh one-step replay directories:

- `Documentation/qualification_logs/open_vessel_free_surface_remaining_20260526/test02_replay_abs_only_prune1e5_step382_residual_diag_20260605_case/`
- `Documentation/qualification_logs/open_vessel_free_surface_remaining_20260526/test10_replay_cap3_step90_residual_diag_20260605_case/`

Both were run with the accepted pressure-update guard and Newton field residual diagnostic enabled.

## Results

| Case | Accepted active/wet pressure update | Support | Solve-time Pressure residual norm | Solve-time Pressure max row | Update / residual norm | Update / max row |
| --- | ---: | --- | ---: | ---: | ---: | ---: |
| Test02 step382 replay | 2,112,209.8407 Pa | full wet | 5.20234e-08 | 1.78358e-08 | 4.06e13 | 1.18e14 |
| Test10 step90 replay | 1,075.558213 Pa | full wet | 1.65093e-05 | 8.19709e-06 | 6.51e7 | 1.31e8 |

The runtime pressure-update guard again matched the offline direct VTU pressure-update audit exactly within `1e-6 Pa` in both cases.

The line-search trial residual rows for `Pressure` are therefore already tiny when the accepted full-wet pressure update is present. Test02 also shows a large initial pressure residual (`4.45394e4` norm) at the first combined Jacobian/residual assembly, which collapses to `5.20234e-08` on the accepted line-search trial while the MPa pressure jump remains.

## Interpretation

This rules out a simple explanation where the accepted pressure jump is just an obviously unconverged pressure equation residual. The Newton solve has made the pressure rows satisfy the assembled equations on the solve-time cut context.

The remaining active-volume hypothesis is sharper: the assembled pressure/continuity equations on the line-search generated cut-volume context are admitting a pressure state that is residual-consistent but physically implausible for these open-vessel free-surface replays. That points toward formulation consistency, scaling, or anchoring in the active-volume pressure path, not merely residual tolerance.

Follow-up: `Documentation/open_vessel_free_surface_vms_pressure_path_control_20260605.md` disables the residual-based VMS/PSPG branch on the same replay inputs. Test02 still accepts a tiny `Pressure` residual with a larger active/wet pressure excursion, while Test10 fails direct factorization with zero pressure rows. That rules out VMS disablement as a fix and sharpens the next target to row-level pressure support/rank decomposition.

## Artifacts

- `Documentation/qualification_logs/open_vessel_free_surface_remaining_20260526/test02_replay_abs_only_prune1e5_step382_residual_diag_pressure_update_direct_audit_20260605.json`
- `Documentation/qualification_logs/open_vessel_free_surface_remaining_20260526/test02_replay_abs_only_prune1e5_step382_residual_diag_cut_context_pressure_residual_20260605.json`
- `Documentation/qualification_logs/open_vessel_free_surface_remaining_20260526/test10_replay_cap3_step90_residual_diag_pressure_update_direct_audit_20260605.json`
- `Documentation/qualification_logs/open_vessel_free_surface_remaining_20260526/test10_replay_cap3_step90_residual_diag_cut_context_pressure_residual_20260605.json`
- `Documentation/qualification_logs/open_vessel_free_surface_remaining_20260526/test02_replay_abs_only_prune1e5_step382_vms_disabled_pressure_update_direct_audit_20260605.json`
- `Documentation/qualification_logs/open_vessel_free_surface_remaining_20260526/test02_replay_abs_only_prune1e5_step382_vms_disabled_cut_context_pressure_residual_20260605.json`
- `Documentation/qualification_logs/open_vessel_free_surface_remaining_20260526/test10_replay_cap3_step90_vms_disabled_cut_context_pressure_residual_20260605.json`
