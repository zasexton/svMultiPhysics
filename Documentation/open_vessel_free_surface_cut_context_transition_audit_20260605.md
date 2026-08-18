# Open-Vessel Cut-Context Pressure Transition Audit

Date: 2026-06-05

Scope: targeted follow-up for the Test02/Test10 one-step runtime-guard replays. This audit checks whether the accepted pressure jump is introduced before or after the accepted-step level-set maintenance cut-context rebuild.

## Diagnostic Added

- `tests/cases/fluid/open_vessel_free_surface/audit_cut_context_pressure_transition.py`
  parses solver logs for `diagnostic=cut_context_rebuild`, `diagnostic=cut_context_refresh_skip`, `diagnostic=accepted_pressure_update_guard`, nonlinear completion, and accepted-step lifecycle lines.
- `tests/test_open_vessel_cut_context_pressure_transition.py`
  covers lifecycle ordering, context delta reporting, and runtime/offline pressure-update matching.

## Results

| Case | Guard line | Accepted-step rebuild line | Runtime/offline pressure match | Runtime max update | Support | Post-acceptance refresh immediate driver |
| --- | ---: | ---: | --- | ---: | --- | --- |
| Test02 step382 replay | 576 | 579 | exact within 1e-6 Pa | 2,112,209.8407 Pa | full wet | ruled out |
| Test10 step90 replay | 982 | 985 | exact within 1e-6 Pa | 1,075.558213 Pa | full wet | ruled out |

The accepted pressure-update diagnostic fires before the `provenance=accepted_step` maintenance rebuild in both runs. The runtime guard value also matches the offline direct source-to-replay VTU audit exactly within the configured tolerance. Therefore the post-acceptance level-set maintenance refresh is not the immediate source of either accepted pressure increment.

The solve-time context is still relevant. In both runs the pressure jump is produced on the `provenance=line_search_trial` cut context:

| Case | Initial -> solve active cut count | Initial -> solve active volume relative change | Solve -> accepted-step count changes | Solve -> accepted-step active physical volume relative change |
| --- | ---: | ---: | ---: | ---: |
| Test02 | 0 | 1.75e-6 | 0 | 2.23e-7 |
| Test10 | -4 | 4.09e-7 | 0 | 4.51e-7 |

The Test10 initial-to-solve transition changes cut topology counts (`active_cut_cells` 240 -> 236, `active_full_wet_cells` 480 -> 484, `active_pruned_volume_regions` 0 -> 8), but those changes occur before the accepted pressure guard fires and are part of the solve-time line-search context. The accepted-step maintenance rebuild after the guard changes no audited counts in either case.

## Interpretation

This rules out one specific geometry-refresh mechanism: a pressure field that is acceptable during the solve but becomes bad only after post-acceptance level-set maintenance rebuilds the cut context. The pressure excursion is already present when the step is accepted.

This does not rule out the solve-time active-volume path. The remaining target is now narrower: the active-volume pressure/continuity residual, pressure stabilization, and inactive support/anchor interaction on the line-search cut context that the Newton solve actually used.

## Artifacts

- `Documentation/qualification_logs/open_vessel_free_surface_remaining_20260526/test02_replay_abs_only_prune1e5_step382_runtime_guard_cut_context_transition_20260605.json`
- `Documentation/qualification_logs/open_vessel_free_surface_remaining_20260526/test10_replay_cap3_step90_runtime_guard_cut_context_transition_20260605.json`
