# Open-Vessel Linear Pressure Cut-Volume Patch

Date: 2026-06-05

Scope: targeted hydrostatic/linear-pressure patch progress for the Test02/Test10 root-cause goal. This note follows the accepted pressure-update guard and the cut-adjacent pressure stabilization contribution audit. It is not a qualification closure.

## What Changed

Added `tests/cases/fluid/open_vessel_free_surface/audit_linear_pressure_cut_volume_patch.py`.

The diagnostic builds a synthetic two-tetra patch with one shared cut-adjacent face and a linear pressure field. It evaluates two support modes using the same reconstructed pressure ghost-penalty proxy as `audit_pressure_stabilization_contribution.py`, and now also assembles a P1 active-volume proxy for the direct PSPG pressure-gradient split `inner(grad(q), tau_m * grad(p))` with its matching hydrostatic body-force cancellation term. The retained-support PSPG proxy also runs a zero-mean pseudoinverse solve probe for unit pressure-row loads so the patch can distinguish hydrostatic residual consistency from boundary-row solve amplification. It now compares constant-null topology diagnostics for one-cell boundary pair completion, one-cell-to-shared support completion, one-cell-to-active support completion, shared-row Schur support completion, direct-support-gap plus same-sign-patch Schur completion, existing-edge support balancing, and incident-support count balancing. These probes test which pressure-support topology changes reduce amplification without breaking hydrostatic cancellation or the constant-pressure null.

- retained cut-volume support, where all pressure vertices in both cells remain active;
- pruned trace-only cut-adjacent support, where only the shared face vertices remain active and the off-trace dry vertices are constrained to zero.
- fixed pruned support, where cut-adjacent support is skipped when no retained generated volume remains.

Added `tests/test_open_vessel_linear_pressure_cut_volume_patch.py` to verify the expected patch behavior.

Updated `Code/Source/solver/FE/Constraints/LevelSetActiveSideVertexDirichletConstraint.cpp` so cut-adjacent facet support is applied only when the generated interface marker still has retained generated-volume rules. If a facet handle exists but no retained generated volume remains, the diagnostic support mode records `cut_adjacent_facets_skipped_no_retained_volume`.

Updated the C++ constraint tests in `Code/Source/solver/FE/Tests/Unit/Constraints/test_LevelSetActiveSideVertexDirichletConstraint.cpp` so pruned generated-volume support no longer keeps shared high-order trace pressure DOFs active by facet metadata alone.

## Verification

Focused checks passed:

```bash
python -m py_compile tests/cases/fluid/open_vessel_free_surface/audit_linear_pressure_cut_volume_patch.py
pytest -q tests/test_open_vessel_linear_pressure_cut_volume_patch.py
build/svMultiPhysics-build/bin/test_fe_constraints --gtest_filter='LevelSetActiveSideVertexDirichletConstraint.*'
```

Pytest result: `1 passed in 0.81s`.

Filtered C++ result: `14 tests` from `LevelSetActiveSideVertexDirichletConstraint`, all passed.

Follow-up focused check after adding the PSPG hydrostatic balance and boundary solve-amplification proxies:

```bash
python -m py_compile tests/cases/fluid/open_vessel_free_surface/audit_linear_pressure_cut_volume_patch.py
pytest -q tests/test_open_vessel_linear_pressure_cut_volume_patch.py
```

Pytest result: `1 passed in 0.67s`.

Follow-up focused check after adding the constant-null boundary pair-completion probe:

```bash
python -m py_compile tests/cases/fluid/open_vessel_free_surface/audit_linear_pressure_cut_volume_patch.py
pytest -q tests/test_open_vessel_linear_pressure_cut_volume_patch.py
```

Pytest result: `1 passed in 0.51s`.

Follow-up focused check after adding shared-support and active-support topology completion probes:

```bash
python -m py_compile tests/cases/fluid/open_vessel_free_surface/audit_linear_pressure_cut_volume_patch.py
pytest -q tests/test_open_vessel_linear_pressure_cut_volume_patch.py
```

Pytest result: `1 passed in 0.44s`.

Follow-up focused check after adding incident-support count balancing:

```bash
python -m py_compile tests/cases/fluid/open_vessel_free_surface/audit_linear_pressure_cut_volume_patch.py
pytest -q tests/test_open_vessel_linear_pressure_cut_volume_patch.py
```

Pytest result: `1 passed in 0.59s`.

Follow-up focused check after separating direct-support-gap plus same-sign-patch Schur-only topology from the edge-balance stage:

```bash
python -m py_compile tests/cases/fluid/open_vessel_free_surface/audit_linear_pressure_cut_volume_patch.py
pytest -q tests/test_open_vessel_linear_pressure_cut_volume_patch.py
```

Pytest result: `1 passed in 0.82s`.

Follow-up focused check after adding shared-row Schur support completion:

```bash
python -m py_compile tests/cases/fluid/open_vessel_free_surface/audit_linear_pressure_cut_volume_patch.py
pytest -q tests/test_open_vessel_linear_pressure_cut_volume_patch.py
```

Pytest result: `1 passed in 0.61s`.

Follow-up solver-level check after adding shared-row Schur graph-completion replay mode:

```bash
cmake --build build/svMultiPhysics-build --target svmultiphysics -j2
pytest -q tests/test_open_vessel_pressure_matrix_support_samples.py tests/test_open_vessel_pressure_graph_completion_selector.py
```

Focused build/parser result: build completed; pytest result: `2 passed in 0.61s`. The short replay windows accepted one step in both cases, but still triggered the active/wet pressure-update guards: Test02 reached `225590.272690 Pa` on tiny-cut row `10624`, with the original full-wet row `10676` still at `193366.433797 Pa`; Test10 reached `319.184466 Pa` on a full-wet row. This verifies solver-level leverage for the Schur topology while ruling out local post-assembly Schur fill as the complete fix.

The JSON artifact is:

- `Documentation/qualification_logs/open_vessel_free_surface_remaining_20260526/linear_pressure_cut_volume_patch_audit_20260605.json`
- `Documentation/qualification_logs/open_vessel_free_surface_remaining_20260526/linear_pressure_cut_volume_patch_topology_completion_20260606.json`

## Evidence

The retained cut-volume support case passes the linear-pressure patch:

- support mode: `retained_cut_volume+cut_adjacent_facets`
- active vertices: `[0, 1, 2, 3, 4]`
- constrained vertices: `[]`
- preserves linear pressure state: `true`
- `||jump(grad(p))||`: 0.0 Pa/m
- current ghost-penalty energy proxy: 0.0
- direct PSPG pressure-gradient max row action: `0.5`
- hydrostatic PSPG total max row action after matching body-force cancellation: `0.0`
- PSPG pressure block preserves constant-pressure null row sum: `true`
- PSPG pressure-gradient proxy matrix rank: `4`
- weakest one-cell boundary row: row `3`, diagonal `0.041666666666666664`, row absolute sum `0.08333333333333333`
- zero-mean solve response at row `3`: `15.98399999999999`
- strongest shared row: row `0`, diagonal `0.625`, row absolute sum `1.25`
- zero-mean solve response at row `0`: `1.5839999999999999`
- weakest-to-strongest target response ratio: `10.090909090909085`
- uniform scale-10 probe max row response: `1.598399999999999`
- uniform scale-10 weakest-to-strongest target response ratio: `10.090909090909085`
- boundary pair-completion edge weight: `0.041666666666666664`
- boundary pair-completion edge count: `1`
- boundary pair-completion max row response: `8.277333333333328`
- boundary pair-completion weakest-to-strongest target response ratio: `6.1587301587301555`
- boundary pair-completion preserves hydrostatic cancellation: `true`
- boundary pair-completion preserves constant-pressure null row sum: `true`
- one-cell-to-shared support completion edge count: `6`
- one-cell-to-shared support completion weakest-to-strongest target response ratio: `6.595813204508855`
- one-cell-to-shared support completion preserves hydrostatic cancellation: `true`
- one-cell-to-active support completion edge count: `7`
- one-cell-to-active support completion contribution count: `8`
- one-cell-to-active support completion weakest-to-strongest target response ratio: `5.768740031897927`
- one-cell-to-active support completion preserves hydrostatic cancellation: `true`
- shared-row Schur support completion edge count: `6`
- shared-row Schur support completion contribution count: `6`
- shared-row Schur support completion weakest-to-strongest target response ratio: `6.153374233128836`
- shared-row Schur support completion max row response: `8.024000000000001`
- shared-row Schur support completion preserves hydrostatic cancellation: `true`
- shared-row Schur support completion preserves constant-pressure null row sum: `true`
- direct-support-gap plus same-sign-patch Schur-only weakest-to-strongest target response ratio: `6.153374233128836`
- direct-support-gap plus same-sign-patch Schur-only preserves hydrostatic cancellation: `true`
- direct-support-gap plus same-sign-patch Schur-only preserves constant-pressure null row sum: `true`
- direct-support-gap plus same-sign-patch Schur-plus-gap-balance weakest-to-strongest target response ratio: `2.168696625002024`
- direct-support-gap plus same-sign-patch balance-stage ratio reduction over Schur-only: `2.837360542820549`
- existing-edge support-balance weakest-to-strongest target response ratio: `4.75`
- incident-support count balance edge count: `4`
- incident-support count balance max edge scale: `2.0`
- incident-support count balance weakest-to-strongest target response ratio: `8.317073170731707`
- incident-support count balance preserves hydrostatic cancellation: `true`
- incident-support count balance preserves constant-pressure null row sum: `true`

The full-volume one-cell boundary topology control keeps the same pressure-block amplification class without a small cut fraction:

- support mode: `full_active_volume+one_cell_boundary_rows`
- active volume fractions: `[1.0, 1.0]`
- hydrostatic PSPG total max row action after matching body-force cancellation: `0.0`
- PSPG pressure block preserves constant-pressure null row sum: `true`
- weakest one-cell boundary row: row `3`, diagonal `0.16666666666666666`, row absolute sum `0.3333333333333333`
- zero-mean solve response at row `3`: `4.320000000000001`
- strongest shared row: row `0`, diagonal `1.0`, row absolute sum `2.0`
- zero-mean solve response at row `0`: `0.7199999999999996`
- weakest-to-strongest target response ratio: `6.000000000000004`
- uniform scale-10 weakest-to-strongest target response ratio: `6.000000000000004`
- boundary pair-completion edge weight: `0.16666666666666666`
- boundary pair-completion max row response: `2.5199999999999996`
- boundary pair-completion weakest-to-strongest target response ratio: `3.4999999999999996`
- boundary pair-completion preserves hydrostatic cancellation: `true`
- boundary pair-completion preserves constant-pressure null row sum: `true`
- one-cell-to-shared support completion edge count: `6`
- one-cell-to-shared support completion weakest-to-strongest target response ratio: `3.546296296296298`
- one-cell-to-shared support completion preserves hydrostatic cancellation: `true`
- one-cell-to-active support completion edge count: `7`
- one-cell-to-active support completion contribution count: `8`
- one-cell-to-active support completion weakest-to-strongest target response ratio: `2.9047619047619047`
- one-cell-to-active support completion preserves hydrostatic cancellation: `true`
- shared-row Schur support completion edge count: `6`
- shared-row Schur support completion contribution count: `6`
- shared-row Schur support completion weakest-to-strongest target response ratio: `3.2058823529411793`
- shared-row Schur support completion max row response: `2.18`
- shared-row Schur support completion preserves hydrostatic cancellation: `true`
- shared-row Schur support completion preserves constant-pressure null row sum: `true`
- direct-support-gap plus same-sign-patch Schur-only weakest-to-strongest target response ratio: `3.2058823529411793`
- direct-support-gap plus same-sign-patch Schur-only preserves hydrostatic cancellation: `true`
- direct-support-gap plus same-sign-patch Schur-only preserves constant-pressure null row sum: `true`
- direct-support-gap plus same-sign-patch Schur-plus-gap-balance weakest-to-strongest target response ratio: `2.1688311688311694`
- direct-support-gap plus same-sign-patch balance-stage ratio reduction over Schur-only: `1.4781613244100045`
- existing-edge support-balance weakest-to-strongest target response ratio: `4.750000000000001`
- incident-support count balance edge count: `4`
- incident-support count balance max edge scale: `2.0`
- incident-support count balance weakest-to-strongest target response ratio: `4.75`
- incident-support count balance preserves hydrostatic cancellation: `true`
- incident-support count balance preserves constant-pressure null row sum: `true`

The pruned trace-only support case exposes a pressure-support hazard:

- support mode: `cell_patch+cut_adjacent_facets`
- active vertices: `[0, 1, 2]`
- constrained vertices: `[3, 4]`
- preserves linear pressure state: `false`
- `||jump(grad(p))||`: 20.0 Pa/m
- current ghost-penalty energy proxy: 598205.3832520385
- direct PSPG pressure-gradient max row action: `1.6666666666666665`
- hydrostatic PSPG total max row action after matching body-force cancellation: `1.708333333333333`
- PSPG pressure block preserves constant-pressure null row sum: `true`

The fixed pruned support case removes that ghost-penalty coupling:

- support mode: `cell_patch+cut_adjacent_facets_skipped_no_retained_volume`
- active vertices: `[]`
- constrained vertices: `[0, 1, 2, 3, 4]`
- cut-adjacent face applied: `false`
- preserves zero ghost-penalty coupling: `true`
- direct PSPG pressure-gradient max row action: `0.0`
- hydrostatic PSPG total max row action: `0.0`

Interpretation: the pressure ghost-penalty form preserves a linear pressure state when retained cut-volume support keeps the neighboring pressure DOFs active. The same linear state becomes inconsistent when support falls back to trace-only cut-adjacent DOFs and off-trace dry pressure DOFs are constrained to zero. The support-path fix prevents that trace-only coupling when the generated active volume has been pruned.

The PSPG pressure-gradient proxy adds a separate conclusion. On retained support, the direct `grad(p)` split has nonzero boundary-row action and a rank-4 pressure self-block while preserving the constant-pressure null row sum. It is hydrostatic-consistent only after the matching nonpressure body-force residual cancels it on the same active-volume support. When off-trace pressure vertices are constrained while the active-volume operator still sees retained cells, that cancellation breaks. The solve-amplification probe adds that a constant-null, hydrostatic-consistent pressure-gradient block can still produce a much larger nonconstant response at the weakest one-cell boundary row than at the strongest shared row. The full-volume topology control keeps a `6x` boundary response ratio even with active volume fractions `[1.0, 1.0]`, ruling out tiny cut fraction as the only patch-level cause. The uniform scale probe reduces absolute response but preserves the boundary/shared response ratio, matching the Test02/Test10 replay evidence that a global PSPG pressure-gradient multiplier is only directional evidence. The boundary pair-completion probe changes the pressure support topology while preserving the same constant null and matched hydrostatic cancellation, reducing the retained ratio from `10.09x` to `6.16x` and the full-volume ratio from `6.0x` to `3.5x`. A one-cell-to-shared support completion also preserves those invariants, but it is weaker (`6.596x` retained and `3.546x` full-volume). A broader one-cell-to-active completion is stronger in the full-volume topology control (`2.905x`) and also improves the retained case (`5.769x`), but it is less local than pair completion or existing-edge balance. The shared-row Schur support completion derives neighbor-neighbor edges from the existing pressure-gradient graph with `w_i*w_j/sum(w)` around each shared row; it preserves the invariants, nearly matches pair completion on the retained patch (`6.153x`), and is stronger than pair completion on the full-volume topology control (`3.206x`). The direct-support-gap plus same-sign-patch probe now exposes that same Schur-only topology explicitly for the formulation predicate: it preserves hydrostatic cancellation and the constant-pressure null at `6.153x` retained and `3.206x` full-volume, while the subsequent gap-row balance stage further lowers the ratios to `2.169x` in both controls. The incident-support count balance uses only endpoint active incident-cell count deficits; it preserves the same invariants but only lowers the retained ratio to `8.317x` and the full-volume ratio to `4.75x`. This rules out a simple claim that the direct PSPG pressure-gradient term is intrinsically residual-inconsistent on a retained linear/hydrostatic patch, and it also rules out shared-support-only completion or incident-cell-count-only balancing as the strongest patch target. The remaining target is a formulation-derived topology or edge-balance rule that changes boundary pressure support without breaking hydrostatic cancellation or constant-null behavior.

## Hypothesis Status

Active retained cut-volume pressure consistency: partially supported for this minimal patch. With retained volume support, the reconstructed first-derivative pressure ghost penalty sees zero jump for a linear pressure state.

Active retained cut-volume PSPG hydrostatic consistency: supported for this minimal patch when pressure state, body force, and active-volume support are matched. The direct PSPG pressure-gradient split alone has nonzero boundary action and pressure self-rank, but the total hydrostatic PSPG action is zero and the constant-pressure null row sum is preserved.

Direct PSPG boundary pressure-block solve amplification: supported as a minimal proxy. In the retained patch, the weakest one-cell boundary row has a zero-mean target solve response `10.09x` the strongest shared row response even though the matrix preserves the constant-pressure null and matched hydrostatic cancellation. With full active-volume fractions, the same topology still gives a `6.0x` response ratio. This mirrors the production evidence that the remaining Test02/Test10 pressure jumps are nonconstant boundary pressure modes after the constrained matrix solve, not constant gauge leakage or only tiny cut-volume support.

Uniform PSPG pressure-gradient scaling as a complete fix: ruled out in the patch proxy. Scaling the retained pressure-gradient block by `10x` reduces the absolute max target response from `15.984` to `1.5984`, but the weakest-to-strongest response ratio remains `10.09x`; the full-volume topology control likewise keeps its `6.0x` ratio. This matches the replay result where scale-up reduces the original sampled rows but leaves shifted boundary/cut updates.

Constant-null boundary pair completion as a next formulation direction: supported as an offline diagnostic prototype, not yet a production fix. Adding one pressure edge between the one-cell boundary rows with weight equal to the smallest positive one-cell boundary diagonal preserves hydrostatic cancellation and the constant-pressure null, while reducing the retained boundary/shared response ratio to `6.1587301587301555` and the full-volume topology ratio to `3.4999999999999996`. The probe changes the response shape, unlike uniform scaling, but still needs a solver-level formulation and Test02/Test10 replay check before it can be treated as resolved.

Shared-support-only completion as the whole topology fix: weakened by the patch proxy. Constant-null edges from one-cell boundary rows to multi-cell shared rows preserve hydrostatic cancellation and the constant-pressure null, but reduce the retained ratio only to `6.595813204508855`, weaker than pair completion, active completion, and existing-edge balance.

Broader one-cell-to-active completion as a formulation direction: supported as an offline diagnostic prototype, not yet a production fix. Distributing constant-null edges from each one-cell boundary row to every other active row reduces the retained ratio to `5.768740031897927` and the full-volume ratio to `2.9047619047619047`, while preserving hydrostatic cancellation and the constant-pressure null. This is the strongest full-volume patch result, but it is broader than a local existing-edge or pair-completion rule and needs a formulation-derived stencil before any replay.

Shared-row Schur support completion as a formulation direction: supported as an offline diagnostic prototype, not yet a production fix. Deriving neighbor-neighbor edges from existing PSPG pressure-gradient support via `w_i*w_j/sum(w)` around shared rows reduces the retained ratio to `6.153374233128836` and the full-volume ratio to `3.2058823529411793` while preserving hydrostatic cancellation and the constant-pressure null. This is the strongest local topology-derived patch result so far, because it improves over manual pair completion in the full-volume control without relying on endpoint incident-cell-count scaling.

Direct-support-gap plus same-sign-patch Schur topology as a formulation predicate: supported as an offline diagnostic prototype, not yet a production fix. The patch now records the Schur-only part of that predicate separately from the edge-balance stage. Schur-only preserves hydrostatic cancellation and the constant-pressure null while matching the shared-row Schur ratios (`6.153374233128836` retained and `3.2058823529411793` full-volume). The balance stage further reduces the patch ratios to `2.168696625002024` and `2.1688311688311694`, but solver replays show broad or balanced post-assembly forms can destabilize Test02. The useful next target is therefore formulation-side topology and a locally constrained balance/coupling rule, not a raw promotion of the diagnostic balance mutation.

Direct-PSPG predicate derivation readiness: still unresolved. The refreshed target map records `predicate_derivation_readiness=coverage_complete_but_no_formulation_side_derivation`. The combined `direct_support_gap_or_same_sign_pressure_action_patch` predicate covers the audited Test02/Test10 direct rows, but the inputs still depend on top-update rows, pressure-update values, and post-assembly matrix samples. That makes it evidence for the formulation shape, not an implementation rule. The next rule must derive an equivalent target from solve-time active cut-volume direct PSPG pressure-gradient topology/coupling.

Graph-completion candidate-readiness audit: implemented in `tests/cases/fluid/open_vessel_free_surface/audit_direct_pspg_graph_completion_candidate_readiness.py`, with coverage in `tests/test_open_vessel_direct_pspg_graph_completion_candidate_readiness.py`. The generated artifact `Documentation/qualification_logs/open_vessel_free_surface_remaining_20260526/test02_test10_direct_pspg_graph_completion_candidate_readiness_20260606.json` shows the closest pre-linear support-gap graph-completion selectors are still too broad: Schur-only selects `879/7` Test02 candidates and `251/12` Test10 candidates, local depth-1 Schur-only selects `652/7` and `188/12`, and Schur plus edge balance selects `879/7` and `251/12`. All three clear the Test10 guard, but all three fail Test02 nonlinear convergence, so the next rule remains a narrower formulation-side direct PSPG support/coupling rule rather than a broad pre-linear pressure-patch selector.

Formulation-side candidate predicate audit: implemented in `tests/cases/fluid/open_vessel_free_surface/audit_direct_pspg_formulation_side_candidate_predicates.py`, with coverage in `tests/test_open_vessel_direct_pspg_formulation_side_candidate_predicates.py`. The generated artifact `Documentation/qualification_logs/open_vessel_free_surface_remaining_20260526/test02_test10_direct_pspg_formulation_side_candidate_predicates_20260606.json` identifies `sparse_direct_self_or_same_sign_pressure_action_patch` as the preferred next candidate: sparse direct pressure-gradient self entries cover Test02's isolated row, while same-sign direct PSPG pressure-action patch coverage covers the remaining Test02 rows and all Test10 rows (`7/7` and `12/12` exact audited coverage). Same-sign coverage alone and zero-Galerkin/nonpressure coupling plus same-sign coverage remain partial because they miss the isolated Test02 row. This is still a diagnostic candidate, not a fix, until the same predicate is emitted globally before pressure updates are known.

Global direct-PSPG candidate emission: implemented in `Code/Source/solver/FE/TimeStepping/NewtonSolver.cpp` behind `SVMP_NS_PRESSURE_ROW_CONTRIBUTION_DIAGNOSTIC=1` and `SVMP_NS_DIRECT_PSPG_FORMULATION_CANDIDATE_DIAGNOSTIC=1`, with parser coverage in `tests/test_open_vessel_direct_pspg_global_candidate_emission.py`. The generated artifact `Documentation/qualification_logs/open_vessel_free_surface_remaining_20260526/test02_test10_direct_pspg_global_candidate_emission_20260606.json` records `candidate_emission_covers_audited_targets`: Test02 covers all `7/7` audited direct targets with `866` preferred global candidates, and Test10 covers all `12/12` with `251`. Candidate lists are untruncated. This rules out missing global diagnostic emission as the next blocker.

Global direct-PSPG candidate selectivity: implemented in `tests/cases/fluid/open_vessel_free_surface/audit_direct_pspg_global_candidate_selectivity.py`, with coverage in `tests/test_open_vessel_direct_pspg_global_candidate_selectivity.py`. The generated artifact `Documentation/qualification_logs/open_vessel_free_surface_remaining_20260526/test02_test10_direct_pspg_global_candidate_selectivity_20260607.json` records `global_candidate_selector_overbroad_matrix_proxy_not_formulation_ready`: the raw preferred set is `866/7` candidates per audited Test02 target and `251/12` per Test10 target, sparse direct-self alone remains `545/7` and `217/12`, and the matrix pressure-action proxy covers all positive direct PSPG rows in both cases. The sparse-seeded pressure-action component gate is also overbroad, yielding one component with `866/7` Test02 rows per audited target and one with `251/12` Test10 rows per target. Bounded sparse-seeded pressure-action neighborhoods are also overbroad: radius `1` selects `818/7` Test02 and `251/12` Test10 rows while covering the audited targets, and radius `2` expands to `866/7` and `251/12`. The global direct-self support-ratio gate is not sufficient either: `sparse_or_moderate` direct-self ratio candidates miss six of seven Test02 targets while selecting `572/7` rows per target, and they cover Test10 only with an overbroad `217/12` rows per target. The graph-local support-ratio gate is also incomplete: moderate graph-local direct-self contrast misses six of seven Test02 targets while selecting `584/7` rows per target, and covers Test10 only with an overbroad `211/12` rows per target. Pressure-action degree, edge-sum, and self-dominance filters also fail as complete gates: moderate degree selects `167/7` Test02 and `99/12` Test10 rows but misses targets, moderate edge-sum misses Test02 while selecting `561/7` rows, and self-dominant rows do not cover either target set. The mesh boundary-provenance artifact `Documentation/qualification_logs/open_vessel_free_surface_remaining_20260526/test02_test10_direct_pspg_boundary_provenance_selectivity_20260607.json` also rules out literal mesh boundary and incident-cell support selectors: preferred boundary coverage is `3/7` Test02 and `9/12` Test10, one-cell boundary support covers `0/7` and `0/12`, and sparse-or-moderate boundary support covers `0/7` and `9/12`. The cut-state provenance artifact `Documentation/qualification_logs/open_vessel_free_surface_remaining_20260526/test02_test10_direct_pspg_cut_state_provenance_selectivity_20260607.json` rules out simple source point activity, `phi`, and incident `WetVolumeFraction` gates: inactive-point and dry-or-cut support are overbroad (`634/7`, `120/12`, `697/7`, `158/12`), while dry-only and cut-incident support miss audited targets (`6/7` Test02 dry-only, `1/7` Test02 cut-incident, and `0/12` Test10 cut-incident). The same-sign dependency-readiness artifact `Documentation/qualification_logs/open_vessel_free_surface_remaining_20260526/test02_test10_direct_pspg_same_sign_dependency_readiness_20260607.json` rules out promoting the exact same-sign patch oracle: it depends on pressure-update signs, has no complete non-update-dependent candidate, and all tested pre-update proxy gates fail. The active pressure-support topology artifact `Documentation/qualification_logs/open_vessel_free_surface_remaining_20260526/test02_test10_direct_pspg_active_pressure_support_selectivity_20260607.json` rules out constrained pressure-neighbor exposure as the missing gate (`0/7` and `0/12` target coverage), and shows sparse unconstrained direct-self topology is still overbroad or incomplete (`545/7` with `1/7` Test02 targets and `217/12` with `12/12` Test10 targets). The residual-sign selectivity artifact `Documentation/qualification_logs/open_vessel_free_surface_remaining_20260526/test02_test10_direct_pspg_residual_sign_selectivity_20260607.json` rules out using the operator residual sign graph as the pre-update same-sign substitute: same-sign residual action misses Test02 row `10676` while selecting `857/7` rows, and sparse-seeded residual components or sparse-self plus residual action cover both cases only by selecting `862/7` Test02 rows and all `251/12` Test10 rows. The null/balance artifact `Documentation/qualification_logs/open_vessel_free_surface_remaining_20260526/test02_test10_direct_pspg_null_balance_selectivity_20260607.json` rules out simple constant-null leakage or diagonal-balance gates: high row-sum leakage and diagonal dominance cover `0/7` and `0/12` audited targets, while null-preserving and balanced-diagonal rows cover targets only with overbroad `861/7`, `245/12`, `864/7`, and `247/12` candidate sets. This rules out promoting raw global emitted candidates, sparse-seeded pressure-action components, bounded sparse-seeded pressure-action neighborhoods, simple direct-self support-ratio deficiency, graph-local direct-self support contrast, pressure-action degree/edge-sum/self-dominance filters, raw mesh boundary/incident support, simple source cut-state provenance, post-update same-sign pressure-action connectivity, direct constrained-neighbor exposure, sparse unconstrained pressure-neighbor deficiency, pre-update residual-sign pressure-action connectivity, direct PSPG row-sum leakage, or diagonal-balance topology as the formulation rule; the remaining work is deriving a formulation-side PSPG support/coupled-patch provenance gate before replaying a support/coupling prototype.

Solver-level shared-row Schur graph completion as a complete fix: ruled out as a post-assembly matrix mutation. The replay mode applies the same `w_i*w_j/sum(w)` topology to the Newton matrix and improves over shared-pressure-neighbor pairing, but it still accepts guard-triggering updates: `225590.272690 Pa` in Test02 and `319.184466 Pa` in Test10. The result keeps Schur-derived topology as a formulation signal, not as a mode to promote.

Incident-support count normalization as a complete support rule: ruled out in the patch proxy. Scaling existing pressure-gradient edges only by endpoint incident-cell-count deficits preserves hydrostatic cancellation and the constant-pressure null, and uses modest `2x` maximum edge scaling, but it only lowers the retained ratio to `8.317073170731707` and the full-volume ratio to `4.75`. That is weaker than row-abs existing-edge balancing in the retained case and weaker than pair or active completion in the full-volume topology control.

Trace-only/pruned pressure support interaction: supported as a real hazard. If a cut-adjacent face is retained but off-trace pressure DOFs are constrained, an otherwise linear pressure field produces a nonzero pressure-gradient jump.

Trace-only/pruned PSPG support interaction: supported as the same class of hazard. If pressure DOFs are constrained inconsistently with the retained active-volume support, the matched hydrostatic PSPG cancellation is broken.

Trace-only/pruned pressure support fix: implemented for the pressure constraint path. Cut-adjacent facet support no longer activates pressure DOFs unless the generated active volume has retained volume rules.

Local ghost-penalty face as the direct source of the largest Test02/Test10 accepted full-wet updates: still weakened by the saved-state contribution audit, because the worst accepted full-wet update in both cases has zero incident reconstructed cut-adjacent faces.

Pressure support/constraint path as a seed for a global pressure mode: strengthened. The patch shows a concrete mechanism by which cut-adjacent pressure support and inactive pressure constraints can inject pressure inconsistency before it appears away from the interface.

Cut-adjacent support pressure-window audit: implemented in `tests/cases/fluid/open_vessel_free_surface/audit_cut_adjacent_support_pressure_window.py`, with coverage in `tests/test_open_vessel_cut_adjacent_support_pressure_window.py`. The generated artifact `Documentation/qualification_logs/open_vessel_free_surface_remaining_20260526/test02_test10_cut_adjacent_support_pressure_window_20260606.json` joins each accepted pressure-update guard to the nearest pre-guard cut context and pressure active-support constraint diagnostic. Test02 has retained cut-volume pressure support before the row `10676` guard (`3358` retained volume-support cells, `0` cut-adjacent-only support cells, `0` generated pruned-volume rules). Test10 has recently pruned tiny generated volume before the row `3526` guard (`8` generated pruned-volume rules, `3.5259659448027454e-10` generated pruned volume), but pressure support still comes from retained cut-volume support (`720` volume-support cells, `0` cut-adjacent-only support cells). Both accepted worst rows are full-wet. This rules out trace-only cut-adjacent support as the immediate driver of those full-wet replay jumps; the Test10 pruned-volume signal remains context, not a direct trace-only pressure-support path.

## Next Narrow Tests

1. Run short Test02/Test10 replay windows only if needed to check whether this support-path fix changes accepted full-wet pressure updates. The saved-state diagnostics show the max update away from local cut-adjacent faces, so a before/after should remain concise and targeted.

2. The solver-level wall-tangential topology probe has now been run on the short Test02/Test10 replay windows. `SVMP_NS_PSPG_BOUNDARY_TANGENTIAL_PRESSURE_GRADIENT_SCALE=1` reaches Test02 row `10676` and all `23/23` sampled important Test02 rows, lowering the accepted full-wet update to `370071.857167 Pa`. It also reaches all `19/19` sampled important Test10 rows and lowers the accepted full-wet update to `591.865160 Pa`. Combining wall-normal and wall-tangential pressure-gradient support gives Test02 `366719.965806 Pa`, only a tiny improvement over tangential-only, and Test10 `622.609410 Pa`, worse than tangential-only; both still trigger their guards. A local edge-completion predictor on those full-gradient max rows supports diagonal-scale constant-null edge completion for Test02, but finds no logged Test10 pressure neighbor below the `100 Pa` guard. A solver-level weak-self graph-cycle prototype with the same full-gradient support lowers Test02 to `226684.935365 Pa` and Test10 to `474.046852 Pa`, but both still trigger their guards and Test02 shifts to a tiny-cut-supported row. A structured strongest-existing-pressure-neighbor graph-completion mode is not better: Test02 remains `226803.140015 Pa`, and Test10 worsens to `573.379695 Pa`. A solver-level shared-row Schur completion replay improves over shared-pressure-neighbor pairing but still leaves Test02 at `225590.272690 Pa` and Test10 at `319.184466 Pa`, so local Schur fill is also diagnostic-only. `SVMP_NS_PSPG_BOUNDARY_TANGENTIAL_MOMENTUM_RESIDUAL_SCALE=1` adds the raw wall-tangential full residual; it lowers Test10 to `472.150770 Pa` but shifts the maximum to a cut-supported row, and it worsens Test02 to `726276.622250 Pa` despite adding sampled velocity coupling. These probes support topology-directed boundary pressure support but rule out a wall-tangential self-block term, a full wall-gradient pressure self block, a purely local edge to the existing Test10 pressure neighbors, an unstructured weak-row graph cycle, strongest-existing-pressure-neighbor completion, local shared-row Schur completion, or raw wall full-residual coupling alone as the production fix.

3. Completed by the cut-adjacent support pressure-window audit above. The accepted windows do not include trace-only cut-adjacent pressure support before the full-wet guard jumps; Test10 includes recently pruned tiny generated volume, but the active pressure constraint remains retained-cut-volume-backed with zero cut-adjacent-only support cells.
