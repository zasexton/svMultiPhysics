# Unfitted Level-Set Assembly Diagnostics - 2026-05-15

## Scope

This note records the current assembly-focused diagnostics for the
unfitted free-surface remediation path. The probes use GMRES for the linear
solve and the MMS traveling-interface case as the primary assembly
correctness check.

## MMS Traveling-Interface GMRES Probe

Probe:

```text
run_test05_velocity_growth_smoke.py --case mms2d --steps 1 --disable-vtk-output
--linear-solver-type GMRES --linear-relative-tolerance 1e-4
--linear-absolute-tolerance 1e-8 --linear-max-iterations 300
--enable-jacobian-check --jacobian-check-components velocity,pressure
--max-jacobian-check-relative-error 1e-4
--enable-newton-direction-check --max-newton-direction-relative-error 2e-4
--enable-linear-solve-history --enable-form-block-diagnostics
--require-cut-context-solution-source-diagnostics
--require-assembly-timing-diagnostics --enable-interior-face-timing
--enable-cut-volume-timing --enable-jit-specialization-trace
--require-assembly-topology-consistency
```

Result: passed.

Key diagnostics:

- GMRES accepted the one-step solve with one nonlinear iteration and `350`
  linear iterations at relative residual `9.99138625978304e-05`.
- The finite-difference Jacobian check with `velocity,pressure` selected
  reported relative mismatch `1.06359e-05`.
- The Newton direction check reported relative error `9.99139e-05`, matching
  the requested GMRES residual scale.
- Cut-context refresh sources were ordered correctly:
  `fe_vector=2`, `state_vector_fe_ordered=10`, missing source count `0`.
- Interior-face timing assembled `53` faces, matching the generated
  cut-adjacent facet count.
- Cut-volume timing used indexed traversal and matched the generated rule
  counts: maximum `180` rules assembled, `18` partial rules, `162` full rules,
  and `756` quadrature points.
- Phase summaries separated the operator modes:
  interior-face maximum total was `0.378835 s` for `matrix=1,vector=1` and
  `0.1582 s` for `matrix=1,vector=0`.
- JIT specialization traces reported `26` specialized compiles and `11`
  generic compiles. Runtime compiles were only cut-volume variants:
  `CutVolume/Residual=8`, `CutVolume/Tangent=16`.

This rules out a gross velocity/pressure residual-vs-Jacobian assembly
sign or block-placement mismatch in the MMS fixed-geometry probe.

## MMS GMRES Component-Sweep Probe

The preserved no-output run in
`/tmp/dam_break_mms2d_3ecrriz2/mms_traveling_interface_2d` used GMRES with
separate finite-difference perturbation sweeps for `velocity` and `pressure`.

Key diagnostics:

- The smoke script parsed two component sweeps: `velocity` and `pressure`.
- GMRES converged in `536` linear iterations to relative residual
  `9.99808e-05` with diagonal FSILS preconditioning.
- The latest whole-vector finite-difference Jacobian relative mismatch was
  `2.84069e-09`.
- The maximum active row/column component-block relative mismatch was
  `2.622930704152584e-04`, from the `velocity -> Pressure` block.
- The other active component-block relative mismatches were
  `velocity -> Velocity[0] = 1.5417375999076666e-05`,
  `velocity -> Velocity[1] = 9.12815679170967e-06`,
  `pressure -> Velocity[0] = 4.533614704234893e-07`,
  `pressure -> Velocity[1] = 2.8026322634344274e-07`, and
  `pressure -> Pressure = 2.840199543899658e-09`.
- The Newton direction check reported relative error `9.99808e-05`, matching
  the requested GMRES residual scale.

This narrows the assembly-mismatch investigation: the fixed-geometry
velocity/pressure matrix action is consistent by block with finite-difference
residual changes on the MMS traveling-interface case. Remaining nonlinear
stagnation should be investigated in geometry coupling, stabilization
conditioning, or solver/preconditioner behavior rather than as a gross
velocity/pressure block assembly mismatch.

## Why `phi` Is Excluded From This Jacobian Filter

The `velocity,pressure` filter intentionally excludes `phi` because this
finite-difference check freezes the cut context while perturbing selected
solution entries. A `phi` perturbation changes the level-set geometry,
active-domain classification, cut-volume quadrature, and cut-adjacent facet
set. Those derivatives are geometry-update derivatives, not fixed-geometry
fluid-block assembly derivatives.

Including `phi` in the same finite-difference check therefore mixes two
questions:

- whether the velocity/pressure residual and tangent are assembled
  consistently for the current cut context;
- whether the monolithic level-set geometry derivative is represented in the
  tangent.

The current assembly diagnostic is aimed at the first question. `phi` should
be checked separately with a geometry-aware perturbation path or with an
explicitly documented fixed-context expectation.

## D18 Diagnostic Timeout Evidence

The preserved D18 no-output diagnostic timeout in
`/tmp/dam_break_d18_kzs02yer/spheric_test05_wet_bed_d18` showed that the
topology and cut-volume counts are internally consistent, but the run remains
too slow for benchmark qualification.

Key diagnostics from that timeout:

- Interior-face timing considered `24648` faces and assembled `1764`, matching
  the generated cut-adjacent facet count.
- Cut-volume timing used indexed traversal and matched generated rule counts.
- The dominant first slow phase was interior-face `matrix=1,vector=1`:
  maximum total `95.34688 s`, maximum kernel `94.705668 s`.
- Later interior-face `matrix=1,vector=0` phases were much smaller:
  maximum total `0.560661 s`, maximum kernel `0.001402 s`.
- A separate cut-volume `matrix=1,vector=0` phase had maximum total
  `6.762353 s`, dominated by insertion at `6.653295 s`.
- Runtime JIT specialization compiles in that timeout were cut-volume-only:
  `CutVolume/Residual=6`, `CutVolume/Tangent=12`.
- Hydrostatic initialization reported no active pressure gauge constraints in
  that fixture copy (`gauge_constraints=0`, `checked_gauge_constraints=0`).

The D18 evidence does not indicate a missing cut-adjacent facet set or a
cut-volume rule-count mismatch. The remaining D18 blockers are concentrated in
the first residual-plus-tangent interior-face kernel phase and a later
cut-volume matrix insertion spike.

## Current Qualification Status

D18/D38 benchmark qualification remains blocked until the residual-plus-tangent
interior-face kernel cost and the cut-volume matrix insertion spike are
reduced or otherwise explained by a repeatable, bounded diagnostic.
