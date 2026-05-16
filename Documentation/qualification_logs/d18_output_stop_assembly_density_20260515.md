# D18 Output Stop And Assembly Density Probe

Date: 2026-05-15

## Stop Event

The D18 final-output GMRES qualification attempt was stopped after roughly
42 minutes because the case was still in the solve phase. The wrapper did not
write `/tmp/d18_final_output_capped_scale_20260515.json`, and the preserved
temporary run directory did not contain solver output files. This makes the
current blocker solver throughput rather than VTK writing.

## No-Output Reference

The completed D18 no-output configured-time GMRES run reached 312 accepted
steps at time 0.156 with all nonlinear and linear solves converged. The parsed
diagnostics show:

- Assembly timing records: 892.
- Nonlinear iterations: 578.
- Extra assembly timing records beyond nonlinear iterations: 314.
- Assembly timing records per accepted step: 2.858974.
- Extra assembly timing records per accepted step: 1.006410.
- Cut-context rebuilds: 2404.
- Cut-context rebuilds per accepted step: 7.705128.
- Cut-context rebuild provenance counts: accepted 890,
  jacobian_and_residual 890, before_physics_solve 312, accepted_step 312.

Assembly timing remained dominated by repeated cut-adjacent and cut-volume
traversal:

- Total assembly time sum: 3478.8 s.
- Interior-face assembly sum: 2001.4 s.
- Cut-volume assembly sum: 1329.0 s.
- Cell-term assembly sum: 148.4 s.

## Short Probe Comparison

A 3-step D18 no-output GMRES probe with the default no-line-search path
completed successfully and produced:

- Assembly timing records: 8.
- Nonlinear iterations: 3.
- Extra assembly timing records beyond nonlinear iterations: 5.
- Assembly timing records per accepted step: 2.666667.
- Cut-context rebuilds per accepted step: 6.0.
- Cut-context rebuild provenance counts: accepted 6,
  jacobian_and_residual 6, before_physics_solve 3, accepted_step 3.

A 3-step D18 no-output GMRES probe with `SVMP_NEWTON_LINE_SEARCH=1` also
completed the solver portion, but failed the smoke threshold that requires a
capped cut-adjacent scale because the first three steps did not encounter a
capped sliver. Its diagnostics showed:

- Assembly timing records: 8.
- Cut-context rebuild provenance counts: accepted 6,
  before_physics_solve 3, jacobian_and_residual 3, line_search_trial 3,
  accepted_step 3.

The line-search path replaced three final combined assembly refreshes with
three residual trial refreshes, but it did not reduce the total number of
expensive assembly traversals in the short window.

## Conclusion

The output-enabled D18 run should remain blocked until the solver reduces
repeated cut-adjacent and cut-volume assembly work per accepted step. The next
guardrail is to parse and threshold assembly-density metrics in the smoke
script so future no-output runs can fail quickly when assembly traversal or
cut-context rebuild density regresses.
