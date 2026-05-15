# Linear Sloshing Eigen Factorization Diagnostics - 2026-05-15

This note records the direct Eigen diagnostic probe for
`tests/cases/fluid/open_vessel_free_surface/unfitted_level_set/linear_sloshing_2d`.

## Probe Setup

- Built `svmultiphysics` with `FE_ENABLE_EIGEN=ON`.
- Ran a temporary one-step copy of `linear_sloshing_2d`.
- Disabled VTK output and PVD combination in the temporary copy only.
- Kept the fixture solver selection as `<LS type="Direct">` with
  `<Linear_algebra type="eigen">`.

## Result

The temporary run reached the Eigen backend and selected the direct solver, then
failed during `Eigen::SparseLU` factorization:

```text
backend=eigen
linear solver method=direct
EigenLinearSolver (direct): factorization failed
```

The new factorization diagnostic reported no nonfinite entries and no
structural empty rows, but did report zero numerical rows and columns in the
pressure block:

```text
info=numerical_issue rows=1156 cols=1156 structural_nnz=50470 numeric_nnz=16566
nonfinite_entries=0 structural_empty_rows=0 zero_rows=68 zero_cols=68
zero_diag=68 rhs_norm=350.30780308029796
Pressure{begin=867,end=1156,zero_rows=68,zero_cols=68,
zero_rows_first_local=189|190|191|192|193|194|195|196,
zero_cols_first_local=189|190|191|192|193|194|195|196}
```

The initial mesh analytical verifier still passed at `time=0`, so the fixture
initial fields and analytical comparison data are consistent. The direct
factorization failure is therefore controlled by the assembled monolithic
operator, not by the analytical fixture definition.

## Interpretation

The failing direct system contains inactive or unanchored pressure DOFs whose
assembled rows and columns are numerically zero. This is a formulation/assembly
issue for the unfitted active-domain path: passive-side pressure unknowns are
left in the monolithic system without equations or constraints. A direct sparse
factorization exposes the defect immediately, while iterative or block solvers
can obscure it through preconditioning and tolerance behavior.

The next implementation step should constrain or eliminate inactive pressure
DOFs for the generated wet active domain, then rerun the one-step direct Eigen
probe and the analytical verifier.

## Run-Length Follow-Up

The direct diagnostic now reports all zero pressure rows and columns as
run-length ranges. In the one-step `linear_sloshing_2d` probe, the final
nonlinear assembly still failed factorization with:

```text
zero_rows=68 zero_cols=68
zero_row_runs=1056-1091|1124-1155
Pressure{begin=867,end=1156,zero_rows=68,zero_cols=68,
zero_row_runs_local=189-224|257-288,
zero_col_runs_local=189-224|257-288}
```

Mapping those pressure-local ranges back to the initial mesh shows that every
zero row is on the dry side of the interface (`phi > 0`). A gauge-only rerun
with a valid wet pressure constraint still failed with the same final
pressure-local zero-row ranges, so the direct factorization failure is not
only a missing pressure reference.

Disabling cut-cell stabilization in the temporary copy increased the final
pressure zero-row count from 68 to 85 and restored the broader initial ranges:

```text
Pressure{zero_rows=85,zero_cols=85,
zero_row_runs_local=166-168|171-172|183-184|187-224|235-236|239-240|251-252|255-288}
```

This indicates that cut-adjacent stabilization currently fills some dry
pressure rows, but it does not provide a complete or principled elimination of
inactive pressure unknowns. The factorization failure is therefore exposing a
real active-domain algebra problem: pressure DOFs outside the wet support
remain in the monolithic system, and stabilization can only partially mask the
singularity.
