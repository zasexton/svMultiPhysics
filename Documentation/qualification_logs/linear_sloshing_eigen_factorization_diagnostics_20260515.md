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
