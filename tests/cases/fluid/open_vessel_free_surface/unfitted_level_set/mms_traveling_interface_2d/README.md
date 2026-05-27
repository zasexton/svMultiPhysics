# MMS Traveling Interface 2D

This is a manufactured moving free-surface test for the new OOP fluid solver
with an unfitted level-set active domain. Negative `phi` denotes liquid.
The default generated mesh uses biquadratic Quad9 cells and `Element_order=2`
so edge and cell-center level-set DOFs define the moving implicit geometry.
The fluid block uses the Taylor-Hood P2/P1 velocity-pressure pair while the
moving high-order pressure ghost penalty remains disabled.

The exact fields are

```text
X(t) = (U0/Omega)*sin(Omega*t)
U(t) = U0*cos(Omega*t)
h(x,t) = H0 + A*cos(k*(x-X(t)))
phi(x,y,t) = y - h(x,t)
u(x,y,t) = [U(t), 0, 0]
p(x,y,t) = rho*g*(h(x,t)-y)
```

The free-surface pressure is exactly zero at `phi=0`, and the velocity
gradient is zero, so viscous free-surface stress does not alter the scalar
pressure check.

The required manufactured source is

```text
s_x(x,t) = Udot(t) + g*h_x(x,t)
         = -U0*Omega*sin(Omega*t) - g*A*k*sin(k*(x-X(t)))
s_y(x,t) = 0
```

The OOP residual convention is `rho*(... - f) + grad(p) - div(stress)`, so
hydrostatic balance uses `grad(p)=rho*f`. With `Force_y=-g`, the source above
is the acceleration-like extra body term required to make the MMS exact.

## Solver Source Wiring

The generated XML sets
`Momentum_source_temporal_and_spatial_values_file_path=bc/momentum_source.dat`,
so the OOP Navier-Stokes residual consumes the time- and spatially-varying
manufactured acceleration in addition to constant `Force_x/y/z`.
`manufactured_source_samples.csv` is still generated as an independent audit of
the source values used by the verifier.

Default parameters:

- `L = 1.0`
- `H0 = 0.5`
- `H_tank = 0.75`
- `A = 0.02`
- `k = 6.28318530718`
- `U0 = 0.1`
- `Omega = 6.28318530718`
- `period = 1`
- `final_time = 1`
- `time_step = 0.005`
- `time_steps = 200`
- `spectral_radius_of_infinite_time_step = 0.5`
- `element_order = 2`

The free-surface boundary requests `Generated_interface_geometry=HighOrderImplicit`
with `Implicit_cut_quadrature_backend=SayeHyperrectangle`, fail-fast fallback,
requested cut volume/interface quadrature order 2, and subdivision depth
6.
The qualified high-order smoke contract uses the Eigen direct backend. The
accepted one-step evidence is from a temporary copy with VTK/time-series output
disabled and `Number_of_time_steps=1`; the default generated case keeps the
full-period 200-step setup for follow-up transient studies.

The high-order implicit path is refreshed-frozen for the Navier-Stokes
linearization: generated interface and cut-volume quadrature are rebuilt from
the current nonlinear level-set state, but high-order quadrature sensitivities
are not part of the default Jacobian. This case also enables the Navier-Stokes
PDE velocity extension, so the cut context must retain both active and
inactive generated cut-volume sides. Physical Navier-Stokes volume terms remain
one-fluid and active-side only. The inactive-side velocity clamp is disabled in
this mode so the extension PDE can solve those inactive velocity DOFs.

The case uses
`Cut_cell_pressure_stabilization_policy=DisabledForRefreshedFrozenHighOrder`:
velocity cut-cell stabilization remains enabled, while pressure jump penalties
are omitted for high-order implicit geometry with refreshed-frozen tangents
until that pressure stabilization is separately qualified for this moving
free-surface Jacobian contract.

The velocity ghost penalty is intentionally left unscaled by cut metadata
(`Use_cut_metadata_scale=false`) in this high-order MMS. The refreshed-frozen
moving-interface probe accepts with the standard h-scaled velocity penalty,
while the metadata scale can over-amplify small-cut facets and stall the
coarse nonlinear acceptance gate.

The current full-size high-order smoke uses
`Cut_cell_velocity_max_derivative_order=1` and
`Cut_cell_velocity_gradient_penalty=0.1`. A JIT+Eigen direct one-step copy at
`Time_step_size=0.005`, depth 6, and fluid `Max_iterations=60` accepted in 9
Newton iterations with fallback-free initial and accepted-step high-order cut
contexts and zero reported wet-volume drift. Broader transient and refinement
validation remain separate qualification work.

The default level-set advection velocity is the wet-extension prescribed field
generated from the fluid velocity. For topology diagnostics, regenerate with `--level-set-velocity-source constant --level-set-constant-velocity 0.1 0.0 0.0` to use a prescribed constant horizontal transport velocity and remove feedback from the solved fluid velocity field. This is a topology/isolation diagnostic; for `Omega != 0`, it is not the exact time-dependent MMS velocity `U0*cos(Omega*t)` after the initial instant. Regenerate with `--level-set-exact-inflow` to add exact time/space side-wall level-set inflow data for boundary-state diagnostics.
Level-set transport defaults to the advective form with SUPG enabled; diagnostics
can regenerate with `--disable-level-set-supg` or
`--level-set-transport-form conservative_divergence` to isolate transport-form
and stabilization sensitivity.

High-order acceptance runs should use an executable built with
`FE_ENABLE_LLVM_JIT=ON`. The solver XML requests `jit=true`, but a build without
LLVM JIT support falls back to the symbolic interpreter path and makes the
high-order cut-volume matrix assembly impractical for this MMS gate.

## Generate

```bash
python3 generate_case.py
```

Useful linear-geometry comparison override:

```bash
python3 generate_case.py --element-order 1
```

## Run

```bash
/path/to/svmultiphysics solver.xml
```

## Verify

```bash
python3 verify_expected_results.py
```

To check the generated initial condition without running the solver:

```bash
python3 verify_expected_results.py mesh/background/mesh-complete.mesh.vtu --time 0
```

The verifier reconstructs and deduplicates `phi=0`, fits the cosine mode,
checks area and y-centroid, compares `phi`, velocity, and pressure against the
exact fields, interpolates pressure onto the free surface, and reports analytic
momentum and level-set residual audits.

Key metrics:

- `phi_rms_error`, `phi_max_abs_error`: level-set advection error.
- `interface_cos_coeff`, `interface_sin_coeff`, `interface_shift_error`:
  reconstructed free-surface phase/translation.
- `area_relative_error`, `centroid_y_error`: cut-volume active-domain checks.
- `velocity_relative_l2_error`, `velocity_mean_x_error`: exact uniform-flow
  checks in the legacy mesh-size-dependent wet region.
- `pressure_relative_rms_error`: exact free-surface-gauge pressure check in the
  legacy mesh-size-dependent wet region.
- `bulk_velocity_relative_l2_error`, `bulk_pressure_relative_rms_error`: exact
  field checks in a fixed-clearance bulk wet region for more comparable compact
  refinement diagnostics.
- `quadrature_phi_l2_error`, `quadrature_velocity_relative_l2_error`,
  `quadrature_pressure_relative_rms_error`, and their `bulk_quadrature_*`
  counterparts: FE-style cell-quadrature weighted diagnostics used to separate
  integral field-error trends from mesh-node sampling artifacts.
- `interior_phi_l2_error`, `quadrature_interior_phi_l2_error`: fixed
  side-wall-clearance level-set diagnostics used to distinguish domain-interior
  transport behavior from side-wall boundary-band effects.
- `interface_pressure_rms`, `interface_pressure_max_abs`: direct pressure check
  on reconstructed `phi=0`; finite-sample checks are enforced, while the
  absolute pressure magnitude is diagnostic-only for this moving high-order cut
  case.
- `manufactured_residual_x_max`, `manufactured_residual_y_max`,
  `level_set_residual_max`: analytic residual audits for sign/source mistakes.
