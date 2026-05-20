# MMS Traveling Interface 2D

This is a manufactured moving free-surface test for the new OOP fluid solver
with an unfitted level-set active domain. Negative `phi` denotes liquid.
The default generated mesh uses biquadratic Quad9 cells and `Element_order=2`
so edge and cell-center level-set DOFs define the moving implicit geometry.

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
- `element_order = 2`

The free-surface boundary requests `Generated_interface_geometry=HighOrderImplicit`
with `Implicit_cut_quadrature_backend=SayeHyperrectangle`, fail-fast fallback,
and requested cut volume/interface quadrature order 2.

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
  checks in the wet region.
- `pressure_relative_rms_error`: exact free-surface-gauge pressure check in the
  wet region.
- `interface_pressure_rms`, `interface_pressure_max_abs`: direct pressure check
  on reconstructed `phi=0`.
- `manufactured_residual_x_max`, `manufactured_residual_y_max`,
  `level_set_residual_max`: analytic residual audits for sign/source mistakes.
