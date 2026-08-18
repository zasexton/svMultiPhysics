# Linear Sloshing 2D

This is a small-amplitude standing-wave free-surface regression test for the
new OOP incompressible Navier-Stokes solver with an unfitted level-set active
domain. Negative `phi` denotes liquid.

See `LEVEL_SET_FREE_SURFACE_SUPPORT_STATUS.md` for the current support-status
notes, known remaining issues, and acceptance criteria for considering the
unfitted level-set free-surface method fully supported.

The analytic reference is the linearized inviscid potential-flow solution in a
rectangular tank. It is exact for impermeable slip walls and zero surface
tension, not for viscous no-slip Navier-Stokes. The solver XML uses exact
time/space Dirichlet velocity data on the left, right, and bottom walls because
that is the currently supported OOP fallback for the slip-wall analytic data.
The level-set transport equation is coupled directly to the fluid `Velocity`
field and includes a weak kinematic residual on the generated free-surface
interface. The interface term keeps the zero contour tied to the free-surface
normal velocity instead of letting the volume transport average the standing
wave velocity through the tank depth.

This case enables the Navier-Stokes unfitted free-surface PDE velocity
extension. The active wet-domain Navier-Stokes terms still assemble only on
`Active_domain=LevelSetNegative`; the extension term assembles on the inactive
generated cut-volume side, so the active cut context must retain both generated
volume sides. This remains a one-fluid free-surface benchmark with a passive
exterior, not a two-phase jump-condition case. In this mode, inactive velocity
DOFs are not homogeneously clamped; they are owned by the extension PDE.

Default parameters:

- `L = 1.0`
- `H0 = 0.5`
- `H_tank = 0.75`
- `A = 0.005`
- `k = 1*pi/L = 3.14159265359`
- `omega = 5.31655337432`
- `period = 1.1818155231`
- `final_time = 0.246211567314`

The free surface is

```text
h(x,t) = H0 + A*cos(k*x)*cos(omega*t)
phi(x,y,t) = y - h(x,t)
```

The pressure reference is zero at the free surface to linear order:

```text
p = rho*g*(H0-y)
  + rho*(A*omega^2/k)*(cosh(k*y)/sinh(k*H0))*cos(k*x)*cos(omega*t)
```

## Analytic Reference

This test uses the first antisymmetric standing sloshing mode from linear
potential-flow theory in a rectangular tank. The assumptions are:

- incompressible liquid
- inviscid and irrotational motion
- small free-surface amplitude
- zero surface tension
- impermeable slip walls and bottom
- a linearized free-surface kinematic and dynamic condition at `y=H0`

With the vertical coordinate measured upward from the tank bottom, the mode
family is

```text
k_n = n*pi/L
omega_n^2 = g*k_n*tanh(k_n*H0)
eta_n(x,t) = A*cos(k_n*x)*cos(omega_n*t)
```

For the default `n=1`, `L=1`, and `H0=0.5`, this gives
`omega = 5.31655337432 rad/s`.

The velocity potential used by the verifier is

```text
Phi(x,y,t) =
  -(A*omega/k)*(cosh(k*y)/sinh(k*H0))*cos(k*x)*sin(omega*t)
```

so that `u = grad(Phi)`. The dynamic pressure follows from the linearized
Bernoulli relation:

```text
p = rho*g*(H0-y) - rho*d(Phi)/dt
```

Using `omega^2 = g*k*tanh(k*H0)`, the dynamic pressure may also be written as

```text
p_dyn =
  rho*g*A*(cosh(k*y)/cosh(k*H0))*cos(k*x)*cos(omega*t)
```

which is algebraically equivalent to the pressure formula above. These
expressions are therefore appropriate as a linear sloshing regression target,
but they should not be interpreted as the exact solution of a nonlinear,
viscous, no-slip Navier-Stokes free-surface problem.

References:

- O. M. Faltinsen and A. N. Timokha, *Sloshing*, Cambridge University Press,
  2009. This is the primary reference for linear and nonlinear sloshing modal
  theory in partially filled tanks.
- O. M. Faltinsen, R. Firoozkoohi, and A. N. Timokha, "Analytical modeling of
  liquid sloshing in a two-dimensional rectangular tank with a slat screen,"
  *Journal of Engineering Mathematics* 70, 93-109, 2011.
  https://doi.org/10.1007/s10665-010-9397-5
- W. Xue et al., "Numerical Investigation of Sloshing in Rectangular Tank with
  Permeable Baffle," *Journal of Marine Science and Engineering* 8(9), 671,
  2020. Section 3.1 compares a clean rectangular-tank linear sloshing case
  against Faltinsen's analytical solution and reports the same `L=1`, `h=0.5`
  lowest natural frequency, about `5.316 rad/s`.
  https://www.mdpi.com/2077-1312/8/9/671
- For the finite-depth linear-wave velocity and pressure forms used to check
  signs and hydrostatic plus dynamic pressure decomposition, see:
  https://www.coastalwiki.org/wiki/Shallow-water_wave_theory and
  https://geo.libretexts.org/Bookshelves/Oceanography/Coastal_Dynamics_%28Bosboom_and_Stive%29/11%3A_Appendix_A_-_Linear_wave_theory

## Generate

```bash
python3 generate_case.py
```

Useful smoke-test override:

```bash
python3 generate_case.py --nx 16 --ny 16 --time-steps 100 --time-step 0.00492423134627
```

## Run

```bash
/path/to/svmultiphysics solver.xml
```

The XML saves VTK output at the requested cadence and combines the time series
into `result.pvd`.

## Verify

```bash
python3 verify_expected_results.py
```

To check the generated initial condition without running the solver:

```bash
python3 verify_expected_results.py mesh/background/mesh-complete.mesh.vtu --time 0
```

The verifier reconstructs the `phi=0` crossings, deduplicates them, fits the
standing-wave mode, clips the active `phi<=0` liquid area, compares velocity
and pressure in wet nodes, and checks pressure interpolated onto the free
surface. The default tolerances are smoke/regression tolerances for a coarse
mesh, not an accuracy benchmark.

Key metrics:

- `interface_mean`, `interface_cos_coeff`, `interface_sin_coeff`: modal fit of
  the reconstructed free surface.
- `interface_l2_height_error`, `interface_max_height_error`: geometric error
  against the analytic free-surface height.
- `relative_area_error`: active-liquid volume conservation for the zero-mean
  standing wave.
- `velocity_relative_l2_error`: wet-region velocity error against the
  potential-flow field.
- `pressure_relative_rms_error`: absolute-gauge pressure error.
- `pressure_relative_rms_error_after_constant_offset_removal`: pressure-plane
  error after removing one gauge offset.
- `interface_pressure_rms`, `interface_pressure_max_abs`: direct pressure check
  on the reconstructed free surface.
