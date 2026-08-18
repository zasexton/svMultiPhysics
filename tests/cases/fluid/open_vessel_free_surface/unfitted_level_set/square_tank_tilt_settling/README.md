# Square Tank Tilt Settling

This is a small 2D smoke/regression case for the new OOP incompressible
Navier-Stokes unfitted level-set free-surface path. The tank starts half full
with a horizontal level set. At `t = 0`, gravity is resolved in a frame tilted
by 10 degrees, so the transient should move toward the analytic hydrostatic
line recorded in `expected_results.json`.

Negative `phi` values denote the active wet region. The OOP VTK output writes
the background mesh for context and preserves unmasked `Velocity`, `Pressure`,
and `Divergence` values so reconstructed-interface diagnostics can interpolate
fields across `phi = 0`. It also writes an `ActiveFluid` point indicator. In
ParaView, threshold or clip to `ActiveFluid = 1` or `phi <= 0` before
interpreting pressure and velocity as active-fluid fields.
For the initial horizontal surface, dry vertices in retained cut cells carry
the signed continuation `rho*g*(fill_height-y)`. Those negative coefficients
make the P1 pressure trace exactly zero at `phi=0`; they are interpolation data,
not a modeled gas pressure.
For interface advection, the level-set equation uses a generated prescribed
`LevelSetAdvectionVelocity` rebuilt from the wet-region physical velocity and
extended into dry vertices. This avoids treating dry-side `Velocity` values as
physical transport data for `phi`.

The expected static balance is

```text
grad(p) = rho * body_force
body_force = (g sin(theta), -g cos(theta), 0)
p = rho * dot(body_force, x - x_ref)
```

With `theta = 10 degrees` and `x_ref = (0.5, 0.5, 0)`, the zero-pressure free
surface is perpendicular to the body force:

```text
y = intercept + slope*x
slope = tan(theta)
intercept = 0.5 - 0.5*slope
```

For the half-full unit square, this gives area `0.5`, centroid
`(0.5293878301180774, 0.25259093367714697)`, and zero equilibrium velocity.

Regenerate the case files with:

```bash
python3 generate_case.py
```

Run from this directory with:

```bash
/path/to/svmultiphysics solver.xml
```

The XML saves every transient step as `result_*.vtu` and writes `result.pvd`
so the full advancing level-set and flow fields can be loaded as one time
series.

Check the latest saved result with:

```bash
python3 verify_expected_results.py
```

The verifier checks final interface geometry, active-fluid area/centroid,
wet-region velocity, wet-region pressure, hydrostatic pressure gradient,
pressure on the reconstructed free surface, a diagnostic Q1 reconstructed
normal/tangential traction trace, probe pressures, and consistency between
`solver.xml` and `expected_results.json`.

The default mesh is intentionally coarse (`9 x 9` quads). Treat this as a fast
CI smoke/regression case, not a high-accuracy benchmark. The default transient
run is a finite-time settling smoke, not a fully damped long-time settling
study: it gates bounded motion toward the analytic slope, active-domain volume,
pressure-gradient, and free-surface pressure diagnostics, but does not claim
the final interface has reached the exact hydrostatic line. For an isolated
static-balance check with strict final-equilibrium slope/intercept checks,
regenerate a companion mode with:

```bash
python3 generate_case.py --initial-state equilibrium --time-steps 5
```

That mode initializes `phi`, pressure, and velocity directly from the analytic
hydrostatic equilibrium and is better suited to strict velocity-decay checks.
Short refined transient probes can use `--verification-profile early_transient`
to defer final-equilibrium slope/intercept and pressure-gradient gates, or
`--verification-profile transient_pressure` to re-enable the hydrostatic
pressure-gradient check while still deferring final-equilibrium intercept
closure.
For staged contact-line-aware checks, `--verification-profile
transient_pressure_interior` keeps the same hydrostatic pressure-gradient,
volume, finite-interface, and probe-pressure gates, applies the free-surface
pressure RMS/max tolerances to interior interface samples, and keeps
wall-contact endpoint pressure as bounded diagnostics.
Stabilization sweeps should be generated explicitly with
`--use-cut-metadata-scale`, `--cut-cell-metadata-scale-cap`,
`--cut-cell-velocity-gradient-penalty`, and
`--cut-cell-pressure-gradient-penalty`; the default smoke keeps cut metadata
scaling disabled, leaves the metadata cap unset, and uses unit
velocity/pressure gradient penalties.
The reconstructed stress metrics are diagnostic-only and use Q1 cell gradients
from the written background field. They are kept to distinguish a true
normal-traction residual from a pressure-only interpretation issue; the staged
acceptance gates still use the interpolated pressure trace unless explicitly
changed in a later verifier contract.
The verifier also reports segment-length-weighted interface-pressure metrics.
Those diagnostics approximate a boundary-measure RMS over reconstructed cut
segments and help separate broad free-surface pressure error from localized
endpoint overshoot. The point-sample and segment-endpoint max locations are
reported for localization; these diagnostics do not relax the existing
point-sample pressure gates.
Boundary/contact-line and interior subsets are reported separately so staged
transient failures can distinguish wall-interface endpoint pressure overshoot
from interior free-surface pressure error.
