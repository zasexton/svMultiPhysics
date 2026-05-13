# Moving Free-Surface Validation Cases

This document defines the validation targets for the moving free-surface
implementation.  Each case is intended to be run for both supported surface
representations when the required features are enabled:

- fitted ALE free surface
- unfitted level-set free surface with generated interface geometry

Common reported quantities:

- free-surface elevation or interface position
- pressure along a vertical probe line
- maximum velocity magnitude
- water volume or area
- generated-interface measure and normals for level-set runs
- mesh quality metrics for fitted ALE runs

## Generated Literature Geometry Fixtures

The following mesh fixtures expand the open-vessel coverage beyond the small
analytic examples. They are generated with PyVista, TetGen, and the MMG
executables bundled with the `svv2` environment:

```bash
conda run -n svv2 python tests/cases/fluid/open_vessel_free_surface/generate_validation_meshes.py
```

Generated fitted ALE fixtures:

- `tests/cases/fluid/open_vessel_free_surface/fitted_ale/spheric_test10_lateral_water_1x`

Generated unfitted level-set fixtures:

- `tests/cases/fluid/open_vessel_free_surface/unfitted_level_set/spheric_test10_lateral_water_1x`
- `tests/cases/fluid/open_vessel_free_surface/unfitted_level_set/spheric_test05_wet_bed_d18`
- `tests/cases/fluid/open_vessel_free_surface/unfitted_level_set/spheric_test05_wet_bed_d38`
- `tests/cases/fluid/open_vessel_free_surface/unfitted_level_set/spheric_test02_dambreak_obstacle`

Each generated case includes:

- `solver.xml`
- `pressure_gauge.csv`
- `benchmark.json`
- `mesh/.../mesh-complete.mesh.vtu`
- `mesh/.../mesh-surfaces/*.vtp`

The benchmark metadata records the published dimensions and source URLs so the
meshes can be regenerated without searching through the literature again.

## Implementation Ownership

Navier-Stokes owns free-surface boundary semantics for both fitted ALE and
unfitted level-set cases. For unfitted cases, Navier-Stokes consumes
`FE::level_set` generated interface domains, curvature helpers, cut-cell
metadata, and level-set diagnostics, then installs the free-surface pressure
jump, surface-tension, kinematic, and stabilization terms. Reusable level-set
transport, volume, reinitialization, diagnostics, restart, and generated
interface lifecycle code belongs in `Code/Source/solver/FE/LevelSet`.

## Open Tank At Rest

Purpose: verify that the half-filled open-vessel examples remain in hydrostatic
balance under gravity.

Starting points:

- `tests/cases/fluid/open_vessel_free_surface/fitted_ale/solver.xml`
- `tests/cases/fluid/open_vessel_free_surface/unfitted_level_set/solver.xml`

Setup:

- water density `998.2`
- dynamic viscosity `1.003e-3`
- gravity `(0, -9.81, 0)`
- atmospheric pressure `0`
- initial surface at the hydrostatic pressure reference point
- zero initial velocity

Reference behavior:

- the free surface remains flat
- velocity stays near zero
- pressure remains `p = rho g (eta - y)` up to the gauge offset
- volume remains constant

Acceptance criteria:

- maximum velocity stays below the selected nonlinear tolerance scale
- free-surface displacement remains below one percent of the cell height
- hydrostatic pressure error converges under mesh refinement
- relative volume drift is below `1.0e-4` for fitted ALE and below the configured
  level-set volume-correction tolerance for unfitted runs

## Small-Amplitude Sloshing

Purpose: validate gravity-wave motion in a rectangular tank against linear
sloshing theory.

Setup:

- rectangular tank of length `L` and water depth `h`
- initial surface perturbation `eta(x, 0) = a cos(pi x / L)` with `a / h <= 0.01`
- zero or analytically compatible initial velocity
- surface tension disabled unless explicitly testing capillary-gravity waves

Reference frequency for mode `n`:

```text
k_n = n pi / L
omega_n^2 = g k_n tanh(k_n h)
```

With surface tension enabled:

```text
omega_n^2 = (g k_n + gamma k_n^3 / rho) tanh(k_n h)
```

Measured outputs:

- surface elevation at one or more probe points
- dominant frequency from zero crossings or spectral peak
- volume drift over the run

Acceptance criteria:

- dominant frequency within two percent of the analytic value on the reference
  mesh
- frequency converges toward the analytic value with refinement
- no secular mean-surface drift beyond the volume-conservation tolerance

## SPHERIC Test 10 Sloshing Wave Impact

Purpose: add a published sloshing tank geometry with pressure-gauge data and
large free-surface motion.

Starting points:

- fitted ALE: `tests/cases/fluid/open_vessel_free_surface/fitted_ale/spheric_test10_lateral_water_1x/solver.xml`
- unfitted level set: `tests/cases/fluid/open_vessel_free_surface/unfitted_level_set/spheric_test10_lateral_water_1x/solver.xml`

Setup:

- rectangular tank length `0.900 m`
- 1x tank breadth `0.062 m`
- lateral-water fill height `0.093 m`
- water at `19 C`
- published roll-angle history and pressure records from SPHERIC Test 10

Reference behavior:

- lateral wave impact timing and pressure history at the reported side-wall
  sensor
- overturning and breaking-wave onset for the low-fill water case
- volume conservation across repeated sloshing periods

Acceptance criteria:

- impact timing follows the published pressure trace after tank-motion forcing
  is wired into the executable case
- pressure peaks converge toward the experimental envelope under refinement
- fitted ALE is restricted to pre-breaking or non-overturning comparisons;
  unfitted level set owns the breaking-wave comparison

## Capillary Wave Oscillation

Purpose: validate surface-tension curvature and pressure-jump coupling.

Setup:

- small sinusoidal interface perturbation with wavelength `lambda`
- gravity disabled for a pure capillary check, or enabled for a capillary-gravity
  check
- low viscosity for the oscillation-frequency check
- amplitude `a / lambda <= 0.01`

Reference frequency:

```text
k = 2 pi / lambda
omega^2 = gamma k^3 / rho
```

For finite depth or gravity, use:

```text
omega^2 = (g k + gamma k^3 / rho) tanh(k h)
```

Measured outputs:

- interface amplitude over time
- pressure jump across the interface
- curvature diagnostics

Acceptance criteria:

- dominant frequency within two percent of the reference value
- measured pressure jump matches `gamma kappa` in phase with curvature
- curvature and interface normals converge under refinement

## Capillary Wave Decay

Purpose: validate viscous damping of small capillary waves.

Setup:

- same geometry and perturbation as the capillary oscillation case
- viscosity enabled
- small amplitude to remain in the linear regime

Deep-water reference decay rate:

```text
eta_amplitude(t) = eta_amplitude(0) exp(-2 nu k^2 t)
nu = mu / rho
```

Measured outputs:

- logarithmic decrement of interface amplitude
- oscillation frequency
- kinetic plus surface-energy trend

Acceptance criteria:

- fitted exponential decay rate within five percent of `2 nu k^2`
- decay-rate error decreases under time-step and mesh refinement
- total energy decreases monotonically after the initial transient

## Dam Break Or Water-Column Collapse

Purpose: validate large free-surface deformation after the methods are stable on
small-amplitude cases.

Setup:

- rectangular water column of initial width `a` and height `H`
- dry or low-density void region represented by the free surface
- no surface tension for the standard gravity-collapse comparison
- wall no-slip or slip policy recorded with the result

Reference behavior:

- early-time front location compared with the Martin-Moyce water-column collapse
  benchmark or an equivalent published data set
- splash and impact timing compared qualitatively until a higher-resolution
  reference is established

Measured outputs:

- leading-front position versus time
- free-surface profile snapshots
- volume drift
- maximum velocity and time-step history

Acceptance criteria:

- nondimensional front-position curve follows the selected benchmark envelope
- qualitative splash timing is stable under refinement
- no negative volume or interface loss occurs in the level-set representation

### SPHERIC Test 05 Wet-Bed Dam Break

Starting points:

- `tests/cases/fluid/open_vessel_free_surface/unfitted_level_set/spheric_test05_wet_bed_d18/solver.xml`
- `tests/cases/fluid/open_vessel_free_surface/unfitted_level_set/spheric_test05_wet_bed_d38/solver.xml`

Setup:

- initial dam height `d0 = 0.15 m`
- wet-bed depths `d = 0.018 m` and `d = 0.038 m`
- thin three-dimensional extrusion of the published two-dimensional setup
- level-set field initialized as the union of the wet bed and retained water
  column
- negative level-set values denote the water side
- fluid volume terms use `Active_domain=LevelSetNegative` and
  `Active_domain_method=CutVolume`
- `Active_domain_method=SmoothedIndicator` is available only as a diagnostic
  fallback. If `Active_domain_smoothing_width` is omitted or set to zero, the
  transition width is derived from local cell diameter; accepted D18/D38 runs
  continue to require exact cut-volume integration.

Reference behavior:

- leading-front velocity over the first meters of travel
- free-surface profiles in the published digitized snapshot window

### SPHERIC Test 02 Three-Dimensional Dam Break With Obstacle

Starting point:

- `tests/cases/fluid/open_vessel_free_surface/unfitted_level_set/spheric_test02_dambreak_obstacle/solver.xml`

Setup:

- tank `3.22 m x 1.00 m x 1.00 m`
- initial water column length `0.58 m`, height `0.55 m`, width `1.00 m`
  at the right end of the tank
- fixed obstacle `0.40 m x 0.16 m x 0.16 m`
- obstacle centered at `x = 0.82 m`, `z = 0.50 m`

Reference behavior:

- impact time near the obstacle
- water-height gauge histories
- pressure response on the obstacle face
- splash height and reflected-wave timing

## Long Transient Volume Conservation

Purpose: quantify free-surface volume preservation over many advection and
mesh-motion updates.

Setup:

- quiescent tank, sloshing tank, or translating level-set interface
- run for at least `1000` time steps or ten dominant sloshing periods
- save volume diagnostics at every output interval

Measured outputs:

- water volume or area
- relative volume change
- level-set correction shift and correction iterations
- fitted mesh boundary displacement measure

Acceptance criteria:

- fitted ALE relative volume drift below `1.0e-4` on the reference mesh
- unfitted level-set relative volume drift below the configured correction
  tolerance
- no monotone volume loss trend remains after correction is enabled

## Contact-Angle Static Meniscus

Purpose: validate wall contact-line and prescribed contact-angle behavior.

Setup:

- narrow rectangular channel or open tank with a prescribed wall contact angle
  `theta`
- surface tension enabled
- gravity optional; include gravity for capillary-rise comparison
- initial interface close to the expected static meniscus

Reference behavior:

- interface normal at the wall satisfies the prescribed contact angle
- pressure jump follows the Young-Laplace relation
- with gravity, capillary rise or depression follows the standard static balance

Measured outputs:

- contact angle reconstructed from the interface normal and wall normal
- static interface profile
- pressure jump across the interface
- active contact measure and skipped-contact diagnostics

Acceptance criteria:

- reconstructed contact angle within two degrees of the prescribed value away
  from deliberately skipped degeneracies
- static pressure jump matches `gamma kappa`
- contact diagnostics report deterministic active contact measure under
  refinement and MPI partitioning
