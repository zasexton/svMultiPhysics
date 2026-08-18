# Bioreactor Physics Additions Plan

## Summary

This plan describes the Physics-folder additions needed to simulate a sparged,
agitated bioreactor for oxygen transfer, gas dispersion, impeller operation,
species transport, cell oxygen uptake, and biological stress metrics.

The current Physics library already provides useful foundations:

- incompressible Navier-Stokes VMS for liquid flow
- moving-domain and free-surface hooks in the Navier-Stokes module
- harmonic and pseudo-elastic mesh-motion modules
- Poisson-style scalar diffusion baselines
- Newtonian and Carreau-Yasuda fluid viscosity models
- FSI and thermal interface coupling contracts
- equation module registry and parameter parsing infrastructure

Those pieces are not sufficient for a complete sparged bioreactor model. The
missing Physics capabilities are mostly equation sets, constitutive closures,
couplings, and bioreactor-specific outputs. They should consume the generic FE
infrastructure for state groups, admissibility, bounded updates, paired
exchange, reductions, exposure accumulation, wall-distance queries,
moving-frame metadata, and sliding-interface metadata.

## Scope Boundary

Physics should own:

- equation sets
- source and sink laws
- constitutive closures
- interphase transfer, chemistry, biology, turbulence, and impeller models
- module-level parameter validation
- bioreactor-specific outputs and constraints

Physics should not duplicate FE responsibilities:

- DOF ownership
- assembly loops
- sparsity construction
- linear algebra backends
- generic reductions
- generic bounded-update infrastructure
- generic state-group metadata
- generic geometry queries
- generic sliding-interface transfer mechanics

Physics should also not own whole-scenario orchestration. A stirred, sparged
bioreactor is a coupled simulation setup assembled from reusable Physics
modules. The composition of liquid flow, rotating machinery, turbulence,
multiphase gas, scalar transport, mass transfer, uptake, chemistry, heat, and
outputs should be specified through solver XML and application-level module
orchestration, not through a dedicated `BioreactorModule`.

## Implementation Phases

### Phase 1: Transient Scalar Transport

Goal: add a reusable transport equation for dissolved oxygen, dissolved CO2,
tracers, cell density, gas species scalars, and temperature-like transported
quantities.

Proposed location:

`Physics/Formulations/Transport/`

Needed types:

- `ScalarTransportModule`
- `ScalarTransportOptions`
- `TransportBCFactories`
- `TransportRegister`
- optional `TransportPostProcessing`

Equation:

```text
dC/dt + div(u C) = div(D_eff grad(C)) + S(C, x, t)
```

with optional ALE form:

```text
dC/dt + div((u - w) C) = div(D_eff grad(C)) + S(C, x, t)
```

Checklist:

- [ ] Add scalar transport options for field name, space, diffusivity,
      velocity-field coupling, source model, and transient behavior.
- [ ] Support constant, parameter-driven, field-driven, and callback-driven
      diffusivity.
- [ ] Support effective diffusivity `D_eff = D_m + nu_t / Sc_t` when a
      turbulence module provides `nu_t`.
- [ ] Register transient mass, convection, diffusion, and source residuals.
- [ ] Support Dirichlet, Neumann flux, Robin flux, and weak Dirichlet boundary
      conditions.
- [ ] Support ALE mesh velocity by reusing existing moving-domain terminals.
- [ ] Attach FE state-group metadata for multiple transported scalars when
      requested.
- [ ] Attach FE admissibility descriptors for nonnegative or bounded scalars.
- [ ] Use FE bounded-update policies for optional check-only, clamp, or reject
      behavior.
- [ ] Add unit tests for steady diffusion, transient mass, convection,
      diffusion, source terms, boundary fluxes, and bounded-state metadata.
- [ ] Add manufactured-solution tests for static and ALE transport.

Bioreactor reuse:

- dissolved oxygen
- dissolved CO2
- tracer mixing
- viable cell concentration, if transported
- gas-phase species, if modeled as scalar fields
- temperature, if heat is implemented as transported scalar

### Phase 2: N-Phase Eulerian Multiphase Flow

Goal: add a reusable Eulerian multiphase-flow formulation for `N` phases,
where each phase may be liquid, gas, solid-like dispersed material, or a
generic continuum. A sparged gas-liquid bioreactor is one configuration of this
general formulation, not the module identity.

Proposed location:

`Physics/Formulations/Multiphase/`

Needed types:

- `EulerianMultiphaseModule`
- `EulerianMultiphaseOptions`
- `PhaseDescriptor`
- `PhaseContinuityModel`
- `PhaseMomentumModel`
- `InterphaseTransferModel`
- `InterphaseMomentumModels`
- `MultiphaseBCFactories`
- `MultiphaseRegister`

Phase descriptor:

- phase name
- phase type: liquid, gas, solid-like dispersed, or generic
- phase role: continuous, dispersed, or unresolved mixture component
- material/property model
- volume fraction field `alpha_k`
- velocity model: shared mixture velocity, phase velocity, or algebraic slip
- density/compressibility model
- optional diameter or interfacial-area provider for dispersed phases

Core constraints:

```text
sum_k alpha_k = 1
0 <= alpha_k <= 1
```

Generic phase continuity:

```text
partial(alpha_k rho_k) / partial t
  + div(alpha_k rho_k u_k)
  = sum_j Gamma_jk
```

where `Gamma_jk` is mass transfer from phase `j` to phase `k`.

Generic phase momentum for full N-fluid mode:

```text
partial(alpha_k rho_k u_k) / partial t
  + div(alpha_k rho_k u_k tensor u_k)
  = -alpha_k grad(p)
    + div(alpha_k tau_k)
    + alpha_k rho_k b
    + sum_j M_jk
    + phase source terms
```

where `M_jk` is pairwise interphase momentum exchange. Conservative pairwise
closures should satisfy:

```text
M_jk = -M_kj
```

Fidelity levels:

- mixture or shared-velocity model
- drift-flux or algebraic-slip model
- two-fluid gas-liquid model as a special case
- full N-fluid Euler-Euler model

Relationship to Navier-Stokes:

- The multiphase formulation owns phase hydrodynamic unknowns such as
  `alpha_k`, `u_k` or mixture velocity, and shared or phase pressure.
- A full Euler-Euler multiphase model is not several independent
  Navier-Stokes modules wrapped together. Its equations are volume-fraction
  weighted and coupled through phase constraints and interphase exchange.
- The implementation should still reuse Navier-Stokes building blocks where
  practical: stress and viscosity models, pressure gauge handling,
  stabilization patterns, ALE and moving-frame support, boundary-condition
  helpers, body-force/source-term helpers, and turbulence closures.
- Early single-phase screening can continue to use the existing
  `IncompressibleNavierStokesVMSModule` with scalar transport and empirical
  transfer sources. Once `alpha_k` fields and interphase exchange matter, the
  setup should move to the multiphase formulation.

Bioreactor first-use configuration:

- phase 0: continuous liquid culture medium
- phase 1: dispersed sparged gas bubbles
- gas injection at sparger boundaries or source regions
- gas escape at top boundaries
- drag, buoyancy, turbulent dispersion, and optional lift, virtual mass, wall
  lubrication, and bubble-induced turbulence

Checklist:

- [ ] Define generic phase descriptors and phase state groups for `N` volume
      fraction fields.
- [ ] Add admissibility descriptors for `0 <= alpha_k <= 1` and
      `sum_k alpha_k = 1`.
- [ ] Add phase-continuity equations with configurable phase sources,
      interphase mass transfer, inlets, and outlets.
- [ ] Add mixture/shared-velocity and algebraic-slip modes before requiring
      full N-fluid momentum for every phase.
- [ ] Add optional full phase-momentum equations for selected phases.
- [ ] Add pairwise interphase momentum closures with equal-and-opposite
      conservation metadata.
- [ ] Add generic drag and buoyancy source terms.
- [ ] Add optional turbulent-dispersion force.
- [ ] Add optional lift, virtual mass, and wall-lubrication force models.
- [ ] Add generic phase inlet and phase outlet boundary/source options.
- [ ] Add gas sparger and top degassing as named configurations of generic
      phase inlet/outlet behavior, not as module-level assumptions.
- [ ] Allow dispersed-phase diameter or interfacial-area providers to be
      supplied by closure models or PBM.
- [ ] Add outputs for phase holdup, phase residence proxy, phase escape rate,
      and loading indicators.
- [ ] Add tests for `N`-phase volume-fraction bounds, phase-continuity source
      accounting, pairwise transfer conservation, drag source signs, inlet
      accounting, and outlet accounting.

Bioreactor reuse:

- local gas holdup `alpha_g`
- gas residence and escape
- dispersed gas source for oxygen transfer
- flooding/loading indicators

Future reuse:

- liquid-liquid extraction
- emulsions
- immiscible multiphase liquid systems
- multiple dispersed gas or liquid phases
- solid-like dispersed phase extensions if supported later

### Phase 3: Bubble Size Closures And Population Balance

Goal: compute or prescribe bubble size and interfacial area for mass transfer,
and add a transported population balance formulation when bubble-size
distribution dynamics are needed.

This phase has two distinct parts:

- Bubble size and interfacial-area closures are algebraic or constitutive
  models. They are not separate physics formulations.
- The population balance model is a distinct physics formulation because it
  owns transported unknowns and breakup/coalescence source terms.

Proposed locations:

`Physics/Formulations/PopulationBalance/`

`Physics/Materials/Multiphase/`

Needed types:

- `BubbleSizeModel`
- `ConstantBubbleDiameterModel`
- `AlgebraicSauterDiameterModel`
- `PopulationBalanceModule`
- `PBMBreakupKernel`
- `PBMCoalescenceKernel`
- `InterfacialAreaModel`

Closure-only model:

```text
d_b = constant
```

or:

```text
d_32 = f(epsilon, sigma, rho_l, mu_l, Q_g, alpha_g, medium properties)
```

with interfacial area:

```text
a = 6 alpha_g / d_32
```

Population balance formulation:

Use bubble volume `v` or bubble diameter `d` as an internal coordinate. Bubble
volume is often cleaner because coalescence conserves gas volume.

The number density is:

```text
n(x, v, t)
```

where `n dv` is the number of bubbles per physical volume with bubble volumes
in `[v, v + dv]`.

The general population balance equation is:

```text
partial n / partial t + div(u_g n) + partial(G n) / partial v
  = B_break - D_break + B_coal - D_coal
```

where:

- `u_g` is the gas-phase velocity
- `G = dv/dt` is the optional bubble growth or shrinkage rate
- `B_break` is birth of bubbles of size `v` from breakup of larger bubbles
- `D_break` is loss of bubbles of size `v` by breakup
- `B_coal` is birth of bubbles of size `v` from coalescence of smaller bubbles
- `D_coal` is loss of bubbles of size `v` by coalescence with other bubbles

Breakup terms can be written as:

```text
D_break(v) = g(v) n(v)

B_break(v) = integral_v^infinity beta(v | v') g(v') n(v') dv'
```

where:

- `g(v)` is the breakup frequency
- `beta(v | v')` is the daughter-size distribution

Coalescence terms can be written as:

```text
D_coal(v) = n(v) integral_0^infinity K(v, v') n(v') dv'

B_coal(v) = 1/2 integral_0^v K(v', v - v') n(v') n(v - v') dv'
```

where `K(v, v')` is the collision/coalescence kernel.

For CFD, this is usually discretized as bins or moments. A bin formulation is:

```text
partial n_i / partial t + div(u_g n_i)
  = S_i,break + S_i,coal + S_i,growth
```

for size class `i`.

Derived quantities from the distribution are:

```text
alpha_g = integral V_b(v) n(v) dv
a       = integral A_b(v) n(v) dv
```

For spherical bubbles using diameter `d`:

```text
alpha_g = integral (pi / 6) d^3 n(d) dd

a = integral pi d^2 n(d) dd

d_32 = integral d^3 n(d) dd / integral d^2 n(d) dd
```

and equivalently:

```text
a = 6 alpha_g / d_32
```

Checklist:

- [ ] Add constant bubble diameter model for early screening.
- [ ] Add algebraic `d_32` model using local turbulence, surface tension,
      density, viscosity, gas flow, and closure constants.
- [ ] Add interfacial area helper `a = 6 alpha_g / d_32`.
- [ ] Keep constant diameter, algebraic `d_32`, and interfacial-area models in
      closure/material code rather than treating them as standalone
      formulations.
- [ ] Add PBM bin or moment state groups using FE indexed scalar-set metadata.
- [ ] Add PBM advection and source terms for bubble bins or moments.
- [ ] Add breakup kernel interface.
- [ ] Add coalescence kernel interface.
- [ ] Ensure PBM sources conserve gas volume across breakup and coalescence
      within configured tolerance.
- [ ] Add derived `alpha_g`, `a`, and `d_32` outputs from bin or moment states.
- [ ] Add closure hooks for medium effects such as proteins, surfactants,
      antifoam, salts, and serum.
- [ ] Add bounded update and admissibility checks for nonnegative bin values.
- [ ] Add tests for constant diameter, algebraic diameter, interfacial area,
      bin conservation, breakup source balance, and coalescence source balance.

Bioreactor reuse:

- local `d_32`
- local interfacial area `a`
- bubble breakup/coalescence sensitivity to impeller design and medium
  properties

### Phase 4: Interphase Mass Transfer Coupling

Goal: add a generic Physics coupling module that transfers species or phase
mass between already-registered phases and transported species. Gas-liquid
oxygen dissolution is one configuration of this coupling, not the coupling
module identity.

Proposed location:

`Physics/Coupling/InterphaseMassTransfer/`

Needed types:

- `InterphaseMassTransferModule`
- `InterphaseMassTransferOptions`
- `InterphaseSpeciesPair`
- `InterphaseEquilibriumModel`
- `InterphaseRateLaw`
- `HenryLawModel`
- `LiquidSideMassTransferModel`
- `InterfacialAreaProvider`

This is a distinct Physics module, but it is not a standalone formulation or
equation set unless species transport is incorrectly bundled into it. It should
couple existing modules by contributing source terms.

Coupled Physics inputs:

- multiphase formulation: phase identities, `alpha_k`, phase velocities,
  pressure, phase densities, and phase roles
- scalar or species transport formulation: donor and receiver species fields
- bubble-size closure or PBM: interfacial area `a`, `d_32`, or
  distribution-derived area
- materials and thermodynamics: Henry constants, partition coefficients,
  diffusivities, surface tension, density, viscosity, and medium corrections
- turbulence module: optional `epsilon`, `k`, `nu_t`, wall distance, or scalar
  diffusivity inputs for rate closures
- heat and chemistry modules: optional temperature, CO2 chemistry, pH, and
  reaction coupling

Core source term:

```text
S_s,receiver = +R_s
S_s,donor    = -R_s
```

Gas-liquid oxygen dissolution is a Henry-law configuration:

```text
R_O2 = k_L a (C*_O2 - C_O2,l)
```

where `C*_O2` is computed from gas composition, pressure, temperature, medium
properties, and the chosen Henry-law convention.

CO2 stripping is the reverse configuration:

```text
donor phase: liquid
receiver phase: gas
species: CO2
```

Mass-transfer modes:

- Dilute species transfer: source/sink terms are added only to species
  transport equations. Phase masses and volume fractions are not changed
  appreciably. This is the first target for oxygen dissolution in a cell
  culture bioreactor.
- Phase mass transfer: evaporation, condensation, dissolving phases, or other
  cases where bulk phase mass changes appreciably. This mode must also provide
  `Gamma_jk` terms to multiphase phase-continuity equations and consistent
  momentum and energy source terms when needed.

The coupling should read phase velocity, pressure, phase fraction, turbulence,
temperature, and interfacial-area fields only to evaluate rate closures. It
should not solve phase momentum, pressure, or volume-fraction equations.

Checklist:

- [ ] Add donor/receiver phase descriptors that reference phases registered by
      the multiphase formulation.
- [ ] Add donor/receiver species descriptors that reference scalar or species
      transport fields.
- [ ] Support dilute species-transfer mode with paired species source terms.
- [ ] Support optional phase-mass-transfer mode that contributes `Gamma_jk`
      terms to multiphase phase continuity.
- [ ] Add optional momentum and energy source hooks for phase-mass-transfer
      cases that require them.
- [ ] Add Henry-law model with explicit convention selection:
      `C* = H p` or `C* = p / H`.
- [ ] Add generic partition/equilibrium model hooks for non-Henry interphase
      transfer.
- [ ] Support temperature, pressure, and medium-property dependence in Henry
      constants.
- [ ] Support local gas partial pressure from fixed gas composition or
      transported gas species.
- [ ] Support hydrostatic pressure correction.
- [ ] Add liquid-side mass-transfer coefficient models based on Sherwood
      correlations, surface-renewal closures, or user callbacks.
- [ ] Consume interfacial area from constant bubble diameter, algebraic
      `d_32`, PBM, or callback.
- [ ] Use FE paired-exchange descriptors for equal-and-opposite interphase
      source terms.
- [ ] Add options for liquid-only oxygen source when gas depletion is neglected.
- [ ] Add full donor/receiver species exchange when both species fields are
      transported.
- [ ] Add outputs for local rate coefficient, `a`, effective transfer
      coefficient, OTR/stripping rate, and total transfer.
- [ ] Add tests for Henry-law convention, transfer-source sign, paired exchange
      conservation, hydrostatic pressure effect, zero-area behavior, dilute
      species-transfer mode, and phase-mass-transfer mode.

Bioreactor reuse:

- oxygen transfer from bubbles into liquid
- CO2 stripping from liquid into gas
- optional headspace exchange

Future reuse:

- liquid-liquid extraction
- evaporation and condensation
- dissolution or degassing in non-bioreactor processes
- reactive interphase transfer when coupled to chemistry modules

### Phase 5: Turbulence Models And Turbulence Outputs

Goal: provide turbulence fields needed by mass transfer, bubble breakup,
mixing, wall models, and cell-stress metrics.

Turbulence modules should be reusable Physics providers, not replacements for
existing hydrodynamic formulations. Flow modules solve momentum and pressure;
turbulence modules solve or derive turbulence quantities and expose closures
that other modules consume.

Proposed location:

`Physics/Formulations/Turbulence/`

Needed types:

- `TurbulenceProvider`
- `TurbulenceModelModule`
- `AlgebraicTurbulenceModel`
- `KEpsilonModule`
- `KOmegaSSTModule`
- `TurbulentViscosityProvider`
- `TurbulencePostProcessing`

Provider inputs:

- velocity field
- density and molecular viscosity
- wall distance from FE boundary-distance service
- optional phase fields from the multiphase formulation
- optional moving-frame or rotating-region metadata

Provider outputs:

- turbulent viscosity `nu_t`
- turbulent kinetic energy `k`
- dissipation rate `epsilon` or specific dissipation rate `omega`
- turbulent diffusivity `D_t = nu_t / Sc_t`
- production, destruction, and dissipation diagnostics

Relationship to existing and planned modules:

- Navier-Stokes VMS: keep the existing `IncompressibleNavierStokesVMSModule`
  as the single-phase liquid-flow base. For early screening, VMS remains a
  stabilization and may optionally provide surrogate turbulence diagnostics.
  For RANS mode, a turbulence module provides `nu_t`, and Navier-Stokes uses:

```text
mu_eff = mu + rho nu_t
```

- Scalar transport: transport modules consume turbulent diffusivity:

```text
D_eff = D_m + nu_t / Sc_t
```

  This applies to dissolved oxygen, dissolved CO2, tracer, heat, and transported
  cell-density fields.

- N-phase multiphase flow: the multiphase formulation owns phase velocities,
  pressure, and volume fractions. Turbulence modules provide closures such as
  phase or mixture `nu_t`, turbulent dispersion, bubble-induced turbulence
  hooks, effective phase diffusivities, and turbulence-modified drag or slip
  correlations.

- Bubble-size closures and PBM: algebraic `d_32`, breakup frequency, and PBM
  kernels consume turbulence inputs such as `epsilon`, `k`, and `nu_t`.

- Interphase mass transfer: rate laws consume turbulence fields for
  liquid-side or phase-side transfer coefficients, for example:

```text
k_L = f(D, epsilon, viscosity, bubble diameter, slip velocity, ...)
```

- Rotating machinery: MRF and sliding-mesh modules provide moving-frame data,
  wall velocities, and flow forcing. Turbulence modules may consume those
  fields or frame metadata for production terms and near-wall models, but they
  do not own impeller physics.

Coupling modes:

- Lagged or segregated coupling: solve flow, update turbulence, then update
  transport and transfer closures. This is the preferred first implementation.
- Monolithic coupling: solve flow and turbulence fields in one coupled system
  when robustness or accuracy requires it.

Checklist:

- [ ] Keep the existing VMS module available for stabilized liquid-flow
      screening.
- [ ] Add a documented VMS-derived turbulence surrogate only for early design
      screening, if useful.
- [ ] Add a provider interface so other modules can request `nu_t`, `k`,
      `epsilon`, `omega`, and `D_t` without depending on a specific turbulence
      formulation.
- [ ] Add prescribed or algebraic `nu_t` and `epsilon` providers for first
      coupled transport and mass-transfer tests.
- [ ] Add RANS `k-epsilon` module for engineering gas-liquid modeling.
- [ ] Add optional RNG or realizable `k-epsilon` variants.
- [ ] Add optional `k-omega SST` module for wall-sensitive cases.
- [ ] Couple RANS turbulent viscosity back to Navier-Stokes through effective
      viscosity without duplicating the momentum equation.
- [ ] Use FE boundary-distance service for wall-distance and wall-function
      inputs.
- [ ] Register fields for `k`, `epsilon`, `omega`, and `nu_t` as needed.
- [ ] Attach admissibility descriptors for nonnegative turbulence quantities.
- [ ] Add turbulent diffusivity coupling for scalar transport through
      `nu_t / Sc_t`.
- [ ] Expose turbulence fields to multiphase, PBM, interphase mass transfer,
      rotating machinery, and output modules through provider interfaces.
- [ ] Add outputs for `k`, `epsilon`, `nu_t`, turbulent Reynolds number,
      Kolmogorov length, and shear-related quantities.
- [ ] Add tests for provider lookup, effective-viscosity coupling, positive
      production/destruction accounting, wall-distance dependency, turbulent
      diffusivity coupling, PBM/mass-transfer consumption, and bounded update
      behavior.

Bioreactor reuse:

- bubble breakup
- `k_L` closures
- scalar mixing
- local energy dissipation
- shear and exposure metrics

### Phase 6: Rotating-Frame And Impeller Hydrodynamic Coupling

Goal: add reusable rotating-frame, moving-wall, rotor/stator, and power-output
support for flow and multiphase formulations. This is a hydrodynamic modeling
capability and coupling layer, not a standalone physics formulation.

Rotating machinery should not claim to solve an independent rotating-machinery
PDE system:

- MRF source terms are momentum-equation terms.
- Moving blade walls are flow boundary conditions.
- Sliding mesh and rotor/stator transfer are geometry/interface orchestration.
- Torque and power are QoIs/functionals.
- Gas loading, gas cavity, and flooding indicators are diagnostics derived from
  flow and gas fields.

Proposed locations:

`Physics/Coupling/RotatingFrame/`

targeted extensions to:

`Physics/Formulations/NavierStokes/`

targeted extensions to:

`Physics/Formulations/Multiphase/`

and postprocessing in:

`Physics/QoI/`

Needed types:

- `RotatingFrameInstaller`
- `RotatingRegionOptions`
- `RotatingFrameSourceTerms`
- `MovingWallBCFactories`
- `RotorStatorInterfaceOptions`
- `RotatingHydrodynamicsPostProcessing`
- `TorquePowerFunctionals`

Module relationships:

- Navier-Stokes: consumes MRF source terms, moving-wall boundary conditions,
  torque functionals, and power outputs.
- Multiphase: consumes phase-momentum MRF terms, rotating-region gas
  dispersion options, and gas loading or cavity diagnostics.
- Turbulence: consumes rotating-frame and near-blade flow information for
  production terms and wall/near-blade models.
- Scalar transport: couples only through the velocity fields generated by the
  hydrodynamic formulation; it should not depend directly on rotating-frame
  helpers.
- PBM and interphase mass transfer: couple indirectly through turbulence, gas
  holdup, slip velocity, residence time, and interfacial area.

A helper module may still be useful as an installer, but it should explicitly
register rotating regions, frame bindings, source terms, boundary terms, and
QoIs into flow or multiphase systems rather than advertising an independent
physics equation.

Checklist:

- [ ] Add rotating-region options that bind Physics regions to FE frame
      descriptors.
- [ ] Add MRF source terms for Coriolis and centrifugal acceleration.
- [ ] Expose MRF source-term builders for Navier-Stokes and multiphase
      momentum equations.
- [ ] Add frame-aware velocity interpretation for flow convection and wall
      velocity.
- [ ] Add moving-wall no-slip support for blades and shafts.
- [ ] Reuse FE sliding-interface metadata for rotor/stator transfer and
      invalidation.
- [ ] Add rotor/stator interface options that configure existing FE sliding
      and transfer infrastructure.
- [ ] Add torque functional on blade boundaries.
- [ ] Add power functional `P = torque * angular_velocity`.
- [ ] Add outputs for tip speed, power number, flow number inputs, and gassed
      power ratio hooks.
- [ ] Add gas loading, gas cavity, and flooding diagnostics only as derived
      outputs that consume multiphase fields.
- [ ] Add tests for rotating-source signs, zero-rotation equivalence, frame
      lookup failure behavior, torque functional accounting, and sliding-map
      invalidation metadata.

Bioreactor reuse:

- impeller-driven flow
- blade power draw
- rotor/stator handling
- gas cavity and flooding indicators when coupled to gas phase

### Phase 7: Material And Property Models

Goal: provide simulation-ready property closures for liquid medium, gas
mixtures, mass transfer, surface tension, and biological media effects.

Proposed locations:

`Physics/Materials/Fluid/`

`Physics/Materials/Gas/`

`Physics/Materials/Multiphase/`

`Physics/Materials/MassTransfer/`

Needed types:

- `LiquidMediumProperties`
- `GasMixtureProperties`
- `SurfaceTensionModel`
- `DiffusivityModel`
- `HenryConstantModel`
- `MediumCorrectionModel`

Checklist:

- [ ] Add constant property models for baseline cases.
- [ ] Add temperature-dependent density, viscosity, diffusivity, and Henry
      constant options.
- [ ] Add gas-mixture composition helpers for oxygen, nitrogen, carbon dioxide,
      and enriched oxygen.
- [ ] Add surface-tension models and medium correction hooks.
- [ ] Add viscosity and diffusivity options for representative culture media.
- [ ] Add property validation for units and physical ranges.
- [ ] Add tests for unit consistency, temperature dependence, gas partial
      pressure, and invalid parameter rejection.

Bioreactor reuse:

- medium-dependent bubble behavior
- oxygen saturation
- gas transport
- heat and chemistry coupling

### Phase 8: Cellular Oxygen Uptake And Biological Stress

Goal: model biological oxygen consumption and stress/damage indicators in
Physics, while using FE only for generic reductions and exposure accumulation.

Proposed locations:

`Physics/Formulations/Biology/`

`Physics/Materials/Biology/`

Needed types:

- `CellUptakeModel`
- `ConstantOURModel`
- `MonodOxygenUptakeModel`
- `CellDensityTransportModule`
- `BiologicalStressMetricModule`

Checklist:

- [ ] Add constant uptake model `OUR = q_O2 X_v`.
- [ ] Add Monod model `OUR = q_O2,max C / (K_O + C) X_v`.
- [ ] Support uniform viable cell concentration as a parameter.
- [ ] Support transported viable cell concentration through scalar transport.
- [ ] Add source coupling from uptake model into dissolved oxygen transport.
- [ ] Add optional growth, death, and metabolism hooks.
- [ ] Add stress metric models based on shear rate, stress, energy dissipation,
      Kolmogorov length, bubble-burst exposure, or user callbacks.
- [ ] Use FE exposure accumulation for time above or below Physics-defined
      thresholds.
- [ ] Add tests for constant uptake, oxygen-limited uptake, zero-cell behavior,
      source sign, bounded cell density, and exposure metric accumulation.

Bioreactor reuse:

- OUR
- minimum viable dissolved oxygen constraint
- biological damage metrics
- culture-type-specific stress thresholds

### Phase 9: CO2, Carbonate Chemistry, And pH

Goal: add optional chemistry support for CO2 stripping and pH behavior in cell
culture media.

Proposed location:

`Physics/Formulations/Chemistry/`

Needed types:

- `DissolvedCO2TransportModule`
- `CarbonateEquilibriumModel`
- `PHModel`
- `GasCO2ExchangeModel`

Checklist:

- [ ] Reuse scalar transport for dissolved CO2.
- [ ] Reuse interphase mass transfer for CO2 stripping.
- [ ] Add bicarbonate/carbonate equilibrium or kinetic chemistry model.
- [ ] Add pH calculation from carbonate chemistry and buffer parameters.
- [ ] Add optional metabolic CO2 source.
- [ ] Add headspace/off-gas CO2 outputs.
- [ ] Add tests for CO2 transfer direction, equilibrium consistency, pH
      monotonicity, and conservation where applicable.

Bioreactor reuse:

- CO2 stripping constraints
- pH risk assessment
- off-gas validation

### Phase 10: Heat Transfer

Goal: add transient heat transport and thermal source terms for bioreactor
operation.

Proposed location:

`Physics/Formulations/Heat/`

Needed types:

- `HeatTransportModule`
- `HeatTransferBCFactories`
- `BioreactorHeatSourceModel`

Checklist:

- [ ] Build heat transport on the scalar transport infrastructure where
      possible.
- [ ] Add volumetric heat capacity and thermal conductivity properties.
- [ ] Add advective heat transport by liquid velocity.
- [ ] Add metabolic and power-dissipation heat sources.
- [ ] Add wall/jacket Robin heat-transfer boundaries.
- [ ] Couple temperature to material and mass-transfer properties.
- [ ] Add tests for transient heat capacity, wall heat flux, heat-source sign,
      and temperature-dependent property coupling.

Bioreactor reuse:

- thermal control
- temperature-dependent oxygen solubility
- viscosity and diffusivity variation

### Phase 11: Solver XML Orchestration Interface

Goal: make each reusable Physics module easy to compose through solver XML for
a stirred, sparged tank, without adding a dedicated bioreactor Physics module.

No new location should be added under:

`Physics/Formulations/Bioreactor/`

Do not add:

- `BioreactorModule`
- `BioreactorOptions`
- `BioreactorRegister`
- `BioreactorOutputRegistration`

The scenario-level orchestration belongs in the solver XML configuration and
the application translator/builder path that instantiates registered Physics
modules and couplings.

Checklist:

- [ ] Ensure each reusable Physics module has a stable equation or coupling
      registration name for solver XML.
- [ ] Ensure module options can reference fields, state groups, boundary
      markers, regions, moving frames, and interface names declared elsewhere
      in the XML.
- [ ] Ensure coupling modules can reference donor and receiver fields by name
      without hard-coding a bioreactor-specific composition layer.
- [ ] Ensure output requests can be declared from XML and mapped to the
      appropriate module-local output registration.
- [ ] Ensure module dependencies are explicit, for example interphase mass
      transfer requiring donor/receiver species plus either prescribed or
      modeled interfacial area.
- [ ] Ensure validation-oriented parameters such as measured `kLa`, torque, gas
      holdup, bubble size, and mixing time targets can be supplied as data for
      outputs or tests, not as hidden defaults.
- [ ] Provide XML-level examples for model hierarchy levels:
  - [ ] well-mixed tank model
  - [ ] single-phase liquid CFD with empirical distributed oxygen source
  - [ ] Euler-Euler gas/liquid CFD with constant bubble size
  - [ ] Euler-Euler gas/liquid CFD with PBM
  - [ ] transient sliding mesh or LES-style validation path where supported
- [ ] Add translator/builder tests outside individual Physics modules that load
      representative XML configurations and verify the expected modules,
      fields, state groups, couplings, admissibility checks, and outputs are
      registered.

Bioreactor orchestration reuse:

- complete setup path through solver XML
- explicit and editable scenario composition
- reusable Physics modules that remain valid outside bioreactor cases
- reproducible validation configurations

### Phase 12: Bioreactor Outputs And Constraints

Goal: expose bioreactor metrics using generic FE postprocessing primitives and
Physics-specific definitions.

Proposed locations:

`Physics/QoI/`

module-local postprocessing files

Needed outputs:

- local dissolved oxygen
- minimum dissolved oxygen
- percentile dissolved oxygen
- volume fraction below critical dissolved oxygen
- local `k_L`
- local interfacial area
- local and global `kLa`
- local and total OTR
- OUR
- oxygen transfer margin
- gas holdup
- bubble diameter and interfacial area
- gas escape rate
- torque
- power
- power per volume
- shear rate
- turbulent energy dissipation
- Kolmogorov length
- exposure time above stress thresholds
- mixing-time tracer metrics
- optional CO2, pH, and heat metrics

Checklist:

- [ ] Register DO threshold volume using FE threshold reductions.
- [ ] Register DO percentile outputs using FE percentile reductions.
- [ ] Register shear and energy-dissipation exposure using FE exposure
      accumulation.
- [ ] Register OTR and OUR as Physics functionals.
- [ ] Register gas holdup, interfacial area, and bubble size reductions.
- [ ] Register torque and power functionals on impeller boundaries.
- [ ] Register flooding/loading indicator outputs.
- [ ] Register optional CO2 stripping, pH, and heat-balance outputs.
- [ ] Add tests for each output registration path and for deterministic
      reduction behavior.

## Validation Plan

The model should not be considered predictive until validated against
representative experimental data.

Checklist:

- [ ] Validate liquid flow or mixing time against tracer data.
- [ ] Validate torque and gassed/ungassed power draw.
- [ ] Validate gas holdup.
- [ ] Validate bubble size distribution or Sauter mean diameter.
- [ ] Validate dynamic gassing-out `kLa` with probe-response correction.
- [ ] Validate dissolved oxygen probe data at one or more locations.
- [ ] Validate off-gas oxygen and CO2 when gas species are modeled.
- [ ] Validate CO2 and pH behavior when chemistry is modeled.
- [ ] Validate cell viability, productivity, or damage metrics for the relevant
      culture type.
- [ ] Repeat validation in representative culture medium, not only water.

## Recommended Development Order

1. Scalar transport module.
2. Oxygen uptake source model.
3. Interphase mass-transfer coupling with prescribed `alpha_g` and `d_32`.
4. Generic bioreactor output registration for DO, OTR, OUR, and low-DO volume.
5. Rotating-frame/MRF source terms and torque/power functionals.
6. Turbulence outputs or RANS module, depending on target fidelity.
7. Euler-Euler gas/liquid model with constant bubble diameter.
8. Algebraic bubble-size and interfacial-area closures.
9. PBM for final mass-transfer validation.
10. CO2, pH, and heat modules as required by the experimental scope.
11. Solver XML orchestration examples and translator/builder coverage.
12. Validation cases and regression tests.

## Near-Term Minimal Viable Bioreactor Model

A useful first Physics implementation can be much smaller than the final model:

- existing incompressible Navier-Stokes VMS for liquid flow
- new scalar transport for dissolved oxygen
- simple distributed oxygen source with prescribed or algebraic `kLa`
- constant or Monod cellular oxygen uptake
- DO threshold and percentile outputs
- power and shear outputs from the existing flow fields

This first model can support early impeller screening, but it cannot predict
gas holdup, bubble residence, bubble-size distribution, or local interfacial
area without the multiphase and bubble-size additions.

## Complete Model Definition

A complete sparged bioreactor model requires:

- liquid flow with impeller motion or rotating-frame terms
- turbulence fields or turbulence surrogate outputs
- gas injection and gas holdup
- bubble size or population balance
- interphase area for dispersed gas bubbles
- interphase oxygen transfer
- dissolved oxygen transport
- cellular oxygen uptake
- shear and exposure metrics
- torque and power metrics
- top gas escape and optional headspace exchange
- optional CO2 stripping, pH, and heat transfer
- validation against representative experiments
