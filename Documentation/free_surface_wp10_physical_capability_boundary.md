# WP-10 physical-capability boundary record

Status: the one-phase model boundary is explicit and the initial
incompressible two-fluid core is staged. The two-fluid method has not passed
the required physical qualification progression. Gas dynamics are absent.
FSR-08, WP-10, and Q7 remain open.

Scope: AD-6, the free-surface momentum and transport models, and the
qualification progression required before any two-fluid or gas-sensitive
claim. The runner binds each declared source to exact bytes and records the
current Git revision, whether the worktree is clean, and a hash of the raw
porcelain status. A dirty execution remains explicitly dirty; it is not
misrepresented as evidence from the recorded revision alone.

## Decision

AD-6 is split into three capability records that may not borrow evidence from
one another:

1. a one-phase incompressible liquid with prescribed exterior pressure;
2. a staged incompressible two-fluid extension; and
3. a gas model validated for the compressibility or other gas dynamics
   required by the target regime.

The first model is implemented and explicitly labeled, but its physical claims
remain bounded by the applicable Q0--Q6 matrices. The second now has a
production core inside a deliberately narrow initial envelope, but no WP-10
physical exit has passed. The third is not implemented.

## One-phase source contract

`IncompressibleNavierStokesVMSOptions` defaults the
`FreeSurfacePhysicalModel` enum to
`OnePhaseLiquidPrescribedExteriorPressure`. It owns one velocity field, one
pressure field, one density, and one viscosity value or constitutive
viscosity model. A free surface owns a scalar `external_pressure`. The
boundary operator uses that value as the prescribed exterior traction
reference. It does not solve exterior momentum or an exterior pressure field.

The one-phase effective artifact emits the capability label
`one_phase_liquid_sharp_interface` and physical model
`one_phase_liquid_prescribed_exterior_pressure`. Explicit legacy output cannot
inherit that current model record. Level-set transport independently labels
its nonlocally conservative and locally conservative indicator variants as
one-phase transport.

The canonical one-phase equation selectors remain
`Free_surface_physical_model` and `FreeSurfacePhysicalModel`. Their supported
one-phase value is `OnePhaseLiquidPrescribedExteriorPressure`; absence selects
that default for the one-phase module. Fitted surface/contact combinations
outside the implemented envelope fail before field mutation.

## Staged incompressible two-fluid contract

The dedicated `IncompressibleTwoFluid` route owns two velocity fields, two
pressure fields, two constant densities, two constant viscosities, one shared
level-set field, and one generated material interface. Its artifact schema 3
uses capability label
`incompressible_two_phase_sharp_interface_initial_envelope`.

The current envelope is fixed-Eulerian affine C0 P1 Triangle3/Tetra4 with
constant phase properties, CutVolume restriction on both complementary sides,
phasewise VMS/PSPG pressure stabilization, phasewise small-cut aggregation,
one weighted symmetric interface coupling, and one shared pressure gauge. The
interface form includes complementary viscosity weights, velocity and traction
coupling, surface tension, and an optional manufactured prescribed pressure
jump.

Strong physical velocity data on an exterior marker are stored once and
installed identically on both phase restrictions. Finite literals and
nonempty spatial or time-dependent coefficients are supported; raw form
expressions, duplicate markers, shared/phase-local marker overlap, and
nonhomogeneous phase-local data fail before mutation. Artifact schema 3
publishes the shared-data policy together with marker, active-component, and
value-kind provenance.

The parser requires every material coefficient and rejects unknown,
duplicated, unused, misplaced, nonfinite, or unsupported equation, domain,
boundary, and nested-block controls. Translation, semantic validation,
cross-equation dependency pairing, field compatibility, solver-envelope
checks, and future-field probes complete before live field or operator
mutation. The complete builder admits exactly one paired level-set/two-fluid
system with FSILS BlockSchur and the canonical role order

```text
(material_interface_level_set,
 conservative_phase_indicator,
 negative_velocity, positive_velocity,
 negative_pressure, positive_pressure).
```

The first four roles form the computational-primary block and both pressures
form the constraint block. There is no generic solver-layout fallback.

The conservative indicator declares the two-fluid owner and samples the
complementary weighted common interface velocity at every graph node. The
momentum weak form retains sharp phase-local bulk values and the same common
trace at the interface. A geometry correction must publish phasewise raw and
corrected mass/momentum records and is rejected when the declared momentum
tolerance is exceeded; no hidden velocity update is applied.

Accepted-stage records include interface measure, velocity and traction jump,
phase flux, prescribed pressure-jump error when applicable, interface work,
phase volume/mass/momentum/kinetic energy, momentum reconciliation, canonical
phasewise aggregation, pressure-stabilization configuration, and the shared
coupled nonlinear/linear solve report. Phase-resolved iterations and separately
resolved pressure-stabilization work carry explicit unavailable reasons when
the coupled backend cannot provide them.

This implementation boundary is not a physical qualification result. Contact,
moving mesh, variable material laws, turbulence, phase change, compressible
gas, trapped-gas pressure, and cushioning remain outside the two-fluid
artifact. High density-ratio robustness is a qualification question, not an
inference from the existence of a BlockSchur layout.

## Supplemental one-phase subguard

The committed scope guard parses representative XML, JSON, and decoded mapping
inputs and rejects two-phase, jump, enrichment, and gas markers with
`unsupported_two_phase_or_jump_free_surface_scope`. It remains executable
containment evidence for paths that declare the one-phase model. It is not the
global two-fluid parser and must not be applied to a dedicated
`IncompressibleTwoFluid` equation.

The production equation translator and module register are authoritative for
the new two-fluid route. The supplemental subguard continues to prove that a
one-phase configuration cannot silently acquire two-fluid or gas-sensitive
meaning.

## Capability ledger

| Capability | Current state | Permitted statement |
|---|---|---|
| One-phase incompressible liquid | Implemented and explicitly labeled; broader qualification remains governed by Q0--Q6 | Liquid dynamics with prescribed exterior pressure, within the separately qualified one-phase envelope |
| Incompressible two-fluid flow | Initial production core staged; WP-10 physical exits not yet passed | Core formulation, parser, dependency, transport, telemetry, and solver-envelope behavior only |
| Compressible or otherwise gas-dynamic flow | Not implemented | No trapped-gas pressure, cushioning, ambient-pressure threshold, aerodynamic breakup, or late-atomization claim |

## Staged WP-10 capabilities

Matrix v5 labels the following as `STAGED_UNQUALIFIED`:

- phasewise density and viscosity;
- both-phase velocity and pressure fields;
- interface velocity and stress conditions;
- separate phase pressure fields and manufactured pressure-jump treatment;
- stabilization and aggregation on both phases;
- phasewise mass accounting;
- phase-flux/momentum-flux reconciliation; and
- the exact initial high-ratio solver route.

This status means that the implementation and focused identity tests exist. It
does not mean that a physical convergence, conservation, conditioning, or
literature gate has passed. Every WP-10 exit is therefore
`BLOCKED_BY_MISSING_QUALIFICATION`, not blocked by a wholly absent core.

## Deferred gas-model requirement

Incompressible two-fluid support does not qualify gas-sensitive phenomena.
Before those claims, the implemented gas model must reproduce the pressure,
inertia, viscosity, and compressibility effects relevant to the benchmark
nondimensional regime. A gas formulation, thermodynamic closure, interface
coupling, conservation ledger, and robust solver remain absent.

Consequently dry splash, entrainment, roof-impact pressure, trapped gas,
air cushioning, ambient-pressure splash thresholds, aerodynamic sheet
breakup, and late atomization remain outside the current model.

## Frozen v5 core-boundary evidence

The matrix
`tests/cases/fluid/free_surface_wp10_capability_boundary_matrix.json` and
wrapper
`tests/cases/fluid/run_free_surface_wp10_capability_boundary_qualification.py`
freeze only categorical core evidence:

- 13 exact source checks retain the one-phase boundary and bind the two-fluid
  module, interface form, parser/register, application dependency builder,
  material-interface level-set transport, and accepted-stage telemetry;
- the one-phase subguard retains 3 accepted, 21 rejected, and 2 structurally
  invalid fixtures;
- five binary groups freeze 40 exact tests across FE, Physics, and Application;
- the accepted claim is `staged_two_fluid_capability_boundary`; and
- FSR-08, WP-10, Q7, incompressible two-fluid physical qualification, and
  gas-sensitive qualification requests are rejected before binary execution.

The v5 runner's 16 unit tests and validation-only route pass. Clean-source
`amarsden` job `41545273` executed all five groups at commit
`1d1a4e96e49541ab5f884371c5ca1ac3c80be94b`: 4 one-phase Physics, 4 one-phase
Application, 5 material-interface FE, 7 two-fluid Physics, and 20 two-fluid
Application tests all passed. The job completed `0:0` in 39 seconds with
330,800 KiB peak batch RSS. Its `summary.json` SHA-256 is
`2c9622ab0af04c61ddd3e91c8507be81411d7c2ea2c5ed774533791322a8859f`.
The durable execution catalog is
[`record.json`](qualification_logs/free_surface_wp10_capability_boundary_v5_20260901_41545273/record.json).
No numerical threshold is invented: these source and unit gates establish the
staged capability boundary, not physical validation.

## Required WP-10 progression

Freeze dimensional inputs, nondimensional groups, references, uncertainty
bands, mesh/time sequences, cut offsets, side reversal, rank counts, and solver
limits before running, then execute in this order:

1. constant-state cancellation and interface action/reaction;
2. planar pressure jump;
3. planar viscous traction jump;
4. two-fluid hydrostatics;
5. static circular and spherical drops;
6. material-side reversal;
7. both-phase volume and mass conservation;
8. phase-flux/momentum-flux consistency under bounded correction;
9. conditioning and solver iterations over the declared high-ratio range;
10. two-fluid capillary waves;
11. Hysing case 1 and a rising bubble; and
12. Hysing case 2 against a predeclared intercode range.

Each record must identify the exact model, source revision, phase properties,
interface representation, pressure treatment, stabilization, solver,
configuration, and raw phase/momentum ledgers. The planar and static gates are
hard prerequisites for moving benchmarks.

## Q7 remains blocked

The incompressible Q7 branch starts only after the preceding WP-10 exits pass.
Its order is static/jump evidence, Hysing case 1, two-fluid capillary waves and
a rising bubble, then Hysing case 2 with its post-breakup result treated as an
intercode range.

The gas-sensitive Q7 branch remains separate and begins only after an
applicable gas model is implemented and qualified. No Q7 case may reuse a
one-phase result as two-fluid or gas evidence. WP-10, Q7, and FSR-08 remain
unchecked.

## Source evidence map

- One-phase option and artifact boundary:
  `Code/Source/solver/Physics/Formulations/NavierStokes/IncompressibleNavierStokesVMSModule.h`
  and `.cpp`
- Dedicated two-fluid artifact, semantic validation, and registration:
  `Code/Source/solver/Physics/Formulations/NavierStokes/IncompressibleTwoFluidModule.cpp`
- Weighted interface form:
  `Code/Source/solver/Physics/Formulations/NavierStokes/IncompressibleTwoFluidInterface.cpp`
- Production parser and dependency pre-registration:
  `Code/Source/solver/Physics/Formulations/NavierStokes/NavierStokesRegister.cpp`
- Cross-equation dependency and exact solver layout:
  `Code/Source/solver/Application/Core/SimulationBuilder.cpp`
- Material-interface conservative transport:
  `Code/Source/solver/FE/LevelSet/LevelSetTransport.cpp`
- Accepted-stage transaction and publication:
  `Code/Source/solver/FE/Systems/FESystem.cpp`
- Supplemental one-phase XML/JSON/mapping containment:
  `tests/cases/fluid/free_surface_one_phase_scope_guard.py`
- Frozen executable inventory:
  `tests/cases/fluid/free_surface_wp10_capability_boundary_matrix.json`
