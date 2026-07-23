# WP-10 physical-capability boundary record

Status: the one-phase model boundary is explicit; the staged physical
extensions and their qualification are absent. FSR-08, WP-10, and Q7 remain
open.

Scope: AD-6, the free-surface momentum and transport model, and the
qualification progression required before any two-fluid or gas-sensitive
claim. The evidence is evaluated against the source revision recorded by the
runner rather than against an informal working-tree snapshot.

## Decision

AD-6 is split into three capability records that may not borrow evidence from
one another:

1. a one-phase incompressible liquid with prescribed exterior pressure;
2. an incompressible two-fluid extension; and
3. a gas model validated for the compressibility or other gas dynamics
   required by the target regime.

Only the first model exists. Its presence is not by itself a qualification
result: each one-phase capability still depends on the applicable Q0--Q6
matrices. The second and third models have no production implementation and
therefore have no qualifying simulation matrix.

## Current source contract

`IncompressibleNavierStokesVMSOptions` owns one velocity field, one pressure
field, one density, and one viscosity value or constitutive viscosity model.
A free surface owns a scalar `external_pressure`. The boundary operator uses
that value as the prescribed exterior traction reference. It does not solve
an exterior momentum equation or an exterior pressure field.

The effective Navier--Stokes artifact labels the supported current model
`one_phase_liquid_sharp_interface`. Level-set transport independently labels
its nonlocally conservative and locally conservative indicator variants as
one-phase transport. The locally conservative indicator is a liquid-geometry
transport variable; it is not a second-fluid momentum or pressure field.

The committed qualification scope guard parses representative XML, JSON, and
already-decoded mapping inputs. It rejects keys, tags, model values, and
name/key/option/parameter wrappers when they carry the frozen normalized
`two_phase`, `two_fluid`, `multiphase`, `jump`, `gas`, or
`pressure_enrichment` markers. Rejection uses the stable diagnostic
`unsupported_two_phase_or_jump_free_surface_scope`. This is executable
control-layer containment evidence. It is not a production formulation,
production-schema qualification, or the missing physical implementation.

The authoritative geometry snapshot retains negative- and positive-side
quadrature on cut cells. References there to a two-sided volume family are
geometric completeness rules only. Both geometric sides do not imply that
both sides have solved fluid physics.

## Capability ledger

| Capability | Current state | Permitted statement |
|---|---|---|
| One-phase incompressible liquid | Implemented and explicitly labeled; broader qualification remains governed by Q0--Q6 | Liquid dynamics with prescribed exterior pressure, within the separately qualified one-phase envelope |
| Incompressible two-fluid flow | Not implemented | No two-fluid hydrostatic, jump, Hysing, rising-bubble, or high-density-ratio claim |
| Compressible or otherwise gas-dynamic flow | Not implemented | No trapped-gas pressure, cushioning, ambient-pressure threshold, aerodynamic breakup, or late-atomization claim |

## Missing incompressible two-fluid implementation

WP-10 cannot close until production contains and qualifies all of the
following as one coherent method:

- phasewise density and viscosity;
- either both-phase velocity and pressure fields or a demonstrated stable
  one-field jump formulation;
- interface velocity and stress conditions;
- pressure-space treatment appropriate to the pressure jump;
- stabilization acting on both phases;
- phasewise mass accounting;
- phase-flux and momentum-flux consistency across material jumps;
- bounded phase reconciliation that cannot create unreported momentum; and
- solvers whose convergence and conditioning remain acceptable over the
  predeclared density-ratio range.

No current low-level free-surface test establishes any item in this list.

## Missing gas-model implementation

Incompressible two-fluid support would not automatically qualify gas-sensitive
phenomena. Before those claims, the implemented gas model must reproduce the
pressure, inertia, viscosity, and compressibility effects relevant to the
benchmark nondimensional regime. A compressible or otherwise independently
validated gas formulation, its thermodynamic closure, interface coupling,
conservation ledger, and robust solver are absent.

Consequently dry splash, entrainment, roof-impact pressure, trapped gas,
air cushioning, ambient-pressure splash thresholds, aerodynamic sheet
breakup, and late atomization remain outside the current model.

## Frozen low-level boundary evidence

The matrix
`tests/cases/fluid/free_surface_wp10_capability_boundary_matrix.json` and
wrapper
`tests/cases/fluid/run_free_surface_wp10_capability_boundary_qualification.py`
freeze only the following evidence:

- the momentum artifact emits its one-phase capability label and prescribed
  exterior-pressure state;
- the two level-set transport modes emit one-phase labels;
- the production option record still contains a single liquid
  velocity/pressure/density/viscosity state; and
- the committed one-phase scope guard executes a frozen positive/negative
  contract over representative XML, JSON, and mapping encodings, including
  tag, attribute, model-value, coupled wrapper-name/value, nested-list, and
  structured nested-value and XML tail forms. JSON configuration roots must
  be mappings; a non-mapping root is structural invalidity rather than an
  unsupported-physics diagnostic.

The wrapper accepts only the claim `one_phase_capability_boundary`. It rejects
requests for FSR-08 closure, WP-10 closure, Q7 closure, incompressible
two-fluid qualification, or gas-sensitive qualification before executing any
test binary. Validation-only and full execution both run the frozen scope-guard
contract and require its exact rejection diagnostic. There are deliberately
no invented numerical thresholds:
artifact-label tests are categorical containment checks, not physical
validation.

## Required WP-10 progression

After the formulation exists, freeze thresholds and execute:

1. planar pressure and viscous jumps;
2. two-fluid hydrostatics;
3. a static drop;
4. material-side reversal;
5. both-phase mass conservation;
6. high-density-ratio conditioning and solver convergence;
7. phase-flux versus momentum-flux consistency; and
8. representative serial/MPI, cut-position, rotation, spatial, and temporal
   refinements.

Each record must identify the exact model, source revision, phase properties,
interface representation, pressure treatment, stabilization, solver,
configuration, and raw phase/momentum ledgers.

## Q7 remains blocked

Q7 starts only after the preceding WP-10 exits pass. Its order is:

1. the static and jump matrix;
2. Hysing case 1;
3. two-fluid capillary waves and a rising bubble;
4. Hysing case 2 with its post-breakup result treated as an intercode range;
5. air-cushioning and trapped-gas cases appropriate to the implemented gas
   model; and
6. ambient-pressure and dry-wall splash sweeps.

No Q7 case may reuse a one-phase result as evidence. The present repository
has neither the required physical implementation nor a frozen applicable Q7
matrix, so Q7 and WP-10 must remain unchecked.

## Source evidence map

- One liquid field pair and material state:
  `Code/Source/solver/Physics/Formulations/NavierStokes/IncompressibleNavierStokesVMSModule.h`
- Prescribed exterior pressure and the momentum capability artifact:
  `Code/Source/solver/Physics/Formulations/NavierStokes/IncompressibleNavierStokesVMSModule.cpp`
- One-phase transport capability artifacts:
  `Code/Source/solver/Application/Translators/LevelSetEquationTranslator.cpp`
- Unsupported two-fluid/jump input containment:
  `tests/cases/fluid/free_surface_one_phase_scope_guard.py`
- Artifact-label unit evidence:
  `Code/Source/solver/Physics/Tests/Unit/test_MovingDomainPhysics.cpp` and
  `Code/Source/solver/Application/Tests/Unit/test_LevelSetEquationTranslator.cpp`
