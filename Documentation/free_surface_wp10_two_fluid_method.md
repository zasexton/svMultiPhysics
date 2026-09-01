# WP-10 incompressible two-fluid unfitted method

Status: implementation contract. This record fixes the discrete signs,
weights, ownership, diagnostics, and qualification order for the staged
incompressible two-fluid extension. It does not claim that the formulation or
any WP-10 exit is complete.

## Capability boundary

The existing one-phase formulation remains a distinct supported model. It
solves one liquid velocity and pressure field and interprets exterior pressure
as prescribed traction data. The formulation specified here is a separate
incompressible two-fluid model. It does not silently promote a one-phase case,
and its evidence cannot be borrowed by the one-phase record or by a future
compressible-gas record.

The initial production envelope is deliberately narrow:

- one affine C0 P1 velocity and one affine C0 P1 pressure field per phase;
- Triangle3 in two dimensions or Tetra4 in three dimensions;
- one LinearCorner level-set interface on each coupled equation system;
- constant positive density and Newtonian dynamic viscosity in each phase;
- constant surface tension, with no Marangoni term;
- a fixed Eulerian background mesh;
- complementary cut-volume integration, sharp generated-interface coupling,
  and phase-local small-cut stabilization; and
- no contact line, phase change, turbulence closure, or compressible gas.

Every route outside this envelope must fail before fields, constraints, forms,
or operators mutate the equation system.

## Strong problem and orientation

Let the authoritative level set satisfy

```text
Omega_minus = {x : phi(x) < 0},
Omega_plus  = {x : phi(x) > 0}.
```

The generated normal `n` points from `Omega_minus` to `Omega_plus`. For phase
`s` in `{minus, plus}`, with density `rho_s`, viscosity `mu_s`, velocity `u_s`,
and pressure `p_s`, define

```text
sigma_s = -p_s I + 2 mu_s epsilon(u_s),
epsilon(u_s) = sym(grad(u_s)).
```

Each phase satisfies incompressible Navier--Stokes on its own cut domain. The
material interface conditions are

```text
u_minus - u_plus = 0,
(sigma_minus - sigma_plus) n = gamma kappa n,
```

where positive `kappa` is consistent with the normal above. The constant-gamma
implementation uses the variation of discrete interface area rather than an
independently reconstructed curvature wherever possible.

## Pressure space

The phases own different pressure fields. A pressure function can therefore
jump across the generated interface without enriching either phase's bulk
basis. This is a two-field jump representation, not a continuous pressure
field with a postprocessed offset.

The interface stress form couples the two pressure fields. Only their common
constant mode remains a pressure nullspace when no physical pressure datum is
present. The two phase restrictions must not each publish a fictitious
absolute-pressure anchor merely because the interface has nonzero measure.
The coupled formulation owns one explicit gauge decision for the combined
pressure pair.

## Weighted symmetric interface form

For positive viscosities define

```text
omega_minus = mu_plus  / (mu_minus + mu_plus),
omega_plus  = mu_minus / (mu_minus + mu_plus),
mu_h        = 2 mu_minus mu_plus / (mu_minus + mu_plus).
```

For any phase pair `a = (a_minus, a_plus)`, use

```text
jump(a)       = a_minus - a_plus,
average_w(a)  = omega_minus a_minus + omega_plus a_plus,
average_c(a)  = omega_plus a_minus + omega_minus a_plus.
```

The complementary average is required by the exact identity

```text
t_minus dot v_minus - t_plus dot v_plus
  = average_w(t) dot jump(v)
  + jump(t) dot average_c(v).
```

With phase tests `v_s` and `q_s`, define

```text
t_s(u_s,p_s) = (2 mu_s epsilon(u_s) - p_s I) n,
z_s(v_s,q_s) = (2 mu_s epsilon(v_s) - q_s I) n.
```

The symmetric interface residual is

```text
R_interface =
  - integral_Gamma average_w(t) dot jump(v)
  - integral_Gamma average_w(z) dot jump(u)
  + integral_Gamma beta jump(u) dot jump(v)
  + gamma integral_Gamma (I - n tensor n) : grad(average_c(v)).
```

The last term is the discrete surface-area variation. On a closed smooth
interface it is equivalent to the traction-jump load with the sign convention
above. It uses the same generated normal and measure as all other interface
terms.

For planar manufactured-jump verification, an optional target
`Delta p = p_minus - p_plus` adds

```text
+ integral_Gamma Delta p n dot average_c(v).
```

This is the residual load corresponding to
`(sigma_minus-sigma_plus)n = -Delta p n`; the same target is retained in the
accepted-stage pressure and stress-jump diagnostics. It is an explicit
manufactured load, not a postprocessing-only expected value.

For transient flow the penalty scale is

```text
beta = gamma_N * (mu_h / h_n + rho_h h_n / dt_eff),
rho_h = 2 rho_minus rho_plus / (rho_minus + rho_plus).
```

The inertial contribution is omitted for a genuinely steady operator. The
dimensionless `gamma_N` is positive, interface-local configuration. Polynomial
order scaling is fixed by the initial P1 envelope and must become explicit
before higher order is admitted.

## Reuse and ownership

Each phase reuses the established incompressible VMS/PSPG volume operator on a
complementary cut-volume side. The coupled two-fluid owner must configure the
phase instances, rather than asking users to coordinate two independent
one-phase modules.

The phase-local volume owners provide:

- transient, convective, viscous, pressure, body-force, VMS, and PSPG terms;
- sharp restriction of physical boundary conditions to the matching phase;
- inactive-side constraints that do not alter physical momentum;
- pressure jump stabilization on the phase's cut-adjacent facets; and
- small-cut aggregation for that phase's velocity and pressure fields.

The two-fluid owner alone provides:

- interface velocity and stress coupling;
- surface energy;
- the combined pressure gauge contract;
- interface diagnostics and accepted-step history;
- the interface transport-velocity declaration; and
- phase/momentum flux reconciliation.

Neither phase-local owner may install prescribed exterior traction on the
internal material interface.

## Interface transport and conservation

There is no phase change in the initial envelope. The level set must therefore
move with the common material-interface normal velocity. The transport owner
must consume an explicitly declared interface velocity whose trace is
`average_c(u)`; it may not select one phase by name without recording the
equivalence error.

The momentum weak form retains sharp phase-local velocity values in each bulk
restriction and uses `average_c(u)` on the generated interface. The
conservative indicator graph requires one velocity extension on every graph
node, so its declared material-interface route samples `average_c(u)` at every
node. Its level-set field and marker identify the owning declaration; the
nodal level-set sign does not switch the graph extension back to a phase-local
bulk value.

At every accepted stage, record

```text
Q_minus = integral_Gamma u_minus dot n,
Q_plus  = integral_Gamma u_plus  dot n,
Q_jump  = Q_minus - Q_plus,
Mflux_s = rho_s Q_s.
```

Any bounded phase correction or reconciliation must publish its raw phase
volume change, corrected phase volume change, and implied phasewise momentum
change. A correction is rejected unless the momentum update is either applied
consistently or proven zero to the declared tolerance. No correction may be
hidden inside geometry refresh.

## Required accepted-stage diagnostics

The coupled operator publishes raw integrals before derived norms:

- interface measure;
- squared velocity jump and its normal and tangential components;
- phasewise normal flux and their difference;
- phasewise traction and the stress-jump residual;
- pressure jump mean, squared norm, and prescribed jump error when applicable;
- surface-energy work;
- Nitsche consistency, adjoint, and penalty work separately;
- phase volume, mass, momentum, and kinetic energy;
- phase-transport and momentum-flux reconciliation; and
- canonical phasewise aggregation and pressure-stabilization configuration;
- nonlinear and linear iterations plus convergence reasons for the shared
  coupled solve; and
- explicit unavailable reasons for phase-resolved iteration counts and
  pressure-stabilization work when the coupled backend cannot separate them.

Absence and inapplicability are distinct from a numeric zero. Diagnostics are
staged with operator state, committed only after step acceptance, replay-safe,
and communicator-consistent.

## Solver contract

The first supported linear strategy uses the exact canonical unknown-role
order

```text
(level_set, phase_indicator,
 u_minus, u_plus,
 p_minus, p_plus).
```

FSILS BlockSchur groups the level set, conservative phase indicator, and both
phase velocities into one computational-primary block. The two phase
pressures form the constraint block. The layout is admitted only for the exact
paired level-set/two-fluid equation system; additional equations, another
backend, another solver method, or a malformed field layout fail before the
live system is mutated. There is no generic layout fallback.

The velocity portion of the primary block includes the interface penalty and
phase-local viscous, transient, and convective contributions. Pressure
preconditioning uses phasewise pressure mass or
pressure-convection-diffusion approximations scaled with the matching density
and viscosity. The interface coupling is retained in the preconditioned
operator; treating it only as an outer residual is not a qualified high-ratio
route.

The qualification matrix records spectrum estimates where available, Krylov
iterations, nonlinear iterations, convergence reasons, and iteration spread
over cut position, mesh size, density ratio, and viscosity ratio. A solver that
converges only after case-specific tolerances or unrecorded regularization does
not satisfy WP-10.

## Implementation sequence

Completed items below establish only the direct formulation boundary. WP-10
remains open until every remaining item and the qualification progression pass.

- [x] Add the phase-pair option and immutable effective-configuration artifact.
- [x] Add complementary phase fields and phase-local cut-volume operators
  without internal-interface exterior traction.
- [x] Add the weighted symmetric velocity/stress interface form and one
  combined gauge contract.
- [x] Add phase-local aggregation and stabilization identity tests under side
  reversal and partition changes.
- [x] Add accepted-stage interface, mass, momentum, and flux histories.
- [x] Connect the common interface velocity to conservative level-set
  transport and make every correction momentum-explicit.
- [x] Add the production parser and fail-closed configuration matrix only after
  the direct C++ formulation gates pass.
- [ ] Freeze and run the complete WP-10 qualification matrix from a clean
  source tree.

The phase-local identity item was completed on 2026-08-31. The serial fixture
holds each physical material fixed while reversing the level-set sign and
swapping the owning cut side. The two-rank fixture repeats both material
assignments with block and METIS cell partitions. In physical vertex, cell,
and face identity, both fixtures require exact rooted aggregation selection,
polynomial-extension weights, cut-adjacent facet scales, retained phase
volume, PSPG pressure-gradient action, and pressure ghost-penalty action.
Algebraic DOF fingerprints and the subsequently closed coupled pressure-gauge
row are deliberately outside this comparison because they change with
owner-contiguous numbering; the underlying provisional aggregation rows and
globalized phase operators are the tested invariant.

The staged core checkpoint was completed on 2026-09-01 from immutable `Code/`
digest `d40c0763054b40927c7422705465ae37cd5699f9dde5e66bcda0d77d906a3049`.
`amarsden` build job `41541399` produced the three focused binaries. FE job
`41541411` passed all 5 selected material-interface transport tests,
Application job `41542616` passed all 43 selected parser, dependency,
builder, transport, and accepted-stage telemetry tests, and Physics job
`41543288` passed all 27 selected serial tests plus both selected two-rank
tests with explicit evidence from ranks 0 and 1. This closes the three staged
implementation items above; it is not physical qualification, and the final
matrix item remains unchecked.

## Qualification progression

The implementation is exercised in this order:

1. exact constant-state cancellation and interface action/reaction;
2. planar prescribed pressure jump;
3. planar viscous traction jump;
4. two-fluid hydrostatics;
5. static circular and spherical drops over mesh, cut-offset, side-reversal,
   density-ratio, and viscosity-ratio sweeps;
6. both-phase volume and mass conservation;
7. phase-flux/momentum-flux consistency with deliberate bounded correction;
8. conditioning and solver iterations through the predeclared high-ratio
   range;
9. two-fluid capillary waves;
10. Hysing case 1 and a rising bubble; and
11. Hysing case 2 reported against a predeclared intercode range.

Every test records dimensional inputs, nondimensional groups, reference-data
provenance, mesh and time refinement, cut offsets, material-side reversal,
rank count, and uncertainty or intercode bands. The low-level planar and
static tests are hard prerequisites for the moving benchmarks.

## Deferred gas-sensitive scope

An incompressible positive phase supplies positive-phase inertia and viscosity,
but it does not establish compressible cushioning or trapped-gas pressure.
Roof-impact pressure, ambient-pressure splash thresholds, trapped gas,
aerodynamic breakup, and late atomization remain blocked until a separately
specified gas formulation, thermodynamic closure, interface coupling, and
qualification record exist.

WP-10 completion can establish the incompressible two-fluid capability without
claiming those deferred regimes. Q7 must retain separate incompressible and
gas-sensitive gates.
