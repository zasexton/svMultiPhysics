# WP-5 contact-line architecture and qualification boundary

Status: low-level prerequisite implemented; FSR-05, WP-5, Q4, and physical
dynamic-wetting qualification remain open.

Audit basis: repository HEAD
`6306165b436ba6d81b50a2e348025038bf049fa2`, plus the dirty tracked source
state reviewed on 2026-07-23. A result produced from a dirty source tree does
not establish source-to-binary correspondence.

## Verdict

The audited low-level implementation has one explicit Ren--E convention,
separate sharp contact and wetted-wall operators, accepted-stage provenance,
and wall-aware level-set maintenance. The retired prescribed-angle
codimension-two level-set residual and its penalty parameter are absent from
the production residual. Prescribed-angle geometry is instead owned by
accepted-state wall-aware repair, while its momentum contribution is the
Young wall-energy variation.

Focused serial tests cover sign and orientation reversals, both liquid-side
conventions, the two surface-tension forms, velocity and level-set
Jacobians, dissipation, dry-wall exclusion, fitted pinning, prescribed
wall-aware repair, authoritative accepted snapshots, and transactional
volume-correction rejection. The distributed repair and accepted-frame
conflict tests pass at two ranks. The same rank-independent tests were also
audited at four ranks; one test-only two-rank assertion was generalized to
an at-least-two-rank precondition.

This evidence is a prerequisite, not closure. It contains no mesh/time/slip
or numerical-wall-width refinement campaign, no integral advancing/receding
benchmark, and no public experimental uncertainty comparison.

## Sign, frame, and Ren--E law

Let \(n\) be the outward liquid-interface normal and \(n_w\) the wall normal
pointing outward from liquid into solid. The through-liquid dynamic angle is

\[
\cos\theta_d=-n\mathbin{\cdot}n_w.
\]

The wall-tangential footprint direction is

\[
m=\frac{n-(n\mathbin{\cdot}n_w)n_w}
        {\left\lVert n-(n\mathbin{\cdot}n_w)n_w\right\rVert},
\]

oriented outward from the wetted footprint. Contact speed is
\(V_{\mathrm{CL}}=u\mathbin{\cdot}m\); positive speed is advancing. The
implemented constitutive residual is

\[
r_{\mathrm{CL}}=
\frac{V_{\mathrm{CL}}}{M}
-\gamma\left(\cos\theta_e-\cos\theta_d\right),
\]

so the zero-residual law is

\[
V_{\mathrm{CL}}=\gamma M
\left(\cos\theta_e-\cos\theta_d\right).
\]

Reversing wall orientation together with the physical contact configuration,
or switching the active level-set side, preserves that law and the
through-liquid angle convention. A nontransverse interface/wall
intersection or a generated wall normal inconsistent with the configured
physical normal fails closed.

The line-friction dissipation is

\[
D_{\mathrm{line}}=\int_{\mathcal C}
\frac{V_{\mathrm{CL}}^2}{M}\,d\ell ,
\]

and the sharp wetted-wall Navier dissipation is

\[
D_{\mathrm{wall}}=\int_{\Gamma_{w,\mathrm{wet}}}
\frac{\mu}{\ell_s}\lvert u_t\rvert^2\,dS .
\]

Mobility \(M\), slip length \(\ell_s\), optional numerical wall width, and
mesh size are independent quantities. The sharp matrix fixes numerical wall
width to zero.

## Surface and wall-energy ownership

With curvature traction, the explicit contact force contains the complete
Ren--E angle gap. With variational surface stress, the surface conormal
already supplies the dynamic-angle part, so the separate conservative wall
term contains only the equilibrium Young contribution. Adding the complete
gap again in that path would double count contact work.

Prescribed contact angle has two distinct owners:

- accepted-state wall-aware level-set repair owns contact geometry; and
- the momentum form owns the Young wall-energy variation.

Both `SurfaceStress` and `CurvatureTraction` contact configurations register
the discrete-functional declaration consumed by accepted-stage history and
wall-aware maintenance. A conflicting pre-existing declaration is rejected
before velocity or pressure fields are added. Prescribed unfitted contact
also requires an explicit active liquid side so the through-liquid wall frame
and maintenance declaration cannot silently assume a phase orientation.

There is no prescribed-angle level-set residual, residual penalty,
Jacobian row, or gauge constraint. The former contact-angle penalty input is
retired. Fitted prescribed and fitted dynamic contact angles remain
unsupported because a true fitted codimension-two integration entity is not
available; those configurations fail before system mutation. Fitted pinning
is retained only with coupled ALE.

Dynamic unfitted contact uses the generated codimension-two contact marker
and the generated sharp active-boundary marker for the wetted wall. The
contact and wall operators therefore share the authoritative cut snapshot
without using a diffuse wall indicator. Dynamic contact with cut-volume
active-domain integration requires zero smoothing width.

## Accepted-stage provenance

Every accepted dynamic-contact record binds the following to one
generalized-alpha stage:

- accepted step, accepted time, stage time, and stage fraction;
- communicator-consistent content fingerprints for the previous and endpoint
  algebraic states;
- a composite stage fingerprint over those content identities, the accepted
  snapshot, stage time and fraction, and reconstructed stage solution;
- the pre-maintenance endpoint revision, which exactly matches every contact
  stage endpoint content fingerprint, and the separately retained
  post-maintenance accepted-state content fingerprint in `state_revision`;
- complete geometry snapshot and source-value revisions;
- wall normal, footprint direction, oriented contact-line tangent, and
  contact position;
- dynamic angle, contact speed, and advancing/receding classification;
- wall-slip speed and constitutive residual; and
- line-friction and wall-slip dissipation.

The stage solution is reconstructed at commit readiness from the finalized
endpoint, its authoritative geometry is refreshed transactionally, and the
endpoint geometry is restored afterward. At the start of the accepted-step
callback, after the time history accepts that endpoint but before any
accepted-step maintenance, its algebraic revision is captured and bound into
every contact-stage record. Maintenance may advance the accepted state
revision without rewriting that captured identity. Missing, stale,
rank-inconsistent, endpoint-inconsistent, or declaration-inconsistent stage
geometry or algebraic provenance is rejected rather than recorded.
Binding first verifies communicator-consistent stage coverage and every
recomputed composite stage fingerprint. Endpoint and composite revisions are
updated only after the complete collective preflight succeeds, so asymmetric
stage solutions or metadata leave all stage records unchanged.

The frozen serial application test explicitly emulates the four relevant
transitions in order: generalized-alpha endpoint finalization, commit-ready
stage reconstruction, `TimeHistory::acceptStep()` (including its backend
counter bump), and accepted-callback content capture/bind. It intentionally
does not claim a full nonlinear `TimeLoop` integration run: such a fixture
would require a second coupled solve/mesh campaign, while this frozen matrix
is limited to low-level prerequisite evidence.

These WP-5 history revisions fingerprint gathered FE-ordered algebraic
content; they are not rank-local backend mutation counters. This prerequisite
does not provide Q0's still-open nine-field atomic accepted-state identity or
artifact contract.

## Wall-aware reinitialization

Accepted dynamic-angle contact patches are preserved by one positive common
scale. This leaves their zero crossing and unit normal unchanged while
allowing signed-distance repair away from the accepted contact.

For prescribed angle, the accepted contact point, physical wall normal, and
oriented contact-line tangent define a complete physical frame. The target
normal is

\[
n_\star=-\cos(\theta_e)n_w+\sin(\theta_e)m_w,
\]

where \(m_w\) is the oriented wall conormal derived from the accepted line
tangent and wall normal. The affine target
\(\phi_\star(x)=n_\star\mathbin{\cdot}(x-x_{\mathrm{CL}})\) preserves the
accepted contact point and enforces the target angle in two and three
dimensions. Positive rescaling of the incoming level set does not change
the prescribed result.

Distributed repair gathers the complete constraint payload, requires one
owner for every coefficient and coordinate, rejects mixed snapshot revisions
or conflicting frames, and canonicalizes duplicate ghost observations.
Serial and distributed paths consume the same canonical constraint set.

## Accepted versus rejected maintenance work

Prospective endpoint maintenance runs inside an explicit transaction.
Reinitialization, reconciliation, and correction rows are staged with their
algebraic, snapshot, topology, and extension-map revisions. A later
contact-motion budget failure rejects the complete transaction and restores
the candidate solution, geometry, accounting state, and accepted history.
Only an accepted endpoint commits its rows. Rejected attempts remain
separate and contribute zero to accepted accounting.

This transaction boundary is prerequisite evidence for truthful accounting;
it is not the complete WP-8 energy balance and does not establish WP-5
physical accuracy.

## Frozen prerequisite evidence

The matrix
`tests/cases/fluid/free_surface_wp5_contact_line_qualification_matrix.json`
and wrapper
`tests/cases/fluid/run_free_surface_wp5_contact_line_qualification.py`
freeze 43 low-level tests. The matrix is byte-frozen at SHA-256
`80b9c62256566ae39193a091171fff67ab37dc169398f288a96f8e280de9ab18`.
The wrapper accepts only
`low_level_prerequisite`, rejects FSR-05, WP-5, and Q4 closure requests
before execution, verifies the canonical matrix bytes and architecture
record, and propagates an explicit open disposition into artifacts.

The two MPI groups are frozen at two ranks. Four-rank audit runs exercise the
same rank-independent repair and accepted-frame tests but do not replace the
still-open representative partition campaign.

## Open qualification campaigns

All required physical and refinement campaigns remain unclaimed:

1. Four advancing/receding cases on three meshes and three time steps.
2. Five-angle bottom- and side-wall sessile-drop relaxation.
3. Reusken spreading and contracting drops.
4. Resolved-slip dynamic wetting at slip-to-mesh ratios 2, 4, and 8.
5. Numerical wall-width ratios 0, 0.5, 1, and 2 with an independent-limit
   check.
6. A public capillary-rise comparison with reported uncertainty.
7. Representative partition sweeps after the low-level MPI gate.

These campaigns must freeze their dimensional parameters, initial data,
reference observables, error norms, convergence thresholds, and uncertainty
policy before execution.

## Closure rule

No low-level test count, serial/MPI equivalence check, or contact-line
diagnostic closes FSR-05, WP-5, or Q4. Closure requires the open campaigns to
pass at one immutable source revision with complete source, binary,
configuration, and result provenance.

## Source evidence map

- Input allowlist and aliases:
  `Code/Source/solver/Parameters.cpp`.
- Input translation:
  `Code/Source/solver/Physics/Formulations/NavierStokes/NavierStokesRegister.cpp`.
- Ren--E, wall energy, sharp wall/contact ownership, and fitted capability
  boundary:
  `Code/Source/solver/Physics/Formulations/NavierStokes/IncompressibleNavierStokesVMSModule.{h,cpp}`.
- Accepted-stage and maintenance transactions:
  `Code/Source/solver/Application/Core/ApplicationDriver.{h,cpp}`.
- Wall-aware repair:
  `Code/Source/solver/FE/LevelSet/LevelSetReinitialization.{h,cpp}`.
- Physics regressions:
  `Code/Source/solver/Physics/Tests/Unit/test_MovingDomainPhysics.cpp` and
  `Code/Source/solver/Physics/Tests/Unit/test_NavierStokesLegacyBCs.cpp`.
- Serial and distributed maintenance regressions:
  `Code/Source/solver/FE/Tests/Unit/LevelSet/test_LevelSetReinitialization{,MPI}.cpp`
  and
  `Code/Source/solver/Application/Tests/Unit/test_ApplicationDriverLevelSetWorkflows{,MPI}.cpp`.
