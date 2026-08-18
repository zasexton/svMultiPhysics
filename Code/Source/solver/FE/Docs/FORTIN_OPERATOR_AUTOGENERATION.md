# Fortin Operator Autogeneration

## Purpose

The FE analysis pipeline can now generate theorem-backed evidence for selected
mixed finite element pairs when `InfSupPairCertification` has been requested by
the numeric summary planner. The implementation is intentionally conservative:
it emits a certified `InfSupCondition` only when the coupling, spaces, mesh
scope, domain assumptions, and boundary/nullspace handling match a registry
entry.

Unknown spaces, custom elements, ambiguous couplings, equal-order unstabilized
pairs, and stabilized surrogate pairs do not receive Fortin certificates. They
remain eligible for the existing numeric and stabilized evidence paths:
`InfSupEstimate`, `SchurComplement`, and `StabilizationAdequacy`.

## Pipeline

The implementation lives in `Analysis/FortinOperatorAutogeneration.*`.

1. `FESystem::runProblemAnalysis()` populates canonical field metadata:
   element family, Sobolev family, continuity class, mapping transform,
   reference-cell family, polynomial orders, enrichment metadata, conformity
   metadata, and conservative domain/boundary/nullspace flags.
2. `MixedCouplingClassifier` scans normalized contributions and available
   `FormExpr` DAGs. It recognizes divergence-pressure forms, integration-by-
   parts variants, H(div) divergence constraints, trace/mortar-style generic
   multiplier blocks, and stabilization surrogate evidence without relying on
   field names.
3. `FortinTheoremRegistry` matches a classified coupling and two field
   descriptors against theorem entries. A miss carries structured rejection
   reasons such as missing metadata, unsupported pair, wrong order relation,
   wrong mesh family, missing boundary/nullspace assumption, or stabilized
   surrogate.
4. `FortinCandidateBuilder` combines coupling, theorem, mesh, domain, boundary,
   and global constraint evidence into a `FortinCandidate`. Constructive or
   commuting-projection entries also get a structured projection plan.
5. `FortinCertificationAnalyzer` runs after `NumericSummaryPlanner`. Complete
   candidates become `InfSupPairCertificationSummary` objects and certified
   `InfSupCondition` claims. Incomplete and blocked candidates emit diagnostic
   issues only. Each run also emits a separate analyzer run-log summary with
   attempted, certified, incomplete, blocked, unsupported, and skipped counts.
6. `LocalFortinProjectionBuilder` is optional. By default it records theorem
   metadata and projection-plan metadata only. When local construction or
   preservation verification is requested, it builds reference-element
   constrained projection matrices for theorem-backed H1 Taylor-Hood/MINI-style
   candidates and H(div) RT/BDM commuting-projection candidates. Trace and
   mortar projections still return `Unsupported` with diagnostics until trace
   DOF metadata is complete.

## Local Projection Construction

The optional local builder is deliberately reference-element scoped. It does
not assemble a global Fortin operator in the default analysis path and it does
not create a certificate for an unsupported theorem candidate. For supported
candidate pairs it:

1. Builds target, source, and multiplier reference bases from the candidate and
   field metadata. H1 constructive entries use a componentwise target basis and
   a higher-order componentwise Lagrange source basis. MINI-style entries add
   bubble enrichment to the target when the enrichment metadata is visible.
   RT/BDM entries use the H(div) basis as both source and target.
2. Assembles reference-element matrices by quadrature: target mass `M`,
   target/source mass coupling `B`, target divergence moments `D`, and source
   divergence moments `C`.
3. Selects an independent row set of `D` and solves one constrained projection
   system per source basis column:
   `[M D^T; D 0] [x; lambda] = [B; C]`.
4. Stores the resulting row-major local projection matrix with shape metadata
   and verifies the full divergence-moment residual `D P - C`.
5. Optionally reports a Frobenius-norm estimate of the reference matrix. This
   estimate is diagnostic numeric evidence only; symbolic theorem constants
   remain the certification authority unless a theorem entry explicitly
   advertises numeric bounds.

Failed solves, unsupported reference cells, incomplete candidates, and failed
preservation checks produce noncertifying diagnostics. They do not downgrade
into a false Fortin certificate.

## Analyzer Run Logs

`ProblemAnalysisReport` contains a separate `run_logs` vector for compact
multi-step analyzer summaries. This is intentionally separate from claims and
issues: claims state mathematical conclusions, issues report user-facing
problems, and run logs describe what an analyzer attempted, skipped, certified,
or rejected.

Fortin certification run logs include:

- analyzer and summary identifiers
- status: `certified`, `partial`, `blocked`, `incomplete`, `unsupported`, or
  `skipped`
- attempted, certified, incomplete, blocked, unsupported, and skipped counts
- candidate detail lines with variables, status, coupling family, theorem id,
  and rejection reasons
- deduplicated diagnostics from candidate construction

The same data is available through `ProblemAnalysisReport::print()`,
`printApplicationLog()`, and `printTraceLog()`. Trace output includes separate
`run_log_detail` and `run_log_diagnostic` entries for downstream tooling.

## Certification Contract

A generated certificate must identify:

- primal and multiplier variables
- coupling family and contribution scope
- theorem or construction id
- primal and multiplier space families, element families, and polynomial orders
- topological dimension and reference-cell family
- mesh family and shape-regularity assumptions
- Lipschitz or other required domain assumptions
- boundary and nullspace handling, such as strong Dirichlet support, mean-zero
  pressure, gauge fixing, or compatible H(div) normal-trace scope
- whether the evidence is a known stable pair, explicit Fortin construction, or
  commuting projection
- symbolic or numeric beta/Fortin norm-bound metadata required by the theorem

The analyzer must not invent numeric constants. If the literature only gives a
uniform constant depending on shape regularity and domain class, the summary
records symbolic bound availability and leaves the numeric value unset.

## Theorem Registry Policy

Registry entries must be mathematical-space entries, not physics entries. Do not
branch on names such as velocity, pressure, displacement, fluid, solid, or
poroelasticity. Physics modules participate by emitting the same field,
contribution, boundary, and constraint metadata used by every other problem.

To add a theorem entry:

1. Add the precise space requirements: Sobolev family, element family,
   continuity, mapping transform, order relation, dimension, and supported
   reference cells.
2. State the mesh, domain, and boundary/nullspace assumptions in strings that
   can be copied into diagnostics.
3. Choose the weakest valid evidence kind:
   `KnownStablePair`, `ExplicitFortinConstruction`, or `CommutingProjection`.
4. Mark beta and Fortin norm bounds as `Symbolic`, `Scoped`, or `Numeric`.
   Leave them `Unavailable` if the theorem does not provide that part of the
   proof path.
5. Add tests for an accepted match and for each important miss mode.

Stabilized equal-order formulations should not be represented as Fortin entries
unless the exact stabilized formulation and its proof assumptions are encoded.
Otherwise they are stabilization evidence, not stable-pair certification
evidence.

## Initial Entries

The initial registry contains:

- `fortin:taylor-hood-p2-p1-simplex`: H1 Lagrange vector primal order 2 with
  H1/L2 Lagrange scalar multiplier order 1 on shape-regular simplex meshes.
  Evidence is an explicit Fortin construction with symbolic beta and Fortin
  norm bounds.
- `fortin:mini-p1bubble-p1-simplex`: bubble-enriched H1 Lagrange primal order
  1 with Lagrange multiplier order 1 on shape-regular simplex meshes. Evidence
  is an explicit stable-pair/Fortin construction with symbolic bounds.
- `fortin:rtk-dgk-hdiv-divergence`: Raviart-Thomas H(div) primal order `k`
  with DG/L2 multiplier order `k`. Evidence is a commuting projection.
- `fortin:bdmk-dgkminus1-hdiv-divergence`: BDM H(div) primal order `k` with
  DG/L2 multiplier order `k - 1`. Evidence is a commuting projection.

Trace, mortar, and other multiplier families are intentionally deferred until
their FE-space metadata is strong enough to identify the exact theorem scope.

## Literature Rationale

The basic Fortin criterion is the basis of the analyzer: if the continuous
coupling satisfies an inf-sup condition with constant `beta` and there is a
bounded operator `Pi_h : V -> V_h` preserving the mixed constraint against all
`q_h in Q_h`, then the discrete inf-sup constant is bounded below by
`beta / ||Pi_h||`. This is the only reason the analyzer may turn a registry
match into a certificate. A concise statement appears in Chen's Taylor-Hood
Fortin construction and in Schoberl's abstract mixed FEM notes. The general
background is the Brezzi-Fortin and Boffi-Brezzi-Fortin mixed FEM theory.

The Taylor-Hood entry is justified by Fortin constructions for continuous
quadratic velocity and continuous affine pressure spaces. Falk gives a
two-dimensional construction for triangular and rectangular Taylor-Hood spaces.
Diening, Storn, and Tscherpel give a local Fortin operator for the lowest-order
Taylor-Hood element in arbitrary dimension and explicitly discuss the role of
shape regularity, divergence preservation, and local stability.

The MINI entry follows Arnold, Brezzi, and Fortin's bubble-enriched P1/P1
Stokes element: continuous piecewise linear velocity enriched by element
bubbles paired with piecewise linear pressure satisfies the usual inf-sup
condition.

The RT/DG and BDM/DG entries are commuting-projection certificates. Raviart and
Thomas introduced the H(div)-conforming mixed family for second-order elliptic
problems. Brezzi, Douglas, and Marini introduced BDM families as alternatives
to Raviart-Thomas-Nedelec spaces. Finite element exterior calculus explains the
shared principle used by the analyzer: stable discretization follows when the
finite element spaces form a subcomplex and admit bounded cochain projections
that commute with the differential operator. For these divergence pairs, the
projection-plan metadata records preservation of the projected divergence.

The optional local matrix builder follows the algebraic part of these proofs:
it verifies the exact reference-element moment identity used by Fortin and
commuting-projection arguments. The matrix is not treated as an independent
global proof; it is extra evidence that the selected theorem-backed local
spaces can realize the required moment constraints with the exposed FE basis
APIs.

## References

- F. Brezzi and M. Fortin, *Mixed and Hybrid Finite Element Methods*,
  Springer, 1991.
- D. Boffi, F. Brezzi, and M. Fortin, *Mixed Finite Element Methods and
  Applications*, Springer, 2013, https://books.google.com/books?id=mRhAAAAAQBAJ.
- L. Chen, "A simple construction of a Fortin operator for the two dimensional
  Taylor-Hood element", Computers and Mathematics with Applications, 2014,
  https://www.math.uci.edu/~chenlong/Papers/infsupTaylorHood.pdf.
- R. S. Falk, "A Fortin operator for two-dimensional Taylor-Hood elements",
  ESAIM: M2AN 42(3), 2008, https://eudml.org/doc/250343.
- L. Diening, J. Storn, and T. Tscherpel, "Fortin operator for the Taylor-Hood
  element", Numerische Mathematik 150, 2022,
  https://doi.org/10.1007/s00211-021-01260-1.
- D. N. Arnold, F. Brezzi, and M. Fortin, "A stable finite element for the
  Stokes equations", Calcolo 21, 1984,
  https://doi.org/10.1007/BF02576171.
- P. A. Raviart and J. M. Thomas, "A mixed finite element method for second
  order elliptic problems", Lecture Notes in Mathematics 606, Springer, 1977.
- F. Brezzi, J. Douglas Jr., and L. D. Marini, "Two Families of Mixed Finite
  Elements for Second Order Elliptic Problems", Numerische Mathematik 47, 1985,
  https://eudml.org/doc/133032.
- D. N. Arnold, R. S. Falk, and R. Winther, "Finite element exterior calculus:
  from Hodge theory to numerical stability", Bulletin of the AMS 47, 2010,
  https://arxiv.org/abs/0906.4325.
- D. N. Arnold and J. Guzman, "Local L2-bounded commuting projections in FEEC",
  ESAIM: M2AN 55(5), 2021, https://doi.org/10.1051/m2an/2021054.
