# Fortin Operator Autogeneration Plan

## Purpose

Add physics-agnostic FE infrastructure that can automatically attempt Fortin
operator evidence when `InfSupPairCertification` is requested by the analysis
pipeline.

This should be conservative. The system may automatically discover candidate
mixed pairs and assemble supporting metadata, but it must only emit certified
`InfSupPairCertificationSummary` evidence for known theorem-backed space pairs
or explicitly supported constructive Fortin projections.

Unknown pairs, equal-order unstabilized pairs, and stabilized surrogate pairs
should continue to route through `InfSupEstimate`, `SchurComplement`, and
`StabilizationAdequacy` evidence rather than claiming a Fortin certificate.

## Certification Contract

A generated Fortin certificate must support the inf-sup claim by furnishing:

- primal and multiplier variables
- coupling family and coupling scope
- theorem or construction id
- primal and multiplier space families
- polynomial orders and element families
- mesh family, dimension, and shape-regularity assumptions
- domain and boundary-condition assumptions
- nullspace or gauge handling
- stable-pair evidence or Fortin-operator evidence
- beta lower-bound metadata when the theorem requires it
- Fortin norm bound metadata when the proof path uses an explicit Fortin map

The producer must not mark a pair certified when any theorem assumption is
missing or contradicted.

## Step 1: Enrich Space Metadata

Goal: make each discrete variable sufficiently self-describing for theorem
matching.

Concrete Completion Checklist:

- [ ] Add canonical FE metadata fields to `FieldDescriptor` or a closely related
      FE-side analysis descriptor:
      `space_family`, `element_family`, `basis_family`,
      `value_structure`, `polynomial_order`, `component_orders`,
      `continuity`, `sobolev_family`, `mapping_transform`,
      `reference_cell_family`, `conformity`, and `metadata_confidence`.
- [ ] Represent at least these space families as explicit enum values:
      Lagrange, bubble-enriched Lagrange, Raviart-Thomas, BDM, Nedelec, DG,
      trace, mortar, dual mortar, static-condensed enrichment, and unknown.
- [ ] Store scalar, vector, tensor, normal-trace, tangential-trace, and
      multiplier value structure independently from the element or space
      family.
- [ ] Store polynomial order per field and, where the FE API exposes it, per
      component or per subspace; preserve reduced-order and enriched-order
      variants without flattening them to one scalar value.
- [ ] Store continuity class as a distinct field with values for continuous,
      discontinuous, normal-continuous, tangential-continuous, trace-only,
      interface-only, and unknown.
- [ ] Store Sobolev family as `H1`, `Hdiv`, `Hcurl`, `L2`, trace, mortar, DG, or
      unknown; do not infer Sobolev family from a variable name or physics
      equation label.
- [ ] Store mapping and transform metadata using identity, covariant Piola,
      contravariant Piola, trace pullback, normal-trace pullback,
      tangential-trace pullback, mortar interface map, or unknown.
- [ ] Store reference-cell family and dimension using simplex, tensor-product,
      wedge, pyramid, mixed, face-simplex, face-tensor-product, or unknown.
- [ ] Store enrichment metadata for bubble degree, macroelement tag, reduced
      variant, hierarchical basis flag, static condensation status, and whether
      the enrichment is visible to analysis.
- [ ] Store conformity metadata needed by commuting-projection theorems,
      including global conformity, element-local conformity, boundary
      conformity, sequence membership, and whether orientation data is present.
- [ ] Populate the descriptor from `FunctionSpace` construction and existing FE
      space factories before analysis starts; add a single adapter layer when
      direct population from every concrete space class is not practical.
- [ ] Populate mesh-derived fields from mesh/reference-cell descriptors rather
      than from physics modules, including dimension, cell family, face family,
      mixed-cell status, and shape-regularity summary availability.
- [ ] Populate boundary, nullspace, and gauge metadata into
      `ProblemAnalysisContext` so theorem matching can verify pressure gauges,
      mean-zero constraints, pinned multipliers, and Dirichlet support.
- [ ] Define fallback values of `Unknown` for every metadata field that the FE
      library cannot infer; pair the fallback with a missing-field diagnostic
      code so certification is blocked explainably.
- [ ] Add trace and mortar metadata placeholders even before certification is
      supported: parent volume field id, trace side, interface marker,
      orientation availability, trace basis id, trace quadrature availability,
      paired interface id, mortar primal side, mortar multiplier side, dual
      basis id, and intersection quadrature availability.
- [ ] Expose all space descriptors through `ProblemAnalysisContext` using stable
      field ids so `FESystem`, `NumericSummaryPlanner`,
      `FortinCertificationAnalyzer`, `FortinCandidateBuilder`, and
      `InfSupAnalyzer` observe the same metadata.
- [ ] Add descriptor tests for H1 Lagrange fields, DG pressure fields, RT fields,
      BDM fields, Nedelec fields, bubble-enriched MINI-style fields, and
      vector-valued H1 fields once those spaces are present in FE tests.
- [ ] Add descriptor tests for trace and mortar spaces that assert incomplete
      metadata remains explicit and noncertifying until trace/mortar support is
      completed.
- [ ] Add negative tests proving custom or unknown spaces remain analyzable,
      appear in diagnostics, and do not produce Fortin certificates.

Acceptance criteria:

- [ ] A `ProblemAnalysisContext` can expose all space metadata needed for a
      theorem match without consulting physics-specific code.
- [ ] Missing metadata is explicit and blocks certification rather than causing
      false-positive certification.

## Step 2: Add Mixed Coupling Classification From DAG and Contributions

Goal: identify the mathematical coupling that an inf-sup theorem would apply to.

Concrete Completion Checklist:

- [ ] Add a `MixedCouplingDescriptor` type in FE analysis with stable ids for
      primal variable, multiplier variable, trial/test role, operator tag,
      contribution ids, matrix block ids, domain marker scope, boundary marker
      scope, interface marker scope, and classification source.
- [ ] Define coupling-family enum values for divergence-pressure,
      gradient-divergence adjoint, mixed Hdiv divergence, trace multiplier,
      mortar constraint, curl/div de Rham, generic multiplier constraint,
      stabilized surrogate, numeric-estimate-only, ambiguous, and unsupported.
- [ ] Add classifier confidence values for strong evidence, weak evidence,
      ambiguous evidence, contradicted evidence, and missing evidence.
- [ ] Implement DAG classifiers for `q * div(u)` and equivalent forms where the
      DAG exposes a divergence operator on the primal field and an L2-like
      multiplier test field.
- [ ] Implement DAG classifiers for `grad(q) dot u` and integration-by-parts
      variants only when contribution metadata or boundary terms support the
      divergence-pressure interpretation.
- [ ] Implement H(div) classifiers for normal-flux or divergence constraints
      using RT/BDM-like primal descriptors, DG/L2-like multiplier descriptors,
      and operator tags from the DAG or contribution descriptor.
- [ ] Implement trace-multiplier classification only to the descriptor level
      until trace local construction is complete; require trace operator tags,
      parent-field ids, interface markers, and side/orientation metadata before
      reporting strong evidence.
- [ ] Implement mortar classification only as noncertifying candidate discovery
      until mortar metadata is complete; require primal side, multiplier side,
      interface pairing, and matching/nonmatching interface classification.
- [ ] Implement generic Lagrange multiplier constraint classification from
      `ContributionDescriptor` roles and block pairing metadata without relying
      on variable names such as velocity, pressure, displacement, or lambda.
- [ ] Record whether evidence came from DAG structure, contribution role,
      matrix-block metadata, boundary/interface metadata, or a combination of
      those sources.
- [ ] Preserve contribution ids and scope so rejection diagnostics can point to
      the exact terms that triggered or blocked candidate creation.
- [ ] Treat algebraically equivalent DAG rewrites as the same coupling when the
      operator tree and contribution metadata identify the same bilinear form.
- [ ] Route conflicting classifications to an incomplete descriptor with
      `AmbiguousCoupling` rejection detail instead of choosing an arbitrary
      theorem.
- [ ] Route stabilized surrogate classifications to stabilization and numeric
      evidence paths; do not send them to Fortin theorem matching unless a
      registry entry explicitly declares that stabilized formulation.
- [ ] Add accepted classifier tests for divergence-pressure, gradient-divergence
      adjoint, RT/DG divergence, BDM/DG divergence, and generic multiplier
      constraint blocks.
- [ ] Add rejection tests for same-field blocks, diagonal-only blocks, missing
      multiplier role, missing divergence operator, wrong domain/interface
      scope, and ambiguous DAG rewrites.
- [ ] Add trace/mortar classifier tests that verify descriptor creation plus
      noncertifying status when orientation, face quadrature, or pairing
      metadata is absent.

Acceptance criteria:

- [ ] The FE analyzer can identify candidate inf-sup pairs and coupling family
      without physics-specific formulation names.
- [ ] Ambiguous DAGs produce incomplete candidate descriptors, not
      certifications.

## Step 3: Implement a Fortin Theorem Registry

Goal: match candidate pairs against known stable-pair or constructive Fortin
theorems.

Concrete Completion Checklist:

- [ ] Add an FE-side `FortinTheoremRegistry` with deterministic lookup by
      coupling family, primal space family, multiplier space family, dimension,
      cell family, polynomial-order relation, and theorem scope.
- [ ] Define `FortinTheoremEntry` fields for theorem id, display name,
      literature reference id, pair family, coupling family, primal family,
      multiplier family, supported dimensions, supported reference cells,
      required mesh assumptions, required domain assumptions, required boundary
      assumptions, required nullspace/gauge handling, order relation,
      supported mappings, required conformity, proof path, and certification
      status.
- [ ] Include proof-path values for known stable pair, explicit Fortin
      construction, commuting-projection construction, local constructive check,
      numerical estimate only, stabilized surrogate, and unsupported.
- [ ] Include beta-bound metadata fields for bound availability, symbolic bound
      label, numeric lower bound when known, mesh-scope dependence, polynomial
      order dependence, and assumptions needed for the bound.
- [ ] Include Fortin-norm-bound metadata fields for bound availability,
      symbolic label, norm family, local/global scope, estimate source, and
      optional reference-element estimate support.
- [ ] Define machine-readable rejection reasons for unsupported pair, missing
      metadata, wrong order relation, wrong mesh family, wrong dimension,
      missing domain assumption, missing boundary condition, missing gauge,
      missing nullspace handling, unsupported mapping, unsupported trace
      orientation, unsupported mortar pairing, stabilized surrogate, numerical
      evidence only, and ambiguous coupling.
- [ ] Add initial theorem entries for Taylor-Hood-like H1 velocity and pressure
      pairings only where space metadata, polynomial relation, mesh family,
      boundary support, and pressure gauge assumptions match the theorem.
- [ ] Add initial theorem entries for MINI-type bubble-enriched H1/P1 pairs when
      bubble metadata, enrichment visibility, and macroelement assumptions are
      present.
- [ ] Add entries for RT_k/DG_k mixed Hdiv divergence pairs with
      contravariant-Piola mapping, DG multiplier metadata, supported cell
      families, and commuting-projection proof path.
- [ ] Add entries for BDM_k/DG_{k-1} mixed Hdiv divergence pairs with the same
      mapping and conformity requirements.
- [ ] Add theorem-scoped placeholders for trace and mortar pairs that return
      noncertifying diagnostics until trace basis, face quadrature, orientation,
      interface pairing, and mortar dual-basis metadata are available.
- [ ] Explicitly mark equal-order stabilized pairs as stabilized surrogate
      candidates, not Fortin-certified pairs, unless a theorem entry is added
      for that exact stabilized formulation and all theorem assumptions are
      represented.
- [ ] Require every registry hit to emit an explanation object containing the
      matched fields, matched assumptions, bound metadata, and any optional
      local-construction request.
- [ ] Require every registry miss to emit a complete rejection list instead of a
      single best-effort error string.
- [ ] Add theorem-match tests for every accepted initial entry and every
      rejection reason above.
- [ ] Add stabilized-surrogate tests proving equal-order stabilized pairs are
      routed to stabilization adequacy and numeric inf-sup evidence, not Fortin
      certification.

Acceptance criteria:

- [ ] Registry matches are deterministic, physics-agnostic, and explainable.
- [ ] Registry misses include a reason: unsupported pair, missing metadata,
      wrong order relation, wrong mesh family, or missing boundary/nullspace
      assumption.

## Step 4: Add a Fortin Candidate Builder

Goal: produce a concrete evidence candidate from coupling, space, mesh, and
theorem metadata.

Concrete Completion Checklist:

- [ ] Add `FortinCandidateBuilder` under FE analysis and keep it independent of
      physics modules.
- [ ] Consume `MixedCouplingDescriptor`, field descriptors, function-space
      descriptors, mesh metadata, domain metadata, boundary descriptors,
      nullspace/gauge metadata, stabilization metadata, and registry lookup
      results.
- [ ] Build candidates only from stable field ids and contribution ids exposed
      by `ProblemAnalysisContext`; do not reconstruct variable meaning from
      names or equation labels.
- [ ] Match candidates against `FortinTheoremRegistry` and preserve all matched
      theorem explanation fields on the candidate.
- [ ] Produce `FortinCandidate` fields for theorem id, pair family, proof path,
      primal variable, multiplier variable, coupling family, coupling scope,
      domain scope, interface scope, assumption evidence, rejection reasons,
      local projection plan, and certification readiness.
- [ ] Distinguish stable-pair-only candidates, explicit Fortin-operator
      candidates, commuting-projection candidates, stabilized-surrogate
      candidates, and numeric-estimate-only candidates.
- [ ] Implement assumption gates for mesh family, dimension, cell family,
      shape-regularity evidence, domain regularity, boundary support, gauge or
      mean-zero pressure, multiplier pinning, nullspace treatment, mapping
      transform, and conformity.
- [ ] Record missing assumptions as incomplete candidate diagnostics and
      contradicted assumptions as hard blockers.
- [ ] Preserve rejected candidates in diagnostics so the run log can explain
      why no certification summary was produced.
- [ ] For constructive theorem entries, generate a projection-plan descriptor
      containing reference cell, source space, target space, interpolation or
      projection type, local moment constraints, correction space, preserved
      quantity, norm family, quadrature requirement, and optional residual
      check.
- [ ] For stable-pair-only theorem entries, generate no local operator request
      unless the analysis options explicitly request an auxiliary local
      construction.
- [ ] Include global constraint handling in the candidate: pressure gauge,
      mean-zero constraint, pinned multiplier, compatible trace constraint,
      strong Dirichlet support, or explicit nullspace removal.
- [ ] Include mesh-quality evidence from existing `MeshGeometryQualitySummary`
      where available, and report `MissingMeshQualityEvidence` when the theorem
      requires it but no summary exists.
- [ ] Avoid assembling large dense global operators in the default analyzer
      path; the candidate builder should produce metadata and optional local
      construction plans only.
- [ ] Gate expensive local projection matrix construction, residual checks, and
      norm-bound estimates behind an analysis option.
- [ ] Emit noncertifying diagnostics for trace/mortar candidates until the
      trace/mortar completion delta is implemented.
- [ ] Add complete-candidate tests for Taylor-Hood-like metadata, MINI-like
      metadata, RT/DG metadata, and BDM/DG metadata.
- [ ] Add incomplete-candidate tests for missing pressure gauge, missing
      mean-zero metadata, missing Dirichlet support, unknown mesh family,
      missing shape-regularity evidence, wrong order relation, ambiguous
      coupling, and trace/mortar missing metadata.

Acceptance criteria:

- [ ] Candidate building can run during analysis without physics modules
      registering custom metadata.
- [ ] Constructive evidence is represented as structured metadata even before
      optional local matrices are generated.

## Step 5: Add Optional Local Fortin Projection Construction

Goal: provide stronger evidence for theorem entries that have constructive
projections.

Concrete Completion Checklist:

Completed H1/H(div) local projection scope:

- [x] Add reference-element projection builder interfaces.
- [x] Support local interpolation/projection basis evaluations and
      coefficient-space matrices for supported spaces.
- [x] Support local mass or moment projection solves.
- [x] Support bubble or correction-space solves for MINI-like constructions.
- [x] Support divergence-preserving projection checks for Hdiv pairs.
- [x] Support commuting-projection checks for de Rham sequence pairs where FE
      spaces expose the necessary transforms. The implemented scope is the
      H(div) RT/BDM divergence sequence covered by the initial registry.
- [x] Produce local operator shape metadata without requiring global assembly.
- [x] Optionally compute local norm-bound estimates when requested.
- [x] Verify preservation identities on reference elements using quadrature with
      sufficient order.
- [x] Surface local construction failures as noncertifying diagnostics.
- [x] Keep the default analysis path theorem-match based; enable full local
      projection construction only through an analysis option.
- [x] Add unit tests for local preservation identities on supported reference
      elements.

Trace/mortar local projection scope still to finish:

- [ ] Add trace-space projection plan support that names the parent volume
      field, trace field, interface marker, side, orientation convention, trace
      basis family, trace polynomial order, and trace quadrature rule.
- [ ] Add mortar projection plan support that names primal side, multiplier
      side, interface pairing, matching or nonmatching interface status,
      multiplier basis, dual/biorthogonal basis availability, and intersection
      quadrature requirement.
- [ ] Add oriented face basis evaluation support for trace spaces, including
      side-aware face maps and normal/tangential component extraction.
- [ ] Add face quadrature selection with theorem-required order and residual
      quadrature order separated from assembly quadrature order.
- [ ] Add matching-interface trace residual checks for preservation of the
      interface constraint in the appropriate trace norm.
- [ ] Add nonmatching-interface residual checks only after intersection
      quadrature and side-pair mapping metadata are available.
- [ ] Add dual mortar or biorthogonal multiplier projection support before any
      mortar theorem entry can become certifying.
- [ ] Keep unsupported trace or mortar local construction results diagnostic and
      noncertifying until every theorem assumption above is present.
- [ ] Add tests that trace/mortar local projection requests return
      noncertifying unsupported diagnostics today rather than falling through to
      a certificate.

Trace/Mortar Completion Delta:

- [ ] Add theorem-scoped trace/mortar registry entries with explicit
      assumptions for trace basis, parent volume basis, interface topology,
      orientation, quadrature, multiplier basis, and matching/nonmatching
      interface support.
- [ ] Add `TraceSpace` and `MortarSpace` analysis descriptor metadata for parent
      field ids, side ids, trace DOF layout, trace polynomial order, mortar
      primal side, mortar multiplier side, dual basis id, and interface pairing
      ids.
- [ ] Add oriented face basis and quadrature support that can evaluate trace
      basis functions, normal/tangential components, and multiplier basis
      functions using a shared orientation convention.
- [ ] Add matching-interface trace residual verification that checks the
      theorem-preserved interface moments with sufficient quadrature order.
- [ ] Add nonmatching-interface intersection quadrature before certifying mortar
      constraints on nonmatching meshes.
- [ ] Add dual mortar or biorthogonal multiplier support before certifying
      mortar Fortin operators that rely on multiplier-space projections.
- [ ] Keep all trace/mortar theorem matches and local projection attempts
      noncertifying diagnostics until the registry, descriptor, face-basis,
      quadrature, and residual assumptions are all present.

Acceptance criteria:

- [x] Supported constructive pairs can produce explicit Fortin-operator evidence
      metadata.
- [x] Unsupported or failed local constructions never downgrade into a false
      certificate.
- [ ] Trace/mortar constructive pairs remain noncertifying until the completion
      delta is implemented and tested.

Implementation review:

- Complete for theorem-backed H1 Taylor-Hood/MINI-style divergence pairs and
  H(div) RT/BDM divergence pairs through `LocalFortinProjectionBuilder`.
- The builder constructs reference-element constrained projection matrices,
  verifies divergence-moment preservation residuals, stores local shape and
  row-major matrix metadata, and optionally reports a reference Frobenius-norm
  estimate.
- Trace and mortar local operators remain deliberately unsupported until
  trace-space DOF metadata is strong enough to identify the exact theorem
  scope; this is reported as an `Unsupported` local construction result, not as
  a certificate.

## Step 6: Produce `InfSupPairCertificationSummary`

Goal: convert successful candidates into the summary type consumed by
`InfSupAnalyzer`.

Concrete Completion Checklist:

- [ ] Add an FE-side summary producer that runs when the request plan includes
      `InfSupPairCertification`.
- [ ] Run the producer from FE analysis infrastructure, not from physics
      modules, and consume candidates produced from `ProblemAnalysisContext`.
- [ ] Populate summary identity fields: primal variable id/name, multiplier
      variable id/name, coupling family, coupling scope, theorem id, proof path,
      pair family, and certification status.
- [ ] Populate primal and multiplier space fields: space family, element
      family, polynomial order, component orders, value structure, continuity,
      Sobolev family, mapping transform, reference cell family, and enrichment
      metadata.
- [ ] Populate assumption fields from the candidate: mesh family, dimension,
      shape-regularity evidence, domain assumption evidence, boundary-condition
      scope, nullspace/gauge handling, pressure mean-zero or pinning evidence,
      and interface scope where relevant.
- [ ] Populate known-stable-pair evidence when the theorem entry is stable-pair
      based, including theorem id, reference id, assumption list, and beta-bound
      metadata.
- [ ] Populate Fortin-operator evidence when the theorem entry has constructive
      evidence, including projection-plan id, local operator shape metadata,
      residual verification result, norm-bound metadata, and optional
      reference-element matrix metadata.
- [ ] Populate beta lower-bound fields only when the theorem entry provides an
      applicable numeric bound or symbolic scoped lower bound.
- [ ] Populate Fortin norm bound only when the theorem entry requires and
      provides it, or when optional local projection construction has produced a
      scoped estimate that the theorem entry allows.
- [ ] Convert complete candidates to certified
      `InfSupPairCertificationSummary` objects only after all registry and
      candidate gates pass.
- [ ] Convert incomplete candidates to diagnostics and run-log entries; do not
      place incomplete objects in the certified evidence path unless the
      existing analyzer explicitly defines a partial-summary channel.
- [ ] Ensure `AnalysisSummarySet::has(InfSupPairCertification)` is true only
      when at least one certified summary is available to `InfSupAnalyzer`.
- [ ] Ensure rejected or incomplete candidates remain available through
      diagnostics/run-log summaries even when `AnalysisSummarySet` does not
      expose certified Fortin evidence.
- [ ] Add summary-producer tests for stable-pair-only certification,
      constructive Fortin certification, optional local matrix metadata,
      missing-bound behavior, incomplete candidate behavior, and multiple
      candidate ordering.

Acceptance criteria:

- [ ] `AnalysisSummarySet::has(InfSupPairCertification)` becomes true only for
      summary evidence that is meaningful to `InfSupAnalyzer`.
- [ ] Missing evidence remains visible in run logs.

## Step 7: Integrate With Planner and Analyzer Flow

Goal: make automatic Fortin attempts happen when useful and only once per
analysis point.

Concrete Completion Checklist:

- [ ] Keep `NumericSummaryPlanner` requesting `InfSupPairCertification` for
      candidate true mixed stable-pair claims identified from coupling,
      contribution, and space metadata.
- [ ] Continue suppressing `InfSupPairCertification` for stabilized surrogate
      claims; request `StabilizationAdequacy`, `InfSupEstimate`, and related
      numeric evidence instead.
- [ ] Route equal-order unstabilized pairs to numeric inf-sup diagnostics and
      explicit rejection reasons unless a theorem registry entry supports the
      exact pair.
- [ ] Define default analyzer ordering so space descriptors and coupling
      descriptors are available before the Fortin candidate builder runs, and
      certified summaries are available before `InfSupAnalyzer` evaluates the
      requested claim.
- [ ] Add a producer hook near other post-tangent summary producers when matrix,
      mesh-quality, or local projection residual metadata is needed.
- [ ] Add a pre-tangent producer path for theorem matches that only require
      symbolic space, mesh, boundary, and nullspace metadata.
- [ ] Ensure the Fortin producer is keyed by analysis point, mesh id, field ids,
      contribution ids, and request options so it does not rerun for every
      linear solve.
- [ ] Add duplicate-run avoidance for repeated planner requests and for multiple
      analyzers consuming the same `InfSupPairCertification` summary.
- [ ] Include Fortin candidate attempts, registry matches, rejection reasons,
      local-construction request status, and certification outcomes in the run
      log.
- [ ] Make analyzer output distinguish certified stable pair, certified
      constructive Fortin pair, Fortin candidate incomplete, stabilized
      surrogate, numeric estimate only, unsupported pair, and ambiguous
      coupling.
- [ ] Ensure `InfSupAnalyzer` upgrades a claim to certified only when the
      summary passes all existing gates and the summary status is certified.
- [ ] Ensure `StabilizationAdequacy` remains the correct route for stabilized
      equal-order formulations and that those results are never reported as
      Fortin certificates by implication.
- [ ] Add integration tests for planner request behavior, producer ordering,
      duplicate-run avoidance, analyzer upgrade behavior, stabilized-surrogate
      routing, and trace/mortar unsupported diagnostics.

Acceptance criteria:

- [ ] Fortin attempts are automatic when `InfSupPairCertification` is requested.
- [ ] Ustruct-like stabilized surrogate pairs do not attempt or require Fortin
      certification.
- [ ] The run log explains both successful and failed attempts.

## Step 8: Add FE-Only Regression Tests

Goal: lock down certification behavior without relying on physics modules.

Concrete Completion Checklist:

- [ ] Add planner regression tests proving true mixed-pair claims request
      `InfSupPairCertification`.
- [ ] Add planner regression tests proving stabilized surrogate claims suppress
      `InfSupPairCertification` and request stabilization/numeric evidence
      instead.
- [ ] Add planner regression tests proving equal-order unstabilized claims stay
      numeric-estimate-only unless a matching theorem entry exists.
- [ ] Add descriptor regression tests for H1 Lagrange, vector H1 Lagrange,
      bubble-enriched MINI-like spaces, DG spaces, RT spaces, BDM spaces,
      Nedelec spaces, unknown/custom spaces, and trace/mortar placeholders.
- [ ] Add coupling-classifier regression tests for accepted
      divergence-pressure forms, integration-by-parts adjoint forms, RT/DG
      divergence forms, BDM/DG divergence forms, generic multiplier constraint
      blocks, and rejected or ambiguous DAGs.
- [ ] Add theorem-registry regression tests for every supported theorem entry,
      including Taylor-Hood-like, MINI-like, RT/DG, and BDM/DG entries.
- [ ] Add theorem-registry rejection tests for wrong polynomial order relation,
      wrong space family, wrong value structure, wrong mapping transform, wrong
      cell family, unknown mesh family, missing boundary condition, missing
      gauge/nullspace handling, missing domain assumption, stabilized surrogate
      status, trace unsupported status, and mortar unsupported status.
- [ ] Add candidate-builder tests for complete Taylor-Hood-like metadata,
      complete MINI-like metadata, complete RT/DG metadata, complete BDM/DG
      metadata, missing pressure gauge, missing mean-zero metadata, incomplete
      boundary scope, missing mesh-quality evidence, contradicted assumptions,
      and ambiguous coupling.
- [ ] Add summary-producer tests for certified stable-pair summaries,
      certified constructive summaries, optional local projection metadata,
      incomplete candidates staying noncertifying, rejected candidates appearing
      in diagnostics, and `AnalysisSummarySet` availability rules.
- [ ] Add analyzer integration tests that a complete certified summary upgrades
      the inf-sup claim and an incomplete or rejected candidate does not.
- [ ] Add stabilized-surrogate integration tests proving equal-order stabilized
      pairs route to `StabilizationAdequacy` and numeric inf-sup evidence, not
      Fortin certification.
- [ ] Add local projection tests for reference-element shape metadata,
      row-major local matrix metadata, divergence-moment preservation residuals,
      MINI-style correction-space residuals, RT/BDM commuting-projection
      residuals, optional Frobenius-norm estimate reporting, and local
      construction failure diagnostics.
- [ ] Add trace/mortar unsupported-path tests for trace descriptor gaps, mortar
      descriptor gaps, missing orientation, missing face quadrature, missing
      interface pairing, missing dual basis, and nonmatching-interface
      quadrature absence.
- [ ] Add run-log tests that successful, incomplete, unsupported, ambiguous,
      stabilized-surrogate, and numeric-estimate-only outcomes include theorem
      ids or rejection reasons as appropriate.
- [ ] Keep all tests FE-only: no Ustruct, Navier-Stokes, solid mechanics, or
      other physics module registration should be required.

Acceptance criteria:

- [ ] All certification behavior is covered at FE unit-test level.
- [ ] No test needs Ustruct, Navier-Stokes, or any other physics module.

## Step 9: Add Diagnostics and Documentation

Goal: make automatic Fortin attempts explainable to future developers.

Concrete Completion Checklist:

- [ ] Add structured rejection reasons to Fortin candidate diagnostics using the
      registry and candidate-builder reason codes.
- [ ] Include matched theorem id, pair family, proof path, coupling family,
      variable ids, contribution ids, and scope in successful diagnostics.
- [ ] Include missing metadata fields, contradicted assumptions, unsupported
      metadata values, and ambiguous classification details in failed
      diagnostics.
- [ ] Include whether the evidence is stable-pair-only, constructive Fortin,
      commuting-projection, stabilized surrogate, numeric-estimate-only, or
      unsupported.
- [ ] Add run-log records for attempted pairs with fields for analysis point,
      field ids, contribution ids, coupling family, theorem lookup status,
      rejection reasons, local projection status, certification status, and
      summary id.
- [ ] Add run-log summaries for attempted, certified, incomplete, unsupported,
      ambiguous, stabilized-surrogate, and numeric-estimate-only pairs.
- [ ] Add documentation sections for the certification contract, metadata flow,
      planner request flow, registry lookup flow, candidate builder, local
      projection options, summary production, analyzer consumption, and run-log
      interpretation.
- [ ] Add developer guidance for theorem entries: required fields, how to encode
      order relations, how to encode mesh/domain/boundary assumptions, how to
      cite literature, how to declare proof path, how to declare beta/Fortin
      bounds, and how to add rejection tests.
- [ ] Add literature traceability for each theorem id using stable reference ids
      and short notes on the theorem assumptions represented in metadata.
- [ ] Document why physics modules should not manually encode Fortin
      certificates unless they are registering generic FE-space or theorem
      metadata through the FE interfaces.
- [ ] Document that custom spaces require either registry support or they remain
      numeric-estimate-only with explicit diagnostics.
- [ ] Document trace/mortar limitations as deliberate noncertifying diagnostics
      until the trace/mortar completion delta is implemented.
- [ ] Add examples of successful run-log output, missing-metadata output,
      stabilized-surrogate output, and trace/mortar unsupported output.

Acceptance criteria:

- [ ] A developer can understand why a pair did or did not certify from the run
      log.
- [ ] A developer can add a new theorem-backed pair without modifying physics
      modules.

## Suggested Implementation Order

Checklist:

- [ ] Implement space metadata enrichment.
- [ ] Implement mixed coupling descriptor and DAG/contribution classifier.
- [ ] Implement theorem registry with no local projection construction.
- [ ] Implement candidate builder using theorem matching only.
- [ ] Implement summary producer and analyzer integration.
- [ ] Add tests for Taylor-Hood-like and RT/DG theorem matches.
- [ ] Add rejection tests for equal-order unstabilized and stabilized surrogate
      pairs.
- [ ] Add optional local projection construction only after theorem-match
      certification is stable.
- [ ] Add trace/mortar completion work only after descriptor, theorem, face
      basis, quadrature, and residual requirements are represented.
- [ ] Add additional theorem entries incrementally.

## Non-Goals

- Do not infer or invent Fortin theorems from arbitrary DAGs.
- Do not certify custom spaces without theorem registry support.
- Do not require physics modules to manually register certification metadata.
- Do not replace numeric `InfSupEstimate`; it remains valuable evidence for
  unsupported and stabilized-surrogate cases.
- Do not treat stabilized equal-order methods as true stable-pair Fortin
  certificates unless a specific theorem entry is added for that stabilized
  formulation.
