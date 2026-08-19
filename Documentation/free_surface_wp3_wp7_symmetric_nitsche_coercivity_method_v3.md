# WP-3/WP-7 accepted-state symmetric Nitsche coercivity prerequisite, version 3

## Status and claim boundary

Version 3 adds a predeclared positive energy floor to the supported
generated-boundary symmetric velocity-Nitsche route. The method value is

\[
c_* = \frac{1}{4}.
\]

Its intended accepted claim is
`accepted_state_coercivity_policy_prerequisite`. At the generic FE layer, the
certificate set together with the sealed form-bound penalties and policy gate
proves the following inequality conditional on a matching bulk viscous energy
\(K\). The low-level form binding authenticates the boundary route, not that
bulk term.

For the supported Navier--Stokes production route, the module supplies this
premise by installing the canonical constant-viscosity active-volume
symmetric-gradient term on the same velocity field. Let \(A_N\) denote only
that bulk term plus the generated-boundary symmetric consistency and penalty
terms, not the complete mixed Navier--Stokes operator. Every exact
current-state certificate must be current, and the gate must prove

\[
A_N(v,v)
\ge c_*\left(K(v,v)+\sum_i P_i(v,v)\right)
\]

before the production operator may assemble. A state that cannot prove this
inequality is rejected; the penalty is never selected or increased by the
certificate path. A generic FE request that supplies only a boundary route
exercises the conditional trace/penalty gate and carries no claim that its
arbitrary operator contains the required \(K\).

This is a uniform contract over the states accepted by the declared method.
It is not an existence theorem for every cut or mesh family. It does not
establish mixed inf-sup stability, conditioning, solver robustness,
manufactured-solution convergence, Navier-slip stability, or any physical
campaign exit. FSR-16, FSR-07, WP-3, WP-7, and Q1 therefore remain open.

The version-2 method, matrix, runner, and certified evidence archive remain
historical and byte-bound to their original revision. Version 3 does not
reinterpret their deliberately null method-wide lower bound.

## Supported production route

The underlying trace certificate and its exact factorized proof are those
specified by version 2. The supported envelope remains:

- reference-frame affine P1 product-H1 velocity fields;
- linear `Triangle3` and `Tetra4` cells;
- finite positive constant Newtonian viscosity;
- generated weak-Dirichlet velocity boundaries using the symmetric Nitsche
  variant;
- installation through the Navier--Stokes production module, whose canonical
  momentum residual contains the matching active-volume viscous term;
- a current cut context, generated source-value revision, finalized
  trace-eligible small-cut aggregation, and closed affine constraints;
- the fixed certificate input, dimension, work, and memory caps stated by the
  version-2 method.

Quadrilateral and hexahedral Q1 cells, higher order, non-affine or
current-frame geometry, variable viscosity, rootless aggregation, physical
Navier slip or Robin terms, and unsymmetric coercivity claims remain outside
the envelope and fail closed or carry no symmetric claim as appropriate.

## Accepted-state inequality

For generated route \(i\), let \(K_i\) be the nonnegative active viscous
energy on the support certified for that route. The exact aggregate-trace
certificate proves

\[
\int_{\Gamma_i}\frac{h_n}{\mu}
  |2\mu\varepsilon(v)n|^2
\le C_i K_i(v,v).
\]

The installed form binding seals the actual production penalty multiplier
\(\alpha_i\). Let \(P_i\) be the corresponding nonnegative route penalty
energy, and let \(B_i\) denote one of the equal boundary pairings in the
symmetric consistency term. On the supported production route,
\(A_N=K+P-2\sum_i B_i\). Thus the symmetric pair contributes
\(-2B_i(v,v)\), and the single-pair bound is

\[
|B_i(v,v)|
\le
\sqrt{\frac{C_i}{\alpha_i}K_i(v,v)P_i(v,v)}.
\]

Let \(K\) be the operator's total active viscous energy over its unique
velocity fields, so each certified support energy satisfies \(K_i\le K\),
and let \(P=\sum_iP_i\). For all symmetric routes on one operator, define
the outward upper risk ratio

\[
R_{\mathrm{op}}
= \operatorname{round}_{\uparrow}
  \sum_i\frac{C_i}{\alpha_i}.
\]

The routewise bounds and Cauchy--Schwarz over the route index give

\[
\sum_i |B_i(v,v)|
\le \sqrt{K(v,v)}\sum_i
     \sqrt{\frac{C_i}{\alpha_i}P_i(v,v)}
\le \sqrt{R_{\mathrm{op}}K(v,v)P(v,v)}.
\]

Using \(2\sqrt{KP}\le K+P\) then gives

\[
A_N(v,v)
\ge
K(v,v)+P(v,v)-2\sum_i|B_i(v,v)|
\ge
K(v,v)+P(v,v)-2\sqrt{R_{\mathrm{op}}K(v,v)P(v,v)}
\ge
\left(1-\sqrt{R_{\mathrm{op}}}\right)
\left(K(v,v)+P(v,v)\right).
\]

Consequently, a requested floor \(c\in(0,1)\) is proven whenever

\[
R_{\mathrm{op}}\le (1-c)^2.
\]

For \(c_*=1/4\), the exact-real risk threshold is \(9/16\).

## Rounding-safe acceptance rule

Acceptance does not compute `1 - sqrt(R)` in binary floating point. That
subtraction can amplify a square-root rounding error through cancellation,
so stepping only the final result downward is not a proof.

Instead, the implementation forms a downward-safe binary cap for
\((1-c)^2\). It evaluates the positive complement, steps it once toward zero,
squares that lower complement, and steps the nonzero square once toward zero.
Each step can only decrease the exact-real threshold; underflow to zero is
conservative. The outward upper \(R_{\mathrm{op}}\) is accepted only when it
is no larger than this downward cap. Therefore an accepted comparison proves
the exact-real inequality even though the cap may reject a borderline state.

After this direct proof, the record stores exactly the configured \(c\) as
its certified symmetric energy-ratio lower bound. It does not store a tighter
value inferred from a rounded square root. Current-state assembly preflight
recomputes the downward cap, rechecks the grouped ratio, and requires the
stored lower-bound bits to equal the policy-floor bits.

## Policy, provenance, and lifecycle binding

The production option and FE install request default to the predeclared
binary-exact value `0.25`. A symmetric generated-boundary request must be
finite and strictly between zero and one. The generic unsymmetric installer
ignores this field and stores exact zero because the diagonal consistency
terms cancel and no symmetric threshold applies. Production configuration
also requires the serialized option to be finite and strictly between zero
and one.

For every supported symmetric route, the floor is bound into:

- the form-derived trace policy;
- deterministic policy ordering and the collective policy signature;
- current certificate validation and the certificate-cache digest;
- effective Physics configuration provenance;
- the disabled version-3 diagnostic case and summary records.

The remaining policy binding is unchanged: operator, velocity field and
complete space signature, physical and generated boundary markers, volume
interface marker, viscosity, penalty gamma, actual polynomial order,
polynomial scaling flag, effective multiplier, symmetry variant, exact-proof
dimension cap, source formulation-record index, and form-binding digest.

Registration validates the production option and every route before adding
velocity or pressure fields. Setup validates the immutable policy table.
Relevant context or constraint changes invalidate the stale certificate
cache. Refresh constructs every current certificate in temporary storage,
coordinates local failures across ranks, applies the grouped floor gate, and
publishes no partial replacement when any route fails. Assembly checks the
current revisions, aggregation ownership and digest, exact certificate
digest, source formulation binding, floor, grouped risk, and stored bound
before touching the requested matrix or vector.

## Required version-3 evidence

The source-level prerequisite requires at least:

1. analytic acceptance above the floor and rejection of a strictly positive
   but subfloor bound with an empty certificate cache;
2. multi-route grouping, showing that the operator-level sum rather than each
   route in isolation controls acceptance;
3. unsymmetric continuity-only behavior with an absent energy bound and exact
   zero policy floor;
4. invalid production-floor rejection before field, operator, or policy
   mutation;
5. policy, provenance, digest, revision, and stale-cache checks;
6. serial and MPI current-state consensus, including a distributed subfloor
   rejection that cannot publish a partial success;
7. the declared 108-case fraction, side, orientation, and mesh-scale matrix,
   with every accepted case recording `0.25` and satisfying the direct risk
   gate.

Until those tests are frozen against one clean commit, executed from fresh
builds, and archived with finalized source and matrix hashes, this document is
an implementation contract rather than qualification evidence. Completion of
that prerequisite still does not by itself check WP-3, WP-7, or Q1.
