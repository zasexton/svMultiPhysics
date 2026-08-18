# WP-3/WP-7 certified aggregate-trace prerequisite, version 2

## Status and claim boundary

This document specifies the version-2 certified aggregate-trace prerequisite
for the generated-boundary velocity-Nitsche route. Its only accepted claim is
`joint_low_level_prerequisite`.

The matching matrix carries the authoritative lifecycle status. While that
status is `DRAFT_UNEXECUTED`, its implementation-source hashes are only an
observation, not an execution baseline, and full execution is disabled. The
runner permits execution only after the sources and runner are deliberately
frozen and the matrix is promoted to `FROZEN_BEFORE_EXECUTION`.

The method-wide coercivity lower bound remains deliberately `null`, with
uniform-bound status `UNFROZEN_NO_BOUND_INVENTED`. FSR-16, FSR-07, WP-3,
WP-7, and Q1 all remain `OPEN`. In particular, a positive certificate for the
supported finite-dimensional route is not a mesh-family theorem and does not
close any of those outcomes.

The factorized exact-dyadic hardening of the patch quotient becomes
qualification evidence only through a frozen, hash-matched source inventory
and a successful execution of every required group. A draft source snapshot
or an implementation result obtained outside that binding is not
qualification evidence. In every lifecycle state, the closure dispositions
above remain unchanged by this prerequisite alone.

The v2 matrix byte-locks these parent artifacts:

- v1 matrix
  `a75bbec8efe800f049375f190c07a121b3e365098da783b43ec1ba9df9610589`;
- v1 runner
  `353c49c10881fd13acececb80cdf000c70abf7937f10021a2816d04d90bb9181`;
- v1 method document
  `abc782ef828b3fd3996257f5544f85221d9c6d047b1cf730848ac93b695c6ead`.

Version 2 is additive. It does not alter, supersede, or retroactively expand
the frozen v1 claim. The parent hashes preserve the v1 bytes; replay of that
runner also requires its hash-matched shared execution base. The evolving
current shared base is intentionally rejected by v1 rather than being
silently reinterpreted as historical evidence.

## Production form and notation

Consider one generated active-boundary route \(i\) on physical boundary
marker \(\Gamma_i\). For a constant positive dynamic viscosity \(\mu\), let

\[
K(v,v)
  = \int_{\Omega_a} 2\mu\,\varepsilon(v):\varepsilon(v)
\]

be the active-volume viscous energy after applying the current closed affine
constraints. Let the normal viscous traction be

\[
t(v) = 2\mu\,\varepsilon(v)n .
\]

For a weak velocity-Dirichlet route, the production penalty is

\[
P_i(v,v)
  = \alpha_i \int_{\Gamma_i}\frac{\mu}{h_n}|v|^2 ,
\qquad
\alpha_i =
\begin{cases}
\gamma_i p^2, & \text{when polynomial-order scaling is enabled},\\
\gamma_i, & \text{otherwise}.
\end{cases}
\]

Here \(h_n\) is the same normal mesh scale used by the production Nitsche
form. The canonical FE helper requires the exact state-field and test-function
terminal roles with matching complete function-space signatures. It seals
\(\gamma_i\), that signature, the actual polynomial order, the scaling flag,
and the resulting \(\alpha_i\) in an opaque binding. Before sealing, it also
requires the prescribed value to have the same vector dimension as the
velocity field and walks its complete expression DAG to reject state, trial,
test, discrete-field, previous-solution, coupled boundary-reduction,
auxiliary, material-state, constitutive, integral, current-geometry, and
geometry-variation dependencies. Ordinary constant and spatial/time callback
data remain admissible. This prevents an apparently inhomogeneous value from
silently reintroducing a velocity or geometry tangent outside the certified
route.
Installation verifies that the binding's exact generated-interface
symmetric-gradient route anchor occurs once as an unscaled top-level additive
summand in the original residual, then
publishes the derived policy against that formulation record. Certification
consumes the sealed \(\alpha_i\); it neither chooses nor mutates the penalty
and does not infer a coefficient from an arbitrary assembled form.

For a symmetric route, the diagonal consistency contribution has the
potentially negative magnitude

\[
2\left|\int_{\Gamma_i} t(v)\cdot v\right|.
\]

For the unsymmetric Nitsche variant, the two diagonal consistency terms
cancel. The trace certificate remains useful as a revision-bound continuity
diagnostic, but no symmetric coercivity threshold or symmetric energy-ratio
claim is applied.

## Certified aggregate trace inequality

The certificate establishes, on the exact finite-dimensional tangent space
of a boundary-carrying support patch,

\[
\int_{\Gamma_i}
  \frac{h_n}{\mu}|2\mu\varepsilon(v)n|^2
\;\le\;
C_i
\int_{\operatorname{supp}_i}
  2\mu\,\varepsilon(v):\varepsilon(v).
\]

The present implementation removes the common positive factor \(2\mu\) from
the numerator and denominator Gram forms before exact certification. This
leaves the quotient unchanged and avoids viscosity-dependent overflow or
underflow. It does not extend the result to variable or constitutive
viscosity.

The supported envelope is intentionally narrow:

- reference-frame affine P1 product velocity fields;
- linear `Triangle3` and `Tetra4` cells;
- constant, positive, finite dynamic viscosity;
- a current free-surface cut context and generated source-value revision;
- closed affine constraints with a current layout revision;
- a current finalized, trace-eligible small-cut aggregation prolongation;
- exactly one face-connected active feature per certified aggregation patch;
- synthetic singleton patches for full-active boundary-carrying cells;
- at most 128 terminal tangent degrees of freedom per patch, together with
  fixed implementation caps on cells, patches, rules, quadrature data,
  serialized data, and modeled dense memory;
- at most 32 retained quotient coordinates after structural rigid modes are
  removed by the exact coordinate gauge.

Rootless aggregation, non-affine geometry, quadrilateral or hexahedral Q1,
higher order, variable viscosity, and current-frame certification are outside
this envelope and fail closed.

## Patch construction and factorized exact bound

The finalized aggregation report supplies canonical active cells, retained
volume-rule identities, patch support, slave rows, owners, and revision
metadata. Certification reconstructs the closed tangent rows from canonical
DOF owners and gathers each retained volume and generated boundary
contribution from its declared provider. All participating ranks perform the
same deterministic factorized exact certification.

For each boundary-carrying patch, the implementation retains:

- raw affine symmetric-strain Gram factors for the retained support energy;
- raw affine normal-traction Gram factors for the scaled boundary trace;
- positive quadrature weights as exact sums, with positive geometric scale
  factors such as \(h_n\) retained as exact products;
- the finalized affine prolongation and retained-coordinate gauge as one
  exact sparse raw-to-quotient tangent map;
- structural rigid-mode candidates appropriate to the spatial dimension.

Affine P1 gradients and tractions must be bitwise invariant at every
quadrature point of the corresponding contribution. Quadrature-point weights
are accumulated as positive terms, not as one rounded binary64 total. The
strain form retains its diagonal/off-diagonal row multipliers; the traction
form retains its separate integer multiplier and the \(h_n\) product factor.
Every primitive binary64 value is interpreted as an exact dyadic. A block's
`map_rows` are required to be strictly increasing, so duplicate and ordering
validation is deterministic and linear in the bounded input size.

A rigid-mode quotient is applied only after the raw factors annihilate the
structural rigid parameters exactly, the sparse tangent reproduces those
parameters exactly, and modular elimination proves full column rank. The
elimination also selects an exactly nonsingular set of coordinate anchors.
Deleting those coordinates produces the retained principal quotient without
a rounded orthogonal projection. Each transformed Gram coefficient remains
an exact sum of raw-factor/tangent-coefficient products until the integer
matrices are materialized; no formed binary64 tangent row or matrix is used by
the acceptance proof.

The structural rigid gauge is followed by an exact secondary common-kernel
check. Both materialized positive forms must be PSD, and the backend proves

\[
\operatorname{rank}(D_q+N_q)=\operatorname{rank}(D_q)
\]

with fraction-free integer congruence. For PSD forms this equality is exactly
equivalent to \(\ker D_q\subseteq\ker N_q\). A deterministic exact pivot set
then selects a full-rank principal coordinate gauge for the remaining finite
quotient. A rank increase rejects the certificate because the generalized
bound would be infinite. This secondary step covers exact zero-energy
directions left by binary64 affine prolongation without classifying a small
positive mode as zero.

A fixed-cap, fraction-free integer congruence proves that the factorized
final denominator is positive definite and the numerator is positive
semidefinite. For a finite nonnegative binary64 candidate \(q\), the same
oracle proves

\[
qD_q-N_q \succeq 0
\]

exactly. Since \(D_q\succ0\), that predicate is monotone in \(q\). An ordered
binary64 bit search returns a representable value that was itself proved;
failure of the maximum finite value rejects the certificate. A
zero-dimensional retained map is accepted as the vacuous SPD/PSD quotient
with coefficient zero, but only after the complete factor, sparse-map,
modeled-input, and worst-case-work preflight succeeds.

The factorized backend fails closed at fixed caps: 16,384 total Gram blocks;
262,144 factor rows and 1,048,576 positive weight terms per form; four positive
product factors per block; 1,048,576 sparse-map entries, raw factor
coefficients, and transform visits; 8,388,608 outer pairs; 64 MiB modeled
factor input; 2,000,000 worst-case exact arithmetic updates; 262,144 integer
bits; quotient dimension 32; and 64 binary64 search steps. Count and
modeled-byte caps are enforced before the corresponding caller-owned content
is scanned. No tolerance, diagonal shift, or small-positive-mode deletion
participates in acceptance.

Formed dense binary64 numerator and denominator matrices exist only for the
optional cyclic-Jacobi diagnostics on the same retained quotient. When that
solver resolves the spectrum, its padded value may raise the published
coefficient; monotonicity of the already-proved dyadic inequality makes that
larger finite value safe. When it cannot resolve a tiny positive mode, the
factorized exact-dyadic coefficient remains the sole authority and the
floating spectral fields remain unavailable. The patch value is therefore
not merely a computed floating largest eigenvalue.

Aggregation supports may overlap and aggregation patches do not partition the
active cells. Consequently, the global certificate is

\[
C_i =
\max_{T\ \mathrm{active}}
\operatorname{round}_{\uparrow}
\left(
  \sum_{\substack{q:\;T\in\operatorname{supp}(q)}} C_q
\right).
\]

This cellwise outward-rounded overlap sum is generally tighter than the
diagnostic product of maximum patch bound and maximum support overlap.
Every generated boundary rule is assigned exactly once to either a canonical
aggregation patch or a synthetic full-active patch.

The certificate records the physical boundary marker, generated active
marker, volume-interface marker, viscosity, communicator size, aggregation
digest, canonical certificate digest, revision stamps, patch counts, support
overlap, retained volume, boundary measure, and conservative bounds. The
canonical digest is scoped to the communicator and algebraic partition.
Rank-local revision numbers are intentionally not part of that digest.
The revised digest also binds the exact proof-input kind, factorized
materialization and sparse-map flags, Gram block/row and positive-weight-term
counts, transform and outer-pair work counts, modeled bytes, nonzero
factorized-input provenance digest, pre- and post-common-kernel dimensions,
the exact common-kernel proof/application flags, eliminated-coordinate count
and identities, retained ranks, failing-neighbor
availability and, when available, the adjacent failing binary64 value, the
directly proved coefficient, proof/search counts, and maximum integer bit
width.

## Symmetric penalty gate

Combining the trace inequality with Cauchy-Schwarz gives

\[
2\left|\int_{\Gamma_i}t(v)\cdot v\right|
\le
2\sqrt{\frac{C_i}{\alpha_i}}\,
  \sqrt{K(v,v)P_i(v,v)}.
\]

For all symmetric generated-boundary routes attached to the same production
operator, define the outward-rounded group risk

\[
R_{\mathrm{op}}
  = \operatorname{round}_{\uparrow}
    \sum_i \frac{C_i}{\alpha_i}.
\]

Cauchy-Schwarz over the routes and
\(2\sqrt{KP}\le K+P\) then give the finite-space ratio

\[
A_{\mathrm{op}}(v,v)
\ge
\left(1-\sqrt{R_{\mathrm{op}}}\right)
\left(K(v,v)+\sum_i P_i(v,v)\right).
\]

The production policy requires the strict condition
\(R_{\mathrm{op}}<1\). The stored lower ratio is
\(1-\sqrt{R_{\mathrm{op}}}\), rounded downward, and must remain finite and
strictly positive. Exact zero is preserved by the ratio and sum operations,
so a dry route stores \(R_{\mathrm{op}}=0\) and the exact lower ratio one
rather than a test-side replacement. If the gate fails, certificate refresh
throws collectively and publishes no partial certificate cache.

This is a finite-dimensional guarantee for the exact current state. It is not
the matrix field `method_coercivity_lower_bound`, which remains `null`, and it
is not a claim of a cut-, mesh-, element-, or regime-uniform constant.

For an unsymmetric route, the record retains \(C_i/\alpha_i\), sets the
grouped symmetric ratio to zero when there are no symmetric routes on that
operator, and leaves the symmetric energy-ratio value absent. An
unsymmetric route is therefore not rejected merely because
\(C_i/\alpha_i\ge1\).

## Production lifecycle and fail-closed assembly

A generated-boundary Nitsche trace policy is committed by the formulation
installer before system setup for each supported generated weak-velocity
route backed by small-cut aggregation. The policy binds the operator,
velocity field, physical boundary marker, volume-interface marker, generated
active marker, viscosity, configured penalty, actual polynomial order,
effective penalty, symmetry variant, source formulation-record index,
nonzero binding-metadata digest, and dense-dimension cap.

Registration fails before adding velocity or pressure fields unless every
generated weak-velocity route uses `CutVolume` with small-cut aggregation, a
native mesh, finite positive constant Newtonian viscosity, and the affine P1
product-H1 triangle/tetrahedron envelope certified here. The same read-only
preflight constructs the canonical form binding for every generated route and
rejects forbidden state, coupled-data, or variational-geometry dependence in
prescribed boundary values. A route-count mismatch also fails before any
system mutation. Synthetic mesh-access harnesses outside this native-mesh
aggregation envelope are therefore rejection tests, not production Nitsche
evidence.

Setup validates the policy table. If setup occurs before a cut context is
available, the policy remains registered and the certificate cache is empty.
After a current cut context and finalized aggregation exist, constraint-state
refresh eagerly certifies every policy. Context replacement and relevant
constraint or layout changes invalidate the cache and require recertification.

Assembly checks the operator's policy records before touching the requested
matrix or vector. It rejects an absent, incomplete, stale, mismatched, or
nonpositive symmetric record before output mutation. The validation binds:

- the current constraint revision;
- cut-context content revision;
- free-surface snapshot revision;
- generated source-value revision;
- affine-constraint layout revision;
- ownership of the current finalized aggregation report;
- aggregation and certificate digests;
- every policy marker and coefficient;
- the source formulation record and binding-metadata digest;
- the effective penalty and grouped symmetric ratio.

In MPI, policy count and policy signature must agree across the field
communicator. Certification and failure coordination are collective.
Canonical report and certificate digests must agree even when rank-local
revision values differ. No rank may publish a successful partial result while
another rank fails.

## Navier-slip separation

The certified route in this version is generated-boundary weak velocity
Dirichlet Nitsche. A physical Navier-slip or Robin contribution with
coefficient

\[
\frac{\mu}{\ell_{\mathrm{slip}}}
\]

is a physical boundary coefficient, not a Nitsche penalty multiplier. The
certificate does not rescale, replace, or qualify that coefficient.
Navier-slip closure remains outside this matrix.

## Version-2 evidence groups

The matching matrix defines four groups and 26 distinct tests.

### Exact dense and factorized dyadic quotient group

Sixteen serial math tests include the ten legacy formed-dense exact tests.
Those require direct proof of a diagonal equality bound and an exact-zero
numerator bound without an invented failing neighbor, retention of a one-ULP
positive mode, exact three-by-three Bareiss division, and a symmetric pivot
around a zero leading coordinate. They also require fail-closed rejection of
a late indefinite pivot, a semidefinite denominator, an indefinite numerator,
an upper bound larger than every finite binary64 value, a quotient above the
fixed dimension cap, and malformed matrix input.

The six factorized tests additionally require:

1. certification from a raw rank-one Gram factor when entrywise binary64
   matrix formation would destroy exact positive semidefiniteness;
2. preservation of a sparse tangent coefficient sum that would otherwise
   round away a half-ULP term;
3. exact positive sum/product scales whose formed binary64 values would
   underflow or overflow;
4. strict-SPD rejection of a singular denominator, exact secondary quotient
   of a compatible common kernel, rejection when the numerator acts on that
   kernel, and fail-closed rejection of zero scale, nonfinite factor,
   non-strict `map_rows`, and genuinely nonmonotone sparse offsets;
5. multi-term positive weights plus numerator and denominator integer,
   product, and row-multiplier paths with linear, not squared, scaling; and
6. a vacuous zero-dimensional quotient after full bounded preflight, with
   factorized proof flags, counts, work metrics, and a nonzero provenance
   digest.

### Serial certificate group

Eight serial assembly tests require:

1. helper rejection for non-state velocity expressions, non-test test
   expressions, mismatched function spaces, an invalid Nitsche variant,
   prescribed values with transitive state/trial/test, coupled auxiliary, or
   geometry-variation dependencies, and prescribed vectors with the wrong
   value dimension; and install-time rejection, before operator or
   analysis-metadata mutation, when an opaque canonical route binding is
   absent from the residual, its exact anchor is inserted more than once, or
   the anchor is scaled or nested below a non-additive operator;
2. a full-active unit triangle with analytic quotient \(4\), a conservative
   outward bound, eager symmetric policy validation, deterministic repetition,
   physical-marker rejection, and stale-cache rejection before sentinel
   matrix mutation;
3. an actual rooted cut-square prolongation with retained volume \(79/128\),
   generated boundary measure \(1/8\), three rigid null modes, and analytic
   quotient \(32/79\);
4. fail-closed rejection of a trace-eligible finalized aggregation report
   whose only patch is rootless, because rootless support is outside the
   certified envelope;
5. fail-closed rejection when generated domains were imported without an
   authoritative free-surface snapshot: the aggregation rows remain
   structurally trace eligible, but the report lacks the snapshot revision
   required by certification;
6. fail-closed rejection of a scalar H1 field outside the required product
   velocity space;
7. rejection of an insufficient symmetric configured penalty with no
   partially published record;
8. retention of an unsymmetric diagnostic certificate without a symmetric
   threshold or energy-ratio value.

### Exact two-rank MPI group

One test must run at exactly two ranks. Its rooted aggregate crosses rank
ownership, closes rank-one slave rows onto rank-zero masters, and has analytic
quotient \(32/79\). Both ranks must agree on nonzero canonical report and
certificate digests and on report shape. Rank-local context revisions are
expected to differ. Collective certification and assembly preflight must both
succeed without skip.

### 108-case diagnostic group

The physics diagnostic is deliberately named with a `DISABLED_` prefix.
The qualification runner must enable that test explicitly and must not
interpret its source-level disabled name as an accepted disabled result.

It covers:

- wall fractions `0`, `1e-8`, `1e-6`, `1e-4`, `1e-2`, `0.1`, `0.25`,
  `0.49`, and `1`;
- axis-aligned and oblique orientations;
- physical scales `1/2`, `1/3`, and `1/4`;
- negative and positive active sides.

The product contains 108 cases: 96 wet and 12 dry. The configured penalty
multiplier is 12. Every case requires one nonzero deterministic,
revision-matched certificate, a nonzero aggregation digest, a nonzero
form-binding metadata digest with a matching source formulation record, a
finite nonnegative upper bound strictly below 12, a grouped symmetric ratio
below one, and a positive finite-space energy ratio. Every patch must expose
internally consistent factorized exact common-kernel metadata, and the full
sweep must apply the exact secondary quotient on at least one production
patch.

For a dry case, the exact contract is zero generated boundary rules, zero
patches, zero trace upper bound, and energy ratio one. For a wet case, the
boundary-rule count, patch count, and upper bound are positive. The sampled
generalized eigenvalue from the v1-style finite spectrum must be no smaller
than the analytic finite-space ratio after subtracting the eigensolver
tolerance, up to the comparison tolerance `1e-11`.

The diagnostic emits 108 structured case records and one structured summary.
Its eight GTest properties require:

- case count equal to `108`;
- maximum upper bound less than `12`;
- minimum finite-sample lower ratio greater than `0`;
- minimum sampled eigenvalue gap at least `-1e-11`;
- exact common-kernel quotient patch count greater than `0`;
- method coercivity lower bound equal to the string `null`;
- uniform-bound status equal to `UNFROZEN_NO_BOUND_INVENTED`;
- accepted claim equal to `joint_low_level_prerequisite`.

Those values become evidence only when evaluated by a successful frozen
execution. Their presence in a draft matrix or this derivation does not report
them as executed evidence.

## Evidence still required

The following remain outside the v2 prerequisite:

- quadrilateral and hexahedral Q1 certification;
- higher-order and non-affine geometry;
- variable or constitutive viscosity;
- traction, pressure flux, outflow, physical Robin, Navier slip, and all
  other supported sharp-boundary routes;
- mixed velocity-pressure and inf-sup stability;
- manufactured velocity, pressure, divergence, and convergence rates;
- transient, advection-dominated, and broader viscous regimes;
- four-or-more ranks, repartition sweeps, and rank-count invariance;
- connected/disconnected topology policies and continuous node crossing;
- production preconditioner robustness and iteration spread;
- exact-current, checksum-bound execution of the existing native-mesh
  production Nitsche wet/dry sweep and assembly preflight, plus repartitioned
  MPI production-route coverage;
- a uniform method coercivity lower bound.

Until those exits are satisfied under their owning qualifications, FSR-16,
FSR-07, WP-3, WP-7, and Q1 remain open.

## Validation and promotion

The v2 matrix may be checked structurally while its status is
`DRAFT_UNEXECUTED`; a full runner must reject execution in that state. Hashes
recorded in a draft inventory are observations of a moving source snapshot,
not a frozen baseline or qualification evidence. They become authoritative
only when every inventory entry is locked to the recorded implementation
commit, the reciprocal matrix/runner binding is complete, and the matrix
status is `FROZEN_BEFORE_EXECUTION`.

Earlier synthetic single-tetra and structured-channel Nitsche success cases
did not provide the native mesh, aggregation state, and authoritative snapshot
required by this method. They now verify fail-closed registration instead.
Archived results that name their former scaling checks remain historical and
checksum-bound; they are not current-tree evidence. Replacement Nitsche
qualification must use the native production fixture and its rooted aggregate
certificate path.

Before promotion, every source in the inventory must be stable and
intentionally SHA-256 locked. The stable implementation inventory is first
committed; that commit remains an ancestor of the later clean qualification
checkout, and every inventoried blob must match both the recorded commit and
the checkout. The v2 runner itself and normalized matrix digest must then be
frozen, discovery must return all 16 named `DenseLinearAlgebra.Exact*` tests,
and the disabled diagnostic route must be explicitly enabled. Only then may
the matrix be changed to `FROZEN_BEFORE_EXECUTION` and used to collect
checksum-bound evidence.

Execution also remains fail closed on resources. The runner requires
`MemAvailable >= 10240 MiB` before it starts, retains `4096 MiB` host-memory
and filesystem-free-space floors while every build or test subprocess
session is live, forces clean builds to one job, caps each spawned build
process at `1536 MiB` of address space, keeps each focused test process at
its `1024 MiB` group ceiling, applies per-file output limits, and pins common
threaded numerical runtimes to one thread. Each of the four focused test
groups has a `64 MiB` artifact envelope. Build-target inventory decoding is
limited to `8 MiB`. Linked-library provenance runs in the same monitored
process-session path with a `60 s`, `256 MiB`, and `4 MiB` time, memory, and
output envelope and fails closed on launch, monitoring, process, or parse
failure. Resource-floor, monitoring, cleanup, process-coverage, or bounded
post-processing failures are method failures rather than qualification
evidence.
