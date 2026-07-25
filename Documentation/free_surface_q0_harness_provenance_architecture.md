# Q0 harness and provenance architecture record

Status: executable control prerequisites available; Q0 remains open.

Scope: the Q0 harness, provenance, and configuration-containment campaign,
including negative cases, defined by
`free_surface_boundary_unfitted_audit_20260720.md`. This record does not alter
that audit or authorize its Q0 checkbox.

## Closure decision

Q0 cannot close from the current executable evidence.

The committed WP-0 matrix supplies 24 deterministic configuration tests and
WP-0 has a separate archived qualification record. Those exact 24 names now
also have a dedicated `Physics_FreeSurfaceConfiguration_WP0` CTest with a
frozen timeout, processor count, and labels. The tracked push/pull-request
workflow routes both its Ubuntu and macOS jobs through actions whose unit-test
steps invoke unfiltered `ctest --verbose`. This is executable CI registration,
not evidence that a hosted runner executed the same revision. The central
campaign validator defines a strict artifact layout and rejects malformed
layouts, source-tree and child-program hash mismatches, missing metrics,
incomplete references, checksum drift, and promotion based only on
prerequisite evidence. Those controls are valuable and executable.

They are not a Q0 campaign execution. The central registry still declares the
Q0 campaign state and its campaign-evidence child unresolved. No artifact at
one source revision contains the complete accepted-step history and
provenance required by Q0. Evidence from WP-1, WP-2, and WP-10 demonstrates
useful runner patterns, but artifacts scoped to other work packages cannot be
reclassified as Q0 accepted-step evidence.

## Creditable prerequisite evidence

The frozen prerequisite harness checks:

- all 20 exact committed source definitions and required semantics for the
  audit criteria,
  WP-0, WP-1, WP-2, and WP-10 matrices and runners, and the central campaign
  registry and validator, plus the executable WP-10 one-phase scope guard.
  The inventory also pins the Physics CTest registration, the tracked
  push/pull-request test workflow, and the Ubuntu and macOS composite actions.
  The WP-10 definition transitively pins its schema-2 physical-model artifact,
  canonical XML parser boundary, and direct-map production boundary. It
  freezes seven dedicated C++ test names and records the binaries when its
  groups execute; Q0 does not reclassify those tests as Q0 campaign execution;
- all 24 WP-0 configuration-containment test names, including negative cases,
  in an executable Physics test binary;
- all 44 committed Python control tests covering exact result inventories,
  duplicate tests, explicit supplemental sources, MPI output copies,
  source-boundary containment, frozen XML/JSON/mapping one-phase scope-guard
  acceptance and rejection, early closure rejection, artifact layout,
  source-tree and child hashes, references, expected metrics, and checksums;
- refusal to promote a campaign from prerequisite-only evidence; and
- the machine-readable unresolved Q0 state.

The exact matrix SHA-256 is
`74c6e2a01178b5d2946edf1e9572a000d6e32117bbb7bf30de9a5f1514274108`.
The runner accepts only the canonical, non-symbolic-link matrix path and
rejects any byte mutation before discovery or execution. Every path component
of every frozen source definition is checked for symbolic links before path
resolution.

Execution passes the 44 frozen pytest node identifiers directly to pytest.
The resulting JUnit record must contain exactly one completed, passing
testcase for every identifier: missing, unexpected, duplicate, skipped,
failed, or errored testcases and inconsistent suite totals all fail the
prerequisite gate.

Before either test group starts, the runner freezes the test-binary bytes, the
identity and complete bytes of its required CMake cache, and the normalized
dynamic-library manifest. It requires the Physics test project, Physics
source directory, enabled Physics tests, and matching build directory. The
same records are rebuilt after both groups and must be byte-for-byte
identical; discovery must also have hashed the same binary.

## Exact remaining closure gaps

Eight exits remain:

1. The complete WP-0 invalid-input matrix is registered as one exact CTest in
   the tracked push/pull-request Ubuntu and macOS CI chain, but no hosted CI
   execution artifact from this same source revision has been archived.
2. Accepted step, time, nonlinear stage, and state/geometry/map revision
   histories are not archived under the campaign contract.
3. Raw and post-maintenance global, component, film, sheet, rim, and satellite
   inventories are incomplete.
4. The complete kinetic, gravitational, surface, wall, gas-when-applicable,
   dissipation, external, stabilization, extension, pruning, and maintenance
   energy/work account remains blocked by WP-8.
5. Per-accepted-step extension, cut geometry, solver, rejection, time-step
   reduction, fallback, rollback, wall-clock, and peak-memory telemetry is
   incomplete.
6. Compiler, libraries, options, machine, mesh/reference checksums,
   dimensional parameters, nondimensional groups, and thresholds have not
   been archived together for one Q0 execution.
7. The same-revision Q0 campaign-evidence child is not registered.
8. No complete, checksummed Q0 campaign artifact has been archived.

These are completion requirements, not prospective test names. A new test
name or a passing control harness cannot substitute for the missing
accepted-step records.

## Claim boundary

The only accepted claim is `q0_control_prerequisite`. The runner rejects every
claim ending in `_closure` and explicitly rejects Q0 qualification, Q0
campaign-pass, and physical-gate-ready requests before validating binary or
output arguments. Validation and list-only modes write no artifacts.

A successful execution artifact is labeled
`q0_control_prerequisite_nonclosure`. It records:

- matrix, runner, and source-definition hashes;
- source commit, tree, tracked diff, untracked-source inventory, status, and
  combined dirty-tree hash;
- test-binary, CMake-cache, and selected compiler/option hashes;
- canonical dynamic-library records containing the requested name, resolved
  regular-file path, file size, and hash of the actual library bytes, plus a
  sorted virtual-dependency inventory and canonical manifest hash;
- machine, requested rank/thread topology, enforced resource limits,
  discovery, and test outcomes; and
- the explicit open-exit and nonclosure disposition.

Dynamic-loader addresses are excluded from provenance. Library output is
sorted before its manifest hash is computed, and any missing, unresolvable,
non-regular, or duplicate library record fails closed.

The matrix deliberately does not hash its runner, its self-tests, or this
record, which would make their maintenance circular. Their non-circular trust
boundary is the recorded source commit and tree. A dirty-tree execution also
records every untracked source hash, while the runner hash is recorded
directly and checked again after execution. These controls establish
prerequisite-artifact provenance only; they do not make the artifact
promotable Q0 campaign evidence.

That artifact remains outside the central Q0 campaign layout and is incapable
of promotion.

## Promotion rule

Q0 can close only after the central registry registers a Q0 campaign-evidence
program, all eight exits above are removed with same-revision evidence, and a
complete immutable artifact passes the central validator with a requested Q0
claim. Until then:

- `q0_closed` is false;
- `audit_q0_checkbox_may_be_checked` is false; and
- every later physical campaign must treat Q0 as unresolved.
