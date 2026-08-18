# AuxiliaryState Ragged Layout Follow-On Plan

**Date**: 2026-04-22
**Status**: Complete; bounded follow-on to fixed-stride scope completion
**Depends on**: `Documentation/plan_auxiliary_state_scope_completion.md`

## Goal

Implement deployment-path ragged auxiliary blocks without weakening the fixed-stride
contracts for `Global`, `Node`, `Cell`, `QuadraturePoint`, and `Region`.
Ragged means each materialized entity has a variable number of auxiliary
components, represented by canonical per-entity offsets.

## Scope

- Add scope-correct ragged indexing and metadata for `Node`, `Cell`, and
  `QuadraturePoint`.
- Preserve stable original entity ids for restricted deployments.
- Define restart/remap metadata for ragged restricted subsets.
- Support partitioned local ODE/DAE runtime for ragged blocks when every local
  entity slice can be advanced independently.
- Keep monolithic support limited to cases with a proven assembly strategy:
  local condensation for independent `Cell`/`QuadraturePoint` slices, or
  explicit sparse/owner-backed rows for `Node` only after row ownership and
  block descriptors are available.

## Non-Goals

- Do not replace FE fields with ragged auxiliary storage.
- Do not add face-quadrature QP scope; `QuadraturePoint` remains cell-volume
  quadrature only.
- Do not support cross-entity ragged coupling in `AuxiliaryModel`; those
  formulations must use `AuxiliaryOperator`.
- Do not route ragged full rows into FSILS without explicit row-owner metadata,
  compatible distributed sparsity, and zero dropped-entry tests.
- Do not implement monolithic semismooth/complementarity handling in this plan.

## Design Decisions

- **Canonical offsets**: every ragged block stores `component_offsets` of size
  `entity_count + 1`; entity `e` owns components
  `[component_offsets[e], component_offsets[e + 1])`.
- **Stable entity ids**: restricted ragged blocks use the same stable original
  entity-id mapping as fixed-stride blocks.  Runtime entity-local inputs are
  addressed by original entity id, not by materialized local index.
- **QP identity**: QP ragged metadata stores covered cell ids, `qpOffsets`, and
  ragged component offsets over the flattened covered QP entity list.
- **Node ownership**: ragged node blocks preserve owned nodes before ghost nodes.
  Component offsets are local to that owned-plus-ghost entity ordering.
- **Cell/QP monolithic default**: independent ragged `Cell` and
  `QuadraturePoint` monolithic blocks use local condensation only.  Coupled or
  nonlocal ragged models are rejected from `AuxiliaryModel`.
- **Restart compatibility**: restart schemas must validate scope name,
  deployment region identity, entity ids, QP covered-cell ids, `qpOffsets`, and
  ragged component offsets before accepting state data.

## Implementation Checklist

- [x] Add ragged-aware indexing metadata to `AuxiliaryBlockIndexing` instead of
  using the current generic cell-shaped fallback in
  `AuxiliaryStateManager::registerBlockRagged()`.
- [x] Extend `AuxiliaryStateManager::registerBlockRagged()` to dispatch to
  scope-correct ragged indexing for `Node`, `Cell`, `QuadraturePoint`, and
  `Region` where supported.
- [x] Add ragged entity metadata to `AuxiliaryEntityRemapMetadata`, including
  `component_offsets` and QP `qpOffsets` metadata when applicable.
- [x] Add deployment-path API validation for `.layoutMode(Ragged)` that requires
  an explicit component-offset provider or a model-provided entity-size callback.
- [x] Wire `FESystem` deployment-region expansion into ragged registration for
  restricted `Node`, `Cell`, and `QuadraturePoint` blocks.
- [x] Update gather/scatter helpers so partitioned runtime slices ragged
  per-entity state, history, rates, residuals, and event contexts correctly.
- [x] Extend consistent initialization and failure-policy handling to ragged
  partitioned mixed ODE/algebraic blocks.
- [x] Add Cell/QP ragged local-condensation eligibility checks and reject
  cross-entity auxiliary-output coupling before assembly.
- [x] Add Node ragged monolithic row ownership only if each ragged row can be
  represented with explicit backend owner metadata and compatible sparsity.
- [x] Keep FSILS insertion owned-row authoritative and fail tests on any
  dropped entries.

## Tests

- [x] `AuxiliaryStateIndexing`: ragged `Node`, `Cell`, `QuadraturePoint`, and
  `Region` indexing reports entity count, owned count, offsets, and storage size
  correctly.
- [x] `AuxiliaryStateManager`: ragged registration is scope-correct and no
  longer falls back to generic cell indexing for non-cell scopes.
- [x] Restart/remap: restricted ragged `Node`, `Cell`, and `QuadraturePoint`
  schemas reject changed entity ids, QP covered-cell ids, `qpOffsets`, or
  component offsets.
- [x] Runtime: ragged partitioned fixed-dimension ODE state/history/residual
  slicing works for `Node`, `Cell`, and `QuadraturePoint`, and variable-width
  ragged entities are rejected before stepper scratch buffers are used.
- [x] Runtime: ragged partitioned mixed DAE initialization and failure-policy
  reject/restore behavior work for `Node`, `Cell`, and `QuadraturePoint`.
- [x] MPI: ragged node owned/ghost sync preserves owned-row authority and
  refreshes ghost components after owner updates.
- [x] Monolithic Cell/QP: independent ragged local condensation matches a dense
  serial reference and produces zero dropped FSILS entries in MPI.
- [x] Negative tests: ragged cross-entity coupling, unsupported FE-backed
  entity-local bindings, and incompatible QP layouts fail with explicit
  diagnostics.

## Exit Criteria

- Deployment-path ragged blocks no longer fail solely because
  `layoutMode(Ragged)` was requested.
- Every supported ragged scope has documented entity identity, component-offset
  semantics, restart metadata, and runtime lifecycle coverage.
- Unsupported ragged monolithic cases fail at setup with diagnostics that name
  the missing ownership, condensation, or coupling contract.
- Serial and MPI tests prove zero off-owner writes and zero dropped FSILS
  entries for any ragged path that reaches FSILS.

## Progress Log

- [x] 2026-04-22: Added ragged-aware `AuxiliaryBlockIndexing` factories and metadata for `Node`, `Cell`, `QuadraturePoint`, and `Region`; updated `AuxiliaryStateManager::registerBlockRagged()` to preserve scope-specific indexing instead of falling back to Cell; added QP ragged registration with covered-cell `qpOffsets`; and validated component offsets in restart schemas.
- [x] 2026-04-22: Added deployment-path ragged size contracts with preferred `raggedEntitySize()` providers and advanced `raggedComponentOffsets()` offsets; validation now requires exactly one contract for `.layoutMode(Ragged)`, rejects ambiguous fixed-stride/ragged metadata, and carries the accepted contract into `FESystem` diagnostics while runtime materialization remains explicitly blocked.
- [x] 2026-04-22: Wired `FESystem` ragged deployment registration for resolved `Node`, `Cell`, `QuadraturePoint`, and `Region` scopes by converting provider/explicit contracts into canonical component offsets, registering through scope-correct ragged manager APIs, and preserving component offsets in remap/restart metadata. Monolithic ragged deployment remains explicitly blocked until the planned owner-backed/condensed assembly contracts are complete.
- [x] 2026-04-22: Added exact-width ragged gather/scatter/runtime validation for partitioned advancement, including per-entity work, committed, and history slices before residual/rate/event contexts are constructed. Uniform ragged `Node`, `Cell`, and `QuadraturePoint` blocks now advance through the fixed-dimension runtime, while variable-width entities fail with an explicit model-dimension diagnostic instead of reaching stepper scratch buffers.
- [x] 2026-04-22: Closed the ragged partitioned mixed DAE lifecycle for fixed-dimension entity slices. Consistent initialization now has regression coverage proving ragged `Node`, `Cell`, and `QuadraturePoint` committed slices are initialized onto the algebraic manifold before advance; non-rejecting local failure restores each ragged work slice from initialized committed state; and rejecting failure throws for each supported ragged runtime scope.
- [x] 2026-04-22: Added restricted ragged restart/remap validation coverage for `Node`, `Cell`, and `QuadraturePoint` scopes. Tests now prove restart schemas preserve stable entity ids, canonical component offsets, and QP covered-cell/`qpOffsets` identity, and reject mutated payload metadata before accepting restart data.
- [x] 2026-04-22: Added ragged negative coverage and setup-time diagnostics for entity-local auxiliary-output coupling, scope-mismatched FE-backed entity-local providers, and incompatible `QuadraturePoint` `qpOffsets`, so unsupported ragged paths fail during layout finalization before runtime or assembly.
- [x] 2026-04-22: Enabled ragged monolithic `Cell`/`QuadraturePoint` deployment only through the independent local-condensation contract, kept other ragged monolithic scopes blocked until row-owner/sparsity proofs exist, rejected variable-width Cell/QP slices against the fixed-dimension `AuxiliaryModel` contract, and added setup-time tests for uniform Cell/QP acceptance plus variable-width and auxiliary-output coupling rejection.
- [x] 2026-04-22: Enabled ragged monolithic `Node` deployment only through the owner-backed local-condensation path. The contract now requires setup-time backend row-owner metadata, derives one backend owner for every ragged component row from a C0 nodal FE row, rejects variable-width node slices against the fixed-dimension `AuxiliaryModel`, and has serial plus 2-rank MPI coverage for missing owner metadata, accepted owner-backed layout, and variable-width rejection.
- [x] 2026-04-22: Made FSILS matrix insertion counters authoritative for owned-row assembly by routing direct block insertion through resolved slots and counting unresolved valid owned-row slots as dropped entries. Added a 2-rank backend regression proving off-owner writes and missing owned-row structural entries are collectively visible to tests, while existing ragged/local-condensed MPI assembly paths still report zero off-owner writes and zero dropped entries.
- [x] 2026-04-22: Added 2-rank MPI coverage for ragged restricted `Node` owned/ghost synchronization with unequal per-entity component widths on owned and ghost slices. The regression proves restart metadata carries ragged component offsets, explicit sync refreshes ghost component slices from their owner rank, and commit copies only the owned ragged prefix before repopulating stale ghost data from owner updates.
- [x] 2026-04-22: Added 2-rank MPI dense-reference parity coverage for ragged monolithic `Cell` and `QuadraturePoint` local condensation. The tests materialize uniform ragged widths equal to the mixed `AuxiliaryModel` dimension, prove the effective locally condensed FSILS matrix/residual matches the dense reference, and assert zero off-owner writes plus zero dropped FSILS entries.
