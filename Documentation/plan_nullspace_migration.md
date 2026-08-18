# Migration Plan: NullspaceAnalyzer → Analysis Pipeline

**Date**: 2026-03-20
**Branch**: `issue-449-modern-mesh-core`
**Goal**: Remove the redundant NullspaceAnalyzer call from FormsInstaller, completing
the gauge candidate population migration to the ContributionDescriptor/NullspaceHint path.

---

## Background

There are three parallel nullspace detection paths:

| Path | When | Mechanism | Production? |
|------|------|-----------|-------------|
| 1. Legacy | Definition time | NullspaceAnalyzer → GaugeCandidate | Yes (FormsInstaller) |
| 2. Contribution | Setup time | NullspaceHint → GaugeCandidate | Yes (SystemSetup) |
| 3. Analysis | On-demand | KernelAnalyzer → PropertyClaim → GaugeAdapter | No (test-only) |

Path 2 was added during Phase 20 cleanup (SystemSetup.cpp lines 1318-1344). It converts
`NullspaceHint` entries from `ContributionDescriptor` objects into `GaugeCandidate` objects
and adds them to the `GaugeRegistry`. The `FormContributionLowerer` (called by FormsInstaller
at line 1126) populates these hints using **identical classification logic** to NullspaceAnalyzer.

Both NullspaceAnalyzer and FormContributionLowerer:
- Delegate to `FormStructureAnalyzer` for DAG walking
- Use the same filter conditions: `only_through_annihilating_ops`, `!has_absolute_value`, `!has_time_derivative`
- Classify into the same families: `ScalarConstant`, `ComponentwiseConstant`, `KernelOfSymGrad`
- Map `has_stabilization` → Medium confidence, otherwise High

The only structural difference: NullspaceAnalyzer walks the whole residual;
FormContributionLowerer walks per-block sub-expressions (but only emits hints for
diagonal blocks, which is mathematically equivalent — the diagonal block captures
the field's self-coupling structure that determines nullspace properties).

Path 1 is therefore fully redundant. GaugeRegistry candidate deduplication already
prevents double-counting when both paths run.

---

## Steps

### Step 1 — Remove NullspaceAnalyzer call from FormsInstaller.cpp

- [x] Delete the `populateGaugeRegistry` lambda and its call (lines 1017-1033)
- [x] Remove `#include "Forms/NullspaceAnalyzer.h"` from FormsInstaller.cpp
- [x] The contributions path (FormContributionLowerer → NullspaceHint → SystemSetup conversion) already handles everything

### Step 2 — Verify equivalence via existing tests

- [x] Run `test_fe_analysis` (241 tests) — includes roundtrip test `Phase9.FullPipelineGaugeRoundtrip`
- [x] Run `test_fe_systems` (GaugeIntegration tests) — verifies gauge enforcement works
- [x] Run `test_fe_forms` — NullspaceAnalyzer unit tests still pass as standalone
- [x] Run all 6 integration test cases (Channel2D, Channel2D_Simple, iliac_artery, pipe_RCR_3d, pipe_simple, vortex_shedding)

### Step 3 — Update test_NullspaceAnalyzer.cpp tests

- [x] These 15 tests test the NullspaceAnalyzer class directly — they remain valid as standalone unit tests
- [x] No changes needed; the class still exists as a utility

### Step 4 — Mark NullspaceAnalyzer as utility-only

- [x] Add comment to NullspaceAnalyzer.h noting it is no longer called in production code
- [x] Keep it available for direct use in tests and tools
- [x] Do NOT delete — it's a useful standalone analysis utility

### Step 5 — Clean up GaugeAdapter

- [x] `GaugeAdapter::populateRegistryFromReport()` is not needed in production — the NullspaceHint→GaugeCandidate path handles everything
- [x] Add comment to GaugeAdapter.h noting that Path 2 (NullspaceHint) is the production path
- [x] Keep GaugeAdapter for test roundtrip validation; no production integration needed

---

### Implementation Note — Watermark Fix

During implementation, a watermark bug was discovered: `contributions_def_count_`
was only snapshotted during `setup()`, but `invalidateSetup()` (called by
`addField()`, `addOperator()`, and within `installFormulation()`) would truncate
contributions to this watermark (initially 0), wiping definition-time contributions.

Fix: `FESystem::addContribution()` now updates `contributions_def_count_` when
`!is_setup_`, ensuring definition-time contributions are always preserved across
`invalidateSetup()` calls.

---

## Risk Assessment

**Low risk.** The NullspaceHint path is already running in production alongside
NullspaceAnalyzer. Removing the NullspaceAnalyzer call just removes the redundant
first registration — the setup-time registration covers it.

**Edge case (multi-field):** For NS-VMS, FormContributionLowerer analyzes per-block
expressions while NullspaceAnalyzer analyzes the whole residual. Nullspace hints are
only emitted from diagonal blocks (`is_diagonal` guard), which is equivalent — the
diagonal block captures the self-coupling structure that determines nullspace.

## What We Do NOT Need

- Do NOT integrate `GaugeAdapter::populateRegistryFromReport()` into production — Path 2 is simpler and already works
- Do NOT move gauge resolution timing — it already happens at setup time after NullspaceHint conversion
- Do NOT delete NullspaceAnalyzer class — useful standalone utility
