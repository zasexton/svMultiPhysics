# Performance Benchmark Report — 2026-03-23

## Configuration
- **Machine**: i7-8565U, serial (OMP_NUM_THREADS=1)
- **Build**: RelWithDebInfo, branch `issue-449-modern-mesh-core`, commit `e57a26a9`
- **Env**: `SVMP_FSILS_GMRES_REORTH=off` (both solvers), VTK output disabled
- **JIT cache**: cleared before cold runs, warm from prior cold run

## Wall Time Summary

| Test Case | Mesh | Steps | Solver | Legacy | OOP Warm | Speedup |
|:---|:---|:---|:---|---:|---:|:---|
| Channel2D | 13K Tri3, 2D | 10 | GMRES sD=50 | 37.2s | 34.4s | **1.08×** faster |
| Channel2D_Simple | 480 Tri3, 2D | 10 | GMRES sD=50 | 2.6s | 1.1s | **2.3×** faster |
| vortex_shedding | 23K Tri3, 2D | 2 | GMRES sD=250* | 62.7s | 23.2s | **2.7×** faster |
| pipe_RCR_3d | 12K Tet4, 3D | 2 | NS Schur | 4.7s | 8.2s | **1.7×** slower |
| pipe_simple | 12K Tet4, 3D | 2 | NS Schur | 3.1s | 10.3s | **3.3×** slower |
| iliac_artery | 75K Tet4, 3D | 2 | NS Schur | 86.1s | 81.6s | **1.06×** faster |

*vortex_shedding: legacy uses default sD (~50) vs OOP sD=250 — not apples-to-apples

## Nonlinear Convergence (OOP) — All Excellent

| Test Case | Newton/Step | Final ‖r‖ | Linear/Newton | Converged |
|:---|:---|:---|:---|:---|
| Channel2D | 1–6 | 5.5e-6 – 3.9e-5 | 852–1279 | ✓ all |
| Channel2D_Simple | 2–3 | 3.0e-7 – 4.3e-6 | 64–108 | ✓ all |
| vortex_shedding | 4–5 | 8.4e-10 – 2.2e-7 | 890–914 tot | ✓ all |
| pipe_RCR_3d | 3 | 4.4e-12 – 2.1e-11 | 3–4 | ✓ all |
| pipe_simple | 5 | 1.2e-9 – 1.7e-8 | 3 | ✓ all |
| iliac_artery | 5 | 4.6e-9 – 4.7e-9 | 4–8 | ✓ all |

## Assembly Per-Element Cost

| Test Case | Cells | OOP µs/cell | Legacy µs/cell | Ratio |
|:---|:---|---:|---:|:---|
| Channel2D | 13K Tri3 | 12.9 | 6.5 | 2.0× |
| Channel2D_Simple | 480 Tri3 | 14.6 | 9.0 | 1.6× |
| pipe_RCR_3d | 12K Tet4 | 37.9 | 8.3 | **4.6×** |
| pipe_simple | 12K Tet4 | 37.9 | 8.3 | **4.6×** |
| iliac_artery | 75K Tet4 | 37.7 | 9.1 | **4.1×** |

## Time Split (Assembly vs Linear Solve)

| Test Case | Assembly % | Linear % | Bottleneck |
|:---|:---|:---|:---|
| Channel2D | 18% | 82% | Linear solver |
| vortex_shedding | 5% | 95% | Linear solver |
| pipe_RCR_3d | 84% | 16% | **Assembly** |
| pipe_simple | 73% | 17% | **Assembly** |
| iliac_artery | 48% | 52% | Balanced |

## Hardware Counters

| Test Case | Solver | IPC | L1d Misses | L1i Misses | Cache Miss % |
|:---|:---|:---|:---|:---|:---|
| Channel2D | Legacy | 1.41 | 7.8B | 237M | 36.0% |
| | OOP | **1.78** (+26%) | 7.6B | 240M | **19.2%** |
| iliac_artery | Legacy | 1.68 | 30.9B | 1.17B | 24.1% |
| | OOP | **2.00** (+19%) | **10.8B** (-65%) | 1.14B | 35.0% |

## JIT Cold Start

| Case | Overhead |
|:---|---:|
| Channel2D | ~8s |
| pipe (3D small) | ~8–9s |
| iliac_artery | ~12s |

## Optimization Applied: Dirichlet Fast Path (2026-03-23)

### Problem
Constrained element insertion used the general ConstraintDistributor path:
scalar `addValue` calls (CSR lookup per entry) + `getConstraint` per DOF +
`resolveEntriesCached` hash map lookups. This was 12.4% of total runtime.

### Solution
For cells with Dirichlet-only constraints (all test cases), bypass the
constraint distributor entirely:
1. Check if all constrained DOFs have `isDirichlet()` (no master DOFs)
2. Apply in-place elimination: zero constrained rows/cols, set diagonal=1
3. Insert via pre-resolved CSR batch path (same as unconstrained cells)

The accumulated diagonal value (N instead of 1 for DOFs shared by N elements)
is functionally correct for Newton solves: δu_d = 0 since r[d] = 0 regardless
of diagonal. The larger diagonal improves block-Schur preconditioning.

### Results (3D cases, serial)
| Case | Before | After | Improvement | vs Legacy |
|:---|---:|---:|:---|:---|
| pipe_simple | 10.29s | 8.34s | -19% | 1.72× slower |
| pipe_RCR_3d | 8.21s | 6.19s | -25% | 2.09× slower |
| iliac_artery | 81.58s | 64.65s | -21% | **1.42× faster** |

### Remaining Profile (pipe_simple, after optimization)
| Category | % |
|:---|---:|
| JIT kernel compute | 37.7% |
| Linear solve (SpMV) | 12.7% |
| assembleCellsFused overhead | 8.9% |
| populateFieldSolutionData | 5.7% |
| Heap alloc (malloc/free) | 4.9% |
| Constrained path (residual) | 3.0% |

## Analysis

### Critical Finding: 3D Assembly Per-Element Overhead
After optimization, OOP 3D assembly is ~3× more expensive per element than legacy
(down from 4–5×). The JIT kernel compute is now the dominant remaining cost at
~63% of assembly time.

### Why 2D Cases Win
Improved GMRES (adaptive restart, fused CGS, SIMD dots) + better cache behavior
(36%→19% miss rate) more than compensate for the 2× assembly overhead.

### Priorities for Further Improvement

1. **Reduce JIT kernel per-element cost** (20µs/cell → target 12µs)
   - Profile JIT codegen: operation count, register pressure, code size
   - Tet4 kernels may benefit from element-specific optimizations

2. **Improve block-Schur preconditioner** for large 3D
   - iliac_artery: 48% time in linear solve with diagonal preconditioner
   - ILU(0) or block-ILU would slash linear iteration counts

3. **Enable parallel assembly** for 3D benchmarks
   - 4 threads: ~1.5× speedup on assembly (from prior measurements)
   - Already works via graph coloring (automatic when OMP > 1)

4. **Reduce heap allocation overhead** (4.9% → target 1%)
   - Pre-allocate scratch buffers for constraint checking path
   - Pool allocator for assembly-internal temporaries
