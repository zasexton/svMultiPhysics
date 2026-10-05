# Build Configuration And JIT Code Generation

This note records how the solver is compiled today, the opt-in build options
that make it faster without changing results, and the JIT code-generation
settings.  The measurements use the free-surface reference cases (static and
sessile drops, capillary wave, sloshing, 3D tanks and the first step of the 3D
sphere proxy), serial runs, on Intel Skylake (Xeon Gold 5118) nodes.

## Production Build

The production build is CMake `Release` with GCC 12.4.0 for the compiler's
generic x86-64 target: `-O3 -DNDEBUG`, no `-march`, no link-time optimization
(LTO), no profile-guided optimization (PGO).  Some FE translation units add
`-ftree-vectorize -funroll-loops` (FSILS, assembly, basis), and
`Interfaces/detail/ProducerArithmeticAssessment.cpp` is pinned to
`-ffp-contract=off -fno-lto -msse2`.  GCC's default `-ffp-contract=fast` has
no effect for the generic target, which has no FMA instructions.  LLVM 17 is
linked statically; its own code generation speed is not affected by these
options.

## Build Options

All options are off by default and apply to every C and C++ target
(`Code/CMake/SimVascularBuildOptimization.cmake`; the top-level superbuild
forwards them).

| Option | Values | Effect |
|--------|--------|--------|
| `SV_ENABLE_LTO` | `OFF`, `ON` | Link-time optimization (`CMAKE_INTERPROCEDURAL_OPTIMIZATION`; GCC `-flto=auto -fno-fat-lto-objects`). |
| `SV_PGO` | `OFF`, `GENERATE`, `USE` | Profile-guided optimization, GCC 11 or newer.  `GENERATE` builds an instrumented solver; `USE` compiles with the recorded profile and `-fprofile-partial-training`, so code the training did not run stays optimized normally. |
| `SV_PGO_PROFILE_DIR` | path | Profile directory (default `<build>/pgo-profile`).  Profile files are named relative to the build directory, so a `USE` build in another directory with the same layout finds them. |

The existing `FE_ENABLE_IPO` option enables LTO for the FE library only.

## Recommended Configuration

`SV_ENABLE_LTO=ON` with `SV_PGO=USE`, generic instruction set, default JIT
settings.  Outputs are bitwise identical to the production build, with the
same accepted steps, outer passes, Newton and linear iterations, and the
reference cases run 8.5-11% faster (3-4% on cases outside the training set).

Recipe (configure arguments as in the production build, from `Code/`):

```bash
# 1. Instrumented build.
cmake -S Code -B build-pgo-gen -DCMAKE_BUILD_TYPE=Release <usual arguments> \
      -DSV_PGO=GENERATE -DSV_PGO_PROFILE_DIR=$PROFILE
cmake --build build-pgo-gen -j 12 --target svmultiphysics

# 2. Training: run representative cases with build-pgo-gen/bin/svmultiphysics.
#    Every process that exits normally merges its counts into $PROFILE
#    (concurrent runs are safe).  Runs killed by a time limit add nothing.

# 3. Optimized build, any build directory.
cmake -S Code -B build-opt -DCMAKE_BUILD_TYPE=Release <usual arguments> \
      -DSV_ENABLE_LTO=ON -DSV_PGO=USE -DSV_PGO_PROFILE_DIR=$PROFILE
cmake --build build-opt -j 12 --target svmultiphysics
```

The study trained on the eight 2D reference cases, `tank3d_L8` and the first
step of the sphere proxy.  The instrumented solver is 3.5 to 4 times slower
(the sphere step took 82 minutes), so the 2D cases and `tank3d_L8` alone (about
10 minutes, three runs at a time) are a reasonable training set.  A stale
profile never changes results: functions whose source changed lose their
profile and are optimized as without PGO (GCC prints a coverage-mismatch
warning), so the gain decays as the code moves on.  Regenerate the profile
after larger changes to the hot paths.

LTO alone is bitwise identical but gives no measurable speed-up; it shrinks the
solver by 10% and its build is not slower.  With LTO, every test executable is
also linked with LTO, which makes test builds slower.

## Measured Effect

Fourteen variants ran each case at the same time on one node (so all saw the
same load), first with an empty JIT object cache (cold) and then reusing it
(warm).  Wall times are sums over the nine reference cases; speed-up is base
time over variant time.  Build times are for the solver target with 12 jobs.

| Variant | Flags | Cold s | Warm s | Speed-up cold / warm | Held-out cold / warm | Bitwise | Build s | Size MB |
|---------|-------|--------|--------|----------------------|----------------------|---------|---------|---------|
| base | production | 752.1 | 649.5 | 1 / 1 | 1 / 1 | yes (= reference job) | 422 | 101.6 |
| lto | `SV_ENABLE_LTO=ON` | 752.8 | 647.4 | 0.999 / 1.003 | 0.983 / 1.003 | yes | 346 | 91.6 |
| pgo | `SV_PGO=USE` | 702.4 | 601.0 | 1.071 / 1.081 | 1.010 / 1.024 | yes | 466 | 109.1 |
| ltopgo | LTO + PGO | 692.8 | 586.3 | 1.085 / 1.108 | 1.027 / 1.044 | yes | 365 | 98.9 |
| v3 | `-march=x86-64-v3 -ffp-contract=off` | 772.7 | 670.1 | 0.973 / 0.969 | 0.949 / 0.945 | no | - | 102.6 |
| v3fma | `-march=x86-64-v3` | 765.6 | 664.0 | 0.982 / 0.978 | 0.992 / 0.983 | no | - | 102.5 |
| ltopgov3 | LTO + PGO + v3 | 720.3 | 612.0 | 1.044 / 1.061 | 0.947 / 0.936 | no | 375 | 99.6 |

Held-out cases are `tank3d_L16` and `tank3d_L16_wet`, which the PGO training
did not run.  Repeated runs of identical code differ by about 1%.

First step of the 3D sphere proxy (`sphere_proxy_L8_kagl`, 6.4 GB, run in
pairs beside other work): LTO with PGO took 1530 s against 1554 s for base in
the same pair (1.6%).  Newton time drops by 9% and the cut-quadrature backend
by 23%, but 74% of the step is geometry work outside Newton, which gains under
1%.  An earlier perf profile of this case puts most of that work in the
certified interval arithmetic of `Interfaces/detail/ProducerArithmeticAssessment.cpp`
(compiled with pinned flags and without LTO) and in curvature sample
collection.  LTO, PGO and LTO with PGO all give bitwise identical sphere
outputs.

On an AMD Milan node (EPYC 7543, cold runs beside a test build, so timings
vary by up to 5% between identical binaries), LTO with PGO was 6.1% faster on
the reference cases (2.6% held out).  The same options applied through
`SV_ENABLE_LTO=ON SV_PGO=USE` to a newer source tree, with the profile of the
older one, were 4.6% faster than the generic build of that tree (4.1% held
out).

Per phase (sums over the nine reference cases, warm cache, seconds):

| Variant | Setup | Newton | Assembly | Cut volumes | Linear solve | Outside Newton |
|---------|-------|--------|----------|-------------|--------------|----------------|
| base | 40.6 | 255.9 | 138.9 | 82.2 | 65.8 | 347.8 |
| ltopgo | 40.0 | 214.6 | 114.2 | 66.5 | 58.3 | 326.8 |

"Outside Newton" is the time-loop time outside the Newton solves: cut and
geometry rebuild, curvature projection, constraints and diagnostics.  LTO with
PGO cuts assembly by 18%, cut volumes by 19%, the linear solve by 11% and the
work outside Newton by 6%.  JIT kernel execution is below 0.1% of the samples
in perf profiles of these cases.

## Instruction Set

`-march=x86-64-v3` (AVX2 and FMA, available on both the Skylake and the Milan
nodes) is 2-3% slower (5-8% on the 3D tanks; assembly takes 8-15% longer) and
not bitwise identical, even with `-ffp-contract=off`.  Two things change
besides the compiler's own code generation:

- Eigen uses 4-wide packets and FMA intrinsics, which changes the order of
  its reductions;
- the JIT hardware profile reads the SIMD width from the compiler's
  instruction-set macros, so a v3 build reports 256-bit SIMD and switches the
  JIT SIMD batch path (qualified for two lanes only) off.

The JIT part is not the cause of the slowdown: a v3 build whose hardware
profile was compiled for the generic target is just as slow.  The build options
therefore offer no instruction-set setting.

The largest field difference over the reference cases is 4e-9 relative to the
field scale (parasitic velocity in `drop_L8_kagc`); all others are below 4e-12,
and step, outer-pass, Newton and linear-iteration counts are unchanged.  The 3D
tanks are at rest: their exact velocity is zero and the computed one is
round-off of about 2e-14, which changes by up to 1e-14 in absolute terms.

## JIT Code Generation

Current settings (`Forms/JIT/JITEngine.cpp`, `LLVMGen.cpp`, `HardwareProfile.cpp`):

- optimization level 3 from `PhysicsJITPolicy` (O3 IR pipeline, aggressive
  code generator);
- target: the host CPU with all host features (`skylake-avx512` or `znver3`);
- strict floating point (`JITFastMathMode::Strict`), but LLVMGen emits
  `llvm.fmuladd` for scalar `a*b + c` at level 2 and above, and LLVM fuses it
  into FMA on both node types;
- the hardware profile of a generic build reports 128-bit SIMD and 16
  registers, so the IR pipeline has no target machine (loop and SLP
  vectorization are effectively off) and the SIMD batch path is two lanes wide.

Measured with environment overrides (`Forms/JIT/README.md`), against the same
binary with default settings:

| Setting | Cold / warm speed-up | JIT compilation | Bitwise |
|---------|----------------------|-----------------|---------|
| `SVMP_JIT_OPT_LEVEL=2` | 1.001 / 1.001 | unchanged | yes |
| `SVMP_JIT_OPT_LEVEL=1` | 1.021 / 1.002 | -14% | no (no `fmuladd`) |
| `SVMP_JIT_TARGET_AWARE=1` | 0.853 / 0.996 | 2.2 times longer | yes |
| `SVMP_JIT_CPU=x86-64-v3` | 0.998 / 1.006 | unchanged | yes |
| `SVMP_JIT_FP_CONTRACT=on` | 1.007 / 1.007 (3D tanks 1.03-1.05) | unchanged | no |
| `SVMP_JIT_FP_CONTRACT=off` | 0.996 / 0.998 | unchanged | no |

No JIT setting is worth changing by default.  Contraction everywhere
(`SVMP_JIT_FP_CONTRACT=on`) speeds up cut-volume assembly of the 3D tanks by
about 12% but changes round-off; the largest field difference is the same 4e-9
of `drop_L8_kagc`, and the counts are unchanged.

JIT compilation costs 10-15 seconds per run with an empty object cache, 14% of
the cold wall time of the reference cases and up to half of a short run.  A
shared, persistent `SVMP_JIT_CACHE_DIR` removes it; warm and cold runs are
bitwise identical.

## Object Cache Across Node Types

Every kernel cache key includes the CPU name and the full feature list of the
code-generation target, and the hardware profile (cache sizes).  With the
default host target, objects compiled on an AVX-512 node are never loaded on an
AVX2 node, even in a shared cache directory.  To share objects between node
types, set `SVMP_JIT_CPU=x86-64-v3` and the same `SVMP_CACHE_PROFILE` on all
nodes.  Checked with one cache directory: with the host target a Milan run
after a Skylake run compiled every kernel again (no disk hits); with
`SVMP_JIT_CPU=x86-64-v3` and a pinned `SVMP_CACHE_PROFILE` it loaded every
kernel the Skylake run had compiled.  All four runs gave bitwise identical
outputs.
