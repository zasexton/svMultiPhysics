# WP-0 configuration containment qualification

Disposition: `PASS`

The frozen `free_surface_configuration_containment_v1` matrix ran against source commit `ffef62d3bd7af3f125074297d6f98c81e3cd916f`. All 24 predeclared tests completed, with zero failures, errors, disabled tests, skips, missing tests, or unexpected tests.

The matrix covers incomplete aliases; all four typed contact alternatives; ambiguous, unknown, duplicate-owner, and cross-model keys; explicit schema migration; validation before system mutation; boundary-local and registration-order-invariant Nitsche assembly; independence of generic weak velocity policy; distinct fitted tangential mesh policies and ownership; fail-closed fitted exclusions; and the complete effective-configuration snapshot.

The tracked source tree was clean. Committed CMake currently names source files that are present only as local supplemental inputs, so the detached build declared and hashed all 12 such inputs. Their paths, sizes, and hashes are in `manifest.json` and `build.json`; the combined source-state hash is `96f0693d54cd986257669618547b4ff90116c29b76fcbd7f1f720b6ddb62264c`. The isolated optional-module test translation unit contains only:

```cpp
// Isolated configuration qualification does not exercise this optional module.
```

The test binary hash is `b958326419baf1fa1dab4286c22a01b7d9ac1f6f536abefe4303a0100af31959`. The run completed in `0.2057570629986003` seconds with sampled peak resident memory of `29384` KiB, inside the frozen 120-second, 2048-MiB, and 16-MiB envelopes. `checksums.txt` verifies every raw runner artifact.
