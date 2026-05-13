# FE Level-Set Services

`svmp::FE::level_set` owns reusable level-set infrastructure. The namespace is
physics-neutral: it knows about FE systems, fields, function spaces, generated
interfaces, cut-cell integration data, and diagnostics, but it does not encode
Navier-Stokes, material, contact-line, or free-surface policy.

Physics modules remain responsible for equation-specific defaults and
formulation-specific residuals. The application translator owns legacy
level-set equation input compatibility, translates those choices into
`FE::level_set` options, and calls the FE services directly.

## Public API

The stable entry point is the aggregate header:

```cpp
#include "FE/LevelSet/LevelSet.h"
```

Use specific headers when compile time or dependency clarity matters:

| Header | Public surface |
|--------|----------------|
| `FE/LevelSet/LevelSetOptions.h` | option structs, source enums, cadence helpers |
| `FE/LevelSet/LevelSetTransport.h` | `installLevelSetTransport()` |
| `FE/LevelSet/LevelSetInterfaceLifecycle.h` | generated interface domain creation and update lifecycle |
| `FE/LevelSet/LevelSetVolume.h` | cut-cell volume measurement and global-shift correction |
| `FE/LevelSet/LevelSetReinitialization.h` | signed-distance repair from generated interface geometry |
| `FE/LevelSet/LevelSetDiagnostics.h` | scalar diagnostics for volume and signed-distance quality |
| `FE/LevelSet/LevelSetRestart.h` | restart records for level-set fields and generated interfaces |

APIs in `FE/LevelSet` are public FE APIs unless they are in an anonymous
namespace or marked internal in the header. Keep new reusable level-set
algorithms in this directory and expose them through the smallest header that
matches their scope.

## Free-Surface Consumers

Navier-Stokes owns free-surface boundary semantics. Fitted ALE free surfaces use
the boundary marker and moving-mesh quantities declared by the Navier-Stokes
module. Unfitted free surfaces consume `FE::level_set` generated interfaces,
level-set curvature helpers, and cut-cell metadata, then install
Navier-Stokes-specific pressure-jump, surface-tension, kinematic, and
stabilization terms on the resulting interface domain.

The application level-set equation translator is an input adapter. It preserves
legacy XML names, builds FE options, and calls
`FE::level_set::installLevelSetTransport()`. It must not contain reusable
transport, volume, reinitialization, diagnostics, or restart algorithms.

## Compatibility

No Physics compatibility headers remain for the migrated level-set APIs. New
code should include `FE/LevelSet/...` directly and use
`svmp::FE::level_set` names. User-facing level-set XML names remain accepted by
the application input adapter for backward compatibility.

## Module Authoring

When a Physics module needs level-set services:

1. Include the FE header that owns the service, usually
   `FE/LevelSet/LevelSet.h` or a narrower `FE/LevelSet/...` header.
2. Parse module-specific input in the application or Physics module and
   translate it into the matching `FE::level_set` option struct.
3. Pass the target `FESystem`, fields, spaces, markers, and options to the FE
   service.
4. Keep equation-specific residuals and boundary laws in the Physics module.
5. Add reusable level-set algorithms, restart records, diagnostics, and
   generated-interface utilities to `FE/LevelSet`, not to Physics.
