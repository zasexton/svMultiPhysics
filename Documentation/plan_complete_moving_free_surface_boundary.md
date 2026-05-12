# Complete Moving Free-Surface Boundary Plan

## Purpose

This plan lists the remaining FE-library and Physics-library work needed to run a complete moving free-surface water simulation in the new OOP solver, such as an open vessel that is partially filled with water.

The implementation must keep FE infrastructure physics-agnostic. FE should provide generic geometry, interface, integration, stabilization, field-evaluation, and diagnostic services. Physics modules should own Navier-Stokes free-surface equations, mesh-motion or level-set evolution choices, fluid properties, contact-line models, and solver coupling.

## Current Baseline

Already implemented:

- FE Forms has generic trace and projection helpers:
  - `normalTrace`
  - `normalProjection`
  - `tangentialProjection`
  - `projectNormal`
  - `projectTangent`
  - `tangentialTrace`
- FE Forms has generic surface and level-set geometry helpers:
  - `unitNormalFromLevelSet`
  - `meanCurvatureFromLevelSet`
  - `meanCurvatureFromNormal`
  - `curvatureVectorFromNormal`
- FE Forms already exposes boundary and interface integration vocabulary through `.ds(marker)` and `.dI(marker)`.
- Navier-Stokes OOP has a `FreeSurfaceBoundary` option with two implementations:
  - `FittedALE`
  - `UnfittedLevelSet`
- Navier-Stokes OOP can apply the dynamic stress balance
  `sigma(u,p)n = (-p_ext + gamma * kappa)n`
  on a fitted boundary marker or an existing interface marker.
- The fitted path reserves boundary markers through the existing FE `BoundaryConditionManager`, so incompatible BCs are detected without adding free-surface-specific logic to FE.
- The unfitted path can consume an existing scalar level-set field and an existing interface marker.

Not yet complete:

- Fitted ALE free surfaces do not yet have a complete mesh-motion coupling workflow that evolves the free-surface geometry from the fluid velocity.
- Unfitted level-set free surfaces do not yet have FE-generated cut-interface quadrature from `phi = 0`.
- Unfitted level-set free surfaces do not yet have a level-set transport, reinitialization, or volume-correction Physics workflow.
- No complete water-in-open-vessel OOP example case exists yet.

## Target Capabilities

The final implementation should support two complete free-surface strategies.

### Fitted ALE Free Surface

The free surface is a boundary of the fluid mesh. Its motion is represented by mesh displacement and mesh velocity.

Required behavior:

- Apply dynamic stress balance on the moving surface.
- Evolve the mesh boundary so the normal mesh velocity matches the normal fluid velocity.
- Maintain mesh quality through a mesh-motion formulation.
- Allow tangential mesh-motion policy choices.
- Support surface tension with curvature from current geometry.
- Support contact-line behavior where the free surface meets vessel walls.

### Unfitted Level-Set Free Surface

The free surface is the zero contour/surface of a scalar field inside a background fluid mesh.

Required behavior:

- Transport the level-set field with the fluid velocity.
- Generate integration geometry for the current `phi = 0` interface.
- Apply dynamic stress balance on the generated interface.
- Maintain level-set quality through reinitialization or signed-distance repair.
- Track water volume and support volume correction.
- Support contact-angle or contact-line treatment at solid walls.

## Implementation Order

Recommended order:

1. Complete fitted ALE free-surface coupling first.
2. Add FE cut-interface generation and quadrature.
3. Add level-set transport and reinitialization.
4. Couple unfitted free-surface stress to generated interfaces.
5. Add full open-vessel examples and validation cases.

This order produces a usable moving free surface earlier, while the more complex unfitted infrastructure is built behind generic FE interfaces.

## FE Library Work

### 1. Complete Generic Moving-Boundary Geometry Support

Goal: make fitted ALE free-surface residuals fully differentiable with respect to moving mesh state.

Concrete changes:

- Audit moving-boundary terminals in FE Forms:
  - `currentNormal()`
  - `currentMeasure()`
  - `meshVelocity()`
  - `meshDisplacement()`
  - `referenceNormal()`
  - `referenceMeasure()`
- Ensure these terminals are supported inside `.ds(marker)` residuals.
- Ensure symbolic tangent generation covers normal and measure dependence on mesh displacement.
- Ensure `GeometryTangentPath` policies work for boundary terms.
- Add analysis/debug metadata that can identify forms depending on moving-boundary geometry.

Checklist:

- [x] Confirm `currentNormal()` lowers correctly on boundary integrals.
- [x] Confirm `currentMeasure()` lowers correctly on boundary integrals.
- [x] Confirm derivatives of `currentNormal()` with respect to mesh displacement are available.
- [x] Confirm derivatives of `currentMeasure()` with respect to mesh displacement are available.
- [x] Add finite-difference tests for moving-boundary normal tangents.
- [x] Add finite-difference tests for moving-boundary measure tangents.
- [x] Add mixed fluid/mesh residual tests that include boundary geometry terms.
- [x] Add diagnostics for invalid or missing moving-geometry tangent paths.

Done when:

- A residual using `currentNormal()` and `currentMeasure()` on `.ds(marker)` matches finite-difference Jacobians with respect to mesh displacement.

### 2. Add Physics-Agnostic Cut-Interface Generation

Goal: generate interface integration domains from scalar level-set fields without embedding any free-surface assumptions in FE.

Concrete changes:

- Add an FE interface-domain builder, for example:
  - `FE/Interfaces/LevelSetInterfaceDomain.h`
  - `FE/Interfaces/LevelSetInterfaceDomain.cpp`
  - `FE/Interfaces/CutInterfaceQuadrature.h`
- Input:
  - scalar FE field id or scalar field evaluator
  - interface marker
  - zero isovalue, initially `0.0`
  - mesh access
- Output:
  - per-cell interface fragments
  - quadrature points
  - quadrature weights
  - generated interface normals
  - owning cell ids
  - optional plus/minus side tags
- Register generated interface domains so existing `.dI(marker)` lowering can consume them.

Checklist:

- [x] Define a generic cut-interface data model.
- [x] Implement 2D cell cuts for triangles and quads.
- [x] Implement 3D cell cuts for tetrahedra.
- [x] Add extension points for hexes, wedges, and pyramids.
- [x] Handle edge cases:
  - [x] no cut
  - [x] full-cell zero field rejection or fallback
  - [x] vertex cuts
  - [x] edge cuts
  - [x] nearly tangent cuts
  - [x] very small fragments
- [x] Assign stable interface marker ids.
- [x] Expose generated interfaces through the same integration path used by `.dI(marker)`.
- [x] Add serial tests for generated interface fragment counts.
- [x] Add MPI tests for ownership and global area consistency.

Done when:

- A scalar level set can generate an interface marker that a Forms residual can integrate over with `.dI(marker)`.

### 3. Generalize Interface Quadrature And Field Evaluation

Goal: evaluate FE fields and gradients accurately on generated embedded interfaces.

Concrete changes:

- Extend quadrature infrastructure to support generated interface points.
- Evaluate H1 scalar and vector fields at cut-interface quadrature points.
- Evaluate gradients at cut-interface quadrature points.
- Support side restrictions where required by existing interface expression semantics.
- Preserve existing `.dI(marker)` behavior for already-materialized interfaces.

Checklist:

- [x] Add cut-interface quadrature rules for linear fragments.
- [x] Add configurable quadrature order.
- [x] Evaluate scalar H1 fields on cut-interface quadrature points.
- [x] Evaluate vector H1 fields on cut-interface quadrature points.
- [x] Evaluate scalar gradients on cut-interface quadrature points.
- [x] Evaluate vector gradients on cut-interface quadrature points.
- [x] Add integration tests for constants.
- [x] Add integration tests for linear fields.
- [x] Add integration tests for field gradients.
- [x] Add tests comparing generated normals with level-set-gradient normals.

Done when:

- `.dI(marker)` integrates constants and linear fields over generated plane cuts with expected accuracy.

### 4. Add Generic Cut-Cell Stabilization Infrastructure

Goal: provide FE mechanisms needed by unfitted methods without making FE know the physics equation.

Concrete changes:

- Add generic discovery of facets adjacent to cut cells.
- Add ghost-penalty style integration support over selected interior facets.
- Add vocabulary helpers for jumps of gradients or normal derivatives, reusing existing jump/average semantics where possible.
- Let Physics modules opt into stabilization terms by composing Forms expressions.

Checklist:

- [x] Identify cut cells from a generated interface domain.
- [x] Identify interior facets adjacent to cut cells.
- [x] Expose a marker or facet-set handle for cut-adjacent facets.
- [x] Support integration over that facet set.
- [x] Reuse existing jump/average operators where valid.
- [x] Add tests for facet-set construction.
- [x] Add tests for ghost-penalty residual assembly.
- [x] Add conditioning-oriented regression tests for small cut fragments.

Done when:

- A Physics module can author generic cut-cell stabilization terms over cut-adjacent facets using Forms expressions.

### 5. Add Interface Diagnostics And Output

Goal: make generated interfaces inspectable and debuggable.

Concrete changes:

- Add VTK/debug output for generated interface fragments.
- Add output fields:
  - level-set value
  - interface normal
  - curvature estimate
  - interface marker
  - owning cell id
  - cut-cell volume fraction
- Add integration summaries:
  - total interface area or length
  - enclosed volume or area where available
  - number of cut cells
  - number of interface fragments

Checklist:

- [x] Add interface geometry writer.
- [x] Add interface summary statistics.
- [ ] Add output for interface normals.
- [ ] Add output for curvature estimates.
- [ ] Add output for cut-cell volume fractions.
- [ ] Add test output for plane, circle, and sphere cuts.

Done when:

- A failing unfitted simulation can produce enough interface diagnostics to inspect the generated free surface.

## Physics Library Work

### 1. Complete Fitted ALE Free-Surface Coupling

Goal: evolve a fitted mesh free surface using the fluid velocity.

Concrete changes:

- Extend Navier-Stokes free-surface options with fitted ALE kinematic policies:
  - normal mesh velocity equals normal fluid velocity
  - optional tangential mesh policy
  - penalty or Nitsche enforcement
- Add or extend mesh-motion boundary conditions so the mesh solver can consume free-surface kinematics.
- Couple the fluid and mesh-motion modules through existing FE/Physics coupling infrastructure.
- Use current-geometry normal and measure in fitted free-surface terms when ALE is active.
- Add current-geometry curvature support for fitted surfaces.

Checklist:

- [ ] Define fitted free-surface kinematic policy options.
- [ ] Add mesh-motion BC that accepts normal velocity or displacement constraints from fluid state.
- [ ] Add tangential mesh policy:
  - [ ] free tangential mesh motion
  - [ ] smoothing-only tangential motion
  - [ ] prescribed tangential motion
- [ ] Add penalty enforcement for `(u - w_mesh) . n = 0`.
- [ ] Add Nitsche enforcement if needed after penalty path is verified.
- [ ] Add fitted-surface curvature from current mesh geometry.
- [ ] Add current-normal/current-measure support to fitted dynamic stress path.
- [ ] Add input translation for fitted ALE free-surface kinematic options.
- [ ] Add coupled fluid plus mesh-motion setup tests.

Done when:

- A fitted free-surface boundary moves with the normal fluid velocity in an ALE simulation, and the coupled Jacobian passes finite-difference checks.

### 2. Add Free-Surface Contact-Line Models

Goal: define behavior where the free surface meets vessel walls.

Concrete changes:

- Add Physics-side contact-line options:
  - pinned contact line
  - prescribed contact angle
  - contact-line mobility
  - optional wall slip model
- Keep FE responsible only for generic boundary/interface geometry queries.
- Add residual terms or boundary constraints in Physics modules.

Checklist:

- [ ] Add free-surface contact-line option struct.
- [ ] Add parser/input options for contact-line model.
- [ ] Implement pinned contact line for fitted ALE.
- [ ] Implement prescribed contact angle for fitted ALE.
- [ ] Define unfitted wall-contact detection requirements.
- [ ] Implement prescribed contact angle for unfitted level-set surfaces.
- [ ] Add tests for static meniscus/contact-angle geometry where practical.

Done when:

- Open-vessel simulations can specify how the free surface behaves at vessel walls.

### 3. Add Level-Set Transport Module

Goal: evolve the unfitted free surface by advecting a scalar level-set field.

Concrete changes:

- Add a Physics formulation module for level-set advection:
  - `Physics/Formulations/LevelSet/LevelSetTransportModule.h`
  - `Physics/Formulations/LevelSet/LevelSetTransportModule.cpp`
  - `Physics/Formulations/LevelSet/LevelSetRegister.cpp`
- Equation:
  - `d(phi)/dt + u . grad(phi) = 0`
- Use existing FE Forms and time-integration infrastructure.
- Couple velocity from Navier-Stokes as a prescribed or coupled field.
- Add SUPG or equivalent advection stabilization.

Checklist:

- [ ] Add level-set field options.
- [ ] Add velocity source options.
- [ ] Add residual form for transient advection.
- [ ] Add SUPG stabilization.
- [ ] Add inflow/outflow treatment for level-set boundaries.
- [ ] Add field registration and input translation.
- [ ] Add unit tests for constant velocity translation.
- [ ] Add Jacobian tests for coupled and prescribed velocity modes.

Done when:

- A level-set plane or circle advects with a prescribed velocity with expected convergence.

### 4. Add Level-Set Reinitialization And Volume Correction

Goal: maintain a usable signed-distance field and conserve water volume over long simulations.

Concrete changes:

- Add reinitialization options:
  - PDE-based reinitialization
  - fast marching or projection-based repair if available
  - reinitialization cadence
- Add volume correction options:
  - target initial volume
  - global shift correction
  - optional local correction
- Add diagnostics:
  - signed-distance error
  - water volume
  - volume loss per step

Checklist:

- [ ] Define reinitialization option struct.
- [ ] Implement signed-distance repair path.
- [ ] Add volume calculation from level-set/cut-cell data.
- [ ] Implement global level-set shift correction.
- [ ] Add reinitialization cadence controls.
- [ ] Add output diagnostics for volume and signed-distance error.
- [ ] Add tests for volume preservation under pure advection.
- [ ] Add tests for signed-distance maintenance near the interface.

Done when:

- Long-running level-set advection preserves interface quality and volume within configured tolerances.

### 5. Couple Unfitted Free Surface To Generated Interfaces

Goal: make `UnfittedLevelSet` free-surface boundaries fully operational from a level-set field alone.

Concrete changes:

- At each time step or nonlinear update, trigger FE interface generation from the current level-set state.
- Bind generated interface markers to Navier-Stokes `FreeSurfaceBoundary`.
- Apply dynamic stress balance on the generated `.dI(interface_marker)`.
- Add optional cut-cell stabilization terms for the Navier-Stokes system.
- Ensure interface updates are compatible with time integration, restart, and MPI.

Checklist:

- [ ] Add generated-interface lifecycle orchestration.
- [ ] Update interface geometry after level-set state changes.
- [ ] Preserve marker identity across time steps.
- [ ] Couple generated interface marker to Navier-Stokes free-surface options.
- [ ] Add optional cut-cell stabilization options to Navier-Stokes.
- [ ] Add restart data for level-set fields and generated-interface metadata as needed.
- [ ] Add MPI tests for generated interface consistency.

Done when:

- A user can specify `UnfittedLevelSet` with a level-set field and run Navier-Stokes free-surface stress on the generated interface without manually providing interface markers.

### 6. Add Water/Open-Vessel Simulation Setup Support

Goal: make the common open-vessel water case straightforward and reproducible.

Concrete changes:

- Add input options for water material properties:
  - density
  - viscosity
  - surface tension
  - gravity
  - atmospheric pressure
- Add hydrostatic pressure initialization utility.
- Ensure pressure gauge/nullspace handling is compatible with free surfaces.
- Add complete example cases:
  - fitted ALE open tank
  - unfitted level-set open tank

Checklist:

- [ ] Add example fitted ALE input file.
- [ ] Add example unfitted level-set input file.
- [ ] Add mesh and initial conditions for half-filled vessel.
- [ ] Add gravity/body-force setup.
- [ ] Add hydrostatic initialization helper.
- [ ] Add pressure gauge/nullspace configuration.
- [ ] Add output requests for surface position, volume, pressure, and velocity.

Done when:

- The repository contains an executable OOP open-vessel water example for each supported implementation.

## Verification Tests

### FE Verification

Checklist:

- [ ] Plane level set cuts triangles exactly in 2D.
- [ ] Plane level set cuts tetrahedra exactly in 3D.
- [ ] Circle interface length converges under mesh refinement.
- [ ] Sphere interface area converges under mesh refinement.
- [ ] Level-set normal matches analytic normal for plane, circle, and sphere.
- [ ] Curvature converges for circle and sphere.
- [ ] `.dI(marker)` integrates constants over generated interfaces.
- [ ] `.dI(marker)` integrates linear fields over generated interfaces.
- [ ] Field gradients evaluate correctly at cut-interface quadrature points.
- [ ] Moving-boundary normal tangent matches finite differences.
- [ ] Moving-boundary measure tangent matches finite differences.
- [ ] Parallel interface generation produces consistent global area and fragment counts.

### Physics Verification

Checklist:

- [ ] Static flat water surface with gravity remains at rest.
- [ ] Hydrostatic pressure field matches analytic pressure.
- [ ] External-pressure free-surface traction has the expected sign.
- [ ] Surface-tension pressure jump matches Laplace law for a circular interface.
- [ ] Surface-tension pressure jump matches Laplace law for a spherical interface.
- [ ] Fitted ALE surface moves with prescribed normal velocity.
- [ ] Fitted ALE fluid plus mesh Jacobian matches finite differences.
- [ ] Level-set transport preserves a translating plane.
- [ ] Level-set transport preserves a translating circle to expected accuracy.
- [ ] Reinitialization preserves signed-distance quality near the interface.
- [ ] Volume correction maintains water volume within tolerance.
- [ ] Fitted and unfitted implementations agree on a simple static flat-surface case.

## Validation Cases

Checklist:

- [ ] Open tank at rest, half-filled with water.
- [ ] Small-amplitude sloshing frequency compared with analytic theory.
- [ ] Capillary wave oscillation benchmark.
- [ ] Capillary wave decay benchmark if viscosity is included.
- [ ] Dam-break or water-column collapse benchmark after stability is mature.
- [ ] Long transient volume conservation benchmark.
- [ ] Contact-angle static meniscus benchmark after contact-line support is added.

## Required Example Cases

### Fitted ALE Open Vessel

Checklist:

- [ ] Mesh with water domain and top free-surface boundary marker.
- [ ] Wall no-slip or slip boundary conditions.
- [ ] Free-surface dynamic stress boundary.
- [ ] Free-surface kinematic mesh-motion coupling.
- [ ] Mesh-motion smoothing equation.
- [ ] Gravity.
- [ ] Hydrostatic initial pressure.
- [ ] Pressure gauge/nullspace configuration.
- [ ] Time-step and nonlinear solver settings.
- [ ] Output for free-surface displacement, pressure, velocity, and volume.

### Unfitted Level-Set Open Vessel

Checklist:

- [ ] Background mesh covering vessel water region.
- [ ] Initial level-set field for half-filled water.
- [ ] Level-set transport module.
- [ ] Interface generation from level set.
- [ ] Navier-Stokes unfitted free-surface stress boundary.
- [ ] Cut-cell stabilization.
- [ ] Reinitialization and volume correction settings.
- [ ] Gravity.
- [ ] Hydrostatic initial pressure.
- [ ] Pressure gauge/nullspace configuration.
- [ ] Output for level set, generated interface, pressure, velocity, and volume.

## Risk Areas

- Cut-interface generation must be robust for small and nearly tangent cuts.
- Surface curvature can be noisy on both fitted and unfitted surfaces.
- Level-set methods can lose volume without correction.
- Contact-line behavior is physically and numerically sensitive.
- Strong coupling between fluid, free surface, and mesh motion may require robust nonlinear solver settings.
- MPI consistency for generated interfaces and cut-cell stabilization must be addressed early.

## Completion Criteria

The moving free-surface feature should be considered complete when:

- Fitted ALE and unfitted level-set implementations each have at least one complete open-vessel example.
- The fitted ALE path evolves the free-surface mesh consistently with fluid velocity.
- The unfitted path generates its own interface geometry from the level-set field.
- Both paths pass hydrostatic, pressure-jump, and basic motion verification tests.
- At least one sloshing or capillary-wave validation case is documented.
- Diagnostics are sufficient to inspect interface position, normals, curvature, and volume conservation.
