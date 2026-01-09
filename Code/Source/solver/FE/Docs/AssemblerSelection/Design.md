# FE Assembler Auto-Selection — Design

## Opt-in Surface (Backward Compatible)

Selection is controlled by `systems::SetupOptions::assembler_name`:
- `"StandardAssembler"` (default): preserves current behavior.
- `"Auto"`: enables rule-based selection from form/system characteristics.
- Explicit names (e.g., `"WorkStreamAssembler"`) are treated as explicit opt-in and are validated against required capabilities (e.g., DG).

Auto-selection behavior is further gated by `assembly::AssemblyOptions::auto_policy`:
- `AutoSelectionPolicy::Conservative` (default): only selects assemblers deemed safe given current capabilities and form requirements.
- Future policies may be added, but must remain explicitly opt-in.

Orthogonal features are enabled via `assembly::AssemblyOptions` flags:
- `cache_element_data` → caching decorator
- `use_batching` / `batch_size` → vectorization decorator (initially a structural wrapper)
- `schedule_elements` / `schedule_strategy` → scheduling decorator (new options)

Iterative-solver leverage is opt-in via `systems::SetupOptions::auto_register_matrix_free`:
- When enabled (single-field only for now), eligible cell-only linear operators are also registered as matrix-free backends.

Diagnostics:
- `systems::FESystem::assemblerSelectionReport()` exposes a human-readable report of the selection + decorators applied.

## Characteristics Model

### `assembly::FormCharacteristics`
Summarizes compiled-form needs used for selection/validation:
- `has_cell_terms`, `has_boundary_terms`, `has_interior_face_terms`
- `has_global_terms` (reporting)
- `has_field_requirements` (hard requirement)
- `has_parameter_specs` (reporting)
- `max_time_derivative_order`
- `required_data` (union across kernels/terms)
- convenience booleans: `needs_solution`, `needs_material_state`, `needs_dg`

### `assembly::SystemCharacteristics`
Summarizes system-level context:
- `num_fields`
- `dimension`
- `num_cells`
- `num_dofs_total`, `max_dofs_per_cell`, `max_polynomial_order`
- `num_threads` (resolved)
- `mpi_world_size` (if MPI initialized; otherwise 1)

## Factory Interface

Add a new factory entrypoint that accepts:
- `AssemblyOptions`
- `assembler_name` (base selection or `"Auto"`)
- `FormCharacteristics` + `SystemCharacteristics`

The factory:
1. Selects a **base** assembler (`StandardAssembler` unless explicitly requested otherwise).
2. Applies **decorators** in a deterministic order (e.g., scheduling → caching → vectorization).
3. Validates required capabilities and throws on incompatibility (DG, full context, solution/transient support, material state, field requirements, multi-field DOF offsets).
4. Optionally emits a selection report (stored by Systems during `FESystem::setup()`).

## Composition via `DecoratorAssembler`

Introduce `assembly::DecoratorAssembler`:
- Holds `std::unique_ptr<Assembler> base_`.
- Forwards all configuration, assembly, lifecycle, and query methods.
- Enables stacking orthogonal behaviors without re-implementing the full `StandardAssembler` surface.

Existing wrappers are generalized:
- `CachedAssembler` becomes a decorator around any base assembler.
- `SymbolicAssembler` becomes a decorator around any base assembler (default base: `StandardAssembler`), while retaining convenience APIs for compiling/evaluating `forms::FormExpr` directly.
