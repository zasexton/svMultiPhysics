# Darcy Node Pressure Constraints Through Poisson H1

## Goal

Allow Darcy pressure solves to reuse the existing scalar Poisson H1 physics path while accepting node-based Dirichlet pressure constraints from the solver input file:

```xml
<Node_pressure_constraints>
  <Id_type>Global_vertex_gid</Id_type>
  <Values_file_path>known_pressure_nodes.csv</Values_file_path>
</Node_pressure_constraints>
```

The boundary pressure conditions remain ordinary face-marker Dirichlet boundary conditions. The new feature adds pressure constraints at named mesh vertices, including interior vertices.

## Intended User Model

The Darcy pressure equation is assembled through `PoissonModule` because the pressure-only Darcy equation has the same scalar elliptic form:

```text
-div(K grad p) = f
```

For the initial implementation:

- The pressure space is scalar H1.
- The node constraints target mesh vertices only.
- The supported node id mode is `Global_vertex_gid`.
- The values file is a CSV with one node id and one pressure value per row.
- The pressure values are steady constants.

Example CSV:

```csv
node_id,pressure
10231,1250.0
10877,980.0
```

## Main Implementation Steps

### 1. Add Input Data Structures

Add a typed input record to `Code/Source/solver/Physics/Core/EquationModuleInput.h`:

```cpp
struct NodePressureConstraintInput {
  std::string id_type{};
  std::string values_file_path{};
};

struct EquationModuleInput {
  ...
  std::optional<NodePressureConstraintInput> node_pressure_constraints{};
};
```

Keep this generic enough for Darcy-through-Poisson, but do not put arbitrary solver semantics into `BoundaryConditionInput`; these constraints are not face-based boundary conditions.

Checklist:

- [x] `EquationModuleInput` should use `std::optional<NodePressureConstraintInput>`
- [x] Add the chosen data structure to `Code/Source/solver/Physics/Core/EquationModuleInput.h`.
- [x] Include any needed standard headers, such as `<optional>`, only if the chosen representation requires them.
- [x] Keep defaults empty so existing equations behave identically when the XML block is absent.
- [x] Confirm existing code that copies, constructs, or logs `EquationModuleInput` still compiles.
- [x] Confirm node pressure constraints are not added to `BoundaryConditionInput`.

### 2. Parse the XML Element

Add a small parameter holder in `Code/Source/solver/Parameters.h`:

```cpp
class NodePressureConstraintsParameters : public ParameterList {
public:
  static const std::string xml_element_name_;

  NodePressureConstraintsParameters();

  Parameter<std::string> id_type;
  Parameter<std::string> values_file_path;

  bool value_set{false};
  void set_values(tinyxml2::XMLElement* elem);
};
```

Implementation in `Code/Source/solver/Parameters.cpp`:

- Set `xml_element_name_ = "Node_pressure_constraints"`.
- Register `Id_type` as required or default it to `Global_vertex_gid`.
- Register `Values_file_path` as required.
- Set `value_set = true` in `set_values()`.
- Reject unknown child elements.

Add a member to `EquationParameters`:

```cpp
NodePressureConstraintsParameters node_pressure_constraints;
```

Then extend `EquationParameters::set_values()` to recognize the new nested element:

```cpp
} else if (name == NodePressureConstraintsParameters::xml_element_name_) {
  node_pressure_constraints.set_values(item);
```

This mirrors how `Add_BC`, body forces, and domain-specific nested input are currently handled.

Checklist:

- [x] Declare `NodePressureConstraintsParameters` in `Code/Source/solver/Parameters.h`.
- [x] Define `NodePressureConstraintsParameters::xml_element_name_` as `"Node_pressure_constraints"` in `Code/Source/solver/Parameters.cpp`.
- [x] Register `Id_type` with default or required value `Global_vertex_gid`.
- [x] Register `Values_file_path` as required.
- [x] Implement `set_values()` using the same child-element parsing pattern used by other parameter classes.
- [x] Set `value_set = true` only after the XML element has been parsed.
- [x] Reject unknown nested elements with a clear error message.
- [x] Add `NodePressureConstraintsParameters node_pressure_constraints;` to `EquationParameters`.
- [x] Extend `EquationParameters::set_values()` to dispatch `Node_pressure_constraints` to `node_pressure_constraints.set_values(item)`.
- [x] Update `EquationParameters::print_parameters()` if solver input diagnostics should show this block.
- [x] Confirm existing XML files without `Node_pressure_constraints` still parse unchanged.

### 3. Translate XML Parameters Into EquationModuleInput

Extend `Code/Source/solver/Application/Translators/EquationTranslator.cpp`.

After the existing equation-level parameter snapshot:

```cpp
if (eq_params.node_pressure_constraints.value_set) {
  svmp::Physics::NodePressureConstraintInput npc{};
  npc.id_type = eq_params.node_pressure_constraints.id_type.value();
  npc.values_file_path = eq_params.node_pressure_constraints.values_file_path.value();
  input.node_pressure_constraints = std::move(npc);
}
```

Validation at translation time:

- `Id_type` must be exactly `Global_vertex_gid` for the first implementation.
- `Values_file_path` must be non-empty.
- Keep path resolution consistent with existing solver file path behavior. If the application already resolves paths relative to the solver XML, use that helper. Otherwise document that the path is relative to the solver working directory.

Checklist:

- [x] Add translation from `eq_params.node_pressure_constraints` into `EquationModuleInput`.
- [x] Validate `Id_type == "Global_vertex_gid"` for the first implementation.
- [x] Validate `Values_file_path` is non-empty.
- [x] Preserve the raw file path or resolve it consistently with existing solver file path behavior.
- [x] Add concise OOP solver diagnostic logging showing that node pressure constraints were detected.
- [x] Keep all behavior unchanged when `node_pressure_constraints.value_set` is false.
- [x] Add or update translator-level tests that construct `EquationParameters` with and without this block.

### 4. Add Poisson/Darcy Option Fields

Extend `Code/Source/solver/Physics/Formulations/Poisson/PoissonModule.h`:

```cpp
enum class NodeIdType {
    GlobalVertexGid,
};

struct NodeDirichletBC {
    FE::GlobalIndex node_id{-1};
    FE::Real value{0.0};
};

struct NodeDirichletBCSet {
    NodeIdType id_type{NodeIdType::GlobalVertexGid};
    std::vector<NodeDirichletBC> values{};
};
```

Add to `PoissonOptions`:

```cpp
NodeDirichletBCSet node_dirichlet{};
```

Use Poisson terminology internally because the module is scalar elliptic. The Darcy-facing input parser can describe these as pressure constraints.

Checklist:

- [x] Add `NodeIdType` to the Poisson formulation namespace.
- [x] Add `NodeDirichletBC` with `node_id` and scalar `value`.
- [x] Add `NodeDirichletBCSet` with `id_type` and `values`.
- [x] Add `node_dirichlet` to `PoissonOptions`.
- [x] Initialize all new fields with safe defaults.
- [x] Ensure the new option names do not conflict with existing face-marker `DirichletBC`.
- [x] Confirm all existing Poisson users compile without setting the new option.

### 5. Parse the CSV File

Add helper functions in `Code/Source/solver/Physics/Formulations/Poisson/PoissonRegister.cpp`, or a small nearby helper file if this starts to grow:

```cpp
NodeIdType parse_node_id_type(std::string_view id_type);
std::vector<PoissonOptions::NodeDirichletBC>
read_node_pressure_csv(const std::string& path);
void apply_node_pressure_constraints(const EquationModuleInput& input,
                                     PoissonOptions& options);
```

CSV rules:

- Accept an optional header row.
- Required columns: `node_id`, `pressure`.
- Allow blank lines and lines starting with `#`.
- Trim whitespace.
- Reject non-integer node ids.
- Reject non-finite pressure values.
- Reject duplicate node ids with conflicting values.
- Allow duplicate node ids if the pressure values match within the same tolerance used for Dirichlet conflicts.

Call `apply_node_pressure_constraints(input, options)` from `create_poisson_from_input()` after `apply_scalar_bcs(input, options)`.

Checklist:

- [x] Implement `parse_node_id_type(std::string_view)` with a clear error for unsupported values.
- [x] Implement `read_node_pressure_csv(const std::string& path)`.
- [x] Accept a header row containing `node_id` and `pressure`.
- [x] Accept a no-header two-column file if desired for convenience.
- [x] Ignore blank lines.
- [x] Ignore comment lines beginning with `#`.
- [x] Trim whitespace around fields.
- [x] Parse node ids as non-negative integer `FE::GlobalIndex` values.
- [x] Parse pressure values as finite `FE::Real` values.
- [x] Reject malformed rows with path and line number in the error message.
- [x] Detect duplicate node ids.
- [x] Accept duplicate node ids only when values match within the selected tolerance.
- [x] Reject duplicate node ids with conflicting values.
- [x] Implement `apply_node_pressure_constraints(input, options)`.
- [x] Call `apply_node_pressure_constraints(input, options)` from `create_poisson_from_input()` after face-based scalar BC parsing.
- [x] Add unit tests for valid, missing, malformed, duplicate, and conflicting CSV inputs.

### 6. Add a Setup-Time Vertex Constraint

Add a new FE constraint:

```text
Code/Source/solver/FE/Constraints/VertexDirichletConstraint.h
Code/Source/solver/FE/Constraints/VertexDirichletConstraint.cpp
```

The class should implement `FE::constraints::ISystemConstraint`, not `FE::constraints::Constraint`, because vertex ids must be lowered after field DOFs are finalized during `FESystem::setup()`.

Proposed constructor:

```cpp
struct VertexDirichletValue {
    GlobalIndex vertex_id{-1};
    Real value{0.0};
};

enum class VertexIdMode {
    GlobalVertexGid,
    LocalVertexId,
};

class VertexDirichletConstraint final : public ISystemConstraint {
public:
    VertexDirichletConstraint(FieldId field,
                              std::vector<VertexDirichletValue> values,
                              VertexIdMode mode);

    void apply(const systems::FESystem& system,
               AffineConstraints& constraints) override;

    bool updateValues(const systems::FESystem& system,
                      AffineConstraints& constraints,
                      double time,
                      double dt) override;

    bool isTimeDependent() const noexcept override;
};
```

Initial implementation details:

- Require a valid scalar field id.
- Require scalar H1/C0 space.
- Require `fieldDofHandler(field).getEntityDofMap()`.
- For `GlobalVertexGid`, resolve the input id through `system.mesh()->base().global_to_local_vertex(gid)` when the concrete mesh is available.
- For ranks that do not store a requested global vertex gid, skip locally.
- Use `EntityDofMap::getVertexDofs(local_vertex)` to get the field-local pressure DOF.
- Add `system.fieldDofOffset(field)` to get the global monolithic DOF.
- Add the constraint only if `system.dofHandler().getPartition().locallyOwned().contains(dof)`.
- Call `constraints.addDirichlet(dof, value)`.

Validation:

- For scalar P1 H1, each constrained vertex should have exactly one vertex DOF.
- For higher-order scalar H1, still constrain only the vertex DOF. Do not interpret high-order edge, face, or cell interpolation points as "nodes" in this feature.
- If a constrained node is also constrained by a boundary face Dirichlet value, let `AffineConstraints::addDirichlet()` detect conflict. Matching values are allowed; conflicting values should fail.
- In MPI, optionally allreduce a "found" flag for each requested global gid so a typo in the CSV fails instead of silently being ignored on all ranks.

Add the new files to `Code/Source/solver/FE/CMakeLists.txt`:

- `Constraints/VertexDirichletConstraint.h`
- `Constraints/VertexDirichletConstraint.cpp`

Checklist:

- [x] Add `Code/Source/solver/FE/Constraints/VertexDirichletConstraint.h`.
- [x] Add `Code/Source/solver/FE/Constraints/VertexDirichletConstraint.cpp`.
- [x] Define `VertexDirichletValue`.
- [x] Define `VertexIdMode` with at least `GlobalVertexGid`.
- [x] Implement `VertexDirichletConstraint` as an `ISystemConstraint`.
- [x] Validate `field != INVALID_FIELD_ID` in the constructor.
- [x] Store a moved copy of the vertex/value list.
- [x] Reject empty or negative vertex ids where appropriate.
- [x] In `apply()`, validate the field is scalar H1/C0.
- [x] In `apply()`, retrieve `system.fieldDofHandler(field).getEntityDofMap()`.
- [x] In `GlobalVertexGid` mode, resolve global vertex gids through `system.mesh()->base().global_to_local_vertex(gid)`.
- [x] Decide and implement fallback behavior if `system.mesh()` is unavailable, such as throwing a clear unsupported-mode error.
- [x] Skip requested global gids that are not stored on the local rank.
- [x] Use `EntityDofMap::getVertexDofs(local_vertex)` to find the pressure field DOF.
- [x] Require exactly one vertex DOF for the scalar H1 pressure target.
- [x] Add `system.fieldDofOffset(field)` to convert field-local DOFs to monolithic DOFs.
- [x] Add the Dirichlet line only when the monolithic DOF is locally owned.
- [x] Let `AffineConstraints::addDirichlet()` detect conflicting constraints.
- [x] Return `false` from `updateValues()` for the first steady implementation.
- [x] Return `false` from `isTimeDependent()`.
- [x] Add MPI allreduce validation so missing global gids fail globally.
- [x] Add the new header and source to `Code/Source/solver/FE/CMakeLists.txt`.
- [x] Add serial unit coverage for scalar H1 vertex constraints.
- [x] Add MPI unit coverage for owner-only insertion and missing global gid detection.

### 7. Lower Poisson Options to the FE Constraint

In `Code/Source/solver/Physics/Formulations/Poisson/PoissonModule.cpp`, after the pressure field is added and before/around boundary condition application:

```cpp
if (!options_.node_dirichlet.values.empty()) {
    std::vector<FE::constraints::VertexDirichletValue> values;
    values.reserve(options_.node_dirichlet.values.size());
    for (const auto& in : options_.node_dirichlet.values) {
        values.push_back({in.node_id, in.value});
    }

    system.addSystemConstraint(
        std::make_unique<FE::constraints::VertexDirichletConstraint>(
            u_id,
            std::move(values),
            FE::constraints::VertexIdMode::GlobalVertexGid));
}
```

This keeps face-marker Dirichlet pressure conditions on the existing `BoundaryConditionManager` path and uses the new setup-time constraint only for vertex pressure values.

Checklist:

- [x] Include `Constraints/VertexDirichletConstraint.h` in `PoissonModule.cpp`.
- [x] After `u_id` is created, check `!options_.node_dirichlet.values.empty()`.
- [x] Convert `PoissonOptions::NodeDirichletBC` values into `FE::constraints::VertexDirichletValue`.
- [x] Map `PoissonOptions::NodeIdType::GlobalVertexGid` to `FE::constraints::VertexIdMode::GlobalVertexGid`.
- [x] Throw a clear error for any unsupported node id mode.
- [x] Register the constraint with `system.addSystemConstraint(...)`.
- [x] Keep existing face-marker Dirichlet handling through `BoundaryConditionManager`.
- [x] Verify ordering does not hide conflicts between boundary face constraints and node constraints.
- [x] Confirm existing Poisson cases without node constraints assemble unchanged.

### 8. Add a Darcy Alias That Reuses Poisson

If the user-facing equation type should be Darcy rather than heat/Poisson, add a factory alias in `Code/Source/solver/Physics/Formulations/Poisson/PoissonRegister.cpp`.

Option A: register a Darcy alias to the same factory:

```cpp
SVMP_REGISTER_EQUATION("darcy", &create_poisson_from_input);
```

Option B: add a tiny wrapper so field/output names are pressure-specific:

```cpp
std::unique_ptr<PhysicsModule>
create_darcy_pressure_from_input(const EquationModuleInput& input,
                                 FE::systems::FESystem& system)
{
  ...
  PoissonOptions options{};
  options.field_name = "Pressure";
  ...
  return std::make_unique<PoissonModule>(space, options);
}

SVMP_REGISTER_EQUATION("darcy", &create_darcy_pressure_from_input);
```

Prefer option B if output variable names matter. The actual module can remain `PoissonModule`.

Checklist:

- [x] Decide the user-facing equation type string, such as `"darcy"`.
- [x] Decide whether plain aliasing is enough or whether a wrapper factory is needed.
- [x] If output names matter, implement `create_darcy_pressure_from_input()`.
- [x] Set `PoissonOptions::field_name = "Pressure"` in the Darcy wrapper.
- [x] Reuse the same H1 space inference used by `create_poisson_from_input()`.
- [x] Reuse scalar material/property parsing or add Darcy-specific aliases for permeability/source naming.
- [x] Reuse `apply_scalar_bcs(input, options)` for face-marker pressure conditions.
- [x] Reuse `apply_node_pressure_constraints(input, options)` for interior known pressures.
- [x] Register the alias with `SVMP_REGISTER_EQUATION`.
- [x] Add a small input example showing Darcy-through-Poisson with the new XML block.

### 9. Testing Plan

Add focused tests in the smallest useful layers.

Parser and translator tests:

- XML with `Node_pressure_constraints` populates `EquationModuleInput`.
- Missing `Values_file_path` fails.
- Unsupported `Id_type` fails.

CSV parser tests:

- Header and no-header files both parse.
- Blank lines and comments are ignored.
- Duplicate matching values are accepted.
- Duplicate conflicting values fail.
- Bad node id and bad pressure fail.

FE constraint tests:

- Serial scalar H1 P1 mesh: global vertex gid maps to one pressure DOF and constrains it to the requested value.
- Boundary zero pressure plus interior nonzero pressure both appear in `AffineConstraints`.
- Boundary zero pressure plus same-node nonzero node pressure fails with a Dirichlet conflict.

Physics integration tests:

- Reuse a simple Poisson manufactured problem and pin one interior vertex through the CSV path.
- Verify the constrained solution entry equals the CSV value.
- Verify normal boundary face Dirichlet conditions still work.

MPI tests:

- CSV contains one global vertex gid owned by rank 0 and one owned by another rank.
- Each constrained DOF is inserted only by its owner.
- A missing global vertex gid fails globally, not only on the local rank that parsed it.

Checklist:

- [x] Add parser tests for the new XML block.
- [x] Add translator tests for `EquationModuleInput` population.
- [x] Add CSV parser tests for valid header and no-header files.
- [x] Add CSV parser tests for blank lines and comments.
- [x] Add CSV parser tests for malformed rows.
- [x] Add CSV parser tests for duplicate matching and duplicate conflicting values.
- [x] Add serial FE constraint tests for scalar H1 P1 vertex constraints.
- [x] Add conflict tests with an existing face-marker Dirichlet constraint.
- [x] Add a physics integration test that pins an interior pressure vertex through the CSV path.
- [x] Add a test that normal face-marker Dirichlet pressure conditions still work.
- [x] Add MPI tests for rank-owned insertion.
- [x] Add MPI tests for missing global vertex gid detection.
- [x] Wire new tests into the relevant CMake test lists.
- [x] Run the focused FE, Physics, and Application tests.
- [x] Run at least one representative existing Poisson/heat case to check regression risk.

### 10. Definition of Done

- The requested XML block is accepted inside the equation input.
- `known_pressure_nodes.csv` can prescribe pressure values at interior mesh vertices.
- Homogeneous surface pressure Dirichlet conditions continue to use existing face-marker `Add_BC` input.
- The implementation works with scalar H1 P1 pressure in serial and MPI.
- Conflicting pressure constraints fail with a clear message.
- The Darcy user-facing path can run through the existing Poisson module without duplicating the scalar elliptic assembly code.

Checklist:

- [x] Example solver XML with `Node_pressure_constraints` parses successfully.
- [x] Example `known_pressure_nodes.csv` is read successfully.
- [x] Interior vertex pressure values are present in the final `AffineConstraints`.
- [x] Boundary face pressure values are still installed through the existing marker-based path.
- [x] Same-node matching constraints are accepted.
- [x] Same-node conflicting constraints fail clearly.
- [x] Serial scalar H1 P1 Darcy-through-Poisson case solves with constrained interior pressure values.
- [x] MPI scalar H1 P1 Darcy-through-Poisson case solves with constrained interior pressure values.
- [x] Missing global vertex gids fail globally in MPI.
- [x] Existing Poisson/heat tests continue to pass.
- [x] Existing face-marker BC behavior is unchanged.
- [x] User-facing documentation includes the XML snippet and CSV format.

Qualification note:

- [x] Targeted Darcy node-pressure parser, translator, FE constraint, serial physics, and MPI physics tests pass.
- [x] `PoissonSquareSteadyMPI.DirichletLeftRight_NeumannTopBottom_Linear_2Ranks` passes after aligning MPI setup rank metadata and backend row ownership assembly policy.
