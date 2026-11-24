# FE Derived Result Postprocessing

FE derived result postprocessing lets physics modules register output-only fields
using the same `FE::forms::FormExpr` vocabulary used for formulations. The FE
library owns registration, validation, expression evaluation, and mesh-field
attachment. Physics modules own the physical meaning and naming of each derived
field.

XML user-defined derived expressions are intentionally not supported yet. Initial
registrations are C++ helpers colocated with each physics formulation.

## Ownership Split

- `Physics/Formulations/.../*PostProcessing.h` defines physics-specific helpers.
- `FE/PostProcessing` stores physics-agnostic derived result definitions.
- `FESystem` owns the registry and appends registered results during output.
- `Application` only asks FE to append derived fields before mesh saving.

The FE layer must not depend on `Physics/`, Darcy-specific names, or VTK-specific
semantics beyond writing through mesh fields.

## Darcy Flux Example

```cpp
using namespace svmp::FE::forms;
using namespace svmp::FE::post;

auto p = StateField(pressure_id, pressure_space, "Pressure");
auto K = FormExpr::constant(diffusion);

system.addDerivedResult(
    DerivedResultBuilder("Darcy_flux")
        .scope(DerivedResultScope::Cell)
        .policy(DerivedResultPolicy::CellAverage)
        .shape(FE::systems::FEQuantityShape::vector(pressure_space.topological_dimension()))
        .expression(-K * grad(p))
        .build());
```

For H1 pressure, `grad(Pressure)` is generally discontinuous across cell
interfaces. The natural initial output is therefore cell data, not point data.
The Darcy physics path also registers `Darcy_flux_node` as
`Vertex/PatchAverage`, which averages adjacent cell fluxes to vertices for
visualization-oriented point data.

## Scalar And Vector Examples

```cpp
system.addDerivedResult(
    DerivedResultBuilder("Temperature_squared")
        .scope(DerivedResultScope::Vertex)
        .policy(DerivedResultPolicy::PointValue)
        .shape(FE::systems::FEQuantityShape::scalar())
        .expression(T * T)
        .build());
```

```cpp
system.addDerivedResult(
    DerivedResultBuilder("Cell_body_force")
        .scope(DerivedResultScope::Cell)
        .policy(DerivedResultPolicy::CellAverage)
        .shape(FE::systems::FEQuantityShape::vector(3))
        .expression(FormExpr::asVector({fx, fy, fz}))
        .build());
```

## Scopes

- `Vertex`: writes to mesh vertex fields and VTK `PointData`.
- `Cell`: writes to mesh volume fields and VTK `CellData`.
- `Face` and `BoundaryFace`: registered and validated, evaluator support is
  deferred until face-field output semantics are finalized.
- `Edge`: registered and validated, evaluator support is deferred.
- `QuadraturePoint`: not a direct mesh-field scope; values must later be
  projected or stored through a dedicated quadrature layout.

## Policies

- `PointValue`: direct vertex evaluation for continuous expressions.
- `PatchAverage`: cell-derived values recovered to adjacent vertices for
  visualization.
- `CellCentroid`: one evaluation at the cell centroid.
- `CellAverage`: quadrature-weighted cell average.
- `FaceCentroid`, `FaceAverage`, `EdgeAverage`, `QuadratureValue`,
  `ProjectToCell`, and `ProjectToVertex`: reserved for future evaluator support.

Raw `Vertex/PointValue` output rejects differential operators such as
`grad(H1 field)`. Use `Cell/CellAverage` for honest cell fluxes, or
`Vertex/PatchAverage` for visualization-oriented recovered gradients.

## Output

The application output path evaluates primary fields first, appends FE derived
result fields, and then saves the mesh. `Cell/CellAverage` and
`Vertex/PatchAverage` derived expressions are evaluated through FE functional
kernels, which request LLVM JIT acceleration by default when the FE library is
built with JIT support. During one output pass, repeated `CellAverage` and
`PatchAverage` requests for the same expression, shape, marker, and referenced
field set share one computed cell-average buffer so cell and recovered vertex
outputs do not repeat the same expression sweep. Existing VTK writers emit
direct mesh fields as point or cell data:

```text
PointData:
  primary fields
  vertex-scope derived fields

CellData:
  GlobalCellID
  cell-scope derived fields
```

Unsupported scopes fail with a diagnostic naming the derived result, scope, and
policy.

## Scalar History

Scalars worth a time series (a quantity, a scalar global field, an auxiliary
input, a state or output of a 0D model) are recorded on the FE system with
`post::record`, `post::recordAuxiliaryState` and `post::recordAuxiliaryOutput`
(`FE/PostProcessing/ScalarHistory.h`), or `F.record` in a formulation builder.
Recording installs nothing. The Navier-Stokes RCR, RCRCR and resistance
outlets record the flow rate `Q`, their states (`X`; `P1`, `P2`) and `P_out`;
the Poisson outlets record `Q`, `X` and their flux.

A deck writes the history with one general key (decision D64, item 5):

```xml
<GeneralSimulationParameters>
  <Write_scalar_history> true </Write_scalar_history>
</GeneralSimulationParameters>
```

Off by default. With the key on, the new solver writes `scalar_history.csv` in
the results folder (`Save_results_in_folder`, else the working directory): a
header `step,time,<column>,...` with columns named `<instance>.<value>` (for
example `rcr_windkessel_b3.P_out`), then one row per accepted time step (one
row, step 1, for a steady solve). Each number is the shortest decimal that
reads back to the same double; quantities are summed exactly and the other
values are those of rank 0, so for the same states the file is the same at
any rank count. Rank 0 writes it.

Each kind of column samples its own stage of the accepted step:

| Column | Value written for step n+1 |
|---|---|
| quantity (`post::record(system, quantity)`) | recomputed from the accepted state u_{n+1} |
| global field | its value in the accepted state u_{n+1} |
| auxiliary input (the outlets' `Q`) | the input's cache: its value at the last refresh of the step; for an input refreshed at every nonlinear iteration, the stage state of the last residual evaluation (backward Euler: u_{n+1}; first-order generalized alpha: u_n + alpha_f (u_{n+1} - u_n)) |
| auxiliary state (`X`, `P1`, `P2`) | the committed state after the step |
| auxiliary output (`P_out`, the Poisson flux) | the output at the committed state with the cached inputs |

So for an RCR outlet `P_out = X + Rp Q` holds row by row with the written
`Q`, while under generalized alpha that `Q` is the flow of the stage state,
not of u_{n+1}; a quantity of the same integrand gives the flow of u_{n+1}
(`test_GlobalMultiplier.cpp`,
`AGeneralizedAlphaOutletHistorySamplesEachColumnAtItsStage`).

## Deferred Scope Design Notes

`Face` fields should attach to `EntityKind::Face` and initially remain mesh
fields. A VTK writer extension should choose between writing a face sidecar mesh
or adding an explicit face-output file because ordinary unstructured-grid cell
data cannot represent both volume-cell and face-cell tuples in one array.

`BoundaryFace` is a face subset selected by marker. Evaluation should skip faces
whose `boundary_label` does not match the registered marker. Interior faces do
not participate in `BoundaryFace`; `Face` applies to all local faces unless a
future region selector is added.

Distributed face and edge fields should follow the mesh ownership model: owned
entities are authoritative, ghost entities may be filled for visualization, and
writers should either emit owned-only sidecar entities or synchronize ghost
values before output. Interior faces shared by ranks need a deterministic owner
from the distributed mesh topology.

`Edge` fields should attach to `EntityKind::Edge` and follow the same owned-plus
ghost rule as face fields. For simplex and tensor-product elements, edge
averages should use edge quadrature in physical coordinates.

`QuadraturePoint` results should not be attached directly to mesh fields. They
need an FE-owned storage layout keyed by cell, quadrature rule, and component.
Ordinary VTK output should use `ProjectToCell` or `ProjectToVertex` until a
dedicated quadrature output format is added.

## Adding A New Physics Derived Result

1. Add a formulation-local `*PostProcessing.h` helper.
2. Build the expression from `FormExpr` terminals such as `StateField`.
3. Choose the output scope based on mathematical continuity.
4. Choose a compatible policy.
5. Declare the result shape explicitly.
6. Register the helper from the module `registerOn()` path.
7. Add a physics test that verifies the field appears with the expected mesh
   association and values.
