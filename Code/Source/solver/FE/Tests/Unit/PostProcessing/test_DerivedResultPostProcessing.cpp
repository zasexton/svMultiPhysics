/**
 * @file test_DerivedResultPostProcessing.cpp
 * @brief Unit tests for FE derived-result postprocessing infrastructure.
 */

#include <gtest/gtest.h>

#include "Assembly/AssemblyKernel.h"
#include "Assembly/StandardAssembler.h"
#include "Core/FEException.h"
#include "Dofs/EntityDofMap.h"
#include "Forms/FormExpr.h"
#include "Forms/Vocabulary.h"
#include "Mesh/Core/MeshBase.h"
#include "Mesh/Mesh.h"
#include "Mesh/Topology/CellShape.h"
#include "PostProcessing/DerivedResultBuilder.h"
#include "PostProcessing/DerivedResultOutput.h"
#include "PostProcessing/DerivedResultRegistry.h"
#include "Spaces/H1Space.h"
#include "Systems/FESystem.h"

#include <memory>
#include <span>
#include <vector>

namespace {

using svmp::CellFamily;
using svmp::CellShape;
using svmp::EntityKind;
using svmp::FieldScalarType;
using svmp::Mesh;
using svmp::MeshBase;
using svmp::FE::ElementType;
using svmp::FE::FieldId;
using svmp::FE::InvalidArgumentException;
using svmp::FE::NotImplementedException;
using svmp::FE::Real;
using svmp::FE::forms::FormExpr;
using svmp::FE::post::DerivedResultBuilder;
using svmp::FE::post::DerivedResultDefinition;
using svmp::FE::post::DerivedResultOverwritePolicy;
using svmp::FE::post::DerivedResultPolicy;
using svmp::FE::post::DerivedResultRegistry;
using svmp::FE::post::DerivedResultScope;
using svmp::FE::post::componentCount;
using svmp::FE::post::derivedResultFieldData;
using svmp::FE::post::ensureDerivedResultField;
using svmp::FE::post::meshEntityKind;
using svmp::FE::post::validateDerivedResultDefinition;
using svmp::FE::spaces::H1Space;
using svmp::FE::systems::FEQuantityShape;
using svmp::FE::systems::FESystem;
using svmp::FE::systems::FieldSpec;
using svmp::FE::systems::SystemStateView;

std::shared_ptr<Mesh> buildSingleQuadMesh()
{
    auto base = std::make_shared<MeshBase>();

    const std::vector<svmp::real_t> x_ref = {
        0.0, 0.0,
        1.0, 0.0,
        1.0, 1.0,
        0.0, 1.0
    };
    const std::vector<svmp::offset_t> cell2vertex_offsets = {0, 4};
    const std::vector<svmp::index_t> cell2vertex = {0, 1, 2, 3};

    CellShape shape{};
    shape.family = CellFamily::Quad;
    shape.num_corners = 4;
    shape.order = 1;
    base->build_from_arrays(/*spatial_dim=*/2, x_ref, cell2vertex_offsets, cell2vertex, {shape});
    base->finalize();

    return svmp::create_mesh(std::move(base));
}

DerivedResultDefinition constantCellResult(std::string name, double value)
{
    return DerivedResultBuilder(std::move(name))
        .scope(DerivedResultScope::Cell)
        .policy(DerivedResultPolicy::CellAverage)
        .shape(FEQuantityShape::scalar())
        .expression(FormExpr::constant(value))
        .build();
}

std::shared_ptr<const H1Space> quadH1Space()
{
    return std::make_shared<H1Space>(ElementType::Quad4, /*order=*/1);
}

void setVertexValue(FESystem& system,
                    FieldId field,
                    svmp::FE::GlobalIndex vertex,
                    Real value,
                    std::vector<Real>& u)
{
    const auto* entity_map = system.fieldDofHandler(field).getEntityDofMap();
    ASSERT_NE(entity_map, nullptr);
    const auto vdofs = entity_map->getVertexDofs(vertex);
    ASSERT_FALSE(vdofs.empty());
    const auto dof = vdofs[0] + system.fieldDofOffset(field);
    ASSERT_GE(dof, 0);
    ASSERT_LT(static_cast<std::size_t>(dof), u.size());
    u[static_cast<std::size_t>(dof)] = value;
}

} // namespace

TEST(DerivedResultBuilder, RequiresNameShapeAndExpression)
{
    EXPECT_THROW(DerivedResultBuilder("")
                     .shape(FEQuantityShape::scalar())
                     .expression(FormExpr::constant(1.0))
                     .build(),
                 InvalidArgumentException);

    EXPECT_THROW(DerivedResultBuilder("missing_shape")
                     .expression(FormExpr::constant(1.0))
                     .build(),
                 InvalidArgumentException);

    EXPECT_THROW(DerivedResultBuilder("missing_expression")
                     .shape(FEQuantityShape::scalar())
                     .build(),
                 InvalidArgumentException);

    auto def = constantCellResult("constant", 2.0);
    EXPECT_EQ(def.name, "constant");
    EXPECT_EQ(def.scope, DerivedResultScope::Cell);
    EXPECT_EQ(def.policy, DerivedResultPolicy::CellAverage);
    EXPECT_EQ(componentCount(def.shape), 1u);
    EXPECT_TRUE(def.expression.isValid());
    EXPECT_TRUE(def.enabled);
}

TEST(DerivedResultRegistry, RejectsDuplicatesAndPreservesOrder)
{
    DerivedResultRegistry registry;
    const auto first = registry.registerDefinition(constantCellResult("first", 1.0));
    const auto second = registry.registerDefinition(constantCellResult("second", 2.0));

    EXPECT_TRUE(first.valid());
    EXPECT_TRUE(second.valid());
    EXPECT_TRUE(registry.contains("first"));
    EXPECT_TRUE(registry.contains("second"));
    EXPECT_EQ(registry.get(first).name, "first");
    EXPECT_EQ(registry.get("second").name, "second");
    ASSERT_EQ(registry.all().size(), 2u);
    EXPECT_EQ(registry.all()[0].name, "first");
    EXPECT_EQ(registry.all()[1].name, "second");

    EXPECT_THROW(registry.registerDefinition(constantCellResult("first", 3.0)),
                 InvalidArgumentException);
}

TEST(DerivedResultValidation, RejectsInvalidDefinitions)
{
    auto space = quadH1Space();
    const auto field = static_cast<FieldId>(7);

    auto invalid_name = constantCellResult("bad name", 1.0);
    EXPECT_THROW(validateDerivedResultDefinition(invalid_name), InvalidArgumentException);

    DerivedResultDefinition invalid_expr;
    invalid_expr.name = "invalid_expr";
    invalid_expr.shape = FEQuantityShape::scalar();
    EXPECT_THROW(validateDerivedResultDefinition(invalid_expr), InvalidArgumentException);

    auto test_fn = DerivedResultBuilder("test_fn")
        .scope(DerivedResultScope::Cell)
        .policy(DerivedResultPolicy::CellAverage)
        .shape(FEQuantityShape::scalar())
        .expression(FormExpr::testFunction(*space, "v"))
        .build();
    EXPECT_THROW(validateDerivedResultDefinition(test_fn), InvalidArgumentException);

    auto trial_fn = DerivedResultBuilder("trial_fn")
        .scope(DerivedResultScope::Cell)
        .policy(DerivedResultPolicy::CellAverage)
        .shape(FEQuantityShape::scalar())
        .expression(FormExpr::trialFunction(*space, "u"))
        .build();
    EXPECT_THROW(validateDerivedResultDefinition(trial_fn), InvalidArgumentException);

    auto integral = DerivedResultBuilder("integral")
        .scope(DerivedResultScope::Cell)
        .policy(DerivedResultPolicy::CellAverage)
        .shape(FEQuantityShape::scalar())
        .expression(FormExpr::constant(1.0).dx())
        .build();
    EXPECT_THROW(validateDerivedResultDefinition(integral), InvalidArgumentException);

    auto auxiliary = DerivedResultBuilder("auxiliary")
        .scope(DerivedResultScope::Cell)
        .policy(DerivedResultPolicy::CellAverage)
        .shape(FEQuantityShape::scalar())
        .expression(FormExpr::auxiliaryInput("unsupported_runtime_value"))
        .build();
    EXPECT_THROW(validateDerivedResultDefinition(auxiliary), InvalidArgumentException);

    auto boundary_face = DerivedResultBuilder("boundary_face")
        .scope(DerivedResultScope::BoundaryFace)
        .policy(DerivedResultPolicy::FaceAverage)
        .shape(FEQuantityShape::scalar())
        .expression(FormExpr::constant(1.0))
        .build();
    EXPECT_THROW(validateDerivedResultDefinition(boundary_face), InvalidArgumentException);

    auto incompatible = DerivedResultBuilder("incompatible")
        .scope(DerivedResultScope::Vertex)
        .policy(DerivedResultPolicy::CellAverage)
        .shape(FEQuantityShape::scalar())
        .expression(FormExpr::constant(1.0))
        .build();
    EXPECT_THROW(validateDerivedResultDefinition(incompatible), InvalidArgumentException);

    const auto pressure = FormExpr::stateField(field, *space, "p");
    auto raw_vertex_gradient = DerivedResultBuilder("raw_vertex_gradient")
        .scope(DerivedResultScope::Vertex)
        .policy(DerivedResultPolicy::PointValue)
        .shape(FEQuantityShape::vector(2))
        .expression(svmp::FE::forms::grad(pressure))
        .build();
    EXPECT_THROW(validateDerivedResultDefinition(raw_vertex_gradient), InvalidArgumentException);
}

TEST(DerivedResultValidation, AllowsFutureScopesButEvaluatorCanRejectThem)
{
    DerivedResultRegistry registry;
    auto face_result = DerivedResultBuilder("face_average")
        .scope(DerivedResultScope::Face)
        .policy(DerivedResultPolicy::FaceAverage)
        .shape(FEQuantityShape::scalar())
        .expression(FormExpr::constant(1.0))
        .build();

    EXPECT_NO_THROW(registry.registerDefinition(std::move(face_result)));
    ASSERT_EQ(registry.all().size(), 1u);
    EXPECT_EQ(registry.all()[0].scope, DerivedResultScope::Face);
}

TEST(DerivedResultTypes, MapsDirectMeshFieldScopes)
{
    ASSERT_TRUE(meshEntityKind(DerivedResultScope::Vertex).has_value());
    EXPECT_EQ(*meshEntityKind(DerivedResultScope::Vertex), EntityKind::Vertex);
    ASSERT_TRUE(meshEntityKind(DerivedResultScope::Cell).has_value());
    EXPECT_EQ(*meshEntityKind(DerivedResultScope::Cell), EntityKind::Volume);
    EXPECT_FALSE(meshEntityKind(DerivedResultScope::QuadraturePoint).has_value());
}

TEST(DerivedResultOutput, CreatesAndValidatesMeshFields)
{
    auto mesh = buildSingleQuadMesh();
    auto& local_mesh = mesh->local_mesh();

    const auto vertex_handle = ensureDerivedResultField(local_mesh,
                                                        EntityKind::Vertex,
                                                        "derived_vertex",
                                                        2,
                                                        DerivedResultOverwritePolicy::Reject);
    EXPECT_TRUE(local_mesh.has_field(EntityKind::Vertex, "derived_vertex"));
    EXPECT_EQ(local_mesh.field_type(vertex_handle), FieldScalarType::Float64);
    EXPECT_EQ(local_mesh.field_components(vertex_handle), 2u);
    EXPECT_EQ(local_mesh.field_entity_count(vertex_handle), local_mesh.n_vertices());

    auto vertex_data = derivedResultFieldData(local_mesh, vertex_handle, 2);
    ASSERT_EQ(vertex_data.size(), local_mesh.n_vertices() * 2u);
    for (const double value : vertex_data) {
        EXPECT_EQ(value, 0.0);
    }

    const auto cell_handle = ensureDerivedResultField(local_mesh,
                                                      EntityKind::Volume,
                                                      "derived_cell",
                                                      1,
                                                      DerivedResultOverwritePolicy::Reject);
    EXPECT_EQ(local_mesh.field_components(cell_handle), 1u);
    EXPECT_EQ(local_mesh.field_entity_count(cell_handle), local_mesh.n_cells());
}

TEST(DerivedResultOutput, HandlesIncompatibleExistingFields)
{
    auto mesh = buildSingleQuadMesh();
    auto& local_mesh = mesh->local_mesh();
    (void)local_mesh.attach_field(EntityKind::Vertex,
                                  "existing",
                                  FieldScalarType::Float64,
                                  1);

    EXPECT_THROW(ensureDerivedResultField(local_mesh,
                                          EntityKind::Vertex,
                                          "existing",
                                          2,
                                          DerivedResultOverwritePolicy::Reject),
                 InvalidArgumentException);
    EXPECT_THROW(ensureDerivedResultField(local_mesh,
                                          EntityKind::Vertex,
                                          "existing",
                                          2,
                                          DerivedResultOverwritePolicy::ReplaceCompatible),
                 InvalidArgumentException);

    const auto replacement = ensureDerivedResultField(local_mesh,
                                                      EntityKind::Vertex,
                                                      "existing",
                                                      2,
                                                      DerivedResultOverwritePolicy::ReplaceAny);
    EXPECT_EQ(local_mesh.field_components(replacement), 2u);
}

TEST(FESystemDerivedResults, RegistersDefinitionsThroughSystem)
{
    auto mesh = buildSingleQuadMesh();
    FESystem system(mesh);

    const auto handle = system.addDerivedResult(constantCellResult("system_constant", 5.0));
    EXPECT_TRUE(handle.valid());
    ASSERT_EQ(system.derivedResults().size(), 1u);
    EXPECT_EQ(system.derivedResults()[0].name, "system_constant");

    EXPECT_THROW(system.addDerivedResult(constantCellResult("system_constant", 6.0)),
                 InvalidArgumentException);
}

TEST(FESystemDerivedResults, NoRegisteredResultsIsNoOp)
{
    auto mesh = buildSingleQuadMesh();
    FESystem system(mesh);
    SystemStateView state;

    EXPECT_NO_THROW(system.appendDerivedResultFields(mesh->local_mesh(), state));
    EXPECT_FALSE(mesh->local_mesh().has_field(EntityKind::Volume, "anything"));
}

TEST(FESystemDerivedResults, EvaluatesCellConstantExpression)
{
    auto mesh = buildSingleQuadMesh();
    auto space = quadH1Space();
    FESystem system(mesh);
    const auto pressure = system.addField(FieldSpec{.name = "Pressure", .space = space, .components = 1});
    system.addOperator("mass");
    system.addCellKernel("mass", pressure, std::make_shared<svmp::FE::assembly::MassKernel>(1.0));
    system.setup();

    system.addDerivedResult(constantCellResult("cell_constant", 2.5));
    SystemStateView state;
    std::vector<Real> u(static_cast<std::size_t>(system.dofHandler().getNumDofs()), 0.0);
    state.u = std::span<const Real>(u.data(), u.size());

    system.appendDerivedResultFields(mesh->local_mesh(), state);

    const auto handle = mesh->local_mesh().field_handle(EntityKind::Volume, "cell_constant");
    const auto* data = mesh->local_mesh().field_data_as<double>(handle);
    ASSERT_NE(data, nullptr);
    ASSERT_EQ(mesh->local_mesh().field_entity_count(handle), 1u);
    EXPECT_NEAR(data[0], 2.5, 1e-12);
}

TEST(FESystemDerivedResults, EvaluatesCellCentroidVectorExpression)
{
    auto mesh = buildSingleQuadMesh();
    auto space = quadH1Space();
    FESystem system(mesh);
    const auto pressure = system.addField(FieldSpec{.name = "Pressure", .space = space, .components = 1});
    system.addOperator("mass");
    system.addCellKernel("mass", pressure, std::make_shared<svmp::FE::assembly::MassKernel>(1.0));
    system.setup();

    auto vector_result = DerivedResultBuilder("centroid_vector")
        .scope(DerivedResultScope::Cell)
        .policy(DerivedResultPolicy::CellCentroid)
        .shape(FEQuantityShape::vector(2))
        .expression(FormExpr::asVector({FormExpr::constant(1.25), FormExpr::constant(-3.5)}))
        .build();
    system.addDerivedResult(std::move(vector_result));

    std::vector<Real> u(static_cast<std::size_t>(system.dofHandler().getNumDofs()), 0.0);
    SystemStateView state;
    state.u = std::span<const Real>(u.data(), u.size());
    system.appendDerivedResultFields(mesh->local_mesh(), state);

    const auto handle = mesh->local_mesh().field_handle(EntityKind::Volume, "centroid_vector");
    EXPECT_EQ(mesh->local_mesh().field_components(handle), 2u);
    const auto* data = mesh->local_mesh().field_data_as<double>(handle);
    ASSERT_NE(data, nullptr);
    EXPECT_NEAR(data[0], 1.25, 1e-12);
    EXPECT_NEAR(data[1], -3.5, 1e-12);
}

TEST(FESystemDerivedResults, EvaluatesDarcyStyleGradientExpression)
{
    auto mesh = buildSingleQuadMesh();
    auto space = quadH1Space();
    FESystem system(mesh);
    const auto pressure = system.addField(FieldSpec{.name = "Pressure", .space = space, .components = 1});
    system.addOperator("mass");
    system.addCellKernel("mass", pressure, std::make_shared<svmp::FE::assembly::MassKernel>(1.0));
    system.setup();

    const auto p = FormExpr::stateField(pressure, *space, "Pressure");
    auto darcy_flux = DerivedResultBuilder("Darcy_flux")
        .scope(DerivedResultScope::Cell)
        .policy(DerivedResultPolicy::CellAverage)
        .shape(FEQuantityShape::vector(2))
        .expression(-2.0 * svmp::FE::forms::grad(p))
        .build();
    system.addDerivedResult(std::move(darcy_flux));

    // p(x,y) = 2x + 3y on the unit quad, so -2 grad(p) = (-4, -6).
    std::vector<Real> u(static_cast<std::size_t>(system.dofHandler().getNumDofs()), 0.0);
    setVertexValue(system, pressure, 0, 0.0, u);
    setVertexValue(system, pressure, 1, 2.0, u);
    setVertexValue(system, pressure, 2, 5.0, u);
    setVertexValue(system, pressure, 3, 3.0, u);

    SystemStateView state;
    state.u = std::span<const Real>(u.data(), u.size());
    system.appendDerivedResultFields(mesh->local_mesh(), state);

    const auto handle = mesh->local_mesh().field_handle(EntityKind::Volume, "Darcy_flux");
    EXPECT_EQ(mesh->local_mesh().field_components(handle), 2u);
    EXPECT_EQ(mesh->local_mesh().field_entity_count(handle), 1u);

    const auto* data = mesh->local_mesh().field_data_as<double>(handle);
    ASSERT_NE(data, nullptr);
    EXPECT_NEAR(data[0], -4.0, 1e-12);
    EXPECT_NEAR(data[1], -6.0, 1e-12);
}

TEST(FESystemDerivedResults, EvaluatesPatchAverageRecoveredGradient)
{
    auto mesh = buildSingleQuadMesh();
    auto space = quadH1Space();
    FESystem system(mesh);
    const auto pressure = system.addField(FieldSpec{.name = "Pressure", .space = space, .components = 1});
    system.addOperator("mass");
    system.addCellKernel("mass", pressure, std::make_shared<svmp::FE::assembly::MassKernel>(1.0));
    system.setup();

    const auto p = FormExpr::stateField(pressure, *space, "Pressure");
    auto recovered_gradient = DerivedResultBuilder("Recovered_grad_p")
        .scope(DerivedResultScope::Vertex)
        .policy(DerivedResultPolicy::PatchAverage)
        .shape(FEQuantityShape::vector(2))
        .expression(svmp::FE::forms::grad(p))
        .build();
    system.addDerivedResult(std::move(recovered_gradient));

    std::vector<Real> u(static_cast<std::size_t>(system.dofHandler().getNumDofs()), 0.0);
    setVertexValue(system, pressure, 0, 0.0, u);
    setVertexValue(system, pressure, 1, 2.0, u);
    setVertexValue(system, pressure, 2, 5.0, u);
    setVertexValue(system, pressure, 3, 3.0, u);

    SystemStateView state;
    state.u = std::span<const Real>(u.data(), u.size());
    system.appendDerivedResultFields(mesh->local_mesh(), state);

    const auto handle = mesh->local_mesh().field_handle(EntityKind::Vertex, "Recovered_grad_p");
    EXPECT_EQ(mesh->local_mesh().field_components(handle), 2u);
    EXPECT_EQ(mesh->local_mesh().field_entity_count(handle), 4u);

    const auto* data = mesh->local_mesh().field_data_as<double>(handle);
    ASSERT_NE(data, nullptr);
    for (std::size_t v = 0; v < 4; ++v) {
        EXPECT_NEAR(data[2 * v], 2.0, 1e-12);
        EXPECT_NEAR(data[2 * v + 1], 3.0, 1e-12);
    }
}

TEST(FESystemDerivedResults, EvaluatesVertexPointValueExpression)
{
    auto mesh = buildSingleQuadMesh();
    auto space = quadH1Space();
    FESystem system(mesh);
    const auto pressure = system.addField(FieldSpec{.name = "Pressure", .space = space, .components = 1});
    system.addOperator("mass");
    system.addCellKernel("mass", pressure, std::make_shared<svmp::FE::assembly::MassKernel>(1.0));
    system.setup();

    const auto p = FormExpr::stateField(pressure, *space, "Pressure");
    auto pressure_copy = DerivedResultBuilder("Pressure_copy")
        .scope(DerivedResultScope::Vertex)
        .policy(DerivedResultPolicy::PointValue)
        .shape(FEQuantityShape::scalar())
        .expression(p)
        .build();
    system.addDerivedResult(std::move(pressure_copy));

    std::vector<Real> u(static_cast<std::size_t>(system.dofHandler().getNumDofs()), 0.0);
    setVertexValue(system, pressure, 0, 1.0, u);
    setVertexValue(system, pressure, 1, 2.0, u);
    setVertexValue(system, pressure, 2, 4.0, u);
    setVertexValue(system, pressure, 3, 8.0, u);

    SystemStateView state;
    state.u = std::span<const Real>(u.data(), u.size());
    system.appendDerivedResultFields(mesh->local_mesh(), state);

    const auto handle = mesh->local_mesh().field_handle(EntityKind::Vertex, "Pressure_copy");
    EXPECT_EQ(mesh->local_mesh().field_components(handle), 1u);
    EXPECT_EQ(mesh->local_mesh().field_entity_count(handle), 4u);

    const auto* data = mesh->local_mesh().field_data_as<double>(handle);
    ASSERT_NE(data, nullptr);
    EXPECT_NEAR(data[0], 1.0, 1e-12);
    EXPECT_NEAR(data[1], 2.0, 1e-12);
    EXPECT_NEAR(data[2], 4.0, 1e-12);
    EXPECT_NEAR(data[3], 8.0, 1e-12);
}

TEST(FESystemDerivedResults, UnsupportedScopeFailsClearlyDuringEvaluation)
{
    auto mesh = buildSingleQuadMesh();
    auto space = quadH1Space();
    FESystem system(mesh);
    const auto pressure = system.addField(FieldSpec{.name = "Pressure", .space = space, .components = 1});
    system.addOperator("mass");
    system.addCellKernel("mass", pressure, std::make_shared<svmp::FE::assembly::MassKernel>(1.0));
    system.setup();

    auto face_result = DerivedResultBuilder("face_average")
        .scope(DerivedResultScope::Face)
        .policy(DerivedResultPolicy::FaceAverage)
        .shape(FEQuantityShape::scalar())
        .expression(FormExpr::constant(1.0))
        .build();
    system.addDerivedResult(std::move(face_result));

    std::vector<Real> u(static_cast<std::size_t>(system.dofHandler().getNumDofs()), 0.0);
    SystemStateView state;
    state.u = std::span<const Real>(u.data(), u.size());

    EXPECT_THROW(system.appendDerivedResultFields(mesh->local_mesh(), state),
                 NotImplementedException);
}
