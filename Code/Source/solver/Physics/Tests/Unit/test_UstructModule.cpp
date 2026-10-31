/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#include <gtest/gtest.h>

#include "Physics/Formulations/Ustruct/UstructBCFactories.h"
#include "Physics/Formulations/Ustruct/UstructModule.h"
#include "Physics/Materials/Solid/IsochoricNeoHookeanPK1.h"
#include "Physics/Tests/Unit/PhysicsTestHelpers.h"

#include "FE/Forms/JIT/LLVMJITBuildInfo.h"
#include "FE/Assembly/GlobalSystemView.h"
#include "FE/Dofs/EntityDofMap.h"
#include "FE/Forms/PointEvaluator.h"
#include "FE/Spaces/SpaceFactory.h"
#include "FE/Systems/FESystem.h"
#include "FE/Systems/TimeIntegrator.h"
#include "FE/Systems/TransientSystem.h"
#include "FE/Tests/Unit/Forms/FormsTestHelpers.h"

#include <array>
#include <memory>
#include <span>
#include <stdexcept>
#include <string_view>
#include <utility>
#include <vector>

namespace svmp {
namespace Physics {
namespace test {

namespace {

[[nodiscard]] FE::GlobalIndex vertexComponentDof(const FE::systems::FESystem& system,
                                                 FE::FieldId field,
                                                 FE::GlobalIndex vertex,
                                                 int component)
{
    const auto* entity_map = system.fieldDofHandler(field).getEntityDofMap();
    if (!entity_map) {
        throw std::runtime_error("field has no entity DOF map");
    }
    const auto dofs = entity_map->getVertexDofs(vertex);
    if (component < 0 || static_cast<std::size_t>(component) >= dofs.size()) {
        throw std::runtime_error("invalid vertex component");
    }
    return system.fieldDofOffset(field) + dofs[static_cast<std::size_t>(component)];
}

const FE::post::DerivedResultDefinition&
findDerivedResult(const FE::systems::FESystem& system, std::string_view name)
{
    for (const auto& result : system.derivedResults()) {
        if (result.name == name) {
            return result;
        }
    }
    throw std::runtime_error("missing derived result");
}

} // namespace

TEST(UstructModule, RejectsDuplicateUnknownFieldBeforeFurtherRegistration)
{
    auto mesh = std::make_shared<SingleTetraMeshAccess>();
    auto displacement_space = FE::spaces::VectorSpace(
        FE::spaces::SpaceType::H1, FE::ElementType::Tetra4, 1, 3);
    auto pressure_space = FE::spaces::SpaceFactory::create_h1(FE::ElementType::Tetra4, 1);
    for (const bool transient : {false, true}) {
        FE::systems::FESystem system(mesh);
        const auto existing = system.addField(
            {.name = "custom_displacement", .space = displacement_space, .components = 3});
        formulations::ustruct::UstructOptions options;
        options.displacement_field_name = "custom_displacement";
        options.enable_time_derivative_terms = transient;
        options.deviatoric_pk1_model =
            materials::solid::isochoricNeoHookeanPK1(3.0).model();
        formulations::ustruct::UstructModule module(displacement_space, pressure_space, options);
        EXPECT_THROW(module.registerOn(system), FE::InvalidArgumentException);
        EXPECT_EQ(system.findFieldByName(options.displacement_field_name), existing);
        EXPECT_EQ(system.registeredFieldCount(), 1u);
        EXPECT_FALSE(system.hasOperator("equations"));
        EXPECT_TRUE(system.formulationRecords().empty());
        EXPECT_TRUE(system.derivedResults().empty());
        EXPECT_TRUE(system.boundaryConditionDescriptors().empty());
    }
}

TEST(UstructModule, DuplicateOperatorPreservesOrderedFieldsBeforePostprocessing)
{
    auto mesh = std::make_shared<SingleTetraMeshAccess>();
    auto displacement_space = FE::spaces::VectorSpace(
        FE::spaces::SpaceType::H1, FE::ElementType::Tetra4, 1, 3);
    auto pressure_space = FE::spaces::SpaceFactory::create_h1(FE::ElementType::Tetra4, 1);
    for (const bool transient : {false, true}) {
        SCOPED_TRACE(transient);
        FE::systems::FESystem system(mesh);
        formulations::ustruct::UstructOptions options;
        options.enable_time_derivative_terms = transient;
        options.deviatoric_pk1_model =
            materials::solid::isochoricNeoHookeanPK1(3.0).model();
        options.ct_m = options.ct_c = 0.0;
        formulations::ustruct::UstructModule first(displacement_space, pressure_space, options);
        first.registerOn(system);
        const auto field_count = system.registeredFieldCount();
        const auto derived_results = system.derivedResults();
        const auto cells = system.operatorDefinition("equations").cells;
        ASSERT_FALSE(cells.empty());
        options.displacement_field_name = "second_displacement";
        options.velocity_field_name = "second_velocity";
        options.pressure_field_name = "second_pressure";
        options.register_def_grad_output = true;
        formulations::ustruct::UstructModule second(displacement_space, pressure_space, options);
        try {
            second.registerOn(system);
            FAIL() << "expected duplicate operator";
        } catch (const FE::InvalidArgumentException& error) {
            EXPECT_NE(std::string(error.what()).find("operator 'equations' already exists"), std::string::npos);
        }
        EXPECT_EQ(system.findFieldByName(options.displacement_field_name), field_count);
        EXPECT_EQ(system.findFieldByName(options.pressure_field_name), field_count + (transient ? 2 : 1));
        EXPECT_EQ(system.findFieldByName(options.velocity_field_name),
                  transient ? static_cast<FE::FieldId>(field_count + 1) : FE::INVALID_FIELD_ID);
        EXPECT_EQ(system.registeredFieldCount(), 2 * field_count);
        ASSERT_EQ(system.derivedResults().size(), derived_results.size());
        for (std::size_t i = 0; i < derived_results.size(); ++i) {
            EXPECT_EQ(system.derivedResults()[i].name, derived_results[i].name);
            EXPECT_EQ(system.derivedResults()[i].referenced_fields, derived_results[i].referenced_fields);
        }
        EXPECT_TRUE(system.boundaryConditionDescriptors().empty());
        EXPECT_EQ(system.formulationRecords().size(), 1u);
        const auto& after = system.operatorDefinition("equations").cells;
        ASSERT_EQ(after.size(), cells.size());
        for (std::size_t i = 0; i < cells.size(); ++i) {
            EXPECT_EQ(after[i].kernel, cells[i].kernel);
        }
    }
}

TEST(UstructModule, ExplicitUnavailableJITFailsAfterBoundaryWork)
{
    if (FE::forms::jit::llvmJITEnabled()) {
        GTEST_SKIP() << "requires a build without LLVM JIT";
    }
    auto mesh = std::make_shared<SingleTetraMeshAccess>();
    FE::systems::FESystem system(mesh);
    auto displacement_space = FE::spaces::VectorSpace(
        FE::spaces::SpaceType::H1, FE::ElementType::Tetra4, 1, 3);
    auto pressure_space = FE::spaces::SpaceFactory::create_h1(FE::ElementType::Tetra4, 1);
    formulations::ustruct::UstructOptions options;
    options.deviatoric_pk1_model =
        materials::solid::isochoricNeoHookeanPK1(3.0).model();
    options.ct_m = options.ct_c = 0.0;
    options.jit_policy.enable = true;
    options.jit_policy.enable_was_explicitly_set = true;
    options.displacement_dirichlet.push_back({
        .boundary_marker = 17, .value = {FE::Real{0.0}, FE::Real{0.0}, FE::Real{0.0}}});
    formulations::ustruct::UstructModule module(displacement_space, pressure_space, options);
    EXPECT_THROW(module.registerOn(system), std::runtime_error);
    EXPECT_EQ(system.registeredFieldCount(), 3u);
    EXPECT_TRUE(system.hasOperator("equations"));
    const auto& descriptors = system.boundaryConditionDescriptors();
    ASSERT_EQ(descriptors.size(), 6u);
    const std::array<FE::FieldId, 2> constrained_fields{
        system.findFieldByName(options.displacement_field_name),
        system.findFieldByName(options.velocity_field_name)};
    for (std::size_t field = 0; field < constrained_fields.size(); ++field) {
        ASSERT_NE(constrained_fields[field], FE::INVALID_FIELD_ID);
        for (int component = 0; component < 3; ++component) {
            const auto& descriptor = descriptors[3 * field + static_cast<std::size_t>(component)];
            EXPECT_EQ(descriptor.primary_variable,
                      FE::analysis::VariableKey::field(constrained_fields[field], component));
            EXPECT_EQ(descriptor.component, component);
            EXPECT_EQ(descriptor.boundary_marker, 17);
            EXPECT_EQ(descriptor.trace_kind, FE::analysis::TraceKind::Value);
            EXPECT_EQ(descriptor.enforcement_kind, FE::analysis::EnforcementKind::Strong);
        }
    }
    EXPECT_TRUE(system.formulationRecords().empty());
    EXPECT_TRUE(system.operatorDefinition("equations").cells.empty());
}

TEST(UstructModule, RegistersMixedDisplacementPressureSystem)
{
    auto mesh = std::make_shared<SingleTetraMeshAccess>();
    FE::systems::FESystem system(mesh);

    auto displacement_space =
        FE::spaces::VectorSpace(FE::spaces::SpaceType::H1, FE::ElementType::Tetra4, /*order=*/1, /*components=*/3);
    auto pressure_space = FE::spaces::SpaceFactory::create_h1(FE::ElementType::Tetra4, /*order=*/1);

    formulations::ustruct::UstructOptions options{};
    options.density = 1.2;
    auto deviatoric_model =
        materials::solid::isochoricNeoHookeanPK1(/*mu=*/3.0).model();
    options.deviatoric_pk1_model = deviatoric_model;
    options.penalty_parameter = 0.0;
    options.ct_m = 0.0;
    options.ct_c = 0.0;

    formulations::ustruct::UstructModule module(
        displacement_space, pressure_space, std::move(options));
    module.registerOn(system);

    system.setup({}, makeSingleTetraSetupInputs());

    EXPECT_EQ(system.temporalOrder(), 1);
    ASSERT_NE(system.findFieldByName("Displacement"), FE::INVALID_FIELD_ID);
    ASSERT_NE(system.findFieldByName("Velocity"), FE::INVALID_FIELD_ID);
    ASSERT_NE(system.findFieldByName("Pressure"), FE::INVALID_FIELD_ID);

    ASSERT_FALSE(system.formulationRecords().empty());
    const auto& record = system.formulationRecords().front();
    EXPECT_EQ(record.active_fields, (std::vector<FE::FieldId>{
        system.findFieldByName("Displacement"), system.findFieldByName("Velocity"),
        system.findFieldByName("Pressure")}));
    ASSERT_FALSE(record.constitutive_laws.empty());
    const auto& law = record.constitutive_laws.front();
    EXPECT_EQ(law.primary_field, system.findFieldByName("Displacement"));
    EXPECT_EQ(law.tensor_rank, "rank2");
    EXPECT_EQ(law.model.get(), deviatoric_model.get());

    const auto n = system.dofHandler().getNumDofs();
    ASSERT_EQ(n, 28);
}

TEST(UstructModule, RegistersRequestedNodePatchPostprocessing)
{
    auto mesh = std::make_shared<SingleTetraMeshAccess>();
    FE::systems::FESystem system(mesh);

    auto displacement_space =
        FE::spaces::VectorSpace(FE::spaces::SpaceType::H1, FE::ElementType::Tetra4, /*order=*/1, /*components=*/3);
    auto pressure_space = FE::spaces::SpaceFactory::create_h1(FE::ElementType::Tetra4, /*order=*/1);

    formulations::ustruct::UstructOptions options{};
    options.density = 1.2;
    options.deviatoric_pk1_model =
        materials::solid::isochoricNeoHookeanPK1(/*mu=*/3.0).model();
    options.penalty_parameter = 0.0;
    options.ct_m = 0.0;
    options.ct_c = 0.0;
    options.register_def_grad_output = true;
    options.register_jacobian_output = true;
    options.register_divergence_output = true;
    options.register_strain_output = true;
    options.register_stress_output = true;
    options.register_cauchy_stress_output = true;
    options.register_von_mises_stress_output = true;

    formulations::ustruct::UstructModule module(
        displacement_space, pressure_space, std::move(options));
    module.registerOn(system);

    ASSERT_EQ(system.derivedResults().size(), 7u);
    for (const char* name : {"Def_grad", "Jacobian", "Divergence", "Strain", "Stress", "Cauchy_stress", "VonMises_stress"}) {
        const auto& result = findDerivedResult(system, name);
        EXPECT_EQ(result.scope, FE::post::DerivedResultScope::Vertex) << name;
        EXPECT_EQ(result.policy, FE::post::DerivedResultPolicy::PatchAverage) << name;
    }

    EXPECT_EQ(findDerivedResult(system, "Def_grad").shape.components, 9);
    EXPECT_EQ(findDerivedResult(system, "Jacobian").shape.components, 1);
    EXPECT_EQ(findDerivedResult(system, "Divergence").shape.components, 1);
    EXPECT_EQ(findDerivedResult(system, "Strain").shape.components, 6);
    EXPECT_EQ(findDerivedResult(system, "Stress").shape.components, 6);
    EXPECT_EQ(findDerivedResult(system, "Cauchy_stress").shape.components, 6);
    EXPECT_EQ(findDerivedResult(system, "VonMises_stress").shape.components, 1);

    const auto d_id = system.findFieldByName("Displacement");
    const auto v_id = system.findFieldByName("Velocity");
    const auto p_id = system.findFieldByName("Pressure");
    ASSERT_NE(d_id, FE::INVALID_FIELD_ID);
    ASSERT_NE(v_id, FE::INVALID_FIELD_ID);
    ASSERT_NE(p_id, FE::INVALID_FIELD_ID);

    EXPECT_EQ(findDerivedResult(system, "Def_grad").referenced_fields,
              std::vector<FE::FieldId>{d_id});
    EXPECT_EQ(findDerivedResult(system, "Jacobian").referenced_fields,
              std::vector<FE::FieldId>{d_id});
    EXPECT_EQ(findDerivedResult(system, "Strain").referenced_fields,
              std::vector<FE::FieldId>{d_id});
    EXPECT_EQ(findDerivedResult(system, "Divergence").referenced_fields,
              (std::vector<FE::FieldId>{d_id, v_id}));
    EXPECT_EQ(findDerivedResult(system, "Stress").referenced_fields,
              (std::vector<FE::FieldId>{d_id, p_id}));
    EXPECT_EQ(findDerivedResult(system, "Cauchy_stress").referenced_fields,
              (std::vector<FE::FieldId>{d_id, p_id}));
    EXPECT_EQ(findDerivedResult(system, "VonMises_stress").referenced_fields,
              (std::vector<FE::FieldId>{d_id, p_id}));
}

TEST(UstructModule, QuasiStaticOptionDropsTemporalDerivatives)
{
    auto mesh = std::make_shared<SingleTetraMeshAccess>();
    FE::systems::FESystem system(mesh);

    auto displacement_space =
        FE::spaces::VectorSpace(FE::spaces::SpaceType::H1, FE::ElementType::Tetra4, /*order=*/1, /*components=*/3);
    auto pressure_space = FE::spaces::SpaceFactory::create_h1(FE::ElementType::Tetra4, /*order=*/1);

    formulations::ustruct::UstructOptions options{};
    options.density = 1.2;
    options.deviatoric_pk1_model =
        materials::solid::isochoricNeoHookeanPK1(/*mu=*/3.0).model();
    options.penalty_parameter = 0.0;
    options.ct_m = 0.0;
    options.ct_c = 0.0;
    options.enable_time_derivative_terms = false;

    formulations::ustruct::UstructModule module(
        displacement_space, pressure_space, std::move(options));
    module.registerOn(system);

    system.setup({}, makeSingleTetraSetupInputs());

    EXPECT_EQ(system.temporalOrder(), 0);
    EXPECT_FALSE(system.isTransient());
    EXPECT_FALSE(system.hasExplicitTimeDependency());
    EXPECT_FALSE(system.hasTimeDependentConstraints());
    EXPECT_FALSE(system.requiresTimeAdvancement());
    ASSERT_NE(system.findFieldByName("Displacement"), FE::INVALID_FIELD_ID);
    ASSERT_EQ(system.findFieldByName("Velocity"), FE::INVALID_FIELD_ID);
    ASSERT_NE(system.findFieldByName("Pressure"), FE::INVALID_FIELD_ID);
    EXPECT_EQ(system.dofHandler().getNumDofs(), 16);
    ASSERT_EQ(system.formulationRecords().size(), 1u);
    EXPECT_EQ(system.formulationRecords().front().active_fields,
              (std::vector<FE::FieldId>{system.findFieldByName("Displacement"),
                                        system.findFieldByName("Pressure")}));
}

TEST(UstructModule, InstallsComponentStrongConstraintsOnDisplacementAndVelocity)
{
    constexpr int marker = 17;
    auto mesh = std::make_shared<FE::forms::test::SingleTetraOneBoundaryFaceMeshAccess>(marker);
    FE::systems::FESystem system(mesh);

    auto displacement_space =
        FE::spaces::VectorSpace(FE::spaces::SpaceType::H1, mesh, /*order=*/1, /*components=*/3);
    auto pressure_space = FE::spaces::Space(FE::spaces::SpaceType::H1, mesh, /*order=*/1);

    formulations::ustruct::UstructOptions options{};
    options.density = 1.0;
    options.deviatoric_pk1_model =
        materials::solid::isochoricNeoHookeanPK1(/*mu=*/3.0).model();
    options.penalty_parameter = 0.0;
    options.ct_m = 0.0;
    options.ct_c = 0.0;
    options.displacement_dirichlet.push_back(formulations::ustruct::UstructOptions::DisplacementDirichletBC{
        .boundary_marker = marker,
        .value = {
            formulations::ustruct::UstructOptions::ScalarValue{1.5},
            formulations::ustruct::UstructOptions::ScalarValue{0.0},
            formulations::ustruct::UstructOptions::ScalarValue{-2.0}},
        .active_components = {true, false, true},
    });
    options.velocity_dirichlet.push_back(formulations::ustruct::UstructOptions::VelocityDirichletBC{
        .boundary_marker = marker,
        .value = {
            formulations::ustruct::UstructOptions::ScalarValue{0.0},
            formulations::ustruct::UstructOptions::ScalarValue{-0.25},
            formulations::ustruct::UstructOptions::ScalarValue{0.0}},
        .active_components = {false, true, false},
    });

    formulations::ustruct::UstructModule module(
        displacement_space, pressure_space, std::move(options));
    module.registerOn(system);
    system.setup({}, makeSingleTetraSetupInputs());

    const auto d_id = system.findFieldByName("Displacement");
    const auto v_id = system.findFieldByName("Velocity");
    ASSERT_NE(d_id, FE::INVALID_FIELD_ID);
    ASSERT_NE(v_id, FE::INVALID_FIELD_ID);

    for (FE::GlobalIndex vertex : {FE::GlobalIndex{0}, FE::GlobalIndex{1}, FE::GlobalIndex{2}}) {
        const auto d0 = vertexComponentDof(system, d_id, vertex, 0);
        const auto d1 = vertexComponentDof(system, d_id, vertex, 1);
        const auto d2 = vertexComponentDof(system, d_id, vertex, 2);
        const auto v0 = vertexComponentDof(system, v_id, vertex, 0);
        const auto v1 = vertexComponentDof(system, v_id, vertex, 1);
        const auto v2 = vertexComponentDof(system, v_id, vertex, 2);

        EXPECT_TRUE(system.constraints().isConstrained(d0));
        EXPECT_NEAR(system.constraints().getInhomogeneity(d0), 1.5, 1e-12);
        EXPECT_FALSE(system.constraints().isConstrained(d1));
        EXPECT_TRUE(system.constraints().isConstrained(d2));
        EXPECT_NEAR(system.constraints().getInhomogeneity(d2), -2.0, 1e-12);

        EXPECT_TRUE(system.constraints().isConstrained(v0));
        EXPECT_NEAR(system.constraints().getInhomogeneity(v0), 0.0, 1e-12);
        EXPECT_TRUE(system.constraints().isConstrained(v1));
        EXPECT_NEAR(system.constraints().getInhomogeneity(v1), -0.25, 1e-12);
        EXPECT_TRUE(system.constraints().isConstrained(v2));
        EXPECT_NEAR(system.constraints().getInhomogeneity(v2), 0.0, 1e-12);
    }

    for (int component = 0; component < 3; ++component) {
        EXPECT_FALSE(system.constraints().isConstrained(
            vertexComponentDof(system, d_id, FE::GlobalIndex{3}, component)));
        EXPECT_FALSE(system.constraints().isConstrained(
            vertexComponentDof(system, v_id, FE::GlobalIndex{3}, component)));
    }
}

TEST(UstructModule, FollowerPressureExpressionUsesTimeScalarCoefficient)
{
    formulations::ustruct::UstructOptions::FollowerPressureBC bc{};
    bc.boundary_marker = 3;
    bc.pressure = formulations::ustruct::UstructOptions::ScalarValue{
        FE::forms::TimeScalarCoefficient(
            [](FE::Real x, FE::Real /*y*/, FE::Real /*z*/, FE::Real t) -> FE::Real {
                return x + static_cast<FE::Real>(2.0) * t;
            })};

    const auto expr = formulations::ustruct::Factories::followerPressureExpression(bc, "p_stage");

    FE::forms::PointEvalContext ctx;
    ctx.x = {1.0, 0.0, 0.0};
    ctx.time = 0.25;
    EXPECT_DOUBLE_EQ(FE::forms::evaluateScalarAt(expr, ctx), 1.5);

    ctx.time = 0.75;
    EXPECT_DOUBLE_EQ(FE::forms::evaluateScalarAt(expr, ctx), 2.5);
}

TEST(UstructModule, FollowerPressureRampExpressionUsesFormTime)
{
    formulations::ustruct::UstructOptions::FollowerPressureBC bc{};
    bc.boundary_marker = 3;
    bc.ramp = formulations::ustruct::UstructOptions::FollowerPressureBC::LinearRamp{
        /*start_time=*/0.2,
        /*end_time=*/0.7,
        /*start_value=*/10.0,
        /*end_value=*/30.0};

    const auto expr = formulations::ustruct::Factories::followerPressureExpression(bc, "p_ramp");

    FE::forms::PointEvalContext ctx;
    ctx.x = {0.0, 0.0, 0.0};
    ctx.time = 0.1;
    EXPECT_DOUBLE_EQ(FE::forms::evaluateScalarAt(expr, ctx), 10.0);

    ctx.time = 0.45;
    EXPECT_DOUBLE_EQ(FE::forms::evaluateScalarAt(expr, ctx), 20.0);

    ctx.time = 1.0;
    EXPECT_DOUBLE_EQ(FE::forms::evaluateScalarAt(expr, ctx), 30.0);
}

TEST(UstructModule, ZeroStateHasZeroResidual)
{
    auto mesh = std::make_shared<SingleTetraMeshAccess>();
    FE::systems::FESystem system(mesh);

    auto displacement_space =
        FE::spaces::VectorSpace(FE::spaces::SpaceType::H1, FE::ElementType::Tetra4, /*order=*/1, /*components=*/3);
    auto pressure_space = FE::spaces::SpaceFactory::create_h1(FE::ElementType::Tetra4, /*order=*/1);

    formulations::ustruct::UstructOptions options{};
    options.density = 1.0;
    options.deviatoric_pk1_model =
        materials::solid::isochoricNeoHookeanPK1(/*mu=*/3.0).model();
    options.penalty_parameter = 0.0;
    options.ct_m = 0.0;
    options.ct_c = 0.0;

    formulations::ustruct::UstructModule module(
        displacement_space, pressure_space, std::move(options));
    module.registerOn(system);
    system.setup({}, makeSingleTetraSetupInputs());

    const auto n = system.dofHandler().getNumDofs();
    std::vector<FE::Real> u(static_cast<std::size_t>(n), 0.0);
    std::vector<FE::Real> u_prev(static_cast<std::size_t>(n), 0.0);
    std::vector<FE::Real> u_prev2(static_cast<std::size_t>(n), 0.0);

    std::array<std::span<const FE::Real>, 2> history_spans{
        std::span<const FE::Real>(u_prev),
        std::span<const FE::Real>(u_prev2)};
    std::array<double, 2> dt_history{1.0, 1.0};

    FE::systems::SystemStateView state;
    state.dt = 1.0;
    state.dt_prev = 1.0;
    state.effective_dt = 1.0;
    state.u = std::span<const FE::Real>(u);
    state.u_prev = std::span<const FE::Real>(u_prev);
    state.u_prev2 = std::span<const FE::Real>(u_prev2);
    state.u_history = std::span<const std::span<const FE::Real>>(history_spans);
    state.dt_history = std::span<const double>(dt_history);

    auto integrator = std::make_shared<const FE::systems::BDFIntegrator>(1);
    FE::systems::TransientSystem transient(system, std::move(integrator));

    FE::systems::AssemblyRequest req;
    req.op = "equations";
    req.want_vector = true;

    FE::assembly::DenseVectorView residual(n);
    const auto result = transient.assemble(req, state, nullptr, &residual);
    ASSERT_TRUE(result.success) << result.error_message;

    for (FE::GlobalIndex i = 0; i < n; ++i) {
        EXPECT_NEAR(residual[i], 0.0, 1e-12);
    }
}

} // namespace test
} // namespace Physics
} // namespace svmp
