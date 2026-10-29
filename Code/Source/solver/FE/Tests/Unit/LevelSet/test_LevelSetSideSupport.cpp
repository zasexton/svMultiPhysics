/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#include <gtest/gtest.h>

#include "LevelSet/LevelSetSideSupport.h"
#include "Dofs/EntityDofMap.h"
#include "Spaces/H1Space.h"
#include "Systems/FESystem.h"
#include "Systems/VertexFieldInitialization.h"
#include "Assembly/CutIntegrationContext.h"
#include "Geometry/CutQuadrature.h"

#include "Tests/Unit/Forms/FormsTestHelpers.h"

#include "Mesh/Core/MeshBase.h"
#include "Mesh/Fields/MeshFields.h"
#include "Mesh/Mesh.h"
#include "Mesh/Topology/CellShape.h"

#include <array>
#include <algorithm>
#include <cmath>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#if defined(FE_HAS_EIGEN)
#  include "Backends/Eigen/EigenVector.h"
#endif

namespace {

using svmp::FE::GlobalIndex;
using svmp::FE::Real;
using svmp::FE::level_set::LevelSetSideSupportQuery;
using svmp::FE::level_set::LevelSetSideVertexValues;
using svmp::FE::level_set::LevelSetSupportSide;
using svmp::FE::level_set::collectVertexFieldOnLevelSetSide;
using svmp::FE::level_set::levelSetSideSupportVertices;
using svmp::FE::level_set::levelSetSideVertexChoice;
using svmp::FE::level_set::levelSetSideVertexValue;
using svmp::FE::level_set::levelSetVertexSide;
using svmp::FE::level_set::vertexOnLevelSetSide;

// The unit square as two triangles: vertices (0,0), (1,0), (1,1), (0,1).
std::shared_ptr<svmp::Mesh> twoTriangleMesh()
{
    auto base = std::make_shared<svmp::MeshBase>();
    svmp::CellShape shape{};
    shape.family = svmp::CellFamily::Triangle;
    shape.num_corners = 3;
    shape.order = 1;
    base->build_from_arrays(
        /*spatial_dim=*/2,
        std::vector<svmp::real_t>{0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0},
        std::vector<svmp::offset_t>{0, 3, 6},
        std::vector<svmp::index_t>{0, 1, 2, 0, 2, 3},
        std::vector<svmp::CellShape>(2u, shape));
    base->finalize();
    return svmp::create_mesh(std::move(base));
}

TEST(LevelSetSideSupport, AdmissibilityIncludesNonlocalValuesAndStopsAtFirstViolation)
{
    using namespace svmp::FE;
    using namespace level_set;
    systems::FESystem system(twoTriangleMesh());
    const std::vector<Real> phi{-0.25, 0.125, 0.0, -1.0};
    std::vector<VertexSideFailure> failures;
    VertexSideAdmissibilityRequirement requirement{
        .level_set = {phi.data(), 1u, phi.size()},
        .isovalue = 0.0,
        .side = LevelSetSupportSide::Negative,
        .exclusion_margin = 0.25,
        .failure_message = [&](const VertexSideFailure& failure) {
            failures.push_back(failure);
            return "rejected vertex " + std::to_string(failure.requested_vertex);
        }};
    const std::vector<constraints::VertexDirichletValue> valid{
        {3, 6.0}, {99, -100.0}, {0, 7.0}, {3, 8.0}};
    const auto stats = requireVertexValuesOnLevelSetSide(system, {.values = valid}, requirement);
    EXPECT_EQ(stats.checked, 3u);
    EXPECT_EQ(stats.skipped_nonlocal, 1u);
    EXPECT_EQ(stats.input_min, -100.0);
    EXPECT_EQ(stats.input_max, 8.0);
    EXPECT_EQ(stats.signed_gap_min, -1.0);
    EXPECT_EQ(stats.signed_gap_max, -0.25); // Equality at the margin is admissible.
    EXPECT_TRUE(failures.empty());

    const std::vector<constraints::VertexDirichletValue> dry_then_near{{1, 0}, {2, 0}};
    EXPECT_THROW((void)requireVertexValuesOnLevelSetSide(
        system, {.values = dry_then_near}, requirement), std::invalid_argument);
    ASSERT_EQ(failures.size(), 1u);
    EXPECT_EQ(failures.back().reason, VertexSideViolation::OutsideSide);
    EXPECT_EQ(failures.back().requested_vertex, 1);
    EXPECT_EQ(failures.back().local_vertex, 1);
    EXPECT_EQ(failures.back().sample.phi, 0.125);
    EXPECT_TRUE(failures.back().sample.within_margin);

    const std::vector<constraints::VertexDirichletValue> on_interface{{2, 0}};
    EXPECT_THROW((void)requireVertexValuesOnLevelSetSide(
        system, {.values = on_interface}, requirement), std::invalid_argument);
    ASSERT_EQ(failures.size(), 2u);
    EXPECT_EQ(failures.back().reason, VertexSideViolation::WithinMargin);
    requirement.exclusion_margin = 0.0;
    EXPECT_NO_THROW((void)requireVertexValuesOnLevelSetSide(
        system, {.values = on_interface}, requirement));

    const std::vector<constraints::VertexDirichletValue> absent{{99, 2.0}};
    const auto no_local = requireVertexValuesOnLevelSetSide(system, {.values = absent}, requirement);
    EXPECT_EQ(no_local.checked, 0u);
    EXPECT_EQ(no_local.skipped_nonlocal, 1u);
    EXPECT_EQ(no_local.input_min, 2.0);
    EXPECT_EQ(no_local.signed_gap_min, std::numeric_limits<Real>::infinity());
    EXPECT_EQ(no_local.signed_gap_max, -std::numeric_limits<Real>::infinity());
    EXPECT_THROW((void)requireVertexValuesOnLevelSetSide(
        system, {.values = absent, .id_mode = constraints::VertexIdMode::LocalVertexId},
        requirement), std::invalid_argument);
    ASSERT_EQ(failures.size(), 3u);
    EXPECT_EQ(failures.back().reason, VertexSideViolation::OutsideMesh);
}

#if defined(FE_HAS_EIGEN)
TEST(LevelSetSideSupport, VertexInitializationLeavesQuadraticNonvertexCoefficientsUntouched)
{
    using namespace svmp::FE;
    using namespace level_set;
    auto mesh = twoTriangleMesh();
    const auto source = svmp::MeshFields::attach_field(
        mesh->local_mesh(), svmp::EntityKind::Vertex, "value",
        svmp::FieldScalarType::Float64, 1u);
    auto* values = svmp::MeshFields::field_data_as<Real>(mesh->local_mesh(), source);
    std::copy_n(std::array<Real, 4>{10, 12, 15, 13}.data(), 4, values);
    systems::FESystem system(mesh);
    const auto field = system.addField({
        .name = "value", .space = std::make_shared<spaces::H1Space>(ElementType::Triangle3, 2)});
    system.setup();
    backends::EigenVector state(system.fieldDofHandler(field).getNumDofs());
    const auto target = systems::requireVertexField(system, "value", "missing '", "no map");
    std::fill(state.localSpan().begin(), state.localSpan().end(), -37.0);
    const std::vector<constraints::VertexDirichletValue> prescribed{{0, 19}, {99, 500}, {2, 17}};
    const SideVertexInitializationDeclaration declaration{
        .source = [](const std::array<Real, 3>& x) { return Real{20} + x[0]; },
        .comparisons = {.values = prescribed},
        .messages = {.non_finite_prefix = "nonfinite ", .no_entries = "no entries", .no_view = "no view"}};
    const auto result = initializeVertexFieldOnLevelSetSide(system, state, target, declaration);
    EXPECT_EQ(result.initialized_dofs, 4u);
    EXPECT_EQ(result.comparisons.checked, 2u);
    EXPECT_EQ(result.comparisons.skipped, 1u);
    EXPECT_EQ(result.comparisons.maximum_absolute_error, 4.0);
    std::vector<Real> actual(state.localSpan().begin(), state.localSpan().end());
    std::sort(actual.begin(), actual.end());
    EXPECT_EQ(actual, (std::vector<Real>{-37, -37, -37, -37, -37, 20, 20, 21, 21}));

    // The optional same-name operation deliberately projects all P2 coefficients.
    EXPECT_EQ(systems::initializeFieldFromMeshVertexField(system, state, field, "projection"), 9u);
    actual.assign(state.localSpan().begin(), state.localSpan().end());
    std::sort(actual.begin(), actual.end());
    EXPECT_EQ(actual, (std::vector<Real>{10, 11, 11.5, 12, 12.5, 13, 13.5, 14, 15}));
}

TEST(LevelSetSideSupport, InitializationRebindsSupportAndValidatesTheEntireNamedSourceBeforeWriting)
{
    using namespace svmp::FE;
    using namespace level_set;
    auto mesh = twoTriangleMesh();
    const auto attach = [&](const std::string& name, std::size_t components) {
        const auto handle = svmp::MeshFields::attach_field(
            mesh->local_mesh(), svmp::EntityKind::Vertex, name,
            svmp::FieldScalarType::Float64, components);
        return svmp::MeshFields::field_data_as<Real>(mesh->local_mesh(), handle);
    };
    auto* phi = attach("phi", 1u);
    auto* source = attach("source", 2u);
    std::fill_n(phi, 4, 1.0);
    for (std::size_t i = 0; i < 4u; ++i) {
        source[2 * i] = Real{10} + static_cast<Real>(i);
        source[2 * i + 1] = std::numeric_limits<Real>::quiet_NaN();
    }
    source[6] = std::numeric_limits<Real>::infinity(); // Off the active support.
    systems::FESystem system(mesh);
    const auto field = system.addField({
        .name = "value", .space = std::make_shared<spaces::H1Space>(ElementType::Triangle3, 1)});
    system.setup();
    const auto target = systems::requireVertexField(system, "value", "missing '", "no map");
    backends::EigenVector state(system.fieldDofHandler(field).getNumDofs());
    std::fill(state.localSpan().begin(), state.localSpan().end(), -37.0);
    SideVertexInitializationDeclaration declaration{
        .source = systems::RequiredMeshVertexScalarSource{
            .field_name = "source", .messages = {.non_finite_prefix = "invalid source vertex "}},
        .support = LevelSetVertexSupportDeclaration{
            .level_set = {.field_name = "phi"}, .isovalue = 0,
            .side = LevelSetSupportSide::Negative,
            .interface_marker = -1, .use_generated_volume_rules = true},
        .unsupported_value = -2,
        .messages = {.non_finite_prefix = "invalid output vertex ", .no_entries = "no entries", .no_view = "no view"}};
    try {
        (void)initializeVertexFieldOnLevelSetSide(system, state, target, declaration);
        FAIL() << "invalid dry source was accepted";
    } catch (const std::runtime_error& error) {
        EXPECT_EQ(std::string(error.what()), "invalid source vertex 3");
    }
    EXPECT_EQ(std::vector<Real>(state.localSpan().begin(), state.localSpan().end()),
              (std::vector<Real>{-37, -37, -37, -37}));
    source[6] = 13;
    auto result = initializeVertexFieldOnLevelSetSide(system, state, target, declaration);
    EXPECT_EQ(result.statistics.supported_vertices, 0u);
    EXPECT_EQ(result.statistics.value_min, -2.0);
    phi[0] = -1; // Both cells now have negative measure, so every vertex is supported.
    result = initializeVertexFieldOnLevelSetSide(system, state, target, declaration);
    EXPECT_EQ(result.statistics.on_side_vertices, 1u);
    EXPECT_EQ(result.statistics.supported_vertices, 4u);
    EXPECT_EQ(result.statistics.off_side_supported_vertices, 3u);
    EXPECT_EQ(result.statistics.value_min, 10.0);
    EXPECT_EQ(result.statistics.value_max, 13.0);

    // A supplied marker selects generated rules; an unresolved marker keeps the fallback.
    auto context = std::make_shared<assembly::CutIntegrationContext>();
    assembly::CutCellAssemblyMetadata metadata{};
    metadata.cell = 0;
    metadata.parent_entity = 0;
    metadata.side = geometry::CutIntegrationSide::Negative;
    metadata.volume_fraction = 0.25;
    geometry::CutQuadratureRule rule{};
    rule.kind = geometry::CutQuadratureKind::Volume;
    rule.side = geometry::CutIntegrationSide::Negative;
    rule.measure = 0.125;
    rule.parent_measure = 0.5;
    rule.volume_fraction = 0.25;
    context->addGeneratedVolumeRule(51, metadata, rule);
    system.setCutIntegrationContext(context);
    result = initializeVertexFieldOnLevelSetSide(system, state, target, declaration);
    EXPECT_EQ(result.statistics.supported_vertices, 4u);
    declaration.support->interface_marker = 51;
    result = initializeVertexFieldOnLevelSetSide(system, state, target, declaration);
    EXPECT_EQ(result.statistics.supported_vertices, 3u);
    EXPECT_EQ(result.statistics.value_min, -2.0);
    EXPECT_EQ(result.statistics.supported_value_max, 12.0);
}
#endif

} // namespace

TEST(LevelSetSideSupport, VertexSidesAreInclusiveOfTheIsovalue)
{
    EXPECT_TRUE(vertexOnLevelSetSide(-1.0, 0.0, LevelSetSupportSide::Negative));
    EXPECT_TRUE(vertexOnLevelSetSide(0.0, 0.0, LevelSetSupportSide::Negative));
    EXPECT_FALSE(vertexOnLevelSetSide(0.5, 0.0, LevelSetSupportSide::Negative));
    EXPECT_TRUE(vertexOnLevelSetSide(0.5, 0.0, LevelSetSupportSide::Positive));
    EXPECT_TRUE(vertexOnLevelSetSide(0.25, 0.25, LevelSetSupportSide::Positive));
    EXPECT_FALSE(vertexOnLevelSetSide(-0.5, 0.0, LevelSetSupportSide::Positive));
    EXPECT_TRUE(vertexOnLevelSetSide(-0.5, 0.0, LevelSetSupportSide::Everywhere));
}

TEST(LevelSetSideSupport, WithoutANativeMeshTheSupportIsTheVerticesOnTheSide)
{
    auto mesh = std::make_shared<svmp::FE::forms::test::SingleTetraMeshAccess>();
    svmp::FE::systems::FESystem sys(mesh);
    // Two components per vertex: only the first is the level set.
    const std::vector<svmp::FE::Real> phi{-1.0, 9.0, 0.0, 9.0, 0.5, 9.0, 2.0, 9.0};
    LevelSetSideSupportQuery query{};
    query.level_set = {phi.data(), 2u, 4u};
    query.isovalue = 0.0;
    query.side = LevelSetSupportSide::Negative;
    query.message_prefix = "unit";
    EXPECT_EQ(levelSetSideSupportVertices(sys, query, 4),
              (std::vector<unsigned char>{1, 1, 0, 0}));
    query.side = LevelSetSupportSide::Positive;
    EXPECT_EQ(levelSetSideSupportVertices(sys, query, 4),
              (std::vector<unsigned char>{0, 1, 1, 1}));
}

TEST(LevelSetSideSupport, VertexSideAndDistanceToTheIsovalue)
{
    const std::vector<Real> phi{-1.0, 0.0, 1.0e-9, 2.0};
    const svmp::FE::level_set::LevelSetVertexValues level_set{phi.data(), 1u, 4u};
    const auto dry = levelSetVertexSide(level_set, 0, 0.0, LevelSetSupportSide::Positive, 1.0e-8);
    EXPECT_EQ(dry.phi, -1.0);
    EXPECT_EQ(dry.signed_gap, -1.0);
    EXPECT_FALSE(dry.on_side);
    EXPECT_FALSE(dry.within_margin);
    const auto on_interface = levelSetVertexSide(level_set, 1, 0.0, LevelSetSupportSide::Negative, 1.0e-8);
    EXPECT_TRUE(on_interface.on_side);
    EXPECT_TRUE(on_interface.within_margin);
    const auto near = levelSetVertexSide(level_set, 2, 0.0, LevelSetSupportSide::Negative, 1.0e-8);
    EXPECT_FALSE(near.on_side);
    EXPECT_TRUE(near.within_margin);
    const auto shifted = levelSetVertexSide(level_set, 3, 0.5, LevelSetSupportSide::Positive, 1.0e-8);
    EXPECT_EQ(shifted.signed_gap, 1.5);
    EXPECT_TRUE(shifted.on_side);
    EXPECT_FALSE(shifted.within_margin);
}

TEST(LevelSetSideSupport, SideVertexValuesTakeTheSupportedOrTheDryValue)
{
    const std::array<Real, 3> x{0.25, 0.0, 0.0};
    LevelSetSideVertexValues values{};
    values.value = [](const std::array<Real, 3>& p) { return Real{10.0} + p[0]; };
    values.dry_value = -1.0;

    // Unrestricted: every vertex is supported.
    auto choice = levelSetSideVertexChoice(values, 3);
    EXPECT_TRUE(choice.supported);
    EXPECT_TRUE(choice.on_side);
    EXPECT_EQ(levelSetSideVertexValue(values, 3, x), 10.25);

    // A vertex field replaces value(x): its first component.
    const std::vector<Real> field{1.0, 9.0, 2.0, 9.0, 3.0, 9.0, 4.0, 9.0};
    values.field_values = field.data();
    values.field_components = 2u;
    EXPECT_EQ(levelSetSideVertexValue(values, 2, x), 3.0);

    // Restricted to the negative side: the side decides...
    const std::vector<Real> phi{-1.0, 0.0, 0.5, 2.0};
    values.side = LevelSetSideVertexValues::Side{
        .level_set = {phi.data(), 1u, 4u},
        .isovalue = 0.0,
        .side = LevelSetSupportSide::Negative};
    choice = levelSetSideVertexChoice(values, 1);
    EXPECT_TRUE(choice.supported);
    EXPECT_TRUE(choice.on_side);
    EXPECT_EQ(levelSetSideVertexValue(values, 1, x), 2.0);
    choice = levelSetSideVertexChoice(values, 2);
    EXPECT_FALSE(choice.supported);
    EXPECT_FALSE(choice.on_side);
    EXPECT_EQ(levelSetSideVertexValue(values, 2, x), -1.0);

    // ...unless support vertices are given.
    values.side->support = std::vector<unsigned char>{0, 1, 1, 0};
    choice = levelSetSideVertexChoice(values, 0);
    EXPECT_FALSE(choice.supported);
    EXPECT_TRUE(choice.on_side);
    EXPECT_EQ(levelSetSideVertexValue(values, 0, x), -1.0);
    choice = levelSetSideVertexChoice(values, 2);
    EXPECT_TRUE(choice.supported);
    EXPECT_FALSE(choice.on_side);
    EXPECT_EQ(levelSetSideVertexValue(values, 2, x), 3.0);
}

TEST(LevelSetSideSupport, VertexFieldIsCollectedOnOneSide)
{
    svmp::FE::systems::FESystem system(twoTriangleMesh());
    const auto p = system.addField(svmp::FE::systems::FieldSpec{
        .name = "p",
        .space = std::make_shared<svmp::FE::spaces::H1Space>(svmp::FE::ElementType::Triangle3, 1),
        .components = 1});
    system.setup();
    const auto& map = svmp::FE::systems::requireVertexDofMap(system, p, "no vertex DOFs");

    // phi = x - 1/2: vertices 0 and 3 are on the negative side; vertex 1 is
    // supported although it is not.
    const std::vector<Real> phi{-0.5, 0.5, 0.5, -0.5};
    LevelSetSideVertexValues values{};
    values.side = LevelSetSideVertexValues::Side{
        .level_set = {phi.data(), 1u, 4u},
        .isovalue = 0.0,
        .side = LevelSetSupportSide::Negative,
        .support = std::vector<unsigned char>{1, 1, 0, 1}};
    values.value = [](const std::array<Real, 3>& x) { return Real{10.0} + x[1]; };
    values.dry_value = -1.0;

    const auto entries =
        collectVertexFieldOnLevelSetSide(system, p, map, values, "bad ", " here");
    const std::vector<Real> expected{10.0, 10.0, -1.0, 11.0};
    ASSERT_EQ(entries.dofs.size(), 4u);
    ASSERT_EQ(entries.values.size(), 4u);
    const auto offset = system.fieldDofOffset(p);
    for (GlobalIndex vertex = 0; vertex < 4; ++vertex) {
        const auto i = static_cast<std::size_t>(vertex);
        ASSERT_EQ(map.getVertexDofs(vertex).size(), 1u);
        EXPECT_EQ(entries.dofs[i], offset + map.getVertexDofs(vertex)[0]);
        EXPECT_EQ(entries.values[i], expected[i]);
    }
    const auto& statistics = entries.statistics;
    EXPECT_EQ(statistics.on_side_vertices, 2u);
    EXPECT_EQ(statistics.off_side_vertices, 2u);
    EXPECT_EQ(statistics.supported_vertices, 3u);
    EXPECT_EQ(statistics.off_side_supported_vertices, 1u);
    EXPECT_EQ(statistics.value_min, -1.0);
    EXPECT_EQ(statistics.value_max, 11.0);
    EXPECT_EQ(statistics.supported_value_min, 10.0);
    EXPECT_EQ(statistics.supported_value_max, 11.0);

    // Unrestricted, every vertex takes value(x) and no side is counted.
    values.side.reset();
    const auto everywhere =
        collectVertexFieldOnLevelSetSide(system, p, map, values, "bad ", " here");
    EXPECT_EQ(everywhere.statistics.on_side_vertices, 0u);
    EXPECT_EQ(everywhere.statistics.supported_vertices, 0u);
    EXPECT_EQ(everywhere.statistics.value_min, 10.0);
    EXPECT_EQ(everywhere.statistics.supported_value_max, 11.0);

    // A non-finite value names its vertex.
    values.value = [](const std::array<Real, 3>& x) {
        return x[1] > 0.5 ? std::numeric_limits<Real>::quiet_NaN() : Real{1.0};
    };
    try {
        (void)collectVertexFieldOnLevelSetSide(system, p, map, values, "bad ", " here");
        FAIL() << "non-finite value accepted";
    } catch (const std::runtime_error& error) {
        EXPECT_EQ(std::string(error.what()), "bad 2 here");
    }
}
