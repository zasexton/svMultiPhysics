/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#include <gtest/gtest.h>

#include "Systems/VertexFieldInitialization.h"

#include "Dofs/EntityDofMap.h"
#include "Spaces/H1Space.h"
#include "Systems/FESystem.h"
#include "Tests/Unit/Forms/FormsTestHelpers.h"

#if defined(FE_HAS_EIGEN)
#  include "Backends/Eigen/EigenVector.h"
#endif

#include "Mesh/Core/MeshBase.h"
#include "Mesh/Fields/MeshFields.h"
#include "Mesh/Mesh.h"
#include "Mesh/Topology/CellShape.h"

#include <cmath>
#include <limits>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

namespace svmp::FE::systems::test {
namespace {

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

Real* attachVertexField(svmp::Mesh& mesh,
                        const std::string& name,
                        std::size_t components)
{
    const auto handle = svmp::MeshFields::attach_field(
        mesh.local_mesh(), svmp::EntityKind::Vertex, name,
        svmp::FieldScalarType::Float64, components);
    return svmp::MeshFields::field_data_as<Real>(mesh.local_mesh(), handle);
}

MeshVertexFieldMessages fieldMessages()
{
    MeshVertexFieldMessages messages{};
    messages.no_native_mesh = "no native mesh";
    messages.missing_field_prefix = "missing '";
    messages.not_float64 = "not Float64";
    messages.no_components = "no components";
    messages.too_few_entries = "too few entries";
    messages.empty = "empty";
    messages.no_mesh_support = "no mesh support";
    return messages;
}

LocalVertexMessages vertexMessages()
{
    return {.negative_id = "negative id",
            .no_native_mesh = "global id without a native mesh",
            .no_mesh_support = "global id without mesh support"};
}

template <class Exception, class F>
std::string messageOf(F&& f)
{
    try {
        f();
    } catch (const Exception& error) {
        return error.what();
    }
    return "(no exception)";
}

} // namespace

TEST(VertexFieldInitialization, MeshVertexFieldIsValidatedWithTheCallersMessages)
{
    auto mesh = twoTriangleMesh();
    auto* phi = attachVertexField(*mesh, "phi", 2u);
    ASSERT_NE(phi, nullptr);
    for (std::size_t vertex = 0; vertex < 4u; ++vertex) {
        phi[2u * vertex] = static_cast<Real>(vertex) - Real{1.5};
        phi[2u * vertex + 1u] = Real{9.0};
    }
    (void)svmp::MeshFields::attach_field(
        mesh->local_mesh(), svmp::EntityKind::Vertex, "ids",
        svmp::FieldScalarType::Int32, 1u);
    FESystem system(mesh);

    auto messages = fieldMessages();
    const auto view = requireMeshVertexField(system, "phi", 4, messages);
    EXPECT_EQ(view.values, phi);
    EXPECT_EQ(view.components, 2u);
    EXPECT_EQ(view.entity_count, 4u);

    EXPECT_EQ(messageOf<std::runtime_error>([&] {
                  (void)requireMeshVertexField(system, "psi", 4, messages);
              }),
              "missing 'psi'");
    EXPECT_EQ(messageOf<std::runtime_error>([&] {
                  (void)requireMeshVertexField(system, "ids", 4, messages);
              }),
              "not Float64");
    EXPECT_EQ(messageOf<std::runtime_error>([&] {
                  (void)requireMeshVertexField(system, "phi", 5, messages);
              }),
              "too few entries");

    // Only the first component of the first n_vertices vertices is checked,
    // and only with a non-finite message.
    phi[2u * 2u + 1u] = std::numeric_limits<Real>::quiet_NaN();
    messages.non_finite_prefix = "non-finite at vertex ";
    EXPECT_NO_THROW((void)requireMeshVertexField(system, "phi", 4, messages));
    phi[2u * 2u] = std::numeric_limits<Real>::infinity();
    EXPECT_NO_THROW((void)requireMeshVertexField(system, "phi", 2, messages));
    EXPECT_EQ(messageOf<std::runtime_error>([&] {
                  (void)requireMeshVertexField(system, "phi", 4, messages);
              }),
              "non-finite at vertex 2");
    messages.non_finite_prefix.clear();
    EXPECT_NO_THROW((void)requireMeshVertexField(system, "phi", 4, messages));
}

TEST(VertexFieldInitialization, MeshVertexFieldRequiresANativeMesh)
{
    FESystem system(std::make_shared<forms::test::SingleTetraMeshAccess>());
    EXPECT_EQ(messageOf<std::runtime_error>([&] {
                  (void)requireMeshVertexField(system, "phi", 4, fieldMessages());
              }),
              "no native mesh");
}

TEST(VertexFieldInitialization, LocalVertexOfLocalAndGlobalIds)
{
    FESystem without_mesh(std::make_shared<forms::test::SingleTetraMeshAccess>());
    const auto messages = vertexMessages();
    EXPECT_EQ(localVertexOf(without_mesh, constraints::VertexIdMode::LocalVertexId, 7, messages)
                  .value_or(-100),
              7);
    EXPECT_EQ(messageOf<std::invalid_argument>([&] {
                  (void)localVertexOf(without_mesh, constraints::VertexIdMode::LocalVertexId, -1, messages);
              }),
              "negative id");
    EXPECT_EQ(messageOf<std::invalid_argument>([&] {
                  (void)localVertexOf(without_mesh, constraints::VertexIdMode::GlobalVertexGid, 1, messages);
              }),
              "global id without a native mesh");

    FESystem with_mesh(twoTriangleMesh());
    EXPECT_EQ(localVertexOf(with_mesh, constraints::VertexIdMode::GlobalVertexGid, 2, messages)
                  .value_or(-100),
              2);
    EXPECT_FALSE(localVertexOf(with_mesh, constraints::VertexIdMode::GlobalVertexGid, 99, messages)
                     .has_value());
    EXPECT_EQ(messageOf<std::invalid_argument>([&] {
                  (void)localVertexOf(with_mesh, constraints::VertexIdMode::GlobalVertexGid, -3, messages);
              }),
              "negative id");
}

TEST(VertexFieldInitialization, VertexDofMapOfAVertexField)
{
    FESystem system(twoTriangleMesh());
    const auto p = system.addField(FieldSpec{
        .name = "p",
        .space = std::make_shared<spaces::H1Space>(ElementType::Triangle3, 1),
        .components = 1});
    system.setup();
    const auto& map = requireVertexDofMap(system, p, "no vertex DOFs");
    for (GlobalIndex vertex = 0; vertex < 4; ++vertex) {
        EXPECT_EQ(map.getVertexDofs(vertex).size(), 1u);
    }
}

#if defined(FE_HAS_EIGEN)
TEST(VertexFieldInitialization, VectorEntriesAreWrittenThroughOneView)
{
    backends::EigenVector u(4);
    u.zero();
    writeVectorEntries(u, {1, 3}, {2.0, 4.0}, "no entries", "no view");
    const auto values = u.localSpan();
    EXPECT_EQ(values[0], 0.0);
    EXPECT_EQ(values[1], 2.0);
    EXPECT_EQ(values[2], 0.0);
    EXPECT_EQ(values[3], 4.0);
    EXPECT_EQ(messageOf<std::runtime_error>([&] {
                  writeVectorEntries(u, {}, {}, "no entries", "no view");
              }),
              "no entries");
}
#endif

} // namespace svmp::FE::systems::test
