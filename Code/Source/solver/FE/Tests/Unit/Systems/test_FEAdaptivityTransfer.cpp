/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#include <gtest/gtest.h>

#include "Systems/FEAdaptivityTransfer.h"
#include "Systems/FESystem.h"

#include "Assembly/AssemblyKernel.h"
#include "Auxiliary/AuxiliaryStateManager.h"
#include "Dofs/EntityDofMap.h"
#include "Spaces/H1Space.h"
#include "Spaces/ProductSpace.h"

#include "Mesh/Core/MeshBase.h"
#include "Mesh/Mesh.h"
#include "Mesh/Topology/CellShape.h"

#ifdef MESH_HAS_ADAPTIVITY
#include "Mesh/Adaptivity/AdaptivityManager.h"
#include "Mesh/Adaptivity/Options.h"
#endif

#include <algorithm>
#include <array>
#include <cmath>
#include <memory>
#include <span>
#include <vector>

namespace {

using svmp::CellFamily;
using svmp::CellShape;
using svmp::Mesh;
using svmp::MeshBase;
using svmp::FE::ElementType;
using svmp::FE::FieldId;
using svmp::FE::GlobalIndex;
using svmp::FE::Real;
using svmp::FE::assembly::MassKernel;
using svmp::FE::spaces::H1Space;
using svmp::FE::spaces::ProductSpace;
using svmp::FE::systems::AuxiliaryStateScope;
using svmp::FE::systems::AuxiliaryStateSpec;
using svmp::FE::systems::AuxiliaryTransferPolicy;
using svmp::FE::systems::FEAdaptedStateTransferRequest;
using svmp::FE::systems::FEFieldTransferMethod;
using svmp::FE::systems::FEFieldTransferOptions;
using svmp::FE::systems::FESystem;
using svmp::FE::systems::FieldSpec;

std::shared_ptr<Mesh> build_single_quad_mesh()
{
    auto base = std::make_shared<MeshBase>();
    const std::vector<svmp::real_t> x_ref = {
        0.0, 0.0,
        1.0, 0.0,
        1.0, 1.0,
        0.0, 1.0,
    };
    const std::vector<svmp::offset_t> offsets = {0, 4};
    const std::vector<svmp::index_t> conn = {0, 1, 2, 3};
    const std::vector<CellShape> shapes = {{CellFamily::Quad, 4, 1}};
    base->build_from_arrays(2, x_ref, offsets, conn, shapes);
    base->finalize();
    return svmp::create_mesh(std::move(base));
}

MeshBase build_line_mesh_with_gids(const std::vector<svmp::real_t>& x_ref,
                                   const std::vector<svmp::gid_t>& gids)
{
    MeshBase mesh(1);
    std::vector<svmp::offset_t> offsets;
    std::vector<svmp::index_t> conn;
    std::vector<CellShape> shapes;
    offsets.reserve(gids.size());
    offsets.push_back(0);
    for (std::size_t e = 0; e + 1 < gids.size(); ++e) {
        conn.push_back(static_cast<svmp::index_t>(e));
        conn.push_back(static_cast<svmp::index_t>(e + 1));
        offsets.push_back(static_cast<svmp::offset_t>(conn.size()));
        shapes.push_back(CellShape{CellFamily::Line, 2, 1});
    }
    mesh.build_from_arrays(1, x_ref, offsets, conn, shapes);
    mesh.set_vertex_gids(gids);
    mesh.finalize();
    return mesh;
}

#ifdef MESH_HAS_ADAPTIVITY
svmp::AdaptivityOptions adaptivity_options()
{
    svmp::AdaptivityOptions options;
    options.max_refinement_level = 1;
    options.refinement_pattern = svmp::AdaptivityOptions::RefinementPattern::RED;
    options.conformity_mode = svmp::AdaptivityOptions::ConformityMode::ALLOW_HANGING_NODES;
    options.check_quality = false;
    options.enforce_quality_after_refinement = false;
    options.verbosity = 0;
    return options;
}

AuxiliaryStateSpec make_cell_auxiliary_spec()
{
    AuxiliaryStateSpec spec;
    spec.name = "cell_history";
    spec.size = 1;
    spec.scope = AuxiliaryStateScope::Cell;
    spec.transfer_policy = AuxiliaryTransferPolicy::FormulationDefined;
    return spec;
}

template <typename ValueFn>
void write_vertex_state(const FESystem& sys,
                        FieldId field,
                        const MeshBase& mesh,
                        std::vector<Real>& state,
                        ValueFn&& value_fn)
{
    const auto& rec = sys.fieldRecord(field);
    const auto components = static_cast<std::size_t>(std::max(1, rec.components));
    const auto* entity_map = sys.fieldDofHandler(field).getEntityDofMap();
    ASSERT_NE(entity_map, nullptr);
    const auto offset = sys.fieldDofOffset(field);
    for (std::size_t v = 0; v < mesh.n_vertices(); ++v) {
        const auto dofs = entity_map->getVertexDofs(static_cast<GlobalIndex>(v));
        ASSERT_GE(dofs.size(), components);
        std::array<svmp::real_t, 3> xyz{0.0, 0.0, 0.0};
        for (int d = 0; d < mesh.dim(); ++d) {
            xyz[static_cast<std::size_t>(d)] =
                mesh.X_ref()[static_cast<std::size_t>(mesh.dim()) * v + static_cast<std::size_t>(d)];
        }
        for (std::size_t c = 0; c < components; ++c) {
            state[static_cast<std::size_t>(offset + dofs[c])] = value_fn(v, c, xyz);
        }
    }
}

Real read_vertex_state(const FESystem& sys,
                       FieldId field,
                       std::span<const Real> state,
                       std::size_t vertex,
                       std::size_t component)
{
    const auto* entity_map = sys.fieldDofHandler(field).getEntityDofMap();
    EXPECT_NE(entity_map, nullptr);
    const auto dofs = entity_map->getVertexDofs(static_cast<GlobalIndex>(vertex));
    EXPECT_GT(dofs.size(), component);
    const auto offset = sys.fieldDofOffset(field);
    return state[static_cast<std::size_t>(offset + dofs[component])];
}
#endif

} // namespace

TEST(FEAdaptivityTransfer, MissingProvenanceReportsDiagnosticsAndConservativeModePreservesSum)
{
    auto old_mesh = build_line_mesh_with_gids({0.0, 1.0}, {0, 1});
    auto new_mesh = build_line_mesh_with_gids({0.0, 0.5, 1.0}, {0, 42, 1});

    svmp::RefinementDelta empty_delta;
    std::vector<Real> old_values = {2.0, 4.0};
    std::vector<Real> new_values(3, 0.0);

    FEFieldTransferOptions optional_missing;
    optional_missing.require_all_vertices = false;
    auto missing = svmp::FE::systems::transferNodalFieldByVertexProvenance(
        old_mesh, new_mesh, empty_delta, 1, old_values, new_values, optional_missing);
    EXPECT_FALSE(missing.success);
    EXPECT_FALSE(missing.diagnostics.empty());

    svmp::RefinementDelta delta;
    svmp::VertexProvenanceRecord midpoint;
    midpoint.new_vertex_gid = 42;
    midpoint.parent_vertex_weights = {{0, 0.5}, {1, 0.5}};
    midpoint.reference_coordinate_weights = midpoint.parent_vertex_weights;
    delta.new_vertices.push_back(midpoint);

    FEFieldTransferOptions conservative;
    conservative.method = FEFieldTransferMethod::Conservative;
    conservative.require_all_vertices = true;
    std::fill(new_values.begin(), new_values.end(), Real(0));
    auto conserved = svmp::FE::systems::transferNodalFieldByVertexProvenance(
        old_mesh, new_mesh, delta, 1, old_values, new_values, conservative);

    EXPECT_TRUE(conserved.success);
    EXPECT_LE(conserved.conservation_error, 1.0e-12);
    const Real new_sum = new_values[0] + new_values[1] + new_values[2];
    EXPECT_NEAR(new_sum, 6.0, 1.0e-12);
}

#ifdef MESH_HAS_ADAPTIVITY
TEST(FEAdaptivityTransfer, OnMeshAdaptedRebuildsLayoutAndTransfersSolutionHistory)
{
    auto old_mesh = build_single_quad_mesh();
    auto mesh = build_single_quad_mesh();

    auto scalar_space = std::make_shared<H1Space>(ElementType::Quad4, 1);
    auto vector_space = std::make_shared<ProductSpace>(scalar_space, 2);

    FESystem sys(mesh);
    const auto temperature = sys.addField(FieldSpec{.name = "temperature", .space = scalar_space, .components = 1});
    const auto velocity = sys.addField(FieldSpec{.name = "velocity", .space = vector_space, .components = 2});
    sys.addCellKernel("mass_temperature", temperature, std::make_shared<MassKernel>(1.0));
    sys.addCellKernel("mass_velocity", velocity, std::make_shared<MassKernel>(1.0));
    sys.setup();

    const auto old_dofs = static_cast<std::size_t>(sys.dofHandler().getNumDofs());
    std::vector<Real> solution(old_dofs, Real(0));
    write_vertex_state(sys, temperature, old_mesh->local_mesh(), solution,
                       [](std::size_t, std::size_t, const std::array<svmp::real_t, 3>& xyz) {
                           return static_cast<Real>(xyz[0] + 2.0 * xyz[1]);
                       });
    write_vertex_state(sys, velocity, old_mesh->local_mesh(), solution,
                       [](std::size_t, std::size_t component, const std::array<svmp::real_t, 3>& xyz) {
                           return component == 0 ? static_cast<Real>(10.0 + xyz[0])
                                                 : static_cast<Real>(20.0 + xyz[1]);
                       });
    std::vector<Real> previous = solution;
    for (auto& value : previous) {
        value += 100.0;
    }

    svmp::AdaptivityManager manager(adaptivity_options());
    auto adapt = manager.refine(mesh->local_mesh(), {true}, nullptr);
    ASSERT_TRUE(adapt.success) << adapt.summary();
    ASSERT_NE(adapt.refinement_delta, nullptr);

    std::vector<Real> transferred_solution;
    std::vector<Real> transferred_previous;
    FEAdaptedStateTransferRequest request;
    request.solution = solution;
    request.previous_solution = previous;
    request.transferred_solution = &transferred_solution;
    request.transferred_previous_solution = &transferred_previous;

    const auto layout_before = sys.feLayoutRevisionState();
    auto& aux = sys.auxiliaryStateManager();
    aux.registerBlock(make_cell_auxiliary_spec(),
                      old_mesh->local_mesh().n_cells(),
                      std::vector<Real>{7.0});
    aux.setTransferHook("cell_history",
                        [](std::span<const Real> old_data,
                           std::size_t old_count,
                           std::size_t new_count,
                           std::span<Real> output) {
                            ASSERT_EQ(old_count, 1u);
                            ASSERT_FALSE(old_data.empty());
                            ASSERT_GE(output.size(), new_count);
                            for (std::size_t i = 0; i < new_count; ++i) {
                                output[i] = old_data.front() + static_cast<Real>(i);
                            }
                        });

    auto result = sys.onMeshAdapted(old_mesh->local_mesh(),
                                    mesh->local_mesh(),
                                    *adapt.refinement_delta,
                                    adaptivity_options(),
                                    request);

    EXPECT_TRUE(result.dof_handler_rebuilt);
    EXPECT_TRUE(result.constraint_layout_rebuilt);
    EXPECT_TRUE(result.sparsity_rebuilt);
    EXPECT_TRUE(result.solution_transferred);
    EXPECT_TRUE(result.previous_solution_transferred);
    EXPECT_TRUE(result.auxiliary_state_transfer_handled);
    EXPECT_GT(result.values_transferred, 0u);
    EXPECT_GT(result.layout_after.dof_layout, layout_before.dof_layout);
    EXPECT_EQ(transferred_solution.size(), static_cast<std::size_t>(sys.dofHandler().getNumDofs()));
    EXPECT_EQ(transferred_previous.size(), transferred_solution.size());

    const auto& aux_block = aux.getBlock("cell_history");
    ASSERT_EQ(aux_block.entityCount(), mesh->local_mesh().n_cells());
    ASSERT_EQ(aux_block.entityCount(), 4u);
    for (std::size_t i = 0; i < aux_block.entityCount(); ++i) {
        EXPECT_DOUBLE_EQ(aux_block.work()[i], 7.0 + static_cast<Real>(i));
    }

    const auto& adapted_mesh = mesh->local_mesh();
    ASSERT_EQ(adapted_mesh.n_vertices(), 9u);
    for (std::size_t v = 0; v < adapted_mesh.n_vertices(); ++v) {
        std::array<svmp::real_t, 3> xyz{0.0, 0.0, 0.0};
        for (int d = 0; d < adapted_mesh.dim(); ++d) {
            xyz[static_cast<std::size_t>(d)] =
                adapted_mesh.X_ref()[static_cast<std::size_t>(adapted_mesh.dim()) * v + static_cast<std::size_t>(d)];
        }
        const Real expected_temperature = static_cast<Real>(xyz[0] + 2.0 * xyz[1]);
        EXPECT_NEAR(read_vertex_state(sys, temperature, transferred_solution, v, 0),
                    expected_temperature,
                    1.0e-12);
        EXPECT_NEAR(read_vertex_state(sys, temperature, transferred_previous, v, 0),
                    expected_temperature + 100.0,
                    1.0e-12);
        EXPECT_NEAR(read_vertex_state(sys, velocity, transferred_solution, v, 0),
                    static_cast<Real>(10.0 + xyz[0]),
                    1.0e-12);
        EXPECT_NEAR(read_vertex_state(sys, velocity, transferred_solution, v, 1),
                    static_cast<Real>(20.0 + xyz[1]),
                    1.0e-12);
    }
}
#else
TEST(FEAdaptivityTransfer, OnMeshAdaptedCoverageRequiresMeshAdaptivity)
{
    GTEST_SKIP() << "MESH_HAS_ADAPTIVITY is not enabled";
}
#endif
