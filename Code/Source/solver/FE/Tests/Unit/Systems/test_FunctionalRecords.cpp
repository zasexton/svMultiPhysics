/* Copyright (c) Stanford University, The Regents of the
 * University of California, and others.
 *
 * All Rights Reserved.
 *
 * See License file.
 */

// Free-surface record tests that were FE system tests before the records
// moved to the Navier-Stokes free-surface physics.

#include <gtest/gtest.h>

#include "Systems/FunctionalRecords.h"

#include "Core/FEException.h"
#include "Spaces/H1Space.h"
#include "Spaces/ProductSpace.h"
#include "Systems/FESystem.h"
#include "Systems/FreeSurfaceFunctionalRounding.h"
#include "Mesh/Core/MeshBase.h"
#include "Mesh/Mesh.h"
#include "Mesh/Topology/CellShape.h"

#include <cmath>
#include <cstdint>
#include <limits>
#include <memory>
#include <string>
#include <vector>

using svmp::FE::systems::functionalRecords;

using svmp::CellFamily;
using svmp::CellShape;
using svmp::Mesh;
using svmp::MeshBase;

using svmp::FE::ElementType;
using svmp::FE::Real;

using svmp::FE::spaces::H1Space;
using svmp::FE::spaces::ProductSpace;

using svmp::FE::systems::FESystem;
using svmp::FE::systems::FieldSourceKind;
using svmp::FE::systems::FieldSpec;

namespace {

std::shared_ptr<Mesh> build_single_quad_mesh()
{
    auto base = std::make_shared<MeshBase>();

    const std::vector<svmp::real_t> X_ref = {
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
    base->build_from_arrays(/*spatial_dim=*/2, X_ref, cell2vertex_offsets, cell2vertex, {shape});
    base->finalize();

    return svmp::create_mesh(std::move(base));
}

std::shared_ptr<Mesh> build_single_tetra_mesh()
{
    auto base = std::make_shared<MeshBase>();

    const std::vector<svmp::real_t> X_ref = {
        0.0, 0.0, 0.0,
        1.0, 0.0, 0.0,
        0.0, 1.0, 0.0,
        0.0, 0.0, 1.0
    };
    const std::vector<svmp::offset_t> cell2vertex_offsets = {0, 4};
    const std::vector<svmp::index_t> cell2vertex = {0, 1, 2, 3};

    CellShape shape{};
    shape.family = CellFamily::Tetra;
    shape.num_corners = 4;
    shape.order = 1;
    base->build_from_arrays(/*spatial_dim=*/3, X_ref, cell2vertex_offsets, cell2vertex, {shape});
    base->finalize();

    return svmp::create_mesh(std::move(base));
}

Real recursiveSum(const std::vector<Real>& terms)
{
    Real sum{0.0};
    for (const auto term : terms) {
        sum += term;
    }
    return sum;
}

std::shared_ptr<svmp::Mesh> buildRoundingSingleQuadMesh()
{
    auto base = std::make_shared<svmp::MeshBase>();
    const std::vector<svmp::real_t> coordinates = {
        0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0};
    const std::vector<svmp::offset_t> offsets = {0, 4};
    const std::vector<svmp::index_t> vertices = {0, 1, 2, 3};
    svmp::CellShape shape{};
    shape.family = svmp::CellFamily::Quad;
    shape.num_corners = 4;
    shape.order = 1;
    base->build_from_arrays(
        /*spatial_dim=*/2, coordinates, offsets, vertices, {shape});
    base->finalize();
    return svmp::create_mesh(std::move(base));
}

bool fixedFiveHundredTwelveUlpNear(Real a, Real b)
{
    const Real scale = std::max({Real{1.0}, std::abs(a), std::abs(b)});
    return std::abs(a - b) <=
           Real{512.0} * std::numeric_limits<Real>::epsilon() * scale;
}

} // namespace

TEST(FunctionalRecords,
     CompleteStaticConservativeBodyForceRequiresActiveVolumeEnergy)
{
    auto mesh = build_single_quad_mesh();
    auto scalar_space =
        std::make_shared<H1Space>(ElementType::Quad4, /*order=*/1);
    auto velocity_space =
        std::make_shared<ProductSpace>(scalar_space, /*components=*/2);

    FESystem sys(mesh);
    const auto phi = sys.addField(
        FieldSpec{.name = "phi_static_body_force_contract",
                  .space = scalar_space,
                  .components = 1});
    const auto velocity = sys.addField(
        FieldSpec{.name = "u_static_body_force_contract",
                  .space = velocity_space,
                  .components = 2});

    svmp::FE::systems::FreeSurfaceDiscreteFunctionalDeclaration declaration{
        .interface_marker = 91,
        .level_set_field = phi,
        .velocity_field = velocity,
        .geometry_domain_id = "static_body_force_contract",
        .parameters =
            svmp::FE::interfaces::FreeSurfaceDiscreteFunctionalParameters{
                .liquid_side =
                    svmp::FE::geometry::CutIntegrationSide::Negative,
                .surface_tension = 1.0,
            },
        .static_conservative_body_force_complete = true,
        .owner_component = "FESystemTest.StaticBodyForceContract",
    };

    EXPECT_THROW(
        functionalRecords(sys).declareFunctional(declaration),
        svmp::FE::InvalidArgumentException);
    EXPECT_TRUE(functionalRecords(sys).functionalDeclarations().empty());

    declaration.active_volume_energy_parameters =
        svmp::FE::interfaces::FreeSurfaceActiveVolumeEnergyParameters{
            .liquid_side =
                svmp::FE::geometry::CutIntegrationSide::Negative,
            .density = 1.0,
        };
    EXPECT_NO_THROW(functionalRecords(sys).declareFunctional(declaration));
    ASSERT_EQ(functionalRecords(sys).functionalDeclarations().size(), 1u);
    EXPECT_TRUE(functionalRecords(sys).functionalDeclarations()
                    .front()
                    .static_conservative_body_force_complete);
}

TEST(FunctionalRecords,
     AcceptedRecordAcceptsRegroupedLiquidVolumeAndRejectsInconsistency)
{
    using svmp::FE::ElementType;
    using svmp::FE::geometry::CutIntegrationSide;
    namespace interfaces = svmp::FE::interfaces;
    namespace systems = svmp::FE::systems;

    auto mesh = buildRoundingSingleQuadMesh();
    auto scalar_space = std::make_shared<svmp::FE::spaces::H1Space>(
        ElementType::Quad4, /*order=*/1);
    auto velocity_space = std::make_shared<svmp::FE::spaces::ProductSpace>(
        scalar_space, /*components=*/2);
    systems::FESystem system(mesh);
    const auto phi = system.addField(systems::FieldSpec{
        .name = "phi_rounding_bound", .space = scalar_space, .components = 1});
    const auto velocity = system.addField(systems::FieldSpec{
        .name = "u_rounding_bound", .space = velocity_space, .components = 2});
    ASSERT_NO_THROW(functionalRecords(system).declareFunctional(
        systems::FreeSurfaceDiscreteFunctionalDeclaration{
            .interface_marker = 91,
            .level_set_field = phi,
            .velocity_field = velocity,
            .geometry_domain_id = "rounding_bound",
            .parameters =
                interfaces::FreeSurfaceDiscreteFunctionalParameters{
                    .liquid_side = CutIntegrationSide::Negative,
                    .surface_tension = 1.0,
                },
            .active_volume_energy_parameters =
                interfaces::FreeSurfaceActiveVolumeEnergyParameters{
                    .liquid_side = CutIntegrationSide::Negative,
                    .density = 1.0,
                },
            .owner_component = "FreeSurfaceFunctionalRounding.Record",
        }));
    ASSERT_NO_THROW(system.setup({}));

    // 300000 equal weights, as on a uniform 3D mesh: the state sums them per
    // four-point rule, the energy point by point.  The two sums differ by
    // about 6.5e-12 relative, far above 512 ulp and below the derived bound.
    constexpr std::size_t point_count = 300000u;
    constexpr std::size_t points_per_rule = 4u;
    const std::vector<Real> weights(point_count, Real{0.1});
    const Real pointwise_volume = recursiveSum(weights);
    Real rulewise_volume{0.0};
    for (std::size_t begin = 0u; begin < point_count;
         begin += points_per_rule) {
        Real rule{0.0};
        for (std::size_t i = begin; i < begin + points_per_rule; ++i) {
            rule += weights[i];
        }
        rulewise_volume += rule;
    }
    ASSERT_FALSE(fixedFiveHundredTwelveUlpNear(pointwise_volume, rulewise_volume));

    const interfaces::FreeSurfaceGeometryRevision revision{
        .source_id = "field:" + std::to_string(phi),
        .domain_id = "rounding_bound",
        .interface_marker = 91,
        .isovalue = 0.0,
        .source_layout_revision = 3u,
        .source_value_revision = 9u,
        .mesh_geometry_revision = 4u,
        .mesh_topology_revision = 5u,
        .ownership_revision = 6u,
        .numbering_revision = 7u,
        .quadrature_policy_key = 8u,
        .snapshot_revision_key = 101u,
    };
    const Real liquid_gas_area = 0.4;
    const auto make_states = [&](Real energy_volume) {
        interfaces::FreeSurfaceDiscreteFunctionalState state{
            .snapshot_revision_key = 101u,
            .liquid_side = CutIntegrationSide::Negative,
            .surface_tension = 1.0,
            .volume_multiplier = 0.0,
            .owned_liquid_volume = rulewise_volume,
            .owned_liquid_gas_area = liquid_gas_area,
            .liquid_gas_surface_energy = Real{1.0} * liquid_gas_area,
            .volume_constraint_potential = Real{0.0} * rulewise_volume,
        };
        state.total_potential = state.liquid_gas_surface_energy +
                                state.young_wall_energy +
                                state.volume_constraint_potential;
        interfaces::FreeSurfaceActiveVolumeEnergyState energy{
            .snapshot_revision_key = 101u,
            .liquid_side = CutIntegrationSide::Negative,
            .density = 1.0,
            .owned_quadrature_point_count = point_count,
            .owned_liquid_volume = energy_volume,
        };
        return std::vector<systems::AcceptedFreeSurfaceDiscreteFunctionalState>{
            systems::AcceptedFreeSurfaceDiscreteFunctionalState{
                .interface_marker = 91,
                .geometry_revision = revision,
                .cut_topology_revision = 102u,
                .state = state,
                .active_volume_energy = energy,
            }};
    };

    EXPECT_NO_THROW(functionalRecords(system).recordAcceptedFunctionals(
        1u, 0.1, 0.1, 13u, 13u, make_states(pointwise_volume)));
    ASSERT_EQ(functionalRecords(system).acceptedFunctionalHistory().size(), 1u);
    EXPECT_EQ(functionalRecords(system).acceptedFunctionalHistory()
                  .front()
                  .active_volume_energy->owned_liquid_volume,
              pointwise_volume);

    for (const Real relative : {Real{1.0e-10}, Real{-1.0e-10}}) {
        try {
            functionalRecords(system).recordAcceptedFunctionals(
                2u,
                0.2,
                0.1,
                14u,
                14u,
                make_states(pointwise_volume * (Real{1.0} + relative)));
            ADD_FAILURE() << "a 1e-10 relative volume inconsistency was accepted";
        } catch (const svmp::FE::InvalidArgumentException& error) {
            EXPECT_NE(std::string(error.what()).find(
                          "active-volume energy liquid measure is inconsistent"),
                      std::string::npos)
                << error.what();
        }
    }
    EXPECT_EQ(functionalRecords(system).acceptedFunctionalHistory().size(), 1u);
}
