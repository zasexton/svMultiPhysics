#include "Physics/Formulations/NavierStokes/FreeSurface/SnapshotFunctionals.h"
#include "FE/Tests/Support/QuadCutSnapshotFixture.h"

#include <gtest/gtest.h>

#include <cmath>
#include <limits>
#include <numbers>

namespace {
namespace FE = svmp::FE;
namespace interfaces = FE::interfaces;
namespace free_surface = svmp::Physics::formulations::navier_stokes::free_surface;
using namespace FE::testing::quad_cut;
}

TEST(FreeSurfaceSnapshotFunctionals, EvaluatesContactMotionAndValidatesWallLawsOnBothSides)
{
    constexpr int interface_marker = 104;
    constexpr int wall_marker = 10;
    const SingleQuadBoundaryMesh mesh(wall_marker);
    const auto snapshot = verticalCutSnapshot(mesh, interface_marker, wall_marker);
    ASSERT_TRUE(snapshot);
    // The interface bisects the unit square and meets its bottom edge once.
    constexpr FE::Real expected_negative_wall_area = 0.5;
    constexpr FE::Real expected_contact_measure = 1.0;
    free_surface::FreeSurfaceDiscreteFunctionalParameters functional_parameters;
    functional_parameters.active_side =
        FE::geometry::CutIntegrationSide::Negative;
    functional_parameters.surface_tension = 2.0;
    functional_parameters.young_wall_coefficients.push_back(
        {10, std::acos(FE::Real{0.5})});
    functional_parameters.dynamic_contact_coefficients.push_back(
        {.boundary_marker = 10,
         .equilibrium_contact_angle_radians =
             std::acos(FE::Real{0.5}),
         .mobility = FE::Real{0.5},
         .slip_length = FE::Real{0.2},
         .dynamic_viscosity = FE::Real{0.4}});
    functional_parameters.volume_multiplier = 3.0;
    // The layout of the record and the owned rules the declared measures
    // integrate (D49 E4: the values are declared measured 0-forms).
    const auto functional =
        free_surface::freeSurfaceDiscreteFunctionalLayout(
            *snapshot, functional_parameters);
    EXPECT_EQ(functional.snapshot_revision_key,
              snapshot->revision().snapshot_revision_key);
    EXPECT_NEAR(ownedRuleMeasure(
                    *snapshot,
                    interfaces::LevelSetGeometryRuleRole::NegativeVolume),
                snapshot->ledger().owned_retained_negative_physical_volume,
                1.0e-14);
    EXPECT_NEAR(ownedRuleMeasure(
                    *snapshot,
                    interfaces::LevelSetGeometryRuleRole::Interface),
                snapshot->ledger().interface_physical_measure,
                1.0e-14);
    EXPECT_NEAR(ownedRuleMeasure(
                    *snapshot,
                    interfaces::LevelSetGeometryRuleRole::NegativeExteriorBoundary,
                    10),
                expected_negative_wall_area,
                1.0e-14);
    EXPECT_NEAR(ownedRuleMeasure(
                    *snapshot,
                    interfaces::LevelSetGeometryRuleRole::Contact,
                    10),
                expected_contact_measure,
                1.0e-14);
    ASSERT_EQ(functional.walls.size(), 1u);
    EXPECT_EQ(functional.walls.front().boundary_marker, 10);
    ASSERT_TRUE(functional.walls.front()
                    .equilibrium_contact_angle_radians.has_value());
    EXPECT_NEAR(*functional.walls.front()
                     .equilibrium_contact_angle_radians,
                std::acos(FE::Real{0.5}),
                1.0e-14);
    EXPECT_EQ(functional.walls.front().owned_wetted_wall_area, 0.0);
    EXPECT_EQ(functional.owned_liquid_volume, 0.0);
    EXPECT_DOUBLE_EQ(functional.surface_tension, 2.0);
    EXPECT_DOUBLE_EQ(functional.volume_multiplier, 3.0);

    interfaces::SnapshotVectorEvaluator velocity;
    velocity.value = [](FE::GlobalIndex,
                        const std::array<FE::Real, 3>&,
                        const FE::geometry::CutQuadratureProvenance&) {
        return std::array<FE::Real, 3>{{0.25, 0.0, 0.0}};
    };
    const auto dynamic_contact =
        free_surface::evaluateFreeSurfaceDynamicContactState(
            *snapshot, functional_parameters, velocity);
    EXPECT_EQ(dynamic_contact.snapshot_revision_key,
              snapshot->revision().snapshot_revision_key);
    ASSERT_EQ(dynamic_contact.walls.size(), 1u);
    const auto& contact = dynamic_contact.walls.front();
    EXPECT_EQ(contact.boundary_marker, wall_marker);
    EXPECT_EQ(contact.owned_quadrature_point_count, 1u);
    EXPECT_EQ(contact.owned_advancing_point_count, 1u);
    EXPECT_EQ(contact.owned_receding_point_count, 0u);
    EXPECT_EQ(contact.motion,
              free_surface::FreeSurfaceContactMotion::Advancing);
    EXPECT_NEAR(contact.owned_contact_measure,
                expected_contact_measure,
                1.0e-14);
    ASSERT_TRUE(contact.mean_dynamic_angle_radians.has_value());
    ASSERT_TRUE(contact.mean_dynamic_cosine.has_value());
    ASSERT_TRUE(contact.mean_contact_speed.has_value());
    ASSERT_TRUE(contact.mean_constitutive_residual.has_value());
    ASSERT_TRUE(contact.mean_absolute_constitutive_residual.has_value());
    EXPECT_NEAR(*contact.mean_dynamic_angle_radians,
                std::numbers::pi_v<FE::Real> / FE::Real{2.0},
                1.0e-14);
    EXPECT_NEAR(*contact.mean_dynamic_cosine, 0.0, 1.0e-14);
    EXPECT_NEAR(*contact.mean_contact_speed, 0.25, 1.0e-14);
    EXPECT_NEAR(contact.contact_speed_squared_integral,
                FE::Real{0.0625} * expected_contact_measure,
                1.0e-14);
    EXPECT_NEAR(*contact.mean_constitutive_residual, -0.5, 1.0e-14);
    EXPECT_NEAR(*contact.mean_absolute_constitutive_residual,
                0.5,
                1.0e-14);
    EXPECT_NEAR(contact.line_friction_dissipation,
                FE::Real{0.125} * expected_contact_measure,
                1.0e-14);
    EXPECT_EQ(contact.owned_wetted_wall_quadrature_point_count, 2u);
    EXPECT_NEAR(contact.owned_wetted_wall_measure,
                expected_negative_wall_area,
                1.0e-14);
    EXPECT_NEAR(contact.wall_slip_speed_squared_integral,
                FE::Real{0.0625} * expected_negative_wall_area,
                1.0e-14);
    EXPECT_NEAR(contact.wall_slip_dissipation,
                FE::Real{0.125} * expected_negative_wall_area,
                1.0e-14);
    ASSERT_TRUE(contact.mean_wall_slip_speed.has_value());
    EXPECT_NEAR(*contact.mean_wall_slip_speed, 0.25, 1.0e-14);
    EXPECT_NEAR(contact.mean_wall_tangential_velocity[0],
                0.25,
                1.0e-14);
    EXPECT_NEAR(contact.mean_wall_tangential_velocity[1],
                0.0,
                1.0e-14);
    EXPECT_NEAR(contact.mean_wall_normal[0], 0.0, 1.0e-14);
    EXPECT_NEAR(contact.mean_wall_normal[1], -1.0, 1.0e-14);
    EXPECT_NEAR(contact.mean_contact_position[0], 0.5, 1.0e-14);
    EXPECT_NEAR(contact.mean_contact_position[1], 0.0, 1.0e-14);
    EXPECT_NEAR(contact.mean_footprint_direction[0], 1.0, 1.0e-14);
    EXPECT_NEAR(contact.mean_footprint_direction[1], 0.0, 1.0e-14);
    EXPECT_NEAR(contact.contact_line_tangent_integral[2],
                expected_contact_measure,
                1.0e-14);
    EXPECT_NEAR(contact.mean_contact_line_tangent[0], 0.0, 1.0e-14);
    EXPECT_NEAR(contact.mean_contact_line_tangent[1], 0.0, 1.0e-14);
    EXPECT_NEAR(contact.mean_contact_line_tangent[2], 1.0, 1.0e-14);
    EXPECT_NEAR(dynamic_contact.owned_contact_measure,
                contact.owned_contact_measure,
                1.0e-14);
    EXPECT_NEAR(dynamic_contact.line_friction_dissipation,
                contact.line_friction_dissipation,
                1.0e-14);
    EXPECT_NEAR(dynamic_contact.owned_wetted_wall_measure,
                contact.owned_wetted_wall_measure,
                1.0e-14);
    EXPECT_NEAR(dynamic_contact.wall_slip_dissipation,
                contact.wall_slip_dissipation,
                1.0e-14);
    EXPECT_NEAR(dynamic_contact.total_dissipation,
                contact.line_friction_dissipation +
                    contact.wall_slip_dissipation,
                1.0e-14);

    auto prescribed_slip_parameters = functional_parameters;
    auto& prescribed_slip_coefficient =
        prescribed_slip_parameters.dynamic_contact_coefficients.front();
    prescribed_slip_coefficient.law =
        free_surface::FreeSurfaceContactLaw::PrescribedAngle;
    prescribed_slip_coefficient.mobility = 0.0;
    const auto prescribed_slip =
        free_surface::evaluateFreeSurfaceDynamicContactState(
            *snapshot, prescribed_slip_parameters, velocity);
    ASSERT_EQ(prescribed_slip.walls.size(), 1u);
    const auto& prescribed_wall = prescribed_slip.walls.front();
    EXPECT_EQ(prescribed_wall.law,
              free_surface::FreeSurfaceContactLaw::PrescribedAngle);
    EXPECT_DOUBLE_EQ(prescribed_wall.mobility, 0.0);
    EXPECT_NEAR(prescribed_wall.wall_slip_dissipation,
                contact.wall_slip_dissipation,
                1.0e-14);
    EXPECT_DOUBLE_EQ(prescribed_wall.line_friction_dissipation, 0.0);
    EXPECT_DOUBLE_EQ(prescribed_slip.line_friction_dissipation, 0.0);
    ASSERT_TRUE(
        prescribed_wall.mean_constitutive_residual.has_value());
    EXPECT_NEAR(*prescribed_wall.mean_constitutive_residual,
                -functional_parameters.surface_tension * 0.5,
                1.0e-14);
    EXPECT_NEAR(prescribed_slip.total_dissipation,
                prescribed_wall.wall_slip_dissipation,
                1.0e-14);

    EXPECT_THROW(
        (void)free_surface::evaluateFreeSurfaceDynamicContactState(
            *snapshot, functional_parameters, {}),
        std::invalid_argument);
    auto invalid_contact_parameters = functional_parameters;
    invalid_contact_parameters.dynamic_contact_coefficients.front()
        .slip_length = 0.0;
    EXPECT_THROW(
        (void)free_surface::evaluateFreeSurfaceDynamicContactState(
            *snapshot, invalid_contact_parameters, velocity),
        std::invalid_argument);
    invalid_contact_parameters = functional_parameters;
    invalid_contact_parameters.dynamic_contact_coefficients.front()
        .dynamic_viscosity = 0.0;
    EXPECT_THROW(
        (void)free_surface::evaluateFreeSurfaceDynamicContactState(
            *snapshot, invalid_contact_parameters, velocity),
        std::invalid_argument);
    invalid_contact_parameters = prescribed_slip_parameters;
    invalid_contact_parameters.dynamic_contact_coefficients.front()
        .mobility = 0.5;
    EXPECT_THROW(
        (void)free_surface::evaluateFreeSurfaceDynamicContactState(
            *snapshot, invalid_contact_parameters, velocity),
        std::invalid_argument);

    auto positive_parameters = functional_parameters;
    positive_parameters.active_side =
        FE::geometry::CutIntegrationSide::Positive;
    const auto positive_contact =
        free_surface::evaluateFreeSurfaceDynamicContactState(
            *snapshot, positive_parameters, velocity);
    ASSERT_EQ(positive_contact.walls.size(), 1u);
    ASSERT_TRUE(
        positive_contact.walls.front().mean_contact_speed.has_value());
    EXPECT_NEAR(*positive_contact.walls.front().mean_contact_speed,
                -0.25,
                1.0e-14);
    EXPECT_EQ(positive_contact.walls.front().motion,
              free_surface::FreeSurfaceContactMotion::Receding);

    auto invalid_parameters = functional_parameters;
    invalid_parameters.surface_tension = -1.0;
    EXPECT_THROW(
        (void)free_surface::freeSurfaceDiscreteFunctionalLayout(
            *snapshot, invalid_parameters),
        std::invalid_argument);
    invalid_parameters = functional_parameters;
    invalid_parameters.young_wall_coefficients.front()
        .equilibrium_contact_angle_radians = 0.0;
    EXPECT_THROW(
        (void)free_surface::freeSurfaceDiscreteFunctionalLayout(
            *snapshot, invalid_parameters),
        std::invalid_argument);
    invalid_parameters = functional_parameters;
    invalid_parameters.young_wall_coefficients.push_back(
        invalid_parameters.young_wall_coefficients.front());
    EXPECT_THROW(
        (void)free_surface::freeSurfaceDiscreteFunctionalLayout(
            *snapshot, invalid_parameters),
        std::invalid_argument);
}
