#include "LevelSet/LevelSetOptions.h"

#include <gtest/gtest.h>

#include <limits>
#include <variant>

namespace level_set = svmp::FE::level_set;

TEST(LevelSetOptions, DefaultsAreNeutral)
{
    const level_set::LevelSetTransportOptions options{};

    EXPECT_EQ(options.operator_tag, "level_set");
    EXPECT_EQ(options.transport_form, level_set::LevelSetTransportForm::Advective);
    EXPECT_EQ(options.level_set.field_name, "level_set");
    EXPECT_EQ(options.level_set.source, level_set::LevelSetFieldSource::Unknown);
    EXPECT_TRUE(options.level_set.auto_register_field);
    EXPECT_EQ(options.velocity.field_name, "Velocity");
    EXPECT_EQ(options.velocity.source, level_set::LevelSetVelocitySource::CoupledField);
    EXPECT_FALSE(options.velocity.auto_register_field);
    EXPECT_EQ(options.velocity.space, nullptr);
    EXPECT_DOUBLE_EQ(options.velocity.constant_value[0], 0.0);
    EXPECT_DOUBLE_EQ(options.velocity.constant_value[1], 0.0);
    EXPECT_DOUBLE_EQ(options.velocity.constant_value[2], 0.0);
    EXPECT_FALSE(options.supg.enabled);
    EXPECT_DOUBLE_EQ(options.supg.tau_scale, 0.5);
    EXPECT_DOUBLE_EQ(options.supg.velocity_epsilon, 1.0e-12);
    EXPECT_DOUBLE_EQ(options.supg.transient_scale, 2.0);
    EXPECT_TRUE(options.boundaries.inflow.empty());
    EXPECT_TRUE(options.boundaries.outflow.empty());
    EXPECT_FALSE(options.conservative_phase.enabled);
    EXPECT_EQ(options.conservative_phase.liquid_indicator.field_name,
              "liquid_indicator");
    EXPECT_EQ(options.conservative_phase.liquid_indicator.source,
              level_set::LevelSetFieldSource::Unknown);
    EXPECT_TRUE(
        options.conservative_phase.liquid_indicator.auto_register_field);
    EXPECT_EQ(options.conservative_phase.liquid_side,
              level_set::LevelSetPhaseSide::Negative);
    EXPECT_DOUBLE_EQ(options.conservative_phase.invariant_tolerance, 1.0e-12);
    EXPECT_DOUBLE_EQ(
        options.conservative_phase.component_activity_tolerance,
        1.0e-8);
    EXPECT_DOUBLE_EQ(options.conservative_phase.maximum_courant, 1.0);
    EXPECT_TRUE(options.conservative_phase.enforce_courant_limit);
    EXPECT_TRUE(options.conservative_phase.require_constant_preservation);
    EXPECT_FALSE(options.conservative_phase.write_flux_artifacts);
    EXPECT_EQ(options.conservative_phase.flux_artifact_cadence_steps, 1);
    EXPECT_FALSE(options.conservative_phase
                     .classify_nonprimary_components_as_satellites);
    EXPECT_TRUE(options.conservative_phase.fixed_flux_regions.empty());
    EXPECT_EQ(
        options.conservative_phase.boundary_flux_policy,
        level_set::LevelSetConservativePhaseBoundaryFluxPolicy::
            ClosedDomainDiscreteQFluxOnly);
    EXPECT_STREQ(
        level_set::levelSetConservativePhaseBoundaryFluxPolicyName(
            options.conservative_phase.boundary_flux_policy),
        "closed_domain_discrete_q_flux_only");
    EXPECT_DOUBLE_EQ(
        options.conservative_phase.impermeable_normal_velocity_tolerance,
        1.0e-10);
    EXPECT_FALSE(options.conservative_phase
                     .pointwise_impermeable_velocity_tolerance_explicitly_requested);
    EXPECT_TRUE(options.conservative_phase.reconcile_geometry);
    EXPECT_DOUBLE_EQ(options.conservative_phase.geometry_measure_tolerance,
                     1.0e-10);
    EXPECT_EQ(options.conservative_phase.geometry_correction_max_iterations,
              50);
    EXPECT_DOUBLE_EQ(
        options.conservative_phase.maximum_geometry_displacement_fraction,
        0.1);
}

TEST(LevelSetOptions, ConservativePhaseBoundaryFluxPoliciesAreFailClosed)
{
    using Policy =
        level_set::LevelSetConservativePhaseBoundaryFluxPolicy;

    EXPECT_TRUE(level_set::levelSetConservativePhaseBoundaryFluxSatisfied(
        Policy::ClosedDomainDiscreteQFluxOnly,
        5.0e-13,
        2.0e-13,
        1.0e-12));
    EXPECT_FALSE(level_set::levelSetConservativePhaseBoundaryFluxSatisfied(
        Policy::ClosedDomainDiscreteQFluxOnly,
        2.0e-6,
        2.0e-13,
        1.0e-12));
    EXPECT_TRUE(level_set::levelSetConservativePhaseBoundaryFluxSatisfied(
        Policy::GloballyBalancedDiscreteQFlux,
        2.0e-6,
        2.0e-13,
        1.0e-12));
    EXPECT_FALSE(level_set::levelSetConservativePhaseBoundaryFluxSatisfied(
        Policy::GloballyBalancedDiscreteQFlux,
        2.0e-6,
        2.0e-12,
        1.0e-12));
    EXPECT_FALSE(level_set::levelSetConservativePhaseBoundaryFluxSatisfied(
        Policy::GloballyBalancedDiscreteQFlux,
        std::numeric_limits<double>::infinity(),
        0.0,
        1.0e-12));
    EXPECT_STREQ(
        level_set::levelSetConservativePhaseBoundaryFluxPolicyName(
            Policy::GloballyBalancedDiscreteQFlux),
        "globally_balanced_discrete_q_flux");
}

TEST(LevelSetOptions, ExplicitTransportOptions)
{
    level_set::LevelSetTransportOptions options{};
    options.transport_form = level_set::LevelSetTransportForm::ConservativeDivergence;
    options.level_set.field_name = "phi";
    options.level_set.source = level_set::LevelSetFieldSource::PrescribedData;
    options.level_set.auto_register_field = false;
    options.velocity.field_name = "advecting_velocity";
    options.velocity.source = level_set::LevelSetVelocitySource::PrescribedData;
    options.velocity.auto_register_field = true;
    options.velocity.constant_value = {1.0, -2.0, 0.5};
    options.supg.enabled = true;
    options.supg.tau_scale = 0.25;
    options.supg.velocity_epsilon = 1.0e-8;
    options.supg.transient_scale = 1.5;
    options.boundaries.inflow.push_back(level_set::LevelSetInflowBoundary{
        .boundary_marker = 11,
        .value = svmp::FE::Real{2.5},
        .penalty_scale = 3.0,
    });
    options.boundaries.outflow.push_back(
        level_set::LevelSetOutflowBoundary{.boundary_marker = 12});

    EXPECT_EQ(options.level_set.field_name, "phi");
    EXPECT_EQ(options.transport_form, level_set::LevelSetTransportForm::ConservativeDivergence);
    EXPECT_EQ(options.level_set.source, level_set::LevelSetFieldSource::PrescribedData);
    EXPECT_FALSE(options.level_set.auto_register_field);
    EXPECT_EQ(options.velocity.field_name, "advecting_velocity");
    EXPECT_EQ(options.velocity.source, level_set::LevelSetVelocitySource::PrescribedData);
    EXPECT_TRUE(options.velocity.auto_register_field);
    EXPECT_EQ(options.velocity.space, nullptr);
    EXPECT_DOUBLE_EQ(options.velocity.constant_value[0], 1.0);
    EXPECT_DOUBLE_EQ(options.velocity.constant_value[1], -2.0);
    EXPECT_DOUBLE_EQ(options.velocity.constant_value[2], 0.5);
    EXPECT_TRUE(options.supg.enabled);
    EXPECT_DOUBLE_EQ(options.supg.tau_scale, 0.25);
    EXPECT_DOUBLE_EQ(options.supg.velocity_epsilon, 1.0e-8);
    EXPECT_DOUBLE_EQ(options.supg.transient_scale, 1.5);
    ASSERT_EQ(options.boundaries.inflow.size(), 1u);
    EXPECT_EQ(options.boundaries.inflow.front().boundary_marker, 11);
    EXPECT_DOUBLE_EQ(std::get<svmp::FE::Real>(options.boundaries.inflow.front().value), 2.5);
    EXPECT_DOUBLE_EQ(options.boundaries.inflow.front().penalty_scale, 3.0);
    ASSERT_EQ(options.boundaries.outflow.size(), 1u);
    EXPECT_EQ(options.boundaries.outflow.front().boundary_marker, 12);
}

TEST(LevelSetOptions, ConservationDiagnosticsDistinguishRunModes)
{
    const level_set::LevelSetTransportOptions plain{};
    EXPECT_EQ(
        level_set::levelSetConservationDiagnostic(plain),
        level_set::LevelSetConservationDiagnostic::PlainAdvectionNotConservative);
    EXPECT_STREQ(
        level_set::levelSetConservationDiagnosticName(
            level_set::levelSetConservationDiagnostic(plain)),
        "plain_level_set_advection_not_conservative");

    level_set::LevelSetTransportOptions conservative{};
    conservative.transport_form = level_set::LevelSetTransportForm::ConservativeDivergence;
    EXPECT_EQ(
        level_set::levelSetConservationDiagnostic(conservative),
        level_set::LevelSetConservationDiagnostic::
            ConservativeDivergenceAdvectionNotLocallyConservative);
    EXPECT_STREQ(
        level_set::levelSetConservationDiagnosticName(
            level_set::levelSetConservationDiagnostic(conservative)),
        "conservative_divergence_level_set_advection_not_locally_conservative");

    level_set::LevelSetTransportOptions phase_conservative{};
    phase_conservative.conservative_phase.enabled = true;
    EXPECT_EQ(
        level_set::levelSetConservationDiagnostic(phase_conservative),
        level_set::LevelSetConservationDiagnostic::
            ConservativePhaseIndicatorLocallyConservative);
    EXPECT_STREQ(
        level_set::levelSetConservationDiagnosticName(
            level_set::levelSetConservationDiagnostic(phase_conservative)),
        "conservative_p1_phase_indicator_locally_conservative");
}

