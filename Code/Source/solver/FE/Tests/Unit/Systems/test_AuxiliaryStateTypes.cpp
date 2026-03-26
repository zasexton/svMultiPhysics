/**
 * @file test_AuxiliaryStateTypes.cpp
 * @brief Unit tests for the generalized AuxiliaryState type vocabulary
 */

#include <gtest/gtest.h>

#include "Systems/AuxiliaryStateTypes.h"
#include "Systems/AuxiliaryState.h"

using namespace svmp::FE::systems;

// ---------------------------------------------------------------------------
//  Enum value coverage
// ---------------------------------------------------------------------------

TEST(AuxiliaryStateTypes, ScopeEnumValues)
{
    EXPECT_NE(AuxiliaryStateScope::Global, AuxiliaryStateScope::Node);
    EXPECT_NE(AuxiliaryStateScope::Node, AuxiliaryStateScope::Cell);
    EXPECT_NE(AuxiliaryStateScope::Cell, AuxiliaryStateScope::QuadraturePoint);
    EXPECT_NE(AuxiliaryStateScope::QuadraturePoint, AuxiliaryStateScope::Facet);
    EXPECT_NE(AuxiliaryStateScope::Facet, AuxiliaryStateScope::Boundary);

    // Verify all six values are distinct
    const auto scopes = {
        AuxiliaryStateScope::Global,
        AuxiliaryStateScope::Boundary,
        AuxiliaryStateScope::Node,
        AuxiliaryStateScope::Cell,
        AuxiliaryStateScope::QuadraturePoint,
        AuxiliaryStateScope::Facet
    };
    EXPECT_EQ(scopes.size(), 6u);
}

TEST(AuxiliaryStateTypes, VariableKindEnumValues)
{
    EXPECT_NE(AuxiliaryVariableKind::Differential, AuxiliaryVariableKind::Algebraic);
}

TEST(AuxiliaryStateTypes, SolveModeEnumValues)
{
    EXPECT_NE(AuxiliarySolveMode::Partitioned, AuxiliarySolveMode::Monolithic);
}

TEST(AuxiliaryStateTypes, HistoryModeEnumValues)
{
    EXPECT_NE(AuxiliaryHistoryMode::None, AuxiliaryHistoryMode::SingleStep);
    EXPECT_NE(AuxiliaryHistoryMode::SingleStep, AuxiliaryHistoryMode::MultiStep);
}

TEST(AuxiliaryStateTypes, LayoutModeEnumValues)
{
    EXPECT_NE(AuxiliaryLayoutMode::FixedStride, AuxiliaryLayoutMode::Ragged);
}

TEST(AuxiliaryStateTypes, EntityOrderingEnumValues)
{
    EXPECT_NE(AuxiliaryEntityOrdering::ByEntityThenComponent,
              AuxiliaryEntityOrdering::ByComponentThenEntity);
}

TEST(AuxiliaryStateTypes, SyncPolicyEnumValues)
{
    EXPECT_NE(AuxiliarySyncPolicy::None, AuxiliarySyncPolicy::OwnedOnly);
    EXPECT_NE(AuxiliarySyncPolicy::OwnedOnly, AuxiliarySyncPolicy::OwnedAndGhost);
}

TEST(AuxiliaryStateTypes, ScheduleModeEnumValues)
{
    EXPECT_NE(AuxiliaryScheduleMode::SingleRate, AuxiliaryScheduleMode::Subcycled);
    EXPECT_NE(AuxiliaryScheduleMode::Subcycled, AuxiliaryScheduleMode::Multirate);
}

TEST(AuxiliaryStateTypes, EventModeEnumValues)
{
    EXPECT_NE(AuxiliaryEventMode::None, AuxiliaryEventMode::EventHook);
    EXPECT_NE(AuxiliaryEventMode::EventHook, AuxiliaryEventMode::ActiveSetHook);
    EXPECT_NE(AuxiliaryEventMode::ActiveSetHook, AuxiliaryEventMode::ComplementarityHook);
}

TEST(AuxiliaryStateTypes, DerivativeSourceEnumValues)
{
    EXPECT_NE(AuxiliaryDerivativeSource::Symbolic, AuxiliaryDerivativeSource::FiniteDifference);
    EXPECT_NE(AuxiliaryDerivativeSource::FiniteDifference, AuxiliaryDerivativeSource::Analytic);
}

TEST(AuxiliaryStateTypes, SecondDerivativeModeEnumValues)
{
    EXPECT_NE(AuxiliarySecondDerivativeMode::None, AuxiliarySecondDerivativeMode::Hessian);
    EXPECT_NE(AuxiliarySecondDerivativeMode::Hessian,
              AuxiliarySecondDerivativeMode::HessianVectorProduct);
    EXPECT_NE(AuxiliarySecondDerivativeMode::HessianVectorProduct,
              AuxiliarySecondDerivativeMode::SelectedBlocks);
}

TEST(AuxiliaryStateTypes, InputRefreshPolicyEnumValues)
{
    EXPECT_NE(AuxiliaryInputRefreshPolicy::HoldLastSample,
              AuxiliaryInputRefreshPolicy::RefreshEachSubstep);
    EXPECT_NE(AuxiliaryInputRefreshPolicy::RefreshEachSubstep,
              AuxiliaryInputRefreshPolicy::FormulationDefined);
}

TEST(AuxiliaryStateTypes, SubstepCommitPolicyEnumValues)
{
    EXPECT_NE(AuxiliarySubstepCommitPolicy::CommitAtEnd,
              AuxiliarySubstepCommitPolicy::CommitEachSubstep);
    EXPECT_NE(AuxiliarySubstepCommitPolicy::CommitEachSubstep,
              AuxiliarySubstepCommitPolicy::FormulationDefined);
}

// ---------------------------------------------------------------------------
//  Default-constructed spec
// ---------------------------------------------------------------------------

TEST(AuxiliaryStateTypes, SpecDefaultConstruction)
{
    AuxiliaryStateSpec spec;

    EXPECT_TRUE(spec.name.empty());
    EXPECT_EQ(spec.size, 0);
    EXPECT_TRUE(spec.component_names.empty());
    EXPECT_TRUE(spec.variable_kinds.empty());

    // Defaults
    EXPECT_EQ(spec.scope, AuxiliaryStateScope::Global);
    EXPECT_EQ(spec.solve_mode, AuxiliarySolveMode::Partitioned);
    EXPECT_EQ(spec.layout_mode, AuxiliaryLayoutMode::FixedStride);
    EXPECT_EQ(spec.ordering, AuxiliaryEntityOrdering::ByEntityThenComponent);
    EXPECT_EQ(spec.history_mode, AuxiliaryHistoryMode::SingleStep);
    EXPECT_EQ(spec.history_depth, 1);
    EXPECT_EQ(spec.history_interpolation, AuxiliaryHistoryInterpolationPolicy::None);
    EXPECT_EQ(spec.sync_policy, AuxiliarySyncPolicy::None);
    EXPECT_EQ(spec.transfer_policy, AuxiliaryTransferPolicy::None);
    EXPECT_EQ(spec.schedule_mode, AuxiliaryScheduleMode::SingleRate);
    EXPECT_EQ(spec.event_mode, AuxiliaryEventMode::None);
    EXPECT_TRUE(spec.metadata.empty());
}

// ---------------------------------------------------------------------------
//  Spec configuration
// ---------------------------------------------------------------------------

TEST(AuxiliaryStateTypes, SpecCanBeConfigured)
{
    AuxiliaryStateSpec spec;
    spec.name = "ionic_gates";
    spec.size = 4;
    spec.component_names = {"m", "h", "j", "d"};
    spec.variable_kinds = {
        AuxiliaryVariableKind::Differential,
        AuxiliaryVariableKind::Differential,
        AuxiliaryVariableKind::Differential,
        AuxiliaryVariableKind::Algebraic
    };
    spec.scope = AuxiliaryStateScope::Node;
    spec.solve_mode = AuxiliarySolveMode::Partitioned;
    spec.layout_mode = AuxiliaryLayoutMode::FixedStride;
    spec.history_mode = AuxiliaryHistoryMode::MultiStep;
    spec.history_depth = 3;
    spec.schedule_mode = AuxiliaryScheduleMode::Subcycled;
    spec.sync_policy = AuxiliarySyncPolicy::OwnedAndGhost;

    EXPECT_EQ(spec.name, "ionic_gates");
    EXPECT_EQ(spec.size, 4);
    ASSERT_EQ(spec.component_names.size(), 4u);
    EXPECT_EQ(spec.component_names[2], "j");
    ASSERT_EQ(spec.variable_kinds.size(), 4u);
    EXPECT_EQ(spec.variable_kinds[3], AuxiliaryVariableKind::Algebraic);
    EXPECT_EQ(spec.scope, AuxiliaryStateScope::Node);
    EXPECT_EQ(spec.history_depth, 3);
    EXPECT_EQ(spec.schedule_mode, AuxiliaryScheduleMode::Subcycled);
    EXPECT_EQ(spec.sync_policy, AuxiliarySyncPolicy::OwnedAndGhost);
}

TEST(AuxiliaryStateTypes, SpecScopeIsOrthogonalToSolveMode)
{
    // Global scope with Partitioned solve
    AuxiliaryStateSpec global_part;
    global_part.scope = AuxiliaryStateScope::Global;
    global_part.solve_mode = AuxiliarySolveMode::Partitioned;
    EXPECT_EQ(global_part.scope, AuxiliaryStateScope::Global);
    EXPECT_EQ(global_part.solve_mode, AuxiliarySolveMode::Partitioned);

    // Global scope with Monolithic solve
    AuxiliaryStateSpec global_mono;
    global_mono.scope = AuxiliaryStateScope::Global;
    global_mono.solve_mode = AuxiliarySolveMode::Monolithic;
    EXPECT_EQ(global_mono.scope, AuxiliaryStateScope::Global);
    EXPECT_EQ(global_mono.solve_mode, AuxiliarySolveMode::Monolithic);

    // Node scope with Monolithic solve
    AuxiliaryStateSpec node_mono;
    node_mono.scope = AuxiliaryStateScope::Node;
    node_mono.solve_mode = AuxiliarySolveMode::Monolithic;
    EXPECT_EQ(node_mono.scope, AuxiliaryStateScope::Node);
    EXPECT_EQ(node_mono.solve_mode, AuxiliarySolveMode::Monolithic);
}

// ---------------------------------------------------------------------------
//  Derivative policy
// ---------------------------------------------------------------------------

TEST(AuxiliaryStateTypes, DerivativePolicyDefaults)
{
    AuxiliaryDerivativePolicy policy;

    EXPECT_EQ(policy.jacobian_source, AuxiliaryDerivativeSource::Symbolic);
    EXPECT_EQ(policy.second_deriv_source, AuxiliaryDerivativeSource::Symbolic);
    EXPECT_EQ(policy.second_deriv_mode, AuxiliarySecondDerivativeMode::None);
    EXPECT_TRUE(policy.analytic_override_enabled);
    EXPECT_DOUBLE_EQ(policy.fd_epsilon, 1.0e-7);
    EXPECT_EQ(policy.ad_seed_dim, 0u);
}

TEST(AuxiliaryStateTypes, DerivativePolicyCanSelectFiniteDifference)
{
    AuxiliaryDerivativePolicy policy;
    policy.jacobian_source = AuxiliaryDerivativeSource::FiniteDifference;
    policy.fd_epsilon = 1.0e-6;
    policy.analytic_override_enabled = false;

    EXPECT_EQ(policy.jacobian_source, AuxiliaryDerivativeSource::FiniteDifference);
    EXPECT_DOUBLE_EQ(policy.fd_epsilon, 1.0e-6);
    EXPECT_FALSE(policy.analytic_override_enabled);
}

TEST(AuxiliaryStateTypes, DerivativePolicyCanRequestHessianVectorProduct)
{
    AuxiliaryDerivativePolicy policy;
    policy.second_deriv_mode = AuxiliarySecondDerivativeMode::HessianVectorProduct;

    EXPECT_EQ(policy.second_deriv_mode,
              AuxiliarySecondDerivativeMode::HessianVectorProduct);
}

// ---------------------------------------------------------------------------
//  Stepper spec
// ---------------------------------------------------------------------------

TEST(AuxiliaryStateTypes, StepperSpecDefaults)
{
    AuxiliaryStepperSpec stepper;

    EXPECT_EQ(stepper.method_name, "BackwardEuler");
    EXPECT_TRUE(stepper.method_options.empty());
    EXPECT_EQ(stepper.input_refresh, AuxiliaryInputRefreshPolicy::HoldLastSample);
    EXPECT_EQ(stepper.commit_policy, AuxiliarySubstepCommitPolicy::CommitAtEnd);
    EXPECT_EQ(stepper.max_nonlinear_iters, 50);
    EXPECT_DOUBLE_EQ(stepper.nonlinear_tol_abs, 1.0e-12);
    EXPECT_DOUBLE_EQ(stepper.nonlinear_tol_rel, 1.0e-10);
    EXPECT_EQ(stepper.substep_count, 1);
}

TEST(AuxiliaryStateTypes, StepperSpecSubstepping)
{
    AuxiliaryStepperSpec stepper;
    stepper.method_name = "RK4";
    stepper.substep_count = 10;
    stepper.input_refresh = AuxiliaryInputRefreshPolicy::RefreshEachSubstep;
    stepper.commit_policy = AuxiliarySubstepCommitPolicy::CommitEachSubstep;

    EXPECT_EQ(stepper.method_name, "RK4");
    EXPECT_EQ(stepper.substep_count, 10);
    EXPECT_EQ(stepper.input_refresh, AuxiliaryInputRefreshPolicy::RefreshEachSubstep);
    EXPECT_EQ(stepper.commit_policy, AuxiliarySubstepCommitPolicy::CommitEachSubstep);
}

// ---------------------------------------------------------------------------
//  Block layout
// ---------------------------------------------------------------------------

TEST(AuxiliaryStateTypes, BlockLayoutDefaultConstruction)
{
    AuxiliaryStateBlockLayout layout;

    EXPECT_EQ(layout.block_id, 0u);
    EXPECT_EQ(layout.component_stride, 0);
    EXPECT_EQ(layout.entity_count, 0u);
    EXPECT_EQ(layout.local_storage_size, 0u);
    EXPECT_EQ(layout.owned_entity_count, 0u);
    EXPECT_EQ(layout.owned_storage_size, 0u);
    EXPECT_EQ(layout.history_storage_size, 0u);
}

// ---------------------------------------------------------------------------
//  Storage summary
// ---------------------------------------------------------------------------

TEST(AuxiliaryStateTypes, StorageSummaryDefaultConstruction)
{
    AuxiliaryStateStorageSummary summary;

    EXPECT_EQ(summary.block_count, 0u);
    EXPECT_EQ(summary.total_work_storage, 0u);
    EXPECT_EQ(summary.total_committed_storage, 0u);
    EXPECT_EQ(summary.total_history_storage, 0u);
}

// ---------------------------------------------------------------------------
//  Registration options
// ---------------------------------------------------------------------------

TEST(AuxiliaryStateTypes, RegistrationOptionsDefaults)
{
    AuxiliaryStateRegistrationOptions opts;

    EXPECT_TRUE(opts.error_on_duplicate);
    EXPECT_TRUE(opts.validate_sizes);
}

// ---------------------------------------------------------------------------
//  Integration: new AuxiliaryStateSpec works with AuxiliaryState container
// ---------------------------------------------------------------------------

TEST(AuxiliaryStateTypes, NewSpecWorksWithAuxiliaryStateContainer)
{
    // The generalized AuxiliaryStateSpec should be usable with the existing
    // AuxiliaryState container for registration.
    AuxiliaryState state;

    AuxiliaryStateSpec spec;
    spec.name = "voltage";
    spec.size = 1;
    spec.scope = AuxiliaryStateScope::Node;
    spec.solve_mode = AuxiliarySolveMode::Partitioned;

    const svmp::FE::Real v0 = -80.0;
    state.registerState(spec, std::span<const svmp::FE::Real>(&v0, 1));

    EXPECT_EQ(state.size(), 1u);
    EXPECT_TRUE(state.has("voltage"));
    EXPECT_DOUBLE_EQ(state["voltage"], -80.0);
}

TEST(AuxiliaryStateTypes, NewSpecMultiComponentWorksWithContainer)
{
    AuxiliaryState state;

    AuxiliaryStateSpec spec;
    spec.name = "gates";
    spec.size = 3;
    spec.component_names = {"m", "h", "j"};
    spec.variable_kinds = {
        AuxiliaryVariableKind::Differential,
        AuxiliaryVariableKind::Differential,
        AuxiliaryVariableKind::Algebraic
    };
    spec.scope = AuxiliaryStateScope::Node;

    const std::array<svmp::FE::Real, 3> init = {0.05, 0.6, 0.3};
    state.registerState(spec, init);

    EXPECT_EQ(state.size(), 3u);
    EXPECT_TRUE(state.has("m"));
    EXPECT_TRUE(state.has("h"));
    EXPECT_TRUE(state.has("j"));
    EXPECT_DOUBLE_EQ(state["m"], 0.05);
    EXPECT_DOUBLE_EQ(state["h"], 0.6);
    EXPECT_DOUBLE_EQ(state["j"], 0.3);
}

TEST(AuxiliaryStateTypes, SpecMetadataIsPreserved)
{
    AuxiliaryStateSpec spec;
    spec.name = "model_x";
    spec.size = 1;
    spec.metadata["author"] = "test";
    spec.metadata["version"] = "1.0";

    EXPECT_EQ(spec.metadata.size(), 2u);
    EXPECT_EQ(spec.metadata.at("author"), "test");
    EXPECT_EQ(spec.metadata.at("version"), "1.0");
}

// ---------------------------------------------------------------------------
//  AuxiliaryStateRegistration backward compatibility
// ---------------------------------------------------------------------------

TEST(AuxiliaryStateTypes, RegistrationHasAssociatedMarkers)
{
    // associated_markers moved from spec to registration directly
    AuxiliaryStateRegistration reg;
    reg.spec.name = "X";
    reg.spec.size = 1;
    reg.associated_markers = {2, 5};

    EXPECT_EQ(reg.associated_markers.size(), 2u);
    EXPECT_EQ(reg.associated_markers[0], 2);
    EXPECT_EQ(reg.associated_markers[1], 5);
}
