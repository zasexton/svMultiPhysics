/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#include <gtest/gtest.h>

#include "Assembly/GeometricNonlinearityHooks.h"
#include "Assembly/AssemblyKernel.h"
#include "Spaces/H1Space.h"
#include "Spaces/ProductSpace.h"
#include "Systems/FESystem.h"
#include "Systems/GeometricNonlinearity.h"

#include "Mesh/Fields/MeshFields.h"
#include "Mesh/Mesh.h"
#include "Mesh/Motion/MotionFields.h"
#include "Mesh/Topology/CellShape.h"

#include <memory>
#include <vector>

namespace svmp {
namespace FE {
namespace systems {
namespace test {

namespace {

std::shared_ptr<svmp::Mesh> makeQuadMesh()
{
    auto base = std::make_shared<svmp::MeshBase>(2);
    const std::vector<svmp::real_t> x_ref = {
        0.0, 0.0,
        1.0, 0.0,
        1.0, 1.0,
        0.0, 1.0,
    };
    const std::vector<svmp::offset_t> offsets = {0, 4};
    const std::vector<svmp::index_t> conn = {0, 1, 2, 3};
    const std::vector<svmp::CellShape> shapes = {{svmp::CellFamily::Quad, 4, 1}};
    base->build_from_arrays(2, x_ref, offsets, conn, shapes);
    base->finalize();
    return svmp::create_mesh(std::move(base));
}

} // namespace

TEST(GeometricNonlinearity, TrialUpdatePolicySeparatesGeometryStressAndHistory)
{
    GeometricNonlinearityPolicy policy{};
    policy.enabled = true;
    policy.update_current_coordinates_on_trial = true;
    policy.update_strain_and_stress_on_trial = true;
    policy.history_updates_are_trial_until_commit = true;

    EXPECT_TRUE(shouldUpdateFieldAtPoint(policy,
                                         GeometricNonlinearityStateField::CurrentCoordinates,
                                         GeometricNonlinearityUpdatePoint::TrialIterate));
    EXPECT_TRUE(shouldUpdateFieldAtPoint(policy,
                                         GeometricNonlinearityStateField::Strain,
                                         GeometricNonlinearityUpdatePoint::TrialIterate));
    EXPECT_TRUE(shouldUpdateFieldAtPoint(policy,
                                         GeometricNonlinearityStateField::Stress,
                                         GeometricNonlinearityUpdatePoint::TrialIterate));
    EXPECT_FALSE(shouldUpdateFieldAtPoint(policy,
                                          GeometricNonlinearityStateField::HistoryState,
                                          GeometricNonlinearityUpdatePoint::TrialIterate));
    EXPECT_TRUE(shouldUpdateFieldAtPoint(policy,
                                         GeometricNonlinearityStateField::HistoryState,
                                         GeometricNonlinearityUpdatePoint::AcceptedNonlinearState));
}

TEST(GeometricNonlinearity, LineSearchRejectionRollsBackGeometryAndCoupledState)
{
    GeometricNonlinearityPolicy policy{};
    policy.enabled = true;
    policy.rollback_geometry_on_line_search_reject = true;
    policy.rollback_coupled_state_on_line_search_reject = true;

    GeometricNonlinearityTransactionEvent event{};
    event.geometry_state = GeometryTransactionState::Trial;
    event.line_search_rejected = true;

    EXPECT_TRUE(requiresLineSearchRollback(policy, event));
    EXPECT_TRUE(requiresCoupledStateRollback(policy, event));

    event.geometry_state = GeometryTransactionState::Accepted;
    EXPECT_FALSE(requiresLineSearchRollback(policy, event));
    EXPECT_FALSE(requiresCoupledStateRollback(policy, event));
}

TEST(GeometricNonlinearity, ArcLengthContinuationHooksAreOptIn)
{
    GeometricNonlinearityPolicy policy{};
    policy.enabled = true;
    EXPECT_FALSE(policy.arc_length.enabled);

    policy.arc_length.enabled = true;
    policy.arc_length.initial_radius = Real(0.1);
    policy.arc_length.min_radius = Real(1e-4);
    policy.arc_length.max_radius = Real(10);

    EXPECT_TRUE(policy.arc_length.enabled);
    EXPECT_LT(policy.arc_length.min_radius, policy.arc_length.initial_radius);
    EXPECT_GT(policy.arc_length.max_radius, policy.arc_length.initial_radius);
    EXPECT_STREQ(geometricNonlinearityUpdatePointName(
                     GeometricNonlinearityUpdatePoint::AcceptedTimeStep),
                 "AcceptedTimeStep");
    EXPECT_STREQ(geometricNonlinearityStateFieldName(
                     GeometricNonlinearityStateField::DeformationGradient),
                 "DeformationGradient");
}

TEST(GeometricNonlinearity, AssemblyContributionPolicyRequiresConsistentTangents)
{
    assembly::GeometricNonlinearContributionPolicy policy{};
    policy.kind = assembly::GeometricNonlinearContributionKind::FollowerLoadSensitivity;
    policy.contributes_to_tangent = true;
    policy.tangent_is_consistent = true;
    EXPECT_TRUE(assembly::geometricContributionPolicyIsValid(policy));

    policy.tangent_is_consistent = false;
    EXPECT_FALSE(assembly::geometricContributionPolicyIsValid(policy));

    policy.kind = assembly::GeometricNonlinearContributionKind::MaterialStiffness;
    EXPECT_TRUE(assembly::geometricContributionPolicyIsValid(policy));
    EXPECT_STREQ(assembly::geometricNonlinearContributionKindName(
                     assembly::GeometricNonlinearContributionKind::InitialStressStiffness),
                 "InitialStressStiffness");
}

TEST(GeometricNonlinearity, FESystemTrialAcceptRollbackLifecycleUpdatesMeshCoordinates)
{
    auto mesh = makeQuadMesh();
    const auto handles = svmp::motion::attach_motion_fields(*mesh, 2);
    auto* disp = svmp::MeshFields::field_data_as<svmp::real_t>(
        mesh->local_mesh(), handles.displacement);
    ASSERT_NE(disp, nullptr);
    const auto ncomp = svmp::MeshFields::field_components(
        mesh->local_mesh(), handles.displacement);
    ASSERT_EQ(ncomp, 2u);
    for (std::size_t v = 0; v < mesh->n_vertices(); ++v) {
        disp[v * ncomp + 0] = 0.10 + 0.01 * static_cast<Real>(v);
        disp[v * ncomp + 1] = -0.05;
    }

    auto scalar_space = std::make_shared<spaces::H1Space>(ElementType::Quad4, 1);
    auto vector_space = std::make_shared<spaces::ProductSpace>(scalar_space, 2);

    FESystem sys(mesh);
    const auto displacement =
        sys.addMeshDisplacementUnknown("mesh_displacement", vector_space);
    sys.addOperator("mass");
    sys.addCellKernel("mass", displacement, std::make_shared<assembly::MassKernel>(1.0));
    sys.setup();

    GeometricNonlinearityPolicy policy{};
    policy.enabled = true;
    sys.setGeometricNonlinearityPolicy(policy);
    EXPECT_TRUE(sys.geometricNonlinearityEnabled());

    std::vector<Real> state(static_cast<std::size_t>(sys.dofHandler().getNumDofs()), 0.0);
    ASSERT_EQ(sys.syncBoundMeshMotionFieldsToState(state), mesh->n_vertices() * 2u);
    SystemStateView view;
    view.u = state;

    const auto trial = sys.beginGeometricNonlinearityTrial(view);
    EXPECT_EQ(trial.update_point, GeometricNonlinearityUpdatePoint::TrialIterate);
    EXPECT_EQ(trial.geometry_state, GeometryTransactionState::Trial);
    ASSERT_TRUE(mesh->has_current_coords());
    EXPECT_TRUE(sys.meshCoordinateTransactionActive());

    const auto& X_ref = mesh->local_mesh().X_ref();
    const auto& X_cur = mesh->local_mesh().X_cur();
    ASSERT_EQ(X_cur.size(), X_ref.size());
    for (std::size_t v = 0; v < mesh->n_vertices(); ++v) {
        EXPECT_NEAR(X_cur[v * 2u + 0], X_ref[v * 2u + 0] + disp[v * ncomp + 0], 1.0e-12);
        EXPECT_NEAR(X_cur[v * 2u + 1], X_ref[v * 2u + 1] + disp[v * ncomp + 1], 1.0e-12);
    }

    const auto rolled_back = sys.rollbackGeometricNonlinearityTrial();
    EXPECT_EQ(rolled_back.geometry_state, GeometryTransactionState::RolledBack);
    EXPECT_FALSE(sys.meshCoordinateTransactionActive());
    EXPECT_FALSE(mesh->has_current_coords());

    (void)sys.beginGeometricNonlinearityTrial(view);
    const auto accepted = sys.acceptGeometricNonlinearityState(
        view, GeometricNonlinearityUpdatePoint::AcceptedNonlinearState);
    EXPECT_EQ(accepted.geometry_state, GeometryTransactionState::Accepted);
    EXPECT_FALSE(sys.meshCoordinateTransactionActive());
    EXPECT_TRUE(mesh->has_current_coords());

    const auto committed = sys.acceptGeometricNonlinearityState(
        view, GeometricNonlinearityUpdatePoint::AcceptedTimeStep);
    EXPECT_EQ(committed.update_point, GeometricNonlinearityUpdatePoint::AcceptedTimeStep);
    EXPECT_EQ(committed.geometry_state, GeometryTransactionState::Committed);
    EXPECT_FALSE(sys.meshCoordinateTransactionActive());
}

TEST(GeometricNonlinearity, UpdatedLagrangianRebaseCommitsReferenceAndClearsDisplacementField)
{
    auto mesh = makeQuadMesh();
    const auto handles = svmp::motion::attach_motion_fields(*mesh, 2);
    auto* disp = svmp::MeshFields::field_data_as<svmp::real_t>(
        mesh->local_mesh(), handles.displacement);
    ASSERT_NE(disp, nullptr);
    const auto ncomp = svmp::MeshFields::field_components(
        mesh->local_mesh(), handles.displacement);
    for (std::size_t v = 0; v < mesh->n_vertices(); ++v) {
        disp[v * ncomp + 0] = 0.20;
        disp[v * ncomp + 1] = 0.05 * static_cast<Real>(v);
    }

    auto scalar_space = std::make_shared<spaces::H1Space>(ElementType::Quad4, 1);
    auto vector_space = std::make_shared<spaces::ProductSpace>(scalar_space, 2);

    FESystem sys(mesh);
    const auto displacement =
        sys.addMeshDisplacementUnknown("mesh_displacement", vector_space);
    sys.addOperator("mass");
    sys.addCellKernel("mass", displacement, std::make_shared<assembly::MassKernel>(1.0));
    sys.setup();

    GeometricNonlinearityPolicy policy{};
    policy.enabled = true;
    policy.reference_policy = geometry::FiniteDeformationReferencePolicy::UpdatedLagrangian;
    sys.setGeometricNonlinearityPolicy(policy);

    std::vector<Real> state(static_cast<std::size_t>(sys.dofHandler().getNumDofs()), 0.0);
    ASSERT_EQ(sys.syncBoundMeshMotionFieldsToState(state), mesh->n_vertices() * 2u);
    SystemStateView view;
    view.u = state;
    (void)sys.acceptGeometricNonlinearityState(
        view, GeometricNonlinearityUpdatePoint::AcceptedNonlinearState);
    ASSERT_TRUE(mesh->has_current_coords());
    const auto committed_current = mesh->local_mesh().X_cur();

    EXPECT_TRUE(sys.rebaseGeometricNonlinearityReference());
    const auto& rebased_reference = mesh->local_mesh().X_ref();
    ASSERT_EQ(rebased_reference.size(), committed_current.size());
    for (std::size_t i = 0; i < rebased_reference.size(); ++i) {
        EXPECT_NEAR(rebased_reference[i], committed_current[i], 1.0e-12);
    }

    for (std::size_t i = 0; i < mesh->n_vertices() * ncomp; ++i) {
        EXPECT_NEAR(disp[i], 0.0, 1.0e-12);
    }

    std::vector<Real> reset_state(static_cast<std::size_t>(sys.dofHandler().getNumDofs()), 1.0);
    ASSERT_EQ(sys.syncBoundMeshMotionFieldsToState(reset_state), mesh->n_vertices() * 2u);
    for (const auto value : reset_state) {
        EXPECT_NEAR(value, 0.0, 1.0e-12);
    }
}

} // namespace test
} // namespace systems
} // namespace FE
} // namespace svmp
