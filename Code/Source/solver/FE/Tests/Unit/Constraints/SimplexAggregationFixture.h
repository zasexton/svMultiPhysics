/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 * All Rights Reserved. See Copyright-SimVascular.txt for additional details.
 */
#pragma once

#include "Assembly/CutIntegrationContext.h"
#include "Interfaces/LevelSetGeometrySnapshot.h"
#include "LevelSet/LevelSetInterfaceLifecycle.h"
#include "Systems/FESystem.h"
#include "Systems/SystemCollectives.h"

#include <gtest/gtest.h>
#include <array>
#include <memory>
#include <numeric>
#include <span>
#include <utility>
#include <vector>

namespace svmp::FE::constraints::test {

inline void publishSimplexBand(systems::FESystem& system, FieldId phi, std::span<const Real> nodal_values,
                        level_set::LevelSetGeneratedInterfaceLifecycle& lifecycle,
                        geometry::CutIntegrationSide side, int marker)
{
    constexpr Real isovalue = 0.375;
    const auto& mesh = system.meshAccess();
    const auto dimension = mesh.dimension();
    const Real sign = side == geometry::CutIntegrationSide::Negative ? 1 : -1;
    const auto nodal_dof = [&](GlobalIndex cell, std::size_t corner) {
        return system.fieldDofOffset(phi) + system.fieldDofHandler(phi).getCellDofs(cell)[corner];
    };
    std::vector<Real> solution(system.dofHandler().getNumDofs());
    std::vector<GlobalIndex> nodes;
    for (GlobalIndex cell = 0; cell < mesh.numCells(); ++cell) {
        mesh.getCellNodes(cell, nodes);
        for (std::size_t corner = 0; corner < nodes.size(); ++corner) {
            solution[nodal_dof(cell, corner)] = isovalue + sign * nodal_values[nodes[corner]];
        }
    }
    level_set::LevelSetGeneratedInterfaceOptions options;
    options.level_set_field_name = "phi";
    options.domain_id = "simplex-band";
    options.requested_interface_marker = marker;
    options.isovalue = isovalue;
    auto generated = lifecycle.build(system, options, solution);
    ASSERT_TRUE(generated.success) << generated.diagnostic;
    interfaces::LevelSetGeometrySnapshotPolicy policy;
    policy.require_complete_exterior_boundary_partition = false;
    const auto gradients = [&](GlobalIndex cell) {
        std::array<Real, 3> gradient{};
        const Real origin = solution[nodal_dof(cell, 0)];
        for (int axis = 0; axis < dimension; ++axis) {
            gradient[axis] = solution[nodal_dof(cell, axis + 1)] - origin;
        }
        return gradient;
    };
    interfaces::LevelSetGeometryScalarEvaluator scalar;
    scalar.reference_gradient = [&](GlobalIndex cell, const auto&, const auto&) {
        return gradients(cell);
    };
    scalar.value = [&](GlobalIndex cell, const auto& point, const auto&) {
        const auto gradient = gradients(cell);
        return std::inner_product(gradient.begin(), gradient.end(), point.begin(),
                                  solution[nodal_dof(cell, 0)]);
    };
    systems::CollectiveCommunicator communicator;
#if FE_HAS_MPI
    communicator = systems::CollectiveCommunicator(system.dofHandler().mpiComm());
#endif
    auto context = std::make_shared<assembly::CutIntegrationContext>();
    context->addLevelSetGeometrySnapshot(interfaces::buildLevelSetGeometrySnapshot(
        std::move(generated.domain), {}, {}, mesh, policy, scalar, options.domain_id,
        systems::ownershipCollective(communicator)));
    system.setCutIntegrationContext(std::move(context));
}

} // namespace svmp::FE::constraints::test
