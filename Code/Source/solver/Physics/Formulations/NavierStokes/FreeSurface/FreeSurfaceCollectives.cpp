/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#include "Physics/Formulations/NavierStokes/FreeSurface/FreeSurfaceCollectives.h"

#include "Physics/Formulations/NavierStokes/FreeSurface/FreeSurfaceGeometry.h"
#include "Physics/Formulations/NavierStokes/FreeSurface/FreeSurfaceQueries.h"
#include "Physics/Formulations/NavierStokes/FreeSurface/FreeSurfaceSemiImplicitSurfaceTension.h"
#include "Physics/Formulations/NavierStokes/NavierStokesBCFactories.h"

#include "FE/Assembly/Assembler.h"
#include "FE/Assembly/CutIntegrationContext.h"
#include "FE/Assembly/GlobalSystemView.h"
#include "FE/Constraints/ConstraintPlan.h"
#include "FE/Constraints/LevelSetActiveSideVertexDirichletConstraint.h"
#include "FE/Constraints/SmallCutAggregationConstraint.h"
#include "FE/Constraints/ContactPointVelocityConstraint.h"
#include "FE/Constraints/VertexDirichletConstraint.h"
#include "FE/Constitutive/MetadataTaggedModel.h"
#include "FE/Core/Logger.h"
#include "FE/Dofs/EntityDofMap.h"
#include "FE/Elements/ReferenceElement.h"
#include "FE/Forms/CutCellForms.h"
#include "FE/Forms/IntegrationDomain.h"
#include "FE/Forms/SymbolicDifferentiation.h"
#include "FE/Forms/Vocabulary.h"
#include "FE/Geometry/FrameGeometry.h"
#include "FE/Geometry/MappingFactory.h"
#include "FE/LevelSet/LevelSetCellEvaluator.h"
#include "FE/LevelSet/LevelSetGeometry.h"
#include "FE/Backends/Interfaces/GenericVector.h"
#include "FE/Systems/BoundaryConditionManager.h"
#include "FE/Systems/ALEBinding.h"
#include "FE/Systems/FESystem.h"
#include "FE/Systems/FormsInstaller.h"
#include "Interfaces/GeneratedActiveBoundaryDomain.h"
#include "Interfaces/GeneratedInterfaceBoundaryIntersectionDomain.h"
#include "Interfaces/LevelSetInterfaceDomain.h"

#if defined(SVMP_FE_WITH_MESH) && SVMP_FE_WITH_MESH
#  include "Mesh/Fields/MeshFields.h"
#  include "Mesh/Mesh.h"
#endif

#include <algorithm>
#include <cctype>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <array>
#include <cmath>
#include <exception>
#include <initializer_list>
#include <iomanip>
#include <limits>
#include <locale>
#include <memory>
#include <optional>
#include <set>
#include <span>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <tuple>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <variant>
#include <vector>

namespace svmp {
namespace Physics {
namespace formulations {
namespace navier_stokes {

// A rank-local failure is logged, with its rank, before the collective that
// propagates it.  Otherwise only the "another communicator rank rejected"
// text of the other ranks reaches the log when the job is aborted from one of
// them first.
[[maybe_unused]] void logRankLocalCollectiveFailure(
    std::string_view phase,
    const std::exception_ptr& failure)
{
    if (failure == nullptr) {
        return;
    }
    std::string reason = "unknown exception";
    try {
        std::rethrow_exception(failure);
    } catch (const std::exception& error) {
        reason = error.what();
    } catch (...) {
    }
    FE_LOG_ERROR("IncompressibleNavierStokesVMSModule: "
                 "diagnostic=rank_local_collective_failure phase='" +
                 std::string(phase) + "' reason='" + reason + "'");
}

[[nodiscard]] bool coordinateCutContextCallbackLocalPhase(
    const FE::systems::FESystem& system,
    const std::exception_ptr& local_exception,
    bool local_available,
    std::string_view phase)
{
    bool any_failed = local_exception != nullptr;
    bool all_available = local_available && !any_failed;
#if FE_HAS_MPI
    int mpi_initialized = 0;
    int mpi_finalized = 0;
    MPI_Initialized(&mpi_initialized);
    if (mpi_initialized != 0) {
        MPI_Finalized(&mpi_finalized);
    }
    if (mpi_initialized != 0 && mpi_finalized == 0) {
        const auto communicator = system.activeMpiCommunicator();
        if (communicator != MPI_COMM_NULL) {
            int communicator_size = 1;
            MPI_Comm_size(communicator, &communicator_size);
            if (communicator_size > 1) {
                logRankLocalCollectiveFailure(phase, local_exception);
                // 0 = failed, 1 = unavailable, 2 = available. MPI_MIN makes
                // every rank take the same route after this local phase.
                const int local_state =
                    any_failed ? 0 : (local_available ? 2 : 1);
                int global_state = 0;
                MPI_Allreduce(
                    &local_state,
                    &global_state,
                    1,
                    MPI_INT,
                    MPI_MIN,
                    communicator);
                any_failed = global_state == 0;
                all_available = global_state == 2;
            }
        }
    }
#else
    (void)system;
#endif
    if (any_failed) {
        if (local_exception != nullptr) {
            std::rethrow_exception(local_exception);
        }
        throw std::runtime_error(
            "IncompressibleNavierStokesVMSModule: another communicator "
            "rank rejected cut-context callback phase '" +
            std::string(phase) + "'");
    }
    return all_available;
}

} // namespace navier_stokes
} // namespace formulations
} // namespace Physics
} // namespace svmp
