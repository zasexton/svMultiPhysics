/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#include "Physics/Formulations/NavierStokes/NavierStokesGeneratedNitsche.h"

#include "FE/Dofs/EntityDofMap.h"

#if defined(SVMP_FE_WITH_MESH) && SVMP_FE_WITH_MESH
#  include "Mesh/Fields/MeshFields.h"
#  include "Mesh/Mesh.h"
#endif

#include <cstddef>
#include <optional>
#include <stdexcept>
#include <vector>

namespace svmp {
namespace Physics {
namespace formulations {
namespace navier_stokes {

void requireCertifiedNitscheTraces(
    const IncompressibleNavierStokesVMSOptions& options,
    const std::function<std::optional<int>(int)>& generated_active_boundary_for,
    bool small_cut_aggregation,
    const std::vector<FE::forms::bc::GeneratedBoundaryNitscheTraceFormBinding>&
        generated_nitsche_trace_bindings)
{
    std::size_t expected_generated_nitsche_trace_bindings = 0u;
    for (const auto& bc : options.velocity_dirichlet_weak) {
        const int physical_marker =
            FE::forms::bc::detail::boundaryMarkerOrThrow(
                bc,
                "IncompressibleNavierStokesVMSModule::registerOn "
                "generated-boundary Nitsche trace policy");
        const auto generated_marker =
            generated_active_boundary_for(physical_marker);
        if (!generated_marker.has_value()) {
            continue;
        }
        if (!small_cut_aggregation) {
            throw std::logic_error(
                "IncompressibleNavierStokesVMSModule::registerOn: "
                "generated-boundary Nitsche route reached installation "
                "without its preflighted small-cut aggregation policy");
        }
        ++expected_generated_nitsche_trace_bindings;
        if (options.viscosity_model != nullptr) {
            throw std::invalid_argument(
                "IncompressibleNavierStokesVMSModule::registerOn: "
                    "certified generated-boundary Nitsche routes require "
                    "constant Newtonian viscosity");
        }
    }
    if (generated_nitsche_trace_bindings.size() !=
        expected_generated_nitsche_trace_bindings) {
        throw std::logic_error(
            "IncompressibleNavierStokesVMSModule::registerOn: "
            "generated-boundary Nitsche form binding count differs "
            "from the certified production routes");
    }
}

} // namespace navier_stokes
} // namespace formulations
} // namespace Physics
} // namespace svmp
