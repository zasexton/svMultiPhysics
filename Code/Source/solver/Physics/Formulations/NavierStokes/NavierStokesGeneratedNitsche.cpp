/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#include "Physics/Formulations/NavierStokes/NavierStokesGeneratedNitsche.h"

#include "Physics/Formulations/NavierStokes/FreeSurface/FreeSurfaceContactCertificate.h"
#include "Physics/Formulations/NavierStokes/FreeSurface/FreeSurfaceEnergy.h"
#include "Physics/Formulations/NavierStokes/FreeSurface/FreeSurfaceGeometry.h"
#include "Physics/Formulations/NavierStokes/FreeSurface/FreeSurfacePressureGauge.h"
#include "Physics/Formulations/NavierStokes/FreeSurface/FreeSurfaceQueries.h"
#include "Physics/Formulations/NavierStokes/FreeSurface/FreeSurfaceSemiImplicitSurfaceTension.h"
#include "Physics/Formulations/NavierStokes/FreeSurface/FreeSurfaceValidation.h"
#include "Physics/Formulations/NavierStokes/NavierStokesBCFactories.h"
#include "Physics/Formulations/NavierStokes/NavierStokesFields.h"
#include "Physics/Formulations/NavierStokes/NavierStokesInitialConditions.h"
#include "Physics/Formulations/NavierStokes/NavierStokesPreflight.h"

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

void addNitscheTraceRequests(
    FE::systems::FormInstallOptions& install,
    std::vector<FE::forms::bc::GeneratedBoundaryNitscheTraceFormBinding>&
        generated_nitsche_trace_bindings,
    int volume_interface_marker,
    FE::Real minimum_symmetric_energy_ratio)
{
    install.generated_boundary_nitsche_trace_requests.reserve(
        generated_nitsche_trace_bindings.size());
    for (auto& binding :
         generated_nitsche_trace_bindings) {
        install.generated_boundary_nitsche_trace_requests.push_back(
            FE::systems::
                GeneratedBoundaryNitscheTraceInstallRequest{
                    .binding = std::move(binding),
                    .volume_interface_marker =
                        volume_interface_marker,
                    .minimum_symmetric_energy_ratio =
                        minimum_symmetric_energy_ratio,
                });
    }
}

} // namespace navier_stokes
} // namespace formulations
} // namespace Physics
} // namespace svmp
