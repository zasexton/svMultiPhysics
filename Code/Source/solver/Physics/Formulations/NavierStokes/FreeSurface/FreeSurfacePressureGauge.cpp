/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#include "Physics/Formulations/NavierStokes/FreeSurface/FreeSurfacePressureGauge.h"

#include "Physics/Formulations/NavierStokes/FreeSurface/FreeSurfaceContactCertificate.h"
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
#include "FE/Systems/SystemCollectives.h"
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

namespace {

struct EmbeddedFreeSurfaceMeasureEvidence {
    FE::Real stored_measure{0.0};
    FE::Real quadrature_weight_measure{0.0};
    std::uint64_t rule_count{0u};
    std::uint64_t quadrature_point_count{0u};
    bool invalid_measure{false};
};

[[nodiscard]] EmbeddedFreeSurfaceMeasureEvidence
embeddedFreeSurfaceMeasureEvidence(
    const FE::systems::FESystem& system,
    const FE::assembly::CutIntegrationContext& context,
    int interface_marker)
{
    EmbeddedFreeSurfaceMeasureEvidence local;
    std::exception_ptr local_preflight_exception;
    try {
        context.assertAllFreeSurfaceGeometrySnapshotsCurrent(
            system.meshAccess());
        for (const auto* rule :
             context.interfaceRulesForMarker(
                 interface_marker)) {
            if (rule == nullptr) {
                local.invalid_measure = true;
                continue;
            }
            ++local.rule_count;
            if (!std::isfinite(rule->measure)) {
                local.invalid_measure = true;
            } else {
                local.stored_measure += rule->measure;
            }
            for (const auto& point : rule->points) {
                ++local.quadrature_point_count;
                if (!std::isfinite(point.weight)) {
                    local.invalid_measure = true;
                } else {
                    local.quadrature_weight_measure +=
                        point.weight;
                }
            }
        }
    } catch (...) {
        local_preflight_exception =
            std::current_exception();
    }

#if FE_HAS_MPI
    int mpi_initialized = 0;
    int mpi_finalized = 0;
    MPI_Initialized(&mpi_initialized);
    if (mpi_initialized != 0) {
        MPI_Finalized(&mpi_finalized);
    }
    MPI_Comm communicator = MPI_COMM_NULL;
    if (mpi_initialized != 0 &&
        mpi_finalized == 0) {
        communicator =
            system.activeMpiCommunicator();
    }
    if (communicator != MPI_COMM_NULL) {
        int communicator_size = 1;
        MPI_Comm_size(communicator, &communicator_size);
        if (communicator_size > 1) {
            FE::systems::logRankLocalPhaseFailure(
                "IncompressibleNavierStokesVMSModule",
                "embedded_free_surface_measure_preflight",
                local_preflight_exception);
            const int local_ok =
                local_preflight_exception == nullptr
                    ? 1
                    : 0;
            int all_ok = 0;
            MPI_Allreduce(
                &local_ok,
                &all_ok,
                1,
                MPI_INT,
                MPI_MIN,
                communicator);
            if (all_ok == 0) {
                if (local_preflight_exception != nullptr) {
                    std::rethrow_exception(
                        local_preflight_exception);
                }
                throw std::runtime_error(
                    "IncompressibleNavierStokesVMSModule: "
                    "another communicator rank rejected "
                    "embedded free-surface measure preflight");
            }
            const std::array<double, 2> local_measures{
                static_cast<double>(local.stored_measure),
                static_cast<double>(local.quadrature_weight_measure)};
            std::array<double, 2> global_measures{};
            MPI_Allreduce(local_measures.data(), global_measures.data(),
                          static_cast<int>(local_measures.size()), MPI_DOUBLE,
                          MPI_SUM, communicator);

            const std::array<unsigned long long, 2> local_counts{
                static_cast<unsigned long long>(local.rule_count),
                static_cast<unsigned long long>(local.quadrature_point_count)};
            std::array<unsigned long long, 2> global_counts{};
            MPI_Allreduce(local_counts.data(), global_counts.data(),
                          static_cast<int>(local_counts.size()),
                          MPI_UNSIGNED_LONG_LONG, MPI_SUM, communicator);

            const int local_invalid = local.invalid_measure ? 1 : 0;
            int global_invalid = 0;
            MPI_Allreduce(&local_invalid, &global_invalid, 1, MPI_INT,
                          MPI_MAX, communicator);

            local.stored_measure =
                static_cast<FE::Real>(global_measures[0]);
            local.quadrature_weight_measure =
                static_cast<FE::Real>(global_measures[1]);
            local.rule_count =
                static_cast<std::uint64_t>(global_counts[0]);
            local.quadrature_point_count =
                static_cast<std::uint64_t>(global_counts[1]);
            local.invalid_measure = global_invalid != 0;
        }
    }
#else
    (void)system;
#endif
    if (local_preflight_exception != nullptr) {
        std::rethrow_exception(
            local_preflight_exception);
    }
    return local;
}

void requirePositiveEmbeddedFreeSurfaceMeasure(
    const FE::systems::FESystem& system,
    const FE::assembly::CutIntegrationContext* context,
    int interface_marker,
    std::string_view level_set_field_name,
    std::string_view domain_id,
    std::string_view validation_stage)
{
    EmbeddedFreeSurfaceMeasureEvidence evidence;
    if (context != nullptr) {
        evidence = embeddedFreeSurfaceMeasureEvidence(
            system, *context, interface_marker);
    }
    if (context != nullptr && !evidence.invalid_measure &&
        evidence.rule_count > 0u && evidence.quadrature_point_count > 0u &&
        evidence.stored_measure > FE::Real{0.0} &&
        evidence.quadrature_weight_measure > FE::Real{0.0}) {
        return;
    }

    std::ostringstream message;
    message
        << "IncompressibleNavierStokesVMSModule: CutVolume embedded "
           "free-surface pressure anchor requires a positive generated "
           "interface measure before a pressure solve"
        << " validation_stage=" << validation_stage
        << " marker=" << interface_marker
        << " level_set_field='" << level_set_field_name << "'"
        << " domain_id='" << domain_id << "'"
        << " cut_context=" << (context != nullptr ? "present" : "missing")
        << " global_interface_rules=" << evidence.rule_count
        << " global_interface_quadrature_points="
        << evidence.quadrature_point_count
        << " global_stored_measure=" << evidence.stored_measure
        << " global_quadrature_weight_measure="
        << evidence.quadrature_weight_measure
        << " invalid_measure=" << (evidence.invalid_measure ? "true" : "false")
        << " gauge_policy=reject_zero_interface_no_dynamic_gauge_insertion";
    throw std::runtime_error(message.str());
}

class CutVolumePressureAnchorMeasureGuard final
    : public FE::systems::GlobalKernel {
public:
    CutVolumePressureAnchorMeasureGuard(int interface_marker,
                                        std::string level_set_field_name,
                                        std::string domain_id)
        : interface_marker_(interface_marker)
        , level_set_field_name_(std::move(level_set_field_name))
        , domain_id_(std::move(domain_id))
    {
    }

    [[nodiscard]] std::string name() const override
    {
        return "CutVolumePressureAnchorMeasureGuard:" +
               std::to_string(interface_marker_);
    }

    void addSparsityCouplings(
        const FE::systems::FESystem& system,
        FE::sparsity::SparsityPattern&) const override
    {
        // Application setup intentionally precedes its first generated-cut
        // refresh.  A missing context is therefore deferred until refresh or
        // assembly, while a preloaded context must already satisfy the anchor.
        if (system.cutIntegrationContext() != nullptr) {
            validate(system, system.cutIntegrationContext(), "setup");
        }
    }

    [[nodiscard]] FE::assembly::AssemblyResult assemble(
        const FE::systems::FESystem& system,
        const FE::systems::AssemblyRequest&,
        const FE::systems::SystemStateView&,
        FE::assembly::GlobalSystemView*,
        FE::assembly::GlobalSystemView*) override
    {
        validate(system, system.cutIntegrationContext(), "assembly");
        return {};
    }

    void validateContextUpdate(
        const FE::systems::FESystem& system,
        const FE::assembly::CutIntegrationContext* context) const
    {
        validate(system, context, "cut_context_update");
    }

private:
    void validate(const FE::systems::FESystem& system,
                  const FE::assembly::CutIntegrationContext* context,
                  std::string_view stage) const
    {
        requirePositiveEmbeddedFreeSurfaceMeasure(
            system, context, interface_marker_, level_set_field_name_,
            domain_id_, stage);
    }

    int interface_marker_{-1};
    std::string level_set_field_name_{};
    std::string domain_id_{};
};

} // namespace

void anchorPressureOnFreeSurface(
    FE::systems::FESystem& system,
    const std::string& operator_tag,
    FE::FieldId p_id,
    const FreeSurfaceBoundary& boundary)
{
    auto& gauge_registry = system.gaugeRegistry();
    constexpr std::string_view free_surface_pressure_anchor =
        "Unfitted CutVolume embedded free-surface natural traction anchors absolute pressure";
    const bool anchor_already_registered = std::any_of(
        gauge_registry.anchoring().begin(),
        gauge_registry.anchoring().end(),
        [&](const FE::gauge::AnchoringEvidence& evidence) {
            return evidence.field == p_id &&
                   evidence.family ==
                       FE::gauge::NullspaceModeFamily::ScalarConstant &&
                   evidence.source == free_surface_pressure_anchor;
        });
    if (!anchor_already_registered) {
        gauge_registry.addAnchoring(FE::gauge::AnchoringEvidence{
            .field = p_id,
            .component = -1,
            .region = -1,
            .family = FE::gauge::NullspaceModeFamily::ScalarConstant,
            .verdict = FE::gauge::AnchoringVerdict::Anchored,
            .source = std::string(free_surface_pressure_anchor),
        });
    }

    const auto& pressure_anchor_boundary = boundary;
    auto pressure_anchor_measure_guard =
        std::make_shared<CutVolumePressureAnchorMeasureGuard>(
            pressure_anchor_boundary.interface_marker,
            pressure_anchor_boundary.level_set_field_name,
            pressure_anchor_boundary.generated_interface_domain_id);
    system.addGlobalKernel(operator_tag,
                           pressure_anchor_measure_guard);
    system.addCutIntegrationContextUpdateCallback(
        FE::systems::CutIntegrationContextUpdateCallback{
            .name = "navier_stokes_cut_volume_pressure_anchor_measure:" +
                    std::to_string(
                        pressure_anchor_boundary.interface_marker),
            .callback =
                [&system, pressure_anchor_measure_guard](
                    const FE::assembly::CutIntegrationContext* context) {
                    if (!FE::systems::agreeOnLocalPhase(
                            system,
                            std::exception_ptr{},
                            context != nullptr,
                            "cut_volume_pressure_anchor_context",
                            "IncompressibleNavierStokesVMSModule")) {
                        // Use the same null-candidate diagnostic on every
                        // rank without letting a rank-local non-null
                        // candidate enter the measure reductions.
                        pressure_anchor_measure_guard
                            ->validateContextUpdate(system, nullptr);
                        throw std::runtime_error(
                            "IncompressibleNavierStokesVMSModule: "
                            "CutVolume pressure-anchor null context was "
                            "unexpectedly accepted");
                    }
                    pressure_anchor_measure_guard->validateContextUpdate(
                        system, context);
                },
        });
}

} // namespace navier_stokes
} // namespace formulations
} // namespace Physics
} // namespace svmp
