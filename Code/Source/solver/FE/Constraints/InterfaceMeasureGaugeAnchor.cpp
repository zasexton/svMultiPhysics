/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#include "Constraints/InterfaceMeasureGaugeAnchor.h"

#include "Assembly/Assembler.h"
#include "Assembly/CutIntegrationContext.h"
#include "Constraints/GaugeRegistry.h"
#include "Systems/FESystem.h"
#include "Systems/GlobalKernel.h"
#include "Systems/SystemCollectives.h"

#if FE_HAS_MPI
#include <mpi.h>
#endif

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <exception>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>

namespace svmp {
namespace FE {
namespace constraints {

namespace {

struct InterfaceMeasureEvidence {
    Real stored_measure{0.0};
    Real quadrature_weight_measure{0.0};
    std::uint64_t rule_count{0u};
    std::uint64_t quadrature_point_count{0u};
    bool invalid_measure{false};
};

[[nodiscard]] InterfaceMeasureEvidence
interfaceMeasureEvidence(
    const systems::FESystem& system,
    const assembly::CutIntegrationContext& context,
    int interface_marker,
    const InterfaceMeasureGaugeAnchorTexts& texts)
{
    InterfaceMeasureEvidence local;
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
            systems::logRankLocalPhaseFailure(
                texts.owner,
                texts.preflight_phase,
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
                    texts.owner + ": " + texts.preflight_rejection);
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
                static_cast<Real>(global_measures[0]);
            local.quadrature_weight_measure =
                static_cast<Real>(global_measures[1]);
            local.rule_count =
                static_cast<std::uint64_t>(global_counts[0]);
            local.quadrature_point_count =
                static_cast<std::uint64_t>(global_counts[1]);
            local.invalid_measure = global_invalid != 0;
        }
    }
#else
    (void)system;
    (void)texts;
#endif
    if (local_preflight_exception != nullptr) {
        std::rethrow_exception(
            local_preflight_exception);
    }
    return local;
}

void requirePositiveInterfaceMeasure(
    const systems::FESystem& system,
    const assembly::CutIntegrationContext* context,
    int interface_marker,
    std::string_view level_set_field_name,
    std::string_view domain_id,
    std::string_view validation_stage,
    const InterfaceMeasureGaugeAnchorTexts& texts)
{
    InterfaceMeasureEvidence evidence;
    if (context != nullptr) {
        evidence = interfaceMeasureEvidence(
            system, *context, interface_marker, texts);
    }
    if (context != nullptr && !evidence.invalid_measure &&
        evidence.rule_count > 0u && evidence.quadrature_point_count > 0u &&
        evidence.stored_measure > Real{0.0} &&
        evidence.quadrature_weight_measure > Real{0.0}) {
        return;
    }

    std::ostringstream message;
    message
        << texts.owner << ": " << texts.requirement
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
        << " gauge_policy=" << texts.gauge_policy;
    throw std::runtime_error(message.str());
}

} // namespace

InterfaceMeasureGaugeKernel::InterfaceMeasureGaugeKernel(const InterfaceMeasureGaugeAnchor& anchor)
    : interface_marker_(anchor.interface_marker)
    , level_set_field_name_(anchor.level_set_field_name)
    , domain_id_(anchor.domain_id)
    , texts_(anchor.texts)
{
}

std::string InterfaceMeasureGaugeKernel::name() const
{
    return texts_.kernel_name + ":" +
           std::to_string(interface_marker_);
}

void InterfaceMeasureGaugeKernel::addSparsityCouplings(
    const systems::FESystem& system,
    sparsity::SparsityPattern&) const
{
    // Application setup intentionally precedes its first generated-cut
    // refresh.  A missing context is therefore deferred until refresh or
    // assembly, while a preloaded context must already satisfy the anchor.
    if (system.cutIntegrationContext() != nullptr) {
        validate(system, system.cutIntegrationContext(), "setup");
    }
}

assembly::AssemblyResult InterfaceMeasureGaugeKernel::assemble(
    const systems::FESystem& system,
    const systems::AssemblyRequest&,
    const systems::SystemStateView&,
    assembly::GlobalSystemView*,
    assembly::GlobalSystemView*)
{
    validate(system, system.cutIntegrationContext(), "assembly");
    return {};
}

void InterfaceMeasureGaugeKernel::validateContextUpdate(
    const systems::FESystem& system,
    const assembly::CutIntegrationContext* context) const
{
    validate(system, context, "cut_context_update");
}

void InterfaceMeasureGaugeKernel::validate(
    const systems::FESystem& system,
    const assembly::CutIntegrationContext* context,
    std::string_view stage) const
{
    requirePositiveInterfaceMeasure(
        system, context, interface_marker_, level_set_field_name_,
        domain_id_, stage, texts_);
}

void anchorGaugeOnInterfaceMeasure(systems::FESystem& system,
                                   const std::string& operator_tag,
                                   const InterfaceMeasureGaugeAnchor& anchor)
{
    auto& gauge_registry = system.gaugeRegistry();
    const auto& texts = anchor.texts;
    const bool anchor_already_registered = std::any_of(
        gauge_registry.anchoring().begin(),
        gauge_registry.anchoring().end(),
        [&](const gauge::AnchoringEvidence& evidence) {
            return evidence.field == anchor.field &&
                   evidence.family ==
                       gauge::NullspaceModeFamily::ScalarConstant &&
                   evidence.source == texts.anchor_source;
        });
    if (!anchor_already_registered) {
        gauge_registry.addAnchoring(gauge::AnchoringEvidence{
            .field = anchor.field,
            .component = -1,
            .region = -1,
            .family = gauge::NullspaceModeFamily::ScalarConstant,
            .verdict = gauge::AnchoringVerdict::Anchored,
            .source = texts.anchor_source,
        });
    }

    std::shared_ptr<InterfaceMeasureGaugeKernel> measure_guard =
        anchor.kernel ? anchor.kernel
                      : std::make_shared<InterfaceMeasureGaugeKernel>(anchor);
    system.addGlobalKernel(operator_tag, measure_guard);
    system.addCutIntegrationContextUpdateCallback(
        systems::CutIntegrationContextUpdateCallback{
            .name = texts.callback_name + ":" +
                    std::to_string(anchor.interface_marker),
            .callback =
                [&system, measure_guard, owner = texts.owner,
                 context_phase = texts.context_phase,
                 null_context_accepted = texts.null_context_accepted](
                    const assembly::CutIntegrationContext* context) {
                    if (!systems::agreeOnLocalPhase(
                            system,
                            std::exception_ptr{},
                            context != nullptr,
                            context_phase,
                            owner)) {
                        // Use the same null-candidate diagnostic on every
                        // rank without letting a rank-local non-null
                        // candidate enter the measure reductions.
                        measure_guard->validateContextUpdate(system, nullptr);
                        throw std::runtime_error(owner + ": " + null_context_accepted);
                    }
                    measure_guard->validateContextUpdate(system, context);
                },
        });
}

} // namespace constraints
} // namespace FE
} // namespace svmp
