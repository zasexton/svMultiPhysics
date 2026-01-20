/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#include "Assembly/Assembler.h"

#include "Assembly/AssemblerSelection.h"
#include "Assembly/CachedAssembler.h"
#include "Assembly/DeviceAssembler.h"
#include "Assembly/ParallelAssembler.h"
#include "Assembly/ScheduledAssembler.h"
#include "Assembly/StandardAssembler.h"
#include "Assembly/SymbolicAssembler.h"
#include "Assembly/VectorizedAssembler.h"
#include "Assembly/WorkStreamAssembler.h"

#include <algorithm>
#include <cctype>
#include <string>

namespace svmp {
namespace FE {
namespace assembly {

namespace {

constexpr RequiredData kContextRequiredBits =
    // Geometry
    RequiredData::PhysicalPoints | RequiredData::Jacobians | RequiredData::JacobianDets |
    RequiredData::InverseJacobians | RequiredData::Normals | RequiredData::Tangents |
    RequiredData::EntityMeasures |
    // Basis
    RequiredData::BasisValues | RequiredData::BasisGradients | RequiredData::PhysicalGradients |
    RequiredData::BasisHessians | RequiredData::BasisCurls | RequiredData::BasisDivergences |
    // Quadrature
    RequiredData::QuadraturePoints | RequiredData::QuadratureWeights | RequiredData::IntegrationWeights |
    // Solution (values/derivatives)
    RequiredData::SolutionValues | RequiredData::SolutionGradients | RequiredData::SolutionHessians |
    RequiredData::SolutionLaplacians |
    // DG/face data
    RequiredData::FaceOrientations | RequiredData::NeighborData;

std::string normalizeName(std::string_view name)
{
    auto is_space = [](unsigned char c) { return std::isspace(c) != 0; };

    std::string s(name);
    s.erase(s.begin(), std::find_if_not(s.begin(), s.end(), is_space));
    s.erase(std::find_if_not(s.rbegin(), s.rend(), is_space).base(), s.end());

    std::transform(s.begin(), s.end(), s.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return s;
}

} // namespace

std::unique_ptr<Assembler> createAssembler(const AssemblyOptions& options,
                                           std::string_view assembler_name,
                                           const FormCharacteristics& form,
                                           const SystemCharacteristics& system)
{
    return createAssembler(options, assembler_name, form, system, nullptr);
}

std::unique_ptr<Assembler> createAssembler(const AssemblyOptions& options,
                                           std::string_view assembler_name,
                                           const FormCharacteristics& form,
                                           const SystemCharacteristics& system,
                                           std::string* selection_report)
{
    std::string report;
    auto reportLine = [&](std::string_view line) {
        if (!selection_report) return;
        if (!report.empty()) report.push_back('\n');
        report.append(line);
    };

    std::string name = normalizeName(assembler_name);
    if (name.empty()) {
        name = "standardassembler";
    }

    std::unique_ptr<Assembler> base;

    if (name == "auto" || name == "autoassembler") {
        // Conservative policy: only select a base that is known-safe given current requirements.
        // (More aggressive selection remains explicitly opt-in via AssemblyOptions::auto_policy.)
        if (options.auto_policy != AutoSelectionPolicy::Conservative) {
            reportLine("Auto policy: non-Conservative policy requested; falling back to Conservative selection");
        } else {
            reportLine("Auto policy: Conservative");
        }

        reportLine("Auto selection inputs:");
        if (selection_report) {
            reportLine("  - fields=" + std::to_string(system.num_fields) +
                       ", cells=" + std::to_string(system.num_cells) +
                       ", dim=" + std::to_string(system.dimension));
            reportLine("  - dofs=" + std::to_string(system.num_dofs_total) +
                       ", max_dofs_per_cell=" + std::to_string(system.max_dofs_per_cell) +
                       ", p_max=" + std::to_string(system.max_polynomial_order));
            reportLine("  - threads=" + std::to_string(system.num_threads) +
                       ", mpi_world=" + std::to_string(system.mpi_world_size));
            reportLine("  - domains: cell=" + std::string(form.has_cell_terms ? "yes" : "no") +
                       ", boundary=" + std::string(form.has_boundary_terms ? "yes" : "no") +
                       ", interior=" + std::string(form.has_interior_face_terms ? "yes" : "no") +
                       ", interface=" + std::string(form.has_interface_face_terms ? "yes" : "no") +
                       ", global=" + std::string(form.has_global_terms ? "yes" : "no"));
            reportLine("  - needs: DG=" + std::string(form.needsDG() ? "yes" : "no") +
                       ", solution=" + std::string(form.needsSolution() ? "yes" : "no") +
                       ", transient=" + std::string(form.isTransient() ? "yes" : "no") +
                       ", material_state=" + std::string(form.needsMaterialState() ? "yes" : "no") +
                       ", field_requirements=" + std::string(form.needsFieldSolutions() ? "yes" : "no"));
        }

        // Candidate: ParallelAssembler (only when MPI is active and the workload fits current limitations).
        const bool mpi_active = system.mpi_world_size > 1;
        const bool single_field = system.num_fields == 1u;
        const bool needs_full_context = (form.required_data & kContextRequiredBits) != RequiredData::None;

        const bool parallel_applicable =
            mpi_active && single_field &&
            !form.needsDG() &&
            !form.needsSolution() &&
            !form.isTransient() &&
            !form.needsMaterialState() &&
            !form.needsFieldSolutions() &&
            !needs_full_context;

        if (parallel_applicable) {
            reportLine("Auto selected base: ParallelAssembler (MPI active, single-field, minimal kernel requirements)");
            base = createParallelAssembler(options);
        } else {
            if (mpi_active) {
                reportLine("Auto skipped ParallelAssembler: current requirements exceed supported feature set; using StandardAssembler");
            } else {
                reportLine("Auto selected base: StandardAssembler");
            }
            base = createStandardAssembler(options);
        }
    } else if (name == "standard" || name == "standardassembler") {
        reportLine("Explicit selection: StandardAssembler");
        base = createStandardAssembler(options);
    } else if (name == "parallel" || name == "parallelassembler") {
        reportLine("Explicit selection: ParallelAssembler");
        base = createParallelAssembler(options);
    } else if (name == "workstream" || name == "workstreamassembler") {
        reportLine("Explicit selection: WorkStreamAssembler");
        base = createWorkStreamAssembler();
        base->setOptions(options);
    } else if (name == "device" || name == "deviceassembler") {
        reportLine("Explicit selection: DeviceAssembler");
        base = createDeviceAssemblerAuto();
        base->setOptions(options);
    } else if (name == "symbolic" || name == "symbolicassembler") {
        reportLine("Explicit selection: SymbolicAssembler");
        base = createSymbolicAssembler();
        base->setOptions(options);
    } else {
        FE_THROW(FEException, "createAssembler: unknown assembler_name '" + std::string(assembler_name) + "'");
    }

    FE_CHECK_NOT_NULL(base.get(), "createAssembler: base assembler");

    // Decorators are composable and opt-in via AssemblyOptions.
    std::unique_ptr<Assembler> assembled = std::move(base);

    // Ordering/scheduling first (affects traversal order).
    if (options.schedule_elements) {
        const auto strategy = static_cast<ScheduledAssembler::Strategy>(options.schedule_strategy);
        assembled = std::make_unique<ScheduledAssembler>(std::move(assembled), strategy);
        reportLine("Decorator enabled: ScheduledAssembler");
    }

    // Caching next (affects kernel evaluation cost).
    if (options.cache_element_data) {
        assembled = std::make_unique<CachedAssembler>(std::move(assembled));
        reportLine("Decorator enabled: CachedAssembler");
    }

    // Vectorization last (currently a structural decorator).
    if (options.use_batching) {
        assembled = std::make_unique<VectorizedAssembler>(std::move(assembled), options.batch_size);
        reportLine("Decorator enabled: VectorizedAssembler");
    }

    // ------------------------------------------------------------------
    // Hard validation: requirements must be met (no silent fallback).
    // ------------------------------------------------------------------
    const std::string selected_name = assembled->name();

    const bool needs_full_context = (form.required_data & kContextRequiredBits) != RequiredData::None;
    FE_THROW_IF(needs_full_context && !assembled->supportsFullContext(), FEException,
                "createAssembler: selected assembler '" + selected_name +
                    "' does not support required FE context data (quadrature/geometry/basis), but the system requires it");

    FE_THROW_IF(form.needsDG() && !assembled->supportsDG(), FEException,
                "createAssembler: selected assembler '" + selected_name +
                    "' does not support DG (interior-face assembly), but the system requires it");

    FE_THROW_IF(form.needsSolution() && !assembled->supportsSolution(), FEException,
                "createAssembler: selected assembler '" + selected_name +
                    "' does not support solution-dependent kernels, but the system requires solution data");

    FE_THROW_IF(form.isTransient() && !assembled->supportsTimeIntegrationContext(), FEException,
                "createAssembler: selected assembler '" + selected_name +
                    "' does not support time-integration context, but the system is transient");

    FE_THROW_IF(form.isTransient() && !assembled->supportsSolutionHistory(), FEException,
                "createAssembler: selected assembler '" + selected_name +
                    "' does not support solution history, but the system is transient");

    FE_THROW_IF(form.needsMaterialState() && !assembled->supportsMaterialState(), FEException,
                "createAssembler: selected assembler '" + selected_name +
                    "' does not support material state, but the system requires it");

    FE_THROW_IF(form.needsFieldSolutions() && !assembled->supportsFieldRequirements(), FEException,
                "createAssembler: selected assembler '" + selected_name +
                    "' does not support additional field requirements, but the system requires them");

    FE_THROW_IF(system.num_fields > 1u && !assembled->supportsDofOffsets(), FEException,
                "createAssembler: selected assembler '" + selected_name +
                    "' does not support multi-field DOF offsets, but the system has " +
                    std::to_string(system.num_fields) + " fields");

    reportLine("Selected assembler: " + selected_name);

    if (selection_report) {
        *selection_report = std::move(report);
    }

    return assembled;
}

} // namespace assembly
} // namespace FE
} // namespace svmp
