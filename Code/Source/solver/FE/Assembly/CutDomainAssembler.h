/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#ifndef SVMP_FE_ASSEMBLY_CUTDOMAINASSEMBLER_H
#define SVMP_FE_ASSEMBLY_CUTDOMAINASSEMBLER_H

/**
 * @file CutDomainAssembler.h
 * @brief Physics-neutral dispatch of FE kernels over cut-integration rules.
 *
 * This helper owns only the cut-domain assembly loop: path filtering, kernel
 * dispatch, and local output accumulation. The caller remains responsible for
 * preparing the AssemblyContext for the chosen FE spaces, element mapping,
 * current solution state, and optional parameter/JIT constants.
 */

#include "Assembly/AssemblyContext.h"
#include "Assembly/AssemblyKernel.h"
#include "Assembly/CutIntegrationContext.h"
#include "Geometry/CutQuadrature.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <stdexcept>

namespace svmp {
namespace FE {
namespace assembly {

enum class CutDomainKind : std::uint8_t {
    Volume,
    EmbeddedInterface
};

struct CutRuleAssemblyRequest {
    CutIntegrationAssemblyPath path{CutIntegrationAssemblyPath::Standard};
    CutDomainKind domain{CutDomainKind::Volume};
    const geometry::CutQuadratureRule* rule{nullptr};
    const CutCellAssemblyMetadata* metadata{nullptr};
    const CutIntegrationBinding* binding{nullptr};
    std::size_t rule_index{0u};
    int marker{-1};
};

struct CutDomainAssemblyOptions {
    CutIntegrationAssemblyPath path{CutIntegrationAssemblyPath::Standard};
    bool include_volume_rules{true};
    bool include_interface_rules{true};
    int volume_marker{-1};
    geometry::CutIntegrationSide volume_side{geometry::CutIntegrationSide::Interface};
    int interface_marker{-1};
};

struct CutDomainAssemblySummary {
    CutIntegrationAssemblyPath path{CutIntegrationAssemblyPath::Standard};
    std::size_t volume_rule_count{0u};
    std::size_t interface_rule_count{0u};
    std::size_t skipped_rule_count{0u};
    KernelOutput volume_output{};
    KernelOutput interface_output{};
    KernelOutput total_output{};

    [[nodiscard]] bool hasMatrix() const noexcept { return total_output.has_matrix; }
    [[nodiscard]] bool hasVector() const noexcept { return total_output.has_vector; }
};

using CutRuleContextBuilder =
    std::function<void(const CutRuleAssemblyRequest& request, AssemblyContext& context)>;

// The builder may bind non-owning AssemblyContext spans, such as JIT constants,
// auxiliary fields, or material state views. Those backing buffers must outlive
// the immediately following kernel dispatch for the requested cut rule.

[[nodiscard]] inline bool cutBindingVisibleToPath(const CutIntegrationBinding& binding,
                                                  CutIntegrationAssemblyPath path) noexcept
{
    return binding.visible_to_paths.empty() ||
           std::find(binding.visible_to_paths.begin(), binding.visible_to_paths.end(), path) !=
               binding.visible_to_paths.end();
}

[[nodiscard]] inline bool cutVolumeRuleMatchesOptions(
    const geometry::CutQuadratureRule& rule,
    const CutDomainAssemblyOptions& options) noexcept
{
    if (options.volume_marker >= 0 && rule.provenance.marker != options.volume_marker) {
        return false;
    }
    if (options.volume_side != geometry::CutIntegrationSide::Interface &&
        rule.side != options.volume_side) {
        return false;
    }
    return true;
}

inline void accumulateCutKernelOutput(KernelOutput& dst, const KernelOutput& src)
{
    if (!src.has_matrix && !src.has_vector) {
        return;
    }

    if (!dst.has_matrix && !dst.has_vector && dst.n_test_dofs == 0 && dst.n_trial_dofs == 0) {
        dst.n_test_dofs = src.n_test_dofs;
        dst.n_trial_dofs = src.n_trial_dofs;
    } else if (dst.n_test_dofs != src.n_test_dofs ||
               dst.n_trial_dofs != src.n_trial_dofs) {
        throw std::invalid_argument("CutDomainAssembler: inconsistent kernel output dimensions");
    }

    if (src.has_matrix) {
        const auto matrix_size =
            static_cast<std::size_t>(src.n_test_dofs) * static_cast<std::size_t>(src.n_trial_dofs);
        if (!dst.has_matrix) {
            dst.has_matrix = true;
            dst.local_matrix.assign(matrix_size, Real(0.0));
        }
        if (dst.local_matrix.size() != matrix_size || src.local_matrix.size() != matrix_size) {
            throw std::invalid_argument("CutDomainAssembler: inconsistent local matrix storage");
        }
        for (std::size_t i = 0u; i < matrix_size; ++i) {
            dst.local_matrix[i] += src.local_matrix[i];
        }
    }

    if (src.has_vector) {
        const auto vector_size = static_cast<std::size_t>(src.n_test_dofs);
        if (!dst.has_vector) {
            dst.has_vector = true;
            dst.local_vector.assign(vector_size, Real(0.0));
        }
        if (dst.local_vector.size() != vector_size || src.local_vector.size() != vector_size) {
            throw std::invalid_argument("CutDomainAssembler: inconsistent local vector storage");
        }
        for (std::size_t i = 0u; i < vector_size; ++i) {
            dst.local_vector[i] += src.local_vector[i];
        }
    }
}

[[nodiscard]] inline CutDomainAssemblySummary assembleCutDomains(
    const CutIntegrationContext& context,
    AssemblyKernel& kernel,
    const CutRuleContextBuilder& context_builder,
    const CutDomainAssemblyOptions& options = {})
{
    if (!context_builder) {
        throw std::invalid_argument("CutDomainAssembler: context builder is required");
    }

    CutDomainAssemblySummary summary;
    summary.path = options.path;

    const bool has_explicit_bindings = !context.bindings().empty();

    if (options.include_volume_rules) {
        if (options.volume_marker >= 0 &&
            options.volume_side != geometry::CutIntegrationSide::Interface) {
            context.assertGeneratedVolumeRulesCurrentForMarkerAndSide(
                options.volume_marker,
                options.volume_side);
        }
        const auto& volume_rules = context.volumeRules();
        const auto& metadata = context.metadata();
        const auto& bindings = context.bindings();
        for (std::size_t i = 0u; i < volume_rules.size(); ++i) {
            if (!cutVolumeRuleMatchesOptions(volume_rules[i], options)) {
                ++summary.skipped_rule_count;
                continue;
            }
            const auto* binding = has_explicit_bindings && i < bindings.size() ? &bindings[i] : nullptr;
            if (binding != nullptr && !cutBindingVisibleToPath(*binding, options.path)) {
                ++summary.skipped_rule_count;
                continue;
            }
            if (!kernel.hasCell()) {
                ++summary.skipped_rule_count;
                continue;
            }

            CutRuleAssemblyRequest request;
            request.path = options.path;
            request.domain = CutDomainKind::Volume;
            request.rule = &volume_rules[i];
            request.metadata = i < metadata.size() ? &metadata[i] : nullptr;
            request.binding = binding;
            request.rule_index = i;
            request.marker = options.volume_marker >= 0 ? options.volume_marker
                                                        : volume_rules[i].provenance.marker;

            AssemblyContext rule_context;
            context_builder(request, rule_context);
            rule_context.setCutVolumeDomain(request.marker, volume_rules[i].side);

            KernelOutput output;
            kernel.computeCell(rule_context, output);
            accumulateCutKernelOutput(summary.volume_output, output);
            accumulateCutKernelOutput(summary.total_output, output);
            ++summary.volume_rule_count;
        }
    }

    if (options.include_interface_rules) {
        const auto& interface_rules = context.interfaceRules();
        for (std::size_t i = 0u; i < interface_rules.size(); ++i) {
            if (!kernel.hasBoundaryFace() && !kernel.hasSingleSidedInterfaceFace()) {
                ++summary.skipped_rule_count;
                continue;
            }

            CutRuleAssemblyRequest request;
            request.path = options.path;
            request.domain = CutDomainKind::EmbeddedInterface;
            request.rule = &interface_rules[i];
            request.rule_index = i;
            request.marker = options.interface_marker;

            AssemblyContext rule_context;
            context_builder(request, rule_context);

            KernelOutput output;
            kernel.computeBoundaryFace(rule_context, options.interface_marker, output);
            accumulateCutKernelOutput(summary.interface_output, output);
            accumulateCutKernelOutput(summary.total_output, output);
            ++summary.interface_rule_count;
        }
    }

    return summary;
}

} // namespace assembly
} // namespace FE
} // namespace svmp

#endif // SVMP_FE_ASSEMBLY_CUTDOMAINASSEMBLER_H
