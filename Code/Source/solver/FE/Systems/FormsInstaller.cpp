/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#include "Systems/FormsInstaller.h"

#include "Core/FEException.h"

#include "Forms/FormCompiler.h"
#include "Forms/FormKernels.h"

#include "Systems/FESystem.h"

#include <algorithm>

namespace svmp {
namespace FE {
namespace systems {

namespace {

struct DomainDispatch {
    bool has_cell{false};
    bool has_interior{false};
    std::vector<int> boundary_markers{};
};

DomainDispatch analyzeDispatch(const forms::FormIR& ir)
{
    DomainDispatch out;
    out.has_cell = ir.hasCellTerms();
    out.has_interior = ir.hasInteriorFaceTerms();

    if (!ir.hasBoundaryTerms()) {
        return out;
    }

    bool has_all_markers = false;
    std::vector<int> markers;
    for (const auto& term : ir.terms()) {
        if (term.domain != forms::IntegralDomain::Boundary) continue;
        if (term.boundary_marker < 0) {
            has_all_markers = true;
            break;
        }
        markers.push_back(term.boundary_marker);
    }

    if (has_all_markers) {
        out.boundary_markers = {-1};
        return out;
    }

    std::sort(markers.begin(), markers.end());
    markers.erase(std::unique(markers.begin(), markers.end()), markers.end());
    out.boundary_markers = std::move(markers);
    return out;
}

void registerKernel(
    FESystem& system,
    const OperatorTag& op,
    FieldId test_field,
    FieldId trial_field,
    const DomainDispatch& dispatch,
    const KernelPtr& kernel)
{
    if (dispatch.has_cell) {
        system.addCellKernel(op, test_field, trial_field, kernel);
    }

    for (int marker : dispatch.boundary_markers) {
        system.addBoundaryKernel(op, marker, test_field, trial_field, kernel);
    }

    if (dispatch.has_interior) {
        system.addInteriorFaceKernel(op, test_field, trial_field, kernel);
    }
}

} // namespace

KernelPtr installResidualForm(
    FESystem& system,
    const OperatorTag& op,
    FieldId test_field,
    FieldId trial_field,
    const forms::FormExpr& residual_form,
    const FormInstallOptions& options)
{
    forms::FormCompiler compiler(options.compiler_options);
    auto ir = compiler.compileResidual(residual_form);

    const auto dispatch = analyzeDispatch(ir);
    FE_THROW_IF(!dispatch.has_cell && !dispatch.has_interior && dispatch.boundary_markers.empty(), InvalidArgumentException,
                "installResidualForm: compiled residual has no integral terms");

    auto kernel = std::make_shared<forms::NonlinearFormKernel>(std::move(ir), options.ad_mode);
    registerKernel(system, op, test_field, trial_field, dispatch, kernel);
    return kernel;
}

std::vector<std::vector<KernelPtr>> installResidualBlocks(
    FESystem& system,
    const OperatorTag& op,
    std::span<const FieldId> test_fields,
    std::span<const FieldId> trial_fields,
    const forms::BlockBilinearForm& blocks,
    const FormInstallOptions& options)
{
    FE_THROW_IF(test_fields.size() != blocks.numTestFields(), InvalidArgumentException,
                "installResidualBlocks: test_fields size does not match blocks.numTestFields()");
    FE_THROW_IF(trial_fields.size() != blocks.numTrialFields(), InvalidArgumentException,
                "installResidualBlocks: trial_fields size does not match blocks.numTrialFields()");

    forms::FormCompiler compiler(options.compiler_options);
    auto compiled = compiler.compileResidual(blocks);

    std::vector<std::vector<KernelPtr>> kernels;
    kernels.resize(blocks.numTestFields());
    for (auto& row : kernels) {
        row.resize(blocks.numTrialFields());
    }

    for (std::size_t i = 0; i < blocks.numTestFields(); ++i) {
        for (std::size_t j = 0; j < blocks.numTrialFields(); ++j) {
            if (i >= compiled.size() || j >= compiled[i].size() || !compiled[i][j].has_value()) continue;

            auto ir = std::move(compiled[i][j].value());
            const auto dispatch = analyzeDispatch(ir);
            FE_THROW_IF(!dispatch.has_cell && !dispatch.has_interior && dispatch.boundary_markers.empty(), InvalidArgumentException,
                        "installResidualBlocks: compiled residual block has no integral terms");

            auto kernel = std::make_shared<forms::NonlinearFormKernel>(std::move(ir), options.ad_mode);
            registerKernel(system, op, test_fields[i], trial_fields[j], dispatch, kernel);
            kernels[i][j] = std::move(kernel);
        }
    }

    return kernels;
}

std::vector<std::vector<KernelPtr>> installResidualBlocks(
    FESystem& system,
    const OperatorTag& op,
    std::initializer_list<FieldId> test_fields,
    std::initializer_list<FieldId> trial_fields,
    const forms::BlockBilinearForm& blocks,
    const FormInstallOptions& options)
{
    const std::span<const FieldId> test_span(test_fields.begin(), test_fields.size());
    const std::span<const FieldId> trial_span(trial_fields.begin(), trial_fields.size());
    return installResidualBlocks(system, op, test_span, trial_span, blocks, options);
}

} // namespace systems
} // namespace FE
} // namespace svmp

