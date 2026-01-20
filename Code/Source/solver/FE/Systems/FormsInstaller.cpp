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
#include "Forms/WeakForm.h"
#include "Forms/AffineAnalysis.h"

#include "Systems/FESystem.h"
#include "Systems/StrongDirichletConstraint.h"

#include <algorithm>
#include <unordered_set>

namespace svmp {
namespace FE {
namespace systems {

namespace {

struct DomainDispatch {
    bool has_cell{false};
    bool has_interior{false};
    bool has_interface{false};
    std::vector<int> boundary_markers{};
    std::vector<int> interface_markers{};
};

DomainDispatch analyzeDispatch(const forms::FormIR& ir)
{
    DomainDispatch out;
    out.has_cell = ir.hasCellTerms();
    out.has_interior = ir.hasInteriorFaceTerms();
    out.has_interface = ir.hasInterfaceFaceTerms();

    if (!ir.hasBoundaryTerms()) {
        out.boundary_markers.clear();
    } else {
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
        } else {
            std::sort(markers.begin(), markers.end());
            markers.erase(std::unique(markers.begin(), markers.end()), markers.end());
            out.boundary_markers = std::move(markers);
        }
    }

    if (!ir.hasInterfaceFaceTerms()) {
        out.interface_markers.clear();
    } else {
        bool has_all_markers = false;
        std::vector<int> markers;
        for (const auto& term : ir.terms()) {
            if (term.domain != forms::IntegralDomain::InterfaceFace) continue;
            if (term.interface_marker < 0) {
                has_all_markers = true;
                break;
            }
            markers.push_back(term.interface_marker);
        }

        if (has_all_markers) {
            out.interface_markers = {-1};
        } else {
            std::sort(markers.begin(), markers.end());
            markers.erase(std::unique(markers.begin(), markers.end()), markers.end());
            out.interface_markers = std::move(markers);
        }
    }

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

    for (int marker : dispatch.interface_markers) {
        system.addInterfaceFaceKernel(op, marker, test_field, trial_field, kernel);
    }
}

std::unordered_set<FieldId> gatherStateFields(const forms::FormExprNode& node)
{
    std::unordered_set<FieldId> out;
    const auto visit = [&](const auto& self, const forms::FormExprNode& n) -> void {
        if (n.type() == forms::FormExprType::StateField) {
            const auto fid = n.fieldId();
            FE_THROW_IF(!fid || *fid == INVALID_FIELD_ID, InvalidArgumentException,
                        "installCoupledResidual: encountered StateField with invalid FieldId");
            out.insert(*fid);
        }
        for (const auto& child : n.childrenShared()) {
            if (child) self(self, *child);
        }
    };
    visit(visit, node);
    return out;
}

bool containsCoupledTerminals(const forms::FormExprNode& node)
{
    switch (node.type()) {
        case forms::FormExprType::BoundaryFunctionalSymbol:
        case forms::FormExprType::BoundaryIntegralSymbol:
        case forms::FormExprType::BoundaryIntegralRef:
        case forms::FormExprType::AuxiliaryStateSymbol:
        case forms::FormExprType::AuxiliaryStateRef:
            return true;
        default:
            break;
    }

    for (const auto& child : node.childrenShared()) {
        if (child && containsCoupledTerminals(*child)) {
            return true;
        }
    }
    return false;
}

forms::FormExpr lowerStateFields(
    const forms::FormExpr& expr,
    FieldId active_trial_field,
    const FESystem& system)
{
    return expr.transformNodes([&](const forms::FormExprNode& n) -> std::optional<forms::FormExpr> {
        if (n.type() != forms::FormExprType::StateField) {
            return std::nullopt;
        }

        const auto fid = n.fieldId();
        FE_THROW_IF(!fid || *fid == INVALID_FIELD_ID, InvalidArgumentException,
                    "installCoupledResidual: StateField node missing FieldId");

        const auto& rec = system.fieldRecord(*fid);
        FE_CHECK_NOT_NULL(rec.space.get(), "installCoupledResidual: field space");

        const std::string sym = n.toString();
        if (*fid == active_trial_field) {
            return forms::FormExpr::trialFunction(*rec.space, sym);
        }
        return forms::FormExpr::discreteField(*fid, *rec.space, sym);
    });
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
    FE_THROW_IF(!dispatch.has_cell && !dispatch.has_interior && !dispatch.has_interface &&
                    dispatch.boundary_markers.empty() && dispatch.interface_markers.empty(),
                InvalidArgumentException,
                "installResidualForm: compiled residual has no integral terms");

    // Attempt an affine split R(u;v) = a(u,v) + L(v). If successful, install a LinearFormKernel
    // that assembles the Jacobian from `a` and the residual vector from `K*u + L`.
    std::string reason;
    const bool has_coupled_terminals =
        residual_form.isValid() && residual_form.node() != nullptr && containsCoupledTerminals(*residual_form.node());
    auto split = has_coupled_terminals
                     ? std::nullopt
                     : forms::trySplitAffineResidual(
                           residual_form,
                           forms::AffineResidualOptions{.allow_time_derivatives = false, .allow_interior_face_terms = false},
                           &reason);

    KernelPtr kernel{};
    if (split.has_value()) {
        auto bilinear_ir = compiler.compileBilinear(split->bilinear);
        std::optional<forms::FormIR> linear_ir;
        if (split->linear.isValid()) {
            linear_ir = compiler.compileLinear(split->linear);
        }
        kernel = std::make_shared<forms::LinearFormKernel>(
            std::move(bilinear_ir),
            std::move(linear_ir),
            forms::LinearKernelOutput::Both);
    } else {
        (void)reason; // reserved for future diagnostics/telemetry
        kernel = std::make_shared<forms::NonlinearFormKernel>(std::move(ir), options.ad_mode);
    }

    registerKernel(system, op, test_field, trial_field, dispatch, kernel);
    return kernel;
}

void installStrongDirichlet(FESystem& system, std::span<const forms::bc::StrongDirichlet> bcs)
{
    for (const auto& bc : bcs) {
        FE_THROW_IF(!bc.isValid(), InvalidArgumentException,
                    "installStrongDirichlet: invalid StrongDirichlet declaration");
        FE_THROW_IF(bc.value.hasTest() || bc.value.hasTrial(), InvalidArgumentException,
                    "installStrongDirichlet: StrongDirichlet value must not contain test/trial functions");
        system.addSystemConstraint(
            std::make_unique<StrongDirichletConstraint>(bc.field, bc.boundary_marker, bc.value, bc.component));
    }
}

KernelPtr installResidualForm(
    FESystem& system,
    const OperatorTag& op,
    FieldId test_field,
    FieldId trial_field,
    const forms::FormExpr& residual_form,
    std::span<const forms::bc::StrongDirichlet> bcs,
    const FormInstallOptions& options)
{
    auto kernel = installResidualForm(system, op, test_field, trial_field, residual_form, options);
    installStrongDirichlet(system, bcs);
    return kernel;
}

KernelPtr installWeakForm(
    FESystem& system,
    const OperatorTag& op,
    FieldId test_field,
    FieldId trial_field,
    const forms::WeakForm& form,
    const FormInstallOptions& options)
{
    const std::span<const forms::bc::StrongDirichlet> bcs(form.strong_constraints.data(),
                                                          form.strong_constraints.size());
    return installResidualForm(system, op, test_field, trial_field, form.residual, bcs, options);
}

std::vector<KernelPtr> installWeakForm(
    FESystem& system,
    std::initializer_list<OperatorTag> ops,
    FieldId test_field,
    FieldId trial_field,
    const forms::WeakForm& form,
    const FormInstallOptions& options)
{
    std::vector<KernelPtr> kernels;
    kernels.reserve(ops.size());

    for (const auto& op : ops) {
        kernels.push_back(installResidualForm(system, op, test_field, trial_field, form.residual, options));
    }

    const std::span<const forms::bc::StrongDirichlet> bcs(form.strong_constraints.data(),
                                                          form.strong_constraints.size());
    installStrongDirichlet(system, bcs);
    return kernels;
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
            FE_THROW_IF(!dispatch.has_cell && !dispatch.has_interior && !dispatch.has_interface &&
                            dispatch.boundary_markers.empty() && dispatch.interface_markers.empty(),
                        InvalidArgumentException,
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

CoupledResidualKernels installCoupledResidual(
    FESystem& system,
    const OperatorTag& op,
    std::span<const FieldId> test_fields,
    std::span<const FieldId> trial_fields,
    const forms::BlockLinearForm& residual_blocks,
    const FormInstallOptions& options)
{
    FE_THROW_IF(test_fields.size() != residual_blocks.numTestFields(), InvalidArgumentException,
                "installCoupledResidual: test_fields size does not match residual_blocks.numTestFields()");
    FE_THROW_IF(test_fields.empty(), InvalidArgumentException,
                "installCoupledResidual: empty test field list");
    FE_THROW_IF(trial_fields.empty(), InvalidArgumentException,
                "installCoupledResidual: empty trial field list");

    forms::FormCompiler compiler(options.compiler_options);

    CoupledResidualKernels out;
    out.residual.resize(residual_blocks.numTestFields());
    out.jacobian_blocks.resize(residual_blocks.numTestFields());
    for (auto& row : out.jacobian_blocks) {
        row.resize(trial_fields.size());
    }

    for (std::size_t i = 0; i < residual_blocks.numTestFields(); ++i) {
        if (!residual_blocks.hasBlock(i)) {
            continue;
        }

        const auto& base_expr = residual_blocks.block(i);
        FE_THROW_IF(!base_expr.hasTest(), InvalidArgumentException,
                    "installCoupledResidual: residual block has no test function");

        // Determine which state fields are referenced by this residual component.
        const auto* root = base_expr.node();
        FE_CHECK_NOT_NULL(root, "installCoupledResidual: residual block root");
        const auto state_fields = gatherStateFields(*root);

        // Choose an active trial field so the residual can be compiled as a Residual form.
        FieldId active_trial = INVALID_FIELD_ID;
        if (i < trial_fields.size() && state_fields.contains(trial_fields[i])) {
            active_trial = trial_fields[i];
        } else {
            for (FieldId fid : trial_fields) {
                if (state_fields.contains(fid)) {
                    active_trial = fid;
                    break;
                }
            }
        }

        FE_THROW_IF(active_trial == INVALID_FIELD_ID, InvalidArgumentException,
                    "installCoupledResidual: residual block does not reference any StateField; cannot determine active trial field");

        // Install residual kernel (vector-only).
        {
            const auto lowered = lowerStateFields(base_expr, active_trial, system);
            auto ir = compiler.compileResidual(lowered);

            const auto dispatch = analyzeDispatch(ir);
            FE_THROW_IF(!dispatch.has_cell && !dispatch.has_interior && !dispatch.has_interface &&
                            dispatch.boundary_markers.empty() && dispatch.interface_markers.empty(),
                        InvalidArgumentException,
                        "installCoupledResidual: compiled residual has no integral terms");

            auto kernel = std::make_shared<forms::NonlinearFormKernel>(
                std::move(ir), options.ad_mode, forms::NonlinearKernelOutput::VectorOnly);
            registerKernel(system, op, test_fields[i], active_trial, dispatch, kernel);
            out.residual[i] = std::move(kernel);
        }

        // Install Jacobian blocks (matrix-only) for every referenced trial field.
        for (std::size_t j = 0; j < trial_fields.size(); ++j) {
            const FieldId trial = trial_fields[j];
            if (!state_fields.contains(trial)) {
                continue; // dR/d(trial) == 0
            }

            const auto lowered = lowerStateFields(base_expr, trial, system);
            auto ir = compiler.compileResidual(lowered);
            const auto dispatch = analyzeDispatch(ir);
            FE_THROW_IF(!dispatch.has_cell && !dispatch.has_interior && !dispatch.has_interface &&
                            dispatch.boundary_markers.empty() && dispatch.interface_markers.empty(),
                        InvalidArgumentException,
                        "installCoupledResidual: compiled Jacobian block has no integral terms");

            auto kernel = std::make_shared<forms::NonlinearFormKernel>(
                std::move(ir), options.ad_mode, forms::NonlinearKernelOutput::MatrixOnly);
            registerKernel(system, op, test_fields[i], trial, dispatch, kernel);
            out.jacobian_blocks[i][j] = std::move(kernel);
        }
    }

    return out;
}

} // namespace systems
} // namespace FE
} // namespace svmp
