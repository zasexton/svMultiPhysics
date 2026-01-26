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
#include "Forms/JIT/JITKernelWrapper.h"
#include "Forms/WeakForm.h"
#include "Forms/AffineAnalysis.h"

#include "Systems/FESystem.h"
#include "Systems/StrongDirichletConstraint.h"
#include "Spaces/FunctionSpace.h"

#include <algorithm>
#include <unordered_set>

namespace svmp {
namespace FE {
namespace systems {

namespace {

forms::FormExprNode::SpaceSignature signatureFromSpace(const spaces::FunctionSpace& space)
{
    forms::FormExprNode::SpaceSignature sig;
    sig.space_type = space.space_type();
    sig.field_type = space.field_type();
    sig.continuity = space.continuity();
    sig.value_dimension = space.value_dimension();
    sig.topological_dimension = space.topological_dimension();
    sig.polynomial_order = space.polynomial_order();
    sig.element_type = space.element_type();
    return sig;
}

bool signaturesMatch(const forms::FormExprNode::SpaceSignature& a,
                     const forms::FormExprNode::SpaceSignature& b) noexcept
{
    return a.space_type == b.space_type &&
           a.field_type == b.field_type &&
           a.continuity == b.continuity &&
           a.value_dimension == b.value_dimension &&
           a.topological_dimension == b.topological_dimension &&
           a.polynomial_order == b.polynomial_order &&
           a.element_type == b.element_type;
}

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

KernelPtr maybeWrapForJIT(KernelPtr kernel, const FormInstallOptions& options)
{
    if (!kernel || !options.compiler_options.jit.enable) {
        return kernel;
    }
    return std::make_shared<forms::jit::JITKernelWrapper>(std::move(kernel), options.compiler_options.jit);
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
    const auto& test_rec = system.fieldRecord(test_field);
    const auto& trial_rec = system.fieldRecord(trial_field);
    FE_CHECK_NOT_NULL(test_rec.space.get(), "installResidualForm: test field space");
    FE_CHECK_NOT_NULL(trial_rec.space.get(), "installResidualForm: trial field space");

    forms::FormCompiler compiler(options.compiler_options);
    auto ir = compiler.compileResidual(residual_form);

    FE_THROW_IF(!ir.testSpace().has_value(), InvalidArgumentException,
                "installResidualForm: compiled residual missing TestFunction space");
    FE_THROW_IF(!ir.trialSpace().has_value(), InvalidArgumentException,
                "installResidualForm: compiled residual missing TrialFunction space");

    const auto expected_test = signatureFromSpace(*test_rec.space);
    const auto expected_trial = signatureFromSpace(*trial_rec.space);
    FE_THROW_IF(!signaturesMatch(*ir.testSpace(), expected_test), InvalidArgumentException,
                "installResidualForm: TestFunction space does not match registered test_field space");
    FE_THROW_IF(!signaturesMatch(*ir.trialSpace(), expected_trial), InvalidArgumentException,
                "installResidualForm: TrialFunction space does not match registered trial_field space");

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
        if (options.compiler_options.use_symbolic_tangent) {
            kernel = std::make_shared<forms::SymbolicNonlinearFormKernel>(std::move(ir), forms::NonlinearKernelOutput::Both);
        } else {
            kernel = std::make_shared<forms::NonlinearFormKernel>(std::move(ir), options.ad_mode);
        }
    }

    kernel = maybeWrapForJIT(std::move(kernel), options);
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

            KernelPtr kernel{};
            if (options.compiler_options.use_symbolic_tangent) {
                kernel = maybeWrapForJIT(std::make_shared<forms::SymbolicNonlinearFormKernel>(
                                             std::move(ir), forms::NonlinearKernelOutput::Both),
                                         options);
            } else {
                kernel = maybeWrapForJIT(std::make_shared<forms::NonlinearFormKernel>(std::move(ir), options.ad_mode), options);
            }
            registerKernel(system, op, test_fields[i], trial_fields[j], dispatch, kernel);
            kernels[i][j] = kernel;
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
    FE_THROW_IF(options.coupled_residual_from_jacobian_block && options.coupled_residual_install_residual_kernels,
                InvalidArgumentException,
                "installCoupledResidual: coupled_residual_from_jacobian_block requires coupled_residual_install_residual_kernels=false");
    FE_THROW_IF(options.coupled_residual_from_jacobian_block && !options.coupled_residual_install_jacobian_blocks,
                InvalidArgumentException,
                "installCoupledResidual: coupled_residual_from_jacobian_block requires coupled_residual_install_jacobian_blocks=true");

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

        if (options.coupled_residual_install_residual_kernels) {
            // Install residual kernel (vector-only).
            const auto lowered = lowerStateFields(base_expr, active_trial, system);
            auto ir = compiler.compileResidual(lowered);

            const auto dispatch = analyzeDispatch(ir);
            FE_THROW_IF(!dispatch.has_cell && !dispatch.has_interior && !dispatch.has_interface &&
                            dispatch.boundary_markers.empty() && dispatch.interface_markers.empty(),
                        InvalidArgumentException,
                        "installCoupledResidual: compiled residual has no integral terms");

            KernelPtr kernel{};
            if (options.compiler_options.use_symbolic_tangent) {
                kernel = maybeWrapForJIT(std::make_shared<forms::SymbolicNonlinearFormKernel>(
                                             std::move(ir), forms::NonlinearKernelOutput::VectorOnly),
                                         options);
            } else {
                kernel = maybeWrapForJIT(std::make_shared<forms::NonlinearFormKernel>(
                                             std::move(ir), options.ad_mode, forms::NonlinearKernelOutput::VectorOnly),
                                         options);
            }
            registerKernel(system, op, test_fields[i], active_trial, dispatch, kernel);
            out.residual[i] = kernel;
        }

        if (options.coupled_residual_install_jacobian_blocks) {
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

                const auto output = (options.coupled_residual_from_jacobian_block && trial == active_trial)
                    ? forms::NonlinearKernelOutput::Both
                    : forms::NonlinearKernelOutput::MatrixOnly;

                KernelPtr kernel{};
                if (options.compiler_options.use_symbolic_tangent) {
                    kernel = maybeWrapForJIT(std::make_shared<forms::SymbolicNonlinearFormKernel>(std::move(ir), output),
                                             options);
                } else {
                    kernel = maybeWrapForJIT(std::make_shared<forms::NonlinearFormKernel>(
                                                 std::move(ir), options.ad_mode, output),
                                             options);
                }
                registerKernel(system, op, test_fields[i], trial, dispatch, kernel);
                out.jacobian_blocks[i][j] = kernel;
                if (output == forms::NonlinearKernelOutput::Both) {
                    out.residual[i] = kernel;
                }
            }
        }
    }

    return out;
}

} // namespace systems
} // namespace FE
} // namespace svmp
