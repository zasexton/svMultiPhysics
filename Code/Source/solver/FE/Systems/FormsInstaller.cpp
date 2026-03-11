/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#include "Systems/FormsInstaller.h"

#include <cstdlib>

#include "Core/FEException.h"

#include "Forms/BlockForm.h"
#include "Forms/CoupledBlockKernel.h"
#include "Forms/FormCompiler.h"
#include "Forms/FormKernels.h"
#include "Forms/MixedFormIR.h"
#include "Forms/JIT/JITCompiler.h"
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
            FE_THROW_IF(!fid || *fid == CURRENT_SOLUTION_FIELD_ID, InvalidArgumentException,
                        "installCoupledResidual: encountered StateField with CURRENT_SOLUTION_FIELD_ID sentinel "
                        "(coupled residuals must use explicit named field IDs)");
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
        FE_THROW_IF(!fid || *fid == CURRENT_SOLUTION_FIELD_ID, InvalidArgumentException,
                    "installCoupledResidual: StateField node has CURRENT_SOLUTION_FIELD_ID sentinel "
                    "(coupled residuals must use explicit named field IDs)");

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

    // ========================================================================
    // CoupledBlockKernel: wrap all Jacobian block kernels into a single kernel
    // so that assembleCellsFused can share geometry across blocks.
    // ========================================================================
    if (options.coupled_residual_install_jacobian_blocks &&
        options.compiler_options.jit.enable)
    {
        std::vector<forms::CoupledBlockKernel::BlockSpec> block_specs;
        for (std::size_t i = 0; i < out.jacobian_blocks.size(); ++i) {
            for (std::size_t j = 0; j < out.jacobian_blocks[i].size(); ++j) {
                const auto& kernel = out.jacobian_blocks[i][j];
                if (!kernel) continue;

                forms::CoupledBlockKernel::BlockSpec bs;
                bs.test_field = test_fields[i];
                bs.trial_field = trial_fields[j];
                bs.want_matrix = !kernel->isVectorOnly();
                bs.want_vector = !kernel->isMatrixOnly();
                bs.fallback_kernel = kernel;
                block_specs.push_back(std::move(bs));
            }
        }

        if (block_specs.size() >= 2) {
            auto jit_compiler = forms::jit::JITCompiler::getOrCreate(options.compiler_options.jit);
            std::shared_ptr<assembly::AssemblyKernel> coupled =
                std::make_shared<forms::CoupledBlockKernel>(
                    std::move(block_specs), std::move(jit_compiler), options.compiler_options.jit);

            // Register as a single cell term for the first (test, trial) pair.
            // SystemAssembly will detect this and expand into per-block fused terms.
            system.addCellKernel(op, test_fields[0], trial_fields[0], coupled);
        }
    }

    return out;
}

CoupledResidualKernels installCoupledResidualMixed(
    FESystem& system,
    const OperatorTag& op,
    std::span<const FieldId> test_fields,
    std::span<const FieldId> trial_fields,
    const forms::FormExpr& mixed_residual,
    const FormInstallOptions& options)
{
    FE_THROW_IF(test_fields.empty(), InvalidArgumentException,
                "installCoupledResidualMixed: empty test field list");
    FE_THROW_IF(trial_fields.empty(), InvalidArgumentException,
                "installCoupledResidualMixed: empty trial field list");
    FE_THROW_IF(!mixed_residual.isValid(), InvalidArgumentException,
                "installCoupledResidualMixed: invalid form expression");
    FE_THROW_IF(!mixed_residual.hasTest(), InvalidArgumentException,
                "installCoupledResidualMixed: form has no test function");
    FE_THROW_IF(options.coupled_residual_from_jacobian_block && options.coupled_residual_install_residual_kernels,
                InvalidArgumentException,
                "installCoupledResidualMixed: coupled_residual_from_jacobian_block requires coupled_residual_install_residual_kernels=false");
    FE_THROW_IF(options.coupled_residual_from_jacobian_block && !options.coupled_residual_install_jacobian_blocks,
                InvalidArgumentException,
                "installCoupledResidualMixed: coupled_residual_from_jacobian_block requires coupled_residual_install_jacobian_blocks=true");

    // =========================================================================
    // Phase 1: Decompose the mixed expression into per-test-function sub-expressions.
    //
    // The mixed FormExpr is a sum of integrals, each containing exactly one
    // TestFunction space. We recursively traverse the top-level Add/Subtract/Negate
    // structure, find individual integral terms (CellIntegral, BoundaryIntegral, etc.),
    // split their integrands into additive terms, and classify each term by which
    // TestFunction it contains.
    //
    // This is critical: we decompose by TEST function only. We do NOT filter by
    // TrialFunction here. For Residual-kind forms, terms that depend on only the
    // test function (not the trial) still contribute to the residual vector — they
    // are "constant" w.r.t. the trial and must be included.
    // =========================================================================

    // Find unique test function names/signatures in the expression.
    struct TestInfo {
        forms::FormExprNode::SpaceSignature signature{};
        std::string name{};
    };
    std::vector<TestInfo> test_infos;
    {
        const auto find_tests = [&](const auto& self, const forms::FormExprNode& n) -> void {
            if (n.type() == forms::FormExprType::TestFunction) {
                const auto* sig = n.spaceSignature();
                if (sig) {
                    std::string nm = n.toString();
                    bool found = false;
                    for (const auto& info : test_infos) {
                        if (signaturesMatch(info.signature, *sig) && info.name == nm) {
                            found = true;
                            break;
                        }
                    }
                    if (!found) {
                        test_infos.push_back({*sig, nm});
                    }
                }
            }
            for (const auto& child : n.childrenShared()) {
                if (child) self(self, *child);
            }
        };
        find_tests(find_tests, *mixed_residual.node());
    }

    FE_THROW_IF(test_infos.size() != test_fields.size(), InvalidArgumentException,
                "installCoupledResidualMixed: expression has " + std::to_string(test_infos.size()) +
                " TestFunction spaces but " + std::to_string(test_fields.size()) + " test fields provided");

    // Map test_infos ordering to caller's test_fields by space signature.
    std::vector<std::size_t> test_map(test_infos.size(), ~std::size_t{0});
    for (std::size_t mi = 0; mi < test_infos.size(); ++mi) {
        for (std::size_t ci = 0; ci < test_fields.size(); ++ci) {
            const auto& rec = system.fieldRecord(test_fields[ci]);
            if (rec.space && signaturesMatch(test_infos[mi].signature, signatureFromSpace(*rec.space))) {
                test_map[mi] = ci;
                break;
            }
        }
        FE_THROW_IF(test_map[mi] == ~std::size_t{0}, InvalidArgumentException,
                    "installCoupledResidualMixed: could not match test function '" +
                    test_infos[mi].name + "' to any provided test FieldId");
    }

    // Helper: check if an expression subtree contains a TestFunction with a given name.
    const auto containsTestNamed = [](const auto& self, const forms::FormExprNode& n,
                                      const std::string& name) -> bool {
        if (n.type() == forms::FormExprType::TestFunction && n.toString() == name) {
            return true;
        }
        for (const auto& child : n.childrenShared()) {
            if (child && self(self, *child, name)) return true;
        }
        return false;
    };

    // Decompose the mixed expression into per-test sub-expressions by traversing
    // the top-level Add/Subtract/Negate tree and assigning each sub-tree to the
    // test function it contains. This preserves the original expression structure
    // exactly — no re-assembly of integral terms — so the compiled kernels are
    // structurally identical to what would result from manual decomposition.
    //
    // At each Add/Subtract node, we check if both children contain the SAME set of
    // test functions. If so, we keep the node intact (it belongs to one test block).
    // If they contain DIFFERENT test functions, we split the children into separate
    // test blocks. This handles `momentum_form + continuity_form` naturally: the top
    // Add splits, and each sub-form goes to its respective test block.
    std::vector<forms::FormExpr> test_block_exprs(test_infos.size());

    // Find which test function(s) appear in a sub-expression.
    const auto findTestIndices = [&](const forms::FormExprNode& node) -> std::vector<std::size_t> {
        std::vector<std::size_t> indices;
        for (std::size_t ti = 0; ti < test_infos.size(); ++ti) {
            if (containsTestNamed(containsTestNamed, node, test_infos[ti].name)) {
                indices.push_back(ti);
            }
        }
        return indices;
    };

    const auto assignToBlock = [&](std::size_t ti, forms::FormExpr expr, int sign) {
        if (sign < 0) {
            expr = forms::FormExpr::constant(-1.0) * expr;
        }
        if (!test_block_exprs[ti].isValid()) {
            test_block_exprs[ti] = std::move(expr);
        } else {
            test_block_exprs[ti] = test_block_exprs[ti] + std::move(expr);
        }
    };

    const auto decompose = [&](const auto& self, const forms::FormExpr& expr, int sign) -> void {
        if (!expr.isValid()) return;
        const auto& n = *expr.node();
        const auto kids = n.childrenShared();

        // For Add/Subtract/Negate: check if children go to different test blocks.
        if ((n.type() == forms::FormExprType::Add || n.type() == forms::FormExprType::Subtract)
            && kids.size() == 2 && kids[0] && kids[1])
        {
            const auto left_tests = findTestIndices(*kids[0]);
            const auto right_tests = findTestIndices(*kids[1]);

            // If both sides reference the same single test function, keep the node intact.
            if (left_tests.size() == 1 && right_tests.size() == 1 && left_tests[0] == right_tests[0]) {
                assignToBlock(left_tests[0], expr, sign);
                return;
            }

            // Different test functions or multiple: recurse to split.
            const int right_sign = (n.type() == forms::FormExprType::Subtract) ? -sign : sign;
            self(self, forms::FormExpr(kids[0]), sign);
            self(self, forms::FormExpr(kids[1]), right_sign);
            return;
        }
        if (n.type() == forms::FormExprType::Negate && kids.size() == 1 && kids[0]) {
            self(self, forms::FormExpr(kids[0]), -sign);
            return;
        }

        // Leaf integral or non-Add node: assign to the matching test block.
        const auto tests = findTestIndices(n);
        FE_THROW_IF(tests.empty(), InvalidArgumentException,
                    "installCoupledResidualMixed: sub-expression does not contain any recognized TestFunction");
        FE_THROW_IF(tests.size() > 1, InvalidArgumentException,
                    "installCoupledResidualMixed: sub-expression contains multiple TestFunctions (" +
                    test_infos[tests[0]].name + ", " + test_infos[tests[1]].name + ")");
        assignToBlock(tests[0], expr, sign);
    };

    decompose(decompose, mixed_residual, +1);

    // =========================================================================
    // Phase 2: Build a BlockLinearForm and delegate to installCoupledResidual.
    //
    // The per-test sub-expressions are complete residual contributions for each
    // test function. We reorder them to match the caller's test_fields ordering,
    // then use the proven installCoupledResidual path which correctly handles
    // per-trial lowering + compileResidual (including terms without TrialFunction
    // in the residual vector).
    // =========================================================================

    forms::BlockLinearForm residual_blocks(test_fields.size());
    for (std::size_t mi = 0; mi < test_infos.size(); ++mi) {
        const auto caller_ti = test_map[mi];
        FE_THROW_IF(!test_block_exprs[mi].isValid(), InvalidArgumentException,
                    "installCoupledResidualMixed: test block '" + test_infos[mi].name + "' is empty");
        residual_blocks.setBlock(caller_ti, std::move(test_block_exprs[mi]));
    }

    return installCoupledResidual(system, op, test_fields, trial_fields, residual_blocks, options);
}

std::vector<std::vector<KernelPtr>>
installMixedFormIR(
    FESystem& system,
    const OperatorTag& op,
    std::span<const FieldId> test_fields,
    std::span<const FieldId> trial_fields,
    const forms::MixedFormIR& mir,
    const FormInstallOptions& options)
{
    FE_THROW_IF(test_fields.size() != mir.numTestFields(), InvalidArgumentException,
                "installMixedFormIR: test_fields size does not match mir.numTestFields()");
    FE_THROW_IF(trial_fields.size() != mir.numTrialFields(), InvalidArgumentException,
                "installMixedFormIR: trial_fields size does not match mir.numTrialFields()");

    const auto n_test = mir.numTestFields();
    const auto n_trial = mir.numTrialFields();

    std::vector<std::vector<KernelPtr>> result(n_test);
    for (auto& row : result) row.resize(n_trial);

    for (std::size_t i = 0; i < n_test; ++i) {
        for (std::size_t j = 0; j < n_trial; ++j) {
            if (!mir.hasBlock(i, j)) continue;

            // FormIR is move-only; clone from the const MixedFormIR block.
            auto block_ir = mir.block(i, j).clone();

            const auto dispatch = analyzeDispatch(block_ir);
            FE_THROW_IF(!dispatch.has_cell && !dispatch.has_interior && !dispatch.has_interface &&
                            dispatch.boundary_markers.empty() && dispatch.interface_markers.empty(),
                        InvalidArgumentException,
                        "installMixedFormIR: compiled block has no integral terms");

            KernelPtr kernel{};
            switch (block_ir.kind()) {
                case forms::FormKind::Bilinear:
                case forms::FormKind::Linear:
                    // Bilinear/linear forms: direct evaluation, no AD
                    kernel = std::make_shared<forms::FormKernel>(std::move(block_ir));
                    break;
                case forms::FormKind::Residual:
                    // Residual forms: require AD or symbolic tangent for Jacobian
                    if (options.compiler_options.use_symbolic_tangent) {
                        kernel = std::make_shared<forms::SymbolicNonlinearFormKernel>(
                            std::move(block_ir), forms::NonlinearKernelOutput::Both);
                    } else {
                        kernel = std::make_shared<forms::NonlinearFormKernel>(
                            std::move(block_ir), options.ad_mode);
                    }
                    break;
            }

            kernel = maybeWrapForJIT(std::move(kernel), options);
            registerKernel(system, op, test_fields[i], trial_fields[j], dispatch, kernel);
            result[i][j] = std::move(kernel);
        }
    }

    return result;
}

// ============================================================================
// installFormulation — unified entry point
// ============================================================================

namespace {

std::size_t countUniqueTestSpaces(const forms::FormExprNode& root)
{
    struct SpaceInfo {
        forms::FormExprNode::SpaceSignature sig;
        std::string name;
    };
    std::vector<SpaceInfo> found;

    const auto visit = [&](const auto& self, const forms::FormExprNode& n) -> void {
        if (n.type() == forms::FormExprType::TestFunction) {
            const auto* sig = n.spaceSignature();
            if (sig) {
                const std::string nm = n.toString();
                bool exists = false;
                for (const auto& info : found) {
                    if (info.name == nm && signaturesMatch(info.sig, *sig)) {
                        exists = true;
                        break;
                    }
                }
                if (!exists) {
                    found.push_back({*sig, nm});
                }
            }
        }
        for (const auto& child : n.childrenShared()) {
            if (child) self(self, *child);
        }
    };
    visit(visit, root);
    return found.size();
}

bool hasStateFieldNodes(const forms::FormExprNode& root)
{
    if (root.type() == forms::FormExprType::StateField) return true;
    for (const auto& child : root.childrenShared()) {
        if (child && hasStateFieldNodes(*child)) return true;
    }
    return false;
}

} // namespace

CoupledResidualKernels installFormulation(
    FESystem& system,
    const OperatorTag& op,
    std::span<const FieldId> fields,
    const forms::FormExpr& residual,
    const FormInstallOptions& options)
{
    FE_THROW_IF(fields.empty(), InvalidArgumentException,
                "installFormulation: empty field list");
    FE_THROW_IF(!residual.isValid(), InvalidArgumentException,
                "installFormulation: invalid residual expression");

    const auto num_test = countUniqueTestSpaces(*residual.node());
    FE_THROW_IF(num_test == 0, InvalidArgumentException,
                "installFormulation: residual contains no TestFunction");

    if (num_test == 1) {
        // Single-field path.
        FE_THROW_IF(fields.size() != 1, InvalidArgumentException,
                    "installFormulation: expression has 1 TestFunction space but " +
                    std::to_string(fields.size()) + " fields provided");

        // If the expression uses StateField nodes, lower them to TrialFunction.
        forms::FormExpr lowered = residual;
        if (residual.node() && hasStateFieldNodes(*residual.node())) {
            lowered = lowerStateFields(residual, fields[0], system);
        }

        auto kernel = installResidualForm(system, op, fields[0], fields[0], lowered, options);

        CoupledResidualKernels out;
        out.residual = {kernel};
        out.jacobian_blocks = {{kernel}};
        return out;
    }

    // Multi-field path: auto-set optimal coupled assembly options.
    FormInstallOptions coupled_opts = options;
    coupled_opts.coupled_residual_install_residual_kernels = false;
    coupled_opts.coupled_residual_install_jacobian_blocks = true;
    coupled_opts.coupled_residual_from_jacobian_block = true;

    return installCoupledResidualMixed(system, op, fields, fields, residual, coupled_opts);
}

CoupledResidualKernels installFormulation(
    FESystem& system,
    const OperatorTag& op,
    std::initializer_list<FieldId> fields,
    const forms::FormExpr& residual,
    const FormInstallOptions& options)
{
    const std::span<const FieldId> span(fields.begin(), fields.size());
    return installFormulation(system, op, span, residual, options);
}

} // namespace systems
} // namespace FE
} // namespace svmp
