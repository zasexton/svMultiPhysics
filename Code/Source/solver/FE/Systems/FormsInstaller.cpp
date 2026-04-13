/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#include "Systems/FormsInstaller.h"

#include <cstdlib>

#include "Core/FEException.h"
#include "Core/KernelTrace.h"

#include "Auxiliary/AuxiliaryInputRegistry.h"

#include "Forms/BlockForm.h"
#include "Forms/FormCompiler.h"
#include "Forms/FormKernels.h"
#include "Forms/MixedBlockKernelSet.h"
#include "Forms/MixedFormIR.h"
#include "Forms/MonolithicCellKernel.h"
#include "Forms/SymbolicDifferentiation.h"
#include "Forms/JIT/JITCompiler.h"
#include "Forms/JIT/JITKernelWrapper.h"
#include "Forms/WeakForm.h"
#include "Forms/AffineAnalysis.h"

#include "Analysis/FormulationRecord.h"
#include "Analysis/FormExprScanner.h"
#include "Analysis/FormContributionLowerer.h"

#include "Systems/FESystem.h"
#include "Constraints/StrongDirichletConstraint.h"
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

void registerKernelDomains(
    FESystem& system,
    const OperatorTag& op,
    FieldId test_field,
    FieldId trial_field,
    const DomainDispatch& dispatch,
    const KernelPtr& kernel,
    bool include_cell)
{
    if (include_cell && dispatch.has_cell) {
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

[[nodiscard]] bool dispatchHasAnyTerm(const DomainDispatch& dispatch) noexcept
{
    return dispatch.has_cell || dispatch.has_interior || dispatch.has_interface ||
           !dispatch.boundary_markers.empty() || !dispatch.interface_markers.empty();
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

[[nodiscard]] std::shared_ptr<MixedKernelPlan> makeMixedKernelPlan(const FormInstallOptions& options)
{
    auto plan = std::make_shared<MixedKernelPlan>();
    plan->jit_requested = options.compiler_options.jit.enable;
    plan->monolithic_cell_requested = options.compiler_options.jit.enable;
    return plan;
}

void traceMixedKernelPlan(const char* prefix, const MixedKernelPlan& plan)
{
    if (!core::kernelTraceEnabled(core::KernelTraceChannel::Selection)) {
        return;
    }
    core::kernelTraceLog(
        core::KernelTraceChannel::Selection,
        std::string(prefix) + ": " + plan.describe());
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
            std::make_unique<constraints::StrongDirichletConstraint>(bc.field, bc.boundary_marker, bc.value, bc.component));
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

    forms::FormCompiler compiler(options.compiler_options);

    struct PendingKernelInstall {
        FieldId test_field{INVALID_FIELD_ID};
        FieldId trial_field{INVALID_FIELD_ID};
        DomainDispatch dispatch{};
        KernelPtr kernel{};
        bool cell_semantics_owned_by_monolithic{false};
    };

    CoupledResidualKernels out;
    out.residual.resize(residual_blocks.numTestFields());
    out.jacobian_blocks.resize(residual_blocks.numTestFields());
    for (auto& row : out.jacobian_blocks) {
        row.resize(trial_fields.size());
    }

    auto plan = makeMixedKernelPlan(options);
    std::vector<PendingKernelInstall> pending_installs;
    std::vector<forms::MixedBlockKernelSet::BlockSpec> mixed_block_cell_specs;
    std::vector<forms::MonolithicCellKernel::BlockSpec> monolithic_cell_blocks;
    bool monolithic_cell_feasible = plan->monolithic_cell_requested;
    std::string monolithic_disable_reason{};

    for (std::size_t i = 0; i < residual_blocks.numTestFields(); ++i) {
        if (!residual_blocks.hasBlock(i)) {
            continue;
        }

        const auto& base_expr = residual_blocks.block(i);
        FE_THROW_IF(!base_expr.hasTest(), InvalidArgumentException,
                    "installCoupledResidual: residual block has no test function");

        const auto* root = base_expr.node();
        FE_CHECK_NOT_NULL(root, "installCoupledResidual: residual block root");
        const auto state_fields = gatherStateFields(*root);

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

        if (active_trial == INVALID_FIELD_ID) {
            auto linear_ir = compiler.compileLinear(base_expr);
            const auto dispatch = analyzeDispatch(linear_ir);
            if (!dispatchHasAnyTerm(dispatch)) {
                continue;
            }

            auto linear_ir_for_plan = linear_ir.clone();
            KernelPtr kernel = maybeWrapForJIT(
                std::make_shared<forms::FormKernel>(std::move(linear_ir)),
                options);

            out.residual[i] = kernel;
            pending_installs.push_back(PendingKernelInstall{
                .test_field = test_fields[i],
                .trial_field = test_fields[i],
                .dispatch = dispatch,
                .kernel = kernel,
                .cell_semantics_owned_by_monolithic = dispatch.has_cell,
            });

            plan->blocks.push_back(MixedKernelPlanBlock{
                .test_field = test_fields[i],
                .trial_field = test_fields[i],
                .residual_owner_field = test_fields[i],
                .has_cell = dispatch.has_cell,
                .has_boundary = !dispatch.boundary_markers.empty(),
                .has_interior = dispatch.has_interior,
                .has_interface = dispatch.has_interface,
                .want_matrix = false,
                .want_vector = true,
            });

            if (dispatch.has_cell) {
                mixed_block_cell_specs.push_back(forms::MixedBlockKernelSet::BlockSpec{
                    .test_field = test_fields[i],
                    .trial_field = test_fields[i],
                    .want_matrix = false,
                    .want_vector = true,
                    .fallback_kernel = kernel,
                });
                monolithic_cell_blocks.push_back(forms::MonolithicCellKernel::BlockSpec{
                    .test_field = test_fields[i],
                    .trial_field = test_fields[i],
                    .want_matrix = false,
                    .want_vector = true,
                    .fallback_kernel = kernel,
                    .tangent_ir = std::nullopt,
                    .residual_ir = std::optional<forms::FormIR>{std::in_place, std::move(linear_ir_for_plan)},
                });
            }
            continue;
        }

        for (std::size_t j = 0; j < trial_fields.size(); ++j) {
            const FieldId trial = trial_fields[j];
            if (!state_fields.contains(trial)) {
                continue;
            }

            const auto lowered = lowerStateFields(base_expr, trial, system);
            auto residual_ir = compiler.compileResidual(lowered);
            const auto dispatch = analyzeDispatch(residual_ir);
            FE_THROW_IF(!dispatchHasAnyTerm(dispatch), InvalidArgumentException,
                        "installCoupledResidual: compiled Jacobian block has no integral terms");

            const bool owns_row_vector = (trial == active_trial);
            const auto output = owns_row_vector
                ? forms::NonlinearKernelOutput::Both
                : forms::NonlinearKernelOutput::MatrixOnly;

            std::optional<forms::FormIR> tangent_ir_for_plan;
            if (dispatch.has_cell && monolithic_cell_feasible) {
                try {
                    auto tangent_expr = forms::differentiateResidual(lowered);
                    tangent_ir_for_plan = compiler.compileBilinear(tangent_expr);
                } catch (const std::exception& e) {
                    monolithic_cell_feasible = false;
                    monolithic_disable_reason =
                        "installCoupledResidual: symbolic tangent generation failed for (" +
                        std::to_string(test_fields[i]) + "," + std::to_string(trial) + "): " + e.what();
                }
            }

            auto residual_ir_for_plan = residual_ir.clone();
            KernelPtr kernel{};
            if (options.compiler_options.use_symbolic_tangent) {
                kernel = maybeWrapForJIT(
                    std::make_shared<forms::SymbolicNonlinearFormKernel>(std::move(residual_ir), output),
                    options);
            } else {
                kernel = maybeWrapForJIT(
                    std::make_shared<forms::NonlinearFormKernel>(std::move(residual_ir), options.ad_mode, output),
                    options);
            }

            out.jacobian_blocks[i][j] = kernel;
            if (owns_row_vector) {
                out.residual[i] = kernel;
            }

            pending_installs.push_back(PendingKernelInstall{
                .test_field = test_fields[i],
                .trial_field = trial,
                .dispatch = dispatch,
                .kernel = kernel,
                .cell_semantics_owned_by_monolithic = dispatch.has_cell,
            });

            plan->blocks.push_back(MixedKernelPlanBlock{
                .test_field = test_fields[i],
                .trial_field = trial,
                .residual_owner_field = active_trial,
                .has_cell = dispatch.has_cell,
                .has_boundary = !dispatch.boundary_markers.empty(),
                .has_interior = dispatch.has_interior,
                .has_interface = dispatch.has_interface,
                .want_matrix = true,
                .want_vector = owns_row_vector,
            });

            if (dispatch.has_cell && tangent_ir_for_plan.has_value()) {
                monolithic_cell_blocks.push_back(forms::MonolithicCellKernel::BlockSpec{
                    .test_field = test_fields[i],
                    .trial_field = trial,
                    .want_matrix = true,
                    .want_vector = owns_row_vector,
                    .fallback_kernel = kernel,
                    .tangent_ir = std::move(tangent_ir_for_plan),
                    .residual_ir = owns_row_vector ? std::optional<forms::FormIR>{std::in_place, std::move(residual_ir_for_plan)}
                                                  : std::nullopt,
                });
            }
            if (dispatch.has_cell) {
                mixed_block_cell_specs.push_back(forms::MixedBlockKernelSet::BlockSpec{
                    .test_field = test_fields[i],
                    .trial_field = trial,
                    .want_matrix = true,
                    .want_vector = owns_row_vector,
                    .fallback_kernel = kernel,
                });
            }
        }
    }

    plan->semantic_type = MixedKernelSemanticType::MixedBlockSet;
    plan->monolithic_cell_enabled =
        monolithic_cell_feasible && plan->monolithic_cell_requested && monolithic_cell_blocks.size() >= 2u;
    if (plan->monolithic_cell_enabled) {
        plan->semantic_type = MixedKernelSemanticType::MonolithicCell;
    }
    const bool use_mixed_block_cell_kernel =
        !plan->monolithic_cell_enabled && mixed_block_cell_specs.size() >= 2u;

    traceMixedKernelPlan("installCoupledResidual", *plan);
    if (!monolithic_disable_reason.empty() &&
        core::kernelTraceEnabled(core::KernelTraceChannel::Selection)) {
        core::kernelTraceLog(core::KernelTraceChannel::Selection, monolithic_disable_reason);
    }

    for (const auto& pending : pending_installs) {
        registerKernelDomains(
            system,
            op,
            pending.test_field,
            pending.trial_field,
            pending.dispatch,
            pending.kernel,
            (!plan->monolithic_cell_enabled && !use_mixed_block_cell_kernel) ||
                !pending.cell_semantics_owned_by_monolithic);
    }

    if (plan->monolithic_cell_enabled) {
        auto jit_compiler = forms::jit::JITCompiler::getOrCreate(options.compiler_options.jit);
        auto monolithic_kernel = std::make_shared<forms::MonolithicCellKernel>(
            std::move(monolithic_cell_blocks), std::move(jit_compiler), options.compiler_options.jit);
        system.addCellKernel(op, test_fields[0], trial_fields[0], monolithic_kernel);
    } else if (use_mixed_block_cell_kernel) {
        std::shared_ptr<forms::jit::JITCompiler> jit_compiler{};
        if (options.compiler_options.jit.enable) {
            jit_compiler = forms::jit::JITCompiler::getOrCreate(options.compiler_options.jit);
        }
        auto mixed_block_kernel = std::make_shared<forms::MixedBlockKernelSet>(
            std::move(mixed_block_cell_specs), std::move(jit_compiler), options.compiler_options.jit);
        system.addCellKernel(op, test_fields[0], trial_fields[0], mixed_block_kernel);
    }

    out.mixed_plan = std::move(plan);
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

    // Find unique test function rows in the expression. Deduplication uses
    // the field binding (FieldId) when present, otherwise (signature, name).
    // This allows same-name test functions with different field bindings
    // (e.g., TestField(T_f, V, "v") and TestField(C_f, V, "v")) to be treated
    // as distinct rows.
    struct TestInfo {
        forms::FormExprNode::SpaceSignature signature{};
        std::string name{};
        std::optional<FieldId> bound_field{};
    };
    std::vector<TestInfo> test_infos;
    {
        const auto find_tests = [&](const auto& self, const forms::FormExprNode& n) -> void {
            if (n.type() == forms::FormExprType::TestFunction) {
                const auto* sig = n.spaceSignature();
                if (sig) {
                    std::string nm = n.toString();
                    auto fid = n.fieldId();
                    bool found = false;
                    for (const auto& info : test_infos) {
                        if (fid.has_value() && info.bound_field.has_value()) {
                            if (*fid == *info.bound_field) {
                                // Same FieldId: verify consistent space and name
                                FE_THROW_IF(!signaturesMatch(info.signature, *sig) || info.name != nm,
                                    InvalidArgumentException,
                                    "installCoupledResidualMixed: conflicting test "
                                    "functions bound to FieldId " + std::to_string(*fid) +
                                    " — same field must use consistent space and name");
                                found = true;
                                break;
                            }
                        } else if (!fid.has_value() && !info.bound_field.has_value()) {
                            // Neither has field bindings: deduplicate by (signature, name)
                            if (signaturesMatch(info.signature, *sig) && info.name == nm) {
                                found = true; break;
                            }
                        }
                        // One bound, one not: always distinct
                    }
                    if (!found) {
                        test_infos.push_back({*sig, nm, fid});
                    }
                }
            }
            for (const auto& child : n.childrenShared()) {
                if (child) self(self, *child);
            }
        };
        find_tests(find_tests, *mixed_residual.node());
    }

    // Reject duplicate unbound test function names across different spaces.
    for (std::size_t i = 0; i < test_infos.size(); ++i) {
        if (test_infos[i].bound_field.has_value()) continue;
        for (std::size_t j = i + 1; j < test_infos.size(); ++j) {
            if (test_infos[j].bound_field.has_value()) continue;
            FE_THROW_IF(
                test_infos[i].name == test_infos[j].name &&
                !signaturesMatch(test_infos[i].signature, test_infos[j].signature),
                InvalidArgumentException,
                "installCoupledResidualMixed: duplicate TestFunction name '" +
                test_infos[i].name + "' used with different spaces — use distinct names");
        }
    }

    FE_THROW_IF(test_infos.size() != test_fields.size(), InvalidArgumentException,
                "installCoupledResidualMixed: expression has " + std::to_string(test_infos.size()) +
                " TestFunction spaces but " + std::to_string(test_fields.size()) + " test fields provided");

    // Map test_infos ordering to caller's test_fields.
    // Strategy: prefer explicit field bindings (from TestField(field_id, ...)),
    // then fall back to space-signature matching. Reject ambiguous same-space
    // mapping when no bindings are present.
    std::vector<std::size_t> test_map(test_infos.size(), ~std::size_t{0});
    std::vector<bool> field_used(test_fields.size(), false);

    // Pass 1: map by explicit field binding, with space validation
    for (std::size_t mi = 0; mi < test_infos.size(); ++mi) {
        if (!test_infos[mi].bound_field) continue;
        const FieldId bound = *test_infos[mi].bound_field;
        for (std::size_t ci = 0; ci < test_fields.size(); ++ci) {
            if (test_fields[ci] == bound && !field_used[ci]) {
                // Validate that the test function's space matches the field's space
                const auto& rec = system.fieldRecord(test_fields[ci]);
                FE_THROW_IF(rec.space && !signaturesMatch(test_infos[mi].signature,
                                                           signatureFromSpace(*rec.space)),
                            InvalidArgumentException,
                            "installCoupledResidualMixed: test function '" +
                            test_infos[mi].name + "' is bound to FieldId " +
                            std::to_string(bound) + " but its space does not match the "
                            "registered field space");
                test_map[mi] = ci;
                field_used[ci] = true;
                break;
            }
        }
        FE_THROW_IF(test_map[mi] == ~std::size_t{0}, InvalidArgumentException,
                    "installCoupledResidualMixed: test function '" +
                    test_infos[mi].name + "' is bound to FieldId " +
                    std::to_string(bound) + " which is not in the provided field list");
    }

    // Pass 2: map remaining by space signature (only succeeds if unambiguous)
    for (std::size_t mi = 0; mi < test_infos.size(); ++mi) {
        if (test_map[mi] != ~std::size_t{0}) continue;  // already mapped

        std::size_t match_count = 0;
        std::size_t last_match = ~std::size_t{0};
        for (std::size_t ci = 0; ci < test_fields.size(); ++ci) {
            if (field_used[ci]) continue;
            const auto& rec = system.fieldRecord(test_fields[ci]);
            if (rec.space && signaturesMatch(test_infos[mi].signature, signatureFromSpace(*rec.space))) {
                ++match_count;
                last_match = ci;
            }
        }

        FE_THROW_IF(match_count == 0, InvalidArgumentException,
                    "installCoupledResidualMixed: could not match test function '" +
                    test_infos[mi].name + "' to any provided test FieldId");

        FE_THROW_IF(match_count > 1, InvalidArgumentException,
                    "installCoupledResidualMixed: test function '" +
                    test_infos[mi].name + "' matches " + std::to_string(match_count) +
                    " fields with the same space — use TestField(field_id, space, name) "
                    "to resolve the ambiguity");

        test_map[mi] = last_match;
        field_used[last_match] = true;
    }

    // Helper: check if an expression subtree contains a TestFunction matching
    // a given TestInfo. Bound infos match only bound nodes (by FieldId); unbound
    // infos match only unbound nodes (by name). This prevents cross-matching
    // between bound and unbound test functions that share a name.
    const auto containsTestMatching = [](const auto& self, const forms::FormExprNode& n,
                                         const TestInfo& info) -> bool {
        if (n.type() == forms::FormExprType::TestFunction) {
            auto node_fid = n.fieldId();
            if (info.bound_field.has_value()) {
                // Bound info: only matches bound nodes with the same FieldId
                if (node_fid.has_value() && *node_fid == *info.bound_field) return true;
            } else {
                // Unbound info: only matches unbound nodes with the same name
                if (!node_fid.has_value() && n.toString() == info.name) return true;
            }
        }
        for (const auto& child : n.childrenShared()) {
            if (child && self(self, *child, info)) return true;
        }
        return false;
    };

    // Decompose the mixed expression into per-test sub-expressions.
    std::vector<forms::FormExpr> test_block_exprs(test_infos.size());

    // Find which test function(s) appear in a sub-expression.
    const auto findTestIndices = [&](const forms::FormExprNode& node) -> std::vector<std::size_t> {
        std::vector<std::size_t> indices;
        for (std::size_t ti = 0; ti < test_infos.size(); ++ti) {
            if (containsTestMatching(containsTestMatching, node, test_infos[ti])) {
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

    forms::BlockLinearForm residual_blocks(test_fields.size());
    for (std::size_t mi = 0; mi < test_infos.size(); ++mi) {
        const auto caller_ti = test_map[mi];
        FE_THROW_IF(!test_block_exprs[mi].isValid(), InvalidArgumentException,
                    "installCoupledResidualMixed: test block '" + test_infos[mi].name + "' is empty");
        residual_blocks.setBlock(caller_ti, std::move(test_block_exprs[mi]));
    }

    struct PendingKernelInstall {
        FieldId test_field{INVALID_FIELD_ID};
        FieldId trial_field{INVALID_FIELD_ID};
        DomainDispatch dispatch{};
        KernelPtr kernel{};
        bool cell_semantics_owned_by_monolithic{false};
    };

    CoupledResidualKernels out;
    out.residual.resize(test_fields.size());
    out.jacobian_blocks.resize(test_fields.size());
    for (auto& row : out.jacobian_blocks) {
        row.resize(trial_fields.size());
    }

    auto plan = makeMixedKernelPlan(options);
    std::vector<PendingKernelInstall> pending_installs;
    std::vector<forms::MixedBlockKernelSet::BlockSpec> mixed_block_cell_specs;
    std::vector<forms::MonolithicCellKernel::BlockSpec> monolithic_cell_blocks;
    bool monolithic_cell_feasible = plan->monolithic_cell_requested;
    std::string monolithic_disable_reason{};

    forms::FormCompiler compiler(options.compiler_options);

    for (std::size_t i = 0; i < residual_blocks.numTestFields(); ++i) {
        if (!residual_blocks.hasBlock(i)) {
            continue;
        }

        const auto& base_expr = residual_blocks.block(i);
        FE_THROW_IF(!base_expr.isValid(), InvalidArgumentException,
                    "installCoupledResidualMixed: invalid residual block");

        const auto* root = base_expr.node();
        FE_CHECK_NOT_NULL(root, "installCoupledResidualMixed: residual block root");
        const auto state_fields = gatherStateFields(*root);

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

        if (active_trial == INVALID_FIELD_ID) {
            auto linear_ir = compiler.compileLinear(base_expr);
            const auto dispatch = analyzeDispatch(linear_ir);
            if (!dispatchHasAnyTerm(dispatch)) {
                continue;
            }

            auto linear_ir_for_plan = linear_ir.clone();
            KernelPtr kernel = maybeWrapForJIT(
                std::make_shared<forms::FormKernel>(std::move(linear_ir)),
                options);

            out.residual[i] = kernel;
            pending_installs.push_back(PendingKernelInstall{
                .test_field = test_fields[i],
                .trial_field = test_fields[i],
                .dispatch = dispatch,
                .kernel = kernel,
                .cell_semantics_owned_by_monolithic = dispatch.has_cell,
            });

            plan->blocks.push_back(MixedKernelPlanBlock{
                .test_field = test_fields[i],
                .trial_field = test_fields[i],
                .residual_owner_field = test_fields[i],
                .has_cell = dispatch.has_cell,
                .has_boundary = !dispatch.boundary_markers.empty(),
                .has_interior = dispatch.has_interior,
                .has_interface = dispatch.has_interface,
                .want_matrix = false,
                .want_vector = true,
            });

            if (dispatch.has_cell) {
                mixed_block_cell_specs.push_back(forms::MixedBlockKernelSet::BlockSpec{
                    .test_field = test_fields[i],
                    .trial_field = test_fields[i],
                    .want_matrix = false,
                    .want_vector = true,
                    .fallback_kernel = kernel,
                });
                monolithic_cell_blocks.push_back(forms::MonolithicCellKernel::BlockSpec{
                    .test_field = test_fields[i],
                    .trial_field = test_fields[i],
                    .want_matrix = false,
                    .want_vector = true,
                    .fallback_kernel = kernel,
                    .tangent_ir = std::nullopt,
                    .residual_ir = std::optional<forms::FormIR>{std::in_place, std::move(linear_ir_for_plan)},
                });
            }
            continue;
        }

        for (std::size_t j = 0; j < trial_fields.size(); ++j) {
            const FieldId trial = trial_fields[j];
            if (!state_fields.contains(trial)) {
                continue;
            }

            const auto lowered = lowerStateFields(base_expr, trial, system);
            auto residual_ir = compiler.compileResidual(lowered);
            const auto dispatch = analyzeDispatch(residual_ir);
            FE_THROW_IF(!dispatchHasAnyTerm(dispatch), InvalidArgumentException,
                        "installCoupledResidualMixed: compiled residual block has no integral terms");

            const bool owns_row_vector = (trial == active_trial);
            const auto output = owns_row_vector
                ? forms::NonlinearKernelOutput::Both
                : forms::NonlinearKernelOutput::MatrixOnly;

            std::optional<forms::FormIR> tangent_ir_for_plan;
            if (dispatch.has_cell && monolithic_cell_feasible) {
                try {
                    auto tangent_expr = forms::differentiateResidual(lowered);
                    tangent_ir_for_plan = compiler.compileBilinear(tangent_expr);
                } catch (const std::exception& e) {
                    monolithic_cell_feasible = false;
                    monolithic_disable_reason =
                        "installCoupledResidualMixed: symbolic tangent generation failed for (" +
                        std::to_string(test_fields[i]) + "," + std::to_string(trial) + "): " + e.what();
                }
            }

            auto residual_ir_for_plan = residual_ir.clone();
            KernelPtr kernel{};
            if (options.compiler_options.use_symbolic_tangent) {
                kernel = maybeWrapForJIT(
                    std::make_shared<forms::SymbolicNonlinearFormKernel>(std::move(residual_ir), output),
                    options);
            } else {
                kernel = maybeWrapForJIT(
                    std::make_shared<forms::NonlinearFormKernel>(std::move(residual_ir), options.ad_mode, output),
                    options);
            }

            out.jacobian_blocks[i][j] = kernel;
            if (owns_row_vector) {
                out.residual[i] = kernel;
            }

            pending_installs.push_back(PendingKernelInstall{
                .test_field = test_fields[i],
                .trial_field = trial,
                .dispatch = dispatch,
                .kernel = kernel,
                .cell_semantics_owned_by_monolithic = dispatch.has_cell,
            });

            plan->blocks.push_back(MixedKernelPlanBlock{
                .test_field = test_fields[i],
                .trial_field = trial,
                .residual_owner_field = active_trial,
                .has_cell = dispatch.has_cell,
                .has_boundary = !dispatch.boundary_markers.empty(),
                .has_interior = dispatch.has_interior,
                .has_interface = dispatch.has_interface,
                .want_matrix = true,
                .want_vector = owns_row_vector,
            });

            if (dispatch.has_cell && tangent_ir_for_plan.has_value()) {
                monolithic_cell_blocks.push_back(forms::MonolithicCellKernel::BlockSpec{
                    .test_field = test_fields[i],
                    .trial_field = trial,
                    .want_matrix = true,
                    .want_vector = owns_row_vector,
                    .fallback_kernel = kernel,
                    .tangent_ir = std::move(tangent_ir_for_plan),
                    .residual_ir = owns_row_vector ? std::optional<forms::FormIR>{std::in_place, std::move(residual_ir_for_plan)}
                                                  : std::nullopt,
                });
            }
            if (dispatch.has_cell) {
                mixed_block_cell_specs.push_back(forms::MixedBlockKernelSet::BlockSpec{
                    .test_field = test_fields[i],
                    .trial_field = trial,
                    .want_matrix = true,
                    .want_vector = owns_row_vector,
                    .fallback_kernel = kernel,
                });
            }
        }
    }

    plan->semantic_type = MixedKernelSemanticType::MixedBlockSet;
    plan->monolithic_cell_enabled =
        monolithic_cell_feasible && plan->monolithic_cell_requested && monolithic_cell_blocks.size() >= 2u;
    if (plan->monolithic_cell_enabled) {
        plan->semantic_type = MixedKernelSemanticType::MonolithicCell;
    }
    const bool use_mixed_block_cell_kernel =
        !plan->monolithic_cell_enabled && mixed_block_cell_specs.size() >= 2u;

    traceMixedKernelPlan("installCoupledResidualMixed", *plan);
    if (!monolithic_disable_reason.empty() &&
        core::kernelTraceEnabled(core::KernelTraceChannel::Selection)) {
        core::kernelTraceLog(core::KernelTraceChannel::Selection, monolithic_disable_reason);
    }

    for (const auto& pending : pending_installs) {
        registerKernelDomains(
            system,
            op,
            pending.test_field,
            pending.trial_field,
            pending.dispatch,
            pending.kernel,
            (!plan->monolithic_cell_enabled && !use_mixed_block_cell_kernel) ||
                !pending.cell_semantics_owned_by_monolithic);
    }

    if (plan->monolithic_cell_enabled) {
        auto jit_compiler = forms::jit::JITCompiler::getOrCreate(options.compiler_options.jit);
        auto monolithic_kernel = std::make_shared<forms::MonolithicCellKernel>(
            std::move(monolithic_cell_blocks), std::move(jit_compiler), options.compiler_options.jit);
        system.addCellKernel(op, test_fields[0], trial_fields[0], monolithic_kernel);
    } else if (use_mixed_block_cell_kernel) {
        std::shared_ptr<forms::jit::JITCompiler> jit_compiler{};
        if (options.compiler_options.jit.enable) {
            jit_compiler = forms::jit::JITCompiler::getOrCreate(options.compiler_options.jit);
        }
        auto mixed_block_kernel = std::make_shared<forms::MixedBlockKernelSet>(
            std::move(mixed_block_cell_specs), std::move(jit_compiler), options.compiler_options.jit);
        system.addCellKernel(op, test_fields[0], trial_fields[0], mixed_block_kernel);
    }

    out.mixed_plan = std::move(plan);
    return out;
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

    struct PendingKernelInstall {
        FieldId test_field{INVALID_FIELD_ID};
        FieldId trial_field{INVALID_FIELD_ID};
        DomainDispatch dispatch{};
        KernelPtr kernel{};
        bool cell_semantics_owned_by_monolithic{false};
    };

    auto plan = makeMixedKernelPlan(options);
    std::vector<PendingKernelInstall> pending_installs;
    std::vector<forms::MixedBlockKernelSet::BlockSpec> mixed_block_cell_specs;
    std::vector<forms::MonolithicCellKernel::BlockSpec> monolithic_cell_blocks;
    bool monolithic_cell_feasible = plan->monolithic_cell_requested;
    std::string monolithic_disable_reason{};

    for (std::size_t i = 0; i < n_test; ++i) {
        for (std::size_t j = 0; j < n_trial; ++j) {
            if (!mir.hasBlock(i, j)) {
                continue;
            }

            auto block_ir = mir.block(i, j).clone();
            const auto dispatch = analyzeDispatch(block_ir);
            FE_THROW_IF(!dispatchHasAnyTerm(dispatch), InvalidArgumentException,
                        "installMixedFormIR: compiled block has no integral terms");

            const auto kind = block_ir.kind();
            auto block_ir_for_plan = block_ir.clone();
            std::optional<forms::FormIR> tangent_ir_for_plan;
            std::optional<forms::FormIR> residual_ir_for_plan;
            bool want_matrix = false;
            bool want_vector = false;

            KernelPtr kernel{};
            switch (kind) {
                case forms::FormKind::Bilinear:
                    want_matrix = true;
                    tangent_ir_for_plan = std::move(block_ir_for_plan);
                    kernel = std::make_shared<forms::FormKernel>(std::move(block_ir));
                    break;
                case forms::FormKind::Linear:
                    want_vector = true;
                    residual_ir_for_plan = std::move(block_ir_for_plan);
                    kernel = std::make_shared<forms::FormKernel>(std::move(block_ir));
                    break;
                case forms::FormKind::Residual:
                    want_matrix = true;
                    want_vector = true;
                    residual_ir_for_plan = std::move(block_ir_for_plan);
                    if (options.compiler_options.use_symbolic_tangent) {
                        kernel = std::make_shared<forms::SymbolicNonlinearFormKernel>(
                            std::move(block_ir), forms::NonlinearKernelOutput::Both);
                    } else {
                        monolithic_cell_feasible = false;
                        if (monolithic_disable_reason.empty()) {
                            monolithic_disable_reason =
                                "installMixedFormIR: residual MixedFormIR blocks require symbolic tangents "
                                "for MonolithicCellKernel JIT";
                        }
                        kernel = std::make_shared<forms::NonlinearFormKernel>(
                            std::move(block_ir), options.ad_mode);
                    }
                    break;
            }

            kernel = maybeWrapForJIT(std::move(kernel), options);
            result[i][j] = kernel;
            pending_installs.push_back(PendingKernelInstall{
                .test_field = test_fields[i],
                .trial_field = trial_fields[j],
                .dispatch = dispatch,
                .kernel = kernel,
                .cell_semantics_owned_by_monolithic = dispatch.has_cell,
            });

            plan->blocks.push_back(MixedKernelPlanBlock{
                .test_field = test_fields[i],
                .trial_field = trial_fields[j],
                .residual_owner_field = test_fields[i],
                .has_cell = dispatch.has_cell,
                .has_boundary = !dispatch.boundary_markers.empty(),
                .has_interior = dispatch.has_interior,
                .has_interface = dispatch.has_interface,
                .want_matrix = want_matrix,
                .want_vector = want_vector,
            });

            if (dispatch.has_cell && monolithic_cell_feasible) {
                monolithic_cell_blocks.push_back(forms::MonolithicCellKernel::BlockSpec{
                    .test_field = test_fields[i],
                    .trial_field = trial_fields[j],
                    .want_matrix = want_matrix,
                    .want_vector = want_vector,
                    .fallback_kernel = kernel,
                    .tangent_ir = std::move(tangent_ir_for_plan),
                    .residual_ir = std::move(residual_ir_for_plan),
                });
            }
            if (dispatch.has_cell) {
                mixed_block_cell_specs.push_back(forms::MixedBlockKernelSet::BlockSpec{
                    .test_field = test_fields[i],
                    .trial_field = trial_fields[j],
                    .want_matrix = want_matrix,
                    .want_vector = want_vector,
                    .fallback_kernel = kernel,
                });
            }
        }
    }

    plan->semantic_type = MixedKernelSemanticType::MixedBlockSet;
    plan->monolithic_cell_enabled =
        monolithic_cell_feasible && plan->monolithic_cell_requested && monolithic_cell_blocks.size() >= 2u;
    if (plan->monolithic_cell_enabled) {
        plan->semantic_type = MixedKernelSemanticType::MonolithicCell;
    }
    const bool use_mixed_block_cell_kernel =
        !plan->monolithic_cell_enabled && mixed_block_cell_specs.size() >= 2u;

    traceMixedKernelPlan("installMixedFormIR", *plan);
    if (!monolithic_disable_reason.empty() &&
        core::kernelTraceEnabled(core::KernelTraceChannel::Selection)) {
        core::kernelTraceLog(core::KernelTraceChannel::Selection, monolithic_disable_reason);
    }

    for (const auto& pending : pending_installs) {
        registerKernelDomains(
            system,
            op,
            pending.test_field,
            pending.trial_field,
            pending.dispatch,
            pending.kernel,
            (!plan->monolithic_cell_enabled && !use_mixed_block_cell_kernel) ||
                !pending.cell_semantics_owned_by_monolithic);
    }

    if (plan->monolithic_cell_enabled) {
        auto jit_compiler = forms::jit::JITCompiler::getOrCreate(options.compiler_options.jit);
        auto monolithic_kernel = std::make_shared<forms::MonolithicCellKernel>(
            std::move(monolithic_cell_blocks), std::move(jit_compiler), options.compiler_options.jit);
        system.addCellKernel(op, test_fields[0], trial_fields[0], monolithic_kernel);
    } else if (use_mixed_block_cell_kernel) {
        std::shared_ptr<forms::jit::JITCompiler> jit_compiler{};
        if (options.compiler_options.jit.enable) {
            jit_compiler = forms::jit::JITCompiler::getOrCreate(options.compiler_options.jit);
        }
        auto mixed_block_kernel = std::make_shared<forms::MixedBlockKernelSet>(
            std::move(mixed_block_cell_specs), std::move(jit_compiler), options.compiler_options.jit);
        system.addCellKernel(op, test_fields[0], trial_fields[0], mixed_block_kernel);
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
        std::optional<FieldId> bound_field;
    };
    std::vector<SpaceInfo> found;

    const auto visit = [&](const auto& self, const forms::FormExprNode& n) -> void {
        if (n.type() == forms::FormExprType::TestFunction) {
            const auto* sig = n.spaceSignature();
            if (sig) {
                const std::string nm = n.toString();
                auto fid = n.fieldId();
                bool exists = false;
                for (const auto& info : found) {
                    if (fid.has_value() && info.bound_field.has_value()) {
                        if (*fid == *info.bound_field) {
                            FE_THROW_IF(!signaturesMatch(info.sig, *sig) || info.name != nm,
                                InvalidArgumentException,
                                "installFormulation: conflicting test functions bound "
                                "to FieldId " + std::to_string(*fid) +
                                " — same field must use consistent space and name");
                            exists = true;
                            break;
                        }
                    } else if (!fid.has_value() && !info.bound_field.has_value()) {
                        if (info.name == nm && signaturesMatch(info.sig, *sig)) {
                            exists = true; break;
                        }
                    }
                }
                if (!exists) {
                    found.push_back({*sig, nm, fid});
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

/**
 * @brief Split a multi-test-function residual into per-test-function sub-expressions.
 *
 * Returns one FormExpr per test function, in the same order as the test functions
 * appear in the expression. Returns empty if the expression has only one test
 * function (no splitting needed) or if splitting fails.
 *
 * This is a lightweight version of the decomposition in installCoupledResidualMixed
 * that only does the symbolic tree splitting without any compilation.
 */
std::vector<forms::FormExpr> splitByTestFunction(const forms::FormExpr& mixed_residual)
{
    if (!mixed_residual.isValid() || !mixed_residual.node()) return {};

    // Find unique test function rows (field-binding-aware).
    struct TestInfo {
        forms::FormExprNode::SpaceSignature sig;
        std::string name;
        std::optional<FieldId> bound_field;
    };
    std::vector<TestInfo> test_infos;
    bool conflict = false;
    {
        const auto find_tests = [&](const auto& self, const forms::FormExprNode& n) -> void {
            if (conflict) return;
            if (n.type() == forms::FormExprType::TestFunction) {
                const auto* sig = n.spaceSignature();
                if (sig) {
                    std::string nm = n.toString();
                    auto fid = n.fieldId();
                    bool found = false;
                    for (const auto& info : test_infos) {
                        if (fid.has_value() && info.bound_field.has_value()) {
                            if (*fid == *info.bound_field) {
                                if (!signaturesMatch(info.sig, *sig) || info.name != nm)
                                    conflict = true;
                                found = true; break;
                            }
                        } else if (!fid.has_value() && !info.bound_field.has_value()) {
                            if (signaturesMatch(info.sig, *sig) && info.name == nm) {
                                found = true; break;
                            }
                        }
                    }
                    if (!found) {
                        test_infos.push_back({*sig, nm, fid});
                    }
                }
            }
            for (const auto& child : n.childrenShared()) {
                if (child) self(self, *child);
            }
        };
        find_tests(find_tests, *mixed_residual.node());
    }

    if (conflict) return {};  // same-FieldId with inconsistent space or name
    if (test_infos.size() <= 1) return {};

    // Reject duplicate unbound test function names across different spaces.
    for (std::size_t i = 0; i < test_infos.size(); ++i) {
        if (test_infos[i].bound_field.has_value()) continue;
        for (std::size_t j = i + 1; j < test_infos.size(); ++j) {
            if (test_infos[j].bound_field.has_value()) continue;
            if (test_infos[i].name == test_infos[j].name &&
                !signaturesMatch(test_infos[i].sig, test_infos[j].sig)) {
                return {};
            }
        }
    }

    // Check if a subtree contains a test function matching a given TestInfo.
    const auto containsTestMatching = [](const auto& self, const forms::FormExprNode& n,
                                         const TestInfo& info) -> bool {
        if (n.type() == forms::FormExprType::TestFunction) {
            auto node_fid = n.fieldId();
            if (info.bound_field.has_value()) {
                if (node_fid.has_value() && *node_fid == *info.bound_field) return true;
            } else {
                if (!node_fid.has_value() && n.toString() == info.name) return true;
            }
        }
        for (const auto& child : n.childrenShared()) {
            if (child && self(self, *child, info)) return true;
        }
        return false;
    };

    const auto findTestIndices = [&](const forms::FormExprNode& node) -> std::vector<std::size_t> {
        std::vector<std::size_t> indices;
        for (std::size_t ti = 0; ti < test_infos.size(); ++ti) {
            if (containsTestMatching(containsTestMatching, node, test_infos[ti])) {
                indices.push_back(ti);
            }
        }
        return indices;
    };

    std::vector<forms::FormExpr> blocks(test_infos.size());

    const auto assignToBlock = [&](std::size_t ti, forms::FormExpr expr, int sign) {
        if (sign < 0) expr = forms::FormExpr::constant(-1.0) * expr;
        if (!blocks[ti].isValid()) {
            blocks[ti] = std::move(expr);
        } else {
            blocks[ti] = blocks[ti] + std::move(expr);
        }
    };

    const auto decompose = [&](const auto& self, const forms::FormExpr& expr, int sign) -> bool {
        if (!expr.isValid()) return true;
        const auto& n = *expr.node();
        const auto kids = n.childrenShared();

        if ((n.type() == forms::FormExprType::Add || n.type() == forms::FormExprType::Subtract)
            && kids.size() == 2 && kids[0] && kids[1]) {
            const auto left_tests = findTestIndices(*kids[0]);
            const auto right_tests = findTestIndices(*kids[1]);
            if (left_tests.size() == 1 && right_tests.size() == 1 && left_tests[0] == right_tests[0]) {
                assignToBlock(left_tests[0], expr, sign);
                return true;
            }
            const int right_sign = (n.type() == forms::FormExprType::Subtract) ? -sign : sign;
            return self(self, forms::FormExpr(kids[0]), sign)
                && self(self, forms::FormExpr(kids[1]), right_sign);
        }
        if (n.type() == forms::FormExprType::Negate && kids.size() == 1 && kids[0]) {
            return self(self, forms::FormExpr(kids[0]), -sign);
        }

        const auto tests = findTestIndices(n);
        if (tests.size() != 1) return false; // Can't split cleanly
        assignToBlock(tests[0], expr, sign);
        return true;
    };

    if (!decompose(decompose, mixed_residual, +1)) return {};

    return blocks;
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

    // Early validation: reject duplicate test function names across different
    // spaces before creating any side effects (FormulationRecord, contributions).
    if (fields.size() > 1 && residual.node()) {
        struct TFInfo {
            forms::FormExprNode::SpaceSignature sig;
            std::string name;
        };
        std::vector<TFInfo> tf_infos;
        const auto collect_tf = [&](const auto& self, const forms::FormExprNode& n) -> void {
            if (n.type() == forms::FormExprType::TestFunction) {
                const auto* sig = n.spaceSignature();
                if (sig) {
                    const std::string nm = n.toString();
                    bool exists = false;
                    for (const auto& info : tf_infos) {
                        if (signaturesMatch(info.sig, *sig) && info.name == nm) {
                            exists = true;
                            break;
                        }
                    }
                    if (!exists) tf_infos.push_back({*sig, nm});
                }
            }
            for (const auto& child : n.childrenShared()) {
                if (child) self(self, *child);
            }
        };
        collect_tf(collect_tf, *residual.node());

        for (std::size_t i = 0; i < tf_infos.size(); ++i) {
            for (std::size_t j = i + 1; j < tf_infos.size(); ++j) {
                FE_THROW_IF(
                    tf_infos[i].name == tf_infos[j].name &&
                    !signaturesMatch(tf_infos[i].sig, tf_infos[j].sig),
                    InvalidArgumentException,
                    "installFormulation: duplicate TestFunction name '" +
                    tf_infos[i].name + "' used with different spaces — use distinct names");
            }
        }
    }

    // Mutable copies for auxiliary symbol resolution.
    // `resolved` is compiled for assembly. `metadata_resolved` preserves the
    // original AuxiliaryOutputRef view so the monolithic direct-coupling path
    // can still extract dR/d(output) even when the assembled kernel was
    // lowered to direct algebraic input expressions.
    forms::FormExpr resolved = residual;
    forms::FormExpr metadata_resolved = residual;

    // Build the FormulationRecord but do NOT commit it yet. It will be added
    // to the system after installation succeeds, so failures don't leave
    // partial analysis state behind.
    analysis::FormulationRecord rec;
    std::vector<analysis::ContributionDescriptor> pending_contributions;
    {
        rec.operator_tag = op;
        rec.active_fields.assign(fields.begin(), fields.end());
        rec.residual_expr = residual.nodeShared();
        rec.is_mixed = (fields.size() > 1);

        // Scan the DAG for structural properties.
        auto scan = analysis::scanFormExpr(*residual.node());
        rec.has_time_derivative = scan.has_time_derivative;
        rec.has_stabilization_terms = scan.has_stabilization();
        rec.has_interior_face_terms = scan.has_interior_face_terms();
        rec.active_domains = scan.activeDomains();

        // Extract field names from the expression for mixed-form diagnostics.
        // Map FieldId → field name using FESystem's field registry, and collect
        // test/trial function names from the expression tree.
        for (auto fid : fields) {
            const auto& frec = system.fieldRecord(fid);
            rec.field_names.emplace_back(fid, frec.name);
        }

        // Collect test/trial function names from the expression DAG.
        {
            const auto collect_names = [&](const auto& self, const forms::FormExprNode& n) -> void {
                if (n.type() == forms::FormExprType::TestFunction) {
                    const std::string nm = n.toString();
                    if (std::find(rec.test_function_names.begin(), rec.test_function_names.end(), nm)
                        == rec.test_function_names.end()) {
                        rec.test_function_names.push_back(nm);
                    }
                } else if (n.type() == forms::FormExprType::TrialFunction ||
                           n.type() == forms::FormExprType::StateField) {
                    const std::string nm = n.toString();
                    if (std::find(rec.trial_function_names.begin(), rec.trial_function_names.end(), nm)
                        == rec.trial_function_names.end()) {
                        rec.trial_function_names.push_back(nm);
                    }
                }
                for (const auto& child : n.childrenShared()) {
                    if (child) self(self, *child);
                }
            };
            collect_names(collect_names, *residual.node());
        }

        // Build active_variables from FE fields + coupled symbols.
        for (auto fid : fields) {
            rec.active_variables.push_back(analysis::VariableKey::field(fid));
        }
        for (const auto& name : scan.boundary_functional_names) {
            auto vk = analysis::VariableKey::named(
                analysis::VariableKind::BoundaryFunctional, name);
            rec.active_variables.push_back(vk);
            rec.boundary_functional_dependencies.push_back(vk);
        }
        for (const auto& name : scan.auxiliary_state_names) {
            auto vk = analysis::VariableKey::named(
                analysis::VariableKind::AuxiliaryState, name);
            rec.active_variables.push_back(vk);
            rec.auxiliary_state_dependencies.push_back(vk);
        }

        // Generalized auxiliary input dependencies.
        for (const auto& name : scan.auxiliary_input_names) {
            auto vk = analysis::VariableKey::named(
                analysis::VariableKind::AuxiliaryInput, name);
            rec.active_variables.push_back(vk);
            rec.auxiliary_input_dependencies.push_back(vk);
        }

        // Auxiliary output dependencies.
        for (const auto& name : scan.auxiliary_output_names) {
            auto vk = analysis::VariableKey::named(
                analysis::VariableKind::AuxiliaryOutput, name);
            rec.active_variables.push_back(vk);
            rec.auxiliary_output_dependencies.push_back(vk);
        }

        // Auto-resolve AuxiliaryInputSymbol and AuxiliaryOutputSymbol.
        if (!scan.auxiliary_input_names.empty() || !scan.auxiliary_output_names.empty()) {
            // Build input name→slot map from the registry.
            std::unordered_map<std::string, std::size_t> input_slots;
            if (auto* reg = system.auxiliaryInputRegistryIfPresent()) {
                for (const auto& name : scan.auxiliary_input_names) {
                    if (reg->hasInput(name)) {
                        input_slots[name] = reg->slotOf(name);
                    }
                }
            }

            // Build output name→stable-id map from deployed models.
            std::unordered_map<std::string, std::size_t> output_ids;
            for (const auto& name : scan.auxiliary_output_names) {
                std::size_t output_id;
                auto slash = name.find('/');
                if (slash != std::string::npos) {
                    auto inst = name.substr(0, slash);
                    auto oname = name.substr(slash + 1);
                    output_id = system.auxiliaryOutputIdOf(inst, oname);
                } else {
                    output_id = system.auxiliaryOutputIdOf(name);
                }
                if (output_id != static_cast<std::size_t>(-1)) {
                    output_ids[name] = output_id;
                }
            }

            const auto default_field =
                fields.empty() ? INVALID_FIELD_ID : fields.front();
            for (const auto& [qualified_name, output_id] : output_ids) {
                for (const auto domain : rec.active_domains) {
                    analysis::AuxiliaryOutputConsumerRecord consumer;
                    consumer.output_id = static_cast<std::uint32_t>(output_id);
                    consumer.qualified_output_name = qualified_name;
                    consumer.operator_tag = op;
                    consumer.domain_kind = domain;
                    consumer.reference_field = default_field;
                    consumer.test_field = default_field;
                    consumer.trial_field = default_field;
                    rec.auxiliary_output_consumers.push_back(std::move(consumer));
                }
            }

            auto resolve_aux_metadata = [&](const forms::FormExprNode& node)
                -> std::optional<forms::FormExpr> {
                auto sym = node.symbolName();
                if (!sym) return std::nullopt;
                const std::string nm{*sym};
                if (node.type() == forms::FormExprType::AuxiliaryInputSymbol) {
                    auto it = input_slots.find(nm);
                    if (it != input_slots.end())
                        return forms::FormExpr::auxiliaryInputRef(
                            static_cast<std::uint32_t>(it->second));
                }
                if (node.type() == forms::FormExprType::AuxiliaryOutputSymbol) {
                    auto it = output_ids.find(nm);
                    if (it != output_ids.end()) {
                        return forms::FormExpr::auxiliaryOutputRef(
                            static_cast<std::uint32_t>(it->second));
                    }
                    if (auto lowered_output = system.loweredAuxiliaryOutputExpr(nm)) {
                        return *lowered_output;
                    }
                }
                return std::nullopt;
            };
            auto resolve_aux_assembly = [&](const forms::FormExprNode& node)
                -> std::optional<forms::FormExpr> {
                auto sym = node.symbolName();
                if (!sym) return std::nullopt;
                const std::string nm{*sym};
                if (node.type() == forms::FormExprType::AuxiliaryInputSymbol) {
                    auto it = input_slots.find(nm);
                    if (it != input_slots.end())
                        return forms::FormExpr::auxiliaryInputRef(
                            static_cast<std::uint32_t>(it->second));
                }
                if (node.type() == forms::FormExprType::AuxiliaryOutputSymbol) {
                    if (auto lowered_output = system.loweredAuxiliaryOutputExpr(nm)) {
                        return *lowered_output;
                    }
                    auto it = output_ids.find(nm);
                    if (it != output_ids.end())
                        return forms::FormExpr::auxiliaryOutputRef(
                            static_cast<std::uint32_t>(it->second));
                }
                return std::nullopt;
            };
            metadata_resolved = metadata_resolved.transformNodes(resolve_aux_metadata);
            resolved = resolved.transformNodes(resolve_aux_assembly);

            // Preserve AuxiliaryOutputRef in metadata_resolved whenever an
            // output id exists. The assembly expression may still be lowered
            // to a direct form, but the stored metadata expression needs the
            // stable output reference so monolithic dR/d(output) coupling can
            // be recovered exactly during operator assembly.
            rec.residual_expr = metadata_resolved.nodeShared();
        }

        // Block couplings: discover actual active blocks from the per-test
        // decomposition. For each test block, check which StateField/TrialFunction
        // symbols appear to determine the actual (test, trial) couplings.
        // This avoids false dense NxN coupling for zero blocks.
        if (fields.size() > 1) {
            auto test_blocks = splitByTestFunction(resolved);
            if (test_blocks.size() == fields.size()) {
                for (std::size_t ti = 0; ti < fields.size(); ++ti) {
                    if (!test_blocks[ti].isValid()) continue;
                    // Check which fields this test block references
                    const auto state_fields = gatherStateFields(*test_blocks[ti].node());
                    for (std::size_t tj = 0; tj < fields.size(); ++tj) {
                        if (state_fields.contains(fields[tj])) {
                            rec.block_couplings.emplace_back(fields[ti], fields[tj]);
                        }
                    }
                    // Pure source rows (no StateField dependencies) produce no
                    // block_couplings entries. This is correct: block_couplings
                    // documents Jacobian block structure, and a pure forcing
                    // term has zero Jacobian contribution.
                }
            } else {
                // splitByTestFunction could not decompose the expression.
                // Fall back to conservative dense coupling.
                for (auto test_f : fields) {
                    for (auto trial_f : fields) {
                        rec.block_couplings.emplace_back(test_f, trial_f);
                    }
                }
            }
            // Note: if splitting succeeds but all rows are pure-source (no
            // StateField dependencies), block_couplings stays empty. This is
            // correct: it means the Jacobian has no block structure.
        } else if (fields.size() == 1) {
            // Only record a self-coupling if the resolved has trial dependency
            // (TrialFunction or StateField). Source-only residuals (f*v) have
            // no Jacobian block.
            const bool has_trial_dep = resolved.hasTrial() ||
                (resolved.node() && hasStateFieldNodes(*resolved.node()));
            if (has_trial_dep) {
                rec.block_couplings.emplace_back(fields[0], fields[0]);
            }
        }

        // Variable couplings: FE fields coupled to each other, plus FE↔boundary/aux links.
        for (const auto& bf : rec.boundary_functional_dependencies) {
            for (auto fid : fields) {
                rec.variable_couplings.emplace_back(
                    analysis::VariableKey::field(fid), bf);
            }
        }
        for (const auto& aux : rec.auxiliary_state_dependencies) {
            for (auto fid : fields) {
                rec.variable_couplings.emplace_back(
                    analysis::VariableKey::field(fid), aux);
            }
        }

        // Determine affine_split_succeeded by attempting the split on the resolved.
        // This is a lightweight structural walk of the DAG — no compilation involved.
        {
            forms::AffineResidualOptions affine_opts;
            affine_opts.allow_time_derivatives = false;
            affine_opts.allow_interior_face_terms = false;
            auto split = forms::trySplitAffineResidual(resolved, affine_opts);
            rec.affine_split_succeeded = split.has_value();
        }

        // Per-block resolved handles.
        if (fields.size() == 1) {
            // Single-field: the whole resolved is the single block.
            rec.block_residual_exprs.push_back(
                {{fields[0], fields[0]}, metadata_resolved.nodeShared()});
        } else {
            // Multi-field: split by test function to get per-test sub-expressions.
            // Each sub-expression F_i(u1,...,uN; v_i) contains terms for one test field.
            // We store them keyed by (test_field, test_field) — the trial splitting
            // is done by the form compiler during Jacobian block compilation.
            auto test_blocks = splitByTestFunction(metadata_resolved);
            if (test_blocks.size() == fields.size()) {
                for (std::size_t i = 0; i < fields.size(); ++i) {
                    if (test_blocks[i].isValid()) {
                        rec.block_residual_exprs.push_back(
                            {{fields[i], fields[i]}, test_blocks[i].nodeShared()});
                    }
                }
            }
        }

        // Lower the formulation into normalized ContributionDescriptors.
        // Store them pending — they will be committed after installation succeeds.
        pending_contributions = analysis::lowerFormulation(rec);
    }

    const auto num_test = countUniqueTestSpaces(*resolved.node());
    FE_THROW_IF(num_test == 0, InvalidArgumentException,
                "installFormulation: resolved contains no TestFunction");

    // Analysis metadata is committed only after installation succeeds.
    const auto commitRecord = [&]() {
        for (auto& c : pending_contributions) {
            system.addContribution(std::move(c));
        }
        system.addFormulationRecord(std::move(rec));
    };

    // Transactional installation: snapshot operator state, rollback on failure.
    return system.executeWithOperatorRollback_([&]() -> CoupledResidualKernels {

    if (num_test == 1) {
        // Single-field path.
        FE_THROW_IF(fields.size() != 1, InvalidArgumentException,
                    "installFormulation: expression has 1 TestFunction space but " +
                    std::to_string(fields.size()) + " fields provided");

        // If the expression uses StateField nodes, lower them to TrialFunction.
        forms::FormExpr lowered = resolved;
        const bool has_state = resolved.node() && hasStateFieldNodes(*resolved.node());
        if (has_state) {
            lowered = lowerStateFields(resolved, fields[0], system);
        }

        KernelPtr kernel;
        const bool is_source_only = !lowered.hasTrial();
        if (!is_source_only) {
            // Normal resolved with trial dependency → full resolved install
            kernel = installResidualForm(system, op, fields[0], fields[0], lowered, options);
        } else {
            // Source-only resolved (e.g., f*v) → install as linear vector-only kernel
            forms::FormCompiler compiler(options.compiler_options);
            auto ir = compiler.compileLinear(lowered);
            const auto dispatch = analyzeDispatch(ir);
            kernel = std::make_shared<forms::FormKernel>(std::move(ir));
            kernel = maybeWrapForJIT(std::move(kernel), options);
            registerKernel(system, op, fields[0], fields[0], dispatch, kernel);
        }

        commitRecord();

        CoupledResidualKernels out;
        out.residual = {kernel};
        if (!is_source_only) {
            out.jacobian_blocks = {{kernel}};
        } else {
            out.jacobian_blocks = {{nullptr}};
        }
        return out;
    }

    // Multi-field path: build an explicit mixed-kernel plan with either a
    // MixedBlockKernelSet or MonolithicCellKernel for cell-domain work.
    auto result = installCoupledResidualMixed(system, op, fields, fields, resolved, options);

    commitRecord();

    return result;

    }); // executeWithOperatorRollback_
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

// ============================================================================
// installMixedBilinear
// ============================================================================

std::vector<std::vector<KernelPtr>>
installMixedBilinear(
    FESystem& system,
    const OperatorTag& op,
    std::span<const FieldId> test_fields,
    std::span<const FieldId> trial_fields,
    const forms::FormExpr& bilinear,
    const FormInstallOptions& options)
{
    FE_THROW_IF(test_fields.empty(), InvalidArgumentException,
                "installMixedBilinear: empty test field list");
    FE_THROW_IF(trial_fields.empty(), InvalidArgumentException,
                "installMixedBilinear: empty trial field list");
    FE_THROW_IF(!bilinear.isValid(), InvalidArgumentException,
                "installMixedBilinear: invalid form expression");

    forms::FormCompiler compiler(options.compiler_options);
    auto mir = compiler.compileMixed(bilinear, forms::FormKind::Bilinear);

    FE_THROW_IF(test_fields.size() != mir.numTestFields(), InvalidArgumentException,
                "installMixedBilinear: expression has " + std::to_string(mir.numTestFields()) +
                " test spaces but " + std::to_string(test_fields.size()) + " test fields provided");
    FE_THROW_IF(trial_fields.size() != mir.numTrialFields(), InvalidArgumentException,
                "installMixedBilinear: expression has " + std::to_string(mir.numTrialFields()) +
                " trial spaces but " + std::to_string(trial_fields.size()) + " trial fields provided");

    return system.executeWithOperatorRollback_([&]() {
        return installMixedFormIR(system, op, test_fields, trial_fields, mir, options);
    });
}

// ============================================================================
// installMixedLinear
// ============================================================================

std::vector<KernelPtr>
installMixedLinear(
    FESystem& system,
    const OperatorTag& op,
    std::span<const FieldId> test_fields,
    const forms::FormExpr& linear,
    const FormInstallOptions& options)
{
    FE_THROW_IF(test_fields.empty(), InvalidArgumentException,
                "installMixedLinear: empty test field list");
    FE_THROW_IF(!linear.isValid(), InvalidArgumentException,
                "installMixedLinear: invalid form expression");

    forms::FormCompiler compiler(options.compiler_options);
    auto mir = compiler.compileMixed(linear, forms::FormKind::Linear);

    FE_THROW_IF(test_fields.size() != mir.numTestFields(), InvalidArgumentException,
                "installMixedLinear: expression has " + std::to_string(mir.numTestFields()) +
                " test spaces but " + std::to_string(test_fields.size()) + " test fields provided");

    // compileMixed() produces a 1-column layout for linear forms (synthetic
    // trial column). Map it to placeholder trial FieldIds for installation.
    const auto n_trial = mir.numTrialFields();
    std::vector<FieldId> trial_fields_vec;
    if (n_trial <= test_fields.size()) {
        trial_fields_vec.assign(test_fields.begin(), test_fields.begin() + static_cast<std::ptrdiff_t>(n_trial));
    } else {
        trial_fields_vec.resize(n_trial, test_fields[0]);
    }

    return system.executeWithOperatorRollback_([&]() {
        auto block_kernels = installMixedFormIR(system, op, test_fields,
                                                 std::span<const FieldId>(trial_fields_vec), mir, options);

        std::vector<KernelPtr> result(test_fields.size());
        for (std::size_t i = 0; i < test_fields.size() && i < block_kernels.size(); ++i) {
            if (!block_kernels[i].empty()) {
                result[i] = block_kernels[i][0];
            }
        }
        return result;
    });
}

} // namespace systems
} // namespace FE
} // namespace svmp
