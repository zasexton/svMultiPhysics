/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#include "Systems/GlobalRowKernel.h"

#include "Systems/BorderedGlobalViews.h"

#include "Assembly/Assembler.h"
#include "Assembly/GlobalSystemView.h"
#include "Backends/Interfaces/GenericVector.h"
#include "Constraints/AffineConstraints.h"
#include "Core/FEException.h"
#include "Forms/PointEvaluator.h"
#include "Forms/SymbolicDifferentiation.h"
#include "Quadrature/QuadratureFactory.h"
#include "Sparsity/SparsityPattern.h"
#include "Spaces/FunctionSpace.h"
#include "Systems/FESystem.h"
#include "Systems/FunctionalFormEvaluator.h"
#include "Systems/SystemCollectives.h"
#include "Systems/SystemState.h"
#include "Systems/SystemsExceptions.h"

#include <algorithm>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace svmp {
namespace FE {
namespace systems {

namespace {

using forms::FormExpr;
using forms::FormExprNode;
using forms::FormExprType;

/// The expression with every test function replaced by 1 (a global row has one).
[[nodiscard]] FormExpr withUnitTestFunction(const FormExpr& expr)
{
    if (!expr.isValid()) {
        return {};
    }
    return expr.transformNodes([](const FormExprNode& node) -> std::optional<FormExpr> {
        if (node.type() == FormExprType::TestFunction) {
            return FormExpr::constant(Real(1.0));
        }
        return std::nullopt;
    });
}

[[nodiscard]] bool isGlobalSymbol(const FormExprNode& node) noexcept
{
    const auto* signature = node.spaceSignature();
    return signature != nullptr && signature->space_type == spaces::SpaceType::Global;
}

void collectGlobalStateFields(const FormExprNode& node, std::vector<FieldId>& out)
{
    if ((node.type() == FormExprType::StateField || node.type() == FormExprType::DiscreteField) &&
        node.fieldId().has_value()) {
        const auto field = *node.fieldId();
        if (std::find(out.begin(), out.end(), field) == out.end()) {
            out.push_back(field);
        }
    }
    for (const auto& child : node.childrenShared()) {
        if (child) {
            collectGlobalStateFields(*child, out);
        }
    }
}

[[nodiscard]] CollectiveCommunicator systemCommunicator(const FESystem& system)
{
#if FE_HAS_MPI
    return CollectiveCommunicator(system.activeMpiCommunicator());
#else
    (void)system;
    return CollectiveCommunicator{};
#endif
}

/// The value of the monolithic DOF @p dof in @p state (a bordered global DOF from the system).
[[nodiscard]] Real stateValue(const FESystem& system, const SystemStateView& state, GlobalIndex dof)
{
    if (auto view = makeStateReadView(system, state.u_vector, state.u, /*ghosted=*/true)) {
        return view->getVectorEntry(dof);
    }
    FE_THROW_IF(dof < 0 || static_cast<std::size_t>(dof) >= state.u.size(), InvalidArgumentException,
                "GlobalRowKernel: the state has no value for DOF " + std::to_string(dof));
    return state.u[static_cast<std::size_t>(dof)];
}

/**
 * The volume unknown whose element has the largest quadrature order (the
 * first one on a tie).  The global space binds the same order on the cells,
 * so the residual entry and the row's element kernels use one rule.
 */
[[nodiscard]] FieldId quadratureField(const FESystem& system)
{
    FieldId best = INVALID_FIELD_ID;
    int best_order = -1;
    for (const auto field : system.unknownFieldIdsInDofMapOrder()) {
        const auto& rec = system.fieldRecord(field);
        if (rec.scope != FieldScope::VolumeCell || !rec.space) {
            continue;
        }
        const auto& element = rec.space->element();
        const auto rule = element.quadrature();
        const int order = rule ? rule->order()
                               : quadrature::QuadratureFactory::recommended_order(
                                     element.polynomial_order(), false);
        if (order > best_order) {
            best_order = order;
            best = field;
        }
    }
    FE_THROW_IF(best == INVALID_FIELD_ID, InvalidStateException,
                "GlobalRowKernel: the measured terms of a global row need a volume unknown "
                "for their quadrature");
    return best;
}

/// The point expression (global symbols only) evaluated at @p state.
[[nodiscard]] Real evaluatePoint(const FormExpr& expr,
                                 const FESystem& system,
                                 const SystemStateView& state)
{
    const auto& parameters = system.parameterRegistry();
    std::function<std::optional<Real>(std::string_view)> real_parameter;
    if (!parameters.specs().empty()) {
        real_parameter = parameters.makeRealGetter(state);
    } else if (state.getRealParam) {
        real_parameter = state.getRealParam;
    }
    const auto resolved = expr.transformNodes([&](const FormExprNode& node) -> std::optional<FormExpr> {
        if (node.type() != FormExprType::ParameterSymbol) {
            return std::nullopt;
        }
        const auto name = node.symbolName();
        FE_THROW_IF(!name.has_value(), InvalidArgumentException,
                    "GlobalRowKernel: a parameter of a dGlobal term has no name");
        const auto value = real_parameter ? real_parameter(*name) : std::optional<Real>{};
        FE_THROW_IF(!value.has_value(), InvalidArgumentException,
                    "GlobalRowKernel: parameter '" + std::string(*name) +
                        "' of a dGlobal term has no value");
        return FormExpr::constant(*value);
    });

    std::vector<FieldId> fields;
    collectGlobalStateFields(*resolved.node(), fields);
    std::vector<FieldValueEntry> field_values;
    field_values.reserve(fields.size());
    for (const auto field : fields) {
        FieldValueEntry entry;
        entry.field = field;
        entry.n_components = 1;
        entry.components[0] = stateValue(system, state, system.fieldDofOffset(field));
        field_values.push_back(entry);
    }

    std::vector<Real> slots;
    if (parameters.slotCount() > 0u) {
        slots = parameters.evaluateRealSlots(state);
    }

    forms::PointEvalContext context;
    context.time = static_cast<Real>(state.time);
    context.dt = static_cast<Real>(state.dt);
    context.jit_constants = slots;
    context.field_values = field_values;
    return forms::evaluateScalarAt(resolved, context);
}

} // namespace

GlobalRowKernel::GlobalRowKernel(FieldId test_field,
                                 std::string test_field_name,
                                 forms::FormExpr measured_terms,
                                 forms::FormExpr point_integrand)
    : test_field_(test_field)
    , test_field_name_(std::move(test_field_name))
{
    FE_THROW_IF(test_field_ == INVALID_FIELD_ID, InvalidArgumentException,
                "GlobalRowKernel: invalid test field");
    FE_THROW_IF(!measured_terms.isValid() && !point_integrand.isValid(), InvalidArgumentException,
                "GlobalRowKernel: the row of '" + test_field_name_ + "' has no terms");
    measured_value_form_ = withUnitTestFunction(measured_terms);
    if (point_integrand.isValid()) {
        validatePointIntegrand(point_integrand, test_field_name_);
        point_value_ = withUnitTestFunction(point_integrand);
        std::vector<FieldId> fields;
        collectGlobalStateFields(*point_value_.node(), fields);
        for (const auto field : fields) {
            point_derivatives_.emplace_back(
                field,
                forms::directionalDerivativeWrtField(point_value_, field, FormExpr::constant(Real(1.0))));
        }
    }
}

namespace {

/// Whether StateField(@p field) occurs in @p node outside every derivative.
[[nodiscard]] bool occursUndifferentiated(const FormExprNode& node, FieldId field)
{
    switch (node.type()) {
    case FormExprType::Gradient:
    case FormExprType::Divergence:
    case FormExprType::Curl:
    case FormExprType::Hessian:
    case FormExprType::TimeDerivative:
        return false;
    case FormExprType::StateField:
    case FormExprType::DiscreteField:
        return node.fieldId().has_value() && *node.fieldId() == field;
    default:
        break;
    }
    for (const auto& child : node.childrenShared()) {
        if (child && occursUndifferentiated(*child, field)) {
            return true;
        }
    }
    return false;
}

[[nodiscard]] bool isZeroConstant(const FormExpr& expr)
{
    if (!expr.isValid()) {
        return true;
    }
    const auto value = expr.node()->constantValue();
    return expr.node()->type() == FormExprType::Constant && value.has_value() && *value == Real(0.0);
}

} // namespace

bool GlobalRowKernel::anchorsConstantModeOf(FieldId field) const
{
    if (!measured_value_form_.isValid() || !occursUndifferentiated(*measured_value_form_.node(), field)) {
        return false;
    }
    const auto derivative = forms::simplify(
        forms::directionalDerivativeWrtField(measured_value_form_, field, FormExpr::constant(Real(1.0))));
    return !isZeroConstant(derivative);
}

std::vector<FieldId> GlobalRowKernel::pointTermFields() const
{
    std::vector<FieldId> fields;
    fields.reserve(point_derivatives_.size());
    for (const auto& [field, derivative] : point_derivatives_) {
        static_cast<void>(derivative);
        fields.push_back(field);
    }
    return fields;
}

std::string GlobalRowKernel::name() const
{
    return "GlobalRowKernel(" + test_field_name_ + ")";
}

void GlobalRowKernel::validatePointIntegrand(const forms::FormExpr& integrand,
                                             const std::string& test_field_name)
{
    FE_THROW_IF(!integrand.isValid(), InvalidArgumentException,
                "GlobalRowKernel: empty dGlobal integrand");
    const auto fail = [&](const FormExprNode& node, const std::string& why) {
        FE_THROW(InvalidArgumentException,
                 "a dGlobal() term of the row of '" + test_field_name + "' holds " + why +
                     " ('" + node.toString() + "'); dGlobal integrands hold only global symbols, "
                     "constants, parameters, time and the time step, and scalar algebra");
    };
    const auto visit = [&](const auto& self, const FormExprNode& node) -> void {
        switch (node.type()) {
        case FormExprType::Constant:
        case FormExprType::ParameterSymbol:
        case FormExprType::ParameterRef:
        case FormExprType::Time:
        case FormExprType::TimeStep:
        case FormExprType::Negate:
        case FormExprType::Add:
        case FormExprType::Subtract:
        case FormExprType::Multiply:
        case FormExprType::Divide:
        case FormExprType::Power:
        case FormExprType::Minimum:
        case FormExprType::Maximum:
        case FormExprType::AbsoluteValue:
        case FormExprType::Sign:
        case FormExprType::Sqrt:
        case FormExprType::Exp:
        case FormExprType::Log:
            break;
        case FormExprType::TestFunction:
        case FormExprType::StateField:
        case FormExprType::DiscreteField:
            if (!isGlobalSymbol(node)) {
                fail(node, "a field that is not global");
            }
            break;
        case FormExprType::TrialFunction:
            fail(node, "a trial function (state the unknowns of a residual as StateField)");
            break;
        default:
            fail(node, "a node with a spatial or unsupported meaning");
        }
        for (const auto& child : node.childrenShared()) {
            if (child) {
                self(self, *child);
            }
        }
    };
    visit(visit, *integrand.node());
}

void GlobalRowKernel::addSparsityCouplings(const FESystem& system,
                                           sparsity::SparsityPattern& pattern) const
{
    if (system.hasBorderedGlobalFields()) {
        // The row is a bordered row, outside the matrix.
        return;
    }
    const auto row = system.fieldDofOffset(test_field_);
    for (const auto& [field, derivative] : point_derivatives_) {
        static_cast<void>(derivative);
        pattern.addEntry(row, system.fieldDofOffset(field));
    }
}

assembly::AssemblyResult GlobalRowKernel::assemble(const FESystem& system,
                                                   const AssemblyRequest& request,
                                                   const SystemStateView& state,
                                                   assembly::GlobalSystemView* matrix_out,
                                                   assembly::GlobalSystemView* vector_out)
{
    assembly::AssemblyResult result;
    const auto comm = systemCommunicator(system);
    const auto row = system.fieldDofOffset(test_field_);
    // The lowest rank owns a global field's DOFs (its own handler says so
    // under either lowering).
    const bool owner = system.fieldDofHandler(test_field_).getDofMap().isOwnedDof(0);

    if (vector_out != nullptr) {
        if (!request.suppress_constraint_inhomogeneity) {
            // Joint distribution lifts -K g of inhomogeneous constraints into
            // the vector; this entry is the residual at the state and carries
            // no lifting.
            const bool lifted = system.constraints().getStatistics().n_inhomogeneous > 0;
            FE_THROW_IF(collectiveAnyTrue(lifted, comm), NotImplementedException,
                        "GlobalRowKernel: the residual of the global row of '" + test_field_name_ +
                            "' is the residual at the state, without the lifting of inhomogeneous "
                            "constraints; assemble it with suppress_constraint_inhomogeneity (as "
                            "Newton does) or with homogeneous constraints");
        }
        Real value = Real(0.0);
        if (measured_value_form_.isValid()) {
            const auto values = evaluateFunctionalFormsWithQuadratureOf(
                system,
                quadratureField(system),
                std::span<const FormExpr>(&measured_value_form_, 1u),
                state,
                comm);
            value = values.front().value;
        }
        if (owner) {
            if (point_value_.isValid()) {
                value += evaluatePoint(point_value_, system, state);
            }
            vector_out->addVectorEntry(row, value, assembly::AddMode::Add);
            ++result.vector_entries_inserted;
        }
    }

    if (matrix_out != nullptr && owner) {
        for (const auto& [field, derivative] : point_derivatives_) {
            matrix_out->addMatrixEntry(row,
                                       system.fieldDofOffset(field),
                                       evaluatePoint(derivative, system, state),
                                       assembly::AddMode::Add);
            ++result.matrix_entries_inserted;
        }
    }
    return result;
}

} // namespace systems
} // namespace FE
} // namespace svmp
