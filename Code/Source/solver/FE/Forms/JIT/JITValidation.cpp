/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#include "Forms/JIT/JITValidation.h"

#include "Core/FEException.h"
#include "Forms/Tensor/TensorContraction.h"

#include <stdexcept>
#include <string>
#include <utility>

namespace svmp {
namespace FE {
namespace forms {
namespace jit {

namespace {

[[nodiscard]] bool containsExternalCall(const FormExprNode& node)
{
    bool found = false;
    const auto visit = [&](const auto& self, const FormExprNode& n) -> void {
        if (n.type() == FormExprType::Coefficient || n.type() == FormExprType::Constitutive) {
            found = true;
            return;
        }
        for (const auto& child : n.childrenShared()) {
            if (child && !found) self(self, *child);
        }
    };
    visit(visit, node);
    return found;
}

[[nodiscard]] ValidationIssue issue(const FormExprNode& node, std::string message)
{
    ValidationIssue out;
    out.type = node.type();
    out.message = std::move(message);
    out.subexpr = node.toString();
    return out;
}

[[nodiscard]] std::string formatIndexList(const std::vector<forms::tensor::ContractionAnalysis::IndexInfo>& indices)
{
    std::string out;
    for (std::size_t k = 0; k < indices.size(); ++k) {
        if (k > 0) out += ", ";
        out += indices[k].name.empty() ? ("i" + std::to_string(indices[k].id)) : indices[k].name;
    }
    return out;
}

} // namespace

ValidationResult canCompile(const FormExpr& integrand, const ValidationOptions& options)
{
    ValidationResult out;
    out.ok = true;
    out.cacheable = true;

    if (!integrand.isValid() || integrand.node() == nullptr) {
        out.ok = false;
        out.first_issue = ValidationIssue{
            .type = FormExprType::Constant,
            .message = "forms::jit::canCompile: invalid expression",
            .subexpr = {},
        };
        return out;
    }

    const auto& root = *integrand.node();
    bool saw_indexed_access = false;

    const auto visit = [&](const auto& self, const FormExprNode& n) -> void {
        if (!out.ok) return;

        // Name-based terminals must be resolved to slot refs before JIT lowering.
        switch (n.type()) {
            case FormExprType::ParameterSymbol:
                out.ok = false;
                out.first_issue = issue(n, "JIT: ParameterSymbol must be resolved to ParameterRef(slot)");
                return;
            case FormExprType::BoundaryFunctionalSymbol:
            case FormExprType::BoundaryIntegralSymbol:
            case FormExprType::AuxiliaryStateSymbol:
                out.ok = false;
                out.first_issue = issue(n, "JIT: coupled placeholder must be resolved to slot-based refs");
                return;

            // Measure wrappers are handled by FormCompiler -> FormIR decomposition, not by kernel lowering.
            case FormExprType::CellIntegral:
            case FormExprType::BoundaryIntegral:
            case FormExprType::InteriorFaceIntegral:
            case FormExprType::InterfaceIntegral:
                out.ok = false;
                out.first_issue = issue(n, "JIT: measure nodes (dx/ds/dS/dI) are not valid in integrands");
                return;

            case FormExprType::IndexedAccess:
                saw_indexed_access = true;
                break;

            default:
                break;
        }

        if (n.type() == FormExprType::Coefficient) {
            if (options.strictness == Strictness::Strict) {
                out.ok = false;
                out.first_issue = issue(n, "JIT(strict): Coefficient nodes are not allowed (runtime callback)");
                return;
            }
            out.cacheable = false;
        }

        if (n.type() == FormExprType::Constitutive) {
            if (options.strictness == Strictness::Strict) {
                out.ok = false;
                out.first_issue = issue(n, "JIT(strict): Constitutive calls are not allowed (virtual dispatch)");
                return;
            }
            out.cacheable = false;
        }

        // Spatial derivatives of external-call coefficients require explicit derivatives.
        if (n.type() == FormExprType::Gradient ||
            n.type() == FormExprType::Divergence ||
            n.type() == FormExprType::Curl ||
            n.type() == FormExprType::Hessian) {
            const auto kids = n.childrenShared();
            if (!kids.empty() && kids[0] && containsExternalCall(*kids[0])) {
                out.ok = false;
                out.first_issue = issue(n, "JIT: derivative of external-call coefficient/model is not supported; provide explicit derivative expressions");
                return;
            }
        }

        for (const auto& child : n.childrenShared()) {
            if (child) self(self, *child);
        }
    };

    visit(visit, root);

    if (out.ok && saw_indexed_access) {
        const auto a = forms::tensor::analyzeContractions(integrand);
        if (!a.ok) {
            out.ok = false;
            out.first_issue = ValidationIssue{
                .type = FormExprType::IndexedAccess,
                .message = "JIT: invalid Einstein-index usage: " + a.message,
                .subexpr = integrand.toString(),
            };
            return out;
        }
        if (!a.free_indices.empty()) {
            out.ok = false;
            out.first_issue = ValidationIssue{
                .type = FormExprType::IndexedAccess,
                .message = "JIT: result has free indices {" + formatIndexList(a.free_indices) +
                           "} but expected scalar output (integrand must be fully contracted)",
                .subexpr = integrand.toString(),
            };
            return out;
        }
    }

    return out;
}

ValidationResult canCompile(const FormIR& ir, const ValidationOptions& options)
{
    ValidationResult out;
    out.ok = true;
    out.cacheable = true;

    if (!ir.isCompiled()) {
        out.ok = false;
        out.first_issue = ValidationIssue{
            .type = FormExprType::Constant,
            .message = "forms::jit::canCompile(FormIR): FormIR is not compiled",
            .subexpr = {},
        };
        return out;
    }

    for (const auto& term : ir.terms()) {
        if (!term.integrand.isValid() || term.integrand.node() == nullptr) {
            out.ok = false;
            out.first_issue = ValidationIssue{
                .type = FormExprType::Constant,
                .message = "forms::jit::canCompile(FormIR): term integrand is invalid",
                .subexpr = term.debug_string,
            };
            return out;
        }

        auto r = canCompile(term.integrand, options);
        if (!r.ok) {
            out.ok = false;
            out.cacheable = out.cacheable && r.cacheable;
            out.first_issue = std::move(r.first_issue);
            return out;
        }
        out.cacheable = out.cacheable && r.cacheable;
    }

    return out;
}

} // namespace jit
} // namespace forms
} // namespace FE
} // namespace svmp
