/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#include "Forms/BoundaryFunctional.h"

#include "Core/FEException.h"
#include "Forms/FormKernels.h"
#include "Forms/FormIR.h"
#include "Forms/ConstitutiveModel.h"
#include "Forms/JIT/InlinableConstitutiveModel.h"
#include "Forms/JIT/JITFunctionalKernelWrapper.h"

#include <algorithm>
#include <optional>
#include <utility>
#include <vector>

namespace svmp {
namespace FE {
namespace forms {

namespace {

/// The largest topological dimension of the spaces in @p node (0 without one).
[[nodiscard]] int largestTopologicalDimension(const FormExprNode& node)
{
    int dim = 0;
    if (const auto* signature = node.spaceSignature()) {
        dim = signature->topological_dimension;
    }
    for (const auto& child : node.childrenShared()) {
        if (child) {
            dim = std::max(dim, largestTopologicalDimension(*child));
        }
    }
    return dim;
}

/// The stateless inlinable law of a constitutive call node, or null.
[[nodiscard]] const InlinableConstitutiveModel* statelessInlinableLaw(const FormExprNode& node)
{
    if (node.type() != FormExprType::Constitutive || node.constitutiveModel() == nullptr) {
        return nullptr;
    }
    const auto* inlinable = node.constitutiveModel()->inlinable();
    if (inlinable == nullptr || inlinable->stateAccess() != MaterialStateAccess::None) {
        return nullptr;
    }
    return inlinable;
}

/**
 * @p integrand with every call of a law written as forms (a stateless
 * inlinable model, such as constitutive::FormLaw) replaced by its body, as
 * kernel setup expands the calls of a residual.  A functional kernel
 * evaluates the integrand point by point, and such a law has no pointwise
 * evaluation.  Returns @p integrand itself when it holds no such call, or
 * when no space tells the dimension.
 */
[[nodiscard]] FormExpr expandStatelessLaws(const FormExpr& integrand, int dim)
{
    bool expanded = false;
    const auto expandCall = [&](const auto& self, const FormExprNode& call) -> std::vector<FormExpr> {
        const auto* law = statelessInlinableLaw(call);
        std::vector<FormExpr> inputs;
        for (const auto& child : call.childrenShared()) {
            inputs.push_back(self(self, FormExpr(child)));
        }
        InlinableConstitutiveContext context;
        context.dim = dim;
        auto expansion = law->inlineExpand(inputs, context);
        expanded = true;
        std::vector<FormExpr> outputs;
        for (auto& output : expansion.outputs) {
            outputs.push_back(self(self, output));
        }
        return outputs;
    };
    const auto expand = [&](const auto& self, const FormExpr& expr) -> FormExpr {
        if (!expr.isValid()) {
            return expr;
        }
        return expr.transformNodes([&](const FormExprNode& node) -> std::optional<FormExpr> {
            if (statelessInlinableLaw(node) != nullptr) {
                const auto outputs = expandCall(self, node);
                FE_THROW_IF(outputs.empty(), InvalidArgumentException,
                            "compileBoundaryFunctionalKernel: a law call expanded to no output");
                return outputs.front();
            }
            if (node.type() == FormExprType::ConstitutiveOutput) {
                const auto kids = node.childrenShared();
                const auto index = node.constitutiveOutputIndex();
                if (kids.size() == 1u && kids.front() && index.has_value() && *index >= 0 &&
                    statelessInlinableLaw(*kids.front()) != nullptr) {
                    const auto outputs = expandCall(self, *kids.front());
                    FE_THROW_IF(static_cast<std::size_t>(*index) >= outputs.size(), InvalidArgumentException,
                                "compileBoundaryFunctionalKernel: a law output index past its outputs");
                    return outputs[static_cast<std::size_t>(*index)];
                }
            }
            return std::nullopt;
        });
    };
    if (dim == 0) {
        return integrand;
    }
    auto out = expand(expand, integrand);
    return expanded ? out : integrand;
}

} // namespace

namespace detail {

// Reuse the FormCompiler analysis routines for RequiredData / FieldRequirements without
// requiring a full linear/bilinear form (BoundaryFunctionals have no test function).
assembly::RequiredData analyzeRequiredData(const FormExprNode& node, FormKind kind);
std::vector<assembly::FieldRequirement> analyzeFieldRequirements(const FormExprNode& node);

} // namespace detail

BoundaryFunctional boundaryFunctional(const FormExpr& measured_form, std::string name)
{
    const auto* node = measured_form.node();
    FE_THROW_IF(node == nullptr, InvalidArgumentException,
                "forms::boundaryFunctional: invalid measured form");
    FE_THROW_IF(node->hasTest() || node->hasTrial(), InvalidArgumentException,
                "forms::boundaryFunctional: a measured 0-form has no test or trial function");

    const auto* measure = node->exteriorBoundaryMeasure();
    const bool fitted = node->type() == FormExprType::BoundaryIntegral;
    const bool generated = node->type() == FormExprType::InterfaceIntegral && measure != nullptr &&
                           measure->isGeneratedActiveSubset();
    FE_THROW_IF(!fitted && !generated, InvalidArgumentException,
                "forms::boundaryFunctional: expected one exterior-boundary integral, integrand.ds(marker) or "
                "integrand.ds(BoundaryDomain)");

    BoundaryFunctional functional;
    functional.integrand = FormExpr(node->childrenShared().at(0));
    functional.boundary_marker = fitted ? node->boundaryMarker().value_or(-1) : measure->physicalBoundaryMarker();
    if (generated) {
        functional.generated_active_boundary_marker = measure->generatedActiveBoundaryMarker();
    }
    functional.name = std::move(name);
    functional.reduction = BoundaryFunctional::Reduction::Sum;
    FE_THROW_IF(functional.boundary_marker < 0, InvalidArgumentException,
                "forms::boundaryFunctional: ds() over every boundary names no marker; integrate over ds(marker)");
    return functional;
}

std::shared_ptr<assembly::FunctionalKernel>
compileBoundaryFunctionalKernel(const FormExpr& integrand, int boundary_marker)
{
    return compileBoundaryFunctionalKernel(integrand, boundary_marker, SymbolicOptions{});
}

std::shared_ptr<assembly::FunctionalKernel>
compileBoundaryFunctionalKernel(const FormExpr& original_integrand,
                                int boundary_marker,
                                const SymbolicOptions& options)
{
    FE_THROW_IF(!original_integrand.isValid(), InvalidArgumentException,
                "compileBoundaryFunctionalKernel: invalid integrand");
    const auto integrand =
        expandStatelessLaws(original_integrand, largestTopologicalDimension(*original_integrand.node()));

    (void)boundary_marker;

    FE_CHECK_NOT_NULL(integrand.node(), "compileBoundaryFunctionalKernel: integrand node");

    // BoundaryFunctionals must be pure integrands: no coupled placeholders and no measures.
    const auto check = [&](const auto& self, const FormExprNode& n) -> void {
        FE_THROW_IF(n.type() == FormExprType::BoundaryFunctionalSymbol ||
                        n.type() == FormExprType::BoundaryIntegralSymbol ||
                        n.type() == FormExprType::AuxiliaryStateSymbol,
                    InvalidArgumentException,
                    "compileBoundaryFunctionalKernel: coupled placeholders are not allowed in integrands");
        FE_THROW_IF(n.type() == FormExprType::CellIntegral ||
                        n.type() == FormExprType::BoundaryIntegral ||
                        n.type() == FormExprType::InteriorFaceIntegral ||
                        n.type() == FormExprType::InterfaceIntegral,
                    InvalidArgumentException,
                    "compileBoundaryFunctionalKernel: measures (dx/ds/dS/dI) are not allowed in integrands");
        for (const auto& child : n.childrenShared()) {
            if (child) self(self, *child);
        }
    };
    check(check, *integrand.node());

    const auto field_requirements = detail::analyzeFieldRequirements(*integrand.node());
    auto required = detail::analyzeRequiredData(*integrand.node(), FormKind::Linear);
    for (const auto& fr : field_requirements) {
        required |= fr.required;
    }

    auto base = std::make_shared<FunctionalFormKernel>(
        integrand,
        FunctionalFormKernel::Domain::BoundaryFace,
        required,
        field_requirements);

    if (!options.jit.enable) {
        return base;
    }

    return std::make_shared<jit::JITFunctionalKernelWrapper>(
        std::move(base),
        integrand,
        jit::JITFunctionalKernelWrapper::Domain::BoundaryFace,
        options.jit);
}

std::shared_ptr<assembly::FunctionalKernel>
compileBoundaryFunctionalKernel(const BoundaryFunctional& functional)
{
    return compileBoundaryFunctionalKernel(functional, SymbolicOptions{});
}

std::shared_ptr<assembly::FunctionalKernel>
compileBoundaryFunctionalKernel(const BoundaryFunctional& functional,
                                const SymbolicOptions& options)
{
    // Domain functionals use Cell domain instead of BoundaryFace.
    if (functional.is_domain_functional) {
        FE_THROW_IF(!functional.integrand.isValid(), InvalidArgumentException,
                    "compileBoundaryFunctionalKernel: invalid domain integrand");
        FE_CHECK_NOT_NULL(functional.integrand.node(),
                          "compileBoundaryFunctionalKernel: domain integrand node");
        const auto integrand =
            expandStatelessLaws(functional.integrand, largestTopologicalDimension(*functional.integrand.node()));

        const auto field_requirements =
            detail::analyzeFieldRequirements(*integrand.node());
        auto required = detail::analyzeRequiredData(
            *integrand.node(), FormKind::Linear);
        for (const auto& fr : field_requirements) {
            required |= fr.required;
        }

        auto base = std::make_shared<FunctionalFormKernel>(
            integrand,
            FunctionalFormKernel::Domain::Cell,
            required,
            field_requirements);

        if (!options.jit.enable) {
            return base;
        }

        return std::make_shared<jit::JITFunctionalKernelWrapper>(
            std::move(base),
            integrand,
            jit::JITFunctionalKernelWrapper::Domain::Cell,
            options.jit);
    }

    return compileBoundaryFunctionalKernel(
        functional.integrand, functional.boundary_marker, options);
}

} // namespace forms
} // namespace FE
} // namespace svmp
