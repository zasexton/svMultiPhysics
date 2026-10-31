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
#include "Forms/JIT/JITFunctionalKernelWrapper.h"

#include <utility>

namespace svmp {
namespace FE {
namespace forms {

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
compileBoundaryFunctionalKernel(const FormExpr& integrand,
                                int boundary_marker,
                                const SymbolicOptions& options)
{
    FE_THROW_IF(!integrand.isValid(), InvalidArgumentException,
                "compileBoundaryFunctionalKernel: invalid integrand");

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

        const auto field_requirements =
            detail::analyzeFieldRequirements(*functional.integrand.node());
        auto required = detail::analyzeRequiredData(
            *functional.integrand.node(), FormKind::Linear);
        for (const auto& fr : field_requirements) {
            required |= fr.required;
        }

        auto base = std::make_shared<FunctionalFormKernel>(
            functional.integrand,
            FunctionalFormKernel::Domain::Cell,
            required,
            field_requirements);

        if (!options.jit.enable) {
            return base;
        }

        return std::make_shared<jit::JITFunctionalKernelWrapper>(
            std::move(base),
            functional.integrand,
            jit::JITFunctionalKernelWrapper::Domain::Cell,
            options.jit);
    }

    return compileBoundaryFunctionalKernel(
        functional.integrand, functional.boundary_marker, options);
}

} // namespace forms
} // namespace FE
} // namespace svmp
