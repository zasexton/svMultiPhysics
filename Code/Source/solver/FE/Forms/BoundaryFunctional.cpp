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
    return compileBoundaryFunctionalKernel(functional.integrand, functional.boundary_marker);
}

std::shared_ptr<assembly::FunctionalKernel>
compileBoundaryFunctionalKernel(const BoundaryFunctional& functional,
                                const SymbolicOptions& options)
{
    return compileBoundaryFunctionalKernel(functional.integrand, functional.boundary_marker, options);
}

} // namespace forms
} // namespace FE
} // namespace svmp
