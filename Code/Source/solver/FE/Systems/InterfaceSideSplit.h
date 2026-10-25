/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#ifndef SVMP_FE_SYSTEMS_INTERFACESIDESPLIT_H
#define SVMP_FE_SYSTEMS_INTERFACESIDESPLIT_H

/**
 * @file InterfaceSideSplit.h
 * @brief The interface terms of an interface-owned field, one side at a time (D54 T2)
 *
 * A field that lives on an InterfaceMesh (FieldScope::InterfaceFace, or a
 * MortarSpace) is single-valued on the interface, while a volume field has a
 * value on each side.  The assembler evaluates such a pair on one side at a
 * time: a boundary-face context on the minus or the plus cell of each
 * interface face, with the interface field on its face-reference basis
 * (StandardAssembler::assembleInterfaceFaces).  splitInterfaceSides() turns
 * the dI(m) terms of one residual row that hold an interface-owned field into
 * such side terms:
 *
 * - the integrand is expanded into a sum of products (through +, -, *, /,
 *   inner products, restrictions, jump() and avg());
 * - a product holding u.minus() of a volume field goes to the minus side, one
 *   holding u.plus() to the plus side, one holding only interface-owned,
 *   global and geometric symbols to the minus side;
 * - the restrictions are dropped (the context is the side's cell), and the
 *   normal is re-signed: dI's normal points from the minus to the plus side,
 *   so on the plus side n and n.minus() become -n and n.plus() becomes n;
 * - each side's products are integrated as a boundary-face integral on marker
 *   m, which a form kernel evaluates in that context.
 *
 * Refused, with the term named: .minus()/.plus()/jump()/avg() of an
 * interface-owned symbol (it has one value), a volume symbol without a side,
 * a product of the two sides, a product of an interface-owned unknown with
 * another unknown (the single-sided context holds the state of one unknown
 * besides the volume fields of the side's cell), and dI() without a marker.
 * Terms that hold no interface-owned field are left as they are.
 */

#include "Assembly/AssemblyKernel.h"
#include "Core/Types.h"
#include "Forms/FormExpr.h"

#include <memory>
#include <string>
#include <string_view>
#include <vector>

namespace svmp {
namespace FE {
namespace systems {

class FESystem;

/// The products of one row on one side of one interface.
struct InterfaceSidePart {
    int marker{-1};
    assembly::InterfaceEvaluationSide side{assembly::InterfaceEvaluationSide::Minus};
    /// Integrands (one product each, restrictions dropped, normal re-signed).
    std::vector<forms::FormExpr> products;

    /// The products as one boundary-face integral on the marker (or, with
    /// @p only_with, those that hold StateField(only_with)).
    [[nodiscard]] forms::FormExpr integral(FieldId only_with = INVALID_FIELD_ID) const;
};

struct InterfaceSideSplit {
    /// The row without the interface terms of interface-owned fields (invalid
    /// when nothing is left).
    forms::FormExpr rest;
    std::vector<InterfaceSidePart> parts;
};

/// True when a registered field is interface-owned (FieldScope::InterfaceFace).
[[nodiscard]] bool hasInterfaceOwnedFields(const FESystem& system);

/// Splits the row of test field @p row_name (see the file comment).
[[nodiscard]] InterfaceSideSplit splitInterfaceSides(const FESystem& system,
                                                     const forms::FormExpr& row,
                                                     std::string_view row_name);

/**
 * @brief A kernel evaluated on one side of an InterfaceMesh
 *
 * Forwards to @p kernel and reports hasSingleSidedInterfaceFace() and @p side,
 * so the assembler evaluates its boundary-face terms on that side of each
 * interface face.
 */
class SingleSidedInterfaceKernel final : public assembly::AssemblyKernel {
public:
    SingleSidedInterfaceKernel(std::shared_ptr<assembly::AssemblyKernel> kernel,
                               assembly::InterfaceEvaluationSide side);

    [[nodiscard]] assembly::RequiredData getRequiredData() const override;
    [[nodiscard]] std::vector<assembly::FieldRequirement> fieldRequirements() const override;
    [[nodiscard]] assembly::MaterialStateSpec materialStateSpec() const noexcept override;
    [[nodiscard]] std::vector<params::Spec> parameterSpecs() const override;
    [[nodiscard]] std::vector<analysis::ContributionDescriptor> analysisContributions() const override;
    void resolveParameterSlots(
        const std::function<std::optional<std::uint32_t>(std::string_view)>& slot_of_real_param) override;
    void resolveInlinableConstitutives() override;

    void computeCell(const assembly::AssemblyContext& ctx, assembly::KernelOutput& output) override;
    void computeBoundaryFace(const assembly::AssemblyContext& ctx,
                             int boundary_marker,
                             assembly::KernelOutput& output) override;

    [[nodiscard]] bool hasCell() const noexcept override { return false; }
    [[nodiscard]] bool hasBoundaryFace() const noexcept override { return true; }
    [[nodiscard]] bool hasSingleSidedInterfaceFace() const noexcept override { return true; }
    [[nodiscard]] assembly::InterfaceEvaluationSide interfaceEvaluationSide() const noexcept override
    {
        return side_;
    }
    [[nodiscard]] bool hasInteriorFace() const noexcept override { return false; }
    [[nodiscard]] bool hasInterfaceFace() const noexcept override { return false; }

    [[nodiscard]] std::string name() const override;
    [[nodiscard]] assembly::SemanticKernelKind semanticKernelKind() const noexcept override;
    [[nodiscard]] int maxTemporalDerivativeOrder() const noexcept override;
    [[nodiscard]] bool hasExplicitTimeDependency() const noexcept override;
    [[nodiscard]] bool hasStateIndependentMatrix() const noexcept override;
    [[nodiscard]] bool isSymmetric() const noexcept override { return false; }
    [[nodiscard]] bool isMatrixOnly() const noexcept override;
    [[nodiscard]] bool isVectorOnly() const noexcept override;

    [[nodiscard]] const assembly::AssemblyKernel& inner() const noexcept { return *kernel_; }

private:
    std::shared_ptr<assembly::AssemblyKernel> kernel_;
    assembly::InterfaceEvaluationSide side_{assembly::InterfaceEvaluationSide::Minus};
};

} // namespace systems
} // namespace FE
} // namespace svmp

#endif // SVMP_FE_SYSTEMS_INTERFACESIDESPLIT_H
