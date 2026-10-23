/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#include "Systems/FunctionalForms.h"

#include "Core/FEException.h"
#include "Systems/FESystem.h"

#include <algorithm>
#include <string_view>
#include <utility>

namespace svmp {
namespace FE {
namespace systems {

namespace {

[[nodiscard]] bool isBlank(std::string_view text) noexcept
{
    return text.find_first_not_of(" \t\r\n") == std::string_view::npos;
}

[[nodiscard]] bool isSumOfIntegralsNode(const forms::FormExprNode& node)
{
    const auto children = node.childrenShared();
    switch (node.type()) {
    case forms::FormExprType::Add:
    case forms::FormExprType::Subtract:
        return children.size() == 2u && children[0] && children[1] &&
               isSumOfIntegralsNode(*children[0]) && isSumOfIntegralsNode(*children[1]);
    case forms::FormExprType::Negate:
        return children.size() == 1u && children[0] && isSumOfIntegralsNode(*children[0]);
    case forms::FormExprType::CellIntegral:
    case forms::FormExprType::BoundaryIntegral:
    case forms::FormExprType::InteriorFaceIntegral:
    case forms::FormExprType::InterfaceIntegral:
    case forms::FormExprType::CutVolumeIntegral:
        return true;
    default:
        return false;
    }
}

} // namespace

const char* functionalKindName(FunctionalKind kind) noexcept
{
    switch (kind) {
    case FunctionalKind::Stored:
        return "stored";
    case FunctionalKind::Power:
        return "power";
    case FunctionalKind::DissipationRate:
        return "dissipation_rate";
    case FunctionalKind::Measure:
        return "measure";
    }
    return "unknown";
}

bool isSumOfIntegrals(const forms::FormExpr& form)
{
    return form.isValid() && form.node() != nullptr && isSumOfIntegralsNode(*form.node());
}

void declareFunctionalForm(FESystem& system, FunctionalFormDeclaration declaration)
{
    const std::string where = "FE::systems::declareFunctionalForm('" + declaration.name + "'): ";
    FE_THROW_IF(system.isSetup(), InvalidArgumentException,
                where + "functionals cannot be declared after setup");
    FE_THROW_IF(isBlank(declaration.name), InvalidArgumentException, where + "the name must be nonempty");
    FE_THROW_IF(isBlank(declaration.owner), InvalidArgumentException, where + "the owner must be nonempty");
    FE_THROW_IF(std::string_view(functionalKindName(declaration.kind)) == "unknown", InvalidArgumentException,
                where + "the kind is invalid");
    FE_THROW_IF(!isSumOfIntegrals(declaration.form), InvalidArgumentException,
                where + "the form must be a sum of integrals (state it with a measure: dx, ds, dS, dI, dL, "
                        "dCutVolume)");
    FE_THROW_IF(declaration.form.hasTest() || declaration.form.hasTrial(), InvalidArgumentException,
                where + "the form must not contain a test or trial function (a functional is a 0-form)");
    auto& store = system.attachment<FunctionalFormStore>();
    FE_THROW_IF(std::any_of(store.declarations.begin(), store.declarations.end(),
                            [&](const FunctionalFormDeclaration& existing) {
                                return existing.name == declaration.name;
                            }),
                InvalidArgumentException, where + "a functional of this name is already declared");
    store.declarations.push_back(std::move(declaration));
}

std::span<const FunctionalFormDeclaration> functionalForms(const FESystem& system) noexcept
{
    const auto* store = system.findAttachment<FunctionalFormStore>();
    if (store == nullptr) {
        return {};
    }
    return store->declarations;
}

} // namespace systems
} // namespace FE
} // namespace svmp
