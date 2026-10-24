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

/// Whether a sum of integrals has an interior-facet (dS) term.
[[nodiscard]] bool hasInteriorFaceIntegral(const forms::FormExprNode& node)
{
    if (node.type() == forms::FormExprType::InteriorFaceIntegral) {
        return true;
    }
    if (node.type() != forms::FormExprType::Add && node.type() != forms::FormExprType::Subtract &&
        node.type() != forms::FormExprType::Negate) {
        return false;
    }
    for (const auto& child : node.childrenShared()) {
        if (child && hasInteriorFaceIntegral(*child)) {
            return true;
        }
    }
    return false;
}

/// Whether the tree has an integral (a measure).
[[nodiscard]] bool hasIntegral(const forms::FormExprNode& node)
{
    switch (node.type()) {
    case forms::FormExprType::CellIntegral:
    case forms::FormExprType::BoundaryIntegral:
    case forms::FormExprType::InteriorFaceIntegral:
    case forms::FormExprType::InterfaceIntegral:
    case forms::FormExprType::CutVolumeIntegral:
        return true;
    default:
        break;
    }
    for (const auto& child : node.childrenShared()) {
        if (child && hasIntegral(*child)) {
            return true;
        }
    }
    return false;
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
                where + "the form must be a sum of integrals (state it with a measure: dx, ds, dI, dL, "
                        "dCutVolume)");
    FE_THROW_IF(hasInteriorFaceIntegral(*declaration.form.node()), InvalidArgumentException,
                where + "the form has an interior-facet integral (dS), which a functional evaluator does not "
                        "integrate; state the functional with dx, ds, dI, dL or dCutVolume");
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

void declareWallFunctionalForm(FESystem& system, WallFunctionalFormDeclaration declaration)
{
    const std::string where = "FE::systems::declareWallFunctionalForm('" + declaration.prefix + "wall.<b>." +
                              declaration.quantity + "'): ";
    FE_THROW_IF(system.isSetup(), InvalidArgumentException,
                where + "functionals cannot be declared after setup");
    FE_THROW_IF(isBlank(declaration.prefix) || isBlank(declaration.quantity), InvalidArgumentException,
                where + "the prefix and the quantity must be nonempty");
    FE_THROW_IF(isBlank(declaration.owner), InvalidArgumentException, where + "the owner must be nonempty");
    FE_THROW_IF(std::string_view(functionalKindName(declaration.kind)) == "unknown", InvalidArgumentException,
                where + "the kind is invalid");
    FE_THROW_IF(!declaration.integrand.isValid() && declaration.by_wall.empty(), InvalidArgumentException,
                where + "no integrand is given");
    const auto require_integrand = [&](const forms::FormExpr& integrand, const std::string& which) {
        FE_THROW_IF(!integrand.isValid() || integrand.node() == nullptr, InvalidArgumentException,
                    where + "the integrand of " + which + " is invalid");
        FE_THROW_IF(hasIntegral(*integrand.node()), InvalidArgumentException,
                    where + "the integrand of " + which +
                        " has a measure; the evaluator integrates it over the wall's domains");
        FE_THROW_IF(integrand.hasTest() || integrand.hasTrial(), InvalidArgumentException,
                    where + "the integrand of " + which +
                        " must not contain a test or trial function (a functional is a 0-form)");
    };
    if (declaration.integrand.isValid()) {
        require_integrand(declaration.integrand, "every wall");
    }
    for (std::size_t i = 0; i < declaration.by_wall.size(); ++i) {
        const int wall = declaration.by_wall[i].first;
        require_integrand(declaration.by_wall[i].second, "wall " + std::to_string(wall));
        for (std::size_t j = 0; j < i; ++j) {
            FE_THROW_IF(declaration.by_wall[j].first == wall, InvalidArgumentException,
                        where + "wall " + std::to_string(wall) + " is listed twice");
        }
    }
    auto& store = system.attachment<WallFunctionalFormStore>();
    FE_THROW_IF(std::any_of(store.declarations.begin(), store.declarations.end(),
                            [&](const WallFunctionalFormDeclaration& existing) {
                                return existing.prefix == declaration.prefix &&
                                       existing.quantity == declaration.quantity;
                            }),
                InvalidArgumentException, where + "a wall functional of this name is already declared");
    store.declarations.push_back(std::move(declaration));
}

std::span<const WallFunctionalFormDeclaration> wallFunctionalForms(const FESystem& system) noexcept
{
    const auto* store = system.findAttachment<WallFunctionalFormStore>();
    if (store == nullptr) {
        return {};
    }
    return store->declarations;
}

std::string wallFunctionalName(const WallFunctionalFormDeclaration& declaration, int wall)
{
    return declaration.prefix + "wall." + std::to_string(wall) + "." + declaration.quantity;
}

forms::FormExpr wallFunctionalForm(const WallFunctionalFormDeclaration& declaration,
                                   int wall,
                                   std::span<const forms::ContactLineDomain> contact_lines,
                                   std::span<const forms::BoundaryDomain> wetted_parts)
{
    const auto listed = std::find_if(declaration.by_wall.begin(), declaration.by_wall.end(),
                                     [&](const auto& entry) { return entry.first == wall; });
    const forms::FormExpr& integrand = listed != declaration.by_wall.end() ? listed->second : declaration.integrand;
    forms::FormExpr sum{};
    if (!integrand.isValid()) {
        return sum;
    }
    const auto add = [&sum](forms::FormExpr term) { sum = sum.isValid() ? sum + term : std::move(term); };
    if (declaration.measure == WallMeasure::ContactLines) {
        for (const auto& line : contact_lines) {
            add(integrand.dL(line));
        }
    } else {
        for (const auto& part : wetted_parts) {
            add(integrand.ds(part));
        }
    }
    return sum;
}

} // namespace systems
} // namespace FE
} // namespace svmp
