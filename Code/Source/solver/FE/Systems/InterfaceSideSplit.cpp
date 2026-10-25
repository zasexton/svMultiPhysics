/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#include "Systems/InterfaceSideSplit.h"

#include "Core/FEException.h"
#include "Spaces/FunctionSpace.h"
#include "Systems/FESystem.h"

#include <algorithm>
#include <map>
#include <optional>
#include <set>
#include <utility>

namespace svmp {
namespace FE {
namespace systems {

namespace {

using forms::FormExpr;
using forms::FormExprNode;
using forms::FormExprType;
using Side = assembly::InterfaceEvaluationSide;

[[nodiscard]] FormExpr wrap(const std::shared_ptr<FormExprNode>& node)
{
    return FormExpr(node);
}

[[nodiscard]] bool isFieldSymbol(const FormExprNode& node) noexcept
{
    switch (node.type()) {
    case FormExprType::TestFunction:
    case FormExprType::TrialFunction:
    case FormExprType::StateField:
    case FormExprType::DiscreteField:
        return true;
    default:
        return false;
    }
}

[[nodiscard]] bool isGlobalSymbol(const FormExprNode& node) noexcept
{
    const auto* signature = node.spaceSignature();
    return signature != nullptr && signature->space_type == spaces::SpaceType::Global;
}

class Splitter {
public:
    Splitter(const FESystem& system, std::string_view row_name)
        : system_(system)
        , row_name_(row_name)
    {
    }

    /// Whether @p node holds a symbol of an interface-owned field.
    [[nodiscard]] bool holdsInterfaceOwned(const FormExprNode& node) const
    {
        if (isFieldSymbol(node) && isInterfaceOwned(node)) {
            return true;
        }
        for (const auto& child : node.childrenShared()) {
            if (child && holdsInterfaceOwned(*child)) {
                return true;
            }
        }
        return false;
    }

    [[nodiscard]] bool isInterfaceOwned(const FormExprNode& node) const
    {
        if (node.spaceSignature() != nullptr && node.spaceSignature()->space_type == spaces::SpaceType::Mortar) {
            return true;
        }
        const auto id = node.fieldId();
        return id.has_value() && system_.hasField(*id) &&
               system_.fieldRecord(*id).scope == FieldScope::InterfaceFace;
    }

    /// jump() and avg() written with restrictions; a check that no
    /// interface-owned symbol is restricted.
    [[nodiscard]] FormExpr withExplicitSides(const FormExpr& integrand) const
    {
        return integrand.transformNodes([this](const FormExprNode& node) -> std::optional<FormExpr> {
            const auto kids = node.childrenShared();
            switch (node.type()) {
            case FormExprType::Jump:
            case FormExprType::Average:
            case FormExprType::RestrictMinus:
            case FormExprType::RestrictPlus: {
                if (holdsInterfaceOwned(*kids.at(0))) {
                    fail("an interface-owned field has one value on the interface; it takes no .minus(), "
                         ".plus(), jump() or avg()",
                         node);
                }
                const auto inner = withExplicitSides(wrap(kids.at(0)));
                if (node.type() == FormExprType::Jump) {
                    return inner.minus() - inner.plus();
                }
                if (node.type() == FormExprType::Average) {
                    return FormExpr::constant(Real(0.5)) * (inner.minus() + inner.plus());
                }
                return node.type() == FormExprType::RestrictMinus ? inner.minus() : inner.plus();
            }
            default:
                return std::nullopt;
            }
        });
    }

    /// The integrand as a sum of products (signs folded in).
    [[nodiscard]] std::vector<FormExpr> expand(const FormExpr& expr) const
    {
        const auto& node = *expr.node();
        const auto kids = node.childrenShared();
        const auto both = [&](const auto& combine) {
            std::vector<FormExpr> out;
            for (const auto& a : expand(wrap(kids.at(0)))) {
                for (const auto& b : expand(wrap(kids.at(1)))) {
                    out.push_back(combine(a, b));
                }
            }
            return out;
        };
        switch (node.type()) {
        case FormExprType::Add: {
            auto out = expand(wrap(kids.at(0)));
            for (auto& b : expand(wrap(kids.at(1)))) {
                out.push_back(std::move(b));
            }
            return out;
        }
        case FormExprType::Subtract: {
            auto out = expand(wrap(kids.at(0)));
            for (auto& b : expand(wrap(kids.at(1)))) {
                out.push_back(-b);
            }
            return out;
        }
        case FormExprType::Negate: {
            auto out = expand(wrap(kids.at(0)));
            for (auto& a : out) {
                a = -a;
            }
            return out;
        }
        case FormExprType::Multiply:
            return both([](const FormExpr& a, const FormExpr& b) { return a * b; });
        case FormExprType::InnerProduct:
            return both([](const FormExpr& a, const FormExpr& b) { return forms::inner(a, b); });
        case FormExprType::Divide: {
            std::vector<FormExpr> out;
            const auto denominator = wrap(kids.at(1));
            for (const auto& a : expand(wrap(kids.at(0)))) {
                out.push_back(a / denominator);
            }
            return out;
        }
        case FormExprType::RestrictMinus:
        case FormExprType::RestrictPlus: {
            std::vector<FormExpr> out;
            for (const auto& a : expand(wrap(kids.at(0)))) {
                out.push_back(node.type() == FormExprType::RestrictMinus ? a.minus() : a.plus());
            }
            return out;
        }
        default:
            return {expr};
        }
    }

    /// The side a product belongs to: the side of its restricted volume
    /// symbols; Minus when it has none.
    [[nodiscard]] Side sideOf(const FormExpr& product) const
    {
        std::set<Side> sides;
        collectSides(*product.node(), std::nullopt, sides, product);
        if (sides.size() > 1u) {
            fail("a product holds both sides of the interface; write each side's terms apart", *product.node());
        }
        return sides.empty() ? Side::Minus : *sides.begin();
    }

    /// The product on its side: restrictions dropped, the normal re-signed.
    [[nodiscard]] FormExpr onSide(const FormExpr& product, Side side) const
    {
        return product.transformNodes([&](const FormExprNode& node) -> std::optional<FormExpr> {
            const auto kids = node.childrenShared();
            if (node.type() == FormExprType::RestrictMinus || node.type() == FormExprType::RestrictPlus) {
                const Side restricted = node.type() == FormExprType::RestrictMinus ? Side::Minus : Side::Plus;
                const auto inner = wrap(kids.at(0));
                if (restricted == side) {
                    return inner;
                }
                // Of the other side only the normal (-n) and single-valued
                // symbols are known here.
                if (kids.at(0)->type() == FormExprType::Normal) {
                    return -inner;
                }
                if (!readsACell(*kids.at(0))) {
                    return inner;
                }
                fail("a product reads the other side of the interface", node);
            }
            if (node.type() == FormExprType::Normal && side == Side::Plus) {
                // dI's normal points from the minus to the plus side.
                return -FormExpr::normal();
            }
            if (node.type() == FormExprType::CurrentNormal || node.type() == FormExprType::ReferenceNormal) {
                fail("the moving-mesh normals are not supported in single-sided interface terms yet", node);
            }
            return std::nullopt;
        });
    }

    /// The state fields of a product.
    [[nodiscard]] std::set<FieldId> stateFieldsOf(const FormExprNode& node) const
    {
        std::set<FieldId> out;
        collectStateFields(node, out);
        return out;
    }

    [[nodiscard]] bool isInterfaceOwnedField(FieldId field) const
    {
        return system_.hasField(field) && system_.fieldRecord(field).scope == FieldScope::InterfaceFace;
    }

    [[noreturn]] void fail(const std::string& why, const FormExprNode& node) const
    {
        FE_THROW(InvalidArgumentException,
                 "installFormulation: the interface terms of the row of '" + std::string(row_name_) + "': " + why +
                     " ('" + node.toString() + "')");
    }

private:
    void collectSides(const FormExprNode& node,
                      std::optional<Side> restricted,
                      std::set<Side>& sides,
                      const FormExpr& product) const
    {
        if (node.type() == FormExprType::RestrictMinus || node.type() == FormExprType::RestrictPlus) {
            restricted = node.type() == FormExprType::RestrictMinus ? Side::Minus : Side::Plus;
        }
        if (isFieldSymbol(node) && !isInterfaceOwned(node) && !isGlobalSymbol(node)) {
            if (!restricted.has_value()) {
                fail("a volume field on the interface needs its side: write .minus() or .plus()", *product.node());
            }
            sides.insert(*restricted);
        }
        for (const auto& child : node.childrenShared()) {
            if (child) {
                collectSides(*child, restricted, sides, product);
            }
        }
    }

    void collectStateFields(const FormExprNode& node, std::set<FieldId>& out) const
    {
        if ((node.type() == FormExprType::StateField || node.type() == FormExprType::TrialFunction) &&
            node.fieldId().has_value()) {
            out.insert(*node.fieldId());
        }
        for (const auto& child : node.childrenShared()) {
            if (child) {
                collectStateFields(*child, out);
            }
        }
    }

    /// Whether @p node reads a cell: a volume field or a cell-geometric quantity.
    [[nodiscard]] bool readsACell(const FormExprNode& node) const
    {
        if (isFieldSymbol(node) && !isInterfaceOwned(node) && !isGlobalSymbol(node)) {
            return true;
        }
        switch (node.type()) {
        case FormExprType::Normal:
        case FormExprType::CellDiameter:
        case FormExprType::CellVolume:
        case FormExprType::FacetArea:
        case FormExprType::CellDomainId:
        case FormExprType::Jacobian:
        case FormExprType::JacobianInverse:
        case FormExprType::JacobianDeterminant:
            return true;
        default:
            break;
        }
        for (const auto& child : node.childrenShared()) {
            if (child && readsACell(*child)) {
                return true;
            }
        }
        return false;
    }

    const FESystem& system_;
    std::string_view row_name_;
};

/// The top-level integrals of a row, with their signs.
void collectIntegrals(const FormExpr& expr, bool negative, std::vector<std::pair<FormExpr, bool>>& out)
{
    const auto& node = *expr.node();
    const auto kids = node.childrenShared();
    switch (node.type()) {
    case FormExprType::Add:
        collectIntegrals(wrap(kids.at(0)), negative, out);
        collectIntegrals(wrap(kids.at(1)), negative, out);
        return;
    case FormExprType::Subtract:
        collectIntegrals(wrap(kids.at(0)), negative, out);
        collectIntegrals(wrap(kids.at(1)), !negative, out);
        return;
    case FormExprType::Negate:
        collectIntegrals(wrap(kids.at(0)), !negative, out);
        return;
    default:
        out.emplace_back(expr, negative);
        return;
    }
}

} // namespace

FormExpr InterfaceSidePart::integral(FieldId only_with) const
{
    FormExpr sum;
    for (const auto& product : products) {
        if (only_with != INVALID_FIELD_ID) {
            bool holds = false;
            const auto visit = [&](const auto& self, const FormExprNode& node) -> void {
                if ((node.type() == FormExprType::StateField || node.type() == FormExprType::TrialFunction) &&
                    node.fieldId().has_value() && *node.fieldId() == only_with) {
                    holds = true;
                }
                for (const auto& child : node.childrenShared()) {
                    if (child && !holds) {
                        self(self, *child);
                    }
                }
            };
            visit(visit, *product.node());
            if (!holds) {
                continue;
            }
        }
        sum = sum.isValid() ? sum + product : product;
    }
    return sum.isValid() ? sum.ds(marker) : FormExpr{};
}

std::string interfaceOwnedFieldIn(const FESystem& system, const forms::FormExpr& expr)
{
    if (!expr.isValid()) {
        return {};
    }
    const Splitter splitter(system, {});
    const auto visit = [&](const auto& self, const FormExprNode& node) -> std::string {
        if (isFieldSymbol(node) && splitter.isInterfaceOwned(node)) {
            const auto id = node.fieldId();
            return id.has_value() && system.hasField(*id) ? system.fieldRecord(*id).name : node.toString();
        }
        for (const auto& child : node.childrenShared()) {
            if (child) {
                auto name = self(self, *child);
                if (!name.empty()) {
                    return name;
                }
            }
        }
        return {};
    };
    return visit(visit, *expr.node());
}

bool hasInterfaceOwnedFields(const FESystem& system)
{
    for (std::size_t id = 0; id < system.registeredFieldCount(); ++id) {
        const auto field = static_cast<FieldId>(id);
        if (system.hasField(field) && system.fieldRecord(field).scope == FieldScope::InterfaceFace) {
            return true;
        }
    }
    return false;
}

InterfaceSideSplit splitInterfaceSides(const FESystem& system, const forms::FormExpr& row, std::string_view row_name)
{
    InterfaceSideSplit out;
    if (!row.isValid()) {
        return out;
    }
    Splitter splitter(system, row_name);
    std::vector<std::pair<FormExpr, bool>> integrals;
    collectIntegrals(row, false, integrals);

    std::map<std::pair<int, Side>, std::size_t> part_index;
    for (const auto& [integral, negative] : integrals) {
        const auto& node = *integral.node();
        if (!splitter.holdsInterfaceOwned(node)) {
            const auto signed_integral = negative ? -integral : integral;
            out.rest = out.rest.isValid() ? out.rest + signed_integral : signed_integral;
            continue;
        }
        if (node.type() != FormExprType::InterfaceIntegral) {
            splitter.fail("an interface-owned field lives on its interface: integrate it with dI(marker), also on "
                          "an interface built from a boundary label",
                          node);
        }
        const auto marker = node.interfaceMarker();
        if (!marker.has_value() || *marker < 0) {
            splitter.fail("an interface-owned field is integrated on dI() without a marker; name the interface",
                          node);
        }
        const auto integrand = splitter.withExplicitSides(wrap(node.childrenShared().at(0)));
        for (auto product : splitter.expand(integrand)) {
            if (negative) {
                product = -product;
            }
            const auto side = splitter.sideOf(product);
            auto on_side = splitter.onSide(product, side);
            // The single-sided context holds the state of one unknown besides
            // the volume fields of the side's cell.
            const auto states = splitter.stateFieldsOf(*on_side.node());
            const auto interface_states = static_cast<std::size_t>(
                std::count_if(states.begin(), states.end(),
                              [&](FieldId f) { return splitter.isInterfaceOwnedField(f); }));
            if (interface_states > 0u && states.size() > 1u) {
                splitter.fail("a product of an interface-owned unknown with another unknown is not supported on the "
                              "interface yet",
                              *product.node());
            }
            const auto key = std::make_pair(*marker, side);
            auto it = part_index.find(key);
            if (it == part_index.end()) {
                it = part_index.emplace(key, out.parts.size()).first;
                out.parts.push_back(InterfaceSidePart{*marker, side, {}});
            }
            out.parts[it->second].products.push_back(std::move(on_side));
        }
    }
    return out;
}

SingleSidedInterfaceKernel::SingleSidedInterfaceKernel(std::shared_ptr<assembly::AssemblyKernel> kernel,
                                                       assembly::InterfaceEvaluationSide side)
    : kernel_(std::move(kernel))
    , side_(side)
{
    FE_CHECK_NOT_NULL(kernel_.get(), "SingleSidedInterfaceKernel: kernel");
}

assembly::RequiredData SingleSidedInterfaceKernel::getRequiredData() const
{
    return kernel_->getRequiredData();
}

std::vector<assembly::FieldRequirement> SingleSidedInterfaceKernel::fieldRequirements() const
{
    return kernel_->fieldRequirements();
}

assembly::MaterialStateSpec SingleSidedInterfaceKernel::materialStateSpec() const noexcept
{
    return kernel_->materialStateSpec();
}

std::vector<params::Spec> SingleSidedInterfaceKernel::parameterSpecs() const
{
    return kernel_->parameterSpecs();
}

std::vector<analysis::ContributionDescriptor> SingleSidedInterfaceKernel::analysisContributions() const
{
    return kernel_->analysisContributions();
}

void SingleSidedInterfaceKernel::resolveParameterSlots(
    const std::function<std::optional<std::uint32_t>(std::string_view)>& slot_of_real_param)
{
    kernel_->resolveParameterSlots(slot_of_real_param);
}

void SingleSidedInterfaceKernel::resolveInlinableConstitutives()
{
    kernel_->resolveInlinableConstitutives();
}

void SingleSidedInterfaceKernel::computeCell(const assembly::AssemblyContext& /*ctx*/,
                                             assembly::KernelOutput& /*output*/)
{
    FE_THROW(FEException, "SingleSidedInterfaceKernel: no cell terms");
}

void SingleSidedInterfaceKernel::computeBoundaryFace(const assembly::AssemblyContext& ctx,
                                                     int boundary_marker,
                                                     assembly::KernelOutput& output)
{
    kernel_->computeBoundaryFace(ctx, boundary_marker, output);
}

std::string SingleSidedInterfaceKernel::name() const
{
    return "SingleSidedInterfaceKernel(" + kernel_->name() + ", " +
           (side_ == assembly::InterfaceEvaluationSide::Minus ? "minus" : "plus") + ")";
}

assembly::SemanticKernelKind SingleSidedInterfaceKernel::semanticKernelKind() const noexcept
{
    return kernel_->semanticKernelKind();
}

int SingleSidedInterfaceKernel::maxTemporalDerivativeOrder() const noexcept
{
    return kernel_->maxTemporalDerivativeOrder();
}

bool SingleSidedInterfaceKernel::hasExplicitTimeDependency() const noexcept
{
    return kernel_->hasExplicitTimeDependency();
}

bool SingleSidedInterfaceKernel::hasStateIndependentMatrix() const noexcept
{
    return kernel_->hasStateIndependentMatrix();
}

bool SingleSidedInterfaceKernel::isMatrixOnly() const noexcept
{
    return kernel_->isMatrixOnly();
}

bool SingleSidedInterfaceKernel::isVectorOnly() const noexcept
{
    return kernel_->isVectorOnly();
}

} // namespace systems
} // namespace FE
} // namespace svmp
