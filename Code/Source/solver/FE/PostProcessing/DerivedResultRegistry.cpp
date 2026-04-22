#include "PostProcessing/DerivedResultRegistry.h"

#include "Core/FEException.h"

#include <algorithm>
#include <cctype>
#include <sstream>
#include <stdexcept>

namespace svmp {
namespace FE {
namespace post {

namespace {

[[nodiscard]] bool isMeasureNode(forms::FormExprType type) noexcept
{
    using FT = forms::FormExprType;
    return type == FT::CellIntegral ||
           type == FT::BoundaryIntegral ||
           type == FT::InteriorFaceIntegral ||
           type == FT::InterfaceIntegral;
}

void collectReferencedFieldsImpl(const forms::FormExprNode& node,
                                 std::vector<FieldId>& fields)
{
    using FT = forms::FormExprType;
    if (node.type() == FT::StateField || node.type() == FT::DiscreteField) {
        const auto fid = node.fieldId();
        FE_THROW_IF(!fid, InvalidArgumentException,
                    "Derived result expression contains a field node without FieldId");
        if (std::find(fields.begin(), fields.end(), *fid) == fields.end()) {
            fields.push_back(*fid);
        }
    }

    for (const auto* child : node.children()) {
        if (child != nullptr) {
            collectReferencedFieldsImpl(*child, fields);
        }
    }
}

void validateExpressionImpl(const forms::FormExprNode& node,
                            const std::string& result_name)
{
    using FT = forms::FormExprType;
    const auto type = node.type();

    FE_THROW_IF(type == FT::TestFunction, InvalidArgumentException,
                "Derived result '" + result_name + "' must not contain TestFunction nodes");
    FE_THROW_IF(type == FT::TrialFunction, InvalidArgumentException,
                "Derived result '" + result_name + "' must not contain TrialFunction nodes");
    FE_THROW_IF(isMeasureNode(type), InvalidArgumentException,
                "Derived result '" + result_name + "' must not contain integral measure nodes");
    FE_THROW_IF(type == FT::AuxiliaryOutputSymbol || type == FT::AuxiliaryOutputRef ||
                    type == FT::AuxiliaryInputSymbol || type == FT::AuxiliaryInputRef ||
                    type == FT::AuxiliaryStateSymbol || type == FT::AuxiliaryStateRef,
                InvalidArgumentException,
                "Derived result '" + result_name +
                    "' contains auxiliary nodes, which are not supported by output evaluation yet");

    for (const auto* child : node.children()) {
        if (child != nullptr) {
            validateExpressionImpl(*child, result_name);
        }
    }
}

[[nodiscard]] bool hasDifferentialOperator(const forms::FormExprNode& node) noexcept
{
    using FT = forms::FormExprType;
    const auto type = node.type();
    if (type == FT::Gradient || type == FT::Divergence || type == FT::Curl || type == FT::Hessian) {
        return true;
    }
    for (const auto* child : node.children()) {
        if (child != nullptr && hasDifferentialOperator(*child)) {
            return true;
        }
    }
    return false;
}

[[nodiscard]] bool validResultName(std::string_view name) noexcept
{
    if (name.empty()) {
        return false;
    }
    for (const char ch : name) {
        const auto uch = static_cast<unsigned char>(ch);
        if (!(std::isalnum(uch) || ch == '_' || ch == '-' || ch == '.')) {
            return false;
        }
    }
    return true;
}

[[nodiscard]] bool isImplementedScopePolicy(DerivedResultScope scope,
                                            DerivedResultPolicy policy) noexcept
{
    return (scope == DerivedResultScope::Cell &&
            (policy == DerivedResultPolicy::CellAverage ||
             policy == DerivedResultPolicy::CellCentroid)) ||
           (scope == DerivedResultScope::Vertex &&
            (policy == DerivedResultPolicy::PointValue ||
             policy == DerivedResultPolicy::PatchAverage));
}

} // namespace

std::vector<FieldId> collectReferencedFields(const forms::FormExpr& expr)
{
    std::vector<FieldId> fields;
    FE_THROW_IF(!expr.isValid(), InvalidArgumentException,
                "collectReferencedFields: invalid expression");
    collectReferencedFieldsImpl(*expr.node(), fields);
    return fields;
}

void validateDerivedResultDefinition(DerivedResultDefinition& def)
{
    FE_THROW_IF(def.name.empty(), InvalidArgumentException,
                "Derived result name must not be empty");
    FE_THROW_IF(!validResultName(def.name), InvalidArgumentException,
                "Derived result '" + def.name +
                    "' has an invalid mesh-field name; use letters, digits, '_', '-', or '.'");
    FE_THROW_IF(componentCount(def.shape) == 0, InvalidArgumentException,
                "Derived result '" + def.name + "' must have at least one component");
    FE_THROW_IF(!def.expression.isValid(), InvalidArgumentException,
                "Derived result '" + def.name + "' has an invalid expression");

    validateExpressionImpl(*def.expression.node(), def.name);

    if (def.referenced_fields.empty()) {
        def.referenced_fields = collectReferencedFields(def.expression);
    }

    FE_THROW_IF(def.scope == DerivedResultScope::BoundaryFace && !def.marker.has_value(),
                InvalidArgumentException,
                "Derived result '" + def.name + "' uses BoundaryFace scope without a marker");

    const auto invalid_scope_policy = [&] {
        switch (def.policy) {
        case DerivedResultPolicy::CellAverage:
        case DerivedResultPolicy::CellCentroid:
            return def.scope != DerivedResultScope::Cell;
        case DerivedResultPolicy::FaceAverage:
        case DerivedResultPolicy::FaceCentroid:
            return def.scope != DerivedResultScope::Face &&
                   def.scope != DerivedResultScope::BoundaryFace;
        case DerivedResultPolicy::PatchAverage:
            return def.scope != DerivedResultScope::Vertex;
        case DerivedResultPolicy::PointValue:
            return def.scope != DerivedResultScope::Vertex &&
                   def.scope != DerivedResultScope::Edge;
        case DerivedResultPolicy::EdgeAverage:
            return def.scope != DerivedResultScope::Edge;
        case DerivedResultPolicy::QuadratureValue:
            return def.scope != DerivedResultScope::QuadraturePoint;
        case DerivedResultPolicy::ProjectToCell:
            return def.scope != DerivedResultScope::QuadraturePoint &&
                   def.scope != DerivedResultScope::Cell;
        case DerivedResultPolicy::ProjectToVertex:
            return def.scope != DerivedResultScope::QuadraturePoint &&
                   def.scope != DerivedResultScope::Vertex;
        case DerivedResultPolicy::L2Projection:
            return def.scope != DerivedResultScope::Vertex &&
                   def.scope != DerivedResultScope::Cell;
        }
        return true;
    }();

    FE_THROW_IF(invalid_scope_policy, InvalidArgumentException,
                "Derived result '" + def.name + "' has incompatible scope/policy combination: " +
                    std::string(toString(def.scope)) + "/" + std::string(toString(def.policy)));

    FE_THROW_IF(def.scope == DerivedResultScope::Vertex &&
                    def.policy == DerivedResultPolicy::PointValue &&
                    hasDifferentialOperator(*def.expression.node()),
                InvalidArgumentException,
                "Derived result '" + def.name +
                    "' uses a differential operator with Vertex/PointValue; use PatchAverage or a projection policy");

    if (!isImplementedScopePolicy(def.scope, def.policy)) {
        // Registration is allowed so physics-facing APIs are stable, but the
        // evaluator will report a clear not-implemented error until the scope
        // is implemented.
        return;
    }
}

DerivedResultHandle DerivedResultRegistry::registerDefinition(DerivedResultDefinition def)
{
    validateDerivedResultDefinition(def);
    FE_THROW_IF(name_to_id_.find(def.name) != name_to_id_.end(), InvalidArgumentException,
                "Derived result '" + def.name + "' is already registered");

    const auto id = definitions_.size();
    definitions_.push_back(std::move(def));
    name_to_id_.emplace(definitions_.back().name, id);
    return DerivedResultHandle{id};
}

const DerivedResultDefinition& DerivedResultRegistry::get(DerivedResultHandle handle) const
{
    FE_THROW_IF(!handle.valid() || handle.id >= definitions_.size(), InvalidArgumentException,
                "DerivedResultRegistry::get: invalid handle");
    return definitions_[handle.id];
}

const DerivedResultDefinition& DerivedResultRegistry::get(std::string_view name) const
{
    const auto it = name_to_id_.find(std::string(name));
    FE_THROW_IF(it == name_to_id_.end(), InvalidArgumentException,
                "DerivedResultRegistry::get: unknown derived result '" + std::string(name) + "'");
    return definitions_[it->second];
}

bool DerivedResultRegistry::contains(std::string_view name) const noexcept
{
    for (const auto& entry : name_to_id_) {
        if (entry.first == name) {
            return true;
        }
    }
    return false;
}

} // namespace post
} // namespace FE
} // namespace svmp
