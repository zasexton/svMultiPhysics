/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#include "Analysis/FormExprScanner.h"

#include "Forms/ConstitutiveModel.h"
#include "Forms/JIT/InlinableConstitutiveModel.h"

#include <algorithm>
#include <cmath>
#include <exception>
#include <limits>
#include <sstream>
#include <typeinfo>
#include <unordered_set>
#include <utility>

namespace svmp {
namespace FE {
namespace analysis {

namespace {

void addFieldIfAbsent(std::vector<FieldId>& fields, FieldId field)
{
    if (field == INVALID_FIELD_ID) {
        return;
    }
    if (std::find(fields.begin(), fields.end(), field) == fields.end()) {
        fields.push_back(field);
    }
}

void collectPrimaryCandidateFields(const forms::FormExprNode& node,
                                   std::vector<FieldId>& fields)
{
    using FT = forms::FormExprType;

    switch (node.type()) {
        case FT::StateField:
        case FT::DiscreteField:
        case FT::TrialFunction:
            if (const auto field = node.fieldId()) {
                addFieldIfAbsent(fields, *field);
            }
            break;
        default:
            break;
    }

    for (const auto& child : node.childrenShared()) {
        if (child) {
            collectPrimaryCandidateFields(*child, fields);
        }
    }
}

FieldId inferConstitutivePrimaryField(const forms::FormExprNode& node)
{
    std::vector<FieldId> fields;
    for (const auto& child : node.childrenShared()) {
        if (child) {
            collectPrimaryCandidateFields(*child, fields);
        }
    }
    return fields.size() == 1u ? fields.front() : INVALID_FIELD_ID;
}

std::string tensorRankString(forms::ConstitutiveModel::ValueKind kind)
{
    using Kind = forms::ConstitutiveModel::ValueKind;
    switch (kind) {
        case Kind::Scalar: return "scalar";
        case Kind::Vector: return "vector";
        case Kind::Matrix:
        case Kind::SymmetricMatrix:
        case Kind::SkewMatrix:
            return "rank2";
        case Kind::Tensor3: return "rank3";
        case Kind::Tensor4: return "rank4";
    }
    return {};
}

std::string symmetryString(forms::ConstitutiveModel::ValueKind kind)
{
    using Kind = forms::ConstitutiveModel::ValueKind;
    switch (kind) {
        case Kind::Scalar: return "not_applicable";
        case Kind::SymmetricMatrix: return "symmetric";
        case Kind::SkewMatrix: return "skew";
        default: return {};
    }
}

std::string genericConstitutiveName(const forms::ConstitutiveModel& model,
                                    std::size_t output_index)
{
    std::ostringstream os;
    os << "constitutive:";
    if (const auto* inlinable = model.inlinable()) {
        os << "kind:" << inlinable->kindId();
    } else {
        os << "type:" << typeid(model).name();
    }
    os << ":output:" << output_index;
    return os.str();
}

ConstitutiveLawMetadata inferGenericConstitutiveLawMetadata(
    std::shared_ptr<const forms::ConstitutiveModel> model,
    std::size_t output_index,
    FieldId primary_field)
{
    ConstitutiveLawMetadata law;
    if (!model) {
        return law;
    }
    law.name = genericConstitutiveName(*model, output_index);
    law.primary_field = primary_field;
    law.model = model;
    law.state_dependent = !model->stateSpec().empty();
    if (const auto* inlinable = model->inlinable()) {
        law.state_dependent =
            law.state_dependent ||
            inlinable->stateAccess() != forms::MaterialStateAccess::None;
    }

    try {
        const auto spec = model->outputSpec(output_index);
        if (spec.kind.has_value()) {
            law.tensor_rank = tensorRankString(*spec.kind);
            law.symmetry_class = symmetryString(*spec.kind);
        }
    } catch (const std::exception&) {
        // Missing optional output metadata should not prevent generic DAG
        // metadata from being published.
    }

    return law;
}

void fillConstitutiveMetadataDefaults(ConstitutiveLawMetadata& law,
                                      const ConstitutiveLawMetadata& fallback)
{
    if (law.name.empty()) law.name = fallback.name;
    if (law.primary_field == INVALID_FIELD_ID) law.primary_field = fallback.primary_field;
    if (law.tensor_rank.empty()) law.tensor_rank = fallback.tensor_rank;
    if (law.symmetry_class.empty()) law.symmetry_class = fallback.symmetry_class;
    if (!law.model && !law.constant_value_available) law.model = fallback.model;
    law.state_dependent = law.state_dependent || fallback.state_dependent;
    law.time_dependent = law.time_dependent || fallback.time_dependent;
}

struct ScanState {
    FormExprScanResult result{};
    std::unordered_set<const forms::FormExprNode*> seen_constitutive_nodes{};
};

struct ScanContext {
    DomainKind domain{DomainKind::Cell};
    int boundary_marker{-1};
    int interface_marker{-1};
};

[[nodiscard]] bool sameParameterUsage(const FormParameterUsage& lhs,
                                      const FormParameterUsage& rhs) noexcept
{
    return lhs.name == rhs.name &&
           lhs.slot == rhs.slot &&
           lhs.domain == rhs.domain &&
           lhs.boundary_marker == rhs.boundary_marker &&
           lhs.interface_marker == rhs.interface_marker;
}

void addParameterUsageIfAbsent(std::vector<FormParameterUsage>& usages,
                               FormParameterUsage usage)
{
    for (const auto& existing : usages) {
        if (sameParameterUsage(existing, usage)) {
            return;
        }
    }
    usages.push_back(std::move(usage));
}

[[nodiscard]] bool sameCoefficientUsage(const FormCoefficientUsage& lhs,
                                        const FormCoefficientUsage& rhs) noexcept
{
    return lhs.name == rhs.name &&
           lhs.rank == rhs.rank &&
           lhs.time_dependent == rhs.time_dependent &&
           lhs.domain == rhs.domain &&
           lhs.boundary_marker == rhs.boundary_marker &&
           lhs.interface_marker == rhs.interface_marker;
}

void addCoefficientUsageIfAbsent(std::vector<FormCoefficientUsage>& usages,
                                 FormCoefficientUsage usage)
{
    if (usage.name.empty()) {
        return;
    }
    for (const auto& existing : usages) {
        if (sameCoefficientUsage(existing, usage)) {
            return;
        }
    }
    usages.push_back(std::move(usage));
}

[[nodiscard]] bool sameScaleUsage(const FormScaleUsage& lhs,
                                  const FormScaleUsage& rhs) noexcept
{
    return lhs.scale_id == rhs.scale_id &&
           lhs.domain == rhs.domain &&
           lhs.boundary_marker == rhs.boundary_marker &&
           lhs.interface_marker == rhs.interface_marker;
}

void addScaleUsageIfAbsent(std::vector<FormScaleUsage>& usages,
                           FormScaleUsage usage)
{
    if (usage.h_power == 0 &&
        usage.dt_power == 0 &&
        usage.parameter_names.empty() &&
        usage.parameter_slots.empty() &&
        usage.coefficient_names.empty()) {
        return;
    }

    std::sort(usage.parameter_names.begin(), usage.parameter_names.end());
    usage.parameter_names.erase(
        std::unique(usage.parameter_names.begin(), usage.parameter_names.end()),
        usage.parameter_names.end());
    std::sort(usage.parameter_slots.begin(), usage.parameter_slots.end());
    usage.parameter_slots.erase(
        std::unique(usage.parameter_slots.begin(), usage.parameter_slots.end()),
        usage.parameter_slots.end());
    std::sort(usage.coefficient_names.begin(), usage.coefficient_names.end());
    usage.coefficient_names.erase(
        std::unique(usage.coefficient_names.begin(), usage.coefficient_names.end()),
        usage.coefficient_names.end());

    if (usage.scale_id.empty()) {
        std::ostringstream id;
        id << "h^" << usage.h_power << ":dt^" << usage.dt_power;
        for (const auto& parameter : usage.parameter_names) {
            id << ":p=" << parameter;
        }
        for (auto slot : usage.parameter_slots) {
            id << ":slot=" << slot;
        }
        for (const auto& coefficient : usage.coefficient_names) {
            id << ":c=" << coefficient;
        }
        usage.scale_id = id.str();
    }

    for (const auto& existing : usages) {
        if (sameScaleUsage(existing, usage)) {
            return;
        }
    }
    usages.push_back(std::move(usage));
}

[[nodiscard]] FormCoefficientRank coefficientRankForNode(
    const forms::FormExprNode& node) noexcept
{
    if (node.scalarCoefficient() || node.timeScalarCoefficient()) {
        return FormCoefficientRank::Scalar;
    }
    if (node.vectorCoefficient()) {
        return FormCoefficientRank::Vector;
    }
    if (node.matrixCoefficient()) {
        return FormCoefficientRank::Rank2Tensor;
    }
    if (node.tensor3Coefficient()) {
        return FormCoefficientRank::Rank3Tensor;
    }
    if (node.tensor4Coefficient()) {
        return FormCoefficientRank::Rank4Tensor;
    }
    return FormCoefficientRank::Unknown;
}

struct ScaleFactors {
    int h_power{0};
    int dt_power{0};
    std::vector<std::string> parameter_names;
    std::vector<std::uint32_t> parameter_slots;
    std::vector<std::string> coefficient_names;
    bool exact{true};

    [[nodiscard]] bool empty() const noexcept
    {
        return h_power == 0 &&
               dt_power == 0 &&
               parameter_names.empty() &&
               parameter_slots.empty() &&
               coefficient_names.empty();
    }
};

void appendUnique(std::vector<std::string>& dst, std::string value)
{
    if (value.empty()) {
        return;
    }
    if (std::find(dst.begin(), dst.end(), value) == dst.end()) {
        dst.push_back(std::move(value));
    }
}

void appendUnique(std::vector<std::uint32_t>& dst, std::uint32_t value)
{
    if (std::find(dst.begin(), dst.end(), value) == dst.end()) {
        dst.push_back(value);
    }
}

void mergeScaleFactors(ScaleFactors& dst,
                       const ScaleFactors& src,
                       int sign = 1)
{
    const bool src_has_scale = !src.empty();
    dst.h_power += sign * src.h_power;
    dst.dt_power += sign * src.dt_power;
    if (src_has_scale) {
        dst.exact = dst.exact && src.exact;
    }
    for (const auto& name : src.parameter_names) {
        appendUnique(dst.parameter_names, name);
    }
    for (auto slot : src.parameter_slots) {
        appendUnique(dst.parameter_slots, slot);
    }
    for (const auto& name : src.coefficient_names) {
        appendUnique(dst.coefficient_names, name);
    }
}

[[nodiscard]] bool isIntegralExponent(Real value, int& exponent) noexcept
{
    const auto rounded = std::round(static_cast<double>(value));
    if (std::abs(static_cast<double>(value) - rounded) > 1.0e-12) {
        return false;
    }
    if (rounded < static_cast<double>(std::numeric_limits<int>::min()) ||
        rounded > static_cast<double>(std::numeric_limits<int>::max())) {
        return false;
    }
    exponent = static_cast<int>(rounded);
    return true;
}

[[nodiscard]] ScaleFactors extractScaleFactors(const forms::FormExprNode& node)
{
    using FT = forms::FormExprType;

    ScaleFactors factors;
    switch (node.type()) {
        case FT::ParameterSymbol:
            if (auto name = node.symbolName()) {
                appendUnique(factors.parameter_names, std::string(*name));
            }
            return factors;
        case FT::ParameterRef:
            if (auto slot = node.slotIndex()) {
                appendUnique(factors.parameter_slots, *slot);
            }
            return factors;
        case FT::Coefficient:
            appendUnique(factors.coefficient_names, node.toString());
            return factors;
        case FT::CellDiameter:
            factors.h_power = 1;
            return factors;
        case FT::TimeStep:
        case FT::EffectiveTimeStep:
            factors.dt_power = 1;
            return factors;
        case FT::Negate: {
            const auto children = node.childrenShared();
            if (children.size() == 1u && children[0]) {
                return extractScaleFactors(*children[0]);
            }
            break;
        }
        case FT::Multiply: {
            const auto children = node.childrenShared();
            if (children.size() == 2u && children[0] && children[1]) {
                auto lhs = extractScaleFactors(*children[0]);
                auto rhs = extractScaleFactors(*children[1]);
                mergeScaleFactors(factors, lhs);
                mergeScaleFactors(factors, rhs);
                return factors;
            }
            break;
        }
        case FT::Divide: {
            const auto children = node.childrenShared();
            if (children.size() == 2u && children[0] && children[1]) {
                auto lhs = extractScaleFactors(*children[0]);
                auto rhs = extractScaleFactors(*children[1]);
                mergeScaleFactors(factors, lhs);
                mergeScaleFactors(factors, rhs, -1);
                return factors;
            }
            break;
        }
        case FT::Power: {
            const auto children = node.childrenShared();
            if (children.size() == 2u && children[0] && children[1]) {
                auto base = extractScaleFactors(*children[0]);
                auto exponent_value = children[1]->constantValue();
                int exponent = 0;
                if (exponent_value && isIntegralExponent(*exponent_value, exponent)) {
                    base.h_power *= exponent;
                    base.dt_power *= exponent;
                    if (exponent != 1 &&
                        (!base.parameter_names.empty() ||
                         !base.parameter_slots.empty() ||
                         !base.coefficient_names.empty())) {
                        base.exact = false;
                    }
                    return base;
                }
                base.exact = false;
                return base;
            }
            break;
        }
        default:
            break;
    }
    factors.exact = false;
    return factors;
}

void collectScaleUsages(const forms::FormExprNode& node,
                        const ScanContext& context,
                        FormExprScanResult& result)
{
    using FT = forms::FormExprType;
    const auto type = node.type();
    const bool candidate =
        type == FT::ParameterSymbol ||
        type == FT::ParameterRef ||
        type == FT::Coefficient ||
        type == FT::CellDiameter ||
        type == FT::TimeStep ||
        type == FT::EffectiveTimeStep ||
        type == FT::Multiply ||
        type == FT::Divide ||
        type == FT::Power;

    if (candidate) {
        const auto factors = extractScaleFactors(node);
        if (!factors.empty()) {
            FormScaleUsage usage;
            usage.h_power = factors.h_power;
            usage.dt_power = factors.dt_power;
            usage.parameter_names = factors.parameter_names;
            usage.parameter_slots = factors.parameter_slots;
            usage.coefficient_names = factors.coefficient_names;
            usage.domain = context.domain;
            usage.boundary_marker = context.boundary_marker;
            usage.interface_marker = context.interface_marker;
            usage.exact_for_analysis = factors.exact;
            addScaleUsageIfAbsent(result.scale_usages, std::move(usage));
        }
    }

    ScanContext child_context = context;
    if (type == FT::CellIntegral) {
        child_context.domain = DomainKind::Cell;
        child_context.boundary_marker = -1;
        child_context.interface_marker = -1;
    } else if (type == FT::BoundaryIntegral) {
        child_context.domain = DomainKind::Boundary;
        child_context.boundary_marker = node.boundaryMarker().value_or(-1);
        child_context.interface_marker = -1;
    } else if (type == FT::InteriorFaceIntegral) {
        child_context.domain = DomainKind::InteriorFace;
        child_context.boundary_marker = -1;
        child_context.interface_marker = -1;
    } else if (type == FT::InterfaceIntegral) {
        child_context.domain = DomainKind::InterfaceFace;
        child_context.boundary_marker = -1;
        child_context.interface_marker = node.interfaceMarker().value_or(-1);
    }

    for (const auto& child : node.childrenShared()) {
        if (child) {
            collectScaleUsages(*child, child_context, result);
        }
    }
}

void scanConstitutiveNode(const forms::FormExprNode& node, ScanState& state)
{
    if (!state.seen_constitutive_nodes.insert(&node).second) {
        return;
    }

    auto model = node.constitutiveModelShared();
    if (!model) {
        return;
    }

    const auto primary_field = inferConstitutivePrimaryField(node);
    const auto n_outputs = model->outputCount();
    for (std::size_t output = 0u; output < n_outputs; ++output) {
        auto fallback = inferGenericConstitutiveLawMetadata(model, output, primary_field);
        auto law = model->constitutiveLawMetadata(output);
        if (!law) {
            law = std::move(fallback);
        } else {
            fillConstitutiveMetadataDefaults(*law, fallback);
        }
        if (law->primary_field == INVALID_FIELD_ID &&
            primary_field != INVALID_FIELD_ID) {
            law->primary_field = primary_field;
        }
        if (!law->model && !law->constant_value_available) {
            law->model = model;
        }
        addConstitutiveLawIfAbsent(state.result.constitutive_laws,
                                   std::move(*law));
    }
}

void scanNode(const forms::FormExprNode& node,
              ScanState& state,
              const ScanContext& context) {
    using FT = forms::FormExprType;
    auto& result = state.result;
    ScanContext child_context = context;

    switch (node.type()) {
        case FT::TimeDerivative:
            result.has_time_derivative = true;
            break;
        case FT::CellDiameter:
            result.has_cell_diameter = true;
            break;
        case FT::Jump:
            result.has_jump = true;
            break;
        case FT::Average:
            result.has_average = true;
            break;
        case FT::CellIntegral:
            result.has_cell_integral = true;
            child_context.domain = DomainKind::Cell;
            child_context.boundary_marker = -1;
            child_context.interface_marker = -1;
            break;
        case FT::BoundaryIntegral: {
            result.has_boundary_integral = true;
            auto marker = node.boundaryMarker();
            child_context.domain = DomainKind::Boundary;
            child_context.boundary_marker = marker.value_or(-1);
            child_context.interface_marker = -1;
            if (marker && *marker >= 0) {
                if (std::find(result.boundary_markers.begin(),
                              result.boundary_markers.end(), *marker)
                    == result.boundary_markers.end()) {
                    result.boundary_markers.push_back(*marker);
                }
            }
            break;
        }
        case FT::InteriorFaceIntegral:
            result.has_interior_face_integral = true;
            child_context.domain = DomainKind::InteriorFace;
            child_context.boundary_marker = -1;
            child_context.interface_marker = -1;
            break;
        case FT::InterfaceIntegral: {
            result.has_interface_integral = true;
            auto marker = node.interfaceMarker();
            child_context.domain = DomainKind::InterfaceFace;
            child_context.boundary_marker = -1;
            child_context.interface_marker = marker.value_or(-1);
            if (marker && *marker >= 0) {
                if (std::find(result.interface_markers.begin(),
                              result.interface_markers.end(), *marker)
                    == result.interface_markers.end()) {
                    result.interface_markers.push_back(*marker);
                }
            }
            break;
        }
        case FT::BoundaryIntegralSymbol: {
            auto name = node.symbolName();
            if (name) {
                std::string s{*name};
                if (std::find(result.boundary_functional_names.begin(),
                              result.boundary_functional_names.end(), s)
                    == result.boundary_functional_names.end()) {
                    result.boundary_functional_names.push_back(std::move(s));
                }
            }
            break;
        }
        case FT::AuxiliaryStateSymbol: {
            auto name = node.symbolName();
            if (name) {
                std::string s{*name};
                if (std::find(result.auxiliary_state_names.begin(),
                              result.auxiliary_state_names.end(), s)
                    == result.auxiliary_state_names.end()) {
                    result.auxiliary_state_names.push_back(std::move(s));
                }
            }
            break;
        }
        case FT::AuxiliaryInputSymbol: {
            auto name = node.symbolName();
            if (name) {
                std::string s{*name};
                if (std::find(result.auxiliary_input_names.begin(),
                              result.auxiliary_input_names.end(), s)
                    == result.auxiliary_input_names.end()) {
                    result.auxiliary_input_names.push_back(std::move(s));
                }
            }
            break;
        }
        case FT::AuxiliaryOutputSymbol: {
            auto name = node.symbolName();
            if (name) {
                std::string s{*name};
                if (std::find(result.auxiliary_output_names.begin(),
                              result.auxiliary_output_names.end(), s)
                    == result.auxiliary_output_names.end()) {
                    result.auxiliary_output_names.push_back(std::move(s));
                }
            }
            break;
        }
        case FT::Constitutive:
            scanConstitutiveNode(node, state);
            break;
        case FT::ParameterSymbol:
        case FT::ParameterRef: {
            FormParameterUsage usage;
            if (auto name = node.symbolName()) {
                usage.name = std::string(*name);
            }
            usage.slot = node.slotIndex();
            usage.domain = context.domain;
            usage.boundary_marker = context.boundary_marker;
            usage.interface_marker = context.interface_marker;
            addParameterUsageIfAbsent(result.parameter_usages, std::move(usage));
            break;
        }
        case FT::Coefficient: {
            FormCoefficientUsage usage;
            usage.name = node.toString();
            usage.rank = coefficientRankForNode(node);
            usage.time_dependent = node.timeScalarCoefficient() != nullptr;
            usage.domain = context.domain;
            usage.boundary_marker = context.boundary_marker;
            usage.interface_marker = context.interface_marker;
            addCoefficientUsageIfAbsent(result.coefficient_usages,
                                        std::move(usage));
            break;
        }
        default:
            break;
    }

    for (const auto& child : node.childrenShared()) {
        if (child) {
            scanNode(*child, state, child_context);
        }
    }
}

} // namespace

FormExprScanResult scanFormExpr(const forms::FormExprNode& root) {
    ScanState state;
    ScanContext context;
    scanNode(root, state, context);
    collectScaleUsages(root, context, state.result);
    return std::move(state.result);
}

std::vector<DomainKind> FormExprScanResult::activeDomains() const {
    std::vector<DomainKind> domains;
    if (has_cell_integral) domains.push_back(DomainKind::Cell);
    if (has_boundary_integral) domains.push_back(DomainKind::Boundary);
    if (has_interior_face_integral) domains.push_back(DomainKind::InteriorFace);
    if (has_interface_integral) domains.push_back(DomainKind::InterfaceFace);
    if (!boundary_functional_names.empty() || !auxiliary_state_names.empty()) {
        // Presence of coupled-boundary symbols implies a coupled boundary domain
        if (std::find(domains.begin(), domains.end(), DomainKind::CoupledBoundary)
            == domains.end()) {
            domains.push_back(DomainKind::CoupledBoundary);
        }
    }
    // If nothing was detected (e.g., no explicit integral wrappers), default to Cell
    if (domains.empty()) domains.push_back(DomainKind::Cell);
    return domains;
}

} // namespace analysis
} // namespace FE
} // namespace svmp
