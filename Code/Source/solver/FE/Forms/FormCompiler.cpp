/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#include "Forms/FormCompiler.h"

#include "Core/FEException.h"
#include "Forms/BlockForm.h"
#include "Forms/MixedFormIR.h"
#include "Forms/PointEvaluator.h"

#include <algorithm>
#include <cmath>
#include <chrono>
#include <sstream>
#include <stdexcept>
#include <unordered_map>

namespace svmp {
namespace FE {
namespace forms {

struct FormCompiler::Impl {
    SymbolicOptions options{};
};

FormCompiler::FormCompiler()
    : impl_(std::make_unique<Impl>())
{
}

FormCompiler::FormCompiler(SymbolicOptions options)
    : impl_(std::make_unique<Impl>())
{
    impl_->options = std::move(options);
}

FormCompiler::~FormCompiler() = default;

FormCompiler::FormCompiler(FormCompiler&&) noexcept = default;
FormCompiler& FormCompiler::operator=(FormCompiler&&) noexcept = default;

void FormCompiler::setOptions(SymbolicOptions options)
{
    impl_->options = std::move(options);
}

const SymbolicOptions& FormCompiler::options() const noexcept
{
    return impl_->options;
}

namespace detail {

FormExpr makeExprFromNode(const std::shared_ptr<FormExprNode>& node)
{
    return FormExpr(node);
}

void requireNoIndexedAccess(const FormExprNode& node)
{
    const auto visit = [&](const auto& self, const FormExprNode& n) -> void {
        if (n.type() == FormExprType::IndexedAccess) {
            throw std::invalid_argument(
                "FormCompiler: detected indexed access (Einstein notation). "
                "Call forms::einsum(expr) to lower it before compilation.");
        }
        for (const auto& child : n.childrenShared()) {
            if (child) self(self, *child);
        }
    };
    visit(visit, node);
}

void requireNoCoupledPlaceholders(const FormExprNode& node)
{
    const auto visit = [&](const auto& self, const FormExprNode& n) -> void {
        if (n.type() == FormExprType::BoundaryFunctionalSymbol ||
            n.type() == FormExprType::BoundaryIntegralSymbol ||
            n.type() == FormExprType::AuxiliaryStateSymbol ||
            n.type() == FormExprType::AuxiliaryInputSymbol ||
            n.type() == FormExprType::AuxiliaryOutputSymbol) {
            throw std::invalid_argument(
                "FormCompiler: detected unresolved placeholder terminal (" +
                n.toString() + "). "
                "Resolve coupled/auxiliary expressions to slot references before compilation.");
        }
        for (const auto& child : n.childrenShared()) {
            if (child) self(self, *child);
        }
    };
    visit(visit, node);
}

bool spaceSignatureEqual(const FormExprNode::SpaceSignature& a,
                         const FormExprNode::SpaceSignature& b) noexcept
{
    return a.space_type == b.space_type && a.field_type == b.field_type && a.continuity == b.continuity &&
           a.value_dimension == b.value_dimension && a.topological_dimension == b.topological_dimension &&
           a.polynomial_order == b.polynomial_order && a.element_type == b.element_type;
}

struct BoundArgumentInfo {
    std::optional<FormExprNode::SpaceSignature> test_space{};
    std::optional<FormExprNode::SpaceSignature> trial_space{};
    std::optional<std::string> test_name{};
    std::optional<std::string> trial_name{};
};

BoundArgumentInfo analyzeBoundArguments(const FormExprNode& node)
{
    BoundArgumentInfo info;

    const auto visit = [&](const auto& self, const FormExprNode& n) -> void {
        if (n.type() == FormExprType::TestFunction) {
            const auto* sig = n.spaceSignature();
            if (!sig) {
                throw std::invalid_argument("FormCompiler: TestFunction must be bound to a FunctionSpace");
            }
            if (!info.test_space) {
                info.test_space = *sig;
                info.test_name = n.toString();
            } else {
                if (!spaceSignatureEqual(*info.test_space, *sig)) {
                    throw std::invalid_argument(
                        "FormCompiler: multiple TestFunction spaces detected (mixed/multi-field not implemented)");
                }
                const auto nm = n.toString();
                if (info.test_name && nm != *info.test_name) {
                    throw std::invalid_argument(
                        "FormCompiler: multiple TestFunction symbols detected (mixed/multi-field not implemented)");
                }
            }
        }

        if (n.type() == FormExprType::TrialFunction) {
            const auto* sig = n.spaceSignature();
            if (!sig) {
                throw std::invalid_argument("FormCompiler: TrialFunction must be bound to a FunctionSpace");
            }
            if (!info.trial_space) {
                info.trial_space = *sig;
                info.trial_name = n.toString();
            } else {
                if (!spaceSignatureEqual(*info.trial_space, *sig)) {
                    throw std::invalid_argument(
                        "FormCompiler: multiple TrialFunction spaces detected (mixed/multi-field not implemented)");
                }
                const auto nm = n.toString();
                if (info.trial_name && nm != *info.trial_name) {
                    throw std::invalid_argument(
                        "FormCompiler: multiple TrialFunction symbols detected (mixed/multi-field not implemented)");
                }
            }
        }

        for (const auto& child : n.childrenShared()) {
            if (child) self(self, *child);
        }
    };

    visit(visit, node);
    return info;
}

struct MixedBoundArgumentInfo {
    // Unique test spaces and their names, in order of first occurrence
    std::vector<std::pair<FormExprNode::SpaceSignature, std::string>> test_spaces{};
    // Unique trial spaces and their names, in order of first occurrence
    std::vector<std::pair<FormExprNode::SpaceSignature, std::string>> trial_spaces{};
};

MixedBoundArgumentInfo analyzeMixedBoundArguments(const FormExprNode& node)
{
    MixedBoundArgumentInfo info;

    const auto find_or_add = [](
        std::vector<std::pair<FormExprNode::SpaceSignature, std::string>>& spaces,
        const FormExprNode::SpaceSignature& sig,
        const std::string& name) -> std::size_t
    {
        for (std::size_t i = 0; i < spaces.size(); ++i) {
            if (spaceSignatureEqual(spaces[i].first, sig) && spaces[i].second == name) {
                return i;
            }
        }
        spaces.push_back({sig, name});
        return spaces.size() - 1;
    };

    const auto visit = [&](const auto& self, const FormExprNode& n) -> void {
        if (n.type() == FormExprType::TestFunction) {
            const auto* sig = n.spaceSignature();
            if (sig) {
                find_or_add(info.test_spaces, *sig, n.toString());
            }
        }
        if (n.type() == FormExprType::TrialFunction) {
            const auto* sig = n.spaceSignature();
            if (sig) {
                find_or_add(info.trial_spaces, *sig, n.toString());
            }
        }
        for (const auto& child : n.childrenShared()) {
            if (child) self(self, *child);
        }
    };

    visit(visit, node);

    // Reject duplicate names across different spaces. If two test (or trial)
    // functions share the same name but have different space signatures, the
    // name-based block classifier cannot reliably distinguish them.
    for (std::size_t i = 0; i < info.test_spaces.size(); ++i) {
        for (std::size_t j = i + 1; j < info.test_spaces.size(); ++j) {
            if (info.test_spaces[i].second == info.test_spaces[j].second &&
                !spaceSignatureEqual(info.test_spaces[i].first, info.test_spaces[j].first)) {
                throw std::invalid_argument(
                    "FormCompiler::compileMixed: duplicate TestFunction name '" +
                    info.test_spaces[i].second +
                    "' used with different spaces — use distinct names for each field");
            }
        }
    }
    for (std::size_t i = 0; i < info.trial_spaces.size(); ++i) {
        for (std::size_t j = i + 1; j < info.trial_spaces.size(); ++j) {
            if (info.trial_spaces[i].second == info.trial_spaces[j].second &&
                !spaceSignatureEqual(info.trial_spaces[i].first, info.trial_spaces[j].first)) {
                throw std::invalid_argument(
                    "FormCompiler::compileMixed: duplicate TrialFunction name '" +
                    info.trial_spaces[i].second +
                    "' used with different spaces — use distinct names for each field");
            }
        }
    }

    return info;
}

bool integrandContainsTestNamed(const FormExprNode& node, const std::string& test_name)
{
    if (node.type() == FormExprType::TestFunction && node.toString() == test_name) {
        return true;
    }
    for (const auto& child : node.childrenShared()) {
        if (child && integrandContainsTestNamed(*child, test_name)) return true;
    }
    return false;
}

bool integrandContainsTrialNamed(const FormExprNode& node, const std::string& trial_name)
{
    if (node.type() == FormExprType::TrialFunction && node.toString() == trial_name) {
        return true;
    }
    for (const auto& child : node.childrenShared()) {
        if (child && integrandContainsTrialNamed(*child, trial_name)) return true;
    }
    return false;
}

bool isGeometrySensitivityTerminal(FormExprType type) noexcept
{
    switch (type) {
        case FormExprType::CurrentCoordinate:
        case FormExprType::CurrentJacobian:
        case FormExprType::CurrentJacobianDeterminant:
        case FormExprType::CurrentNormal:
        case FormExprType::CurrentMeasure:
        case FormExprType::SurfaceJacobian:
        case FormExprType::MeshVelocity:
        case FormExprType::GeometryTrialVectorVariation:
        case FormExprType::GeometryTrialJacobianVariation:
        case FormExprType::MeshVelocityVariation:
        case FormExprType::CurrentMeasureVariation:
        case FormExprType::CurrentNormalVariation:
        case FormExprType::SurfaceJacobianVariation:
        case FormExprType::Pullback:
        case FormExprType::Pushforward:
            return true;
        default:
            return false;
    }
}

bool containsGeometrySensitivityTerminal(const FormExprNode& node)
{
    if (isGeometrySensitivityTerminal(node.type())) {
        return true;
    }
    for (const auto& child : node.childrenShared()) {
        if (child && containsGeometrySensitivityTerminal(*child)) {
            return true;
        }
    }
    return false;
}

void requireSupportedGeometrySensitivityMode(const FormExprNode& node,
                                             const GeometrySensitivityOptions& options)
{
    if (options.mode == GeometrySensitivityMode::GeometryConstant) {
        return;
    }

    if (options.mesh_motion_field == INVALID_FIELD_ID ||
        options.mesh_motion_field == GEOMETRY_FIELD_ID ||
        options.mesh_motion_field == CURRENT_SOLUTION_FIELD_ID) {
        throw std::invalid_argument(
            "FormCompiler: geometry sensitivity mode requires an explicit mesh-motion FieldId");
    }

    if (containsGeometrySensitivityTerminal(node)) {
        return;
    }
}

assembly::RequiredData analyzeRequiredData(const FormExprNode& node, FormKind kind)
{
    using assembly::RequiredData;

    RequiredData required = RequiredData::None;

    const auto visit = [&](const auto& self, const FormExprNode& n, int order) -> void {
        order = std::clamp(order, 0, 2);

        switch (n.type()) {
            case FormExprType::TestFunction:
                required |= RequiredData::BasisValues;
                if (order >= 1) required |= RequiredData::PhysicalGradients;
                if (order >= 2) required |= RequiredData::BasisHessians;
                break;
            case FormExprType::TrialFunction:
                if (kind == FormKind::Residual) {
                    required |= RequiredData::SolutionValues;
                    required |= RequiredData::BasisValues; // Needed for AD seeding via trialBasisValue().
                    if (order >= 1) {
                        required |= RequiredData::SolutionGradients;
                        required |= RequiredData::PhysicalGradients; // Needed for AD seeding via trialPhysicalGradient().
                    }
                    if (order >= 2) {
                        required |= RequiredData::SolutionHessians;
                        required |= RequiredData::BasisHessians; // Needed for AD seeding via trialPhysicalHessian().
                    }
                } else {
                    required |= RequiredData::BasisValues;
                    if (order >= 1) required |= RequiredData::PhysicalGradients;
                    if (order >= 2) required |= RequiredData::BasisHessians;
                }
                break;
            case FormExprType::PreviousSolutionRef:
                required |= RequiredData::BasisValues;
                if (order >= 1) required |= RequiredData::PhysicalGradients;
                if (order >= 2) required |= RequiredData::BasisHessians;
                break;
            case FormExprType::Coefficient:
            case FormExprType::Coordinate:
                required |= RequiredData::PhysicalPoints;
                break;
            case FormExprType::ReferenceCoordinate:
                required |= RequiredData::QuadraturePoints;
                break;
            case FormExprType::MeshDisplacement:
                required |= RequiredData::MeshDisplacement;
                if (order >= 1) required |= RequiredData::MeshDisplacementGradient;
                break;
            case FormExprType::MeshVelocity:
                required |= RequiredData::MeshVelocity;
                if (order >= 1) required |= RequiredData::MeshVelocityGradient;
                break;
            case FormExprType::MeshAcceleration:
                required |= RequiredData::MeshAcceleration;
                if (order >= 1) required |= RequiredData::MeshAccelerationGradient;
                break;
            case FormExprType::CurrentCoordinate:
                required |= RequiredData::CurrentPhysicalPoints;
                break;
            case FormExprType::PreviousCoordinate:
                required |= RequiredData::PreviousPhysicalPoints;
                break;
            case FormExprType::ReferencePhysicalCoordinate:
                required |= RequiredData::ReferencePhysicalPoints;
                break;
            case FormExprType::PreviousMeshVelocity:
                required |= RequiredData::PreviousMeshVelocity;
                if (order >= 1) required |= RequiredData::PreviousMeshVelocityGradient;
                break;
            case FormExprType::PredictedMeshVelocity:
                required |= RequiredData::PredictedMeshVelocity;
                if (order >= 1) required |= RequiredData::PredictedMeshVelocityGradient;
                break;
            case FormExprType::CurrentJacobian:
                required |= RequiredData::CurrentJacobians;
                break;
            case FormExprType::ReferenceJacobian:
                required |= RequiredData::ReferenceJacobians;
                break;
            case FormExprType::CurrentJacobianDeterminant:
                required |= RequiredData::CurrentJacobians;
                break;
            case FormExprType::CurrentMeasure:
                required |= RequiredData::CurrentMeasures;
                break;
            case FormExprType::ReferenceJacobianDeterminant:
                required |= RequiredData::ReferenceJacobians;
                break;
            case FormExprType::ReferenceMeasure:
                required |= RequiredData::ReferenceMeasures;
                break;
            case FormExprType::CurrentNormal:
                required |= RequiredData::CurrentNormals;
                break;
            case FormExprType::ReferenceNormal:
                required |= RequiredData::ReferenceNormals;
                break;
            case FormExprType::SurfaceJacobian:
                required |= RequiredData::SurfaceJacobians;
                break;
            case FormExprType::GeometryTrialVectorVariation:
            case FormExprType::MeshVelocityVariation:
                required |= RequiredData::BasisValues;
                required |= RequiredData::PhysicalGradients;
                break;
            case FormExprType::GeometryTrialJacobianVariation:
            case FormExprType::CurrentMeasureVariation:
                required |= RequiredData::BasisValues;
                required |= RequiredData::BasisGradients;
                required |= RequiredData::PhysicalGradients;
                required |= RequiredData::CurrentJacobians;
                required |= RequiredData::CurrentMeasures;
                break;
            case FormExprType::CurrentNormalVariation:
            case FormExprType::SurfaceJacobianVariation:
                required |= RequiredData::BasisValues;
                required |= RequiredData::BasisGradients;
                required |= RequiredData::PhysicalGradients;
                required |= RequiredData::CurrentJacobians;
                required |= RequiredData::CurrentMeasures;
                required |= RequiredData::CurrentNormals;
                required |= RequiredData::SurfaceJacobians;
                break;
            case FormExprType::Jacobian:
                required |= RequiredData::Jacobians;
                break;
            case FormExprType::JacobianInverse:
                required |= RequiredData::InverseJacobians;
                break;
            case FormExprType::JacobianDeterminant:
                required |= RequiredData::JacobianDets;
                break;
            case FormExprType::Normal:
                required |= RequiredData::Normals;
                break;
            case FormExprType::CellDiameter:
            case FormExprType::CellVolume:
            case FormExprType::FacetArea:
                required |= RequiredData::EntityMeasures;
                break;
            case FormExprType::MaterialStateOldRef:
            case FormExprType::MaterialStateWorkRef:
                required |= RequiredData::MaterialState;
                break;
            case FormExprType::Pullback:
            case FormExprType::Pushforward:
                required |= RequiredData::ConfigurationTransforms;
                break;
            case FormExprType::StateField: {
                // StateField usually refers to an auxiliary/state variable provided as
                // field solution data (handled via FieldRequirements). We also support
                // a special sentinel FieldId (CURRENT_SOLUTION_FIELD_ID) to represent the
                // current solution state u in symbolic tangent forms.
                const auto fid = n.fieldId();
                if (fid && *fid == CURRENT_SOLUTION_FIELD_ID) {
                    required |= RequiredData::SolutionValues;
                    required |= RequiredData::BasisValues;
                    if (order >= 1) {
                        required |= RequiredData::SolutionGradients;
                        required |= RequiredData::PhysicalGradients;
                    }
                    if (order >= 2) {
                        required |= RequiredData::SolutionHessians;
                        required |= RequiredData::BasisHessians;
                    }
                }
                break;
            }
            case FormExprType::Gradient:
            {
                const auto kids = n.childrenShared();
                if (!kids.empty() && kids[0]) self(self, *kids[0], order + 1);
                return;
            }
            case FormExprType::Divergence:
            case FormExprType::Curl: {
                const auto kids = n.childrenShared();
                if (kids.empty() || !kids[0]) {
                    return;
                }

                // H(curl)/H(div) vector bases provide curl/div directly via basis evaluation; do not
                // require PhysicalGradients/SolutionGradients for these operators.
                if (order == 0) {
                    if (const auto* sig = kids[0]->spaceSignature()) {
                        const bool is_vector_basis_hcurl =
                            (n.type() == FormExprType::Curl &&
                             sig->field_type == FieldType::Vector &&
                             sig->continuity == Continuity::H_curl);
                        const bool is_vector_basis_hdiv =
                            (n.type() == FormExprType::Divergence &&
                             sig->field_type == FieldType::Vector &&
                             sig->continuity == Continuity::H_div);

                        if (is_vector_basis_hcurl) {
                            required |= RequiredData::BasisCurls;
                            self(self, *kids[0], 0);
                            return;
                        }
                        if (is_vector_basis_hdiv) {
                            required |= RequiredData::BasisDivergences;
                            self(self, *kids[0], 0);
                            return;
                        }
                    }
                }

                // Fallback: treat as a first-order spatial operator (component-wise vectors).
                self(self, *kids[0], order + 1);
                return;
            }
            case FormExprType::Hessian: {
                const auto kids = n.childrenShared();
                if (!kids.empty() && kids[0]) self(self, *kids[0], order + 2);
                return;
            }
            case FormExprType::Conditional: {
                const auto kids = n.childrenShared();
                if (kids.size() != 3u || !kids[0] || !kids[1] || !kids[2]) {
                    throw std::invalid_argument("FormCompiler: conditional expects 3 operands");
                }
                self(self, *kids[0], 0);
                self(self, *kids[1], order);
                self(self, *kids[2], order);
                return;
            }
            case FormExprType::Less:
            case FormExprType::LessEqual:
            case FormExprType::Greater:
            case FormExprType::GreaterEqual:
            case FormExprType::Equal:
            case FormExprType::NotEqual: {
                const auto kids = n.childrenShared();
                if (kids.size() != 2u || !kids[0] || !kids[1]) {
                    throw std::invalid_argument("FormCompiler: comparison expects 2 operands");
                }
                self(self, *kids[0], 0);
                self(self, *kids[1], 0);
                return;
            }
            default:
                break;
        }

        for (const auto& child : n.childrenShared()) {
            if (child) self(self, *child, order);
        }
    };

    visit(visit, node, 0);

    // Always need quadrature/integration weights for any integral evaluation.
    required |= RequiredData::IntegrationWeights;
    return required;
}

std::vector<assembly::FieldRequirement> analyzeFieldRequirements(const FormExprNode& node)
{
    using assembly::FieldRequirement;
    using assembly::RequiredData;

    std::unordered_map<FieldId, RequiredData> req_by_field;

    const auto add = [&](FieldId id, RequiredData bits) {
        if (id == INVALID_FIELD_ID) {
            throw std::invalid_argument("FormCompiler: DiscreteField node has invalid FieldId");
        }
        req_by_field[id] |= bits;
    };

    const auto visit = [&](const auto& self, const FormExprNode& n, int order) -> void {
        order = std::clamp(order, 0, 2);
        switch (n.type()) {
            case FormExprType::DiscreteField:
            case FormExprType::StateField: {
                const auto fid = n.fieldId();
                if (!fid) {
                    throw std::logic_error("FormCompiler: DiscreteField/StateField node missing fieldId()");
                }
                if (n.type() == FormExprType::StateField && *fid == CURRENT_SOLUTION_FIELD_ID) {
                    // Special-case: StateField(CURRENT_SOLUTION_FIELD_ID) represents the current
                    // solution state (u), not an external field entry.
                    break;
                }
                RequiredData bits = RequiredData::SolutionValues;
                if (order >= 1) bits |= RequiredData::SolutionGradients;
                if (order >= 2) bits |= RequiredData::SolutionHessians;
                add(*fid, bits);
                break;
            }
            case FormExprType::Gradient:
            case FormExprType::Divergence:
            case FormExprType::Curl: {
                const auto kids = n.childrenShared();
                if (!kids.empty() && kids[0]) self(self, *kids[0], order + 1);
                return;
            }
            case FormExprType::Hessian: {
                const auto kids = n.childrenShared();
                if (!kids.empty() && kids[0]) self(self, *kids[0], order + 2);
                return;
            }
            case FormExprType::Conditional: {
                const auto kids = n.childrenShared();
                if (kids.size() != 3u || !kids[0] || !kids[1] || !kids[2]) {
                    throw std::invalid_argument("FormCompiler: conditional expects 3 operands");
                }
                self(self, *kids[0], 0);
                self(self, *kids[1], order);
                self(self, *kids[2], order);
                return;
            }
            case FormExprType::Less:
            case FormExprType::LessEqual:
            case FormExprType::Greater:
            case FormExprType::GreaterEqual:
            case FormExprType::Equal:
            case FormExprType::NotEqual: {
                const auto kids = n.childrenShared();
                if (kids.size() != 2u || !kids[0] || !kids[1]) {
                    throw std::invalid_argument("FormCompiler: comparison expects 2 operands");
                }
                self(self, *kids[0], 0);
                self(self, *kids[1], 0);
                return;
            }
            default:
                break;
        }

        for (const auto& child : n.childrenShared()) {
            if (child) self(self, *child, order);
        }
    };

    visit(visit, node, 0);

    std::vector<FieldRequirement> out;
    out.reserve(req_by_field.size());
    for (const auto& kv : req_by_field) {
        out.push_back(FieldRequirement{kv.first, kv.second});
    }
    std::sort(out.begin(), out.end(),
              [](const FieldRequirement& a, const FieldRequirement& b) { return a.field < b.field; });
    return out;
}

int analyzeTimeDerivativeOrder(const FormExprNode& node, FormKind kind)
{
    int max_order = 0;
    int dt_count = 0;
    std::optional<int> dt_order{};

    const auto visit = [&](const auto& self, const FormExprNode& n) -> void {
        if (n.type() == FormExprType::TimeDerivative) {
            ++dt_count;
            const int order = n.timeDerivativeOrder().value_or(1);
            if (order <= 0) {
                throw std::invalid_argument("FormCompiler: dt(·,k) requires k >= 1");
            }

            if (!dt_order) {
                dt_order = order;
            } else if (*dt_order != order) {
                throw std::invalid_argument("FormCompiler: multiple dt() orders in one term are not supported");
            }
            max_order = std::max(max_order, order);

            const auto kids = n.childrenShared();
            if (kids.size() != 1 || !kids[0]) {
                throw std::invalid_argument("FormCompiler: dt(·,k) must have exactly 1 operand");
            }
            const auto& operand = *kids[0];
            const auto operand_type = operand.type();

            bool ok = false;
            if (operand_type == FormExprType::TrialFunction) {
                ok = true;
            } else if (operand_type == FormExprType::DiscreteField || operand_type == FormExprType::StateField) {
                // dt(field) is treated as a coefficient value and includes history contributions.
                ok = true;
            }

            if (!ok) {
                throw std::invalid_argument(
                    "FormCompiler: dt(·,k) currently supports TrialFunction and DiscreteField/StateField operands only");
            }
        }

        for (const auto& child : n.childrenShared()) {
            if (child) self(self, *child);
        }
    };

    visit(visit, node);

    // For bilinear forms, dt(TrialFunction) must remain affine:
    // allow multiple dt(TrialFunction) occurrences in *additive* contexts, but reject
    // dt(TrialFunction) multiplied by dt(TrialFunction) (or any expression that can expand
    // to such a product).
    if (kind == FormKind::Bilinear && dt_count > 1) {
        std::unordered_map<const FormExprNode*, int> memo;

        const auto dt_trial_degree = [&](const auto& self, const FormExprNode& n) -> int {
            if (auto it = memo.find(&n); it != memo.end()) {
                return it->second;
            }

            const auto kids = n.childrenShared();
            auto child_degree = [&](std::size_t k) -> int {
                if (kids.size() <= k || !kids[k]) {
                    return 0;
                }
                return self(self, *kids[k]);
            };

            int deg = 0;
            switch (n.type()) {
                case FormExprType::TimeDerivative: {
                    // dt(...) nodes are only allowed on TrialFunction and DiscreteField/StateField (validated above).
                    if (kids.size() == 1u && kids[0] && kids[0]->type() == FormExprType::TrialFunction) {
                        deg = 1;
                    }
                    break;
                }

                // Sum-like nodes: degree is the max over branches.
                case FormExprType::Add:
                case FormExprType::Subtract:
                    deg = std::max(child_degree(0), child_degree(1));
                    break;
                case FormExprType::Negate:
                    deg = child_degree(0);
                    break;
                case FormExprType::Conditional:
                    deg = std::max({child_degree(0), child_degree(1), child_degree(2)});
                    break;
                case FormExprType::Minimum:
                case FormExprType::Maximum:
                    deg = std::max(child_degree(0), child_degree(1));
                    break;
                case FormExprType::SmoothMin:
                case FormExprType::SmoothMax:
                    deg = std::max({child_degree(0), child_degree(1), child_degree(2)});
                    break;

                // Product-like nodes: degree adds (distributivity means degrees combine).
                case FormExprType::Multiply:
                case FormExprType::Divide:
                case FormExprType::InnerProduct:
                case FormExprType::DoubleContraction:
                case FormExprType::OuterProduct:
                case FormExprType::CrossProduct:
                    deg = child_degree(0) + child_degree(1);
                    break;

                case FormExprType::Power: {
                    // Conservative: allow pow(x,1) to preserve degree, reject higher integer powers.
                    const int a = child_degree(0);
                    const int b = child_degree(1);
                    if (a == 0 && b == 0) {
                        deg = 0;
                        break;
                    }
                    if (b == 0 && kids.size() == 2u && kids[1] && kids[1]->type() == FormExprType::Constant) {
                        const auto exp = kids[1]->constantValue();
                        if (exp.has_value()) {
                            const Real e = *exp;
                            const Real ei = std::round(e);
                            if (std::abs(e - ei) < 1e-12 && ei >= 0.0 && ei <= 100.0) {
                                deg = a * static_cast<int>(ei);
                                break;
                            }
                        }
                    }
                    // Non-integer or variable exponent: treat as nonlinear in dt(TrialFunction).
                    deg = (a > 0 || b > 0) ? 2 : 0;
                    break;
                }

                // Constructors: degree is max over components.
                case FormExprType::AsVector:
                case FormExprType::AsTensor: {
                    int mx = 0;
                    for (const auto& k : kids) {
                        if (!k) continue;
                        mx = std::max(mx, self(self, *k));
                    }
                    deg = mx;
                    break;
                }

                // Linear/unary transforms: propagate child degree.
                case FormExprType::Gradient:
                case FormExprType::Divergence:
                case FormExprType::Curl:
                case FormExprType::Hessian:
                case FormExprType::RestrictMinus:
                case FormExprType::RestrictPlus:
                case FormExprType::Jump:
                case FormExprType::Average:
                case FormExprType::Pullback:
                case FormExprType::Pushforward:
                case FormExprType::Component:
                case FormExprType::IndexedAccess:
                case FormExprType::Transpose:
                case FormExprType::Trace:
                case FormExprType::Determinant:
                case FormExprType::Inverse:
                case FormExprType::Cofactor:
                case FormExprType::Deviator:
                case FormExprType::SymmetricPart:
                case FormExprType::SkewPart:
                case FormExprType::Norm:
                case FormExprType::Normalize:
                case FormExprType::AbsoluteValue:
                case FormExprType::Sign:
                case FormExprType::Sqrt:
                case FormExprType::Exp:
                case FormExprType::Log:
                case FormExprType::MatrixExponential:
                case FormExprType::MatrixLogarithm:
                case FormExprType::MatrixSqrt:
                case FormExprType::MatrixPower:
                case FormExprType::MatrixExponentialDirectionalDerivative:
                case FormExprType::MatrixLogarithmDirectionalDerivative:
                case FormExprType::MatrixSqrtDirectionalDerivative:
                case FormExprType::MatrixPowerDirectionalDerivative:
                case FormExprType::HistoryWeightedSum:
                case FormExprType::HistoryConvolution:
                case FormExprType::SymmetricEigenvalue:
                case FormExprType::SymmetricEigenvalueDirectionalDerivative:
                case FormExprType::SymmetricEigenvalueDirectionalDerivativeWrtA:
                case FormExprType::Eigenvalue:
                case FormExprType::SymmetricEigenvector:
                case FormExprType::SpectralDecomposition:
                case FormExprType::SymmetricEigenvectorDirectionalDerivative:
                case FormExprType::SpectralDecompositionDirectionalDerivative:
                case FormExprType::SmoothHeaviside:
                case FormExprType::SmoothAbsoluteValue:
                case FormExprType::SmoothSign:
                    if (!kids.empty() && kids[0]) {
                        deg = child_degree(0);
                    }
                    break;

                // Comparisons / predicates: treat as max over children (rare in dt(Trial) contexts).
                case FormExprType::Less:
                case FormExprType::LessEqual:
                case FormExprType::Greater:
                case FormExprType::GreaterEqual:
                case FormExprType::Equal:
                case FormExprType::NotEqual:
                    deg = std::max(child_degree(0), child_degree(1));
                    break;

                // Terminals and geometry nodes: degree 0.
                default:
                    deg = 0;
                    for (const auto& k : kids) {
                        if (!k) continue;
                        deg = std::max(deg, self(self, *k));
                    }
                    break;
            }

            memo.emplace(&n, deg);
            return deg;
        };

        if (dt_trial_degree(dt_trial_degree, node) > 1) {
            throw std::invalid_argument("FormCompiler: multiple dt(TrialFunction) factors in one integral term are not supported");
        }
    }

    return max_order;
}

void collectIntegralTerms(
    const FormExpr& expr,
    int sign,
    std::vector<IntegralTerm>& out_terms)
{
    if (!expr.isValid()) {
        throw std::invalid_argument("collectIntegralTerms: invalid expression");
    }

    const auto& n = *expr.node();
    const auto children = n.childrenShared();

    const auto collect_integrand_terms = [&](const auto& self,
                                             const FormExpr& integrand_expr,
                                             int integrand_sign,
                                             IntegralDomain domain,
                                             int boundary_marker,
                                             int interface_marker) -> void {
        const auto& in = *integrand_expr.node();
        const auto kids = in.childrenShared();

        switch (in.type()) {
            case FormExprType::Add: {
                if (kids.size() != 2) throw std::logic_error("Add node must have 2 children");
                self(self, makeExprFromNode(kids[0]), integrand_sign, domain, boundary_marker, interface_marker);
                self(self, makeExprFromNode(kids[1]), integrand_sign, domain, boundary_marker, interface_marker);
                return;
            }
            case FormExprType::Subtract: {
                if (kids.size() != 2) throw std::logic_error("Subtract node must have 2 children");
                self(self, makeExprFromNode(kids[0]), integrand_sign, domain, boundary_marker, interface_marker);
                self(self, makeExprFromNode(kids[1]), -integrand_sign, domain, boundary_marker, interface_marker);
                return;
            }
            case FormExprType::Negate: {
                if (kids.size() != 1) throw std::logic_error("Negate node must have 1 child");
                self(self, makeExprFromNode(kids[0]), -integrand_sign, domain, boundary_marker, interface_marker);
                return;
            }
            default:
                break;
        }

        FormExpr integrand = integrand_expr;
        if (integrand_sign < 0) {
            integrand = FormExpr::constant(-1.0) * integrand;
        }

        IntegralTerm term;
        term.domain = domain;
        term.boundary_marker = boundary_marker;
        term.interface_marker = interface_marker;
        term.integrand = std::move(integrand);
        term.debug_string = term.integrand.toString();
        out_terms.push_back(std::move(term));
    };

    switch (n.type()) {
        case FormExprType::Add: {
            if (children.size() != 2) throw std::logic_error("Add node must have 2 children");
            collectIntegralTerms(makeExprFromNode(children[0]), sign, out_terms);
            collectIntegralTerms(makeExprFromNode(children[1]), sign, out_terms);
            return;
        }
        case FormExprType::Subtract: {
            if (children.size() != 2) throw std::logic_error("Subtract node must have 2 children");
            collectIntegralTerms(makeExprFromNode(children[0]), sign, out_terms);
            collectIntegralTerms(makeExprFromNode(children[1]), -sign, out_terms);
            return;
        }
        case FormExprType::Negate: {
            if (children.size() != 1) throw std::logic_error("Negate node must have 1 child");
            collectIntegralTerms(makeExprFromNode(children[0]), -sign, out_terms);
            return;
        }
        case FormExprType::CellIntegral: {
            if (children.size() != 1) throw std::logic_error("CellIntegral node must have 1 child");
            collect_integrand_terms(collect_integrand_terms,
                                    makeExprFromNode(children[0]),
                                    sign,
                                    IntegralDomain::Cell,
                                    /*boundary_marker=*/-1,
                                    /*interface_marker=*/-1);
            return;
        }
        case FormExprType::BoundaryIntegral: {
            if (children.size() != 1) throw std::logic_error("BoundaryIntegral node must have 1 child");
            const int marker = n.boundaryMarker().value_or(-1);
            collect_integrand_terms(collect_integrand_terms,
                                    makeExprFromNode(children[0]),
                                    sign,
                                    IntegralDomain::Boundary,
                                    /*boundary_marker=*/marker,
                                    /*interface_marker=*/-1);
            return;
        }
        case FormExprType::InteriorFaceIntegral: {
            if (children.size() != 1) throw std::logic_error("InteriorFaceIntegral node must have 1 child");
            collect_integrand_terms(collect_integrand_terms,
                                    makeExprFromNode(children[0]),
                                    sign,
                                    IntegralDomain::InteriorFace,
                                    /*boundary_marker=*/-1,
                                    /*interface_marker=*/-1);
            return;
        }
        case FormExprType::InterfaceIntegral: {
            if (children.size() != 1) throw std::logic_error("InterfaceIntegral node must have 1 child");
            const int marker = n.interfaceMarker().value_or(-1);
            collect_integrand_terms(collect_integrand_terms,
                                    makeExprFromNode(children[0]),
                                    sign,
                                    IntegralDomain::InterfaceFace,
                                    /*boundary_marker=*/-1,
                                    /*interface_marker=*/marker);
            return;
        }
        default:
            break;
    }

    throw std::invalid_argument(
        "FormCompiler: top-level expression must be a sum of integrals; got: " + expr.toString());
}

} // namespace detail

FormIR FormCompiler::compileImpl(const FormExpr& form, FormKind kind)
{
    if (!form.isValid()) {
        throw std::invalid_argument("FormCompiler: cannot compile invalid form");
    }

    detail::requireNoCoupledPlaceholders(*form.node());
    detail::requireSupportedGeometrySensitivityMode(
        *form.node(), impl_->options.geometry_sensitivity);
    if (!impl_->options.jit.enable) {
        detail::requireNoIndexedAccess(*form.node());
    }

    FormIR ir;
    ir.setCompiled(false);
    ir.setKind(kind);

    const auto args = detail::analyzeBoundArguments(*form.node());
    ir.setTestSpace(args.test_space);
    ir.setTrialSpace(args.trial_space);

    std::vector<IntegralTerm> terms;
    detail::collectIntegralTerms(form, /*sign=*/+1, terms);

    const bool form_has_geometry_sensitivity_terminals =
        detail::containsGeometrySensitivityTerminal(*form.node());
    const bool geometry_sensitivity_active =
        impl_->options.geometry_sensitivity.mode ==
            GeometrySensitivityMode::MeshMotionUnknowns &&
        form_has_geometry_sensitivity_terminals;

    assembly::RequiredData required = assembly::RequiredData::None;
    std::unordered_map<FieldId, assembly::RequiredData> field_required{};
    int max_time_order = 0;
    bool has_explicit_time_dependency = false;
    for (auto& t : terms) {
        t.time_derivative_order = detail::analyzeTimeDerivativeOrder(*t.integrand.node(), kind);
        max_time_order = std::max(max_time_order, t.time_derivative_order);
        has_explicit_time_dependency = has_explicit_time_dependency || isTimeDependent(t.integrand);

        t.required_data = detail::analyzeRequiredData(*t.integrand.node(), kind);
        for (const auto& fr : detail::analyzeFieldRequirements(*t.integrand.node())) {
            field_required[fr.field] |= fr.required;
        }

        const bool term_has_geometry_sensitivity_terminals =
            geometry_sensitivity_active &&
            detail::containsGeometrySensitivityTerminal(*t.integrand.node());
        if (term_has_geometry_sensitivity_terminals) {
            t.required_data |= assembly::RequiredData::BasisValues;
            t.required_data |= assembly::RequiredData::BasisGradients;
            t.required_data |= assembly::RequiredData::CurrentJacobians;
            t.required_data |= assembly::RequiredData::CurrentMeasures;
            field_required[impl_->options.geometry_sensitivity.mesh_motion_field] |=
                assembly::RequiredData::SolutionValues |
                assembly::RequiredData::SolutionGradients;

            if (t.domain == IntegralDomain::Boundary ||
                t.domain == IntegralDomain::InteriorFace ||
                t.domain == IntegralDomain::InterfaceFace) {
                t.required_data |= assembly::RequiredData::CurrentNormals;
                t.required_data |= assembly::RequiredData::SurfaceJacobians;
            }
        }

        // Face terms require face geometry context (surface measure, normals).
        if (t.domain == IntegralDomain::Boundary ||
            t.domain == IntegralDomain::InteriorFace ||
            t.domain == IntegralDomain::InterfaceFace) {
            t.required_data |= assembly::RequiredData::Normals;
        }

        // Interior-face terms require plus-side (neighbor) context; include
        // DG-oriented flags so assemblers can prepare the correct data.
        if (t.domain == IntegralDomain::InteriorFace || t.domain == IntegralDomain::InterfaceFace) {
            t.required_data |= assembly::RequiredData::NeighborData;
            t.required_data |= assembly::RequiredData::FaceOrientations;
        }
        required |= t.required_data;
    }

    std::vector<assembly::FieldRequirement> field_requirements;
    field_requirements.reserve(field_required.size());
    for (const auto& kv : field_required) {
        field_requirements.push_back(assembly::FieldRequirement{kv.first, kv.second});
    }
    std::sort(field_requirements.begin(), field_requirements.end(),
              [](const assembly::FieldRequirement& a, const assembly::FieldRequirement& b) { return a.field < b.field; });

    ir.setTerms(std::move(terms));
    ir.setRequiredData(required);
    ir.setFieldRequirements(std::move(field_requirements));
    ir.setMaxTimeDerivativeOrder(max_time_order);
    ir.setHasExplicitTimeDependency(has_explicit_time_dependency);
    ir.setGeometrySensitivityOptions(impl_->options.geometry_sensitivity);
    ir.setHasGeometrySensitivityTerminals(form_has_geometry_sensitivity_terminals);
    ir.setCompiled(true);

    std::ostringstream oss;
    oss << "FormIR\n";
    oss << "  kind: ";
    switch (kind) {
        case FormKind::Linear: oss << "linear\n"; break;
        case FormKind::Bilinear: oss << "bilinear\n"; break;
        case FormKind::Residual: oss << "residual\n"; break;
    }
    oss << "  terms: " << ir.terms().size() << "\n";
    for (const auto& t : ir.terms()) {
        oss << "    - ";
        switch (t.domain) {
            case IntegralDomain::Cell: oss << "dx"; break;
            case IntegralDomain::Boundary: oss << "ds(" << t.boundary_marker << ")"; break;
            case IntegralDomain::InteriorFace: oss << "dS"; break;
            case IntegralDomain::InterfaceFace: oss << "dI(" << t.interface_marker << ")"; break;
        }
        if (t.time_derivative_order > 0) {
            oss << " [dt^" << t.time_derivative_order << "]";
        }
        oss << " : " << t.debug_string << "\n";
    }
    ir.setDump(oss.str());

    return ir;
}

FormIR FormCompiler::compileLinear(const FormExpr& form)
{
    if (!form.hasTest()) {
        throw std::invalid_argument("FormCompiler::compileLinear: form has no test function");
    }
    if (form.hasTrial()) {
        throw std::invalid_argument("FormCompiler::compileLinear: form contains TrialFunction");
    }
    return compileImpl(form, FormKind::Linear);
}

FormIR FormCompiler::compileBilinear(const FormExpr& form)
{
    if (!form.hasTest() || !form.hasTrial()) {
        throw std::invalid_argument("FormCompiler::compileBilinear: form must contain both test and trial functions");
    }
    return compileImpl(form, FormKind::Bilinear);
}

FormIR FormCompiler::compileResidual(const FormExpr& residual_form)
{
    if (!residual_form.hasTest()) {
        throw std::invalid_argument("FormCompiler::compileResidual: residual form has no test function");
    }
    if (!residual_form.hasTrial()) {
        throw std::invalid_argument("FormCompiler::compileResidual: residual form has no TrialFunction (unknown)");
    }
    return compileImpl(residual_form, FormKind::Residual);
}

std::vector<std::optional<FormIR>> FormCompiler::compileLinear(const BlockLinearForm& blocks)
{
    std::vector<std::optional<FormIR>> out(blocks.numTestFields());
    for (std::size_t i = 0; i < blocks.numTestFields(); ++i) {
        if (!blocks.hasBlock(i)) {
            out[i] = std::nullopt;
            continue;
        }
        out[i] = compileLinear(blocks.block(i));
    }
    return out;
}

std::vector<std::vector<std::optional<FormIR>>> FormCompiler::compileBilinear(const BlockBilinearForm& blocks)
{
    std::vector<std::vector<std::optional<FormIR>>> out;
    out.resize(blocks.numTestFields());
    for (auto& row : out) {
        row.resize(blocks.numTrialFields());
    }

    for (std::size_t i = 0; i < blocks.numTestFields(); ++i) {
        for (std::size_t j = 0; j < blocks.numTrialFields(); ++j) {
            if (!blocks.hasBlock(i, j)) {
                continue;
            }
            out[i][j] = compileBilinear(blocks.block(i, j));
        }
    }
    return out;
}

std::vector<std::vector<std::optional<FormIR>>> FormCompiler::compileResidual(const BlockBilinearForm& blocks)
{
    std::vector<std::vector<std::optional<FormIR>>> out;
    out.resize(blocks.numTestFields());
    for (auto& row : out) {
        row.resize(blocks.numTrialFields());
    }

    for (std::size_t i = 0; i < blocks.numTestFields(); ++i) {
        for (std::size_t j = 0; j < blocks.numTrialFields(); ++j) {
            if (!blocks.hasBlock(i, j)) {
                continue;
            }
            out[i][j] = compileResidual(blocks.block(i, j));
        }
    }
    return out;
}

MixedFormIR FormCompiler::compile(const FormExpr& form, FormKind kind)
{
    // compile() is the auto-detecting entry point. It delegates to compileMixed(),
    // which already handles both single-field (1×1 MixedFormIR) and multi-field
    // (N×M block-sparse MixedFormIR) cases.
    return compileMixed(form, kind);
}

MixedFormIR FormCompiler::compileMixed(const FormExpr& form, FormKind kind)
{
    if (!form.isValid()) {
        throw std::invalid_argument("FormCompiler::compileMixed: cannot compile invalid form");
    }
    if (!form.hasTest()) {
        throw std::invalid_argument("FormCompiler::compileMixed: form has no test function");
    }

    detail::requireNoCoupledPlaceholders(*form.node());
    if (!impl_->options.jit.enable) {
        detail::requireNoIndexedAccess(*form.node());
    }

    // Collect all test/trial space signatures
    const auto mixed_args = detail::analyzeMixedBoundArguments(*form.node());

    // If single test and single trial, delegate to single-field compilation.
    // For linear forms (no trial), use a 1×1 layout with a synthetic trial
    // column so that installMixedFormIR / installMixedLinear can install the
    // block at (0,0). The trial descriptor is populated with the test field's
    // descriptor to keep the IR self-consistent.
    //
    // Residual/Bilinear forms with no trial function are rejected, consistent
    // with compileResidual() / compileBilinear().
    if (mixed_args.test_spaces.size() <= 1 && mixed_args.trial_spaces.size() <= 1) {
        if (mixed_args.trial_spaces.empty() && kind != FormKind::Linear) {
            throw std::invalid_argument(
                "FormCompiler::compileMixed: " +
                std::string(kind == FormKind::Residual ? "residual" : "bilinear") +
                " form has no TrialFunction — use FormKind::Linear for test-only forms");
        }

        MixedFormIR mir(1, 1);
        mir.setBlock(0, 0, compileImpl(form, kind));
        mir.setKind(kind);
        mir.setSourceExpression(form);

        if (!mixed_args.test_spaces.empty()) {
            std::vector<MixedFieldDescriptor> test_desc;
            test_desc.push_back({INVALID_FIELD_ID, mixed_args.test_spaces[0].second,
                                 mixed_args.test_spaces[0].first,
                                 mixed_args.test_spaces[0].first.value_dimension});
            mir.setTestFields(std::move(test_desc));
        }
        if (!mixed_args.trial_spaces.empty()) {
            std::vector<MixedFieldDescriptor> trial_desc;
            trial_desc.push_back({INVALID_FIELD_ID, mixed_args.trial_spaces[0].second,
                                  mixed_args.trial_spaces[0].first,
                                  mixed_args.trial_spaces[0].first.value_dimension});
            mir.setTrialFields(std::move(trial_desc));
        } else {
            // Linear form: populate trial descriptor from test field so
            // installMixedLinear can map the synthetic trial column.
            if (!mixed_args.test_spaces.empty()) {
                std::vector<MixedFieldDescriptor> trial_desc;
                trial_desc.push_back({INVALID_FIELD_ID,
                                      mixed_args.test_spaces[0].second + "_trial",
                                      mixed_args.test_spaces[0].first,
                                      mixed_args.test_spaces[0].first.value_dimension});
                mir.setTrialFields(std::move(trial_desc));
            }
        }
        return mir;
    }

    // Multi-field: reject Residual/Bilinear with no trial, consistent with
    // the single-field fast path and with compileResidual()/compileBilinear().
    if (mixed_args.trial_spaces.empty() && kind != FormKind::Linear) {
        throw std::invalid_argument(
            "FormCompiler::compileMixed: multi-field " +
            std::string(kind == FormKind::Residual ? "residual" : "bilinear") +
            " form has no TrialFunction — use FormKind::Linear for test-only forms");
    }

    // Multi-field: decompose into blocks.
    // For linear forms (no trial space), use 1 synthetic trial column so
    // per-test blocks can be compiled and installed.
    const std::size_t n_test = mixed_args.test_spaces.size();
    const std::size_t n_trial = std::max(mixed_args.trial_spaces.size(), std::size_t{1});

    MixedFormIR mir(n_test, n_trial);
    mir.setKind(kind);
    mir.setSourceExpression(form);

    // Set field descriptors
    std::vector<MixedFieldDescriptor> test_desc;
    test_desc.reserve(n_test);
    for (const auto& ts : mixed_args.test_spaces) {
        test_desc.push_back({INVALID_FIELD_ID, ts.second, ts.first, ts.first.value_dimension});
    }
    mir.setTestFields(std::move(test_desc));

    std::vector<MixedFieldDescriptor> trial_desc;
    if (!mixed_args.trial_spaces.empty()) {
        trial_desc.reserve(mixed_args.trial_spaces.size());
        for (const auto& ts : mixed_args.trial_spaces) {
            trial_desc.push_back({INVALID_FIELD_ID, ts.second, ts.first, ts.first.value_dimension});
        }
    } else if (n_trial > 0) {
        // Linear form: synthetic trial column descriptor from the first test field
        trial_desc.push_back({INVALID_FIELD_ID,
                              mixed_args.test_spaces[0].second + "_trial",
                              mixed_args.test_spaces[0].first,
                              mixed_args.test_spaces[0].first.value_dimension});
    }
    mir.setTrialFields(std::move(trial_desc));

    // Collect all integral terms from the mixed expression
    std::vector<IntegralTerm> all_terms;
    detail::collectIntegralTerms(form, /*sign=*/+1, all_terms);

    // Build whole-form domain summary from all terms
    MixedFormDomainSummary domain_summary;
    for (const auto& term : all_terms) {
        switch (term.domain) {
            case IntegralDomain::Cell:
                domain_summary.has_cell_terms = true;
                break;
            case IntegralDomain::Boundary:
                domain_summary.has_boundary_terms = true;
                if (std::find(domain_summary.boundary_markers.begin(),
                              domain_summary.boundary_markers.end(),
                              term.boundary_marker) == domain_summary.boundary_markers.end()) {
                    domain_summary.boundary_markers.push_back(term.boundary_marker);
                }
                break;
            case IntegralDomain::InteriorFace:
                domain_summary.has_interior_face_terms = true;
                break;
            case IntegralDomain::InterfaceFace:
                domain_summary.has_interface_face_terms = true;
                if (std::find(domain_summary.interface_markers.begin(),
                              domain_summary.interface_markers.end(),
                              term.interface_marker) == domain_summary.interface_markers.end()) {
                    domain_summary.interface_markers.push_back(term.interface_marker);
                }
                break;
        }
    }
    mir.setDomainSummary(std::move(domain_summary));

    // Classify each term by its (test, trial) block.
    //
    // For bilinear/residual forms with trial functions, terms containing
    // exactly one test AND one trial are placed in the corresponding
    // (test_idx, trial_idx) block. Residual test-only terms (no trial) are
    // NOT classified here — they are handled by the residual installer.
    //
    // For linear forms (no trial space), all terms for a given test function
    // are placed in the synthetic trial column 0.
    // Linear forms (no trial space) use a synthetic trial column and collect
    // all terms by test function only. Residual/bilinear forms with no trial
    // should not reach this path — they produce an empty Nx0 block matrix
    // (which is correct: no Jacobian blocks to classify).
    const bool is_linear_no_trial = mixed_args.trial_spaces.empty() && kind == FormKind::Linear;

    for (std::size_t ti = 0; ti < n_test; ++ti) {
        const auto& test_name = mixed_args.test_spaces[ti].second;

        if (is_linear_no_trial) {
            // Linear form: collect ALL terms for this test function into column 0
            std::vector<IntegralTerm> block_terms;
            BlockProvenance provenance;
            for (std::size_t idx = 0; idx < all_terms.size(); ++idx) {
                const auto& term = all_terms[idx];
                if (detail::integrandContainsTestNamed(*term.integrand.node(), test_name)) {
                    block_terms.push_back(term);
                    provenance.contributing_term_indices.push_back(idx);
                }
            }
            if (block_terms.empty()) continue;

            FormExpr block_expr;
            for (std::size_t k = 0; k < block_terms.size(); ++k) {
                FormExpr term_with_measure;
                switch (block_terms[k].domain) {
                    case IntegralDomain::Cell:
                        term_with_measure = block_terms[k].integrand.dx();
                        break;
                    case IntegralDomain::Boundary:
                        term_with_measure = block_terms[k].integrand.ds(block_terms[k].boundary_marker);
                        break;
                    case IntegralDomain::InteriorFace:
                        term_with_measure = block_terms[k].integrand.dS();
                        break;
                    case IntegralDomain::InterfaceFace:
                        term_with_measure = block_terms[k].integrand.dI(block_terms[k].interface_marker);
                        break;
                }
                if (!block_expr.isValid()) {
                    block_expr = term_with_measure;
                } else {
                    block_expr = block_expr + term_with_measure;
                }
            }

            provenance.source_summary = test_name + ": " + std::to_string(block_terms.size()) + " term(s)";
            try {
                mir.setBlock(ti, 0, compileImpl(block_expr, kind));
                mir.setBlockProvenance(ti, 0, std::move(provenance));
            } catch (const std::invalid_argument& e) {
                throw std::invalid_argument(
                    "FormCompiler::compileMixed: error compiling linear block (" +
                    std::to_string(ti) + ") [test=" + test_name + "]: " + e.what());
            }
            continue;  // skip the trial loop below
        }

        for (std::size_t tj = 0; tj < mixed_args.trial_spaces.size(); ++tj) {
            const auto& trial_name = mixed_args.trial_spaces[tj].second;

            // Collect terms that contain this specific test AND trial
            std::vector<IntegralTerm> block_terms;
            BlockProvenance provenance;
            for (std::size_t idx = 0; idx < all_terms.size(); ++idx) {
                const auto& term = all_terms[idx];
                if (detail::integrandContainsTestNamed(*term.integrand.node(), test_name) &&
                    detail::integrandContainsTrialNamed(*term.integrand.node(), trial_name)) {
                    block_terms.push_back(term);
                    provenance.contributing_term_indices.push_back(idx);
                }
            }

            if (block_terms.empty()) continue;

            // Reconstruct a FormExpr for this block by summing its integral terms
            // Each term already includes its measure (dx/ds/dS/dI)
            FormExpr block_expr;
            for (std::size_t k = 0; k < block_terms.size(); ++k) {
                // The term's integrand is the integrand-only part; we need to re-wrap it
                FormExpr term_with_measure;
                switch (block_terms[k].domain) {
                    case IntegralDomain::Cell:
                        term_with_measure = block_terms[k].integrand.dx();
                        break;
                    case IntegralDomain::Boundary:
                        term_with_measure = block_terms[k].integrand.ds(block_terms[k].boundary_marker);
                        break;
                    case IntegralDomain::InteriorFace:
                        term_with_measure = block_terms[k].integrand.dS();
                        break;
                    case IntegralDomain::InterfaceFace:
                        term_with_measure = block_terms[k].integrand.dI(block_terms[k].interface_marker);
                        break;
                }

                if (!block_expr.isValid()) {
                    block_expr = term_with_measure;
                } else {
                    block_expr = block_expr + term_with_measure;
                }
            }

            // Build source summary for diagnostics
            provenance.source_summary = test_name + "-" + trial_name + ": "
                + std::to_string(block_terms.size()) + " term(s)";

            // Compile this block using the single-field path
            // (analyzeBoundArguments will see exactly one test + one trial)
            try {
                mir.setBlock(ti, tj, compileImpl(block_expr, kind));
                mir.setBlockProvenance(ti, tj, std::move(provenance));
            } catch (const std::invalid_argument& e) {
                // Re-throw with block context including field names
                throw std::invalid_argument(
                    "FormCompiler::compileMixed: error compiling block (" +
                    std::to_string(ti) + ", " + std::to_string(tj) +
                    ") [test=" + test_name + ", trial=" + trial_name + "]: " + e.what());
            }
        }
    }

    return mir;
}

} // namespace forms
} // namespace FE
} // namespace svmp
