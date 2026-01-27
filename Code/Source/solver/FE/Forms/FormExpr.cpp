/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#include "Forms/FormExpr.h"

#include "Forms/ConstitutiveModel.h"
#include "Forms/Index.h"
#include "Forms/Tensor/TensorIndex.h"
#include "Spaces/FunctionSpace.h"

#include <sstream>
#include <stdexcept>

namespace svmp {
namespace FE {
namespace forms {

namespace {

// ============================================================================
// Terminal nodes
// ============================================================================

class TestFunctionNode final : public FormExprNode {
public:
    TestFunctionNode(SpaceSignature signature, std::string name)
        : signature_(std::move(signature)), name_(std::move(name))
    {}

    [[nodiscard]] FormExprType type() const noexcept override { return FormExprType::TestFunction; }
    [[nodiscard]] std::string toString() const override { return name_; }
    [[nodiscard]] bool hasTest() const noexcept override { return true; }
    [[nodiscard]] bool hasTrial() const noexcept override { return false; }
    [[nodiscard]] const SpaceSignature* spaceSignature() const override { return &signature_; }

private:
    SpaceSignature signature_{};
    std::string name_;
};

class TrialFunctionNode final : public FormExprNode {
public:
    TrialFunctionNode(SpaceSignature signature, std::string name)
        : signature_(std::move(signature)), name_(std::move(name))
    {}

    [[nodiscard]] FormExprType type() const noexcept override { return FormExprType::TrialFunction; }
    [[nodiscard]] std::string toString() const override { return name_; }
    [[nodiscard]] bool hasTest() const noexcept override { return false; }
    [[nodiscard]] bool hasTrial() const noexcept override { return true; }
    [[nodiscard]] const SpaceSignature* spaceSignature() const override { return &signature_; }

private:
    SpaceSignature signature_{};
    std::string name_;
};

class DiscreteFieldNode final : public FormExprNode {
public:
    DiscreteFieldNode(FieldId field, SpaceSignature signature, std::string name)
        : field_(field), signature_(std::move(signature)), name_(std::move(name))
    {}

    [[nodiscard]] FormExprType type() const noexcept override { return FormExprType::DiscreteField; }
    [[nodiscard]] std::string toString() const override { return name_; }
    [[nodiscard]] bool hasTest() const noexcept override { return false; }
    [[nodiscard]] bool hasTrial() const noexcept override { return false; }
    [[nodiscard]] const SpaceSignature* spaceSignature() const override { return &signature_; }
    [[nodiscard]] std::optional<FieldId> fieldId() const override { return field_; }

private:
    FieldId field_{INVALID_FIELD_ID};
    SpaceSignature signature_{};
    std::string name_;
};

class StateFieldNode final : public FormExprNode {
public:
    StateFieldNode(FieldId field, SpaceSignature signature, std::string name)
        : field_(field), signature_(std::move(signature)), name_(std::move(name))
    {}

    [[nodiscard]] FormExprType type() const noexcept override { return FormExprType::StateField; }
    [[nodiscard]] std::string toString() const override { return name_; }
    [[nodiscard]] bool hasTest() const noexcept override { return false; }
    [[nodiscard]] bool hasTrial() const noexcept override { return false; }
    [[nodiscard]] const SpaceSignature* spaceSignature() const override { return &signature_; }
    [[nodiscard]] std::optional<FieldId> fieldId() const override { return field_; }

private:
    FieldId field_{INVALID_FIELD_ID};
    SpaceSignature signature_{};
    std::string name_;
};

class ConstantNode final : public FormExprNode {
public:
    explicit ConstantNode(Real value) : value_(value) {}

    [[nodiscard]] FormExprType type() const noexcept override { return FormExprType::Constant; }
    [[nodiscard]] std::string toString() const override {
        std::ostringstream oss;
        oss << value_;
        return oss.str();
    }
    [[nodiscard]] bool hasTest() const noexcept override { return false; }
    [[nodiscard]] bool hasTrial() const noexcept override { return false; }

    [[nodiscard]] std::optional<Real> constantValue() const override { return value_; }

    [[nodiscard]] Real value() const noexcept { return value_; }

private:
    Real value_{0.0};
};

class BoundaryFunctionalSymbolNode final : public FormExprNode {
public:
    BoundaryFunctionalSymbolNode(std::shared_ptr<FormExprNode> integrand,
                                 int boundary_marker,
                                 std::string name)
        : integrand_(std::move(integrand))
        , boundary_marker_(boundary_marker)
        , name_(std::move(name))
    {}

    [[nodiscard]] FormExprType type() const noexcept override { return FormExprType::BoundaryFunctionalSymbol; }
    [[nodiscard]] std::string toString() const override
    {
        const std::string inner = integrand_ ? integrand_->toString() : "<empty>";
        return "boundaryIntegral(" + name_ + ", " + std::to_string(boundary_marker_) + ", " + inner + ")";
    }
    [[nodiscard]] bool hasTest() const noexcept override { return false; }
    [[nodiscard]] bool hasTrial() const noexcept override { return false; }

    [[nodiscard]] std::optional<int> boundaryMarker() const override { return boundary_marker_; }
    [[nodiscard]] std::optional<std::string_view> symbolName() const override { return name_; }

    [[nodiscard]] std::vector<std::shared_ptr<FormExprNode>> childrenShared() const override
    {
        return integrand_ ? std::vector<std::shared_ptr<FormExprNode>>{integrand_}
                          : std::vector<std::shared_ptr<FormExprNode>>{};
    }

    [[nodiscard]] std::vector<const FormExprNode*> children() const override
    {
        return integrand_ ? std::vector<const FormExprNode*>{integrand_.get()}
                          : std::vector<const FormExprNode*>{};
    }

private:
    std::shared_ptr<FormExprNode> integrand_{};
    int boundary_marker_{-1};
    std::string name_{};
};

class AuxiliaryStateSymbolNode final : public FormExprNode {
public:
    explicit AuxiliaryStateSymbolNode(std::string name)
        : name_(std::move(name))
    {}

    [[nodiscard]] FormExprType type() const noexcept override { return FormExprType::AuxiliaryStateSymbol; }
    [[nodiscard]] std::string toString() const override { return "aux(" + name_ + ")"; }
    [[nodiscard]] bool hasTest() const noexcept override { return false; }
    [[nodiscard]] bool hasTrial() const noexcept override { return false; }
    [[nodiscard]] std::optional<std::string_view> symbolName() const override { return name_; }

private:
    std::string name_{};
};

class CoordinateNode final : public FormExprNode {
public:
    [[nodiscard]] FormExprType type() const noexcept override { return FormExprType::Coordinate; }
    [[nodiscard]] std::string toString() const override { return "x"; }
    [[nodiscard]] bool hasTest() const noexcept override { return false; }
    [[nodiscard]] bool hasTrial() const noexcept override { return false; }
};

class ReferenceCoordinateNode final : public FormExprNode {
public:
    [[nodiscard]] FormExprType type() const noexcept override { return FormExprType::ReferenceCoordinate; }
    [[nodiscard]] std::string toString() const override { return "X"; }
    [[nodiscard]] bool hasTest() const noexcept override { return false; }
    [[nodiscard]] bool hasTrial() const noexcept override { return false; }
};

class TimeNode final : public FormExprNode {
public:
    [[nodiscard]] FormExprType type() const noexcept override { return FormExprType::Time; }
    [[nodiscard]] std::string toString() const override { return "t"; }
    [[nodiscard]] bool hasTest() const noexcept override { return false; }
    [[nodiscard]] bool hasTrial() const noexcept override { return false; }
};

class TimeStepNode final : public FormExprNode {
public:
    [[nodiscard]] FormExprType type() const noexcept override { return FormExprType::TimeStep; }
    [[nodiscard]] std::string toString() const override { return "dt"; }
    [[nodiscard]] bool hasTest() const noexcept override { return false; }
    [[nodiscard]] bool hasTrial() const noexcept override { return false; }
};

class EffectiveTimeStepNode final : public FormExprNode {
public:
    [[nodiscard]] FormExprType type() const noexcept override { return FormExprType::EffectiveTimeStep; }
    [[nodiscard]] std::string toString() const override { return "dt_eff"; }
    [[nodiscard]] bool hasTest() const noexcept override { return false; }
    [[nodiscard]] bool hasTrial() const noexcept override { return false; }
};

class IdentityNode final : public FormExprNode {
public:
    IdentityNode() = default;
    explicit IdentityNode(int dim) : dim_(dim) {}

    [[nodiscard]] FormExprType type() const noexcept override { return FormExprType::Identity; }
    [[nodiscard]] std::string toString() const override {
        if (!dim_) {
            return "I";
        }
        return "I(" + std::to_string(*dim_) + ")";
    }
    [[nodiscard]] bool hasTest() const noexcept override { return false; }
    [[nodiscard]] bool hasTrial() const noexcept override { return false; }

    [[nodiscard]] std::optional<int> identityDim() const override { return dim_; }

private:
    std::optional<int> dim_{};
};

class JacobianNode final : public FormExprNode {
public:
    [[nodiscard]] FormExprType type() const noexcept override { return FormExprType::Jacobian; }
    [[nodiscard]] std::string toString() const override { return "J"; }
    [[nodiscard]] bool hasTest() const noexcept override { return false; }
    [[nodiscard]] bool hasTrial() const noexcept override { return false; }
};

class JacobianInverseNode final : public FormExprNode {
public:
    [[nodiscard]] FormExprType type() const noexcept override { return FormExprType::JacobianInverse; }
    [[nodiscard]] std::string toString() const override { return "Jinv"; }
    [[nodiscard]] bool hasTest() const noexcept override { return false; }
    [[nodiscard]] bool hasTrial() const noexcept override { return false; }
};

class JacobianDeterminantNode final : public FormExprNode {
public:
    [[nodiscard]] FormExprType type() const noexcept override { return FormExprType::JacobianDeterminant; }
    [[nodiscard]] std::string toString() const override { return "detJ"; }
    [[nodiscard]] bool hasTest() const noexcept override { return false; }
    [[nodiscard]] bool hasTrial() const noexcept override { return false; }
};

class NormalNode final : public FormExprNode {
public:
    [[nodiscard]] FormExprType type() const noexcept override { return FormExprType::Normal; }
    [[nodiscard]] std::string toString() const override { return "n"; }
    [[nodiscard]] bool hasTest() const noexcept override { return false; }
    [[nodiscard]] bool hasTrial() const noexcept override { return false; }
};

class CellDiameterNode final : public FormExprNode {
public:
    [[nodiscard]] FormExprType type() const noexcept override { return FormExprType::CellDiameter; }
    [[nodiscard]] std::string toString() const override { return "h"; }
    [[nodiscard]] bool hasTest() const noexcept override { return false; }
    [[nodiscard]] bool hasTrial() const noexcept override { return false; }
};

class CellVolumeNode final : public FormExprNode {
public:
    [[nodiscard]] FormExprType type() const noexcept override { return FormExprType::CellVolume; }
    [[nodiscard]] std::string toString() const override { return "vol(K)"; }
    [[nodiscard]] bool hasTest() const noexcept override { return false; }
    [[nodiscard]] bool hasTrial() const noexcept override { return false; }
};

class FacetAreaNode final : public FormExprNode {
public:
    [[nodiscard]] FormExprType type() const noexcept override { return FormExprType::FacetArea; }
    [[nodiscard]] std::string toString() const override { return "area(F)"; }
    [[nodiscard]] bool hasTest() const noexcept override { return false; }
    [[nodiscard]] bool hasTrial() const noexcept override { return false; }
};

class CellDomainIdNode final : public FormExprNode {
public:
    [[nodiscard]] FormExprType type() const noexcept override { return FormExprType::CellDomainId; }
    [[nodiscard]] std::string toString() const override { return "domain(K)"; }
    [[nodiscard]] bool hasTest() const noexcept override { return false; }
    [[nodiscard]] bool hasTrial() const noexcept override { return false; }
};

class CoefficientNode final : public FormExprNode {
public:
    explicit CoefficientNode(std::string name, ScalarCoefficient func)
        : name_(std::move(name)), scalar_func_(std::move(func))
    {
    }

    explicit CoefficientNode(std::string name, TimeScalarCoefficient func)
        : name_(std::move(name)), time_scalar_func_(std::move(func))
    {
    }

    explicit CoefficientNode(std::string name, VectorCoefficient func)
        : name_(std::move(name)), vector_func_(std::move(func))
    {
    }

    explicit CoefficientNode(std::string name, MatrixCoefficient func)
        : name_(std::move(name)), matrix_func_(std::move(func))
    {
    }

    explicit CoefficientNode(std::string name, Tensor3Coefficient func)
        : name_(std::move(name)), tensor3_func_(std::move(func))
    {
    }

    explicit CoefficientNode(std::string name, Tensor4Coefficient func)
        : name_(std::move(name)), tensor4_func_(std::move(func))
    {
    }

    [[nodiscard]] FormExprType type() const noexcept override { return FormExprType::Coefficient; }
    [[nodiscard]] std::string toString() const override { return name_; }
    [[nodiscard]] bool hasTest() const noexcept override { return false; }
    [[nodiscard]] bool hasTrial() const noexcept override { return false; }

    [[nodiscard]] const ScalarCoefficient* scalarCoefficient() const override {
        return scalar_func_ ? &scalar_func_ : nullptr;
    }

    [[nodiscard]] const TimeScalarCoefficient* timeScalarCoefficient() const override {
        return time_scalar_func_ ? &time_scalar_func_ : nullptr;
    }

    [[nodiscard]] const VectorCoefficient* vectorCoefficient() const override {
        return vector_func_ ? &vector_func_ : nullptr;
    }

    [[nodiscard]] const MatrixCoefficient* matrixCoefficient() const override {
        return matrix_func_ ? &matrix_func_ : nullptr;
    }

    [[nodiscard]] const Tensor3Coefficient* tensor3Coefficient() const override {
        return tensor3_func_ ? &tensor3_func_ : nullptr;
    }

    [[nodiscard]] const Tensor4Coefficient* tensor4Coefficient() const override {
        return tensor4_func_ ? &tensor4_func_ : nullptr;
    }

    [[nodiscard]] bool isScalar() const noexcept { return static_cast<bool>(scalar_func_); }
    [[nodiscard]] bool isTimeScalar() const noexcept { return static_cast<bool>(time_scalar_func_); }
    [[nodiscard]] bool isVector() const noexcept { return static_cast<bool>(vector_func_); }
    [[nodiscard]] bool isMatrix() const noexcept { return static_cast<bool>(matrix_func_); }
    [[nodiscard]] bool isTensor3() const noexcept { return static_cast<bool>(tensor3_func_); }
    [[nodiscard]] bool isTensor4() const noexcept { return static_cast<bool>(tensor4_func_); }

    [[nodiscard]] const ScalarCoefficient& scalarFunc() const { return scalar_func_; }
    [[nodiscard]] const TimeScalarCoefficient& timeScalarFunc() const { return time_scalar_func_; }
    [[nodiscard]] const VectorCoefficient& vectorFunc() const { return vector_func_; }
    [[nodiscard]] const MatrixCoefficient& matrixFunc() const { return matrix_func_; }
    [[nodiscard]] const Tensor3Coefficient& tensor3Func() const { return tensor3_func_; }
    [[nodiscard]] const Tensor4Coefficient& tensor4Func() const { return tensor4_func_; }

private:
    std::string name_;
    ScalarCoefficient scalar_func_{};
    TimeScalarCoefficient time_scalar_func_{};
    VectorCoefficient vector_func_{};
    MatrixCoefficient matrix_func_{};
    Tensor3Coefficient tensor3_func_{};
    Tensor4Coefficient tensor4_func_{};
};

class ParameterSymbolNode final : public FormExprNode {
public:
    explicit ParameterSymbolNode(std::string name)
        : name_(std::move(name))
    {}

    [[nodiscard]] FormExprType type() const noexcept override { return FormExprType::ParameterSymbol; }
    [[nodiscard]] std::string toString() const override { return "param(" + name_ + ")"; }
    [[nodiscard]] bool hasTest() const noexcept override { return false; }
    [[nodiscard]] bool hasTrial() const noexcept override { return false; }
    [[nodiscard]] std::optional<std::string_view> symbolName() const override { return name_; }

private:
    std::string name_{};
};

class ParameterRefNode final : public FormExprNode {
public:
    explicit ParameterRefNode(std::uint32_t slot)
        : slot_(slot)
    {}

    [[nodiscard]] FormExprType type() const noexcept override { return FormExprType::ParameterRef; }
    [[nodiscard]] std::string toString() const override
    {
        return "param[" + std::to_string(slot_) + "]";
    }
    [[nodiscard]] bool hasTest() const noexcept override { return false; }
    [[nodiscard]] bool hasTrial() const noexcept override { return false; }
    [[nodiscard]] std::optional<std::uint32_t> slotIndex() const override { return slot_; }

private:
    std::uint32_t slot_{0u};
};

class BoundaryIntegralSymbolNode final : public FormExprNode {
public:
    explicit BoundaryIntegralSymbolNode(std::string name)
        : name_(std::move(name))
    {}

    [[nodiscard]] FormExprType type() const noexcept override { return FormExprType::BoundaryIntegralSymbol; }
    [[nodiscard]] std::string toString() const override { return "integral(" + name_ + ")"; }
    [[nodiscard]] bool hasTest() const noexcept override { return false; }
    [[nodiscard]] bool hasTrial() const noexcept override { return false; }
    [[nodiscard]] std::optional<std::string_view> symbolName() const override { return name_; }

private:
    std::string name_{};
};

class BoundaryIntegralRefNode final : public FormExprNode {
public:
    explicit BoundaryIntegralRefNode(std::uint32_t slot)
        : slot_(slot)
    {}

    [[nodiscard]] FormExprType type() const noexcept override { return FormExprType::BoundaryIntegralRef; }
    [[nodiscard]] std::string toString() const override
    {
        return "integral[" + std::to_string(slot_) + "]";
    }
    [[nodiscard]] bool hasTest() const noexcept override { return false; }
    [[nodiscard]] bool hasTrial() const noexcept override { return false; }
    [[nodiscard]] std::optional<std::uint32_t> slotIndex() const override { return slot_; }

private:
    std::uint32_t slot_{0u};
};

class AuxiliaryStateRefNode final : public FormExprNode {
public:
    explicit AuxiliaryStateRefNode(std::uint32_t slot)
        : slot_(slot)
    {}

    [[nodiscard]] FormExprType type() const noexcept override { return FormExprType::AuxiliaryStateRef; }
    [[nodiscard]] std::string toString() const override
    {
        return "aux[" + std::to_string(slot_) + "]";
    }
    [[nodiscard]] bool hasTest() const noexcept override { return false; }
    [[nodiscard]] bool hasTrial() const noexcept override { return false; }
    [[nodiscard]] std::optional<std::uint32_t> slotIndex() const override { return slot_; }

private:
    std::uint32_t slot_{0u};
};

class MaterialStateOldRefNode final : public FormExprNode {
public:
    explicit MaterialStateOldRefNode(std::uint32_t offset_bytes)
        : offset_bytes_(offset_bytes)
    {}

    [[nodiscard]] FormExprType type() const noexcept override { return FormExprType::MaterialStateOldRef; }
    [[nodiscard]] std::string toString() const override
    {
        return "state_old[" + std::to_string(offset_bytes_) + "]";
    }
    [[nodiscard]] bool hasTest() const noexcept override { return false; }
    [[nodiscard]] bool hasTrial() const noexcept override { return false; }
    [[nodiscard]] std::optional<std::uint32_t> stateOffsetBytes() const override { return offset_bytes_; }

private:
    std::uint32_t offset_bytes_{0u};
};

class MaterialStateWorkRefNode final : public FormExprNode {
public:
    explicit MaterialStateWorkRefNode(std::uint32_t offset_bytes)
        : offset_bytes_(offset_bytes)
    {}

    [[nodiscard]] FormExprType type() const noexcept override { return FormExprType::MaterialStateWorkRef; }
    [[nodiscard]] std::string toString() const override
    {
        return "state[" + std::to_string(offset_bytes_) + "]";
    }
    [[nodiscard]] bool hasTest() const noexcept override { return false; }
    [[nodiscard]] bool hasTrial() const noexcept override { return false; }
    [[nodiscard]] std::optional<std::uint32_t> stateOffsetBytes() const override { return offset_bytes_; }

private:
    std::uint32_t offset_bytes_{0u};
};

class PreviousSolutionRefNode final : public FormExprNode {
public:
    explicit PreviousSolutionRefNode(int steps_back)
        : steps_back_(steps_back)
    {}

    [[nodiscard]] FormExprType type() const noexcept override { return FormExprType::PreviousSolutionRef; }
    [[nodiscard]] std::string toString() const override
    {
        return "u_prev(" + std::to_string(steps_back_) + ")";
    }
    [[nodiscard]] bool hasTest() const noexcept override { return false; }
    [[nodiscard]] bool hasTrial() const noexcept override { return false; }
    [[nodiscard]] std::optional<int> historyIndex() const override { return steps_back_; }

private:
    int steps_back_{1};
};

// ============================================================================
// Unary operator nodes
// ============================================================================

class UnaryNode : public FormExprNode {
public:
    explicit UnaryNode(std::shared_ptr<FormExprNode> child) : child_(std::move(child)) {}

    [[nodiscard]] bool hasTest() const noexcept override { return child_ && child_->hasTest(); }
    [[nodiscard]] bool hasTrial() const noexcept override { return child_ && child_->hasTrial(); }

    [[nodiscard]] std::vector<const FormExprNode*> children() const override {
        return {child_.get()};
    }

    [[nodiscard]] std::vector<std::shared_ptr<FormExprNode>> childrenShared() const override {
        return {child_};
    }

protected:
    std::shared_ptr<FormExprNode> child_;
};

class GradientNode final : public UnaryNode {
public:
    using UnaryNode::UnaryNode;

    [[nodiscard]] FormExprType type() const noexcept override { return FormExprType::Gradient; }
    [[nodiscard]] std::string toString() const override { return "grad(" + child_->toString() + ")"; }
};

class DivergenceNode final : public UnaryNode {
public:
    using UnaryNode::UnaryNode;

    [[nodiscard]] FormExprType type() const noexcept override { return FormExprType::Divergence; }
    [[nodiscard]] std::string toString() const override { return "div(" + child_->toString() + ")"; }
};

class CurlNode final : public UnaryNode {
public:
    using UnaryNode::UnaryNode;

    [[nodiscard]] FormExprType type() const noexcept override { return FormExprType::Curl; }
    [[nodiscard]] std::string toString() const override { return "curl(" + child_->toString() + ")"; }
};

class HessianNode final : public UnaryNode {
public:
    using UnaryNode::UnaryNode;

    [[nodiscard]] FormExprType type() const noexcept override { return FormExprType::Hessian; }
    [[nodiscard]] std::string toString() const override { return "H(" + child_->toString() + ")"; }
};

class TimeDerivativeNode final : public UnaryNode {
public:
    TimeDerivativeNode(std::shared_ptr<FormExprNode> child, int order)
        : UnaryNode(std::move(child)), order_(order)
    {
    }

    [[nodiscard]] FormExprType type() const noexcept override { return FormExprType::TimeDerivative; }
    [[nodiscard]] std::string toString() const override
    {
        if (order_ == 1) {
            return "dt(" + child_->toString() + ")";
        }
        return "dt(" + child_->toString() + "," + std::to_string(order_) + ")";
    }
    [[nodiscard]] std::optional<int> timeDerivativeOrder() const override { return order_; }

private:
    int order_{1};
};

class RestrictMinusNode final : public UnaryNode {
public:
    using UnaryNode::UnaryNode;

    [[nodiscard]] FormExprType type() const noexcept override { return FormExprType::RestrictMinus; }
    [[nodiscard]] std::string toString() const override { return child_->toString() + "(-)"; }
};

class RestrictPlusNode final : public UnaryNode {
public:
    using UnaryNode::UnaryNode;

    [[nodiscard]] FormExprType type() const noexcept override { return FormExprType::RestrictPlus; }
    [[nodiscard]] std::string toString() const override { return child_->toString() + "(+)"; }
};

class JumpNode final : public UnaryNode {
public:
    using UnaryNode::UnaryNode;

    [[nodiscard]] FormExprType type() const noexcept override { return FormExprType::Jump; }
    [[nodiscard]] std::string toString() const override { return "[[" + child_->toString() + "]]"; }
};

class AverageNode final : public UnaryNode {
public:
    using UnaryNode::UnaryNode;

    [[nodiscard]] FormExprType type() const noexcept override { return FormExprType::Average; }
    [[nodiscard]] std::string toString() const override { return "{" + child_->toString() + "}"; }
};

class NegateNode final : public UnaryNode {
public:
    using UnaryNode::UnaryNode;

    [[nodiscard]] FormExprType type() const noexcept override { return FormExprType::Negate; }
    [[nodiscard]] std::string toString() const override { return "-(" + child_->toString() + ")"; }
};

class TransposeNode final : public UnaryNode {
public:
    using UnaryNode::UnaryNode;

    [[nodiscard]] FormExprType type() const noexcept override { return FormExprType::Transpose; }
    [[nodiscard]] std::string toString() const override { return "transpose(" + child_->toString() + ")"; }
};

class TraceNode final : public UnaryNode {
public:
    using UnaryNode::UnaryNode;

    [[nodiscard]] FormExprType type() const noexcept override { return FormExprType::Trace; }
    [[nodiscard]] std::string toString() const override { return "trace(" + child_->toString() + ")"; }
};

class DeterminantNode final : public UnaryNode {
public:
    using UnaryNode::UnaryNode;

    [[nodiscard]] FormExprType type() const noexcept override { return FormExprType::Determinant; }
    [[nodiscard]] std::string toString() const override { return "det(" + child_->toString() + ")"; }
};

class InverseNode final : public UnaryNode {
public:
    using UnaryNode::UnaryNode;

    [[nodiscard]] FormExprType type() const noexcept override { return FormExprType::Inverse; }
    [[nodiscard]] std::string toString() const override { return "inv(" + child_->toString() + ")"; }
};

class CofactorNode final : public UnaryNode {
public:
    using UnaryNode::UnaryNode;

    [[nodiscard]] FormExprType type() const noexcept override { return FormExprType::Cofactor; }
    [[nodiscard]] std::string toString() const override { return "cofactor(" + child_->toString() + ")"; }
};

class DeviatorNode final : public UnaryNode {
public:
    using UnaryNode::UnaryNode;

    [[nodiscard]] FormExprType type() const noexcept override { return FormExprType::Deviator; }
    [[nodiscard]] std::string toString() const override { return "dev(" + child_->toString() + ")"; }
};

class SymmetricPartNode final : public UnaryNode {
public:
    using UnaryNode::UnaryNode;

    [[nodiscard]] FormExprType type() const noexcept override { return FormExprType::SymmetricPart; }
    [[nodiscard]] std::string toString() const override { return "sym(" + child_->toString() + ")"; }
};

class SkewPartNode final : public UnaryNode {
public:
    using UnaryNode::UnaryNode;

    [[nodiscard]] FormExprType type() const noexcept override { return FormExprType::SkewPart; }
    [[nodiscard]] std::string toString() const override { return "skew(" + child_->toString() + ")"; }
};

class NormNode final : public UnaryNode {
public:
    using UnaryNode::UnaryNode;

    [[nodiscard]] FormExprType type() const noexcept override { return FormExprType::Norm; }
    [[nodiscard]] std::string toString() const override { return "norm(" + child_->toString() + ")"; }
};

class NormalizeNode final : public UnaryNode {
public:
    using UnaryNode::UnaryNode;

    [[nodiscard]] FormExprType type() const noexcept override { return FormExprType::Normalize; }
    [[nodiscard]] std::string toString() const override { return "normalize(" + child_->toString() + ")"; }
};

class AbsoluteValueNode final : public UnaryNode {
public:
    using UnaryNode::UnaryNode;

    [[nodiscard]] FormExprType type() const noexcept override { return FormExprType::AbsoluteValue; }
    [[nodiscard]] std::string toString() const override { return "abs(" + child_->toString() + ")"; }
};

class SignNode final : public UnaryNode {
public:
    using UnaryNode::UnaryNode;

    [[nodiscard]] FormExprType type() const noexcept override { return FormExprType::Sign; }
    [[nodiscard]] std::string toString() const override { return "sign(" + child_->toString() + ")"; }
};

class SqrtNode final : public UnaryNode {
public:
    using UnaryNode::UnaryNode;

    [[nodiscard]] FormExprType type() const noexcept override { return FormExprType::Sqrt; }
    [[nodiscard]] std::string toString() const override { return "sqrt(" + child_->toString() + ")"; }
};

class ExpNode final : public UnaryNode {
public:
    using UnaryNode::UnaryNode;

    [[nodiscard]] FormExprType type() const noexcept override { return FormExprType::Exp; }
    [[nodiscard]] std::string toString() const override { return "exp(" + child_->toString() + ")"; }
};

class LogNode final : public UnaryNode {
public:
    using UnaryNode::UnaryNode;

    [[nodiscard]] FormExprType type() const noexcept override { return FormExprType::Log; }
    [[nodiscard]] std::string toString() const override { return "log(" + child_->toString() + ")"; }
};

class SymmetricEigenvalueNode final : public UnaryNode {
public:
    SymmetricEigenvalueNode(std::shared_ptr<FormExprNode> child, int which)
        : UnaryNode(std::move(child)), which_(which)
    {}

    [[nodiscard]] FormExprType type() const noexcept override { return FormExprType::SymmetricEigenvalue; }
    [[nodiscard]] std::string toString() const override
    {
        return "eig_sym(" + child_->toString() + ", " + std::to_string(which_) + ")";
    }
    [[nodiscard]] std::optional<int> eigenIndex() const override { return which_; }

private:
    int which_{0};
};

class SymmetricEigenvalueDirectionalDerivativeNode final : public FormExprNode {
public:
    SymmetricEigenvalueDirectionalDerivativeNode(std::shared_ptr<FormExprNode> a,
                                                 std::shared_ptr<FormExprNode> da,
                                                 int which)
        : a_(std::move(a))
        , da_(std::move(da))
        , which_(which)
    {}

    [[nodiscard]] FormExprType type() const noexcept override { return FormExprType::SymmetricEigenvalueDirectionalDerivative; }
    [[nodiscard]] std::string toString() const override
    {
        return "eig_sym_dd(" + a_->toString() + ", " + da_->toString() + ", " + std::to_string(which_) + ")";
    }
    [[nodiscard]] std::optional<int> eigenIndex() const override { return which_; }

    [[nodiscard]] bool hasTest() const noexcept override
    {
        return (a_ && a_->hasTest()) || (da_ && da_->hasTest());
    }

    [[nodiscard]] bool hasTrial() const noexcept override
    {
        return (a_ && a_->hasTrial()) || (da_ && da_->hasTrial());
    }

    [[nodiscard]] std::vector<const FormExprNode*> children() const override
    {
        return {a_.get(), da_.get()};
    }

    [[nodiscard]] std::vector<std::shared_ptr<FormExprNode>> childrenShared() const override
    {
        return {a_, da_};
    }

private:
    std::shared_ptr<FormExprNode> a_;
    std::shared_ptr<FormExprNode> da_;
    int which_{0};
};

class SymmetricEigenvalueDirectionalDerivativeWrtANode final : public FormExprNode {
public:
    SymmetricEigenvalueDirectionalDerivativeWrtANode(std::shared_ptr<FormExprNode> a,
                                                     std::shared_ptr<FormExprNode> b,
                                                     std::shared_ptr<FormExprNode> da,
                                                     int which)
        : a_(std::move(a))
        , b_(std::move(b))
        , da_(std::move(da))
        , which_(which)
    {}

    [[nodiscard]] FormExprType type() const noexcept override { return FormExprType::SymmetricEigenvalueDirectionalDerivativeWrtA; }
    [[nodiscard]] std::string toString() const override
    {
        return "eig_sym_ddA(" + a_->toString() + ", " + b_->toString() + ", " + da_->toString() + ", " +
               std::to_string(which_) + ")";
    }

    [[nodiscard]] bool hasTest() const noexcept override
    {
        return (a_ && a_->hasTest()) || (b_ && b_->hasTest()) || (da_ && da_->hasTest());
    }
    [[nodiscard]] bool hasTrial() const noexcept override
    {
        return (a_ && a_->hasTrial()) || (b_ && b_->hasTrial()) || (da_ && da_->hasTrial());
    }

    [[nodiscard]] std::optional<int> eigenIndex() const override { return which_; }

    [[nodiscard]] std::vector<const FormExprNode*> children() const override
    {
        return {a_.get(), b_.get(), da_.get()};
    }

    [[nodiscard]] std::vector<std::shared_ptr<FormExprNode>> childrenShared() const override
    {
        return {a_, b_, da_};
    }

private:
    std::shared_ptr<FormExprNode> a_;
    std::shared_ptr<FormExprNode> b_;
    std::shared_ptr<FormExprNode> da_;
    int which_{0};
};

// ============================================================================
// Binary operator nodes
// ============================================================================

class BinaryNode : public FormExprNode {
public:
    BinaryNode(std::shared_ptr<FormExprNode> lhs, std::shared_ptr<FormExprNode> rhs)
        : lhs_(std::move(lhs)), rhs_(std::move(rhs))
    {
    }

    [[nodiscard]] bool hasTest() const noexcept override {
        return (lhs_ && lhs_->hasTest()) || (rhs_ && rhs_->hasTest());
    }
    [[nodiscard]] bool hasTrial() const noexcept override {
        return (lhs_ && lhs_->hasTrial()) || (rhs_ && rhs_->hasTrial());
    }

    [[nodiscard]] std::vector<const FormExprNode*> children() const override {
        return {lhs_.get(), rhs_.get()};
    }

    [[nodiscard]] std::vector<std::shared_ptr<FormExprNode>> childrenShared() const override {
        return {lhs_, rhs_};
    }

protected:
    std::shared_ptr<FormExprNode> lhs_, rhs_;
};

class AddNode final : public BinaryNode {
public:
    using BinaryNode::BinaryNode;
    [[nodiscard]] FormExprType type() const noexcept override { return FormExprType::Add; }
    [[nodiscard]] std::string toString() const override {
        return "(" + lhs_->toString() + " + " + rhs_->toString() + ")";
    }
};

class SubtractNode final : public BinaryNode {
public:
    using BinaryNode::BinaryNode;
    [[nodiscard]] FormExprType type() const noexcept override { return FormExprType::Subtract; }
    [[nodiscard]] std::string toString() const override {
        return "(" + lhs_->toString() + " - " + rhs_->toString() + ")";
    }
};

class MultiplyNode final : public BinaryNode {
public:
    using BinaryNode::BinaryNode;
    [[nodiscard]] FormExprType type() const noexcept override { return FormExprType::Multiply; }
    [[nodiscard]] std::string toString() const override {
        return "(" + lhs_->toString() + " * " + rhs_->toString() + ")";
    }
};

class DivideNode final : public BinaryNode {
public:
    using BinaryNode::BinaryNode;
    [[nodiscard]] FormExprType type() const noexcept override { return FormExprType::Divide; }
    [[nodiscard]] std::string toString() const override {
        return "(" + lhs_->toString() + " / " + rhs_->toString() + ")";
    }
};

class InnerProductNode final : public BinaryNode {
public:
    using BinaryNode::BinaryNode;
    [[nodiscard]] FormExprType type() const noexcept override { return FormExprType::InnerProduct; }
    [[nodiscard]] std::string toString() const override {
        return "inner(" + lhs_->toString() + ", " + rhs_->toString() + ")";
    }
};

class DoubleContractionNode final : public BinaryNode {
public:
    using BinaryNode::BinaryNode;
    [[nodiscard]] FormExprType type() const noexcept override { return FormExprType::DoubleContraction; }
    [[nodiscard]] std::string toString() const override {
        return "doubleContraction(" + lhs_->toString() + ", " + rhs_->toString() + ")";
    }
};

class OuterProductNode final : public BinaryNode {
public:
    using BinaryNode::BinaryNode;
    [[nodiscard]] FormExprType type() const noexcept override { return FormExprType::OuterProduct; }
    [[nodiscard]] std::string toString() const override {
        return "outer(" + lhs_->toString() + ", " + rhs_->toString() + ")";
    }
};

class CrossProductNode final : public BinaryNode {
public:
    using BinaryNode::BinaryNode;
    [[nodiscard]] FormExprType type() const noexcept override { return FormExprType::CrossProduct; }
    [[nodiscard]] std::string toString() const override {
        return "cross(" + lhs_->toString() + ", " + rhs_->toString() + ")";
    }
};

class PowerNode final : public BinaryNode {
public:
    using BinaryNode::BinaryNode;
    [[nodiscard]] FormExprType type() const noexcept override { return FormExprType::Power; }
    [[nodiscard]] std::string toString() const override {
        return "pow(" + lhs_->toString() + ", " + rhs_->toString() + ")";
    }
};

class MinimumNode final : public BinaryNode {
public:
    using BinaryNode::BinaryNode;
    [[nodiscard]] FormExprType type() const noexcept override { return FormExprType::Minimum; }
    [[nodiscard]] std::string toString() const override {
        return "min(" + lhs_->toString() + ", " + rhs_->toString() + ")";
    }
};

class MaximumNode final : public BinaryNode {
public:
    using BinaryNode::BinaryNode;
    [[nodiscard]] FormExprType type() const noexcept override { return FormExprType::Maximum; }
    [[nodiscard]] std::string toString() const override {
        return "max(" + lhs_->toString() + ", " + rhs_->toString() + ")";
    }
};

class ComparisonNode final : public BinaryNode {
public:
    ComparisonNode(FormExprType cmp_type, std::shared_ptr<FormExprNode> lhs, std::shared_ptr<FormExprNode> rhs)
        : BinaryNode(std::move(lhs), std::move(rhs)), cmp_type_(cmp_type)
    {
    }

    [[nodiscard]] FormExprType type() const noexcept override { return cmp_type_; }

    [[nodiscard]] std::string toString() const override {
        std::string op;
        switch (cmp_type_) {
            case FormExprType::Less: op = "<"; break;
            case FormExprType::LessEqual: op = "<="; break;
            case FormExprType::Greater: op = ">"; break;
            case FormExprType::GreaterEqual: op = ">="; break;
            case FormExprType::Equal: op = "=="; break;
            case FormExprType::NotEqual: op = "!="; break;
            default: op = "?"; break;
        }
        return "(" + lhs_->toString() + " " + op + " " + rhs_->toString() + ")";
    }

private:
    FormExprType cmp_type_{FormExprType::Equal};
};

class ComponentNode final : public UnaryNode {
public:
    ComponentNode(std::shared_ptr<FormExprNode> child, int i, int j)
        : UnaryNode(std::move(child))
        , i_(i)
        , j_(j)
    {
    }

    [[nodiscard]] FormExprType type() const noexcept override { return FormExprType::Component; }
    [[nodiscard]] std::string toString() const override {
        if (j_ >= 0) {
            return child_->toString() + "[" + std::to_string(i_) + "," + std::to_string(j_) + "]";
        }
        return child_->toString() + "[" + std::to_string(i_) + "]";
    }

    [[nodiscard]] std::optional<int> componentIndex0() const override { return i_; }
    [[nodiscard]] std::optional<int> componentIndex1() const override { return j_ >= 0 ? std::optional<int>(j_) : std::nullopt; }

private:
    int i_{0};
    int j_{-1};
};

class IndexedAccessNode final : public UnaryNode {
public:
    IndexedAccessNode(std::shared_ptr<FormExprNode> child, const Index& i)
        : UnaryNode(std::move(child))
    {
        rank_ = 1;
        ids_.fill(-1);
        extents_.fill(-1);
        variances_.fill(tensor::IndexVariance::None);
        ids_[0] = i.id();
        extents_[0] = i.extent();
        names_[0] = i.name();
    }

    IndexedAccessNode(std::shared_ptr<FormExprNode> child, const Index& i, const Index& j)
        : UnaryNode(std::move(child))
    {
        rank_ = 2;
        ids_.fill(-1);
        extents_.fill(-1);
        variances_.fill(tensor::IndexVariance::None);
        ids_[0] = i.id();
        ids_[1] = j.id();
        extents_[0] = i.extent();
        extents_[1] = j.extent();
        names_[0] = i.name();
        names_[1] = j.name();
    }

    IndexedAccessNode(std::shared_ptr<FormExprNode> child,
                      int rank,
                      std::array<int, 4> ids,
                      std::array<int, 4> extents,
                      std::array<std::string, 4> names,
                      std::array<tensor::IndexVariance, 4> variances)
        : UnaryNode(std::move(child))
        , rank_(rank)
        , ids_(std::move(ids))
        , extents_(std::move(extents))
        , names_(std::move(names))
        , variances_(std::move(variances))
    {
    }

    [[nodiscard]] FormExprType type() const noexcept override { return FormExprType::IndexedAccess; }

    [[nodiscard]] std::string toString() const override
    {
        if (rank_ < 1 || rank_ > 4) {
            return child_->toString() + "(?)";
        }

        const auto indexLabel = [&](int k) -> std::string {
            const auto& nm = names_[static_cast<std::size_t>(k)];
            if (!nm.empty()) {
                return nm;
            }
            const int id = ids_[static_cast<std::size_t>(k)];
            if (id >= 0) {
                return "i" + std::to_string(id);
            }
            return "?";
        };

        // ---- Special pretty-print cases (tensor-calculus readability) ----
        // Symmetrization / antisymmetrization tags.
        if (rank_ == 2 && (child_->type() == FormExprType::SymmetricPart || child_->type() == FormExprType::SkewPart)) {
            const auto kids = child_->childrenShared();
            const std::string base = (!kids.empty() && kids[0]) ? kids[0]->toString() : child_->toString();
            const char open = (child_->type() == FormExprType::SymmetricPart) ? '(' : '[';
            const char close = (child_->type() == FormExprType::SymmetricPart) ? ')' : ']';
            return base + "_{" + open + indexLabel(0) + indexLabel(1) + close + "}";
        }

        // Kronecker delta (identity tensor) in index form.
        if (rank_ == 2 && child_->type() == FormExprType::Identity) {
            return std::string("\u03B4") + "_{" + indexLabel(0) + indexLabel(1) + "}";
        }

        // Levi-Civita / permutation tensor appearances via common ops.
        if (rank_ == 1 && child_->type() == FormExprType::CrossProduct) {
            const auto kids = child_->childrenShared();
            if (kids.size() == 2u && kids[0] && kids[1]) {
                const std::string i = indexLabel(0);
                const std::string j = "j";
                const std::string k = "k";
                return "(" + std::string("\u03B5") + "_{" + i + j + k + "} * " +
                       kids[0]->toString() + "_{" + j + "} * " +
                       kids[1]->toString() + "_{" + k + "})";
            }
        }
        if (rank_ == 1 && child_->type() == FormExprType::Curl) {
            const auto kids = child_->childrenShared();
            if (kids.size() == 1u && kids[0]) {
                const std::string i = indexLabel(0);
                const std::string j = "j";
                const std::string k = "k";
                return "(" + std::string("\u03B5") + "_{" + i + j + k + "} * grad(" +
                       kids[0]->toString() + ")_{" + k + j + "})";
            }
        }

        // Default indexed access printing.
        std::string out = child_->toString();
        out += "_{";
        for (int k = 0; k < rank_; ++k) {
            out += indexLabel(k);
        }
        out += "}";
        return out;
    }

    [[nodiscard]] std::optional<int> indexRank() const override { return rank_; }
    [[nodiscard]] std::optional<std::array<int, 4>> indexIds() const override { return ids_; }
    [[nodiscard]] std::optional<std::array<int, 4>> indexExtents() const override { return extents_; }
    [[nodiscard]] std::optional<std::array<std::string_view, 4>> indexNames() const override
    {
        std::array<std::string_view, 4> out{};
        for (std::size_t k = 0; k < out.size(); ++k) {
            out[k] = names_[k];
        }
        return out;
    }

    [[nodiscard]] std::optional<std::array<tensor::IndexVariance, 4>> indexVariances() const override
    {
        return variances_;
    }

private:
    int rank_{0};
    std::array<int, 4> ids_{};
    std::array<int, 4> extents_{};
    std::array<std::string, 4> names_{};
    std::array<tensor::IndexVariance, 4> variances_{};
};

class ConditionalNode final : public FormExprNode {
public:
    ConditionalNode(std::shared_ptr<FormExprNode> cond,
                    std::shared_ptr<FormExprNode> then_expr,
                    std::shared_ptr<FormExprNode> else_expr)
        : cond_(std::move(cond))
        , then_(std::move(then_expr))
        , else_(std::move(else_expr))
    {
    }

    [[nodiscard]] FormExprType type() const noexcept override { return FormExprType::Conditional; }
    [[nodiscard]] std::string toString() const override {
        return "if_then_else(" + cond_->toString() + ", " + then_->toString() + ", " + else_->toString() + ")";
    }

    [[nodiscard]] bool hasTest() const noexcept override {
        return (cond_ && cond_->hasTest()) || (then_ && then_->hasTest()) || (else_ && else_->hasTest());
    }
    [[nodiscard]] bool hasTrial() const noexcept override {
        return (cond_ && cond_->hasTrial()) || (then_ && then_->hasTrial()) || (else_ && else_->hasTrial());
    }

    [[nodiscard]] std::vector<const FormExprNode*> children() const override {
        return {cond_.get(), then_.get(), else_.get()};
    }

    [[nodiscard]] std::vector<std::shared_ptr<FormExprNode>> childrenShared() const override {
        return {cond_, then_, else_};
    }

private:
    std::shared_ptr<FormExprNode> cond_;
    std::shared_ptr<FormExprNode> then_;
    std::shared_ptr<FormExprNode> else_;
};

class AsVectorNode final : public FormExprNode {
public:
    explicit AsVectorNode(std::vector<std::shared_ptr<FormExprNode>> components)
        : components_(std::move(components))
    {
    }

    [[nodiscard]] FormExprType type() const noexcept override { return FormExprType::AsVector; }

    [[nodiscard]] std::string toString() const override
    {
        std::string s = "as_vector(";
        for (std::size_t i = 0; i < components_.size(); ++i) {
            if (i > 0) s += ", ";
            s += components_[i] ? components_[i]->toString() : "<null>";
        }
        s += ")";
        return s;
    }

    [[nodiscard]] bool hasTest() const noexcept override
    {
        for (const auto& c : components_) {
            if (c && c->hasTest()) return true;
        }
        return false;
    }

    [[nodiscard]] bool hasTrial() const noexcept override
    {
        for (const auto& c : components_) {
            if (c && c->hasTrial()) return true;
        }
        return false;
    }

    [[nodiscard]] std::vector<std::shared_ptr<FormExprNode>> childrenShared() const override
    {
        return components_;
    }

    [[nodiscard]] std::vector<const FormExprNode*> children() const override
    {
        std::vector<const FormExprNode*> out;
        out.reserve(components_.size());
        for (const auto& c : components_) out.push_back(c.get());
        return out;
    }

private:
    std::vector<std::shared_ptr<FormExprNode>> components_;
};

class AsTensorNode final : public FormExprNode {
public:
    AsTensorNode(int rows,
                 int cols,
                 std::vector<std::shared_ptr<FormExprNode>> entries)
        : rows_(rows)
        , cols_(cols)
        , entries_(std::move(entries))
    {
    }

    [[nodiscard]] FormExprType type() const noexcept override { return FormExprType::AsTensor; }

    [[nodiscard]] std::string toString() const override
    {
        std::string s = "as_tensor(";
        for (int r = 0; r < rows_; ++r) {
            if (r > 0) s += ", ";
            s += "[";
            for (int c = 0; c < cols_; ++c) {
                if (c > 0) s += ", ";
                const auto idx = static_cast<std::size_t>(r * cols_ + c);
                s += entries_[idx] ? entries_[idx]->toString() : "<null>";
            }
            s += "]";
        }
        s += ")";
        return s;
    }

    [[nodiscard]] bool hasTest() const noexcept override
    {
        for (const auto& e : entries_) {
            if (e && e->hasTest()) return true;
        }
        return false;
    }

    [[nodiscard]] bool hasTrial() const noexcept override
    {
        for (const auto& e : entries_) {
            if (e && e->hasTrial()) return true;
        }
        return false;
    }

    [[nodiscard]] std::optional<int> tensorRows() const override { return rows_; }
    [[nodiscard]] std::optional<int> tensorCols() const override { return cols_; }

    [[nodiscard]] std::vector<std::shared_ptr<FormExprNode>> childrenShared() const override
    {
        return entries_;
    }

    [[nodiscard]] std::vector<const FormExprNode*> children() const override
    {
        std::vector<const FormExprNode*> out;
        out.reserve(entries_.size());
        for (const auto& e : entries_) out.push_back(e.get());
        return out;
    }

private:
    int rows_{0};
    int cols_{0};
    std::vector<std::shared_ptr<FormExprNode>> entries_;
};

class ConstitutiveNode final : public FormExprNode {
public:
    ConstitutiveNode(std::shared_ptr<const ConstitutiveModel> model,
                     std::vector<std::shared_ptr<FormExprNode>> inputs)
        : model_(std::move(model))
        , inputs_(std::move(inputs))
    {
        if (inputs_.empty()) {
            throw std::invalid_argument("ConstitutiveNode: requires at least one input");
        }
    }

    [[nodiscard]] FormExprType type() const noexcept override { return FormExprType::Constitutive; }
    [[nodiscard]] std::string toString() const override {
        std::string s = "constitutive(";
        for (std::size_t i = 0; i < inputs_.size(); ++i) {
            if (i > 0) s += ", ";
            s += inputs_[i] ? inputs_[i]->toString() : "<null>";
        }
        s += ")";
        return s;
    }

    [[nodiscard]] bool hasTest() const noexcept override
    {
        for (const auto& c : inputs_) {
            if (c && c->hasTest()) return true;
        }
        return false;
    }

    [[nodiscard]] bool hasTrial() const noexcept override
    {
        for (const auto& c : inputs_) {
            if (c && c->hasTrial()) return true;
        }
        return false;
    }

    [[nodiscard]] const ConstitutiveModel* constitutiveModel() const override { return model_.get(); }
    [[nodiscard]] std::shared_ptr<const ConstitutiveModel> constitutiveModelShared() const override { return model_; }

    [[nodiscard]] std::vector<std::shared_ptr<FormExprNode>> childrenShared() const override
    {
        return inputs_;
    }

    [[nodiscard]] std::vector<const FormExprNode*> children() const override
    {
        std::vector<const FormExprNode*> out;
        out.reserve(inputs_.size());
        for (const auto& c : inputs_) out.push_back(c.get());
        return out;
    }

private:
    std::shared_ptr<const ConstitutiveModel> model_;
    std::vector<std::shared_ptr<FormExprNode>> inputs_;
};

class ConstitutiveOutputNode final : public UnaryNode {
public:
    ConstitutiveOutputNode(std::shared_ptr<FormExprNode> call, std::size_t output_index)
        : UnaryNode(std::move(call))
        , output_index_(output_index)
    {
        if (!child_) {
            throw std::invalid_argument("ConstitutiveOutputNode: requires a call expression");
        }
        if (child_->type() != FormExprType::Constitutive) {
            throw std::invalid_argument("ConstitutiveOutputNode: child must be a constitutive(...) expression");
        }
        const auto* model = child_->constitutiveModel();
        if (!model) {
            throw std::invalid_argument("ConstitutiveOutputNode: child has no model");
        }
        if (output_index_ >= model->outputCount()) {
            throw std::invalid_argument("ConstitutiveOutputNode: output_index out of range");
        }
    }

    [[nodiscard]] FormExprType type() const noexcept override { return FormExprType::ConstitutiveOutput; }
    [[nodiscard]] std::string toString() const override {
        return child_->toString() + ".output(" + std::to_string(output_index_) + ")";
    }

    [[nodiscard]] std::optional<int> constitutiveOutputIndex() const override
    {
        return static_cast<int>(output_index_);
    }

private:
    std::size_t output_index_{0};
};

// ============================================================================
// Integral nodes
// ============================================================================

class CellIntegralNode final : public UnaryNode {
public:
    using UnaryNode::UnaryNode;

    [[nodiscard]] FormExprType type() const noexcept override { return FormExprType::CellIntegral; }
    [[nodiscard]] std::string toString() const override {
        return "integral_K(" + child_->toString() + ") dx";
    }
};

class BoundaryIntegralNode final : public UnaryNode {
public:
    BoundaryIntegralNode(std::shared_ptr<FormExprNode> child, int boundary_marker)
        : UnaryNode(std::move(child)), boundary_marker_(boundary_marker)
    {
    }

    [[nodiscard]] FormExprType type() const noexcept override { return FormExprType::BoundaryIntegral; }
    [[nodiscard]] std::string toString() const override {
        std::string marker_str = boundary_marker_ >= 0 ? std::to_string(boundary_marker_) : "all";
        return "integral_Gamma_" + marker_str + "(" + child_->toString() + ") ds";
    }

    [[nodiscard]] std::optional<int> boundaryMarker() const override { return boundary_marker_; }

private:
    int boundary_marker_{-1};
};

class InteriorFaceIntegralNode final : public UnaryNode {
public:
    using UnaryNode::UnaryNode;

    [[nodiscard]] FormExprType type() const noexcept override { return FormExprType::InteriorFaceIntegral; }
    [[nodiscard]] std::string toString() const override {
        return "integral_F(" + child_->toString() + ") dS";
    }
};

class InterfaceIntegralNode final : public UnaryNode {
public:
    InterfaceIntegralNode(std::shared_ptr<FormExprNode> child, int interface_marker)
        : UnaryNode(std::move(child)), interface_marker_(interface_marker)
    {
    }

    [[nodiscard]] FormExprType type() const noexcept override { return FormExprType::InterfaceIntegral; }
    [[nodiscard]] std::string toString() const override {
        std::string marker_str = interface_marker_ >= 0 ? std::to_string(interface_marker_) : "all";
        return "integral_I_" + marker_str + "(" + child_->toString() + ") dI";
    }

    [[nodiscard]] std::optional<int> interfaceMarker() const override { return interface_marker_; }

private:
    int interface_marker_{-1};
};

} // namespace

// ============================================================================
// FormExpr implementation
// ============================================================================

FormExpr::FormExpr() = default;

FormExpr::FormExpr(std::shared_ptr<FormExprNode> node)
    : node_(std::move(node))
{
}

namespace {

FormExprNode::SpaceSignature makeSpaceSignature(const spaces::FunctionSpace& space)
{
    FormExprNode::SpaceSignature sig;
    sig.space_type = space.space_type();
    sig.field_type = space.field_type();
    sig.continuity = space.continuity();
    sig.value_dimension = space.value_dimension();
    sig.topological_dimension = space.topological_dimension();
    sig.polynomial_order = space.polynomial_order();
    sig.element_type = space.element_type();
    return sig;
}

} // namespace

FormExpr FormExpr::testFunction(const spaces::FunctionSpace& space, std::string name)
{
    return FormExpr(std::make_shared<TestFunctionNode>(makeSpaceSignature(space), std::move(name)));
}

FormExpr FormExpr::trialFunction(const spaces::FunctionSpace& space, std::string name)
{
    return FormExpr(std::make_shared<TrialFunctionNode>(makeSpaceSignature(space), std::move(name)));
}

FormExpr FormExpr::discreteField(FieldId field, const spaces::FunctionSpace& space, std::string name)
{
    if (field == INVALID_FIELD_ID) {
        throw std::invalid_argument("FormExpr::discreteField: invalid FieldId");
    }
    return FormExpr(std::make_shared<DiscreteFieldNode>(field, makeSpaceSignature(space), std::move(name)));
}

FormExpr FormExpr::stateField(FieldId field, const spaces::FunctionSpace& space, std::string name)
{
    return FormExpr(std::make_shared<StateFieldNode>(field, makeSpaceSignature(space), std::move(name)));
}

FormExpr FormExpr::testFunction(const FormExprNode::SpaceSignature& signature, std::string name)
{
    return FormExpr(std::make_shared<TestFunctionNode>(signature, std::move(name)));
}

FormExpr FormExpr::trialFunction(const FormExprNode::SpaceSignature& signature, std::string name)
{
    return FormExpr(std::make_shared<TrialFunctionNode>(signature, std::move(name)));
}

FormExpr FormExpr::discreteField(FieldId field, const FormExprNode::SpaceSignature& signature, std::string name)
{
    if (field == INVALID_FIELD_ID) {
        throw std::invalid_argument("FormExpr::discreteField: invalid FieldId");
    }
    return FormExpr(std::make_shared<DiscreteFieldNode>(field, signature, std::move(name)));
}

FormExpr FormExpr::stateField(FieldId field, const FormExprNode::SpaceSignature& signature, std::string name)
{
    return FormExpr(std::make_shared<StateFieldNode>(field, signature, std::move(name)));
}

FormExpr FormExpr::coefficient(std::string name, ScalarCoefficient func)
{
    return FormExpr(std::make_shared<CoefficientNode>(std::move(name), std::move(func)));
}

FormExpr FormExpr::coefficient(std::string name, TimeScalarCoefficient func)
{
    return FormExpr(std::make_shared<CoefficientNode>(std::move(name), std::move(func)));
}

FormExpr FormExpr::coefficient(std::string name, VectorCoefficient func)
{
    return FormExpr(std::make_shared<CoefficientNode>(std::move(name), std::move(func)));
}

FormExpr FormExpr::coefficient(std::string name, MatrixCoefficient func)
{
    return FormExpr(std::make_shared<CoefficientNode>(std::move(name), std::move(func)));
}

FormExpr FormExpr::coefficient(std::string name, Tensor3Coefficient func)
{
    return FormExpr(std::make_shared<CoefficientNode>(std::move(name), std::move(func)));
}

FormExpr FormExpr::coefficient(std::string name, Tensor4Coefficient func)
{
    return FormExpr(std::make_shared<CoefficientNode>(std::move(name), std::move(func)));
}

FormExpr FormExpr::parameter(std::string name)
{
    if (name.empty()) {
        throw std::invalid_argument("FormExpr::parameter: empty name");
    }
    return FormExpr(std::make_shared<ParameterSymbolNode>(std::move(name)));
}

FormExpr FormExpr::parameterRef(std::uint32_t slot)
{
    return FormExpr(std::make_shared<ParameterRefNode>(slot));
}

FormExpr FormExpr::constant(Real value)
{
    return FormExpr(std::make_shared<ConstantNode>(value));
}

FormExpr FormExpr::boundaryIntegral(FormExpr integrand, int boundary_marker, std::string name)
{
    if (!integrand.isValid()) {
        throw std::invalid_argument("FormExpr::boundaryIntegral: invalid integrand");
    }
    if (boundary_marker < 0) {
        throw std::invalid_argument("FormExpr::boundaryIntegral: boundary_marker must be >= 0");
    }
    if (name.empty()) {
        throw std::invalid_argument("FormExpr::boundaryIntegral: empty name");
    }
    return FormExpr(std::make_shared<BoundaryFunctionalSymbolNode>(integrand.nodeShared(),
                                                                   boundary_marker,
                                                                   std::move(name)));
}

FormExpr FormExpr::boundaryIntegralValue(std::string name)
{
    if (name.empty()) {
        throw std::invalid_argument("FormExpr::boundaryIntegralValue: empty name");
    }
    return FormExpr(std::make_shared<BoundaryIntegralSymbolNode>(std::move(name)));
}

FormExpr FormExpr::boundaryIntegralRef(std::uint32_t slot)
{
    return FormExpr(std::make_shared<BoundaryIntegralRefNode>(slot));
}

FormExpr FormExpr::auxiliaryState(std::string name)
{
    if (name.empty()) {
        throw std::invalid_argument("FormExpr::auxiliaryState: empty name");
    }
    return FormExpr(std::make_shared<AuxiliaryStateSymbolNode>(std::move(name)));
}

FormExpr FormExpr::auxiliaryStateRef(std::uint32_t slot)
{
    return FormExpr(std::make_shared<AuxiliaryStateRefNode>(slot));
}

FormExpr FormExpr::materialStateOldRef(std::uint32_t offset_bytes)
{
    return FormExpr(std::make_shared<MaterialStateOldRefNode>(offset_bytes));
}

FormExpr FormExpr::materialStateWorkRef(std::uint32_t offset_bytes)
{
    return FormExpr(std::make_shared<MaterialStateWorkRefNode>(offset_bytes));
}

FormExpr FormExpr::previousSolution(int steps_back)
{
    if (steps_back <= 0) {
        throw std::invalid_argument("FormExpr::previousSolution: steps_back must be >= 1");
    }
    return FormExpr(std::make_shared<PreviousSolutionRefNode>(steps_back));
}

FormExpr FormExpr::coordinate()
{
    return FormExpr(std::make_shared<CoordinateNode>());
}

FormExpr FormExpr::referenceCoordinate()
{
    return FormExpr(std::make_shared<ReferenceCoordinateNode>());
}

FormExpr FormExpr::time()
{
    return FormExpr(std::make_shared<TimeNode>());
}

FormExpr FormExpr::timeStep()
{
    return FormExpr(std::make_shared<TimeStepNode>());
}

FormExpr FormExpr::effectiveTimeStep()
{
    return FormExpr(std::make_shared<EffectiveTimeStepNode>());
}

FormExpr FormExpr::identity()
{
    return FormExpr(std::make_shared<IdentityNode>());
}

FormExpr FormExpr::identity(int dim)
{
    return FormExpr(std::make_shared<IdentityNode>(dim));
}

FormExpr FormExpr::jacobian()
{
    return FormExpr(std::make_shared<JacobianNode>());
}

FormExpr FormExpr::jacobianInverse()
{
    return FormExpr(std::make_shared<JacobianInverseNode>());
}

FormExpr FormExpr::jacobianDeterminant()
{
    return FormExpr(std::make_shared<JacobianDeterminantNode>());
}

FormExpr FormExpr::normal()
{
    return FormExpr(std::make_shared<NormalNode>());
}

FormExpr FormExpr::cellDiameter()
{
    return FormExpr(std::make_shared<CellDiameterNode>());
}

FormExpr FormExpr::cellVolume()
{
    return FormExpr(std::make_shared<CellVolumeNode>());
}

FormExpr FormExpr::facetArea()
{
    return FormExpr(std::make_shared<FacetAreaNode>());
}

FormExpr FormExpr::cellDomainId()
{
    return FormExpr(std::make_shared<CellDomainIdNode>());
}

FormExpr FormExpr::asVector(std::vector<FormExpr> components)
{
    if (components.empty()) return {};

    std::vector<std::shared_ptr<FormExprNode>> kids;
    kids.reserve(components.size());
    for (const auto& c : components) {
        if (!c.node_) {
            throw std::invalid_argument("FormExpr::asVector: component expression is invalid");
        }
        kids.push_back(c.node_);
    }
    return FormExpr(std::make_shared<AsVectorNode>(std::move(kids)));
}

FormExpr FormExpr::asTensor(std::vector<std::vector<FormExpr>> rows)
{
    if (rows.empty()) return {};

    const std::size_t cols = rows.front().size();
    if (cols == 0) {
        throw std::invalid_argument("FormExpr::asTensor: empty row");
    }
    for (const auto& r : rows) {
        if (r.size() != cols) {
            throw std::invalid_argument("FormExpr::asTensor: all rows must have the same number of columns");
        }
    }

    std::vector<std::shared_ptr<FormExprNode>> entries;
    entries.reserve(rows.size() * cols);
    for (const auto& r : rows) {
        for (const auto& e : r) {
            if (!e.node_) {
                throw std::invalid_argument("FormExpr::asTensor: entry expression is invalid");
            }
            entries.push_back(e.node_);
        }
    }

    return FormExpr(std::make_shared<AsTensorNode>(static_cast<int>(rows.size()),
                                                   static_cast<int>(cols),
                                                   std::move(entries)));
}

FormExpr FormExpr::constitutive(std::shared_ptr<const ConstitutiveModel> model, const FormExpr& input)
{
    if (!model) return {};
    if (!input.node_) return {};
    std::vector<std::shared_ptr<FormExprNode>> inputs;
    inputs.push_back(input.node_);
    return FormExpr(std::make_shared<ConstitutiveNode>(std::move(model), std::move(inputs)));
}

FormExpr FormExpr::constitutive(std::shared_ptr<const ConstitutiveModel> model, std::vector<FormExpr> inputs)
{
    if (!model) return {};
    if (inputs.empty()) return {};

    std::vector<std::shared_ptr<FormExprNode>> nodes;
    nodes.reserve(inputs.size());
    for (auto& e : inputs) {
        if (!e.node_) return {};
        nodes.push_back(std::move(e.node_));
    }
    return FormExpr(std::make_shared<ConstitutiveNode>(std::move(model), std::move(nodes)));
}

FormExpr FormExpr::constitutiveOutput(const FormExpr& call, std::size_t output_index)
{
    if (!call.node_) return {};
    return FormExpr(std::make_shared<ConstitutiveOutputNode>(call.node_, output_index));
}

FormExpr FormExpr::grad() const
{
    if (!node_) return {};
    return FormExpr(std::make_shared<GradientNode>(node_));
}

FormExpr FormExpr::div() const
{
    if (!node_) return {};
    return FormExpr(std::make_shared<DivergenceNode>(node_));
}

FormExpr FormExpr::curl() const
{
    if (!node_) return {};
    return FormExpr(std::make_shared<CurlNode>(node_));
}

FormExpr FormExpr::hessian() const
{
    if (!node_) return {};
    return FormExpr(std::make_shared<HessianNode>(node_));
}

FormExpr FormExpr::dt(int order) const
{
    if (!node_) return {};
    if (order <= 0) {
        throw std::invalid_argument("FormExpr::dt: order must be >= 1");
    }
    return FormExpr(std::make_shared<TimeDerivativeNode>(node_, order));
}

FormExpr FormExpr::minus() const
{
    if (!node_) return {};
    return FormExpr(std::make_shared<RestrictMinusNode>(node_));
}

FormExpr FormExpr::plus() const
{
    if (!node_) return {};
    return FormExpr(std::make_shared<RestrictPlusNode>(node_));
}

FormExpr FormExpr::jump() const
{
    if (!node_) return {};
    return FormExpr(std::make_shared<JumpNode>(node_));
}

FormExpr FormExpr::avg() const
{
    if (!node_) return {};
    return FormExpr(std::make_shared<AverageNode>(node_));
}

FormExpr FormExpr::operator-() const
{
    if (!node_) return {};
    return FormExpr(std::make_shared<NegateNode>(node_));
}

FormExpr FormExpr::operator+(const FormExpr& rhs) const
{
    if (!node_ || !rhs.node_) return {};
    return FormExpr(std::make_shared<AddNode>(node_, rhs.node_));
}

FormExpr FormExpr::operator-(const FormExpr& rhs) const
{
    if (!node_ || !rhs.node_) return {};
    return FormExpr(std::make_shared<SubtractNode>(node_, rhs.node_));
}

FormExpr FormExpr::operator*(const FormExpr& rhs) const
{
    if (!node_ || !rhs.node_) return {};
    return FormExpr(std::make_shared<MultiplyNode>(node_, rhs.node_));
}

FormExpr FormExpr::operator/(const FormExpr& rhs) const
{
    if (!node_ || !rhs.node_) return {};
    return FormExpr(std::make_shared<DivideNode>(node_, rhs.node_));
}

FormExpr FormExpr::operator*(Real scalar) const
{
    return FormExpr::constant(scalar) * (*this);
}

FormExpr FormExpr::operator/(Real scalar) const
{
    return (*this) / FormExpr::constant(scalar);
}

FormExpr FormExpr::inner(const FormExpr& rhs) const
{
    if (!node_ || !rhs.node_) return {};
    return FormExpr(std::make_shared<InnerProductNode>(node_, rhs.node_));
}

FormExpr FormExpr::doubleContraction(const FormExpr& rhs) const
{
    if (!node_ || !rhs.node_) return {};
    return FormExpr(std::make_shared<DoubleContractionNode>(node_, rhs.node_));
}

FormExpr FormExpr::outer(const FormExpr& rhs) const
{
    if (!node_ || !rhs.node_) return {};
    return FormExpr(std::make_shared<OuterProductNode>(node_, rhs.node_));
}

FormExpr FormExpr::cross(const FormExpr& rhs) const
{
    if (!node_ || !rhs.node_) return {};
    return FormExpr(std::make_shared<CrossProductNode>(node_, rhs.node_));
}

FormExpr FormExpr::pow(const FormExpr& exponent) const
{
    if (!node_ || !exponent.node_) return {};
    return FormExpr(std::make_shared<PowerNode>(node_, exponent.node_));
}

FormExpr FormExpr::min(const FormExpr& rhs) const
{
    if (!node_ || !rhs.node_) return {};
    return FormExpr(std::make_shared<MinimumNode>(node_, rhs.node_));
}

FormExpr FormExpr::max(const FormExpr& rhs) const
{
    if (!node_ || !rhs.node_) return {};
    return FormExpr(std::make_shared<MaximumNode>(node_, rhs.node_));
}

FormExpr FormExpr::lt(const FormExpr& rhs) const
{
    if (!node_ || !rhs.node_) return {};
    return FormExpr(std::make_shared<ComparisonNode>(FormExprType::Less, node_, rhs.node_));
}

FormExpr FormExpr::le(const FormExpr& rhs) const
{
    if (!node_ || !rhs.node_) return {};
    return FormExpr(std::make_shared<ComparisonNode>(FormExprType::LessEqual, node_, rhs.node_));
}

FormExpr FormExpr::gt(const FormExpr& rhs) const
{
    if (!node_ || !rhs.node_) return {};
    return FormExpr(std::make_shared<ComparisonNode>(FormExprType::Greater, node_, rhs.node_));
}

FormExpr FormExpr::ge(const FormExpr& rhs) const
{
    if (!node_ || !rhs.node_) return {};
    return FormExpr(std::make_shared<ComparisonNode>(FormExprType::GreaterEqual, node_, rhs.node_));
}

FormExpr FormExpr::eq(const FormExpr& rhs) const
{
    if (!node_ || !rhs.node_) return {};
    return FormExpr(std::make_shared<ComparisonNode>(FormExprType::Equal, node_, rhs.node_));
}

FormExpr FormExpr::ne(const FormExpr& rhs) const
{
    if (!node_ || !rhs.node_) return {};
    return FormExpr(std::make_shared<ComparisonNode>(FormExprType::NotEqual, node_, rhs.node_));
}

FormExpr FormExpr::conditional(const FormExpr& then_expr, const FormExpr& else_expr) const
{
    if (!node_ || !then_expr.node_ || !else_expr.node_) return {};
    return FormExpr(std::make_shared<ConditionalNode>(node_, then_expr.node_, else_expr.node_));
}

FormExpr FormExpr::component(int i, int j) const
{
    if (!node_) return {};
    return FormExpr(std::make_shared<ComponentNode>(node_, i, j));
}

FormExpr FormExpr::operator()(const Index& i) const
{
    if (!node_) return {};
    return FormExpr(std::make_shared<IndexedAccessNode>(node_, i));
}

FormExpr FormExpr::operator()(const Index& i, const Index& j) const
{
    if (!node_) return {};
    return FormExpr(std::make_shared<IndexedAccessNode>(node_, i, j));
}

FormExpr FormExpr::operator()(const Index& i, const Index& j, const Index& k) const
{
    if (!node_) return {};
    std::array<int, 4> ids{};
    std::array<int, 4> ext{};
    std::array<std::string, 4> names{};
    std::array<tensor::IndexVariance, 4> vars{};
    ids.fill(-1);
    ext.fill(-1);
    vars.fill(tensor::IndexVariance::None);
    ids[0] = i.id();
    ids[1] = j.id();
    ids[2] = k.id();
    ext[0] = i.extent();
    ext[1] = j.extent();
    ext[2] = k.extent();
    names[0] = i.name();
    names[1] = j.name();
    names[2] = k.name();
    return FormExpr(std::make_shared<IndexedAccessNode>(node_, 3, ids, ext, std::move(names), vars));
}

FormExpr FormExpr::operator()(const Index& i, const Index& j, const Index& k, const Index& l) const
{
    if (!node_) return {};
    std::array<int, 4> ids{};
    std::array<int, 4> ext{};
    std::array<std::string, 4> names{};
    std::array<tensor::IndexVariance, 4> vars{};
    ids.fill(-1);
    ext.fill(-1);
    vars.fill(tensor::IndexVariance::None);
    ids[0] = i.id();
    ids[1] = j.id();
    ids[2] = k.id();
    ids[3] = l.id();
    ext[0] = i.extent();
    ext[1] = j.extent();
    ext[2] = k.extent();
    ext[3] = l.extent();
    names[0] = i.name();
    names[1] = j.name();
    names[2] = k.name();
    names[3] = l.name();
    return FormExpr(std::make_shared<IndexedAccessNode>(node_, 4, ids, ext, std::move(names), vars));
}

FormExpr FormExpr::operator()(const tensor::TensorIndex& i) const
{
    if (!node_) return {};
    if (i.isFixed()) {
        throw std::invalid_argument("FormExpr::operator()(TensorIndex): fixed indices are not supported (use component())");
    }

    std::array<int, 4> ids{};
    std::array<int, 4> extents{};
    std::array<std::string, 4> names{};
    std::array<tensor::IndexVariance, 4> variances{};
    ids.fill(-1);
    extents.fill(-1);
    variances.fill(tensor::IndexVariance::None);

    ids[0] = i.id;
    extents[0] = i.dimension;
    names[0] = i.name;
    variances[0] = i.variance;
    return indexedAccessRawWithMetadata(FormExpr(node_), 1, ids, extents, variances, std::move(names));
}

FormExpr FormExpr::operator()(const tensor::TensorIndex& i, const tensor::TensorIndex& j) const
{
    if (!node_) return {};
    if (i.isFixed() || j.isFixed()) {
        throw std::invalid_argument("FormExpr::operator()(TensorIndex,TensorIndex): fixed indices are not supported");
    }

    std::array<int, 4> ids{};
    std::array<int, 4> extents{};
    std::array<std::string, 4> names{};
    std::array<tensor::IndexVariance, 4> variances{};
    ids.fill(-1);
    extents.fill(-1);
    variances.fill(tensor::IndexVariance::None);

    ids[0] = i.id;
    ids[1] = j.id;
    extents[0] = i.dimension;
    extents[1] = j.dimension;
    names[0] = i.name;
    names[1] = j.name;
    variances[0] = i.variance;
    variances[1] = j.variance;
    return indexedAccessRawWithMetadata(FormExpr(node_), 2, ids, extents, variances, std::move(names));
}

FormExpr FormExpr::operator()(const tensor::TensorIndex& i,
                              const tensor::TensorIndex& j,
                              const tensor::TensorIndex& k) const
{
    if (!node_) return {};
    if (i.isFixed() || j.isFixed() || k.isFixed()) {
        throw std::invalid_argument("FormExpr::operator()(TensorIndex,...): fixed indices are not supported");
    }

    std::array<int, 4> ids{};
    std::array<int, 4> extents{};
    std::array<std::string, 4> names{};
    std::array<tensor::IndexVariance, 4> variances{};
    ids.fill(-1);
    extents.fill(-1);
    variances.fill(tensor::IndexVariance::None);

    ids[0] = i.id;
    ids[1] = j.id;
    ids[2] = k.id;
    extents[0] = i.dimension;
    extents[1] = j.dimension;
    extents[2] = k.dimension;
    names[0] = i.name;
    names[1] = j.name;
    names[2] = k.name;
    variances[0] = i.variance;
    variances[1] = j.variance;
    variances[2] = k.variance;
    return indexedAccessRawWithMetadata(FormExpr(node_), 3, ids, extents, variances, std::move(names));
}

FormExpr FormExpr::operator()(const tensor::TensorIndex& i,
                              const tensor::TensorIndex& j,
                              const tensor::TensorIndex& k,
                              const tensor::TensorIndex& l) const
{
    if (!node_) return {};
    if (i.isFixed() || j.isFixed() || k.isFixed() || l.isFixed()) {
        throw std::invalid_argument("FormExpr::operator()(TensorIndex,...): fixed indices are not supported");
    }

    std::array<int, 4> ids{};
    std::array<int, 4> extents{};
    std::array<std::string, 4> names{};
    std::array<tensor::IndexVariance, 4> variances{};
    ids.fill(-1);
    extents.fill(-1);
    variances.fill(tensor::IndexVariance::None);

    ids[0] = i.id;
    ids[1] = j.id;
    ids[2] = k.id;
    ids[3] = l.id;
    extents[0] = i.dimension;
    extents[1] = j.dimension;
    extents[2] = k.dimension;
    extents[3] = l.dimension;
    names[0] = i.name;
    names[1] = j.name;
    names[2] = k.name;
    names[3] = l.name;
    variances[0] = i.variance;
    variances[1] = j.variance;
    variances[2] = k.variance;
    variances[3] = l.variance;
    return indexedAccessRawWithMetadata(FormExpr(node_), 4, ids, extents, variances, std::move(names));
}

FormExpr FormExpr::indexedAccessRaw(FormExpr base,
                                   int rank,
                                   std::array<int, 4> ids,
                                   std::array<int, 4> extents)
{
    if (!base.node_) return {};
    if (rank < 1 || rank > 4) {
        throw std::invalid_argument("FormExpr::indexedAccessRaw: rank must be 1..4");
    }
    std::array<std::string, 4> names{};
    for (std::size_t k = 0; k < names.size(); ++k) {
        const int id = ids[k];
        names[k] = (id >= 0) ? ("i" + std::to_string(id)) : std::string{};
    }
    std::array<tensor::IndexVariance, 4> vars{};
    vars.fill(tensor::IndexVariance::None);
    return indexedAccessRawWithMetadata(std::move(base), rank, std::move(ids), std::move(extents), std::move(vars), std::move(names));
}

FormExpr FormExpr::indexedAccessRawWithMetadata(FormExpr base,
                                               int rank,
                                               std::array<int, 4> ids,
                                               std::array<int, 4> extents,
                                               std::array<tensor::IndexVariance, 4> variances,
                                               std::array<std::string, 4> names)
{
    if (!base.node_) return {};
    if (rank < 1 || rank > 4) {
        throw std::invalid_argument("FormExpr::indexedAccessRawWithMetadata: rank must be 1..4");
    }
    for (int k = 0; k < rank; ++k) {
        const auto idx = static_cast<std::size_t>(k);
        if (ids[idx] < 0 || extents[idx] <= 0) {
            throw std::invalid_argument("FormExpr::indexedAccessRawWithMetadata: invalid index id/extent");
        }
        if (names[idx].empty()) {
            names[idx] = "i" + std::to_string(ids[idx]);
        }
    }
    return FormExpr(std::make_shared<IndexedAccessNode>(base.node_, rank, std::move(ids), std::move(extents), std::move(names), std::move(variances)));
}

FormExpr FormExpr::transpose() const
{
    if (!node_) return {};
    return FormExpr(std::make_shared<TransposeNode>(node_));
}

FormExpr FormExpr::trace() const
{
    if (!node_) return {};
    return FormExpr(std::make_shared<TraceNode>(node_));
}

FormExpr FormExpr::det() const
{
    if (!node_) return {};
    return FormExpr(std::make_shared<DeterminantNode>(node_));
}

FormExpr FormExpr::inv() const
{
    if (!node_) return {};
    return FormExpr(std::make_shared<InverseNode>(node_));
}

FormExpr FormExpr::cofactor() const
{
    if (!node_) return {};
    return FormExpr(std::make_shared<CofactorNode>(node_));
}

FormExpr FormExpr::dev() const
{
    if (!node_) return {};
    return FormExpr(std::make_shared<DeviatorNode>(node_));
}

FormExpr FormExpr::sym() const
{
    if (!node_) return {};
    return FormExpr(std::make_shared<SymmetricPartNode>(node_));
}

FormExpr FormExpr::skew() const
{
    if (!node_) return {};
    return FormExpr(std::make_shared<SkewPartNode>(node_));
}

FormExpr FormExpr::norm() const
{
    if (!node_) return {};
    return FormExpr(std::make_shared<NormNode>(node_));
}

FormExpr FormExpr::normalize() const
{
    if (!node_) return {};
    return FormExpr(std::make_shared<NormalizeNode>(node_));
}

FormExpr FormExpr::abs() const
{
    if (!node_) return {};
    return FormExpr(std::make_shared<AbsoluteValueNode>(node_));
}

FormExpr FormExpr::sign() const
{
    if (!node_) return {};
    return FormExpr(std::make_shared<SignNode>(node_));
}

FormExpr FormExpr::sqrt() const
{
    if (!node_) return {};
    return FormExpr(std::make_shared<SqrtNode>(node_));
}

FormExpr FormExpr::exp() const
{
    if (!node_) return {};
    return FormExpr(std::make_shared<ExpNode>(node_));
}

FormExpr FormExpr::log() const
{
    if (!node_) return {};
    return FormExpr(std::make_shared<LogNode>(node_));
}

FormExpr FormExpr::symmetricEigenvalue(int which) const
{
    if (!node_) return {};
    return FormExpr(std::make_shared<SymmetricEigenvalueNode>(node_, which));
}

FormExpr FormExpr::symmetricEigenvalueDirectionalDerivative(const FormExpr& A, const FormExpr& dA, int which)
{
    if (!A.node_ || !dA.node_) return {};
    return FormExpr(std::make_shared<SymmetricEigenvalueDirectionalDerivativeNode>(A.node_, dA.node_, which));
}

FormExpr FormExpr::symmetricEigenvalueDirectionalDerivativeWrtA(const FormExpr& A,
                                                                const FormExpr& B,
                                                                const FormExpr& dA,
                                                                int which)
{
    if (!A.node_ || !B.node_ || !dA.node_) return {};
    return FormExpr(std::make_shared<SymmetricEigenvalueDirectionalDerivativeWrtANode>(A.node_, B.node_, dA.node_, which));
}

FormExpr FormExpr::dx() const
{
    if (!node_) return {};
    return FormExpr(std::make_shared<CellIntegralNode>(node_));
}

FormExpr FormExpr::ds(int boundary_marker) const
{
    if (!node_) return {};
    return FormExpr(std::make_shared<BoundaryIntegralNode>(node_, boundary_marker));
}

FormExpr FormExpr::dS() const
{
    if (!node_) return {};
    return FormExpr(std::make_shared<InteriorFaceIntegralNode>(node_));
}

FormExpr FormExpr::dI(int interface_marker) const
{
    if (!node_) return {};
    return FormExpr(std::make_shared<InterfaceIntegralNode>(node_, interface_marker));
}

namespace {

std::shared_ptr<FormExprNode> transformNodeShared(
    const std::shared_ptr<FormExprNode>& node,
    const FormExpr::NodeTransform& transform)
{
    if (!node) {
        return {};
    }

    if (transform) {
        auto replacement = transform(*node);
        if (replacement.has_value()) {
            if (!replacement->isValid()) {
                throw std::invalid_argument("FormExpr::transformNodes: transform returned an invalid FormExpr");
            }
            return replacement->nodeShared();
        }
    }

    const auto kids = node->childrenShared();
    if (kids.empty()) {
        return node;
    }

    std::vector<std::shared_ptr<FormExprNode>> new_kids;
    new_kids.reserve(kids.size());
    bool changed = false;
    for (const auto& k : kids) {
        auto new_k = transformNodeShared(k, transform);
        if (new_k != k) {
            changed = true;
        }
        new_kids.push_back(std::move(new_k));
    }
    if (!changed) {
        return node;
    }

    switch (node->type()) {
        // Unary operators
        case FormExprType::Negate: return std::make_shared<NegateNode>(new_kids[0]);
        case FormExprType::Gradient: return std::make_shared<GradientNode>(new_kids[0]);
        case FormExprType::Divergence: return std::make_shared<DivergenceNode>(new_kids[0]);
        case FormExprType::Curl: return std::make_shared<CurlNode>(new_kids[0]);
        case FormExprType::Hessian: return std::make_shared<HessianNode>(new_kids[0]);
        case FormExprType::TimeDerivative: {
            const int order = node->timeDerivativeOrder().value_or(1);
            return std::make_shared<TimeDerivativeNode>(new_kids[0], order);
        }
        case FormExprType::BoundaryFunctionalSymbol: {
            const int marker = node->boundaryMarker().value_or(-1);
            const auto name = node->symbolName();
            if (marker < 0 || !name) {
                throw std::logic_error("FormExpr::transformNodes: BoundaryFunctionalSymbol missing metadata");
            }
            return std::make_shared<BoundaryFunctionalSymbolNode>(new_kids[0], marker, std::string(*name));
        }
        case FormExprType::RestrictMinus: return std::make_shared<RestrictMinusNode>(new_kids[0]);
        case FormExprType::RestrictPlus: return std::make_shared<RestrictPlusNode>(new_kids[0]);
        case FormExprType::Jump: return std::make_shared<JumpNode>(new_kids[0]);
        case FormExprType::Average: return std::make_shared<AverageNode>(new_kids[0]);

        case FormExprType::Component: {
            const int i = node->componentIndex0().value_or(0);
            const int j = node->componentIndex1().value_or(-1);
            return std::make_shared<ComponentNode>(new_kids[0], i, j);
        }
	        case FormExprType::IndexedAccess: {
	            const int rank = node->indexRank().value_or(0);
	            const auto ids_opt = node->indexIds();
	            const auto ext_opt = node->indexExtents();
	            if (!ids_opt || !ext_opt) {
	                throw std::logic_error("FormExpr::transformNodes: IndexedAccess missing index metadata");
	            }
	            std::array<std::string, 4> names{};
	            if (const auto names_opt = node->indexNames()) {
	                for (std::size_t k = 0; k < names.size(); ++k) {
	                    names[k] = std::string((*names_opt)[k]);
	                }
	            } else {
	                for (std::size_t k = 0; k < names.size(); ++k) {
	                    const int id = (*ids_opt)[k];
	                    names[k] = (id >= 0) ? ("i" + std::to_string(id)) : std::string{};
	                }
	            }
	            std::array<tensor::IndexVariance, 4> vars{};
	            vars.fill(tensor::IndexVariance::None);
	            if (const auto vars_opt = node->indexVariances()) {
	                vars = *vars_opt;
	            }
	            return std::make_shared<IndexedAccessNode>(new_kids[0], rank, *ids_opt, *ext_opt, std::move(names), vars);
	        }

        case FormExprType::Transpose: return std::make_shared<TransposeNode>(new_kids[0]);
        case FormExprType::Trace: return std::make_shared<TraceNode>(new_kids[0]);
        case FormExprType::Determinant: return std::make_shared<DeterminantNode>(new_kids[0]);
        case FormExprType::Inverse: return std::make_shared<InverseNode>(new_kids[0]);
        case FormExprType::Cofactor: return std::make_shared<CofactorNode>(new_kids[0]);
        case FormExprType::Deviator: return std::make_shared<DeviatorNode>(new_kids[0]);
        case FormExprType::SymmetricPart: return std::make_shared<SymmetricPartNode>(new_kids[0]);
        case FormExprType::SkewPart: return std::make_shared<SkewPartNode>(new_kids[0]);
        case FormExprType::Norm: return std::make_shared<NormNode>(new_kids[0]);
        case FormExprType::Normalize: return std::make_shared<NormalizeNode>(new_kids[0]);
        case FormExprType::AbsoluteValue: return std::make_shared<AbsoluteValueNode>(new_kids[0]);
        case FormExprType::Sign: return std::make_shared<SignNode>(new_kids[0]);
        case FormExprType::Sqrt: return std::make_shared<SqrtNode>(new_kids[0]);
        case FormExprType::Exp: return std::make_shared<ExpNode>(new_kids[0]);
        case FormExprType::Log: return std::make_shared<LogNode>(new_kids[0]);

        case FormExprType::SymmetricEigenvalue: {
            const auto which = node->eigenIndex();
            if (!which.has_value()) {
                throw std::logic_error("FormExpr::transformNodes: SymmetricEigenvalue missing eigen index");
            }
            return std::make_shared<SymmetricEigenvalueNode>(new_kids[0], *which);
        }
        case FormExprType::SymmetricEigenvalueDirectionalDerivative: {
            const auto which = node->eigenIndex();
            if (!which.has_value()) {
                throw std::logic_error("FormExpr::transformNodes: SymmetricEigenvalueDirectionalDerivative missing eigen index");
            }
            return std::make_shared<SymmetricEigenvalueDirectionalDerivativeNode>(new_kids[0], new_kids[1], *which);
        }
        case FormExprType::SymmetricEigenvalueDirectionalDerivativeWrtA: {
            const auto which = node->eigenIndex();
            if (!which.has_value()) {
                throw std::logic_error("FormExpr::transformNodes: SymmetricEigenvalueDirectionalDerivativeWrtA missing eigen index");
            }
            return std::make_shared<SymmetricEigenvalueDirectionalDerivativeWrtANode>(new_kids[0], new_kids[1], new_kids[2], *which);
        }

        case FormExprType::CellIntegral: return std::make_shared<CellIntegralNode>(new_kids[0]);
        case FormExprType::BoundaryIntegral: {
            const int marker = node->boundaryMarker().value_or(-1);
            return std::make_shared<BoundaryIntegralNode>(new_kids[0], marker);
        }
        case FormExprType::InteriorFaceIntegral: return std::make_shared<InteriorFaceIntegralNode>(new_kids[0]);
        case FormExprType::InterfaceIntegral: {
            const int marker = node->interfaceMarker().value_or(-1);
            return std::make_shared<InterfaceIntegralNode>(new_kids[0], marker);
        }

        case FormExprType::Constitutive: {
            auto model = node->constitutiveModelShared();
            if (!model) {
                throw std::logic_error("FormExpr::transformNodes: Constitutive node missing model");
            }
            return std::make_shared<ConstitutiveNode>(std::move(model), std::move(new_kids));
        }
        case FormExprType::ConstitutiveOutput: {
            const auto idx = node->constitutiveOutputIndex();
            if (!idx || *idx < 0) {
                throw std::logic_error("FormExpr::transformNodes: ConstitutiveOutput missing output index");
            }
            return std::make_shared<ConstitutiveOutputNode>(new_kids[0], static_cast<std::size_t>(*idx));
        }

        // Binary operators
        case FormExprType::Add: return std::make_shared<AddNode>(new_kids[0], new_kids[1]);
        case FormExprType::Subtract: return std::make_shared<SubtractNode>(new_kids[0], new_kids[1]);
        case FormExprType::Multiply: return std::make_shared<MultiplyNode>(new_kids[0], new_kids[1]);
        case FormExprType::Divide: return std::make_shared<DivideNode>(new_kids[0], new_kids[1]);
        case FormExprType::InnerProduct: return std::make_shared<InnerProductNode>(new_kids[0], new_kids[1]);
        case FormExprType::DoubleContraction: return std::make_shared<DoubleContractionNode>(new_kids[0], new_kids[1]);
        case FormExprType::OuterProduct: return std::make_shared<OuterProductNode>(new_kids[0], new_kids[1]);
        case FormExprType::CrossProduct: return std::make_shared<CrossProductNode>(new_kids[0], new_kids[1]);
        case FormExprType::Power: return std::make_shared<PowerNode>(new_kids[0], new_kids[1]);
        case FormExprType::Minimum: return std::make_shared<MinimumNode>(new_kids[0], new_kids[1]);
        case FormExprType::Maximum: return std::make_shared<MaximumNode>(new_kids[0], new_kids[1]);

        case FormExprType::Less:
        case FormExprType::LessEqual:
        case FormExprType::Greater:
        case FormExprType::GreaterEqual:
        case FormExprType::Equal:
        case FormExprType::NotEqual:
            return std::make_shared<ComparisonNode>(node->type(), new_kids[0], new_kids[1]);

        // Ternary
        case FormExprType::Conditional:
            return std::make_shared<ConditionalNode>(new_kids[0], new_kids[1], new_kids[2]);

        // Packing
        case FormExprType::AsVector:
            return std::make_shared<AsVectorNode>(std::move(new_kids));
        case FormExprType::AsTensor: {
            const int rows = node->tensorRows().value_or(0);
            const int cols = node->tensorCols().value_or(0);
            return std::make_shared<AsTensorNode>(rows, cols, std::move(new_kids));
        }

        default:
            // Terminals and other leaf nodes should have no children.
            throw std::logic_error("FormExpr::transformNodes: cannot rebuild node type");
    }
}

} // namespace

FormExpr FormExpr::transformNodes(const NodeTransform& transform) const
{
    if (!node_) {
        return {};
    }
    if (!transform) {
        return *this;
    }
    return FormExpr(transformNodeShared(node_, transform));
}

std::string FormExpr::toString() const
{
    return node_ ? node_->toString() : "<empty>";
}

bool FormExpr::hasTest() const noexcept
{
    return node_ && node_->hasTest();
}

bool FormExpr::hasTrial() const noexcept
{
    return node_ && node_->hasTrial();
}

} // namespace forms
} // namespace FE
} // namespace svmp
