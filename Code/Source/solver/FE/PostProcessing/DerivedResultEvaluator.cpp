#include "PostProcessing/DerivedResultEvaluator.h"

#include "Assembly/Assembler.h"
#include "Assembly/GlobalSystemView.h"
#include "Backends/Interfaces/GenericVector.h"
#include "Basis/BasisFunction.h"
#include "Core/FEException.h"
#include "Dofs/DofHandler.h"
#include "Dofs/DofMap.h"
#include "Dofs/EntityDofMap.h"
#include "Elements/Element.h"
#include "Geometry/GeometryMapping.h"
#include "Geometry/MappingFactory.h"
#include "PostProcessing/DerivedResultOutput.h"
#include "Quadrature/QuadratureFactory.h"
#include "Quadrature/QuadratureRule.h"
#include "Spaces/FunctionSpace.h"
#include "Systems/FESystem.h"

#if defined(SVMP_FE_WITH_MESH) && SVMP_FE_WITH_MESH
#include "Mesh/Core/MeshBase.h"
#include "Mesh/Core/MeshTypes.h"
#endif

#include <algorithm>
#include <array>
#include <cmath>
#include <memory>
#include <span>
#include <stdexcept>
#include <unordered_map>

namespace svmp {
namespace FE {
namespace post {

namespace {

using Matrix3 = std::array<std::array<Real, 3>, 3>;
using Vector3 = std::array<Real, 3>;

int defaultGeometryOrder(ElementType element_type) noexcept
{
    switch (element_type) {
    case ElementType::Line3:
    case ElementType::Triangle6:
    case ElementType::Quad8:
    case ElementType::Quad9:
    case ElementType::Tetra10:
    case ElementType::Hex20:
    case ElementType::Hex27:
    case ElementType::Wedge15:
    case ElementType::Wedge18:
    case ElementType::Pyramid13:
    case ElementType::Pyramid14:
        return 2;
    default:
        return 1;
    }
}

[[nodiscard]] Vector3 referenceCentroid(ElementType element_type) noexcept
{
    switch (element_type) {
    case ElementType::Line2:
    case ElementType::Line3:
    case ElementType::Quad4:
    case ElementType::Quad8:
    case ElementType::Quad9:
    case ElementType::Hex8:
    case ElementType::Hex20:
    case ElementType::Hex27:
    case ElementType::Pyramid5:
    case ElementType::Pyramid13:
    case ElementType::Pyramid14:
        return {0.0, 0.0, 0.0};
    case ElementType::Triangle3:
    case ElementType::Triangle6:
        return {Real(1) / Real(3), Real(1) / Real(3), 0.0};
    case ElementType::Tetra4:
    case ElementType::Tetra10:
        return {Real(1) / Real(4), Real(1) / Real(4), Real(1) / Real(4)};
    case ElementType::Wedge6:
    case ElementType::Wedge15:
    case ElementType::Wedge18:
        return {Real(1) / Real(3), Real(1) / Real(3), 0.0};
    default:
        return {0.0, 0.0, 0.0};
    }
}

[[nodiscard]] Matrix3 zeroMatrix() noexcept
{
    return Matrix3{{{0.0, 0.0, 0.0},
                    {0.0, 0.0, 0.0},
                    {0.0, 0.0, 0.0}}};
}

[[nodiscard]] Vector3 zeroVector() noexcept
{
    return {0.0, 0.0, 0.0};
}

[[nodiscard]] Real vectorDot(const Vector3& a, const Vector3& b) noexcept
{
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

[[nodiscard]] Vector3 vectorCross(const Vector3& a, const Vector3& b) noexcept
{
    return {
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    };
}

[[nodiscard]] Real vectorNorm(const Vector3& v) noexcept
{
    return std::sqrt(vectorDot(v, v));
}

struct EvalValue {
    enum class Kind {
        Scalar,
        Vector,
        Tensor
    };

    Kind kind{Kind::Scalar};
    Real s{0.0};
    Vector3 v{zeroVector()};
    Matrix3 m{zeroMatrix()};
    int rows{1};
    int cols{1};

    [[nodiscard]] int components() const noexcept
    {
        switch (kind) {
        case Kind::Scalar: return 1;
        case Kind::Vector: return rows;
        case Kind::Tensor: return rows * cols;
        }
        return 1;
    }
};

[[nodiscard]] EvalValue scalar(Real value) noexcept
{
    EvalValue out;
    out.kind = EvalValue::Kind::Scalar;
    out.s = value;
    out.rows = 1;
    out.cols = 1;
    return out;
}

[[nodiscard]] EvalValue vector(Vector3 value, int dim = 3) noexcept
{
    EvalValue out;
    out.kind = EvalValue::Kind::Vector;
    out.v = value;
    out.rows = std::max(1, std::min(dim, 3));
    out.cols = 1;
    return out;
}

[[nodiscard]] EvalValue tensor(Matrix3 value, int rows = 3, int cols = 3) noexcept
{
    EvalValue out;
    out.kind = EvalValue::Kind::Tensor;
    out.m = value;
    out.rows = std::max(1, std::min(rows, 3));
    out.cols = std::max(1, std::min(cols, 3));
    return out;
}

[[nodiscard]] EvalValue typedZeroLike(const EvalValue& a) noexcept
{
    if (a.kind == EvalValue::Kind::Vector) {
        return vector(zeroVector(), a.rows);
    }
    if (a.kind == EvalValue::Kind::Tensor) {
        return tensor(zeroMatrix(), a.rows, a.cols);
    }
    return scalar(0.0);
}

[[nodiscard]] EvalValue addValues(const EvalValue& a, const EvalValue& b)
{
    FE_THROW_IF(a.kind != b.kind, InvalidArgumentException,
                "DerivedResultEvaluator: cannot add values with different shapes");
    if (a.kind == EvalValue::Kind::Scalar) {
        return scalar(a.s + b.s);
    }
    if (a.kind == EvalValue::Kind::Vector) {
        Vector3 out{};
        for (int i = 0; i < a.rows; ++i) {
            out[static_cast<std::size_t>(i)] =
                a.v[static_cast<std::size_t>(i)] + b.v[static_cast<std::size_t>(i)];
        }
        return vector(out, a.rows);
    }
    Matrix3 out = zeroMatrix();
    for (int r = 0; r < a.rows; ++r) {
        for (int c = 0; c < a.cols; ++c) {
            out[static_cast<std::size_t>(r)][static_cast<std::size_t>(c)] =
                a.m[static_cast<std::size_t>(r)][static_cast<std::size_t>(c)] +
                b.m[static_cast<std::size_t>(r)][static_cast<std::size_t>(c)];
        }
    }
    return tensor(out, a.rows, a.cols);
}

[[nodiscard]] EvalValue negateValue(const EvalValue& a)
{
    if (a.kind == EvalValue::Kind::Scalar) {
        return scalar(-a.s);
    }
    if (a.kind == EvalValue::Kind::Vector) {
        Vector3 out{};
        for (int i = 0; i < a.rows; ++i) {
            out[static_cast<std::size_t>(i)] = -a.v[static_cast<std::size_t>(i)];
        }
        return vector(out, a.rows);
    }
    Matrix3 out = zeroMatrix();
    for (int r = 0; r < a.rows; ++r) {
        for (int c = 0; c < a.cols; ++c) {
            out[static_cast<std::size_t>(r)][static_cast<std::size_t>(c)] =
                -a.m[static_cast<std::size_t>(r)][static_cast<std::size_t>(c)];
        }
    }
    return tensor(out, a.rows, a.cols);
}

[[nodiscard]] EvalValue scaleValue(const EvalValue& a, Real s)
{
    if (a.kind == EvalValue::Kind::Scalar) {
        return scalar(a.s * s);
    }
    if (a.kind == EvalValue::Kind::Vector) {
        Vector3 out{};
        for (int i = 0; i < a.rows; ++i) {
            out[static_cast<std::size_t>(i)] = a.v[static_cast<std::size_t>(i)] * s;
        }
        return vector(out, a.rows);
    }
    Matrix3 out = zeroMatrix();
    for (int r = 0; r < a.rows; ++r) {
        for (int c = 0; c < a.cols; ++c) {
            out[static_cast<std::size_t>(r)][static_cast<std::size_t>(c)] =
                a.m[static_cast<std::size_t>(r)][static_cast<std::size_t>(c)] * s;
        }
    }
    return tensor(out, a.rows, a.cols);
}

[[nodiscard]] EvalValue multiplyValues(const EvalValue& a, const EvalValue& b)
{
    if (a.kind == EvalValue::Kind::Scalar) {
        return scaleValue(b, a.s);
    }
    if (b.kind == EvalValue::Kind::Scalar) {
        return scaleValue(a, b.s);
    }
    if (a.kind == EvalValue::Kind::Tensor && b.kind == EvalValue::Kind::Vector) {
        Vector3 out{};
        for (int r = 0; r < a.rows; ++r) {
            Real sum = 0.0;
            for (int c = 0; c < a.cols; ++c) {
                sum += a.m[static_cast<std::size_t>(r)][static_cast<std::size_t>(c)] *
                       b.v[static_cast<std::size_t>(c)];
            }
            out[static_cast<std::size_t>(r)] = sum;
        }
        return vector(out, a.rows);
    }
    if (a.kind == EvalValue::Kind::Tensor && b.kind == EvalValue::Kind::Tensor) {
        FE_THROW_IF(a.cols != b.rows, InvalidArgumentException,
                    "DerivedResultEvaluator: tensor multiply dimension mismatch");
        Matrix3 out = zeroMatrix();
        for (int r = 0; r < a.rows; ++r) {
            for (int c = 0; c < b.cols; ++c) {
                Real sum = 0.0;
                for (int k = 0; k < a.cols; ++k) {
                    sum += a.m[static_cast<std::size_t>(r)][static_cast<std::size_t>(k)] *
                           b.m[static_cast<std::size_t>(k)][static_cast<std::size_t>(c)];
                }
                out[static_cast<std::size_t>(r)][static_cast<std::size_t>(c)] = sum;
            }
        }
        return tensor(out, a.rows, b.cols);
    }
    FE_THROW(InvalidArgumentException,
             "DerivedResultEvaluator: unsupported multiply operand shapes");
}

[[nodiscard]] EvalValue divideValues(const EvalValue& a, const EvalValue& b)
{
    FE_THROW_IF(b.kind != EvalValue::Kind::Scalar, InvalidArgumentException,
                "DerivedResultEvaluator: division by non-scalar is not supported");
    FE_THROW_IF(std::abs(b.s) == Real(0), InvalidArgumentException,
                "DerivedResultEvaluator: division by zero");
    return scaleValue(a, Real(1) / b.s);
}

[[nodiscard]] EvalValue innerValues(const EvalValue& a, const EvalValue& b)
{
    if (a.kind == EvalValue::Kind::Vector && b.kind == EvalValue::Kind::Vector) {
        return scalar(vectorDot(a.v, b.v));
    }
    if (a.kind == EvalValue::Kind::Tensor && b.kind == EvalValue::Kind::Tensor) {
        Real sum = 0.0;
        for (int r = 0; r < a.rows; ++r) {
            for (int c = 0; c < a.cols; ++c) {
                sum += a.m[static_cast<std::size_t>(r)][static_cast<std::size_t>(c)] *
                       b.m[static_cast<std::size_t>(r)][static_cast<std::size_t>(c)];
            }
        }
        return scalar(sum);
    }
    FE_THROW(InvalidArgumentException,
             "DerivedResultEvaluator: inner product expects vector/vector or tensor/tensor");
}

struct CellSamples {
    std::vector<Vector3> xi;
    std::vector<Real> quadrature_weights;
    std::vector<Vector3> x;
    std::vector<Matrix3> inv_jacobians;
    std::vector<Real> integration_weights;
};

struct FieldSampleData {
    FieldId field{INVALID_FIELD_ID};
    FieldType field_type{FieldType::Scalar};
    int components{1};
    int value_dimension{1};
    std::vector<EvalValue> values;
    std::vector<Vector3> scalar_gradients;
    std::vector<Matrix3> vector_jacobians;
};

[[nodiscard]] Matrix3 toArrayMatrix(const math::Matrix<Real, 3, 3>& m) noexcept
{
    Matrix3 out = zeroMatrix();
    for (int r = 0; r < 3; ++r) {
        for (int c = 0; c < 3; ++c) {
            out[static_cast<std::size_t>(r)][static_cast<std::size_t>(c)] =
                m(static_cast<std::size_t>(r), static_cast<std::size_t>(c));
        }
    }
    return out;
}

[[nodiscard]] CellSamples prepareCellSamples(const assembly::IMeshAccess& mesh,
                                             GlobalIndex cell_id,
                                             const spaces::FunctionSpace& space,
                                             DerivedResultPolicy policy)
{
    const auto cell_type = mesh.getCellType(cell_id);
    const auto& element = space.getElement(cell_type, cell_id);

    CellSamples samples;
    if (policy == DerivedResultPolicy::CellCentroid) {
        samples.xi.push_back(referenceCentroid(cell_type));
        samples.quadrature_weights.push_back(1.0);
    } else {
        auto quad = element.quadrature();
        if (!quad) {
            const int order = quadrature::QuadratureFactory::recommended_order(
                std::max(1, element.polynomial_order()), false);
            quad = quadrature::QuadratureFactory::create(cell_type, order);
        }
        const auto& qpts = quad->points();
        const auto& qwts = quad->weights();
        samples.xi.reserve(qpts.size());
        samples.quadrature_weights.reserve(qwts.size());
        for (std::size_t q = 0; q < qpts.size(); ++q) {
            samples.xi.push_back({qpts[q][0], qpts[q][1], qpts[q][2]});
            samples.quadrature_weights.push_back(qwts[q]);
        }
    }

    std::vector<std::array<Real, 3>> raw_coords;
    mesh.getCellCoordinates(cell_id, raw_coords);
    std::vector<math::Vector<Real, 3>> node_coords(raw_coords.size());
    for (std::size_t i = 0; i < raw_coords.size(); ++i) {
        node_coords[i] = {raw_coords[i][0], raw_coords[i][1], raw_coords[i][2]};
    }

    geometry::MappingRequest map_request;
    map_request.element_type = cell_type;
    map_request.geometry_order = defaultGeometryOrder(cell_type);
    map_request.use_affine = (map_request.geometry_order <= 1);
    auto mapping = geometry::MappingFactory::create(map_request, node_coords);

    samples.x.resize(samples.xi.size());
    samples.inv_jacobians.resize(samples.xi.size());
    samples.integration_weights.resize(samples.xi.size());
    for (std::size_t q = 0; q < samples.xi.size(); ++q) {
        const math::Vector<Real, 3> xi{samples.xi[q][0], samples.xi[q][1], samples.xi[q][2]};
        const auto xp = mapping->map_to_physical(xi);
        const auto invJ = mapping->jacobian_inverse(xi);
        const Real detJ = mapping->jacobian_determinant(xi);
        samples.x[q] = {xp[0], xp[1], xp[2]};
        samples.inv_jacobians[q] = toArrayMatrix(invJ);
        samples.integration_weights[q] = samples.quadrature_weights[q] * std::abs(detJ);
    }

    return samples;
}

[[nodiscard]] std::unique_ptr<assembly::GlobalSystemView>
makeSolutionView(const systems::SystemStateView& state)
{
    if (state.u_vector == nullptr) {
        return nullptr;
    }
    auto* vec = const_cast<backends::GenericVector*>(state.u_vector);
    return vec->createAssemblyView();
}

Real readSolutionEntry(GlobalIndex dof,
                       const systems::SystemStateView& state,
                       const assembly::GlobalSystemView* view)
{
    FE_THROW_IF(dof < 0, InvalidArgumentException,
                "DerivedResultEvaluator: negative DOF index");
    if (view != nullptr) {
        return view->getVectorEntry(dof);
    }
    const auto idx = static_cast<std::size_t>(dof);
    FE_THROW_IF(idx >= state.u.size(), InvalidArgumentException,
                "DerivedResultEvaluator: state vector is smaller than required DOF index");
    return state.u[idx];
}

std::vector<Real> gatherFieldCoefficients(const systems::FESystem& system,
                                          FieldId field,
                                          GlobalIndex cell_id,
                                          const systems::SystemStateView& state,
                                          const assembly::GlobalSystemView* view)
{
    const auto& dof_map = system.fieldDofHandler(field).getDofMap();
    const auto dofs = dof_map.getCellDofs(cell_id);
    const auto offset = system.fieldDofOffset(field);
    std::vector<Real> coeffs(dofs.size());
    for (std::size_t i = 0; i < dofs.size(); ++i) {
        coeffs[i] = readSolutionEntry(dofs[i] + offset, state, view);
    }
    return coeffs;
}

[[nodiscard]] Vector3 transformGradient(const Vector3& grad_ref,
                                        const Matrix3& invJ,
                                        int dim) noexcept
{
    Vector3 grad_phys{0.0, 0.0, 0.0};
    for (int d1 = 0; d1 < dim; ++d1) {
        for (int d2 = 0; d2 < dim; ++d2) {
            grad_phys[static_cast<std::size_t>(d1)] +=
                invJ[static_cast<std::size_t>(d2)][static_cast<std::size_t>(d1)] *
                grad_ref[static_cast<std::size_t>(d2)];
        }
    }
    return grad_phys;
}

FieldSampleData sampleFieldOnCell(const systems::FESystem& system,
                                  FieldId field,
                                  GlobalIndex cell_id,
                                  const CellSamples& samples,
                                  const systems::SystemStateView& state,
                                  const assembly::GlobalSystemView* view)
{
    const auto& rec = system.fieldRecord(field);
    FE_CHECK_NOT_NULL(rec.space.get(), "DerivedResultEvaluator: field space");

    const auto& mesh = system.meshAccess();
    const int dim = mesh.dimension();
    const int n_comp = std::max(1, rec.components);
    const auto cell_type = mesh.getCellType(cell_id);
    const auto& field_element = rec.space->getElement(cell_type, cell_id);
    const auto& field_basis = field_element.basis();
    const auto coeffs = gatherFieldCoefficients(system, field, cell_id, state, view);

    FieldSampleData out;
    out.field = field;
    out.field_type = rec.space->field_type();
    out.components = n_comp;
    out.value_dimension = rec.space->value_dimension();
    out.values.resize(samples.xi.size());
    out.scalar_gradients.resize(samples.xi.size(), zeroVector());
    out.vector_jacobians.resize(samples.xi.size(), zeroMatrix());

    const bool component_blocked =
        rec.space->field_type() == FieldType::Vector &&
        rec.space->space_type() == spaces::SpaceType::Product;

    const int n_scalar_nodes =
        (rec.space->field_type() == FieldType::Vector && component_blocked)
            ? static_cast<int>(rec.space->dofs_per_element(cell_id) /
                               static_cast<std::size_t>(std::max(n_comp, 1)))
            : static_cast<int>(field_element.num_dofs());

    std::vector<Real> basis_values;
    std::vector<basis::Gradient> basis_gradients;

    for (std::size_t q = 0; q < samples.xi.size(); ++q) {
        const math::Vector<Real, 3> xi{samples.xi[q][0], samples.xi[q][1], samples.xi[q][2]};
        field_basis.evaluate_values(xi, basis_values);
        field_basis.evaluate_gradients(xi, basis_gradients);

        if (rec.space->field_type() == FieldType::Scalar || n_comp == 1) {
            Real val = 0.0;
            Vector3 grad_ref = zeroVector();
            for (int i = 0; i < n_scalar_nodes; ++i) {
                const Real ci = coeffs[static_cast<std::size_t>(i)];
                val += ci * basis_values[static_cast<std::size_t>(i)];
                for (int d = 0; d < dim; ++d) {
                    grad_ref[static_cast<std::size_t>(d)] +=
                        ci * basis_gradients[static_cast<std::size_t>(i)][static_cast<std::size_t>(d)];
                }
            }
            out.values[q] = scalar(val);
            out.scalar_gradients[q] = transformGradient(grad_ref, samples.inv_jacobians[q], dim);
            continue;
        }

        Vector3 val = zeroVector();
        Matrix3 jac = zeroMatrix();
        for (int comp = 0; comp < n_comp; ++comp) {
            Vector3 grad_ref = zeroVector();
            for (int i = 0; i < n_scalar_nodes; ++i) {
                const std::size_t coeff_idx = component_blocked
                    ? static_cast<std::size_t>(comp * n_scalar_nodes + i)
                    : static_cast<std::size_t>(i * n_comp + comp);
                const Real ci = coeffs[coeff_idx];
                val[static_cast<std::size_t>(comp)] +=
                    ci * basis_values[static_cast<std::size_t>(i)];
                for (int d = 0; d < dim; ++d) {
                    grad_ref[static_cast<std::size_t>(d)] +=
                        ci * basis_gradients[static_cast<std::size_t>(i)][static_cast<std::size_t>(d)];
                }
            }
            const auto grad_phys = transformGradient(grad_ref, samples.inv_jacobians[q], dim);
            for (int d = 0; d < dim; ++d) {
                jac[static_cast<std::size_t>(comp)][static_cast<std::size_t>(d)] =
                    grad_phys[static_cast<std::size_t>(d)];
            }
        }
        out.values[q] = vector(val, n_comp);
        out.vector_jacobians[q] = jac;
    }

    return out;
}

class ExpressionEvaluator {
public:
    ExpressionEvaluator(const systems::SystemStateView& state,
                        const CellSamples& samples,
                        const std::unordered_map<FieldId, FieldSampleData>& fields,
                        int dim)
        : state_(state)
        , samples_(samples)
        , fields_(fields)
        , dim_(dim)
    {
    }

    [[nodiscard]] EvalValue eval(const forms::FormExprNode& node, std::size_t q) const
    {
        using FT = forms::FormExprType;
        const auto kids = node.children();

        switch (node.type()) {
        case FT::Constant:
            return scalar(node.constantValue().value_or(0.0));
        case FT::TypedZero:
            return scalar(0.0);
        case FT::Time:
            return scalar(state_.time);
        case FT::TimeStep:
            return scalar(state_.dt);
        case FT::EffectiveTimeStep:
            return scalar(state_.effective_dt != 0.0 ? state_.effective_dt : state_.dt);
        case FT::Coordinate:
            return vector(samples_.x[q], dim_);
        case FT::Identity: {
            const int idim = node.identityDim().value_or(dim_);
            Matrix3 I = zeroMatrix();
            for (int i = 0; i < idim && i < 3; ++i) {
                I[static_cast<std::size_t>(i)][static_cast<std::size_t>(i)] = 1.0;
            }
            return tensor(I, idim, idim);
        }
        case FT::Coefficient:
            return evalCoefficient(node, q);
        case FT::StateField:
        case FT::DiscreteField:
            return fieldValue(node, q);
        case FT::Gradient:
            FE_THROW_IF(kids.size() != 1 || kids[0] == nullptr, InvalidArgumentException,
                        "DerivedResultEvaluator: Gradient expects one child");
            return fieldGradient(*kids[0], q);
        case FT::Negate:
            return negateValue(evalUnary(node, q));
        case FT::Add:
            return addValues(evalBinaryA(node, q), evalBinaryB(node, q));
        case FT::Subtract:
            return addValues(evalBinaryA(node, q), negateValue(evalBinaryB(node, q)));
        case FT::Multiply:
            return multiplyValues(evalBinaryA(node, q), evalBinaryB(node, q));
        case FT::Divide:
            return divideValues(evalBinaryA(node, q), evalBinaryB(node, q));
        case FT::InnerProduct:
        case FT::DoubleContraction:
            return innerValues(evalBinaryA(node, q), evalBinaryB(node, q));
        case FT::CrossProduct: {
            const auto a = evalBinaryA(node, q);
            const auto b = evalBinaryB(node, q);
            FE_THROW_IF(a.kind != EvalValue::Kind::Vector || b.kind != EvalValue::Kind::Vector,
                        InvalidArgumentException,
                        "DerivedResultEvaluator: cross product expects vector operands");
            return vector(vectorCross(a.v, b.v), 3);
        }
        case FT::Power: {
            const auto a = evalBinaryA(node, q);
            const auto b = evalBinaryB(node, q);
            FE_THROW_IF(a.kind != EvalValue::Kind::Scalar || b.kind != EvalValue::Kind::Scalar,
                        InvalidArgumentException,
                        "DerivedResultEvaluator: power expects scalar operands");
            return scalar(std::pow(a.s, b.s));
        }
        case FT::Minimum: {
            const auto a = requireScalar(evalBinaryA(node, q), "minimum");
            const auto b = requireScalar(evalBinaryB(node, q), "minimum");
            return scalar(std::min(a, b));
        }
        case FT::Maximum: {
            const auto a = requireScalar(evalBinaryA(node, q), "maximum");
            const auto b = requireScalar(evalBinaryB(node, q), "maximum");
            return scalar(std::max(a, b));
        }
        case FT::Less:
            return scalar(requireScalar(evalBinaryA(node, q), "less") <
                          requireScalar(evalBinaryB(node, q), "less") ? 1.0 : 0.0);
        case FT::LessEqual:
            return scalar(requireScalar(evalBinaryA(node, q), "less_equal") <=
                          requireScalar(evalBinaryB(node, q), "less_equal") ? 1.0 : 0.0);
        case FT::Greater:
            return scalar(requireScalar(evalBinaryA(node, q), "greater") >
                          requireScalar(evalBinaryB(node, q), "greater") ? 1.0 : 0.0);
        case FT::GreaterEqual:
            return scalar(requireScalar(evalBinaryA(node, q), "greater_equal") >=
                          requireScalar(evalBinaryB(node, q), "greater_equal") ? 1.0 : 0.0);
        case FT::Equal:
            return scalar(requireScalar(evalBinaryA(node, q), "equal") ==
                          requireScalar(evalBinaryB(node, q), "equal") ? 1.0 : 0.0);
        case FT::NotEqual:
            return scalar(requireScalar(evalBinaryA(node, q), "not_equal") !=
                          requireScalar(evalBinaryB(node, q), "not_equal") ? 1.0 : 0.0);
        case FT::Conditional: {
            FE_THROW_IF(kids.size() != 3, InvalidArgumentException,
                        "DerivedResultEvaluator: conditional expects three children");
            const auto cond = requireScalar(eval(*kids[0], q), "conditional");
            return cond != 0.0 ? eval(*kids[1], q) : eval(*kids[2], q);
        }
        case FT::AsVector: {
            Vector3 out = zeroVector();
            for (std::size_t i = 0; i < kids.size() && i < 3; ++i) {
                out[i] = requireScalar(eval(*kids[i], q), "asVector");
            }
            return vector(out, static_cast<int>(std::min<std::size_t>(kids.size(), 3)));
        }
        case FT::AsTensor: {
            const int rows = node.tensorRows().value_or(0);
            const int cols = node.tensorCols().value_or(0);
            FE_THROW_IF(rows <= 0 || cols <= 0 || rows > 3 || cols > 3,
                        InvalidArgumentException,
                        "DerivedResultEvaluator: unsupported asTensor shape");
            Matrix3 out = zeroMatrix();
            for (int r = 0; r < rows; ++r) {
                for (int c = 0; c < cols; ++c) {
                    const auto idx = static_cast<std::size_t>(r * cols + c);
                    out[static_cast<std::size_t>(r)][static_cast<std::size_t>(c)] =
                        requireScalar(eval(*kids[idx], q), "asTensor");
                }
            }
            return tensor(out, rows, cols);
        }
        case FT::Component:
            return evalComponent(node, q);
        case FT::Transpose:
            return transpose(evalUnary(node, q));
        case FT::Trace: {
            const auto a = evalUnary(node, q);
            FE_THROW_IF(a.kind != EvalValue::Kind::Tensor, InvalidArgumentException,
                        "DerivedResultEvaluator: trace expects tensor");
            Real tr = 0.0;
            for (int i = 0; i < std::min(a.rows, a.cols); ++i) {
                tr += a.m[static_cast<std::size_t>(i)][static_cast<std::size_t>(i)];
            }
            return scalar(tr);
        }
        case FT::SymmetricPart: {
            const auto a = evalUnary(node, q);
            FE_THROW_IF(a.kind != EvalValue::Kind::Tensor, InvalidArgumentException,
                        "DerivedResultEvaluator: sym expects tensor");
            FE_THROW_IF(a.rows != a.cols, InvalidArgumentException,
                        "DerivedResultEvaluator: sym expects square tensor");
            Matrix3 out = zeroMatrix();
            for (int r = 0; r < a.rows; ++r) {
                for (int c = 0; c < a.cols; ++c) {
                    out[static_cast<std::size_t>(r)][static_cast<std::size_t>(c)] =
                        0.5 * (a.m[static_cast<std::size_t>(r)][static_cast<std::size_t>(c)] +
                               a.m[static_cast<std::size_t>(c)][static_cast<std::size_t>(r)]);
                }
            }
            return tensor(out, a.rows, a.cols);
        }
        case FT::Norm: {
            const auto a = evalUnary(node, q);
            if (a.kind == EvalValue::Kind::Scalar) {
                return scalar(std::abs(a.s));
            }
            if (a.kind == EvalValue::Kind::Vector) {
                return scalar(vectorNorm(a.v));
            }
            Real sum = 0.0;
            for (int r = 0; r < a.rows; ++r) {
                for (int c = 0; c < a.cols; ++c) {
                    const Real x = a.m[static_cast<std::size_t>(r)][static_cast<std::size_t>(c)];
                    sum += x * x;
                }
            }
            return scalar(std::sqrt(sum));
        }
        case FT::AbsoluteValue:
            return scalar(std::abs(requireScalar(evalUnary(node, q), "abs")));
        case FT::Sign: {
            const auto a = requireScalar(evalUnary(node, q), "sign");
            return scalar((a > 0.0) - (a < 0.0));
        }
        case FT::Sqrt:
            return scalar(std::sqrt(requireScalar(evalUnary(node, q), "sqrt")));
        case FT::Exp:
            return scalar(std::exp(requireScalar(evalUnary(node, q), "exp")));
        case FT::Log:
            return scalar(std::log(requireScalar(evalUnary(node, q), "log")));
        default:
            FE_THROW(NotImplementedException,
                     "DerivedResultEvaluator: unsupported expression node '" + node.toString() + "'");
        }
    }

private:
    [[nodiscard]] EvalValue evalCoefficient(const forms::FormExprNode& node, std::size_t q) const
    {
        const auto& x = samples_.x[q];
        if (const auto* c = node.scalarCoefficient()) {
            return scalar((*c)(x[0], x[1], x[2]));
        }
        if (const auto* c = node.timeScalarCoefficient()) {
            return scalar((*c)(x[0], x[1], x[2], state_.time));
        }
        if (const auto* c = node.vectorCoefficient()) {
            const auto v = (*c)(x[0], x[1], x[2]);
            return vector({v[0], v[1], v[2]}, dim_);
        }
        if (const auto* c = node.matrixCoefficient()) {
            const auto m = (*c)(x[0], x[1], x[2]);
            Matrix3 out = zeroMatrix();
            for (int r = 0; r < 3; ++r) {
                for (int col = 0; col < 3; ++col) {
                    out[static_cast<std::size_t>(r)][static_cast<std::size_t>(col)] =
                        m[static_cast<std::size_t>(r)][static_cast<std::size_t>(col)];
                }
            }
            return tensor(out, dim_, dim_);
        }
        FE_THROW(NotImplementedException,
                 "DerivedResultEvaluator: unsupported coefficient type");
    }

    [[nodiscard]] EvalValue fieldValue(const forms::FormExprNode& node, std::size_t q) const
    {
        const auto fid = node.fieldId();
        FE_THROW_IF(!fid, InvalidArgumentException,
                    "DerivedResultEvaluator: field expression without FieldId");
        const auto it = fields_.find(*fid);
        FE_THROW_IF(it == fields_.end(), InvalidArgumentException,
                    "DerivedResultEvaluator: unbound field " + std::to_string(*fid));
        return it->second.values[q];
    }

    [[nodiscard]] EvalValue fieldGradient(const forms::FormExprNode& node, std::size_t q) const
    {
        using FT = forms::FormExprType;
        FE_THROW_IF(node.type() != FT::StateField && node.type() != FT::DiscreteField,
                    NotImplementedException,
                    "DerivedResultEvaluator: Gradient currently supports StateField/DiscreteField children");
        const auto fid = node.fieldId();
        FE_THROW_IF(!fid, InvalidArgumentException,
                    "DerivedResultEvaluator: grad(field) without FieldId");
        const auto it = fields_.find(*fid);
        FE_THROW_IF(it == fields_.end(), InvalidArgumentException,
                    "DerivedResultEvaluator: unbound gradient field " + std::to_string(*fid));
        const auto& field = it->second;
        if (field.field_type == FieldType::Scalar || field.components == 1) {
            return vector(field.scalar_gradients[q], dim_);
        }
        return tensor(field.vector_jacobians[q], field.components, dim_);
    }

    [[nodiscard]] EvalValue evalUnary(const forms::FormExprNode& node, std::size_t q) const
    {
        const auto kids = node.children();
        FE_THROW_IF(kids.size() != 1 || kids[0] == nullptr, InvalidArgumentException,
                    "DerivedResultEvaluator: unary node expects one child");
        return eval(*kids[0], q);
    }

    [[nodiscard]] EvalValue evalBinaryA(const forms::FormExprNode& node, std::size_t q) const
    {
        const auto kids = node.children();
        FE_THROW_IF(kids.size() != 2 || kids[0] == nullptr || kids[1] == nullptr,
                    InvalidArgumentException,
                    "DerivedResultEvaluator: binary node expects two children");
        return eval(*kids[0], q);
    }

    [[nodiscard]] EvalValue evalBinaryB(const forms::FormExprNode& node, std::size_t q) const
    {
        const auto kids = node.children();
        FE_THROW_IF(kids.size() != 2 || kids[0] == nullptr || kids[1] == nullptr,
                    InvalidArgumentException,
                    "DerivedResultEvaluator: binary node expects two children");
        return eval(*kids[1], q);
    }

    [[nodiscard]] Real requireScalar(const EvalValue& value, const char* op) const
    {
        FE_THROW_IF(value.kind != EvalValue::Kind::Scalar, InvalidArgumentException,
                    std::string("DerivedResultEvaluator: ") + op + " expects scalar");
        return value.s;
    }

    [[nodiscard]] EvalValue evalComponent(const forms::FormExprNode& node, std::size_t q) const
    {
        const auto a = evalUnary(node, q);
        const int i = node.componentIndex0().value_or(0);
        const auto j = node.componentIndex1();
        if (a.kind == EvalValue::Kind::Scalar) {
            FE_THROW_IF(i != 0 || j.has_value(), InvalidArgumentException,
                        "DerivedResultEvaluator: scalar component index out of range");
            return a;
        }
        if (a.kind == EvalValue::Kind::Vector) {
            FE_THROW_IF(j.has_value() || i < 0 || i >= a.rows, InvalidArgumentException,
                        "DerivedResultEvaluator: vector component index out of range");
            return scalar(a.v[static_cast<std::size_t>(i)]);
        }
        const int col = j.value_or(0);
        FE_THROW_IF(i < 0 || i >= a.rows || col < 0 || col >= a.cols, InvalidArgumentException,
                    "DerivedResultEvaluator: tensor component index out of range");
        return scalar(a.m[static_cast<std::size_t>(i)][static_cast<std::size_t>(col)]);
    }

    [[nodiscard]] EvalValue transpose(const EvalValue& a) const
    {
        FE_THROW_IF(a.kind != EvalValue::Kind::Tensor, InvalidArgumentException,
                    "DerivedResultEvaluator: transpose expects tensor");
        Matrix3 out = zeroMatrix();
        for (int r = 0; r < a.rows; ++r) {
            for (int c = 0; c < a.cols; ++c) {
                out[static_cast<std::size_t>(c)][static_cast<std::size_t>(r)] =
                    a.m[static_cast<std::size_t>(r)][static_cast<std::size_t>(c)];
            }
        }
        return tensor(out, a.cols, a.rows);
    }

    const systems::SystemStateView& state_;
    const CellSamples& samples_;
    const std::unordered_map<FieldId, FieldSampleData>& fields_;
    int dim_{3};
};

void writeValueToTuple(const EvalValue& value,
                       std::size_t tuple,
                       std::size_t components,
                       std::span<double> data)
{
    const auto offset = tuple * components;
    FE_THROW_IF(offset + components > data.size(), InvalidArgumentException,
                "DerivedResultEvaluator: derived output buffer is too small");
    if (value.kind == EvalValue::Kind::Scalar) {
        FE_THROW_IF(components != 1, InvalidArgumentException,
                    "DerivedResultEvaluator: scalar result component mismatch");
        data[offset] = value.s;
        return;
    }
    if (value.kind == EvalValue::Kind::Vector) {
        FE_THROW_IF(components > static_cast<std::size_t>(value.rows), InvalidArgumentException,
                    "DerivedResultEvaluator: vector result component mismatch");
        for (std::size_t c = 0; c < components; ++c) {
            data[offset + c] = value.v[c];
        }
        return;
    }
    FE_THROW_IF(components > static_cast<std::size_t>(value.rows * value.cols), InvalidArgumentException,
                "DerivedResultEvaluator: tensor result component mismatch");
    std::size_t k = 0;
    for (int r = 0; r < value.rows; ++r) {
        for (int c = 0; c < value.cols; ++c) {
            if (k < components) {
                data[offset + k] = value.m[static_cast<std::size_t>(r)][static_cast<std::size_t>(c)];
            }
            ++k;
        }
    }
}

[[nodiscard]] EvalValue averageValues(const std::vector<EvalValue>& values,
                                      const std::vector<Real>& weights)
{
    FE_THROW_IF(values.empty() || values.size() != weights.size(), InvalidArgumentException,
                "DerivedResultEvaluator: invalid value/weight buffers");
    EvalValue accum = typedZeroLike(values.front());
    Real measure = 0.0;
    for (std::size_t q = 0; q < values.size(); ++q) {
        accum = addValues(accum, scaleValue(values[q], weights[q]));
        measure += weights[q];
    }
    FE_THROW_IF(measure <= 0.0, InvalidArgumentException,
                "DerivedResultEvaluator: non-positive cell measure");
    return scaleValue(accum, Real(1) / measure);
}

void evaluateCellResult(const systems::FESystem& system,
                        const systems::SystemStateView& state,
                        const DerivedResultDefinition& def,
                        std::span<double> out)
{
    const auto& mesh = system.meshAccess();
    const auto view = makeSolutionView(state);
    const int dim = mesh.dimension();
    const auto components = componentCount(def.shape);

    FieldId primary_field = INVALID_FIELD_ID;
    if (!def.referenced_fields.empty()) {
        primary_field = def.referenced_fields.front();
    } else if (system.fieldMap().numFields() > 0) {
        primary_field = 0;
    }
    FE_THROW_IF(primary_field == INVALID_FIELD_ID, InvalidArgumentException,
                "Derived result '" + def.name + "' has no field to define cell geometry");
    const auto& primary_record = system.fieldRecord(primary_field);
    FE_CHECK_NOT_NULL(primary_record.space.get(),
                      "DerivedResultEvaluator: primary field space");

    mesh.forEachCell([&](GlobalIndex cell_id) {
        const auto samples = prepareCellSamples(mesh, cell_id, *primary_record.space, def.policy);
        std::unordered_map<FieldId, FieldSampleData> fields;
        for (const auto field : def.referenced_fields) {
            fields.emplace(field, sampleFieldOnCell(system, field, cell_id, samples, state, view.get()));
        }
        ExpressionEvaluator evaluator(state, samples, fields, dim);

        std::vector<EvalValue> q_values;
        q_values.reserve(samples.xi.size());
        for (std::size_t q = 0; q < samples.xi.size(); ++q) {
            q_values.push_back(evaluator.eval(*def.expression.node(), q));
        }
        const EvalValue result =
            def.policy == DerivedResultPolicy::CellAverage
                ? averageValues(q_values, samples.integration_weights)
                : q_values.front();
        writeValueToTuple(result, static_cast<std::size_t>(cell_id), components, out);
    });
}

void evaluateVertexPointValueResult(const systems::FESystem& system,
                                    const systems::SystemStateView& state,
                                    const DerivedResultDefinition& def,
                                    std::span<double> out)
{
    const auto& mesh = system.meshAccess();
    const int dim = mesh.dimension();
    const auto view = makeSolutionView(state);
    const auto components = componentCount(def.shape);

    FE_THROW_IF(def.referenced_fields.empty(), InvalidArgumentException,
                "Vertex PointValue derived result '" + def.name +
                    "' requires at least one referenced field");

    for (GlobalIndex vertex = 0; vertex < mesh.numVertices(); ++vertex) {
        const auto x = mesh.getNodeCoordinates(vertex);
        CellSamples samples;
        samples.xi.push_back({0.0, 0.0, 0.0});
        samples.x.push_back({x[0], x[1], x[2]});
        samples.integration_weights.push_back(1.0);
        samples.quadrature_weights.push_back(1.0);
        samples.inv_jacobians.push_back(zeroMatrix());

        std::unordered_map<FieldId, FieldSampleData> fields;
        for (const auto field : def.referenced_fields) {
            FieldSampleData data;
            data.field = field;
            const auto& rec = system.fieldRecord(field);
            data.field_type = rec.space ? rec.space->field_type() : FieldType::Scalar;
            data.components = std::max(1, rec.components);
            data.value_dimension = data.components;
            data.values.resize(1);

            const auto* entity_map = system.fieldDofHandler(field).getEntityDofMap();
            FE_CHECK_NOT_NULL(entity_map,
                              "DerivedResultEvaluator: vertex PointValue requires entity DOF map");
            const auto vdofs = entity_map->getVertexDofs(vertex);
            FE_THROW_IF(vdofs.empty(), InvalidArgumentException,
                        "DerivedResultEvaluator: vertex has no field DOFs");

            if (data.components == 1) {
                data.values[0] =
                    scalar(readSolutionEntry(vdofs[0] + system.fieldDofOffset(field), state, view.get()));
            } else {
                Vector3 v = zeroVector();
                FE_THROW_IF(vdofs.size() < static_cast<std::size_t>(data.components), InvalidArgumentException,
                            "DerivedResultEvaluator: vertex DOF component mismatch");
                for (int c = 0; c < data.components; ++c) {
                    v[static_cast<std::size_t>(c)] =
                        readSolutionEntry(vdofs[static_cast<std::size_t>(c)] +
                                              system.fieldDofOffset(field),
                                          state,
                                          view.get());
                }
                data.values[0] = vector(v, data.components);
            }
            fields.emplace(field, std::move(data));
        }

        ExpressionEvaluator evaluator(state, samples, fields, dim);
        const auto value = evaluator.eval(*def.expression.node(), 0);
        writeValueToTuple(value, static_cast<std::size_t>(vertex), components, out);
    }
}

void evaluateVertexPatchAverageResult(const systems::FESystem& system,
                                      const systems::SystemStateView& state,
                                      const DerivedResultDefinition& def,
                                      std::span<double> out)
{
    const auto& mesh = system.meshAccess();
    const auto components = componentCount(def.shape);
    std::vector<double> weights(static_cast<std::size_t>(mesh.numVertices()), 0.0);
    const auto view = makeSolutionView(state);
    const int dim = mesh.dimension();

    FieldId primary_field = INVALID_FIELD_ID;
    if (!def.referenced_fields.empty()) {
        primary_field = def.referenced_fields.front();
    }
    FE_THROW_IF(primary_field == INVALID_FIELD_ID, InvalidArgumentException,
                "Vertex PatchAverage derived result '" + def.name +
                    "' requires at least one referenced field");
    const auto& primary_record = system.fieldRecord(primary_field);
    FE_CHECK_NOT_NULL(primary_record.space.get(),
                      "DerivedResultEvaluator: primary field space");

    mesh.forEachCell([&](GlobalIndex cell_id) {
        auto samples = prepareCellSamples(mesh, cell_id, *primary_record.space,
                                          DerivedResultPolicy::CellAverage);
        std::unordered_map<FieldId, FieldSampleData> fields;
        for (const auto field : def.referenced_fields) {
            fields.emplace(field, sampleFieldOnCell(system, field, cell_id, samples, state, view.get()));
        }
        ExpressionEvaluator evaluator(state, samples, fields, dim);
        std::vector<EvalValue> q_values;
        q_values.reserve(samples.xi.size());
        for (std::size_t q = 0; q < samples.xi.size(); ++q) {
            q_values.push_back(evaluator.eval(*def.expression.node(), q));
        }
        const auto cell_value = averageValues(q_values, samples.integration_weights);

        std::vector<GlobalIndex> nodes;
        mesh.getCellNodes(cell_id, nodes);
        for (const auto vertex : nodes) {
            const auto tuple = static_cast<std::size_t>(vertex);
            const auto offset = tuple * components;
            std::vector<double> tmp(components, 0.0);
            writeValueToTuple(cell_value, 0, components, std::span<double>(tmp));
            for (std::size_t c = 0; c < components; ++c) {
                out[offset + c] += tmp[c];
            }
            weights[tuple] += 1.0;
        }
    });

    for (std::size_t v = 0; v < weights.size(); ++v) {
        if (weights[v] <= 0.0) {
            continue;
        }
        const auto offset = v * components;
        for (std::size_t c = 0; c < components; ++c) {
            out[offset + c] /= weights[v];
        }
    }
}

} // namespace

DerivedResultEvaluator::DerivedResultEvaluator(const systems::FESystem& system,
                                               const systems::SystemStateView& state)
    : system_(system)
    , state_(state)
{
}

void DerivedResultEvaluator::evaluateToMeshField(MeshBase& mesh,
                                                 const DerivedResultDefinition& def,
                                                 const DerivedResultOutputOptions& options) const
{
#if defined(SVMP_FE_WITH_MESH) && SVMP_FE_WITH_MESH
    if (!def.enabled) {
        return;
    }

    const auto kind = meshEntityKind(def.scope);
    FE_THROW_IF(!kind.has_value(), NotImplementedException,
                "Derived result scope '" + std::string(toString(def.scope)) +
                    "' cannot be written directly as a mesh field yet");

    const auto components = componentCount(def.shape);
    const auto handle = ensureDerivedResultField(mesh, *kind, def.name, components, options.overwrite);
    auto data = derivedResultFieldData(mesh, handle, components);

    switch (def.scope) {
    case DerivedResultScope::Cell:
        evaluateCellResult(system_, state_, def, data);
        return;
    case DerivedResultScope::Vertex:
        if (def.policy == DerivedResultPolicy::PointValue) {
            evaluateVertexPointValueResult(system_, state_, def, data);
            return;
        }
        if (def.policy == DerivedResultPolicy::PatchAverage) {
            evaluateVertexPatchAverageResult(system_, state_, def, data);
            return;
        }
        break;
    case DerivedResultScope::Face:
    case DerivedResultScope::BoundaryFace:
    case DerivedResultScope::Edge:
    case DerivedResultScope::QuadraturePoint:
        break;
    }

    FE_THROW(NotImplementedException,
             "Derived result '" + def.name + "' scope/policy is registered but not implemented: " +
                 std::string(toString(def.scope)) + "/" + std::string(toString(def.policy)));
#else
    (void)mesh;
    (void)def;
    (void)options;
    FE_THROW(NotImplementedException,
             "Derived result evaluation to mesh fields requires FE_WITH_MESH");
#endif
}

} // namespace post
} // namespace FE
} // namespace svmp
