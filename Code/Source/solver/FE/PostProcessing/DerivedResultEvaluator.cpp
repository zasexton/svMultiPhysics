#include "PostProcessing/DerivedResultEvaluator.h"

#include "Assembly/Assembler.h"
#include "Assembly/FunctionalAssembler.h"
#include "Assembly/GlobalSystemView.h"
#include "Assembly/TimeIntegrationContext.h"
#include "Backends/Interfaces/GenericVector.h"
#include "Basis/BasisFunction.h"
#include "Core/AlignedAllocator.h"
#include "Core/Alignment.h"
#include "Core/FEException.h"
#include "Core/Logger.h"
#include "Dofs/DofHandler.h"
#include "Dofs/DofMap.h"
#include "Dofs/EntityDofMap.h"
#include "Elements/Element.h"
#include "Forms/BoundaryFunctional.h"
#include "Forms/JIT/ExternalCalls.h"
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
#include <optional>
#include <span>
#include <stdexcept>
#include <unordered_map>
#include <utility>
#include <vector>

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
makeVectorView(const backends::GenericVector* vector)
{
    if (vector == nullptr) {
        return nullptr;
    }
    auto* vec = const_cast<backends::GenericVector*>(vector);
    return vec->createAssemblyView();
}

[[nodiscard]] std::unique_ptr<assembly::GlobalSystemView>
makeSolutionView(const systems::SystemStateView& state)
{
    return makeVectorView(state.u_vector);
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

forms::FormExpr scalarComponentExpression(const DerivedResultDefinition& def,
                                          std::size_t component)
{
    const auto components = componentCount(def.shape);
    FE_THROW_IF(component >= components, InvalidArgumentException,
                "Derived result '" + def.name + "' component index out of range");

    if (components == 1) {
        return def.expression;
    }

    switch (def.shape.kind) {
    case systems::FEQuantityShapeKind::Vector:
        return def.expression.component(static_cast<int>(component));
    case systems::FEQuantityShapeKind::Tensor: {
        const int dim = def.shape.spatial_dim > 0
                            ? def.shape.spatial_dim
                            : static_cast<int>(std::sqrt(static_cast<double>(components)));
        FE_THROW_IF(dim <= 0 || static_cast<std::size_t>(dim * dim) != components,
                    InvalidArgumentException,
                    "Derived result '" + def.name + "' has invalid tensor shape");
        return def.expression.component(static_cast<int>(component / static_cast<std::size_t>(dim)),
                                        static_cast<int>(component % static_cast<std::size_t>(dim)));
    }
    case systems::FEQuantityShapeKind::Scalar:
        break;
    }

    FE_THROW(InvalidArgumentException,
             "Derived result '" + def.name + "' has scalar shape with multiple components");
}

forms::SymbolicOptions derivedResultCompilerOptions()
{
    forms::SymbolicOptions options{};
    options.jit.enable = true;
    return options;
}

int requiredHistoryStatesForExpression(const forms::FormExprNode& node,
                                       const assembly::TimeIntegrationContext* time_context,
                                       const std::string& result_name)
{
    int required = 0;

    const auto visit = [&](const auto& self, const forms::FormExprNode& n) -> void {
        switch (n.type()) {
        case forms::FormExprType::PreviousSolutionRef:
            required = std::max(required, n.historyIndex().value_or(1));
            break;
        case forms::FormExprType::TimeDerivative: {
            const int order = n.timeDerivativeOrder().value_or(1);
            FE_THROW_IF(time_context == nullptr, InvalidArgumentException,
                        "Derived result '" + result_name +
                            "' uses a time derivative but no time-integration context was provided");
            const auto* stencil = time_context->stencil(order);
            FE_THROW_IF(stencil == nullptr, InvalidArgumentException,
                        "Derived result '" + result_name +
                            "' uses an unavailable time-derivative stencil");
            required = std::max(required, stencil->requiredHistoryStates());
            break;
        }
        case forms::FormExprType::HistoryWeightedSum:
        case forms::FormExprType::HistoryConvolution: {
            const auto children = n.children();
            required = std::max(required, static_cast<int>(children.size()));
            break;
        }
        default:
            break;
        }

        for (const auto* child : n.children()) {
            if (child != nullptr) {
                self(self, *child);
            }
        }
    };

    visit(visit, node);
    return required;
}

int requiredHistoryStatesForDerivedResult(const DerivedResultDefinition& def,
                                          const systems::SystemStateView& state)
{
    FE_CHECK_NOT_NULL(def.expression.node(),
                      "DerivedResultEvaluator: derived expression node");
    return requiredHistoryStatesForExpression(*def.expression.node(),
                                              state.time_integration,
                                              def.name);
}

std::vector<std::shared_ptr<assembly::FunctionalKernel>>
makeCellFunctionalComponentKernels(const DerivedResultDefinition& def)
{
    const auto components = componentCount(def.shape);
    std::vector<std::shared_ptr<assembly::FunctionalKernel>> kernels;
    kernels.reserve(components);

    const auto compiler_options = derivedResultCompilerOptions();
    for (std::size_t c = 0; c < components; ++c) {
        forms::BoundaryFunctional functional;
        functional.name = def.name + "[" + std::to_string(c) + "]";
        functional.integrand = scalarComponentExpression(def, c);
        functional.is_domain_functional = true;
        functional.reduction = forms::BoundaryFunctional::Reduction::Sum;
        kernels.push_back(forms::compileBoundaryFunctionalKernel(functional, compiler_options));
    }

    return kernels;
}

void configureDerivedFunctionalAssembler(assembly::FunctionalAssembler& assembler,
                                         const systems::FESystem& system,
                                         const systems::SystemStateView& state,
                                         FieldId primary_field,
                                         const assembly::GlobalSystemView* solution_view,
                                         const assembly::GlobalSystemView* prev_solution_view,
                                         const assembly::GlobalSystemView* prev_solution2_view,
                                         int required_history_states)
{
    const auto& primary_record = system.fieldRecord(primary_field);
    FE_CHECK_NOT_NULL(primary_record.space.get(),
                      "DerivedResultEvaluator: primary field space");

    assembler.setMesh(system.meshAccess());
    assembler.setDofMap(system.fieldDofHandler(primary_field).getDofMap());
    assembler.setSpace(*primary_record.space);
    assembler.setPrimaryField(primary_field);
    assembler.setPrimaryFieldDofOffset(system.fieldDofOffset(primary_field));

    if (solution_view != nullptr) {
        assembler.setSolutionView(solution_view);
    } else if (!state.u.empty()) {
        assembler.setSolution(state.u);
    }

    if (required_history_states > 0) {
        if (!state.u_history.empty()) {
            FE_THROW_IF(static_cast<int>(state.u_history.size()) < required_history_states,
                        InvalidArgumentException,
                        "DerivedResultEvaluator: insufficient solution history for derived result");
            for (int k = 1; k <= required_history_states; ++k) {
                assembler.setPreviousSolutionK(k, state.u_history[static_cast<std::size_t>(k - 1)]);
            }
        } else {
            FE_THROW_IF(required_history_states > 2, InvalidArgumentException,
                        "DerivedResultEvaluator: derived result requires more than two history states");
            if (required_history_states >= 1) {
                if (prev_solution_view != nullptr) {
                    assembler.setPreviousSolutionView(prev_solution_view);
                } else if (!state.u_prev.empty()) {
                    assembler.setPreviousSolution(state.u_prev);
                }
            }
            if (required_history_states >= 2) {
                if (prev_solution2_view != nullptr) {
                    assembler.setPreviousSolution2View(prev_solution2_view);
                } else if (!state.u_prev2.empty()) {
                    assembler.setPreviousSolution2(state.u_prev2);
                }
            }
        }

        if (prev_solution_view != nullptr) {
            assembler.setPreviousSolutionViewK(1, prev_solution_view);
        }
        if (required_history_states >= 2 && prev_solution2_view != nullptr) {
            assembler.setPreviousSolutionViewK(2, prev_solution2_view);
        }
    }

    assembler.setTimeIntegrationContext(state.time_integration);
    assembler.setTime(static_cast<Real>(state.time));
    assembler.setTimeStep(static_cast<Real>(state.dt));
}

void registerDerivedFunctionalFields(assembly::FunctionalAssembler& assembler,
                                     const systems::FESystem& system,
                                     const DerivedResultDefinition& def,
                                     FieldId primary_field)
{
    std::vector<FieldId> referenced = def.referenced_fields;
    if (referenced.empty()) {
        referenced.push_back(primary_field);
    }
    std::sort(referenced.begin(), referenced.end());
    referenced.erase(std::unique(referenced.begin(), referenced.end()), referenced.end());

    for (const auto field : referenced) {
        if (field == primary_field) {
            continue;
        }
        const auto& rec = system.fieldRecord(field);
        FE_CHECK_NOT_NULL(rec.space.get(),
                          "DerivedResultEvaluator: referenced field space");
        const auto& dh = system.fieldDofHandler(field);

        assembly::FieldSolutionBinding binding;
        binding.field = field;
        binding.space = rec.space.get();
        binding.dof_map = &dh.getDofMap();
        binding.dof_offset = system.fieldDofOffset(field);
        // Derived-result assemblers are configured with the primary field-local
        // DofMap, so secondary fields should gather from their own field-local
        // map plus monolithic offset instead of scanning the primary cell DOFs.
        binding.field_global_size = 0;
        binding.field_type = rec.space->field_type();
        binding.value_dimension = rec.components;
        binding.n_components = rec.components;
        assembler.registerFieldBinding(binding);
    }
}

struct DerivedFunctionalRuntimeBindings {
    std::unique_ptr<assembly::GlobalSystemView> solution_view;
    std::unique_ptr<assembly::GlobalSystemView> prev_solution_view;
    std::unique_ptr<assembly::GlobalSystemView> prev_solution2_view;
    std::function<std::optional<Real>(std::string_view)> get_real_param_wrapped;
    std::function<std::optional<params::Value>(std::string_view)> get_param_wrapped;
    forms::jit::external::ExternalCallTableV1 jit_table{};
    std::vector<Real, AlignedAllocator<Real, kFEPreferredAlignmentBytes>> jit_constants;
};

void bindDerivedFunctionalRuntime(assembly::FunctionalAssembler& assembler,
                                  const systems::FESystem& system,
                                  const systems::SystemStateView& state,
                                  DerivedFunctionalRuntimeBindings& bindings)
{
    bindings.solution_view = makeVectorView(state.u_vector);
    bindings.prev_solution_view = makeVectorView(state.u_prev_vector);
    bindings.prev_solution2_view = makeVectorView(state.u_prev2_vector);

    const auto& preg = system.parameterRegistry();
    const bool have_param_contracts = !preg.specs().empty();
    if (have_param_contracts) {
        bindings.get_real_param_wrapped = preg.makeRealGetter(state);
        bindings.get_param_wrapped = preg.makeParamGetter(state);
    }
    assembler.setRealParameterGetter(have_param_contracts
                                         ? &bindings.get_real_param_wrapped
                                         : (state.getRealParam ? &state.getRealParam : nullptr));
    assembler.setParameterGetter(have_param_contracts
                                     ? &bindings.get_param_wrapped
                                     : (state.getParam ? &state.getParam : nullptr));

    bindings.jit_table.context = state.user_data;
    assembler.setUserData(&bindings.jit_table);

    if (have_param_contracts && preg.slotCount() > 0u) {
        const auto slots = preg.evaluateRealSlots(state);
        bindings.jit_constants.assign(slots.begin(), slots.end());
        assembler.setJITConstants(bindings.jit_constants);
    } else {
        assembler.setJITConstants({});
    }
    assembler.setCoupledValues({}, {});
}

void evaluateCellFunctionalResult(const systems::FESystem& system,
                                  const systems::SystemStateView& state,
                                  const DerivedResultDefinition& def,
                                  FieldId primary_field,
                                  const assembly::FunctionalAssembler::CellFunctionalVisitor& visitor)
{
    auto kernels = makeCellFunctionalComponentKernels(def);
    std::vector<assembly::FunctionalKernel*> raw_kernels;
    raw_kernels.reserve(kernels.size());
    for (const auto& kernel : kernels) {
        raw_kernels.push_back(kernel.get());
    }

    assembly::FunctionalAssembler assembler;
    DerivedFunctionalRuntimeBindings bindings;
    bindDerivedFunctionalRuntime(assembler, system, state, bindings);
    const int required_history_states = requiredHistoryStatesForDerivedResult(def, state);
    configureDerivedFunctionalAssembler(assembler,
                                        system,
                                        state,
                                        primary_field,
                                        bindings.solution_view.get(),
                                        bindings.prev_solution_view.get(),
                                        bindings.prev_solution2_view.get(),
                                        required_history_states);
    registerDerivedFunctionalFields(assembler, system, def, primary_field);
    assembler.evaluateCellFunctionals(raw_kernels, visitor);
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

[[nodiscard]] bool usesCellAverageSweep(const DerivedResultDefinition& def) noexcept
{
    if (!def.enabled) {
        return false;
    }
    return (def.scope == DerivedResultScope::Cell &&
            def.policy == DerivedResultPolicy::CellAverage) ||
           (def.scope == DerivedResultScope::Vertex &&
            def.policy == DerivedResultPolicy::PatchAverage);
}

[[nodiscard]] std::string cellAverageCacheKey(const DerivedResultDefinition& def)
{
    std::string key;
    key.reserve(def.expression.toString().size() + 96u + def.referenced_fields.size() * 12u);
    key += "expr=";
    key += def.expression.toString();
    key += "|shape=";
    key += std::to_string(static_cast<int>(def.shape.kind));
    key += ",";
    key += std::to_string(def.shape.components);
    key += ",";
    key += std::to_string(def.shape.spatial_dim);
    key += "|marker=";
    key += def.marker ? std::to_string(*def.marker) : std::string("none");
    key += "|fields=";

    std::vector<FieldId> fields = def.referenced_fields;
    std::sort(fields.begin(), fields.end());
    for (const auto field : fields) {
        key += std::to_string(field);
        key += ",";
    }
    return key;
}

[[nodiscard]] std::unordered_set<std::string>
findReusableCellAverageKeys(const systems::FESystem& system)
{
    std::unordered_map<std::string, std::size_t> counts;
    for (const auto& def : system.derivedResults()) {
        if (usesCellAverageSweep(def)) {
            ++counts[cellAverageCacheKey(def)];
        }
    }

    std::unordered_set<std::string> reusable;
    for (const auto& [key, count] : counts) {
        if (count > 1u) {
            reusable.insert(key);
        }
    }
    return reusable;
}

[[nodiscard]] FieldId primaryFieldForCellAverage(const systems::FESystem& system,
                                                 const DerivedResultDefinition& def)
{
    if (!def.referenced_fields.empty()) {
        return def.referenced_fields.front();
    }
    if (def.scope == DerivedResultScope::Cell && system.fieldMap().numFields() > 0) {
        return 0;
    }
    FE_THROW(InvalidArgumentException,
             "Derived result '" + def.name + "' has no field to define cell-average geometry");
}

[[nodiscard]] std::vector<double>
computeCellAverageValues(const systems::FESystem& system,
                         const systems::SystemStateView& state,
                         const DerivedResultDefinition& def)
{
    const auto& mesh = system.meshAccess();
    const auto components = componentCount(def.shape);
    const auto primary_field = primaryFieldForCellAverage(system, def);
    std::vector<double> values(static_cast<std::size_t>(mesh.numCells()) * components, 0.0);

    evaluateCellFunctionalResult(
        system,
        state,
        def,
        primary_field,
        [&](GlobalIndex cell_id, std::span<const Real> cell_values, Real cell_measure) {
            FE_THROW_IF(cell_values.size() != components, InvalidArgumentException,
                        "DerivedResultEvaluator: JIT component count mismatch for '" + def.name + "'");
            FE_THROW_IF(cell_measure <= 0.0, InvalidArgumentException,
                        "DerivedResultEvaluator: non-positive cell measure");
            const auto offset = static_cast<std::size_t>(cell_id) * components;
            FE_THROW_IF(offset + components > values.size(), InvalidArgumentException,
                        "DerivedResultEvaluator: cell-average cache buffer is too small");
            for (std::size_t c = 0; c < components; ++c) {
                values[offset + c] = static_cast<double>(cell_values[c] / cell_measure);
            }
        });

    return values;
}

void copyCellAverageValues(const std::vector<double>& values,
                           std::size_t components,
                           std::span<double> out)
{
    FE_THROW_IF(values.size() != out.size(), InvalidArgumentException,
                "DerivedResultEvaluator: cached cell-average field size mismatch");
    FE_THROW_IF(components == 0u, InvalidArgumentException,
                "DerivedResultEvaluator: cached cell-average component count is zero");
    std::copy(values.begin(), values.end(), out.begin());
}

void evaluateVertexPatchAverageFromCellValues(const systems::FESystem& system,
                                              const std::vector<double>& cell_values,
                                              std::size_t components,
                                              std::span<double> out)
{
    const auto& mesh = system.meshAccess();
    FE_THROW_IF(cell_values.size() != static_cast<std::size_t>(mesh.numCells()) * components,
                InvalidArgumentException,
                "DerivedResultEvaluator: cached cell-average input size mismatch");

    std::fill(out.begin(), out.end(), 0.0);
    std::vector<double> weights(static_cast<std::size_t>(mesh.numVertices()), 0.0);
    std::vector<GlobalIndex> nodes;

    mesh.forEachCell([&](GlobalIndex cell_id) {
        nodes.clear();
        mesh.getCellNodes(cell_id, nodes);
        const auto cell_offset = static_cast<std::size_t>(cell_id) * components;
        for (const auto vertex : nodes) {
            const auto tuple = static_cast<std::size_t>(vertex);
            const auto vertex_offset = tuple * components;
            FE_THROW_IF(vertex_offset + components > out.size(), InvalidArgumentException,
                        "DerivedResultEvaluator: vertex output buffer is too small");
            for (std::size_t c = 0; c < components; ++c) {
                out[vertex_offset + c] += cell_values[cell_offset + c];
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

void evaluateCellResult(const systems::FESystem& system,
                        const systems::SystemStateView& state,
                        const DerivedResultDefinition& def,
                        std::span<double> out)
{
    const auto& mesh = system.meshAccess();
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

    if (def.policy == DerivedResultPolicy::CellAverage) {
        evaluateCellFunctionalResult(
            system,
            state,
            def,
            primary_field,
            [&](GlobalIndex cell_id, std::span<const Real> values, Real cell_measure) {
                FE_THROW_IF(values.size() != components, InvalidArgumentException,
                            "DerivedResultEvaluator: JIT component count mismatch for '" + def.name + "'");
                FE_THROW_IF(cell_measure <= 0.0, InvalidArgumentException,
                            "DerivedResultEvaluator: non-positive cell measure");
                const auto tuple = static_cast<std::size_t>(cell_id);
                const auto offset = tuple * components;
                FE_THROW_IF(offset + components > out.size(), InvalidArgumentException,
                            "DerivedResultEvaluator: derived output buffer is too small");
                for (std::size_t c = 0; c < components; ++c) {
                    out[offset + c] = static_cast<double>(values[c] / cell_measure);
                }
            });
        return;
    }

    const auto view = makeSolutionView(state);

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
    std::fill(out.begin(), out.end(), 0.0);

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

    std::vector<GlobalIndex> nodes;
    evaluateCellFunctionalResult(
        system,
        state,
        def,
        primary_field,
        [&](GlobalIndex cell_id, std::span<const Real> values, Real cell_measure) {
            FE_THROW_IF(values.size() != components, InvalidArgumentException,
                        "DerivedResultEvaluator: JIT component count mismatch for '" + def.name + "'");
            FE_THROW_IF(cell_measure <= 0.0, InvalidArgumentException,
                        "DerivedResultEvaluator: non-positive cell measure");
            nodes.clear();
            mesh.getCellNodes(cell_id, nodes);
            for (const auto vertex : nodes) {
                const auto tuple = static_cast<std::size_t>(vertex);
                const auto offset = tuple * components;
                for (std::size_t c = 0; c < components; ++c) {
                    out[offset + c] += static_cast<double>(values[c] / cell_measure);
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
    , reusable_cell_average_keys_(findReusableCellAverageKeys(system))
{
    if (!reusable_cell_average_keys_.empty()) {
        FE_LOG_INFO("DerivedResultEvaluator: caching " +
                    std::to_string(reusable_cell_average_keys_.size()) +
                    " reusable cell-average derived expression(s) for this output pass");
    }
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
    const bool can_reuse_cell_average = usesCellAverageSweep(def);
    const std::string cache_key = can_reuse_cell_average ? cellAverageCacheKey(def) : std::string{};
    const bool use_cell_average_cache =
        can_reuse_cell_average &&
        reusable_cell_average_keys_.find(cache_key) != reusable_cell_average_keys_.end();

    switch (def.scope) {
    case DerivedResultScope::Cell:
        if (def.policy == DerivedResultPolicy::CellAverage && use_cell_average_cache) {
            auto& cache = cell_average_cache_[cache_key];
            if (cache.values.empty()) {
                cache.components = components;
                cache.values = computeCellAverageValues(system_, state_, def);
            }
            FE_THROW_IF(cache.components != components, InvalidArgumentException,
                        "DerivedResultEvaluator: cached cell-average component mismatch");
            copyCellAverageValues(cache.values, cache.components, data);
            return;
        }
        evaluateCellResult(system_, state_, def, data);
        return;
    case DerivedResultScope::Vertex:
        if (def.policy == DerivedResultPolicy::PointValue) {
            evaluateVertexPointValueResult(system_, state_, def, data);
            return;
        }
        if (def.policy == DerivedResultPolicy::PatchAverage) {
            if (use_cell_average_cache) {
                auto& cache = cell_average_cache_[cache_key];
                if (cache.values.empty()) {
                    cache.components = components;
                    cache.values = computeCellAverageValues(system_, state_, def);
                }
                FE_THROW_IF(cache.components != components, InvalidArgumentException,
                            "DerivedResultEvaluator: cached cell-average component mismatch");
                evaluateVertexPatchAverageFromCellValues(system_, cache.values, cache.components, data);
                return;
            }
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
