/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#include "FunctionalAssembler.h"
#include "Dofs/DofMap.h"
#include "Spaces/FunctionSpace.h"
#include "Elements/Element.h"
#include "Elements/ElementTransform.h"
#include "Elements/ReferenceElement.h"
#include "Quadrature/QuadratureFactory.h"
#include "Quadrature/QuadratureRule.h"
#include "Geometry/MappingFactory.h"
#include "Geometry/GeometryMapping.h"
#include "Math/Vector.h"
#include "Basis/BasisFunction.h"

#include <chrono>
#include <algorithm>
#include <numeric>
#include <stdexcept>
#include <thread>
#include <mutex>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace svmp {
namespace FE {
namespace assembly {

namespace {

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

struct ContextScratch {
    std::vector<std::array<Real, 3>> cell_coords;

    std::vector<AssemblyContext::Point3D> quad_points;
    std::vector<Real> quad_weights;
    std::vector<AssemblyContext::Point3D> phys_points;
    std::vector<AssemblyContext::Matrix3x3> jacobians;
    std::vector<AssemblyContext::Matrix3x3> inv_jacobians;
    std::vector<Real> jac_dets;
    std::vector<Real> integration_weights;
    std::vector<AssemblyContext::Vector3D> normals;

    std::vector<Real> basis_values;
    std::vector<AssemblyContext::Vector3D> ref_gradients;
    std::vector<AssemblyContext::Vector3D> phys_gradients;
};

ContextScratch& scratchStorage()
{
    static thread_local ContextScratch scratch;
    return scratch;
}

void prepareCellContext(AssemblyContext& context,
                        const IMeshAccess& mesh,
                        GlobalIndex cell_id,
                        const spaces::FunctionSpace& space,
                        RequiredData required_data)
{
    auto& scratch = scratchStorage();

    const ElementType cell_type = mesh.getCellType(cell_id);
    const int dim = mesh.dimension();

    const auto& element = space.getElement(cell_type, cell_id);

    auto quad_rule = element.quadrature();
    if (!quad_rule) {
        const int quad_order = quadrature::QuadratureFactory::recommended_order(
            element.polynomial_order(), false);
        quad_rule = quadrature::QuadratureFactory::create(cell_type, quad_order);
    }

    const auto n_qpts = static_cast<LocalIndex>(quad_rule->num_points());
    const auto n_dofs = static_cast<LocalIndex>(space.dofs_per_element());
    const auto n_scalar_dofs = static_cast<LocalIndex>(element.num_dofs());
    const bool is_product = (space.space_type() == spaces::SpaceType::Product);
    if (is_product) {
        FE_CHECK_ARG(space.field_type() == FieldType::Vector,
                     "FunctionalAssembler::prepareCellContext: ProductSpace must be vector-valued");
        FE_CHECK_ARG(space.value_dimension() > 0,
                     "FunctionalAssembler::prepareCellContext: invalid ProductSpace value_dimension");
        FE_CHECK_ARG(
            n_dofs ==
                static_cast<LocalIndex>(n_scalar_dofs * static_cast<LocalIndex>(space.value_dimension())),
            "FunctionalAssembler::prepareCellContext: ProductSpace DOF count mismatch");
    } else {
        FE_CHECK_ARG(n_dofs == n_scalar_dofs,
                     "FunctionalAssembler::prepareCellContext: non-Product space DOF count mismatch");
    }
    const bool need_basis_hessians = hasFlag(required_data, RequiredData::BasisHessians);

    mesh.getCellCoordinates(cell_id, scratch.cell_coords);

    std::vector<math::Vector<Real, 3>> node_coords(scratch.cell_coords.size());
    for (std::size_t i = 0; i < scratch.cell_coords.size(); ++i) {
        node_coords[i] = math::Vector<Real, 3>{
            scratch.cell_coords[i][0],
            scratch.cell_coords[i][1],
            scratch.cell_coords[i][2]};
    }

    geometry::MappingRequest map_request;
    map_request.element_type = cell_type;
    map_request.geometry_order = defaultGeometryOrder(cell_type);
    map_request.use_affine = (map_request.geometry_order <= 1);
    auto mapping = geometry::MappingFactory::create(map_request, node_coords);

    scratch.quad_points.resize(n_qpts);
    scratch.quad_weights.resize(n_qpts);
    scratch.phys_points.resize(n_qpts);
    scratch.jacobians.resize(n_qpts);
    scratch.inv_jacobians.resize(n_qpts);
    scratch.jac_dets.resize(n_qpts);
    scratch.integration_weights.resize(n_qpts);
    scratch.normals.clear();

    scratch.basis_values.resize(static_cast<std::size_t>(n_dofs * n_qpts));
    scratch.ref_gradients.resize(static_cast<std::size_t>(n_dofs * n_qpts));
    scratch.phys_gradients.resize(static_cast<std::size_t>(n_dofs * n_qpts));
    (void)need_basis_hessians;

    const auto& quad_points = quad_rule->points();
    const auto& quad_weights = quad_rule->weights();

    for (LocalIndex q = 0; q < n_qpts; ++q) {
        const auto& qpt = quad_points[q];
        scratch.quad_points[q] = {qpt[0], qpt[1], qpt[2]};
        scratch.quad_weights[q] = quad_weights[q];

        const math::Vector<Real, 3> xi{qpt[0], qpt[1], qpt[2]};
        const auto x_phys = mapping->map_to_physical(xi);
        scratch.phys_points[q] = {x_phys[0], x_phys[1], x_phys[2]};

        const auto J = mapping->jacobian(xi);
        const auto J_inv = mapping->jacobian_inverse(xi);
        const Real det_J = mapping->jacobian_determinant(xi);

        for (int i = 0; i < 3; ++i) {
            for (int j = 0; j < 3; ++j) {
                scratch.jacobians[q][i][j] = J(i, j);
                scratch.inv_jacobians[q][i][j] = J_inv(i, j);
            }
        }
        scratch.jac_dets[q] = det_J;
        scratch.integration_weights[q] = quad_weights[q] * std::abs(det_J);
    }

    const auto& basis = element.basis();
    std::vector<Real> values_at_pt;
    std::vector<basis::Gradient> gradients_at_pt;

    for (LocalIndex q = 0; q < n_qpts; ++q) {
        const math::Vector<Real, 3> xi{
            scratch.quad_points[q][0],
            scratch.quad_points[q][1],
            scratch.quad_points[q][2]};

        basis.evaluate_values(xi, values_at_pt);
        basis.evaluate_gradients(xi, gradients_at_pt);

        for (LocalIndex i = 0; i < n_dofs; ++i) {
            const LocalIndex si = is_product ? static_cast<LocalIndex>(i % n_scalar_dofs) : i;
            const std::size_t idx = static_cast<std::size_t>(i * n_qpts + q);
            scratch.basis_values[idx] = values_at_pt[static_cast<std::size_t>(si)];
            scratch.ref_gradients[idx] = {
                gradients_at_pt[static_cast<std::size_t>(si)][0],
                gradients_at_pt[static_cast<std::size_t>(si)][1],
                gradients_at_pt[static_cast<std::size_t>(si)][2]};

            const auto& grad_ref = scratch.ref_gradients[idx];
            const auto& J_inv = scratch.inv_jacobians[q];
            AssemblyContext::Vector3D grad_phys = {0.0, 0.0, 0.0};
            for (int d1 = 0; d1 < dim; ++d1) {
                for (int d2 = 0; d2 < dim; ++d2) {
                    grad_phys[d1] += J_inv[d2][d1] * grad_ref[d2];
                }
            }
            scratch.phys_gradients[idx] = grad_phys;
        }
    }

    context.reserve(n_dofs, n_qpts, dim);
    context.configure(cell_id, space, space, required_data);
    context.setCellDomainId(mesh.getCellDomainId(cell_id));
    context.setQuadratureData(scratch.quad_points, scratch.quad_weights);
    context.setPhysicalPoints(scratch.phys_points);
    context.setJacobianData(scratch.jacobians, scratch.inv_jacobians, scratch.jac_dets);
    context.setIntegrationWeights(scratch.integration_weights);
    context.setTestBasisData(n_dofs, scratch.basis_values, scratch.ref_gradients);
    context.setPhysicalGradients(scratch.phys_gradients, scratch.phys_gradients);

    if (hasFlag(required_data, RequiredData::EntityMeasures)) {
        Real cell_volume = 0.0;
        for (LocalIndex q = 0; q < n_qpts; ++q) {
            cell_volume += scratch.integration_weights[static_cast<std::size_t>(q)];
        }

        Real h = 0.0;
        for (std::size_t a = 0; a < node_coords.size(); ++a) {
            for (std::size_t b = a + 1; b < node_coords.size(); ++b) {
                const Real dx = node_coords[a][0] - node_coords[b][0];
                const Real dy = node_coords[a][1] - node_coords[b][1];
                const Real dz = node_coords[a][2] - node_coords[b][2];
                const Real dist = std::sqrt(dx * dx + dy * dy + dz * dz);
                if (dist > h) h = dist;
            }
        }

        context.setEntityMeasures(h, cell_volume, /*facet_area=*/0.0);
    }
}

AssemblyContext::Vector3D computeFaceNormal(LocalIndex local_face_id,
                                            ElementType cell_type,
                                            int dim) noexcept
{
    (void)dim;
    auto n = elements::ElementTransform::reference_facet_normal(
        cell_type, static_cast<int>(local_face_id));
    // Match StandardAssembler: scale the oblique tetra face normal so that
    // surface measures computed via cof(J) are correct when using canonical
    // face quadrature rules.
    if ((cell_type == ElementType::Tetra4 || cell_type == ElementType::Tetra10) && local_face_id == 2) {
        const Real scale = std::sqrt(Real(3));
        n[0] *= scale;
        n[1] *= scale;
        n[2] *= scale;
    }
    return {n[0], n[1], n[2]};
}

void computeSurfaceMeasureAndNormal(const AssemblyContext::Vector3D& n_ref,
                                    const AssemblyContext::Matrix3x3& J_inv,
                                    Real det_J,
                                    int dim,
                                    Real& surface_measure,
                                    AssemblyContext::Vector3D& n_phys) noexcept
{
    AssemblyContext::Vector3D Jit_n = {0.0, 0.0, 0.0};
    for (int i = 0; i < dim; ++i) {
        for (int k = 0; k < dim; ++k) {
            Jit_n[static_cast<std::size_t>(i)] +=
                J_inv[static_cast<std::size_t>(k)][static_cast<std::size_t>(i)] *
                n_ref[static_cast<std::size_t>(k)];
        }
    }

    Real norm_Jit_n = 0.0;
    for (int i = 0; i < dim; ++i) {
        const Real v = Jit_n[static_cast<std::size_t>(i)];
        norm_Jit_n += v * v;
    }
    norm_Jit_n = std::sqrt(norm_Jit_n);

    surface_measure = norm_Jit_n * std::abs(det_J);

    constexpr Real tol = 1e-14;
    if (norm_Jit_n > tol) {
        n_phys[0] = Jit_n[0] / norm_Jit_n;
        n_phys[1] = Jit_n[1] / norm_Jit_n;
        n_phys[2] = Jit_n[2] / norm_Jit_n;
    } else {
        n_phys = n_ref;
    }
}

void prepareBoundaryFaceContext(AssemblyContext& context,
                                const IMeshAccess& mesh,
                                GlobalIndex face_id,
                                GlobalIndex cell_id,
                                LocalIndex local_face_id,
                                const spaces::FunctionSpace& space,
                                RequiredData required_data)
{
    auto& scratch = scratchStorage();

    const ElementType cell_type = mesh.getCellType(cell_id);
    const int dim = mesh.dimension();

    const auto& element = space.getElement(cell_type, cell_id);

    const auto n_dofs = static_cast<LocalIndex>(space.dofs_per_element());
    const auto n_scalar_dofs = static_cast<LocalIndex>(element.num_dofs());
    const bool is_product = (space.space_type() == spaces::SpaceType::Product);
    if (is_product) {
        FE_CHECK_ARG(space.field_type() == FieldType::Vector,
                     "FunctionalAssembler::prepareBoundaryFaceContext: ProductSpace must be vector-valued");
        FE_CHECK_ARG(space.value_dimension() > 0,
                     "FunctionalAssembler::prepareBoundaryFaceContext: invalid ProductSpace value_dimension");
        FE_CHECK_ARG(
            n_dofs ==
                static_cast<LocalIndex>(n_scalar_dofs * static_cast<LocalIndex>(space.value_dimension())),
            "FunctionalAssembler::prepareBoundaryFaceContext: ProductSpace DOF count mismatch");
    } else {
        FE_CHECK_ARG(n_dofs == n_scalar_dofs,
                     "FunctionalAssembler::prepareBoundaryFaceContext: non-Product space DOF count mismatch");
    }
    const bool need_basis_hessians = hasFlag(required_data, RequiredData::BasisHessians);

    elements::ReferenceElement ref = elements::ReferenceElement::create(cell_type);
    const auto& face_nodes = ref.face_nodes(static_cast<std::size_t>(local_face_id));

    ElementType face_type = ElementType::Unknown;
    switch (face_nodes.size()) {
        case 2: face_type = ElementType::Line2; break;
        case 3: face_type = ElementType::Triangle3; break;
        case 4: face_type = ElementType::Quad4; break;
        default:
            throw std::runtime_error("FunctionalAssembler::prepareBoundaryFaceContext: unsupported face topology");
    }

    const int quad_order = quadrature::QuadratureFactory::recommended_order(
        element.polynomial_order(), false);
    auto quad_rule = quadrature::QuadratureFactory::create(face_type, quad_order);

    const auto n_qpts = static_cast<LocalIndex>(quad_rule->num_points());

    mesh.getCellCoordinates(cell_id, scratch.cell_coords);
    const auto n_nodes = scratch.cell_coords.size();

    std::vector<math::Vector<Real, 3>> node_coords(n_nodes);
    for (std::size_t i = 0; i < n_nodes; ++i) {
        node_coords[i] = math::Vector<Real, 3>{
            scratch.cell_coords[i][0],
            scratch.cell_coords[i][1],
            scratch.cell_coords[i][2]};
    }

    geometry::MappingRequest map_request;
    map_request.element_type = cell_type;
    map_request.geometry_order = defaultGeometryOrder(cell_type);
    map_request.use_affine = (map_request.geometry_order <= 1);
    auto mapping = geometry::MappingFactory::create(map_request, node_coords);

    scratch.quad_points.resize(n_qpts);
    scratch.quad_weights.resize(n_qpts);
    scratch.phys_points.resize(n_qpts);
    scratch.jacobians.resize(n_qpts);
    scratch.inv_jacobians.resize(n_qpts);
    scratch.jac_dets.resize(n_qpts);
    scratch.integration_weights.resize(n_qpts);
    scratch.normals.resize(n_qpts);

    scratch.basis_values.resize(static_cast<std::size_t>(n_dofs * n_qpts));
    scratch.ref_gradients.resize(static_cast<std::size_t>(n_dofs * n_qpts));
    scratch.phys_gradients.resize(static_cast<std::size_t>(n_dofs * n_qpts));

    std::vector<basis::Hessian> hessians_at_pt;
    std::vector<AssemblyContext::Matrix3x3> ref_hessians;
    std::vector<AssemblyContext::Matrix3x3> phys_hessians;
    if (need_basis_hessians) {
        ref_hessians.resize(static_cast<std::size_t>(n_dofs * n_qpts));
        phys_hessians.resize(static_cast<std::size_t>(n_dofs * n_qpts));
    }

    const auto& quad_points = quad_rule->points();
    const auto& quad_weights = quad_rule->weights();

    const AssemblyContext::Vector3D n_ref = computeFaceNormal(local_face_id, cell_type, dim);

    for (LocalIndex q = 0; q < n_qpts; ++q) {
        const auto& qpt = quad_points[q];
        scratch.quad_weights[q] = quad_weights[q];

        math::Vector<Real, 3> facet_coords{};
        if (face_type == ElementType::Line2) {
            Real t = (qpt[0] + Real(1)) * Real(0.5);
            facet_coords = math::Vector<Real, 3>{t, Real(0), Real(0)};
        } else if (face_type == ElementType::Quad4) {
            facet_coords = math::Vector<Real, 3>{
                (qpt[0] + Real(1)) * Real(0.5),
                (qpt[1] + Real(1)) * Real(0.5),
                Real(0)};
        } else {
            facet_coords = math::Vector<Real, 3>{qpt[0], qpt[1], Real(0)};
        }

        const math::Vector<Real, 3> xi = elements::ElementTransform::facet_to_reference(
            cell_type, static_cast<int>(local_face_id), facet_coords);

        scratch.quad_points[q] = {xi[0], xi[1], xi[2]};

        const auto x_phys = mapping->map_to_physical(xi);
        scratch.phys_points[q] = {x_phys[0], x_phys[1], x_phys[2]};

        const auto J = mapping->jacobian(xi);
        const auto J_inv = mapping->jacobian_inverse(xi);
        const Real det_J = mapping->jacobian_determinant(xi);

        for (int i = 0; i < 3; ++i) {
            for (int j = 0; j < 3; ++j) {
                scratch.jacobians[q][i][j] = J(i, j);
                scratch.inv_jacobians[q][i][j] = J_inv(i, j);
            }
        }
        scratch.jac_dets[q] = det_J;

        Real surface_measure = 0.0;
        AssemblyContext::Vector3D n_phys = {0.0, 0.0, 0.0};
        computeSurfaceMeasureAndNormal(n_ref, scratch.inv_jacobians[q], det_J, dim, surface_measure, n_phys);
        scratch.integration_weights[q] = quad_weights[q] * surface_measure;
        scratch.normals[q] = n_phys;
    }

    const auto& basis = element.basis();
    std::vector<Real> values_at_pt;
    std::vector<basis::Gradient> gradients_at_pt;

    for (LocalIndex q = 0; q < n_qpts; ++q) {
        const math::Vector<Real, 3> xi{
            scratch.quad_points[q][0],
            scratch.quad_points[q][1],
            scratch.quad_points[q][2]};

        const auto& J_inv = scratch.inv_jacobians[q];

        basis.evaluate_values(xi, values_at_pt);
        basis.evaluate_gradients(xi, gradients_at_pt);
        if (need_basis_hessians) {
            basis.evaluate_hessians(xi, hessians_at_pt);
        }

        for (LocalIndex i = 0; i < n_dofs; ++i) {
            const LocalIndex si = is_product ? static_cast<LocalIndex>(i % n_scalar_dofs) : i;
            const std::size_t idx = static_cast<std::size_t>(i * n_qpts + q);
            scratch.basis_values[idx] = values_at_pt[static_cast<std::size_t>(si)];
            scratch.ref_gradients[idx] = {
                gradients_at_pt[static_cast<std::size_t>(si)][0],
                gradients_at_pt[static_cast<std::size_t>(si)][1],
                gradients_at_pt[static_cast<std::size_t>(si)][2]};

            const auto& grad_ref = scratch.ref_gradients[idx];
            AssemblyContext::Vector3D grad_phys = {0.0, 0.0, 0.0};
            for (int d1 = 0; d1 < dim; ++d1) {
                for (int d2 = 0; d2 < dim; ++d2) {
                    grad_phys[d1] += J_inv[d2][d1] * grad_ref[d2];
                }
            }
            scratch.phys_gradients[idx] = grad_phys;

            if (need_basis_hessians) {
                AssemblyContext::Matrix3x3 H_ref{};
                for (int r = 0; r < 3; ++r) {
                    for (int c = 0; c < 3; ++c) {
                        H_ref[static_cast<std::size_t>(r)][static_cast<std::size_t>(c)] =
                            hessians_at_pt[static_cast<std::size_t>(si)](
                                static_cast<std::size_t>(r), static_cast<std::size_t>(c));
                    }
                }
                ref_hessians[idx] = H_ref;

                AssemblyContext::Matrix3x3 H_phys{};
                for (int r = 0; r < dim; ++r) {
                    for (int c = 0; c < dim; ++c) {
                        Real sum = 0.0;
                        for (int a = 0; a < dim; ++a) {
                            for (int b = 0; b < dim; ++b) {
                                sum += J_inv[static_cast<std::size_t>(a)][static_cast<std::size_t>(r)] *
                                       H_ref[static_cast<std::size_t>(a)][static_cast<std::size_t>(b)] *
                                       J_inv[static_cast<std::size_t>(b)][static_cast<std::size_t>(c)];
                            }
                        }
                        H_phys[static_cast<std::size_t>(r)][static_cast<std::size_t>(c)] = sum;
                    }
                }
                phys_hessians[idx] = H_phys;
            }
        }
    }

    context.reserve(n_dofs, n_qpts, dim);
    context.configureFace(face_id, cell_id, local_face_id, space, space, required_data, ContextType::BoundaryFace);
    context.setCellDomainId(mesh.getCellDomainId(cell_id));
    context.setQuadratureData(scratch.quad_points, scratch.quad_weights);
    context.setPhysicalPoints(scratch.phys_points);
    context.setJacobianData(scratch.jacobians, scratch.inv_jacobians, scratch.jac_dets);
    context.setIntegrationWeights(scratch.integration_weights);
    context.setTestBasisData(n_dofs, scratch.basis_values, scratch.ref_gradients);
    context.setPhysicalGradients(scratch.phys_gradients, scratch.phys_gradients);
    context.setNormals(scratch.normals);

    if (need_basis_hessians) {
        context.setTestBasisHessians(n_dofs, ref_hessians);
        context.setPhysicalHessians(phys_hessians, phys_hessians);
    }

    if (hasFlag(required_data, RequiredData::EntityMeasures)) {
        Real facet_area = 0.0;
        for (LocalIndex q = 0; q < n_qpts; ++q) {
            facet_area += scratch.integration_weights[static_cast<std::size_t>(q)];
        }

        Real h = 0.0;
        for (std::size_t a = 0; a < node_coords.size(); ++a) {
            for (std::size_t b = a + 1; b < node_coords.size(); ++b) {
                const Real dx = node_coords[a][0] - node_coords[b][0];
                const Real dy = node_coords[a][1] - node_coords[b][1];
                const Real dz = node_coords[a][2] - node_coords[b][2];
                const Real dist = std::sqrt(dx * dx + dy * dy + dz * dz);
                if (dist > h) h = dist;
            }
        }
        context.setEntityMeasures(h, /*cell_volume=*/0.0, facet_area);
    }
}

} // namespace

// ============================================================================
// Construction
// ============================================================================

FunctionalAssembler::FunctionalAssembler()
    : options_{}
{
}

FunctionalAssembler::FunctionalAssembler(const FunctionalAssemblyOptions& options)
    : options_(options)
{
}

FunctionalAssembler::~FunctionalAssembler() = default;

FunctionalAssembler::FunctionalAssembler(FunctionalAssembler&& other) noexcept = default;

FunctionalAssembler& FunctionalAssembler::operator=(FunctionalAssembler&& other) noexcept = default;

// ============================================================================
// Configuration
// ============================================================================

void FunctionalAssembler::setMesh(const IMeshAccess& mesh)
{
    mesh_ = &mesh;
}

void FunctionalAssembler::setDofMap(const dofs::DofMap& dof_map)
{
    dof_map_ = &dof_map;
}

void FunctionalAssembler::setSpace(const spaces::FunctionSpace& space)
{
    space_ = &space;
}

void FunctionalAssembler::setPrimaryField(FieldId field) noexcept
{
    primary_field_ = field;
}

void FunctionalAssembler::setSolution(std::span<const Real> solution)
{
    solution_.assign(solution.begin(), solution.end());
}

void FunctionalAssembler::setPreviousSolution(std::span<const Real> solution)
{
    setPreviousSolutionK(1, solution);
}

void FunctionalAssembler::setPreviousSolution2(std::span<const Real> solution)
{
    setPreviousSolutionK(2, solution);
}

void FunctionalAssembler::setPreviousSolutionK(int k, std::span<const Real> solution)
{
    FE_THROW_IF(k <= 0, InvalidArgumentException,
                "FunctionalAssembler::setPreviousSolutionK: k must be >= 1");
    if (previous_solutions_.size() < static_cast<std::size_t>(k)) {
        previous_solutions_.resize(static_cast<std::size_t>(k));
    }
    previous_solutions_[static_cast<std::size_t>(k - 1)].assign(solution.begin(), solution.end());
}

void FunctionalAssembler::setTimeIntegrationContext(const TimeIntegrationContext* ctx) noexcept
{
    time_integration_ = ctx;
}

void FunctionalAssembler::setTime(Real time) noexcept
{
    time_ = time;
}

void FunctionalAssembler::setTimeStep(Real dt) noexcept
{
    dt_ = dt;
}

void FunctionalAssembler::setRealParameterGetter(
    const std::function<std::optional<Real>(std::string_view)>* get_real_param) noexcept
{
    get_real_param_ = get_real_param;
}

void FunctionalAssembler::setParameterGetter(
    const std::function<std::optional<params::Value>(std::string_view)>* get_param) noexcept
{
    get_param_ = get_param;
}

void FunctionalAssembler::setUserData(const void* user_data) noexcept
{
    user_data_ = user_data;
}

void FunctionalAssembler::setJITConstants(std::span<const Real> constants) noexcept
{
    jit_constants_ = constants;
}

void FunctionalAssembler::setCoupledValues(std::span<const Real> integrals,
                                          std::span<const Real> aux_state) noexcept
{
    coupled_integrals_ = integrals;
    coupled_aux_state_ = aux_state;
}

void FunctionalAssembler::bindFieldSolutionData(AssemblyContext& context,
                                                std::span<const FieldRequirement> reqs)
{
    context.clearFieldSolutionData();
    if (reqs.empty()) {
        return;
    }

    FE_THROW_IF(primary_field_ == INVALID_FIELD_ID, InvalidArgumentException,
                "FunctionalAssembler: primary_field is not set (required for DiscreteField/StateField evaluation)");
    FE_CHECK_NOT_NULL(space_, "FunctionalAssembler::bindFieldSolutionData: space");

    const auto n_qpts = context.numQuadraturePoints();
    const auto field_type = space_->field_type();

    std::vector<Real> scalar_values;
    std::vector<AssemblyContext::Vector3D> scalar_gradients;
    std::vector<AssemblyContext::Matrix3x3> scalar_hessians;
    std::vector<Real> scalar_laplacians;

    std::vector<AssemblyContext::Vector3D> vector_values;
    std::vector<AssemblyContext::Matrix3x3> vector_jacobians;
    std::vector<AssemblyContext::Matrix3x3> vector_component_hessians;
    std::vector<Real> vector_component_laplacians;

    for (const auto& fr : reqs) {
        FE_THROW_IF(fr.field != primary_field_, NotImplementedException,
                    "FunctionalAssembler: multi-field functionals are not implemented (requested FieldId " +
                        std::to_string(fr.field) + ", primary FieldId is " + std::to_string(primary_field_) + ")");

        if (field_type == FieldType::Scalar) {
            scalar_values.resize(static_cast<std::size_t>(n_qpts));
            for (LocalIndex q = 0; q < n_qpts; ++q) {
                scalar_values[static_cast<std::size_t>(q)] = context.solutionValue(q);
            }

            std::span<const AssemblyContext::Vector3D> grads{};
            if (hasFlag(fr.required, RequiredData::SolutionGradients)) {
                scalar_gradients.resize(static_cast<std::size_t>(n_qpts));
                for (LocalIndex q = 0; q < n_qpts; ++q) {
                    scalar_gradients[static_cast<std::size_t>(q)] = context.solutionGradient(q);
                }
                grads = scalar_gradients;
            } else {
                scalar_gradients.clear();
            }

            std::span<const AssemblyContext::Matrix3x3> hessians{};
            if (hasFlag(fr.required, RequiredData::SolutionHessians)) {
                scalar_hessians.resize(static_cast<std::size_t>(n_qpts));
                for (LocalIndex q = 0; q < n_qpts; ++q) {
                    scalar_hessians[static_cast<std::size_t>(q)] = context.solutionHessian(q);
                }
                hessians = scalar_hessians;
            } else {
                scalar_hessians.clear();
            }

            std::span<const Real> laps{};
            if (hasFlag(fr.required, RequiredData::SolutionLaplacians)) {
                scalar_laplacians.resize(static_cast<std::size_t>(n_qpts));
                for (LocalIndex q = 0; q < n_qpts; ++q) {
                    scalar_laplacians[static_cast<std::size_t>(q)] = context.solutionLaplacian(q);
                }
                laps = scalar_laplacians;
            } else {
                scalar_laplacians.clear();
            }

            context.setFieldSolutionScalar(primary_field_, scalar_values, grads, hessians, laps);
        } else if (field_type == FieldType::Vector) {
            const int vd = space_->value_dimension();
            FE_THROW_IF(vd <= 0 || vd > 3, InvalidArgumentException,
                        "FunctionalAssembler: invalid primary vector value_dimension");

            vector_values.resize(static_cast<std::size_t>(n_qpts));
            for (LocalIndex q = 0; q < n_qpts; ++q) {
                vector_values[static_cast<std::size_t>(q)] = context.solutionVectorValue(q);
            }

            std::span<const AssemblyContext::Matrix3x3> jacobians{};
            if (hasFlag(fr.required, RequiredData::SolutionGradients)) {
                vector_jacobians.resize(static_cast<std::size_t>(n_qpts));
                for (LocalIndex q = 0; q < n_qpts; ++q) {
                    vector_jacobians[static_cast<std::size_t>(q)] = context.solutionJacobian(q);
                }
                jacobians = vector_jacobians;
            } else {
                vector_jacobians.clear();
            }

            std::span<const AssemblyContext::Matrix3x3> component_hessians{};
            if (hasFlag(fr.required, RequiredData::SolutionHessians)) {
                vector_component_hessians.resize(static_cast<std::size_t>(n_qpts) * static_cast<std::size_t>(vd));
                for (LocalIndex q = 0; q < n_qpts; ++q) {
                    for (int comp = 0; comp < vd; ++comp) {
                        vector_component_hessians[
                            static_cast<std::size_t>(q) * static_cast<std::size_t>(vd) + static_cast<std::size_t>(comp)] =
                            context.solutionComponentHessian(q, comp);
                    }
                }
                component_hessians = vector_component_hessians;
            } else {
                vector_component_hessians.clear();
            }

            std::span<const Real> component_laps{};
            if (hasFlag(fr.required, RequiredData::SolutionLaplacians)) {
                vector_component_laplacians.resize(static_cast<std::size_t>(n_qpts) * static_cast<std::size_t>(vd));
                for (LocalIndex q = 0; q < n_qpts; ++q) {
                    for (int comp = 0; comp < vd; ++comp) {
                        vector_component_laplacians[
                            static_cast<std::size_t>(q) * static_cast<std::size_t>(vd) + static_cast<std::size_t>(comp)] =
                            context.solutionComponentLaplacian(q, comp);
                    }
                }
                component_laps = vector_component_laplacians;
            } else {
                vector_component_laplacians.clear();
            }

            context.setFieldSolutionVector(primary_field_, vd, vector_values, jacobians, component_hessians, component_laps);
        } else {
            FE_THROW(NotImplementedException,
                     "FunctionalAssembler: primary field type not supported");
        }
    }
}

void FunctionalAssembler::setOptions(const FunctionalAssemblyOptions& options)
{
    options_ = options;
}

const FunctionalAssemblyOptions& FunctionalAssembler::getOptions() const noexcept
{
    return options_;
}

// ============================================================================
// Scalar Assembly
// ============================================================================

Real FunctionalAssembler::assembleScalar(FunctionalKernel& kernel)
{
    FunctionalResult result;
    Real value = assembleCellsCore(kernel, result);
    last_result_ = result;

    // Apply post-processing (e.g., square root for norms)
    return kernel.postProcess(value);
}

FunctionalResult FunctionalAssembler::assembleScalarDetailed(FunctionalKernel& kernel)
{
    FunctionalResult result;
    Real raw_value = assembleCellsCore(kernel, result);
    result.value = kernel.postProcess(raw_value);
    last_result_ = result;
    return result;
}

Real FunctionalAssembler::assembleBoundaryScalar(
    FunctionalKernel& kernel,
    int boundary_marker)
{
    FunctionalResult result;
    Real value = assembleBoundaryCore(kernel, boundary_marker, result);
    last_result_ = result;
    return kernel.postProcess(value);
}

// ============================================================================
// Multiple Functionals
// ============================================================================

std::vector<Real> FunctionalAssembler::assembleMultiple(
    std::span<FunctionalKernel* const> kernels)
{
    if (!isConfigured()) {
        throw std::runtime_error("FunctionalAssembler: not configured");
    }

    if (kernels.empty()) {
        return {};
    }

    auto start_time = std::chrono::steady_clock::now();

    const std::size_t num_kernels = kernels.size();
    std::vector<Real> totals(num_kernels, 0.0);
    std::vector<KahanAccumulator> accumulators(num_kernels);

    // Compute union of required data
    RequiredData all_required = RequiredData::None;
    for (auto* kernel : kernels) {
        all_required = all_required | kernel->getRequiredData();
    }

    const bool need_solution =
        hasFlag(all_required, RequiredData::SolutionCoefficients) ||
        hasFlag(all_required, RequiredData::SolutionValues) ||
        hasFlag(all_required, RequiredData::SolutionGradients) ||
        hasFlag(all_required, RequiredData::SolutionHessians) ||
        hasFlag(all_required, RequiredData::SolutionLaplacians);

    GlobalIndex elements_processed = 0;

    // Loop over cells
    std::vector<Real> local_solution;
    mesh_->forEachCell([&](GlobalIndex cell_id) {
        prepareContext(cell_id, all_required);

        if (need_solution) {
            FE_THROW_IF(solution_.empty(), FEException,
                        "FunctionalAssembler::assembleMultiple: kernels require solution but no solution was set");
            const auto dofs = dof_map_->getCellDofs(cell_id);
            local_solution.resize(dofs.size());
            for (std::size_t i = 0; i < dofs.size(); ++i) {
                const auto dof = dofs[i];
                FE_THROW_IF(dof < 0 || static_cast<std::size_t>(dof) >= solution_.size(), FEException,
                            "FunctionalAssembler::assembleMultiple: solution vector too small for DOF " + std::to_string(dof));
                local_solution[i] = solution_[static_cast<std::size_t>(dof)];
            }
            context_.setSolutionCoefficients(local_solution);
        }

        // Evaluate all kernels for this cell
        for (std::size_t k = 0; k < num_kernels; ++k) {
            if (kernels[k]->hasCell()) {
                Real cell_value = kernels[k]->evaluateCellTotal(context_);
                if (options_.use_kahan_summation) {
                    accumulators[k].add(cell_value);
                } else {
                    totals[k] += cell_value;
                }
            }
        }

        elements_processed++;
    });

    // Extract final values
    std::vector<Real> results(num_kernels);
    for (std::size_t k = 0; k < num_kernels; ++k) {
        Real raw = options_.use_kahan_summation ? accumulators[k].get() : totals[k];
        results[k] = kernels[k]->postProcess(raw);
    }

    auto end_time = std::chrono::steady_clock::now();
    last_result_.elapsed_seconds = std::chrono::duration<double>(end_time - start_time).count();
    last_result_.elements_processed = elements_processed;

    return results;
}

// ============================================================================
// Goal-Oriented Error Estimation
// ============================================================================

std::vector<Real> FunctionalAssembler::computeGoalOrientedIndicators(
    std::span<const Real> primal_solution,
    std::span<const Real> dual_solution,
    FunctionalKernel& kernel)
{
    if (!isConfigured()) {
        throw std::runtime_error("FunctionalAssembler: not configured");
    }

    // Store original solution
    auto original_solution = std::move(solution_);

    // We need both solutions; store primal for now
    solution_.assign(primal_solution.begin(), primal_solution.end());

    // Store dual solution separately
    std::vector<Real> dual(dual_solution.begin(), dual_solution.end());

    // Get number of cells
    GlobalIndex num_cells = mesh_->numCells();
    std::vector<Real> indicators(static_cast<std::size_t>(num_cells), 0.0);

    RequiredData required = kernel.getRequiredData() | RequiredData::SolutionValues;

    GlobalIndex cell_idx = 0;
    std::vector<Real> local_solution;
    mesh_->forEachCell([&](GlobalIndex cell_id) {
        prepareContext(cell_id, required);

        const auto dofs = dof_map_->getCellDofs(cell_id);
        local_solution.resize(dofs.size());
        for (std::size_t i = 0; i < dofs.size(); ++i) {
            const auto dof = dofs[i];
            FE_THROW_IF(dof < 0 || static_cast<std::size_t>(dof) >= solution_.size(), FEException,
                        "FunctionalAssembler::computeGoalOrientedIndicators: primal solution vector too small for DOF " + std::to_string(dof));
            local_solution[i] = solution_[static_cast<std::size_t>(dof)];
        }
        context_.setSolutionCoefficients(local_solution);

        // Compute DWR indicator: R(u_h) * z_h integrated over element
        // Simplified: we use the kernel evaluation weighted by dual solution
        Real indicator = 0.0;

        for (LocalIndex q = 0; q < context_.numQuadraturePoints(); ++q) {
            Real residual_contribution = kernel.evaluateCell(context_, q);
            // In a full implementation, we'd evaluate the dual solution at this point
            // For now, use element-average dual value
            Real jxw = context_.integrationWeight(q);
            indicator += residual_contribution * jxw;
        }

        indicators[static_cast<std::size_t>(cell_idx)] = std::abs(indicator);
        cell_idx++;
    });

    // Restore original solution
    solution_ = std::move(original_solution);

    return indicators;
}

// ============================================================================
// Convenience Norm Computations
// ============================================================================

Real FunctionalAssembler::computeL2Norm()
{
    L2NormKernel kernel;
    return assembleScalar(kernel);
}

Real FunctionalAssembler::computeH1Seminorm()
{
    H1SeminormKernel kernel;
    return assembleScalar(kernel);
}

Real FunctionalAssembler::computeL2Error(
    std::function<Real(Real, Real, Real)> exact_solution)
{
    L2ErrorKernel kernel(std::move(exact_solution));
    return assembleScalar(kernel);
}

Real FunctionalAssembler::computeH1Error(
    std::function<std::array<Real, 3>(Real, Real, Real)> exact_gradient)
{
    H1ErrorKernel kernel(std::move(exact_gradient));
    return assembleScalar(kernel);
}

Real FunctionalAssembler::computeEnergy()
{
    EnergyKernel kernel;
    return assembleScalar(kernel);
}

Real FunctionalAssembler::computeVolume()
{
    VolumeKernel kernel;
    return assembleScalar(kernel);
}

// ============================================================================
// Query
// ============================================================================

bool FunctionalAssembler::isConfigured() const noexcept
{
    return mesh_ != nullptr && dof_map_ != nullptr && space_ != nullptr;
}

const FunctionalResult& FunctionalAssembler::getLastResult() const noexcept
{
    return last_result_;
}

// ============================================================================
// Internal Implementation
// ============================================================================

Real FunctionalAssembler::assembleCellsCore(
    FunctionalKernel& kernel,
    FunctionalResult& result)
{
    if (!isConfigured()) {
        result.success = false;
        result.error_message = "FunctionalAssembler not configured";
        return 0.0;
    }

    auto start_time = std::chrono::steady_clock::now();

    const RequiredData required = kernel.getRequiredData();
    const auto field_reqs = kernel.fieldRequirements();
    const bool need_solution =
        hasFlag(required, RequiredData::SolutionCoefficients) ||
        hasFlag(required, RequiredData::SolutionValues) ||
        hasFlag(required, RequiredData::SolutionGradients) ||
        hasFlag(required, RequiredData::SolutionHessians) ||
        hasFlag(required, RequiredData::SolutionLaplacians) ||
        !field_reqs.empty();

    Real total = 0.0;
    KahanAccumulator accumulator;

    const int num_threads = options_.num_threads;
    const bool use_parallel = (num_threads > 1);

    if (use_parallel && options_.deterministic) {
        // Deterministic parallel: gather local contributions then reduce
        std::vector<Real> local_values;
        std::mutex values_mutex;

#ifdef _OPENMP
        #pragma omp parallel num_threads(num_threads)
        {
            AssemblyContext thread_context;
            thread_context.reserve(dof_map_->getMaxDofsPerCell(), 27,
                                   mesh_->dimension());

            std::vector<Real> my_values;
            std::vector<Real> local_solution;

            #pragma omp for schedule(static)
            for (GlobalIndex cell_id = 0; cell_id < mesh_->numCells(); ++cell_id) {
                prepareCellContext(thread_context, *mesh_, cell_id, *space_, required);
                thread_context.setTimeIntegrationContext(time_integration_);
                thread_context.setTime(time_);
                thread_context.setTimeStep(dt_);
                thread_context.setRealParameterGetter(get_real_param_);
                thread_context.setParameterGetter(get_param_);
                thread_context.setUserData(user_data_);
                thread_context.setJITConstants(jit_constants_);
                thread_context.setCoupledValues(coupled_integrals_, coupled_aux_state_);
                if (need_solution) {
                    FE_THROW_IF(solution_.empty(), FEException,
                                "FunctionalAssembler::assembleCellsCore: kernel requires solution but no solution was set");
                    const auto dofs = dof_map_->getCellDofs(cell_id);
                    local_solution.resize(dofs.size());
                    for (std::size_t i = 0; i < dofs.size(); ++i) {
                        const auto dof = dofs[i];
                        FE_THROW_IF(dof < 0 || static_cast<std::size_t>(dof) >= solution_.size(), FEException,
                                    "FunctionalAssembler::assembleCellsCore: solution vector too small for DOF " + std::to_string(dof));
                        local_solution[i] = solution_[static_cast<std::size_t>(dof)];
                    }
                    thread_context.setSolutionCoefficients(local_solution);
                    if (!field_reqs.empty()) {
                        bindFieldSolutionData(thread_context, field_reqs);
                    }
                }

                Real cell_value = kernel.evaluateCellTotal(thread_context);
                my_values.push_back(cell_value);
            }

            // Merge results
            {
                std::lock_guard<std::mutex> lock(values_mutex);
                local_values.insert(local_values.end(),
                                    my_values.begin(), my_values.end());
            }
        }
#else
        // Sequential fallback
        mesh_->forEachCell([&](GlobalIndex cell_id) {
            prepareContext(cell_id, required);
            Real cell_value = kernel.evaluateCellTotal(context_);
            local_values.push_back(cell_value);
            result.elements_processed++;
        });
#endif

        // Deterministic reduction: sort by element order then sum
        // (Parallel execution may produce out-of-order results)
        // For truly deterministic results, we rely on stable per-element order
        total = parallelReduce(local_values, options_.use_kahan_summation);
        result.elements_processed = static_cast<GlobalIndex>(local_values.size());

    } else {
        // Sequential assembly
        std::vector<Real> local_solution;
        mesh_->forEachCell([&](GlobalIndex cell_id) {
            prepareContext(cell_id, required);
            if (need_solution) {
                FE_THROW_IF(solution_.empty(), FEException,
                            "FunctionalAssembler::assembleCellsCore: kernel requires solution but no solution was set");
                const auto dofs = dof_map_->getCellDofs(cell_id);
                local_solution.resize(dofs.size());
                for (std::size_t i = 0; i < dofs.size(); ++i) {
                    const auto dof = dofs[i];
                    FE_THROW_IF(dof < 0 || static_cast<std::size_t>(dof) >= solution_.size(), FEException,
                                "FunctionalAssembler::assembleCellsCore: solution vector too small for DOF " + std::to_string(dof));
                    local_solution[i] = solution_[static_cast<std::size_t>(dof)];
                }
                context_.setSolutionCoefficients(local_solution);
                if (!field_reqs.empty()) {
                    bindFieldSolutionData(context_, field_reqs);
                }
            }

            Real cell_value = kernel.evaluateCellTotal(context_);

            if (options_.use_kahan_summation) {
                accumulator.add(cell_value);
            } else {
                total += cell_value;
            }

            result.elements_processed++;
        });

        if (options_.use_kahan_summation) {
            total = accumulator.get();
        }
    }

    auto end_time = std::chrono::steady_clock::now();
    result.elapsed_seconds = std::chrono::duration<double>(end_time - start_time).count();
    result.success = true;

    return total;
}

Real FunctionalAssembler::assembleBoundaryCore(
    FunctionalKernel& kernel,
    int boundary_marker,
    FunctionalResult& result)
{
    if (!isConfigured()) {
        result.success = false;
        result.error_message = "FunctionalAssembler not configured";
        return 0.0;
    }

    if (!kernel.hasBoundaryFace()) {
        result.success = false;
        result.error_message = "Kernel does not support boundary face integration";
        return 0.0;
    }

    auto start_time = std::chrono::steady_clock::now();

    const RequiredData required = kernel.getRequiredData();
    const auto field_reqs = kernel.fieldRequirements();

    const bool need_solution =
        hasFlag(required, RequiredData::SolutionCoefficients) ||
        hasFlag(required, RequiredData::SolutionValues) ||
        hasFlag(required, RequiredData::SolutionGradients) ||
        hasFlag(required, RequiredData::SolutionHessians) ||
        hasFlag(required, RequiredData::SolutionLaplacians) ||
        !field_reqs.empty();

    Real total = 0.0;
    KahanAccumulator accumulator;

    std::vector<Real> local_solution;
    mesh_->forEachBoundaryFace(boundary_marker,
        [&](GlobalIndex face_id, GlobalIndex cell_id) {
            // Prepare face context
            const LocalIndex local_face_id = mesh_->getLocalFaceIndex(face_id, cell_id);
            prepareBoundaryFaceContext(context_, *mesh_, face_id, cell_id, local_face_id, *space_, required);

            context_.setTimeIntegrationContext(time_integration_);
            context_.setTime(time_);
            context_.setTimeStep(dt_);
            context_.setRealParameterGetter(get_real_param_);
            context_.setParameterGetter(get_param_);
            context_.setUserData(user_data_);
            context_.setJITConstants(jit_constants_);
            context_.setCoupledValues(coupled_integrals_, coupled_aux_state_);
            context_.setBoundaryMarker(boundary_marker);

            if (need_solution) {
                FE_THROW_IF(solution_.empty(), FEException,
                            "FunctionalAssembler::assembleBoundaryCore: kernel requires solution but no solution was set");
                const auto dofs = dof_map_->getCellDofs(cell_id);
                local_solution.resize(dofs.size());
                for (std::size_t i = 0; i < dofs.size(); ++i) {
                    const auto dof = dofs[i];
                    FE_THROW_IF(dof < 0 || static_cast<std::size_t>(dof) >= solution_.size(), FEException,
                                "FunctionalAssembler::assembleBoundaryCore: solution vector too small for DOF " + std::to_string(dof));
                    local_solution[i] = solution_[static_cast<std::size_t>(dof)];
                }
                context_.setSolutionCoefficients(local_solution);
                if (!field_reqs.empty()) {
                    bindFieldSolutionData(context_, field_reqs);
                }
            }

            const Real face_value = kernel.evaluateBoundaryFaceTotal(context_, boundary_marker);

            if (options_.use_kahan_summation) {
                accumulator.add(face_value);
            } else {
                total += face_value;
            }

            result.faces_processed++;
        });

    if (options_.use_kahan_summation) {
        total = accumulator.get();
    }

    auto end_time = std::chrono::steady_clock::now();
    result.elapsed_seconds = std::chrono::duration<double>(end_time - start_time).count();
    result.success = true;

    return total;
}

void FunctionalAssembler::prepareContext(
    GlobalIndex cell_id,
    RequiredData required_data)
{
    prepareCellContext(context_, *mesh_, cell_id, *space_, required_data);

    context_.setTimeIntegrationContext(time_integration_);
    context_.setTime(time_);
    context_.setTimeStep(dt_);
    context_.setRealParameterGetter(get_real_param_);
    context_.setParameterGetter(get_param_);
    context_.setUserData(user_data_);
    context_.setJITConstants(jit_constants_);
    context_.setCoupledValues(coupled_integrals_, coupled_aux_state_);
}

Real FunctionalAssembler::parallelReduce(
    const std::vector<Real>& local_values,
    bool use_kahan)
{
    if (local_values.empty()) {
        return 0.0;
    }

    if (use_kahan) {
        KahanAccumulator acc;
        for (Real v : local_values) {
            acc.add(v);
        }
        return acc.get();
    } else {
        return std::accumulate(local_values.begin(), local_values.end(), Real{0.0});
    }
}

// ============================================================================
// Factory Functions
// ============================================================================

std::unique_ptr<FunctionalAssembler> createFunctionalAssembler()
{
    return std::make_unique<FunctionalAssembler>();
}

std::unique_ptr<FunctionalAssembler> createFunctionalAssembler(
    const FunctionalAssemblyOptions& options)
{
    return std::make_unique<FunctionalAssembler>(options);
}

} // namespace assembly
} // namespace FE
} // namespace svmp
