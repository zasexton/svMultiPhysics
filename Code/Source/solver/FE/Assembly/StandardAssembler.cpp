/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#include "StandardAssembler.h"
#include "Dofs/DofMap.h"
#include "Dofs/DofHandler.h"
#include "Constraints/AffineConstraints.h"
#include "Constraints/ConstraintDistributor.h"
#include "Sparsity/SparsityPattern.h"
#include "Spaces/FunctionSpace.h"
#include "Elements/Element.h"
#include "Elements/ElementTransform.h"
#include "Elements/ReferenceElement.h"
#include "Quadrature/QuadratureFactory.h"
#include "Quadrature/QuadratureRule.h"
#include "Geometry/MappingFactory.h"
#include "Geometry/GeometryMapping.h"
#include "Basis/BasisFunction.h"
#include "Math/Vector.h"
#include "Math/Matrix.h"

#include <chrono>
#include <stdexcept>
#include <cmath>

namespace svmp {
namespace FE {
namespace assembly {

// ============================================================================
// Construction
// ============================================================================

StandardAssembler::StandardAssembler() = default;

StandardAssembler::StandardAssembler(const AssemblyOptions& options)
    : options_(options)
{
}

StandardAssembler::~StandardAssembler() = default;

StandardAssembler::StandardAssembler(StandardAssembler&& other) noexcept = default;

StandardAssembler& StandardAssembler::operator=(StandardAssembler&& other) noexcept = default;

// ============================================================================
// Configuration
// ============================================================================

void StandardAssembler::setDofMap(const dofs::DofMap& dof_map)
{
    dof_map_ = &dof_map;
}

void StandardAssembler::setDofHandler(const dofs::DofHandler& dof_handler)
{
    dof_handler_ = &dof_handler;
    dof_map_ = &dof_handler.getDofMap();
}

void StandardAssembler::setConstraints(const constraints::AffineConstraints* constraints)
{
    constraints_ = constraints;

    if (constraints_ && constraints_->isClosed()) {
        constraint_distributor_ = std::make_unique<constraints::ConstraintDistributor>(*constraints_);
    } else {
        constraint_distributor_.reset();
    }
}

void StandardAssembler::setSparsityPattern(const sparsity::SparsityPattern* sparsity)
{
    sparsity_ = sparsity;
}

void StandardAssembler::setOptions(const AssemblyOptions& options)
{
    options_ = options;
}

const AssemblyOptions& StandardAssembler::getOptions() const noexcept
{
    return options_;
}

bool StandardAssembler::isConfigured() const noexcept
{
    return dof_map_ != nullptr;
}

// ============================================================================
// Lifecycle
// ============================================================================

void StandardAssembler::initialize()
{
    if (!isConfigured()) {
        throw std::runtime_error("StandardAssembler::initialize: assembler not configured");
    }

    // Reserve working storage based on DOF map
    const auto max_dofs = dof_map_->getMaxDofsPerCell();
    const auto max_dofs_size = static_cast<std::size_t>(max_dofs);

    row_dofs_.reserve(max_dofs_size);
    col_dofs_.reserve(max_dofs_size);
    scratch_rows_.reserve(max_dofs_size);
    scratch_cols_.reserve(max_dofs_size);
    scratch_matrix_.reserve(max_dofs_size * max_dofs_size);
    scratch_vector_.reserve(max_dofs_size);

    // Reserve context storage (estimate quadrature points)
    const LocalIndex est_qpts = 27;  // Typical for 3D Q2
    context_.reserve(max_dofs, est_qpts, 3);

    initialized_ = true;
}

void StandardAssembler::finalize(GlobalSystemView* matrix_view, GlobalSystemView* vector_view)
{
    // End assembly phase and trigger finalization
    if (matrix_view) {
        matrix_view->endAssemblyPhase();
        matrix_view->finalizeAssembly();
    }

    if (vector_view && vector_view != matrix_view) {
        vector_view->endAssemblyPhase();
        vector_view->finalizeAssembly();
    }
}

void StandardAssembler::reset()
{
    context_.clear();
    row_dofs_.clear();
    col_dofs_.clear();
    initialized_ = false;
}

// ============================================================================
// Matrix Assembly
// ============================================================================

AssemblyResult StandardAssembler::assembleMatrix(
    const IMeshAccess& mesh,
    const spaces::FunctionSpace& test_space,
    const spaces::FunctionSpace& trial_space,
    AssemblyKernel& kernel,
    GlobalSystemView& matrix_view)
{
    return assembleCellsCore(mesh, test_space, trial_space, kernel,
                             &matrix_view, nullptr, true, false);
}

// ============================================================================
// Vector Assembly
// ============================================================================

AssemblyResult StandardAssembler::assembleVector(
    const IMeshAccess& mesh,
    const spaces::FunctionSpace& space,
    AssemblyKernel& kernel,
    GlobalSystemView& vector_view)
{
    return assembleCellsCore(mesh, space, space, kernel,
                             nullptr, &vector_view, false, true);
}

// ============================================================================
// Combined Assembly
// ============================================================================

AssemblyResult StandardAssembler::assembleBoth(
    const IMeshAccess& mesh,
    const spaces::FunctionSpace& test_space,
    const spaces::FunctionSpace& trial_space,
    AssemblyKernel& kernel,
    GlobalSystemView& matrix_view,
    GlobalSystemView& vector_view)
{
    return assembleCellsCore(mesh, test_space, trial_space, kernel,
                             &matrix_view, &vector_view, true, true);
}

// ============================================================================
// Face Assembly
// ============================================================================

AssemblyResult StandardAssembler::assembleBoundaryFaces(
    const IMeshAccess& mesh,
    int boundary_marker,
    const spaces::FunctionSpace& space,
    AssemblyKernel& kernel,
    GlobalSystemView* matrix_view,
    GlobalSystemView* vector_view)
{
    AssemblyResult result;
    auto start_time = std::chrono::steady_clock::now();

    if (!initialized_) {
        initialize();
    }

    if (!kernel.hasBoundaryFace()) {
        return result;  // Nothing to do
    }

    // Begin assembly phase
    if (matrix_view) matrix_view->beginAssemblyPhase();
    if (vector_view && vector_view != matrix_view) {
        vector_view->beginAssemblyPhase();
    }

    const auto required_data = kernel.getRequiredData();

    // Iterate over boundary faces with given marker
    mesh.forEachBoundaryFace(boundary_marker,
        [&](GlobalIndex face_id, GlobalIndex cell_id) {
            // Get cell DOFs
            auto cell_dofs = dof_map_->getCellDofs(cell_id);
            row_dofs_.assign(cell_dofs.begin(), cell_dofs.end());
            col_dofs_.assign(cell_dofs.begin(), cell_dofs.end());

            // Prepare context for face
            LocalIndex local_face_id = mesh.getLocalFaceIndex(face_id, cell_id);
            prepareContextFace(context_, mesh, face_id, cell_id, local_face_id, space,
                               required_data, ContextType::BoundaryFace);
            context_.setBoundaryMarker(boundary_marker);

            // Compute local contributions
            kernel_output_.clear();
            kernel.computeBoundaryFace(context_, boundary_marker, kernel_output_);

            // Insert into global system
            if (options_.use_constraints && constraint_distributor_) {
                insertLocalConstrained(kernel_output_, row_dofs_, col_dofs_,
                                       matrix_view, vector_view);
            } else {
                insertLocal(kernel_output_, row_dofs_, col_dofs_,
                            matrix_view, vector_view);
            }

            result.boundary_faces_assembled++;
        });

    auto end_time = std::chrono::steady_clock::now();
    result.elapsed_time_seconds = std::chrono::duration<double>(end_time - start_time).count();

    return result;
}

AssemblyResult StandardAssembler::assembleInteriorFaces(
    const IMeshAccess& mesh,
    const spaces::FunctionSpace& test_space,
    const spaces::FunctionSpace& trial_space,
    AssemblyKernel& kernel,
    GlobalSystemView& matrix_view,
    GlobalSystemView* vector_view)
{
    AssemblyResult result;
    auto start_time = std::chrono::steady_clock::now();

    if (!initialized_) {
        initialize();
    }

    if (!kernel.hasInteriorFace()) {
        return result;
    }

    matrix_view.beginAssemblyPhase();
    if (vector_view && vector_view != &matrix_view) {
        vector_view->beginAssemblyPhase();
    }

    const auto required_data = kernel.getRequiredData();

    // Create second context for the "plus" side
    AssemblyContext context_plus;
    context_plus.reserve(dof_map_->getMaxDofsPerCell(), 27, mesh.dimension());

    // Kernel outputs for DG face terms
    KernelOutput output_minus, output_plus, coupling_mp, coupling_pm;

    // Scratch for DOFs
    std::vector<GlobalIndex> minus_dofs, plus_dofs;

    mesh.forEachInteriorFace(
        [&](GlobalIndex face_id, GlobalIndex cell_minus, GlobalIndex cell_plus) {
            // Get DOFs for both cells
            auto minus_cell_dofs = dof_map_->getCellDofs(cell_minus);
            auto plus_cell_dofs = dof_map_->getCellDofs(cell_plus);

            minus_dofs.assign(minus_cell_dofs.begin(), minus_cell_dofs.end());
            plus_dofs.assign(plus_cell_dofs.begin(), plus_cell_dofs.end());

            // Prepare contexts for both sides
            LocalIndex local_face_minus = mesh.getLocalFaceIndex(face_id, cell_minus);
            LocalIndex local_face_plus = mesh.getLocalFaceIndex(face_id, cell_plus);

            prepareContextFace(context_, mesh, face_id, cell_minus, local_face_minus, test_space,
                               required_data, ContextType::InteriorFace);

            prepareContextFace(context_plus, mesh, face_id, cell_plus, local_face_plus, test_space,
                               required_data, ContextType::InteriorFace);

            // Compute DG face contributions
            output_minus.clear();
            output_plus.clear();
            coupling_mp.clear();
            coupling_pm.clear();

            kernel.computeInteriorFace(context_, context_plus,
                                       output_minus, output_plus,
                                       coupling_mp, coupling_pm);

            // Insert contributions (4 blocks for DG)
            // Self-coupling: minus-minus
            if (output_minus.has_matrix || output_minus.has_vector) {
                insertLocal(output_minus, minus_dofs, minus_dofs, &matrix_view, vector_view);
            }

            // Self-coupling: plus-plus
            if (output_plus.has_matrix || output_plus.has_vector) {
                insertLocal(output_plus, plus_dofs, plus_dofs, &matrix_view, vector_view);
            }

            // Cross-coupling: minus-plus (minus rows, plus cols)
            if (coupling_mp.has_matrix) {
                matrix_view.addMatrixEntries(minus_dofs, plus_dofs,
                                             coupling_mp.local_matrix);
            }

            // Cross-coupling: plus-minus (plus rows, minus cols)
            if (coupling_pm.has_matrix) {
                matrix_view.addMatrixEntries(plus_dofs, minus_dofs,
                                             coupling_pm.local_matrix);
            }

            result.interior_faces_assembled++;
        });

    auto end_time = std::chrono::steady_clock::now();
    result.elapsed_time_seconds = std::chrono::duration<double>(end_time - start_time).count();

    return result;
}

// ============================================================================
// Internal Implementation
// ============================================================================

AssemblyResult StandardAssembler::assembleCellsCore(
    const IMeshAccess& mesh,
    const spaces::FunctionSpace& test_space,
    const spaces::FunctionSpace& trial_space,
    AssemblyKernel& kernel,
    GlobalSystemView* matrix_view,
    GlobalSystemView* vector_view,
    bool assemble_matrix,
    bool assemble_vector)
{
    AssemblyResult result;
    auto start_time = std::chrono::steady_clock::now();

    if (!initialized_) {
        initialize();
    }

    // Begin assembly phase
    if (matrix_view && assemble_matrix) {
        matrix_view->beginAssemblyPhase();
    }
    if (vector_view && assemble_vector && vector_view != matrix_view) {
        vector_view->beginAssemblyPhase();
    }

    const bool is_rectangular = (&test_space != &trial_space);
    const auto required_data = kernel.getRequiredData();

    // Iterate over cells
    mesh.forEachCell([&](GlobalIndex cell_id) {
        // Get element DOFs
        auto test_dofs = dof_map_->getCellDofs(cell_id);
        row_dofs_.assign(test_dofs.begin(), test_dofs.end());

        if (is_rectangular) {
            // For rectangular, trial DOFs may differ
            // This would require a separate DOF map for trial space
            // For now, assume same DOF map
            col_dofs_.assign(test_dofs.begin(), test_dofs.end());
        } else {
            col_dofs_.assign(test_dofs.begin(), test_dofs.end());
        }

        // Prepare assembly context
        prepareContext(context_, mesh, cell_id, test_space, trial_space, required_data);

        // Compute local matrix/vector via kernel
        kernel_output_.clear();
        kernel.computeCell(context_, kernel_output_);

        // Insert into global system
        if (options_.use_constraints && constraint_distributor_) {
            insertLocalConstrained(kernel_output_, row_dofs_, col_dofs_,
                                   assemble_matrix ? matrix_view : nullptr,
                                   assemble_vector ? vector_view : nullptr);
        } else {
            insertLocal(kernel_output_, row_dofs_, col_dofs_,
                        assemble_matrix ? matrix_view : nullptr,
                        assemble_vector ? vector_view : nullptr);
        }

        result.elements_assembled++;

        if (kernel_output_.has_matrix) {
            result.matrix_entries_inserted +=
                static_cast<GlobalIndex>(row_dofs_.size() * col_dofs_.size());
        }
        if (kernel_output_.has_vector) {
            result.vector_entries_inserted +=
                static_cast<GlobalIndex>(row_dofs_.size());
        }
    });

    auto end_time = std::chrono::steady_clock::now();
    result.elapsed_time_seconds = std::chrono::duration<double>(end_time - start_time).count();

    return result;
}

void StandardAssembler::prepareContext(
    AssemblyContext& context,
    const IMeshAccess& mesh,
    GlobalIndex cell_id,
    const spaces::FunctionSpace& test_space,
    const spaces::FunctionSpace& trial_space,
    RequiredData required_data)
{
    // 1. Get element type from mesh
    const ElementType cell_type = mesh.getCellType(cell_id);
    const int dim = mesh.dimension();

    // 2. Get elements for test and trial spaces
    const auto& test_element = getElement(test_space, cell_id, cell_type);
    const auto& trial_element = getElement(trial_space, cell_id, cell_type);

    // 3. Get quadrature rule from the element
    auto quad_rule = test_element.quadrature();
    if (!quad_rule) {
        // Fall back to factory-created quadrature if element doesn't provide one
        const int quad_order = quadrature::QuadratureFactory::recommended_order(
            test_element.polynomial_order(), false);
        quad_rule = quadrature::QuadratureFactory::create(cell_type, quad_order);
    }

    const auto n_qpts = static_cast<LocalIndex>(quad_rule->num_points());
    const auto n_test_dofs = static_cast<LocalIndex>(test_element.num_dofs());
    const auto n_trial_dofs = static_cast<LocalIndex>(trial_element.num_dofs());

    // 4. Get cell node coordinates from mesh
    mesh.getCellCoordinates(cell_id, cell_coords_);
    const auto n_nodes = cell_coords_.size();

    // Convert to math::Vector format for geometry mapping
    std::vector<math::Vector<Real, 3>> node_coords(n_nodes);
    for (std::size_t i = 0; i < n_nodes; ++i) {
        node_coords[i] = math::Vector<Real, 3>{
            cell_coords_[i][0], cell_coords_[i][1], cell_coords_[i][2]};
    }

    // 5. Create geometry mapping
    geometry::MappingRequest map_request;
    map_request.element_type = cell_type;
    map_request.geometry_order = 1;  // Linear mapping by default
    map_request.use_affine = true;

    auto mapping = geometry::MappingFactory::create(map_request, node_coords);

    // 6. Resize scratch storage
    scratch_quad_points_.resize(n_qpts);
    scratch_quad_weights_.resize(n_qpts);
    scratch_phys_points_.resize(n_qpts);
    scratch_jacobians_.resize(n_qpts);
    scratch_inv_jacobians_.resize(n_qpts);
    scratch_jac_dets_.resize(n_qpts);
    scratch_integration_weights_.resize(n_qpts);

    const auto test_basis_size = static_cast<std::size_t>(n_test_dofs * n_qpts);
    const auto trial_basis_size = static_cast<std::size_t>(n_trial_dofs * n_qpts);
    scratch_basis_values_.resize(test_basis_size);
    scratch_ref_gradients_.resize(test_basis_size);
    scratch_phys_gradients_.resize(test_basis_size);

    // Storage for trial if different from test
    std::vector<Real> trial_basis_values;
    std::vector<AssemblyContext::Vector3D> trial_ref_gradients;
    std::vector<AssemblyContext::Vector3D> trial_phys_gradients;

    if (&test_space != &trial_space) {
        trial_basis_values.resize(trial_basis_size);
        trial_ref_gradients.resize(trial_basis_size);
        trial_phys_gradients.resize(trial_basis_size);
    }

    // 7. Copy quadrature data and compute physical points and Jacobians
    const auto& quad_points = quad_rule->points();
    const auto& quad_weights = quad_rule->weights();

    for (LocalIndex q = 0; q < n_qpts; ++q) {
        // Copy reference quadrature point
        const auto& qpt = quad_points[q];
        scratch_quad_points_[q] = {qpt[0], qpt[1], qpt[2]};
        scratch_quad_weights_[q] = quad_weights[q];

        // Map to physical space
        const math::Vector<Real, 3> xi{qpt[0], qpt[1], qpt[2]};
        const auto x_phys = mapping->map_to_physical(xi);
        scratch_phys_points_[q] = {x_phys[0], x_phys[1], x_phys[2]};

        // Compute Jacobian
        const auto J = mapping->jacobian(xi);
        const auto J_inv = mapping->jacobian_inverse(xi);
        const Real det_J = mapping->jacobian_determinant(xi);

        // Store as arrays
        for (int i = 0; i < 3; ++i) {
            for (int j = 0; j < 3; ++j) {
                scratch_jacobians_[q][i][j] = J(i, j);
                scratch_inv_jacobians_[q][i][j] = J_inv(i, j);
            }
        }
        scratch_jac_dets_[q] = det_J;

        // Integration weight = quadrature weight * |det(J)|
        scratch_integration_weights_[q] = quad_weights[q] * std::abs(det_J);
    }

    // 8. Evaluate basis functions at quadrature points
    const auto& test_basis = test_element.basis();
    std::vector<Real> values_at_pt;
    std::vector<basis::Gradient> gradients_at_pt;

    for (LocalIndex q = 0; q < n_qpts; ++q) {
        const math::Vector<Real, 3> xi{
            scratch_quad_points_[q][0],
            scratch_quad_points_[q][1],
            scratch_quad_points_[q][2]};

        // Evaluate test basis values and gradients
        test_basis.evaluate_values(xi, values_at_pt);
        test_basis.evaluate_gradients(xi, gradients_at_pt);

        for (LocalIndex i = 0; i < n_test_dofs; ++i) {
            const std::size_t idx = static_cast<std::size_t>(i * n_qpts + q);
            scratch_basis_values_[idx] = values_at_pt[i];
            scratch_ref_gradients_[idx] = {
                gradients_at_pt[i][0],
                gradients_at_pt[i][1],
                gradients_at_pt[i][2]};

            // Transform gradient to physical space: grad_phys = J^{-T} * grad_ref
            const auto& grad_ref = scratch_ref_gradients_[idx];
            const auto& J_inv = scratch_inv_jacobians_[q];
            AssemblyContext::Vector3D grad_phys = {0.0, 0.0, 0.0};

            for (int d1 = 0; d1 < dim; ++d1) {
                for (int d2 = 0; d2 < dim; ++d2) {
                    grad_phys[d1] += J_inv[d2][d1] * grad_ref[d2];  // J^{-T}
                }
            }
            scratch_phys_gradients_[idx] = grad_phys;
        }

        // Evaluate trial basis if different
        if (&test_space != &trial_space) {
            const auto& trial_basis = trial_element.basis();
            trial_basis.evaluate_values(xi, values_at_pt);
            trial_basis.evaluate_gradients(xi, gradients_at_pt);

            for (LocalIndex j = 0; j < n_trial_dofs; ++j) {
                const std::size_t idx = static_cast<std::size_t>(j * n_qpts + q);
                trial_basis_values[idx] = values_at_pt[j];
                trial_ref_gradients[idx] = {
                    gradients_at_pt[j][0],
                    gradients_at_pt[j][1],
                    gradients_at_pt[j][2]};

                // Transform gradient
                const auto& grad_ref = trial_ref_gradients[idx];
                const auto& J_inv = scratch_inv_jacobians_[q];
                AssemblyContext::Vector3D grad_phys = {0.0, 0.0, 0.0};

                for (int d1 = 0; d1 < dim; ++d1) {
                    for (int d2 = 0; d2 < dim; ++d2) {
                        grad_phys[d1] += J_inv[d2][d1] * grad_ref[d2];
                    }
                }
                trial_phys_gradients[idx] = grad_phys;
            }
        }
    }

    // 9. Configure context with basic info
    context.configure(cell_id, test_element, trial_element, required_data);

    // 10. Set all computed data into context
    context.setQuadratureData(scratch_quad_points_, scratch_quad_weights_);
    context.setPhysicalPoints(scratch_phys_points_);
    context.setJacobianData(scratch_jacobians_, scratch_inv_jacobians_, scratch_jac_dets_);
    context.setIntegrationWeights(scratch_integration_weights_);

    // Set test basis data
    context.setTestBasisData(n_test_dofs, scratch_basis_values_, scratch_ref_gradients_);
    context.setPhysicalGradients(scratch_phys_gradients_,
        (&test_space != &trial_space) ? trial_phys_gradients : scratch_phys_gradients_);

    // Set trial basis data if different
    if (&test_space != &trial_space) {
        context.setTrialBasisData(n_trial_dofs, trial_basis_values, trial_ref_gradients);
    }
}

void StandardAssembler::prepareContextFace(
    AssemblyContext& context,
    const IMeshAccess& mesh,
    GlobalIndex face_id,
    GlobalIndex cell_id,
    LocalIndex local_face_id,
    const spaces::FunctionSpace& space,
    RequiredData required_data,
    ContextType type)
{
    // 1. Get element type from mesh
    const ElementType cell_type = mesh.getCellType(cell_id);
    const int dim = mesh.dimension();

    // 2. Get element for the space
    const auto& element = getElement(space, cell_id, cell_type);

    // 3. Determine face element type from reference topology
    elements::ReferenceElement ref = elements::ReferenceElement::create(cell_type);
    const auto& face_nodes = ref.face_nodes(static_cast<std::size_t>(local_face_id));

    ElementType face_type = ElementType::Unknown;
    switch (face_nodes.size()) {
        case 2:
            face_type = ElementType::Line2;
            break;
        case 3:
            face_type = ElementType::Triangle3;
            break;
        case 4:
            face_type = ElementType::Quad4;
            break;
        default:
            throw std::runtime_error("StandardAssembler::prepareContextFace: unsupported face topology");
    }

    // 4. Create a face quadrature rule
    const int quad_order = quadrature::QuadratureFactory::recommended_order(
        element.polynomial_order(), false);
    auto quad_rule = quadrature::QuadratureFactory::create(face_type, quad_order);

    const auto n_qpts = static_cast<LocalIndex>(quad_rule->num_points());
    const auto n_dofs = static_cast<LocalIndex>(element.num_dofs());

    // 5. Get cell node coordinates from mesh
    mesh.getCellCoordinates(cell_id, cell_coords_);
    const auto n_nodes = cell_coords_.size();

    // Convert to math::Vector format
    std::vector<math::Vector<Real, 3>> node_coords(n_nodes);
    for (std::size_t i = 0; i < n_nodes; ++i) {
        node_coords[i] = math::Vector<Real, 3>{
            cell_coords_[i][0], cell_coords_[i][1], cell_coords_[i][2]};
    }

    // 6. Create geometry mapping
    geometry::MappingRequest map_request;
    map_request.element_type = cell_type;
    map_request.geometry_order = 1;
    map_request.use_affine = true;

    auto mapping = geometry::MappingFactory::create(map_request, node_coords);

    // 7. Resize scratch storage
    scratch_quad_points_.resize(n_qpts);
    scratch_quad_weights_.resize(n_qpts);
    scratch_phys_points_.resize(n_qpts);
    scratch_jacobians_.resize(n_qpts);
    scratch_inv_jacobians_.resize(n_qpts);
    scratch_jac_dets_.resize(n_qpts);
    scratch_integration_weights_.resize(n_qpts);
    scratch_normals_.resize(n_qpts);

    const auto basis_size = static_cast<std::size_t>(n_dofs * n_qpts);
    scratch_basis_values_.resize(basis_size);
    scratch_ref_gradients_.resize(basis_size);
    scratch_phys_gradients_.resize(basis_size);

    // 8. Map face quadrature points to element reference coordinates and compute normals/weights
    const auto& quad_points = quad_rule->points();
    const auto& quad_weights = quad_rule->weights();

    const AssemblyContext::Vector3D n_ref = computeFaceNormal(local_face_id, cell_type, dim);

    for (LocalIndex q = 0; q < n_qpts; ++q) {
        const auto& qpt = quad_points[q];
        scratch_quad_weights_[q] = quad_weights[q];

        // Convert quadrature point to facet-local coordinates expected by ElementTransform
        math::Vector<Real, 3> facet_coords{};
        if (face_type == ElementType::Line2) {
            // Line quadrature is on [-1,1]; facet parameterization uses t in [0,1]
            facet_coords = math::Vector<Real, 3>{(qpt[0] + Real(1)) * Real(0.5), Real(0), Real(0)};
        } else if (face_type == ElementType::Quad4) {
            // Quad quadrature is on [-1,1]^2; facet parameterization uses (s,t) in [0,1]^2
            facet_coords = math::Vector<Real, 3>{
                (qpt[0] + Real(1)) * Real(0.5),
                (qpt[1] + Real(1)) * Real(0.5),
                Real(0)};
        } else {
            // Triangle quadrature uses reference simplex coordinates (0<=x,y, x+y<=1)
            facet_coords = math::Vector<Real, 3>{qpt[0], qpt[1], Real(0)};
        }

        // Map to the cell reference coordinates on the requested face
        const math::Vector<Real, 3> xi = elements::ElementTransform::facet_to_reference(
            cell_type, static_cast<int>(local_face_id), facet_coords);

        scratch_quad_points_[q] = {xi[0], xi[1], xi[2]};

        // Compute physical point and mapping Jacobians
        const auto x_phys = mapping->map_to_physical(xi);
        scratch_phys_points_[q] = {x_phys[0], x_phys[1], x_phys[2]};

        const auto J = mapping->jacobian(xi);
        const auto J_inv = mapping->jacobian_inverse(xi);
        const Real det_J = mapping->jacobian_determinant(xi);

        for (int i = 0; i < 3; ++i) {
            for (int j = 0; j < 3; ++j) {
                scratch_jacobians_[q][i][j] = J(i, j);
                scratch_inv_jacobians_[q][i][j] = J_inv(i, j);
            }
        }
        scratch_jac_dets_[q] = det_J;

        Real surface_measure;
        AssemblyContext::Vector3D n_phys;
        computeSurfaceMeasureAndNormal(n_ref, scratch_inv_jacobians_[q], det_J, dim,
                                       surface_measure, n_phys);

        scratch_integration_weights_[q] = quad_weights[q] * surface_measure;
        scratch_normals_[q] = n_phys;
    }

    // 9. Evaluate basis functions at face quadrature points
    const auto& basis = element.basis();
    std::vector<Real> values_at_pt;
    std::vector<basis::Gradient> gradients_at_pt;

    for (LocalIndex q = 0; q < n_qpts; ++q) {
        const math::Vector<Real, 3> xi{
            scratch_quad_points_[q][0],
            scratch_quad_points_[q][1],
            scratch_quad_points_[q][2]};

        basis.evaluate_values(xi, values_at_pt);
        basis.evaluate_gradients(xi, gradients_at_pt);

        for (LocalIndex i = 0; i < n_dofs; ++i) {
            const std::size_t idx = static_cast<std::size_t>(i * n_qpts + q);
            scratch_basis_values_[idx] = values_at_pt[i];
            scratch_ref_gradients_[idx] = {
                gradients_at_pt[i][0],
                gradients_at_pt[i][1],
                gradients_at_pt[i][2]};

            const auto& grad_ref = scratch_ref_gradients_[idx];
            const auto& J_inv = scratch_inv_jacobians_[q];
            AssemblyContext::Vector3D grad_phys = {0.0, 0.0, 0.0};

            for (int d1 = 0; d1 < dim; ++d1) {
                for (int d2 = 0; d2 < dim; ++d2) {
                    grad_phys[d1] += J_inv[d2][d1] * grad_ref[d2];
                }
            }
            scratch_phys_gradients_[idx] = grad_phys;
        }
    }

    // 10. Configure face context and set computed data
    context.configureFace(face_id, cell_id, local_face_id, element, required_data, type);
    context.setQuadratureData(scratch_quad_points_, scratch_quad_weights_);
    context.setPhysicalPoints(scratch_phys_points_);
    context.setJacobianData(scratch_jacobians_, scratch_inv_jacobians_, scratch_jac_dets_);
    context.setIntegrationWeights(scratch_integration_weights_);
    context.setTestBasisData(n_dofs, scratch_basis_values_, scratch_ref_gradients_);
    context.setPhysicalGradients(scratch_phys_gradients_, scratch_phys_gradients_);
    context.setNormals(scratch_normals_);
}

AssemblyContext::Vector3D StandardAssembler::computeFaceNormal(
    LocalIndex local_face_id,
    ElementType cell_type,
    int dim) const
{
    (void)dim;
    const auto n = elements::ElementTransform::reference_facet_normal(
        cell_type, static_cast<int>(local_face_id));
    return {n[0], n[1], n[2]};
}

void StandardAssembler::computeSurfaceMeasureAndNormal(
    const AssemblyContext::Vector3D& n_ref,
    const AssemblyContext::Matrix3x3& J_inv,
    Real det_J,
    int dim,
    Real& surface_measure,
    AssemblyContext::Vector3D& n_phys) const
{
    // Compute the transformation J^{-T} * n_ref.
    //
    // Mathematical derivation:
    // For a mapping x = F(xi) from reference to physical coordinates, the
    // Jacobian is J = dx/dxi. The transformation of area elements is:
    //
    //   dS_phys = ||cof(J) * n_ref|| * dS_ref
    //
    // where cof(J) is the cofactor matrix of J. Using the identity
    // cof(J) = det(J) * J^{-T}, we have:
    //
    //   dS_phys = ||det(J) * J^{-T} * n_ref|| * dS_ref
    //           = |det(J)| * ||J^{-T} * n_ref|| * dS_ref
    //
    // The physical normal direction (unnormalized) is given by:
    //   n_phys_unnorm = J^{-T} * n_ref = (J^{-1})^T * n_ref
    //
    // To apply J^{-T} = (J^{-1})^T to a vector v:
    //   (J^{-T} * v)_i = sum_k J^{-1}_{ki} * v_k
    //
    // This is the transpose action: column i of J^{-1} dotted with v.

    // Compute J^{-T} * n_ref
    AssemblyContext::Vector3D Jit_n = {0.0, 0.0, 0.0};
    for (int i = 0; i < dim; ++i) {
        for (int k = 0; k < dim; ++k) {
            // J^{-T}_{ik} = J^{-1}_{ki}
            Jit_n[i] += J_inv[k][i] * n_ref[k];
        }
    }

    // Compute the norm of J^{-T} * n_ref
    Real norm_Jit_n = 0.0;
    for (int i = 0; i < dim; ++i) {
        norm_Jit_n += Jit_n[i] * Jit_n[i];
    }
    norm_Jit_n = std::sqrt(norm_Jit_n);

    // Surface measure = ||J^{-T} * n_ref|| * |det(J)|
    surface_measure = norm_Jit_n * std::abs(det_J);

    // Physical unit normal = normalize(J^{-T} * n_ref)
    constexpr Real tol = 1e-14;
    if (norm_Jit_n > tol) {
        n_phys[0] = Jit_n[0] / norm_Jit_n;
        n_phys[1] = Jit_n[1] / norm_Jit_n;
        n_phys[2] = Jit_n[2] / norm_Jit_n;
    } else {
        // Degenerate case: fall back to reference normal
        // This should not happen for valid meshes
        n_phys = n_ref;
    }
}

void StandardAssembler::insertLocal(
    const KernelOutput& output,
    std::span<const GlobalIndex> row_dofs,
    std::span<const GlobalIndex> col_dofs,
    GlobalSystemView* matrix_view,
    GlobalSystemView* vector_view)
{
    // Insert matrix entries
    if (matrix_view && output.has_matrix) {
        matrix_view->addMatrixEntries(row_dofs, col_dofs, output.local_matrix);
    }

    // Insert vector entries
    if (vector_view && output.has_vector) {
        vector_view->addVectorEntries(row_dofs, output.local_vector);
    }
}

void StandardAssembler::insertLocalConstrained(
    const KernelOutput& output,
    std::span<const GlobalIndex> row_dofs,
    std::span<const GlobalIndex> col_dofs,
    GlobalSystemView* matrix_view,
    GlobalSystemView* vector_view)
{
    // Check if any DOFs are constrained
    if (!constraints_->hasConstrainedDofs(row_dofs) &&
        !constraints_->hasConstrainedDofs(col_dofs)) {
        // No constraints - direct insertion
        insertLocal(output, row_dofs, col_dofs, matrix_view, vector_view);
        return;
    }

    // Use ConstraintDistributor for constrained assembly
    // This expands constrained DOFs to their masters and distributes contributions

    // For matrix
    if (matrix_view && output.has_matrix && constraint_distributor_) {
        // Create matrix ops adapter
        class MatrixOpsAdapter : public constraints::IMatrixOperations {
        public:
            explicit MatrixOpsAdapter(GlobalSystemView& view) : view_(view) {}

            void addValues(std::span<const GlobalIndex> rows,
                           std::span<const GlobalIndex> cols,
                           std::span<const double> values) override {
                view_.addMatrixEntries(rows, cols, values);
            }

            void addValue(GlobalIndex row, GlobalIndex col, double value) override {
                view_.addMatrixEntry(row, col, value);
            }

            void setDiagonal(GlobalIndex row, double value) override {
                view_.setDiagonal(row, value);
            }

            [[nodiscard]] GlobalIndex numRows() const override { return view_.numRows(); }
            [[nodiscard]] GlobalIndex numCols() const override { return view_.numCols(); }

        private:
            GlobalSystemView& view_;
        };

        MatrixOpsAdapter matrix_ops(*matrix_view);

        // Also need vector ops if vector_view is provided
        if (vector_view && output.has_vector) {
            class VectorOpsAdapter : public constraints::IVectorOperations {
            public:
                explicit VectorOpsAdapter(GlobalSystemView& view) : view_(view) {}

                void addValues(std::span<const GlobalIndex> indices,
                               std::span<const double> values) override {
                    view_.addVectorEntries(indices, values);
                }

                void addValue(GlobalIndex index, double value) override {
                    view_.addVectorEntry(index, value);
                }

                void setValue(GlobalIndex index, double value) override {
                    view_.addVectorEntry(index, value, AddMode::Insert);
                }

                [[nodiscard]] double getValue(GlobalIndex index) const override {
                    return view_.getVectorEntry(index);
                }

                [[nodiscard]] GlobalIndex size() const override {
                    return view_.numRows();
                }

            private:
                GlobalSystemView& view_;
            };

            VectorOpsAdapter vector_ops(*vector_view);

            constraint_distributor_->distributeLocalToGlobal(
                output.local_matrix, output.local_vector,
                row_dofs, matrix_ops, vector_ops);
        } else {
            constraint_distributor_->distributeMatrixToGlobal(
                output.local_matrix, row_dofs, matrix_ops);
        }
    } else if (vector_view && output.has_vector && constraint_distributor_) {
        // Vector-only with constraints
        class VectorOpsAdapter : public constraints::IVectorOperations {
        public:
            explicit VectorOpsAdapter(GlobalSystemView& view) : view_(view) {}

            void addValues(std::span<const GlobalIndex> indices,
                           std::span<const double> values) override {
                view_.addVectorEntries(indices, values);
            }

            void addValue(GlobalIndex index, double value) override {
                view_.addVectorEntry(index, value);
            }

            void setValue(GlobalIndex index, double value) override {
                view_.addVectorEntry(index, value, AddMode::Insert);
            }

            [[nodiscard]] double getValue(GlobalIndex index) const override {
                return view_.getVectorEntry(index);
            }

            [[nodiscard]] GlobalIndex size() const override {
                return view_.numRows();
            }

        private:
            GlobalSystemView& view_;
        };

        VectorOpsAdapter vector_ops(*vector_view);
        constraint_distributor_->distributeRhsToGlobal(
            output.local_vector, row_dofs, vector_ops);
    }
}

const elements::Element& StandardAssembler::getElement(
    const spaces::FunctionSpace& space,
    GlobalIndex cell_id,
    ElementType cell_type) const
{
    return space.getElement(cell_type, cell_id);
}

// ============================================================================
// Factory Functions
// ============================================================================

std::unique_ptr<Assembler> createStandardAssembler()
{
    return std::make_unique<StandardAssembler>();
}

std::unique_ptr<Assembler> createStandardAssembler(const AssemblyOptions& options)
{
    return std::make_unique<StandardAssembler>(options);
}

std::unique_ptr<Assembler> createAssembler(ThreadingStrategy strategy)
{
    switch (strategy) {
        case ThreadingStrategy::Sequential:
            return createStandardAssembler();
        case ThreadingStrategy::Colored:
        case ThreadingStrategy::WorkStream:
        case ThreadingStrategy::Atomic:
            // These would return specialized assemblers
            // For now, fall back to standard
            return createStandardAssembler();
        default:
            return createStandardAssembler();
    }
}

std::unique_ptr<Assembler> createAssembler(const AssemblyOptions& options)
{
    return createAssembler(options.threading);
}

} // namespace assembly
} // namespace FE
} // namespace svmp
