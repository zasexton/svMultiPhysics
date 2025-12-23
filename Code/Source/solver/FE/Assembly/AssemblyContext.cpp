/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#include "AssemblyContext.h"
#include "Elements/Element.h"

#include <stdexcept>
#include <algorithm>

namespace svmp {
namespace FE {
namespace assembly {

// ============================================================================
// AssemblyContext Implementation
// ============================================================================

AssemblyContext::AssemblyContext() = default;

AssemblyContext::~AssemblyContext() = default;

AssemblyContext::AssemblyContext(AssemblyContext&& other) noexcept = default;

AssemblyContext& AssemblyContext::operator=(AssemblyContext&& other) noexcept = default;

void AssemblyContext::reserve(LocalIndex max_dofs, LocalIndex max_qpts, int dim)
{
    dim_ = dim;

    const auto n_dofs = static_cast<std::size_t>(max_dofs);
    const auto n_qpts = static_cast<std::size_t>(max_qpts);
    const auto n_basis = n_dofs * n_qpts;

    // Quadrature data
    quad_points_.reserve(n_qpts);
    quad_weights_.reserve(n_qpts);
    integration_weights_.reserve(n_qpts);

    // Geometry data
    physical_points_.reserve(n_qpts);
    jacobians_.reserve(n_qpts);
    inverse_jacobians_.reserve(n_qpts);
    jacobian_dets_.reserve(n_qpts);
    normals_.reserve(n_qpts);

    // Basis data
    test_basis_values_.reserve(n_basis);
    test_ref_gradients_.reserve(n_basis);
    test_phys_gradients_.reserve(n_basis);
    trial_basis_values_.reserve(n_basis);
    trial_ref_gradients_.reserve(n_basis);
    trial_phys_gradients_.reserve(n_basis);

    // Solution data
    solution_coefficients_.reserve(n_dofs);
    solution_values_.reserve(n_qpts);
    solution_gradients_.reserve(n_qpts);
}

void AssemblyContext::configure(
    GlobalIndex cell_id,
    const elements::Element& test_element,
    const elements::Element& trial_element,
    RequiredData /*required_data*/)
{
    type_ = ContextType::Cell;
    cell_id_ = cell_id;
    face_id_ = -1;
    local_face_id_ = 0;
    boundary_marker_ = -1;

    n_test_dofs_ = static_cast<LocalIndex>(test_element.num_dofs());
    n_trial_dofs_ = static_cast<LocalIndex>(trial_element.num_dofs());
    trial_is_test_ = (&test_element == &trial_element);
}

void AssemblyContext::configureFace(
    GlobalIndex face_id,
    GlobalIndex cell_id,
    LocalIndex local_face_id,
    const elements::Element& element,
    RequiredData /*required_data*/,
    ContextType type)
{
    type_ = type;
    cell_id_ = cell_id;
    face_id_ = face_id;
    local_face_id_ = local_face_id;

    n_test_dofs_ = static_cast<LocalIndex>(element.num_dofs());
    n_trial_dofs_ = n_test_dofs_;
    trial_is_test_ = true;
}

void AssemblyContext::clear()
{
    cell_id_ = -1;
    face_id_ = -1;
    n_test_dofs_ = 0;
    n_trial_dofs_ = 0;
    n_qpts_ = 0;
    trial_is_test_ = true;

    quad_points_.clear();
    quad_weights_.clear();
    integration_weights_.clear();
    physical_points_.clear();
    jacobians_.clear();
    inverse_jacobians_.clear();
    jacobian_dets_.clear();
    normals_.clear();
    test_basis_values_.clear();
    test_ref_gradients_.clear();
    test_phys_gradients_.clear();
    trial_basis_values_.clear();
    trial_ref_gradients_.clear();
    trial_phys_gradients_.clear();
    solution_coefficients_.clear();
    solution_values_.clear();
    solution_gradients_.clear();
}

// ============================================================================
// Quadrature Data Access
// ============================================================================

AssemblyContext::Point3D AssemblyContext::quadraturePoint(LocalIndex q) const
{
    if (q >= n_qpts_) {
        throw std::out_of_range("AssemblyContext::quadraturePoint: index out of range");
    }
    return quad_points_[static_cast<std::size_t>(q)];
}

Real AssemblyContext::quadratureWeight(LocalIndex q) const
{
    if (q >= n_qpts_) {
        throw std::out_of_range("AssemblyContext::quadratureWeight: index out of range");
    }
    return quad_weights_[static_cast<std::size_t>(q)];
}

Real AssemblyContext::integrationWeight(LocalIndex q) const
{
    if (q >= n_qpts_) {
        throw std::out_of_range("AssemblyContext::integrationWeight: index out of range");
    }
    return integration_weights_[static_cast<std::size_t>(q)];
}

// ============================================================================
// Geometry Data Access
// ============================================================================

AssemblyContext::Point3D AssemblyContext::physicalPoint(LocalIndex q) const
{
    if (q >= n_qpts_) {
        throw std::out_of_range("AssemblyContext::physicalPoint: index out of range");
    }
    return physical_points_[static_cast<std::size_t>(q)];
}

Real AssemblyContext::jacobianDet(LocalIndex q) const
{
    if (q >= n_qpts_) {
        throw std::out_of_range("AssemblyContext::jacobianDet: index out of range");
    }
    return jacobian_dets_[static_cast<std::size_t>(q)];
}

AssemblyContext::Matrix3x3 AssemblyContext::jacobian(LocalIndex q) const
{
    if (q >= n_qpts_) {
        throw std::out_of_range("AssemblyContext::jacobian: index out of range");
    }
    return jacobians_[static_cast<std::size_t>(q)];
}

AssemblyContext::Matrix3x3 AssemblyContext::inverseJacobian(LocalIndex q) const
{
    if (q >= n_qpts_) {
        throw std::out_of_range("AssemblyContext::inverseJacobian: index out of range");
    }
    return inverse_jacobians_[static_cast<std::size_t>(q)];
}

AssemblyContext::Vector3D AssemblyContext::normal(LocalIndex q) const
{
    if (type_ == ContextType::Cell) {
        throw std::logic_error("AssemblyContext::normal: not available for cell context");
    }
    if (q >= n_qpts_) {
        throw std::out_of_range("AssemblyContext::normal: index out of range");
    }
    return normals_[static_cast<std::size_t>(q)];
}

// ============================================================================
// Test Basis Function Access
// ============================================================================

Real AssemblyContext::basisValue(LocalIndex i, LocalIndex q) const
{
    if (i >= n_test_dofs_ || q >= n_qpts_) {
        throw std::out_of_range("AssemblyContext::basisValue: index out of range");
    }
    return test_basis_values_[static_cast<std::size_t>(i * n_qpts_ + q)];
}

std::span<const Real> AssemblyContext::basisValues(LocalIndex i) const
{
    if (i >= n_test_dofs_) {
        throw std::out_of_range("AssemblyContext::basisValues: index out of range");
    }
    const auto offset = static_cast<std::size_t>(i * n_qpts_);
    return {test_basis_values_.data() + offset, static_cast<std::size_t>(n_qpts_)};
}

AssemblyContext::Vector3D AssemblyContext::referenceGradient(LocalIndex i, LocalIndex q) const
{
    if (i >= n_test_dofs_ || q >= n_qpts_) {
        throw std::out_of_range("AssemblyContext::referenceGradient: index out of range");
    }
    return test_ref_gradients_[static_cast<std::size_t>(i * n_qpts_ + q)];
}

AssemblyContext::Vector3D AssemblyContext::physicalGradient(LocalIndex i, LocalIndex q) const
{
    if (i >= n_test_dofs_ || q >= n_qpts_) {
        throw std::out_of_range("AssemblyContext::physicalGradient: index out of range");
    }
    return test_phys_gradients_[static_cast<std::size_t>(i * n_qpts_ + q)];
}

// ============================================================================
// Trial Basis Function Access
// ============================================================================

Real AssemblyContext::trialBasisValue(LocalIndex j, LocalIndex q) const
{
    if (j >= n_trial_dofs_ || q >= n_qpts_) {
        throw std::out_of_range("AssemblyContext::trialBasisValue: index out of range");
    }

    if (trial_is_test_) {
        return test_basis_values_[static_cast<std::size_t>(j * n_qpts_ + q)];
    }
    return trial_basis_values_[static_cast<std::size_t>(j * n_qpts_ + q)];
}

AssemblyContext::Vector3D AssemblyContext::trialPhysicalGradient(LocalIndex j, LocalIndex q) const
{
    if (j >= n_trial_dofs_ || q >= n_qpts_) {
        throw std::out_of_range("AssemblyContext::trialPhysicalGradient: index out of range");
    }

    if (trial_is_test_) {
        return test_phys_gradients_[static_cast<std::size_t>(j * n_qpts_ + q)];
    }
    return trial_phys_gradients_[static_cast<std::size_t>(j * n_qpts_ + q)];
}

// ============================================================================
// Solution Data Access
// ============================================================================

void AssemblyContext::setSolutionCoefficients(std::span<const Real> coefficients)
{
    solution_coefficients_.assign(coefficients.begin(), coefficients.end());

    // Compute solution values at quadrature points
    solution_values_.resize(static_cast<std::size_t>(n_qpts_), 0.0);

    for (LocalIndex q = 0; q < n_qpts_; ++q) {
        Real val = 0.0;
        for (LocalIndex i = 0; i < n_test_dofs_; ++i) {
            val += coefficients[static_cast<std::size_t>(i)] * basisValue(i, q);
        }
        solution_values_[static_cast<std::size_t>(q)] = val;
    }

    // Compute solution gradients at quadrature points
    solution_gradients_.resize(static_cast<std::size_t>(n_qpts_), Vector3D{0.0, 0.0, 0.0});

    for (LocalIndex q = 0; q < n_qpts_; ++q) {
        Vector3D grad = {0.0, 0.0, 0.0};
        for (LocalIndex i = 0; i < n_test_dofs_; ++i) {
            auto grad_i = physicalGradient(i, q);
            Real coef = coefficients[static_cast<std::size_t>(i)];
            grad[0] += coef * grad_i[0];
            grad[1] += coef * grad_i[1];
            grad[2] += coef * grad_i[2];
        }
        solution_gradients_[static_cast<std::size_t>(q)] = grad;
    }
}

Real AssemblyContext::solutionValue(LocalIndex q) const
{
    if (solution_values_.empty()) {
        throw std::logic_error("AssemblyContext::solutionValue: solution data not set");
    }
    if (q >= n_qpts_) {
        throw std::out_of_range("AssemblyContext::solutionValue: index out of range");
    }
    return solution_values_[static_cast<std::size_t>(q)];
}

AssemblyContext::Vector3D AssemblyContext::solutionGradient(LocalIndex q) const
{
    if (solution_gradients_.empty()) {
        throw std::logic_error("AssemblyContext::solutionGradient: solution data not set");
    }
    if (q >= n_qpts_) {
        throw std::out_of_range("AssemblyContext::solutionGradient: index out of range");
    }
    return solution_gradients_[static_cast<std::size_t>(q)];
}

// ============================================================================
// Data Setting Methods (called by assembler)
// ============================================================================

void AssemblyContext::setQuadratureData(
    std::span<const Point3D> points,
    std::span<const Real> weights)
{
    if (points.size() != weights.size()) {
        throw std::invalid_argument("AssemblyContext::setQuadratureData: size mismatch");
    }

    n_qpts_ = static_cast<LocalIndex>(points.size());
    quad_points_.assign(points.begin(), points.end());
    quad_weights_.assign(weights.begin(), weights.end());
}

void AssemblyContext::setPhysicalPoints(std::span<const Point3D> points)
{
    physical_points_.assign(points.begin(), points.end());
}

void AssemblyContext::setJacobianData(
    std::span<const Matrix3x3> jacobians,
    std::span<const Matrix3x3> inverse_jacobians,
    std::span<const Real> determinants)
{
    jacobians_.assign(jacobians.begin(), jacobians.end());
    inverse_jacobians_.assign(inverse_jacobians.begin(), inverse_jacobians.end());
    jacobian_dets_.assign(determinants.begin(), determinants.end());
}

void AssemblyContext::setIntegrationWeights(std::span<const Real> weights)
{
    integration_weights_.assign(weights.begin(), weights.end());
}

void AssemblyContext::setTestBasisData(
    LocalIndex n_dofs,
    std::span<const Real> values,
    std::span<const Vector3D> gradients)
{
    n_test_dofs_ = n_dofs;
    if (trial_is_test_) {
        n_trial_dofs_ = n_dofs;
    }
    test_basis_values_.assign(values.begin(), values.end());
    test_ref_gradients_.assign(gradients.begin(), gradients.end());
}

void AssemblyContext::setTrialBasisData(
    LocalIndex n_dofs,
    std::span<const Real> values,
    std::span<const Vector3D> gradients)
{
    n_trial_dofs_ = n_dofs;
    trial_is_test_ = false;
    trial_basis_values_.assign(values.begin(), values.end());
    trial_ref_gradients_.assign(gradients.begin(), gradients.end());
}

void AssemblyContext::setPhysicalGradients(
    std::span<const Vector3D> test_gradients,
    std::span<const Vector3D> trial_gradients)
{
    test_phys_gradients_.assign(test_gradients.begin(), test_gradients.end());

    if (!trial_is_test_) {
        trial_phys_gradients_.assign(trial_gradients.begin(), trial_gradients.end());
    }
}

void AssemblyContext::setNormals(std::span<const Vector3D> normals)
{
    normals_.assign(normals.begin(), normals.end());
}

void AssemblyContext::setSolutionValues(std::span<const Real> values)
{
    solution_values_.assign(values.begin(), values.end());
}

void AssemblyContext::setSolutionGradients(std::span<const Vector3D> gradients)
{
    solution_gradients_.assign(gradients.begin(), gradients.end());
}

// ============================================================================
// AssemblyContextPool Implementation
// ============================================================================

AssemblyContextPool::AssemblyContextPool(
    int num_threads,
    LocalIndex max_dofs,
    LocalIndex max_qpts,
    int dim)
{
    contexts_.reserve(static_cast<std::size_t>(num_threads));

    for (int i = 0; i < num_threads; ++i) {
        auto ctx = std::make_unique<AssemblyContext>();
        ctx->reserve(max_dofs, max_qpts, dim);
        contexts_.push_back(std::move(ctx));
    }
}

AssemblyContext& AssemblyContextPool::getContext(int thread_id)
{
    if (thread_id < 0 || thread_id >= static_cast<int>(contexts_.size())) {
        throw std::out_of_range("AssemblyContextPool::getContext: invalid thread_id");
    }
    return *contexts_[static_cast<std::size_t>(thread_id)];
}

} // namespace assembly
} // namespace FE
} // namespace svmp
