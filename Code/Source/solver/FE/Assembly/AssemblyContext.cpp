/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#include "AssemblyContext.h"
#include "Elements/Element.h"
#include "Spaces/FunctionSpace.h"

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

    test_ref_hessians_.reserve(n_basis);
    test_phys_hessians_.reserve(n_basis);
    trial_ref_hessians_.reserve(n_basis);
    trial_phys_hessians_.reserve(n_basis);

    // Solution data
    solution_coefficients_.reserve(n_dofs);
    solution_values_.reserve(n_qpts);
    solution_vector_values_.reserve(n_qpts);
    solution_gradients_.reserve(n_qpts);
    solution_jacobians_.reserve(n_qpts);
    solution_hessians_.reserve(n_qpts);
    solution_laplacians_.reserve(n_qpts);
}

void AssemblyContext::configure(
    GlobalIndex cell_id,
    const elements::Element& test_element,
    const elements::Element& trial_element,
    RequiredData required_data)
{
    type_ = ContextType::Cell;
    cell_id_ = cell_id;
    face_id_ = -1;
    local_face_id_ = 0;
    boundary_marker_ = -1;
    required_data_ = required_data;

    n_test_dofs_ = static_cast<LocalIndex>(test_element.num_dofs());
    n_trial_dofs_ = static_cast<LocalIndex>(trial_element.num_dofs());
    trial_is_test_ = (&test_element == &trial_element);

    test_field_type_ = FieldType::Scalar;
    trial_field_type_ = FieldType::Scalar;
    test_value_dim_ = 1;
    trial_value_dim_ = 1;

    cell_diameter_ = 0.0;
    cell_volume_ = 0.0;
    facet_area_ = 0.0;

    material_state_old_base_ = nullptr;
    material_state_work_base_ = nullptr;
    material_state_bytes_per_qpt_ = 0;
    material_state_stride_bytes_ = 0;

    test_ref_hessians_.clear();
    test_phys_hessians_.clear();
    trial_ref_hessians_.clear();
    trial_phys_hessians_.clear();

    solution_hessians_.clear();
    solution_laplacians_.clear();
}

void AssemblyContext::configure(
    GlobalIndex cell_id,
    const spaces::FunctionSpace& test_space,
    const spaces::FunctionSpace& trial_space,
    RequiredData required_data)
{
    type_ = ContextType::Cell;
    cell_id_ = cell_id;
    face_id_ = -1;
    local_face_id_ = 0;
    boundary_marker_ = -1;
    required_data_ = required_data;

    test_field_type_ = test_space.field_type();
    trial_field_type_ = trial_space.field_type();
    test_value_dim_ = test_space.value_dimension();
    trial_value_dim_ = trial_space.value_dimension();

    n_test_dofs_ = static_cast<LocalIndex>(test_space.dofs_per_element());
    n_trial_dofs_ = static_cast<LocalIndex>(trial_space.dofs_per_element());
    trial_is_test_ = (&test_space == &trial_space);

    cell_diameter_ = 0.0;
    cell_volume_ = 0.0;
    facet_area_ = 0.0;

    material_state_old_base_ = nullptr;
    material_state_work_base_ = nullptr;
    material_state_bytes_per_qpt_ = 0;
    material_state_stride_bytes_ = 0;

    test_ref_hessians_.clear();
    test_phys_hessians_.clear();
    trial_ref_hessians_.clear();
    trial_phys_hessians_.clear();

    solution_hessians_.clear();
    solution_laplacians_.clear();
}

void AssemblyContext::configureFace(
    GlobalIndex face_id,
    GlobalIndex cell_id,
    LocalIndex local_face_id,
    const elements::Element& element,
    RequiredData required_data,
    ContextType type)
{
    type_ = type;
    cell_id_ = cell_id;
    face_id_ = face_id;
    local_face_id_ = local_face_id;
    required_data_ = required_data;

    n_test_dofs_ = static_cast<LocalIndex>(element.num_dofs());
    n_trial_dofs_ = n_test_dofs_;
    trial_is_test_ = true;

    test_field_type_ = FieldType::Scalar;
    trial_field_type_ = FieldType::Scalar;
    test_value_dim_ = 1;
    trial_value_dim_ = 1;

    cell_diameter_ = 0.0;
    cell_volume_ = 0.0;
    facet_area_ = 0.0;

    material_state_old_base_ = nullptr;
    material_state_work_base_ = nullptr;
    material_state_bytes_per_qpt_ = 0;
    material_state_stride_bytes_ = 0;

    test_ref_hessians_.clear();
    test_phys_hessians_.clear();
    trial_ref_hessians_.clear();
    trial_phys_hessians_.clear();

    solution_hessians_.clear();
    solution_laplacians_.clear();
}

void AssemblyContext::configureFace(
    GlobalIndex face_id,
    GlobalIndex cell_id,
    LocalIndex local_face_id,
    const spaces::FunctionSpace& test_space,
    const spaces::FunctionSpace& trial_space,
    RequiredData required_data,
    ContextType type)
{
    type_ = type;
    cell_id_ = cell_id;
    face_id_ = face_id;
    local_face_id_ = local_face_id;
    required_data_ = required_data;

    test_field_type_ = test_space.field_type();
    trial_field_type_ = trial_space.field_type();
    test_value_dim_ = test_space.value_dimension();
    trial_value_dim_ = trial_space.value_dimension();

    n_test_dofs_ = static_cast<LocalIndex>(test_space.dofs_per_element());
    n_trial_dofs_ = static_cast<LocalIndex>(trial_space.dofs_per_element());
    trial_is_test_ = (&test_space == &trial_space);

    cell_diameter_ = 0.0;
    cell_volume_ = 0.0;
    facet_area_ = 0.0;

    material_state_old_base_ = nullptr;
    material_state_work_base_ = nullptr;
    material_state_bytes_per_qpt_ = 0;
    material_state_stride_bytes_ = 0;

    test_ref_hessians_.clear();
    test_phys_hessians_.clear();
    trial_ref_hessians_.clear();
    trial_phys_hessians_.clear();

    solution_hessians_.clear();
    solution_laplacians_.clear();
}

void AssemblyContext::clear()
{
    cell_id_ = -1;
    face_id_ = -1;
    n_test_dofs_ = 0;
    n_trial_dofs_ = 0;
    n_qpts_ = 0;
    trial_is_test_ = true;

    test_field_type_ = FieldType::Scalar;
    trial_field_type_ = FieldType::Scalar;
    test_value_dim_ = 1;
    trial_value_dim_ = 1;

    cell_diameter_ = 0.0;
    cell_volume_ = 0.0;
    facet_area_ = 0.0;

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
    solution_vector_values_.clear();
    solution_gradients_.clear();
    solution_jacobians_.clear();
    history_solution_data_.clear();
    material_state_old_base_ = nullptr;
    material_state_work_base_ = nullptr;
    material_state_bytes_per_qpt_ = 0;
    material_state_stride_bytes_ = 0;
    time_ = 0.0;
    dt_ = 0.0;
    get_real_param_ = nullptr;
    get_param_ = nullptr;
    user_data_ = nullptr;
    time_integration_ = nullptr;
    required_data_ = RequiredData::None;

    test_ref_hessians_.clear();
    test_phys_hessians_.clear();
    trial_ref_hessians_.clear();
    trial_phys_hessians_.clear();
    solution_hessians_.clear();
    solution_laplacians_.clear();
}

bool AssemblyContext::hasPreviousSolutionData() const noexcept
{
    if (history_solution_data_.size() < 1) {
        return false;
    }
    const auto& data = history_solution_data_[0];
    return !data.values.empty() || !data.vector_values.empty();
}

bool AssemblyContext::hasPreviousSolution2Data() const noexcept
{
    if (history_solution_data_.size() < 2) {
        return false;
    }
    const auto& data = history_solution_data_[1];
    return !data.values.empty() || !data.vector_values.empty();
}

void AssemblyContext::clearPreviousSolutionDataK(int k) noexcept
{
    if (k <= 0 || history_solution_data_.size() < static_cast<std::size_t>(k)) {
        return;
    }
    auto& data = history_solution_data_[static_cast<std::size_t>(k - 1)];
    data.coefficients.clear();
    data.values.clear();
    data.vector_values.clear();
    data.gradients.clear();
    data.jacobians.clear();
}

void AssemblyContext::clearAllPreviousSolutionData() noexcept
{
    for (std::size_t i = 0; i < history_solution_data_.size(); ++i) {
        clearPreviousSolutionDataK(static_cast<int>(i + 1));
    }
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

Real AssemblyContext::cellVolume() const
{
    if (type_ != ContextType::Cell) {
        throw std::logic_error("AssemblyContext::cellVolume: not available for face contexts");
    }
    return cell_volume_;
}

Real AssemblyContext::facetArea() const
{
    if (type_ == ContextType::Cell) {
        throw std::logic_error("AssemblyContext::facetArea: not available for cell context");
    }
    return facet_area_;
}

void AssemblyContext::setEntityMeasures(Real cell_diameter, Real cell_volume, Real facet_area)
{
    cell_diameter_ = cell_diameter;
    cell_volume_ = cell_volume;
    facet_area_ = facet_area;
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

AssemblyContext::Matrix3x3 AssemblyContext::referenceHessian(LocalIndex i, LocalIndex q) const
{
    if (test_ref_hessians_.empty()) {
        throw std::logic_error("AssemblyContext::referenceHessian: basis Hessians not set");
    }
    if (i >= n_test_dofs_ || q >= n_qpts_) {
        throw std::out_of_range("AssemblyContext::referenceHessian: index out of range");
    }
    return test_ref_hessians_[static_cast<std::size_t>(i * n_qpts_ + q)];
}

AssemblyContext::Matrix3x3 AssemblyContext::physicalHessian(LocalIndex i, LocalIndex q) const
{
    if (test_phys_hessians_.empty()) {
        throw std::logic_error("AssemblyContext::physicalHessian: basis Hessians not set");
    }
    if (i >= n_test_dofs_ || q >= n_qpts_) {
        throw std::out_of_range("AssemblyContext::physicalHessian: index out of range");
    }
    return test_phys_hessians_[static_cast<std::size_t>(i * n_qpts_ + q)];
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

AssemblyContext::Matrix3x3 AssemblyContext::trialReferenceHessian(LocalIndex j, LocalIndex q) const
{
    if (trial_is_test_) {
        return referenceHessian(j, q);
    }
    if (trial_ref_hessians_.empty()) {
        throw std::logic_error("AssemblyContext::trialReferenceHessian: trial basis Hessians not set");
    }
    if (j >= n_trial_dofs_ || q >= n_qpts_) {
        throw std::out_of_range("AssemblyContext::trialReferenceHessian: index out of range");
    }
    return trial_ref_hessians_[static_cast<std::size_t>(j * n_qpts_ + q)];
}

AssemblyContext::Matrix3x3 AssemblyContext::trialPhysicalHessian(LocalIndex j, LocalIndex q) const
{
    if (trial_is_test_) {
        return physicalHessian(j, q);
    }
    if (trial_phys_hessians_.empty()) {
        throw std::logic_error("AssemblyContext::trialPhysicalHessian: trial basis Hessians not set");
    }
    if (j >= n_trial_dofs_ || q >= n_qpts_) {
        throw std::out_of_range("AssemblyContext::trialPhysicalHessian: index out of range");
    }
    return trial_phys_hessians_[static_cast<std::size_t>(j * n_qpts_ + q)];
}

// ============================================================================
// Solution Data Access
// ============================================================================

void AssemblyContext::setSolutionCoefficients(std::span<const Real> coefficients)
{
    if (coefficients.size() < static_cast<std::size_t>(n_trial_dofs_)) {
        throw std::invalid_argument("AssemblyContext::setSolutionCoefficients: coefficient size does not match trial DOFs");
    }

    solution_coefficients_.assign(coefficients.begin(),
                                  coefficients.begin() + static_cast<std::size_t>(n_trial_dofs_));

    const bool have_trial_phys_gradients =
        trial_is_test_ ? !test_phys_gradients_.empty() : !trial_phys_gradients_.empty();
    const bool need_gradients = hasFlag(required_data_, RequiredData::SolutionGradients) ||
                                hasFlag(required_data_, RequiredData::SolutionHessians) ||
                                hasFlag(required_data_, RequiredData::SolutionLaplacians) ||
                                (required_data_ == RequiredData::None && have_trial_phys_gradients);

    if (trial_field_type_ == FieldType::Scalar) {
        solution_vector_values_.clear();
        solution_jacobians_.clear();

        // Compute scalar u_h(x_q)
        solution_values_.resize(static_cast<std::size_t>(n_qpts_), 0.0);
        for (LocalIndex q = 0; q < n_qpts_; ++q) {
            Real val = 0.0;
            for (LocalIndex j = 0; j < n_trial_dofs_; ++j) {
                val += solution_coefficients_[static_cast<std::size_t>(j)] * trialBasisValue(j, q);
            }
            solution_values_[static_cast<std::size_t>(q)] = val;
        }

        // Compute grad(u_h)(x_q)
        solution_gradients_.clear();
        if (need_gradients) {
            solution_gradients_.resize(static_cast<std::size_t>(n_qpts_), Vector3D{0.0, 0.0, 0.0});
            for (LocalIndex q = 0; q < n_qpts_; ++q) {
                Vector3D grad = {0.0, 0.0, 0.0};
                for (LocalIndex j = 0; j < n_trial_dofs_; ++j) {
                    const auto grad_j = trialPhysicalGradient(j, q);
                    const Real coef = solution_coefficients_[static_cast<std::size_t>(j)];
                    grad[0] += coef * grad_j[0];
                    grad[1] += coef * grad_j[1];
                    grad[2] += coef * grad_j[2];
                }
                solution_gradients_[static_cast<std::size_t>(q)] = grad;
            }
        }
    } else if (trial_field_type_ == FieldType::Vector) {
        FE_CHECK_ARG(trial_value_dim_ > 0 && trial_value_dim_ <= 3,
                     "AssemblyContext::setSolutionCoefficients: invalid trial value dimension");
        FE_CHECK_ARG((n_trial_dofs_ % static_cast<LocalIndex>(trial_value_dim_)) == 0,
                     "AssemblyContext::setSolutionCoefficients: trial DOFs not divisible by value_dimension");

        const LocalIndex dofs_per_component =
            static_cast<LocalIndex>(n_trial_dofs_ / static_cast<LocalIndex>(trial_value_dim_));

        solution_values_.clear();
        solution_gradients_.clear();

        // Compute vector u_h(x_q)
        solution_vector_values_.resize(static_cast<std::size_t>(n_qpts_), Vector3D{0.0, 0.0, 0.0});
        for (LocalIndex q = 0; q < n_qpts_; ++q) {
            Vector3D u = {0.0, 0.0, 0.0};
            for (int c = 0; c < trial_value_dim_; ++c) {
                Real val_c = 0.0;
                const LocalIndex base = static_cast<LocalIndex>(c) * dofs_per_component;
                for (LocalIndex j = 0; j < dofs_per_component; ++j) {
                    const LocalIndex jj = base + j;
                    val_c += solution_coefficients_[static_cast<std::size_t>(jj)] * trialBasisValue(jj, q);
                }
                u[static_cast<std::size_t>(c)] = val_c;
            }
            solution_vector_values_[static_cast<std::size_t>(q)] = u;
        }

        // Compute Jacobian J_{ij} = du_i/dx_j
        solution_jacobians_.clear();
        if (need_gradients) {
            solution_jacobians_.resize(static_cast<std::size_t>(n_qpts_), Matrix3x3{});
            for (LocalIndex q = 0; q < n_qpts_; ++q) {
                Matrix3x3 J{};
                for (int c = 0; c < trial_value_dim_; ++c) {
                    const LocalIndex base = static_cast<LocalIndex>(c) * dofs_per_component;
                    for (LocalIndex j = 0; j < dofs_per_component; ++j) {
                        const LocalIndex jj = base + j;
                        const auto grad_j = trialPhysicalGradient(jj, q);
                        const Real coef = solution_coefficients_[static_cast<std::size_t>(jj)];
                        J[static_cast<std::size_t>(c)][0] += coef * grad_j[0];
                        J[static_cast<std::size_t>(c)][1] += coef * grad_j[1];
                        J[static_cast<std::size_t>(c)][2] += coef * grad_j[2];
                    }
                }
                solution_jacobians_[static_cast<std::size_t>(q)] = J;
            }
        }
    } else {
        throw FEException("AssemblyContext::setSolutionCoefficients: only scalar and vector trial fields are supported",
                          __FILE__, __LINE__, __func__, FEStatus::NotImplemented);
    }

    const bool need_hessians = hasFlag(required_data_, RequiredData::SolutionHessians) ||
                               hasFlag(required_data_, RequiredData::SolutionLaplacians);
    if (need_hessians) {
        if (trial_field_type_ != FieldType::Scalar) {
            throw FEException("AssemblyContext::setSolutionCoefficients: solution Hessians are only implemented for scalar fields",
                              __FILE__, __LINE__, __func__, FEStatus::NotImplemented);
        }
        if (trial_is_test_) {
            if (test_phys_hessians_.empty()) {
                throw std::logic_error(
                    "AssemblyContext::setSolutionCoefficients: basis Hessians not set (RequiredData::BasisHessians)");
            }
        } else {
            if (trial_phys_hessians_.empty()) {
                throw std::logic_error(
                    "AssemblyContext::setSolutionCoefficients: trial basis Hessians not set (RequiredData::BasisHessians)");
            }
        }

        solution_hessians_.resize(static_cast<std::size_t>(n_qpts_), Matrix3x3{});
        for (LocalIndex q = 0; q < n_qpts_; ++q) {
            Matrix3x3 H{};
            for (LocalIndex j = 0; j < n_trial_dofs_; ++j) {
                const auto Hj = trialPhysicalHessian(j, q);
                const Real coef = solution_coefficients_[static_cast<std::size_t>(j)];
                for (int r = 0; r < 3; ++r) {
                    for (int c = 0; c < 3; ++c) {
                        H[static_cast<std::size_t>(r)][static_cast<std::size_t>(c)] +=
                            coef * Hj[static_cast<std::size_t>(r)][static_cast<std::size_t>(c)];
                    }
                }
            }
            solution_hessians_[static_cast<std::size_t>(q)] = H;
        }

        if (hasFlag(required_data_, RequiredData::SolutionLaplacians)) {
            solution_laplacians_.resize(static_cast<std::size_t>(n_qpts_), 0.0);
            const int dim = dim_;
            for (LocalIndex q = 0; q < n_qpts_; ++q) {
                Real lap = 0.0;
                const auto& H = solution_hessians_[static_cast<std::size_t>(q)];
                for (int d = 0; d < dim; ++d) {
                    lap += H[static_cast<std::size_t>(d)][static_cast<std::size_t>(d)];
                }
                solution_laplacians_[static_cast<std::size_t>(q)] = lap;
            }
        }
    } else {
        solution_hessians_.clear();
        solution_laplacians_.clear();
    }
}

void AssemblyContext::setPreviousSolutionCoefficients(std::span<const Real> coefficients)
{
    setPreviousSolutionCoefficientsK(1, coefficients);
}

void AssemblyContext::setPreviousSolution2Coefficients(std::span<const Real> coefficients)
{
    setPreviousSolutionCoefficientsK(2, coefficients);
}

void AssemblyContext::setPreviousSolutionCoefficientsK(int k, std::span<const Real> coefficients)
{
    FE_THROW_IF(k <= 0, InvalidArgumentException,
                "AssemblyContext::setPreviousSolutionCoefficientsK: k must be >= 1");
    if (coefficients.size() < static_cast<std::size_t>(n_trial_dofs_)) {
        throw std::invalid_argument(
            "AssemblyContext::setPreviousSolutionCoefficientsK: coefficient size does not match trial DOFs");
    }

    if (history_solution_data_.size() < static_cast<std::size_t>(k)) {
        history_solution_data_.resize(static_cast<std::size_t>(k));
    }
    auto& dst = history_solution_data_[static_cast<std::size_t>(k - 1)];

    dst.coefficients.assign(coefficients.begin(),
                            coefficients.begin() + static_cast<std::size_t>(n_trial_dofs_));

    if (trial_field_type_ == FieldType::Scalar) {
        dst.vector_values.clear();
        dst.jacobians.clear();

        dst.values.resize(static_cast<std::size_t>(n_qpts_), 0.0);
        for (LocalIndex q = 0; q < n_qpts_; ++q) {
            Real val = 0.0;
            for (LocalIndex j = 0; j < n_trial_dofs_; ++j) {
                val += dst.coefficients[static_cast<std::size_t>(j)] * trialBasisValue(j, q);
            }
            dst.values[static_cast<std::size_t>(q)] = val;
        }

        dst.gradients.resize(static_cast<std::size_t>(n_qpts_), Vector3D{0.0, 0.0, 0.0});
        for (LocalIndex q = 0; q < n_qpts_; ++q) {
            Vector3D grad = {0.0, 0.0, 0.0};
            for (LocalIndex j = 0; j < n_trial_dofs_; ++j) {
                const auto grad_j = trialPhysicalGradient(j, q);
                const Real coef = dst.coefficients[static_cast<std::size_t>(j)];
                grad[0] += coef * grad_j[0];
                grad[1] += coef * grad_j[1];
                grad[2] += coef * grad_j[2];
            }
            dst.gradients[static_cast<std::size_t>(q)] = grad;
        }
    } else if (trial_field_type_ == FieldType::Vector) {
        FE_CHECK_ARG(trial_value_dim_ > 0 && trial_value_dim_ <= 3,
                     "AssemblyContext::setPreviousSolutionCoefficientsK: invalid trial value dimension");
        FE_CHECK_ARG((n_trial_dofs_ % static_cast<LocalIndex>(trial_value_dim_)) == 0,
                     "AssemblyContext::setPreviousSolutionCoefficientsK: trial DOFs not divisible by value_dimension");

        const LocalIndex dofs_per_component =
            static_cast<LocalIndex>(n_trial_dofs_ / static_cast<LocalIndex>(trial_value_dim_));

        dst.values.clear();
        dst.gradients.clear();

        dst.vector_values.resize(static_cast<std::size_t>(n_qpts_), Vector3D{0.0, 0.0, 0.0});
        for (LocalIndex q = 0; q < n_qpts_; ++q) {
            Vector3D u = {0.0, 0.0, 0.0};
            for (int c = 0; c < trial_value_dim_; ++c) {
                Real val_c = 0.0;
                const LocalIndex base = static_cast<LocalIndex>(c) * dofs_per_component;
                for (LocalIndex j = 0; j < dofs_per_component; ++j) {
                    const LocalIndex jj = base + j;
                    val_c += dst.coefficients[static_cast<std::size_t>(jj)] * trialBasisValue(jj, q);
                }
                u[static_cast<std::size_t>(c)] = val_c;
            }
            dst.vector_values[static_cast<std::size_t>(q)] = u;
        }

        dst.jacobians.resize(static_cast<std::size_t>(n_qpts_), Matrix3x3{});
        for (LocalIndex q = 0; q < n_qpts_; ++q) {
            Matrix3x3 J{};
            for (int c = 0; c < trial_value_dim_; ++c) {
                const LocalIndex base = static_cast<LocalIndex>(c) * dofs_per_component;
                for (LocalIndex j = 0; j < dofs_per_component; ++j) {
                    const LocalIndex jj = base + j;
                    const auto grad_j = trialPhysicalGradient(jj, q);
                    const Real coef = dst.coefficients[static_cast<std::size_t>(jj)];
                    J[static_cast<std::size_t>(c)][0] += coef * grad_j[0];
                    J[static_cast<std::size_t>(c)][1] += coef * grad_j[1];
                    J[static_cast<std::size_t>(c)][2] += coef * grad_j[2];
                }
            }
            dst.jacobians[static_cast<std::size_t>(q)] = J;
        }
    } else {
        throw FEException("AssemblyContext::setPreviousSolutionCoefficientsK: only scalar and vector trial fields are supported",
                          __FILE__, __LINE__, __func__, FEStatus::NotImplemented);
    }
}

Real AssemblyContext::solutionValue(LocalIndex q) const
{
    if (trial_field_type_ != FieldType::Scalar) {
        throw std::logic_error("AssemblyContext::solutionValue: trial field is not scalar-valued");
    }
    if (solution_values_.empty()) {
        throw std::logic_error("AssemblyContext::solutionValue: solution data not set");
    }
    if (q >= n_qpts_) {
        throw std::out_of_range("AssemblyContext::solutionValue: index out of range");
    }
    return solution_values_[static_cast<std::size_t>(q)];
}

AssemblyContext::Vector3D AssemblyContext::solutionVectorValue(LocalIndex q) const
{
    if (trial_field_type_ != FieldType::Vector) {
        throw std::logic_error("AssemblyContext::solutionVectorValue: trial field is not vector-valued");
    }
    if (solution_vector_values_.empty()) {
        throw std::logic_error("AssemblyContext::solutionVectorValue: solution data not set");
    }
    if (q >= n_qpts_) {
        throw std::out_of_range("AssemblyContext::solutionVectorValue: index out of range");
    }
    return solution_vector_values_[static_cast<std::size_t>(q)];
}

Real AssemblyContext::previousSolutionValue(LocalIndex q) const
{
    return previousSolutionValue(q, 1);
}

Real AssemblyContext::previousSolutionValue(LocalIndex q, int k) const
{
    if (trial_field_type_ != FieldType::Scalar) {
        throw std::logic_error("AssemblyContext::previousSolutionValue: trial field is not scalar-valued");
    }
    if (k <= 0 || history_solution_data_.size() < static_cast<std::size_t>(k) ||
        history_solution_data_[static_cast<std::size_t>(k - 1)].values.empty()) {
        throw std::logic_error("AssemblyContext::previousSolutionValue: previous solution data not set");
    }
    if (q >= n_qpts_) {
        throw std::out_of_range("AssemblyContext::previousSolutionValue: index out of range");
    }
    const auto& data = history_solution_data_[static_cast<std::size_t>(k - 1)];
    return data.values[static_cast<std::size_t>(q)];
}

AssemblyContext::Vector3D AssemblyContext::previousSolutionVectorValue(LocalIndex q) const
{
    return previousSolutionVectorValue(q, 1);
}

AssemblyContext::Vector3D AssemblyContext::previousSolutionVectorValue(LocalIndex q, int k) const
{
    if (trial_field_type_ != FieldType::Vector) {
        throw std::logic_error("AssemblyContext::previousSolutionVectorValue: trial field is not vector-valued");
    }
    if (k <= 0 || history_solution_data_.size() < static_cast<std::size_t>(k) ||
        history_solution_data_[static_cast<std::size_t>(k - 1)].vector_values.empty()) {
        throw std::logic_error("AssemblyContext::previousSolutionVectorValue: previous solution data not set");
    }
    if (q >= n_qpts_) {
        throw std::out_of_range("AssemblyContext::previousSolutionVectorValue: index out of range");
    }
    const auto& data = history_solution_data_[static_cast<std::size_t>(k - 1)];
    return data.vector_values[static_cast<std::size_t>(q)];
}

Real AssemblyContext::previousSolution2Value(LocalIndex q) const
{
    return previousSolutionValue(q, 2);
}

AssemblyContext::Vector3D AssemblyContext::previousSolution2VectorValue(LocalIndex q) const
{
    return previousSolutionVectorValue(q, 2);
}

AssemblyContext::Vector3D AssemblyContext::solutionGradient(LocalIndex q) const
{
    if (trial_field_type_ != FieldType::Scalar) {
        throw std::logic_error("AssemblyContext::solutionGradient: trial field is not scalar-valued");
    }
    if (solution_gradients_.empty()) {
        throw std::logic_error("AssemblyContext::solutionGradient: solution data not set");
    }
    if (q >= n_qpts_) {
        throw std::out_of_range("AssemblyContext::solutionGradient: index out of range");
    }
    return solution_gradients_[static_cast<std::size_t>(q)];
}

AssemblyContext::Matrix3x3 AssemblyContext::solutionJacobian(LocalIndex q) const
{
    if (trial_field_type_ != FieldType::Vector) {
        throw std::logic_error("AssemblyContext::solutionJacobian: trial field is not vector-valued");
    }
    if (solution_jacobians_.empty()) {
        throw std::logic_error("AssemblyContext::solutionJacobian: solution Jacobian data not set");
    }
    if (q >= n_qpts_) {
        throw std::out_of_range("AssemblyContext::solutionJacobian: index out of range");
    }
    return solution_jacobians_[static_cast<std::size_t>(q)];
}

AssemblyContext::Vector3D AssemblyContext::previousSolutionGradient(LocalIndex q) const
{
    if (trial_field_type_ != FieldType::Scalar) {
        throw std::logic_error("AssemblyContext::previousSolutionGradient: trial field is not scalar-valued");
    }
    if (history_solution_data_.size() < 1 || history_solution_data_[0].gradients.empty()) {
        throw std::logic_error("AssemblyContext::previousSolutionGradient: previous solution gradient data not set");
    }
    if (q >= n_qpts_) {
        throw std::out_of_range("AssemblyContext::previousSolutionGradient: index out of range");
    }
    return history_solution_data_[0].gradients[static_cast<std::size_t>(q)];
}

AssemblyContext::Matrix3x3 AssemblyContext::previousSolutionJacobian(LocalIndex q) const
{
    if (trial_field_type_ != FieldType::Vector) {
        throw std::logic_error("AssemblyContext::previousSolutionJacobian: trial field is not vector-valued");
    }
    if (history_solution_data_.size() < 1 || history_solution_data_[0].jacobians.empty()) {
        throw std::logic_error("AssemblyContext::previousSolutionJacobian: previous solution Jacobian data not set");
    }
    if (q >= n_qpts_) {
        throw std::out_of_range("AssemblyContext::previousSolutionJacobian: index out of range");
    }
    return history_solution_data_[0].jacobians[static_cast<std::size_t>(q)];
}

AssemblyContext::Vector3D AssemblyContext::previousSolution2Gradient(LocalIndex q) const
{
    if (trial_field_type_ != FieldType::Scalar) {
        throw std::logic_error("AssemblyContext::previousSolution2Gradient: trial field is not scalar-valued");
    }
    if (history_solution_data_.size() < 2 || history_solution_data_[1].gradients.empty()) {
        throw std::logic_error("AssemblyContext::previousSolution2Gradient: previous-previous solution gradient data not set");
    }
    if (q >= n_qpts_) {
        throw std::out_of_range("AssemblyContext::previousSolution2Gradient: index out of range");
    }
    return history_solution_data_[1].gradients[static_cast<std::size_t>(q)];
}

AssemblyContext::Matrix3x3 AssemblyContext::previousSolution2Jacobian(LocalIndex q) const
{
    if (trial_field_type_ != FieldType::Vector) {
        throw std::logic_error("AssemblyContext::previousSolution2Jacobian: trial field is not vector-valued");
    }
    if (history_solution_data_.size() < 2 || history_solution_data_[1].jacobians.empty()) {
        throw std::logic_error("AssemblyContext::previousSolution2Jacobian: previous-previous solution Jacobian data not set");
    }
    if (q >= n_qpts_) {
        throw std::out_of_range("AssemblyContext::previousSolution2Jacobian: index out of range");
    }
    return history_solution_data_[1].jacobians[static_cast<std::size_t>(q)];
}

AssemblyContext::Matrix3x3 AssemblyContext::solutionHessian(LocalIndex q) const
{
    if (solution_hessians_.empty()) {
        throw std::logic_error("AssemblyContext::solutionHessian: solution Hessian data not set");
    }
    if (q >= n_qpts_) {
        throw std::out_of_range("AssemblyContext::solutionHessian: index out of range");
    }
    return solution_hessians_[static_cast<std::size_t>(q)];
}

Real AssemblyContext::solutionLaplacian(LocalIndex q) const
{
    if (solution_laplacians_.empty()) {
        throw std::logic_error("AssemblyContext::solutionLaplacian: solution Laplacian data not set");
    }
    if (q >= n_qpts_) {
        throw std::out_of_range("AssemblyContext::solutionLaplacian: index out of range");
    }
    return solution_laplacians_[static_cast<std::size_t>(q)];
}

// ============================================================================
// Material State
// ============================================================================

void AssemblyContext::setMaterialState(std::byte* cell_state_base,
                                       std::size_t bytes_per_qpt,
                                       std::size_t stride_bytes) noexcept
{
    setMaterialState(cell_state_base, cell_state_base, bytes_per_qpt, stride_bytes);
}

void AssemblyContext::setMaterialState(std::byte* cell_state_old_base,
                                       std::byte* cell_state_work_base,
                                       std::size_t bytes_per_qpt,
                                       std::size_t stride_bytes) noexcept
{
    material_state_old_base_ = (cell_state_old_base != nullptr) ? cell_state_old_base : cell_state_work_base;
    material_state_work_base_ = cell_state_work_base;
    material_state_bytes_per_qpt_ = bytes_per_qpt;
    material_state_stride_bytes_ = stride_bytes;
}

std::span<std::byte> AssemblyContext::materialState(LocalIndex q) const
{
    return materialStateWork(q);
}

std::span<std::byte> AssemblyContext::materialStateWork(LocalIndex q) const
{
    if (material_state_work_base_ == nullptr || material_state_bytes_per_qpt_ == 0 || material_state_stride_bytes_ == 0) {
        throw std::logic_error("AssemblyContext::materialState: material state not set");
    }
    if (material_state_stride_bytes_ < material_state_bytes_per_qpt_) {
        throw std::logic_error("AssemblyContext::materialState: invalid state stride");
    }
    if (q >= n_qpts_) {
        throw std::out_of_range("AssemblyContext::materialState: index out of range");
    }

    auto* ptr = material_state_work_base_ + static_cast<std::size_t>(q) * material_state_stride_bytes_;
    return {ptr, material_state_bytes_per_qpt_};
}

std::span<const std::byte> AssemblyContext::materialStateOld(LocalIndex q) const
{
    if (material_state_work_base_ == nullptr || material_state_bytes_per_qpt_ == 0 || material_state_stride_bytes_ == 0) {
        throw std::logic_error("AssemblyContext::materialStateOld: material state not set");
    }
    if (material_state_stride_bytes_ < material_state_bytes_per_qpt_) {
        throw std::logic_error("AssemblyContext::materialStateOld: invalid state stride");
    }
    if (q >= n_qpts_) {
        throw std::out_of_range("AssemblyContext::materialStateOld: index out of range");
    }

    auto* base = (material_state_old_base_ != nullptr) ? material_state_old_base_ : material_state_work_base_;
    auto* ptr = base + static_cast<std::size_t>(q) * material_state_stride_bytes_;
    return {ptr, material_state_bytes_per_qpt_};
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

void AssemblyContext::setTestBasisHessians(
    LocalIndex n_dofs,
    std::span<const Matrix3x3> hessians)
{
    (void)n_dofs;
    test_ref_hessians_.assign(hessians.begin(), hessians.end());
}

void AssemblyContext::setTrialBasisHessians(
    LocalIndex n_dofs,
    std::span<const Matrix3x3> hessians)
{
    (void)n_dofs;
    trial_ref_hessians_.assign(hessians.begin(), hessians.end());
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

void AssemblyContext::setPhysicalHessians(
    std::span<const Matrix3x3> test_hessians,
    std::span<const Matrix3x3> trial_hessians)
{
    test_phys_hessians_.assign(test_hessians.begin(), test_hessians.end());
    if (!trial_is_test_) {
        trial_phys_hessians_.assign(trial_hessians.begin(), trial_hessians.end());
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
