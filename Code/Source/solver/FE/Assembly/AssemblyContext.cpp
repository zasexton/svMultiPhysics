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
#include <type_traits>

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

    const LocalIndex clamped_max_dofs = std::max<LocalIndex>(max_dofs, 0);
    const LocalIndex clamped_max_qpts = std::max<LocalIndex>(max_qpts, 0);
    const LocalIndex target_max_dofs = std::max(arena_max_dofs_, clamped_max_dofs);
    const LocalIndex target_max_qpts = std::max(arena_max_qpts_, clamped_max_qpts);

    if (target_max_dofs == arena_max_dofs_ &&
        target_max_qpts == arena_max_qpts_ &&
        !arena_storage_.empty()) {
        return;
    }

    const auto n_dofs = static_cast<std::size_t>(target_max_dofs);
    const auto n_qpts = static_cast<std::size_t>(target_max_qpts);
    const auto n_basis = n_dofs * n_qpts;

    constexpr std::size_t arena_align = jit::kJITPointerAlignmentBytes;
    const auto align_up = [](std::size_t value, std::size_t alignment) -> std::size_t {
        const std::size_t mask = alignment - 1u;
        return (value + mask) & ~mask;
    };

    std::size_t arena_bytes = 0;
    const auto reserve_block = [&](std::size_t bytes) -> std::size_t {
        arena_bytes = align_up(arena_bytes, arena_align);
        const std::size_t offset = arena_bytes;
        arena_bytes += bytes;
        return offset;
    };

    const std::size_t off_quad_points = reserve_block(sizeof(Point3D) * n_qpts);
    const std::size_t off_quad_weights = reserve_block(sizeof(Real) * n_qpts);
    const std::size_t off_integration_weights = reserve_block(sizeof(Real) * n_qpts);

    const std::size_t off_physical_points = reserve_block(sizeof(Point3D) * n_qpts);
    const std::size_t off_jacobians = reserve_block(sizeof(Matrix3x3) * n_qpts);
    const std::size_t off_inverse_jacobians = reserve_block(sizeof(Matrix3x3) * n_qpts);
    const std::size_t off_jacobian_dets = reserve_block(sizeof(Real) * n_qpts);
    const std::size_t off_normals = reserve_block(sizeof(Vector3D) * n_qpts);
    const std::size_t off_interleaved_qpoint_geometry = reserve_block(
        sizeof(Real) * n_qpts * static_cast<std::size_t>(AssemblyContext::kInterleavedQPointGeometryStride));

    const std::size_t off_test_basis_values = reserve_block(sizeof(Real) * n_basis);
    const std::size_t off_test_ref_gradients = reserve_block(sizeof(Vector3D) * n_basis);
    const std::size_t off_test_phys_gradients = reserve_block(sizeof(Vector3D) * n_basis);
    const std::size_t off_test_basis_vector_values = reserve_block(sizeof(Vector3D) * n_basis);
    const std::size_t off_test_basis_curls = reserve_block(sizeof(Vector3D) * n_basis);
    const std::size_t off_test_basis_divergences = reserve_block(sizeof(Real) * n_basis);

    const std::size_t off_trial_basis_values = reserve_block(sizeof(Real) * n_basis);
    const std::size_t off_trial_ref_gradients = reserve_block(sizeof(Vector3D) * n_basis);
    const std::size_t off_trial_phys_gradients = reserve_block(sizeof(Vector3D) * n_basis);
    const std::size_t off_trial_basis_vector_values = reserve_block(sizeof(Vector3D) * n_basis);
    const std::size_t off_trial_basis_curls = reserve_block(sizeof(Vector3D) * n_basis);
    const std::size_t off_trial_basis_divergences = reserve_block(sizeof(Real) * n_basis);

    const std::size_t off_test_ref_hessians = reserve_block(sizeof(Matrix3x3) * n_basis);
    const std::size_t off_test_phys_hessians = reserve_block(sizeof(Matrix3x3) * n_basis);
    const std::size_t off_trial_ref_hessians = reserve_block(sizeof(Matrix3x3) * n_basis);
    const std::size_t off_trial_phys_hessians = reserve_block(sizeof(Matrix3x3) * n_basis);

    const std::size_t off_solution_coefficients = reserve_block(sizeof(Real) * n_dofs);
    const std::size_t off_solution_values = reserve_block(sizeof(Real) * n_qpts);
    const std::size_t off_solution_vector_values = reserve_block(sizeof(Vector3D) * n_qpts);
    const std::size_t off_solution_gradients = reserve_block(sizeof(Vector3D) * n_qpts);
    const std::size_t off_solution_jacobians = reserve_block(sizeof(Matrix3x3) * n_qpts);
    const std::size_t off_solution_hessians = reserve_block(sizeof(Matrix3x3) * n_qpts);
    const std::size_t off_solution_laplacians = reserve_block(sizeof(Real) * n_qpts);
    const std::size_t off_solution_component_hessians = reserve_block(sizeof(Matrix3x3) * n_qpts * 3u);
    const std::size_t off_solution_component_laplacians = reserve_block(sizeof(Real) * n_qpts * 3u);

    if (arena_storage_.size() < arena_bytes) {
        arena_storage_.resize(arena_bytes);
    }

    const auto bind = [&](auto& view, std::size_t offset, std::size_t count) {
        using ViewType = std::remove_reference_t<decltype(view)>;
        using T = typename ViewType::value_type;
        auto* ptr = (count == 0 || arena_storage_.empty())
                        ? nullptr
                        : reinterpret_cast<T*>(arena_storage_.data() + offset);
        view.bind(ptr, count);
    };

    bind(quad_points_, off_quad_points, n_qpts);
    bind(quad_weights_, off_quad_weights, n_qpts);
    bind(integration_weights_, off_integration_weights, n_qpts);

    bind(physical_points_, off_physical_points, n_qpts);
    bind(jacobians_, off_jacobians, n_qpts);
    bind(inverse_jacobians_, off_inverse_jacobians, n_qpts);
    bind(jacobian_dets_, off_jacobian_dets, n_qpts);
    bind(normals_, off_normals, n_qpts);
    bind(interleaved_qpoint_geometry_, off_interleaved_qpoint_geometry,
         n_qpts * static_cast<std::size_t>(AssemblyContext::kInterleavedQPointGeometryStride));

    bind(test_basis_values_, off_test_basis_values, n_basis);
    bind(test_ref_gradients_, off_test_ref_gradients, n_basis);
    bind(test_phys_gradients_, off_test_phys_gradients, n_basis);
    bind(test_basis_vector_values_, off_test_basis_vector_values, n_basis);
    bind(test_basis_curls_, off_test_basis_curls, n_basis);
    bind(test_basis_divergences_, off_test_basis_divergences, n_basis);

    bind(trial_basis_values_, off_trial_basis_values, n_basis);
    bind(trial_ref_gradients_, off_trial_ref_gradients, n_basis);
    bind(trial_phys_gradients_, off_trial_phys_gradients, n_basis);
    bind(trial_basis_vector_values_, off_trial_basis_vector_values, n_basis);
    bind(trial_basis_curls_, off_trial_basis_curls, n_basis);
    bind(trial_basis_divergences_, off_trial_basis_divergences, n_basis);

    bind(test_ref_hessians_, off_test_ref_hessians, n_basis);
    bind(test_phys_hessians_, off_test_phys_hessians, n_basis);
    bind(trial_ref_hessians_, off_trial_ref_hessians, n_basis);
    bind(trial_phys_hessians_, off_trial_phys_hessians, n_basis);

    bind(solution_coefficients_, off_solution_coefficients, n_dofs);
    bind(solution_values_, off_solution_values, n_qpts);
    bind(solution_vector_values_, off_solution_vector_values, n_qpts);
    bind(solution_gradients_, off_solution_gradients, n_qpts);
    bind(solution_jacobians_, off_solution_jacobians, n_qpts);
    bind(solution_hessians_, off_solution_hessians, n_qpts);
    bind(solution_laplacians_, off_solution_laplacians, n_qpts);
    bind(solution_component_hessians_, off_solution_component_hessians, n_qpts * 3u);
    bind(solution_component_laplacians_, off_solution_component_laplacians, n_qpts * 3u);

    arena_max_dofs_ = target_max_dofs;
    arena_max_qpts_ = target_max_qpts;
}

void AssemblyContext::ensureArenaCapacity(LocalIndex required_dofs, LocalIndex required_qpts)
{
    const LocalIndex clamped_required_dofs = std::max<LocalIndex>(required_dofs, 0);
    const LocalIndex clamped_required_qpts = std::max<LocalIndex>(required_qpts, 0);
    if (clamped_required_dofs <= arena_max_dofs_ && clamped_required_qpts <= arena_max_qpts_) {
        return;
    }

    const auto snapshot = [](const auto& src) {
        using SrcType = std::remove_reference_t<decltype(src)>;
        using Value = typename SrcType::value_type;
        return std::vector<Value>(src.begin(), src.end());
    };

    const auto quad_points = snapshot(quad_points_);
    const auto quad_weights = snapshot(quad_weights_);
    const auto integration_weights = snapshot(integration_weights_);
    const auto physical_points = snapshot(physical_points_);
    const auto jacobians = snapshot(jacobians_);
    const auto inverse_jacobians = snapshot(inverse_jacobians_);
    const auto jacobian_dets = snapshot(jacobian_dets_);
    const auto normals = snapshot(normals_);
    const auto interleaved_qpoint_geometry = snapshot(interleaved_qpoint_geometry_);

    const auto test_basis_values = snapshot(test_basis_values_);
    const auto test_ref_gradients = snapshot(test_ref_gradients_);
    const auto test_phys_gradients = snapshot(test_phys_gradients_);
    const auto test_basis_vector_values = snapshot(test_basis_vector_values_);
    const auto test_basis_curls = snapshot(test_basis_curls_);
    const auto test_basis_divergences = snapshot(test_basis_divergences_);

    const auto trial_basis_values = snapshot(trial_basis_values_);
    const auto trial_ref_gradients = snapshot(trial_ref_gradients_);
    const auto trial_phys_gradients = snapshot(trial_phys_gradients_);
    const auto trial_basis_vector_values = snapshot(trial_basis_vector_values_);
    const auto trial_basis_curls = snapshot(trial_basis_curls_);
    const auto trial_basis_divergences = snapshot(trial_basis_divergences_);

    const auto test_ref_hessians = snapshot(test_ref_hessians_);
    const auto test_phys_hessians = snapshot(test_phys_hessians_);
    const auto trial_ref_hessians = snapshot(trial_ref_hessians_);
    const auto trial_phys_hessians = snapshot(trial_phys_hessians_);

    const auto solution_coefficients = snapshot(solution_coefficients_);
    const auto solution_values = snapshot(solution_values_);
    const auto solution_vector_values = snapshot(solution_vector_values_);
    const auto solution_gradients = snapshot(solution_gradients_);
    const auto solution_jacobians = snapshot(solution_jacobians_);
    const auto solution_hessians = snapshot(solution_hessians_);
    const auto solution_laplacians = snapshot(solution_laplacians_);
    const auto solution_component_hessians = snapshot(solution_component_hessians_);
    const auto solution_component_laplacians = snapshot(solution_component_laplacians_);

    reserve(clamped_required_dofs, clamped_required_qpts, dim_);

    const auto restore = [](auto& dst, const auto& src) {
        dst.assign(src.begin(), src.end());
    };

    restore(quad_points_, quad_points);
    restore(quad_weights_, quad_weights);
    restore(integration_weights_, integration_weights);
    restore(physical_points_, physical_points);
    restore(jacobians_, jacobians);
    restore(inverse_jacobians_, inverse_jacobians);
    restore(jacobian_dets_, jacobian_dets);
    restore(normals_, normals);
    restore(interleaved_qpoint_geometry_, interleaved_qpoint_geometry);

    restore(test_basis_values_, test_basis_values);
    restore(test_ref_gradients_, test_ref_gradients);
    restore(test_phys_gradients_, test_phys_gradients);
    restore(test_basis_vector_values_, test_basis_vector_values);
    restore(test_basis_curls_, test_basis_curls);
    restore(test_basis_divergences_, test_basis_divergences);

    restore(trial_basis_values_, trial_basis_values);
    restore(trial_ref_gradients_, trial_ref_gradients);
    restore(trial_phys_gradients_, trial_phys_gradients);
    restore(trial_basis_vector_values_, trial_basis_vector_values);
    restore(trial_basis_curls_, trial_basis_curls);
    restore(trial_basis_divergences_, trial_basis_divergences);

    restore(test_ref_hessians_, test_ref_hessians);
    restore(test_phys_hessians_, test_phys_hessians);
    restore(trial_ref_hessians_, trial_ref_hessians);
    restore(trial_phys_hessians_, trial_phys_hessians);

    restore(solution_coefficients_, solution_coefficients);
    restore(solution_values_, solution_values);
    restore(solution_vector_values_, solution_vector_values);
    restore(solution_gradients_, solution_gradients);
    restore(solution_jacobians_, solution_jacobians);
    restore(solution_hessians_, solution_hessians);
    restore(solution_laplacians_, solution_laplacians);
    restore(solution_component_hessians_, solution_component_hessians);
    restore(solution_component_laplacians_, solution_component_laplacians);
}

void AssemblyContext::configure(
    GlobalIndex cell_id,
    const elements::Element& test_element,
    const elements::Element& trial_element,
    RequiredData required_data)
{
    type_ = ContextType::Cell;
    cell_id_ = cell_id;
    cell_domain_id_ = 0;
    face_id_ = -1;
    local_face_id_ = 0;
    boundary_marker_ = -1;
    required_data_ = required_data;

    n_test_dofs_ = static_cast<LocalIndex>(test_element.num_dofs());
    n_trial_dofs_ = static_cast<LocalIndex>(trial_element.num_dofs());
    trial_is_test_ = (&test_element == &trial_element);

    test_field_type_ = test_element.field_type();
    trial_field_type_ = trial_element.field_type();
    test_continuity_ = test_element.continuity();
    trial_continuity_ = trial_element.continuity();
    test_is_vector_basis_ = test_element.basis().is_vector_valued();
    trial_is_vector_basis_ = trial_element.basis().is_vector_valued();
    test_value_dim_ = (test_field_type_ == FieldType::Vector) ? test_element.dimension() : 1;
    trial_value_dim_ = (trial_field_type_ == FieldType::Vector) ? trial_element.dimension() : 1;

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

    test_basis_vector_values_.clear();
    test_basis_curls_.clear();
    test_basis_divergences_.clear();
    trial_basis_vector_values_.clear();
    trial_basis_curls_.clear();
    trial_basis_divergences_.clear();

    solution_hessians_.clear();
    solution_laplacians_.clear();
    solution_component_hessians_.clear();
    solution_component_laplacians_.clear();
    field_solution_data_.clear();
}

void AssemblyContext::configure(
    GlobalIndex cell_id,
    const spaces::FunctionSpace& test_space,
    const spaces::FunctionSpace& trial_space,
    RequiredData required_data)
{
    type_ = ContextType::Cell;
    cell_id_ = cell_id;
    cell_domain_id_ = 0;
    face_id_ = -1;
    local_face_id_ = 0;
    boundary_marker_ = -1;
    required_data_ = required_data;

    test_field_type_ = test_space.field_type();
    trial_field_type_ = trial_space.field_type();
    test_continuity_ = test_space.continuity();
    trial_continuity_ = trial_space.continuity();
    test_is_vector_basis_ = test_space.element().basis().is_vector_valued();
    trial_is_vector_basis_ = trial_space.element().basis().is_vector_valued();
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

    test_basis_vector_values_.clear();
    test_basis_curls_.clear();
    test_basis_divergences_.clear();
    trial_basis_vector_values_.clear();
    trial_basis_curls_.clear();
    trial_basis_divergences_.clear();

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
    cell_domain_id_ = 0;
    face_id_ = face_id;
    local_face_id_ = local_face_id;
    required_data_ = required_data;

    n_test_dofs_ = static_cast<LocalIndex>(element.num_dofs());
    n_trial_dofs_ = n_test_dofs_;
    trial_is_test_ = true;

    test_field_type_ = element.field_type();
    trial_field_type_ = element.field_type();
    test_continuity_ = element.continuity();
    trial_continuity_ = element.continuity();
    test_is_vector_basis_ = element.basis().is_vector_valued();
    trial_is_vector_basis_ = test_is_vector_basis_;
    test_value_dim_ = (test_field_type_ == FieldType::Vector) ? element.dimension() : 1;
    trial_value_dim_ = test_value_dim_;

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

    test_basis_vector_values_.clear();
    test_basis_curls_.clear();
    test_basis_divergences_.clear();
    trial_basis_vector_values_.clear();
    trial_basis_curls_.clear();
    trial_basis_divergences_.clear();

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
    cell_domain_id_ = 0;
    face_id_ = face_id;
    local_face_id_ = local_face_id;
    required_data_ = required_data;

    test_field_type_ = test_space.field_type();
    trial_field_type_ = trial_space.field_type();
    test_continuity_ = test_space.continuity();
    trial_continuity_ = trial_space.continuity();
    test_is_vector_basis_ = test_space.element().basis().is_vector_valued();
    trial_is_vector_basis_ = trial_space.element().basis().is_vector_valued();
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

    test_basis_vector_values_.clear();
    test_basis_curls_.clear();
    test_basis_divergences_.clear();
    trial_basis_vector_values_.clear();
    trial_basis_curls_.clear();
    trial_basis_divergences_.clear();

    solution_hessians_.clear();
    solution_laplacians_.clear();
}

void AssemblyContext::clear()
{
    cell_id_ = -1;
    cell_domain_id_ = 0;
    face_id_ = -1;
    n_test_dofs_ = 0;
    n_trial_dofs_ = 0;
    n_qpts_ = 0;
    trial_is_test_ = true;

    test_field_type_ = FieldType::Scalar;
    trial_field_type_ = FieldType::Scalar;
    test_continuity_ = Continuity::C0;
    trial_continuity_ = Continuity::C0;
    test_is_vector_basis_ = false;
    trial_is_vector_basis_ = false;
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
    interleaved_qpoint_geometry_.clear();
    test_basis_values_.clear();
    test_ref_gradients_.clear();
    test_phys_gradients_.clear();
    test_basis_vector_values_.clear();
    test_basis_curls_.clear();
    test_basis_divergences_.clear();
    trial_basis_values_.clear();
    trial_ref_gradients_.clear();
    trial_phys_gradients_.clear();
    trial_basis_vector_values_.clear();
    trial_basis_curls_.clear();
    trial_basis_divergences_.clear();
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
    material_state_alignment_bytes_ = alignof(std::max_align_t);
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
    solution_component_hessians_.clear();
    solution_component_laplacians_.clear();
    field_solution_data_.clear();
    jit_field_solution_table_.clear();
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
    data.component_hessians.clear();
    data.component_laplacians.clear();
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
    if (test_is_vector_basis_) {
        throw std::logic_error("AssemblyContext::basisValue: scalar basis values not available for vector-basis spaces");
    }
    if (i >= n_test_dofs_ || q >= n_qpts_) {
        throw std::out_of_range("AssemblyContext::basisValue: index out of range");
    }
    return test_basis_values_[static_cast<std::size_t>(i * n_qpts_ + q)];
}

std::span<const Real> AssemblyContext::basisValues(LocalIndex i) const
{
    if (test_is_vector_basis_) {
        throw std::logic_error("AssemblyContext::basisValues: scalar basis values not available for vector-basis spaces");
    }
    if (i >= n_test_dofs_) {
        throw std::out_of_range("AssemblyContext::basisValues: index out of range");
    }
    const auto offset = static_cast<std::size_t>(i * n_qpts_);
    return {test_basis_values_.data() + offset, static_cast<std::size_t>(n_qpts_)};
}

AssemblyContext::Vector3D AssemblyContext::referenceGradient(LocalIndex i, LocalIndex q) const
{
    if (test_is_vector_basis_) {
        throw std::logic_error("AssemblyContext::referenceGradient: scalar basis gradients not available for vector-basis spaces");
    }
    if (i >= n_test_dofs_ || q >= n_qpts_) {
        throw std::out_of_range("AssemblyContext::referenceGradient: index out of range");
    }
    return test_ref_gradients_[static_cast<std::size_t>(i * n_qpts_ + q)];
}

AssemblyContext::Vector3D AssemblyContext::physicalGradient(LocalIndex i, LocalIndex q) const
{
    if (test_is_vector_basis_) {
        throw std::logic_error("AssemblyContext::physicalGradient: scalar basis gradients not available for vector-basis spaces");
    }
    if (i >= n_test_dofs_ || q >= n_qpts_) {
        throw std::out_of_range("AssemblyContext::physicalGradient: index out of range");
    }
    return test_phys_gradients_[static_cast<std::size_t>(q * n_test_dofs_ + i)];
}

AssemblyContext::Vector3D AssemblyContext::basisVectorValue(LocalIndex i, LocalIndex q) const
{
    if (!test_is_vector_basis_) {
        throw std::logic_error("AssemblyContext::basisVectorValue: test space does not use a vector basis");
    }
    if (test_basis_vector_values_.empty()) {
        throw std::logic_error("AssemblyContext::basisVectorValue: vector basis values not set");
    }
    if (i >= n_test_dofs_ || q >= n_qpts_) {
        throw std::out_of_range("AssemblyContext::basisVectorValue: index out of range");
    }
    return test_basis_vector_values_[static_cast<std::size_t>(i * n_qpts_ + q)];
}

AssemblyContext::Vector3D AssemblyContext::basisCurl(LocalIndex i, LocalIndex q) const
{
    if (!test_is_vector_basis_) {
        throw std::logic_error("AssemblyContext::basisCurl: test space does not use a vector basis");
    }
    if (test_basis_curls_.empty()) {
        throw std::logic_error("AssemblyContext::basisCurl: basis curls not set");
    }
    if (i >= n_test_dofs_ || q >= n_qpts_) {
        throw std::out_of_range("AssemblyContext::basisCurl: index out of range");
    }
    return test_basis_curls_[static_cast<std::size_t>(i * n_qpts_ + q)];
}

Real AssemblyContext::basisDivergence(LocalIndex i, LocalIndex q) const
{
    if (!test_is_vector_basis_) {
        throw std::logic_error("AssemblyContext::basisDivergence: test space does not use a vector basis");
    }
    if (test_basis_divergences_.empty()) {
        throw std::logic_error("AssemblyContext::basisDivergence: basis divergences not set");
    }
    if (i >= n_test_dofs_ || q >= n_qpts_) {
        throw std::out_of_range("AssemblyContext::basisDivergence: index out of range");
    }
    return test_basis_divergences_[static_cast<std::size_t>(i * n_qpts_ + q)];
}

AssemblyContext::Matrix3x3 AssemblyContext::referenceHessian(LocalIndex i, LocalIndex q) const
{
    if (test_is_vector_basis_) {
        throw std::logic_error("AssemblyContext::referenceHessian: scalar basis Hessians not available for vector-basis spaces");
    }
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
    if (test_is_vector_basis_) {
        throw std::logic_error("AssemblyContext::physicalHessian: scalar basis Hessians not available for vector-basis spaces");
    }
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
    if (trial_is_vector_basis_) {
        throw std::logic_error("AssemblyContext::trialBasisValue: scalar basis values not available for vector-basis spaces");
    }
    if (j >= n_trial_dofs_ || q >= n_qpts_) {
        throw std::out_of_range("AssemblyContext::trialBasisValue: index out of range");
    }

    if (trial_is_test_) {
        return test_basis_values_[static_cast<std::size_t>(j * n_qpts_ + q)];
    }
    return trial_basis_values_[static_cast<std::size_t>(j * n_qpts_ + q)];
}

AssemblyContext::Vector3D AssemblyContext::trialBasisVectorValue(LocalIndex j, LocalIndex q) const
{
    if (!trial_is_vector_basis_) {
        throw std::logic_error("AssemblyContext::trialBasisVectorValue: trial space does not use a vector basis");
    }
    const auto& storage = trial_is_test_ ? test_basis_vector_values_ : trial_basis_vector_values_;
    if (storage.empty()) {
        throw std::logic_error("AssemblyContext::trialBasisVectorValue: vector basis values not set");
    }
    if (j >= n_trial_dofs_ || q >= n_qpts_) {
        throw std::out_of_range("AssemblyContext::trialBasisVectorValue: index out of range");
    }
    return storage[static_cast<std::size_t>(j * n_qpts_ + q)];
}

AssemblyContext::Vector3D AssemblyContext::trialBasisCurl(LocalIndex j, LocalIndex q) const
{
    if (!trial_is_vector_basis_) {
        throw std::logic_error("AssemblyContext::trialBasisCurl: trial space does not use a vector basis");
    }
    const auto& storage = trial_is_test_ ? test_basis_curls_ : trial_basis_curls_;
    if (storage.empty()) {
        throw std::logic_error("AssemblyContext::trialBasisCurl: basis curls not set");
    }
    if (j >= n_trial_dofs_ || q >= n_qpts_) {
        throw std::out_of_range("AssemblyContext::trialBasisCurl: index out of range");
    }
    return storage[static_cast<std::size_t>(j * n_qpts_ + q)];
}

Real AssemblyContext::trialBasisDivergence(LocalIndex j, LocalIndex q) const
{
    if (!trial_is_vector_basis_) {
        throw std::logic_error("AssemblyContext::trialBasisDivergence: trial space does not use a vector basis");
    }
    const auto& storage = trial_is_test_ ? test_basis_divergences_ : trial_basis_divergences_;
    if (storage.empty()) {
        throw std::logic_error("AssemblyContext::trialBasisDivergence: basis divergences not set");
    }
    if (j >= n_trial_dofs_ || q >= n_qpts_) {
        throw std::out_of_range("AssemblyContext::trialBasisDivergence: index out of range");
    }
    return storage[static_cast<std::size_t>(j * n_qpts_ + q)];
}

AssemblyContext::Vector3D AssemblyContext::trialPhysicalGradient(LocalIndex j, LocalIndex q) const
{
    if (trial_is_vector_basis_) {
        throw std::logic_error("AssemblyContext::trialPhysicalGradient: scalar basis gradients not available for vector-basis spaces");
    }
    if (j >= n_trial_dofs_ || q >= n_qpts_) {
        throw std::out_of_range("AssemblyContext::trialPhysicalGradient: index out of range");
    }

    if (trial_is_test_) {
        return test_phys_gradients_[static_cast<std::size_t>(q * n_test_dofs_ + j)];
    }
    return trial_phys_gradients_[static_cast<std::size_t>(q * n_trial_dofs_ + j)];
}

AssemblyContext::Matrix3x3 AssemblyContext::trialReferenceHessian(LocalIndex j, LocalIndex q) const
{
    if (trial_is_vector_basis_) {
        throw std::logic_error("AssemblyContext::trialReferenceHessian: scalar basis Hessians not available for vector-basis spaces");
    }
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
    if (trial_is_vector_basis_) {
        throw std::logic_error("AssemblyContext::trialPhysicalHessian: scalar basis Hessians not available for vector-basis spaces");
    }
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

    ensureArenaCapacity(std::max(n_test_dofs_, n_trial_dofs_), n_qpts_);

    solution_coefficients_.assign(coefficients.begin(),
                                  coefficients.begin() + static_cast<std::size_t>(n_trial_dofs_));

    const bool coefficients_only =
        hasFlag(required_data_, RequiredData::SolutionCoefficients) &&
        !hasFlag(required_data_, RequiredData::SolutionValues) &&
        !hasFlag(required_data_, RequiredData::SolutionGradients) &&
        !hasFlag(required_data_, RequiredData::SolutionHessians) &&
        !hasFlag(required_data_, RequiredData::SolutionLaplacians);

    if (coefficients_only) {
        solution_values_.clear();
        solution_vector_values_.clear();
        solution_gradients_.clear();
        solution_jacobians_.clear();
        solution_hessians_.clear();
        solution_laplacians_.clear();
        solution_component_hessians_.clear();
        solution_component_laplacians_.clear();
        return;
    }

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
        solution_values_.clear();
        solution_gradients_.clear();
        solution_jacobians_.clear();

        if (trial_is_vector_basis_) {
            FE_THROW_IF(need_gradients, InvalidArgumentException,
                        "AssemblyContext::setSolutionCoefficients: SolutionGradients are not available for vector-basis spaces; use curl()/div() operators instead");

            solution_vector_values_.resize(static_cast<std::size_t>(n_qpts_), Vector3D{0.0, 0.0, 0.0});
            for (LocalIndex q = 0; q < n_qpts_; ++q) {
                Vector3D u = {0.0, 0.0, 0.0};
                for (LocalIndex j = 0; j < n_trial_dofs_; ++j) {
                    const auto phi = trialBasisVectorValue(j, q);
                    const Real coef = solution_coefficients_[static_cast<std::size_t>(j)];
                    u[0] += coef * phi[0];
                    u[1] += coef * phi[1];
                    u[2] += coef * phi[2];
                }
                solution_vector_values_[static_cast<std::size_t>(q)] = u;
            }
        } else {
            FE_CHECK_ARG(trial_value_dim_ > 0 && trial_value_dim_ <= 3,
                         "AssemblyContext::setSolutionCoefficients: invalid trial value dimension");
            FE_CHECK_ARG((n_trial_dofs_ % static_cast<LocalIndex>(trial_value_dim_)) == 0,
                         "AssemblyContext::setSolutionCoefficients: trial DOFs not divisible by value_dimension");

            const LocalIndex dofs_per_component =
                static_cast<LocalIndex>(n_trial_dofs_ / static_cast<LocalIndex>(trial_value_dim_));

            // Compute vector u_h(x_q) for component-wise vector spaces.
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
        }
    } else {
        throw FEException("AssemblyContext::setSolutionCoefficients: only scalar and vector trial fields are supported",
                          __FILE__, __LINE__, __func__, FEStatus::NotImplemented);
    }

    const bool need_hessians = hasFlag(required_data_, RequiredData::SolutionHessians) ||
                               hasFlag(required_data_, RequiredData::SolutionLaplacians);
    if (need_hessians) {
        FE_THROW_IF(trial_is_vector_basis_, InvalidArgumentException,
                    "AssemblyContext::setSolutionCoefficients: SolutionHessians/Laplacians are not available for vector-basis spaces");
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

        if (trial_field_type_ == FieldType::Scalar) {
            solution_component_hessians_.clear();
            solution_component_laplacians_.clear();

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
            } else {
                solution_laplacians_.clear();
            }
        } else if (trial_field_type_ == FieldType::Vector) {
            solution_hessians_.clear();
            solution_laplacians_.clear();

            FE_CHECK_ARG(trial_value_dim_ > 0 && trial_value_dim_ <= 3,
                         "AssemblyContext::setSolutionCoefficients: invalid trial value dimension");
            FE_CHECK_ARG((n_trial_dofs_ % static_cast<LocalIndex>(trial_value_dim_)) == 0,
                         "AssemblyContext::setSolutionCoefficients: trial DOFs not divisible by value_dimension");

            const LocalIndex dofs_per_component =
                static_cast<LocalIndex>(n_trial_dofs_ / static_cast<LocalIndex>(trial_value_dim_));
            const auto total = static_cast<std::size_t>(n_qpts_) * static_cast<std::size_t>(trial_value_dim_);

            solution_component_hessians_.resize(total, Matrix3x3{});
            if (hasFlag(required_data_, RequiredData::SolutionLaplacians)) {
                solution_component_laplacians_.resize(total, 0.0);
            } else {
                solution_component_laplacians_.clear();
            }

            const int dim = dim_;
            for (LocalIndex q = 0; q < n_qpts_; ++q) {
                const auto q_base = static_cast<std::size_t>(q) * static_cast<std::size_t>(trial_value_dim_);
                for (int comp = 0; comp < trial_value_dim_; ++comp) {
                    Matrix3x3 H{};
                    const LocalIndex base = static_cast<LocalIndex>(comp) * dofs_per_component;
                    for (LocalIndex j = 0; j < dofs_per_component; ++j) {
                        const LocalIndex jj = base + j;
                        const auto Hj = trialPhysicalHessian(jj, q);
                        const Real coef = solution_coefficients_[static_cast<std::size_t>(jj)];
                        for (int r = 0; r < 3; ++r) {
                            for (int c = 0; c < 3; ++c) {
                                H[static_cast<std::size_t>(r)][static_cast<std::size_t>(c)] +=
                                    coef * Hj[static_cast<std::size_t>(r)][static_cast<std::size_t>(c)];
                            }
                        }
                    }

                    const auto idx = q_base + static_cast<std::size_t>(comp);
                    solution_component_hessians_[idx] = H;

                    if (!solution_component_laplacians_.empty()) {
                        Real lap = 0.0;
                        for (int d = 0; d < dim; ++d) {
                            lap += H[static_cast<std::size_t>(d)][static_cast<std::size_t>(d)];
                        }
                        solution_component_laplacians_[idx] = lap;
                    }
                }
            }
        } else {
            throw FEException("AssemblyContext::setSolutionCoefficients: solution Hessians are only implemented for scalar and vector fields",
                              __FILE__, __LINE__, __func__, FEStatus::NotImplemented);
        }
    } else {
        solution_hessians_.clear();
        solution_laplacians_.clear();
        solution_component_hessians_.clear();
        solution_component_laplacians_.clear();
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
        dst.component_hessians.clear();
        dst.component_laplacians.clear();

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
        dst.values.clear();
        dst.gradients.clear();
        dst.jacobians.clear();
        dst.component_hessians.clear();
        dst.component_laplacians.clear();

        if (trial_is_vector_basis_) {
            dst.vector_values.resize(static_cast<std::size_t>(n_qpts_), Vector3D{0.0, 0.0, 0.0});
            for (LocalIndex q = 0; q < n_qpts_; ++q) {
                Vector3D u = {0.0, 0.0, 0.0};
                for (LocalIndex j = 0; j < n_trial_dofs_; ++j) {
                    const auto phi = trialBasisVectorValue(j, q);
                    const Real coef = dst.coefficients[static_cast<std::size_t>(j)];
                    u[0] += coef * phi[0];
                    u[1] += coef * phi[1];
                    u[2] += coef * phi[2];
                }
                dst.vector_values[static_cast<std::size_t>(q)] = u;
            }
        } else {
            FE_CHECK_ARG(trial_value_dim_ > 0 && trial_value_dim_ <= 3,
                         "AssemblyContext::setPreviousSolutionCoefficientsK: invalid trial value dimension");
            FE_CHECK_ARG((n_trial_dofs_ % static_cast<LocalIndex>(trial_value_dim_)) == 0,
                         "AssemblyContext::setPreviousSolutionCoefficientsK: trial DOFs not divisible by value_dimension");

            const LocalIndex dofs_per_component =
                static_cast<LocalIndex>(n_trial_dofs_ / static_cast<LocalIndex>(trial_value_dim_));

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

AssemblyContext::Matrix3x3 AssemblyContext::solutionComponentHessian(LocalIndex q, int component) const
{
    if (trial_field_type_ != FieldType::Vector) {
        throw std::logic_error("AssemblyContext::solutionComponentHessian: trial field is not vector-valued");
    }
    if (solution_component_hessians_.empty()) {
        throw std::logic_error("AssemblyContext::solutionComponentHessian: solution component Hessian data not set");
    }
    if (q >= n_qpts_) {
        throw std::out_of_range("AssemblyContext::solutionComponentHessian: index out of range");
    }
    if (component < 0 || component >= trial_value_dim_) {
        throw std::out_of_range("AssemblyContext::solutionComponentHessian: component index out of range");
    }
    const auto idx = static_cast<std::size_t>(q) * static_cast<std::size_t>(trial_value_dim_) +
                     static_cast<std::size_t>(component);
    if (idx >= solution_component_hessians_.size()) {
        throw std::out_of_range("AssemblyContext::solutionComponentHessian: component indexing out of range");
    }
    return solution_component_hessians_[idx];
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

Real AssemblyContext::solutionComponentLaplacian(LocalIndex q, int component) const
{
    if (trial_field_type_ != FieldType::Vector) {
        throw std::logic_error("AssemblyContext::solutionComponentLaplacian: trial field is not vector-valued");
    }
    if (solution_component_laplacians_.empty()) {
        throw std::logic_error("AssemblyContext::solutionComponentLaplacian: solution component Laplacian data not set");
    }
    if (q >= n_qpts_) {
        throw std::out_of_range("AssemblyContext::solutionComponentLaplacian: index out of range");
    }
    if (component < 0 || component >= trial_value_dim_) {
        throw std::out_of_range("AssemblyContext::solutionComponentLaplacian: component index out of range");
    }
    const auto idx = static_cast<std::size_t>(q) * static_cast<std::size_t>(trial_value_dim_) +
                     static_cast<std::size_t>(component);
    if (idx >= solution_component_laplacians_.size()) {
        throw std::out_of_range("AssemblyContext::solutionComponentLaplacian: component indexing out of range");
    }
    return solution_component_laplacians_[idx];
}

void AssemblyContext::clearFieldSolutionData() noexcept
{
    field_solution_data_.clear();
    jit_field_solution_table_.clear();
}

void AssemblyContext::rebuildJITFieldSolutionTable()
{
    jit_field_solution_table_.clear();
    if (field_solution_data_.empty() || n_qpts_ <= 0) {
        return;
    }

    const auto nq = static_cast<std::size_t>(n_qpts_);
    jit_field_solution_table_.reserve(field_solution_data_.size());

    auto flattenXYZ = [](std::span<const Vector3D> a) noexcept -> const Real* {
        return a.empty() ? nullptr : a.data()->data();
    };
    auto flattenMat3 = [](std::span<const Matrix3x3> mats) noexcept -> const Real* {
        if (mats.empty()) return nullptr;
        return &(*mats.data())[0][0];
    };

    for (const auto& f : field_solution_data_) {
        jit::FieldSolutionEntryV1 e;
        e.field_id = static_cast<std::int32_t>(f.id);
        e.field_type = static_cast<std::uint32_t>(f.field_type);
        e.value_dim = static_cast<std::uint32_t>(f.value_dim);

        e.values = f.values.empty() ? nullptr : f.values.data();
        e.gradients_xyz = flattenXYZ(f.gradients);
        e.hessians = flattenMat3(f.hessians);
        e.laplacians = f.laplacians.empty() ? nullptr : f.laplacians.data();

        e.vector_values_xyz = flattenXYZ(f.vector_values);
        e.jacobians = flattenMat3(f.jacobians);
        e.component_hessians = flattenMat3(f.component_hessians);
        e.component_laplacians = f.component_laplacians.empty() ? nullptr : f.component_laplacians.data();

        if (f.field_type == FieldType::Scalar) {
            if (!f.history_values.empty()) {
                e.history_values = f.history_values.data();
                e.history_count = static_cast<std::uint32_t>(f.history_values.size() / nq);
            }
        } else if (f.field_type == FieldType::Vector) {
            if (!f.history_vector_values.empty()) {
                e.history_vector_values_xyz = flattenXYZ(f.history_vector_values);
                e.history_count = static_cast<std::uint32_t>(f.history_vector_values.size() / nq);
            }
        }

        jit_field_solution_table_.push_back(e);
    }
}

void AssemblyContext::setFieldSolutionScalar(FieldId field,
                                             std::span<const Real> values,
                                             std::span<const Vector3D> gradients,
                                             std::span<const Matrix3x3> hessians,
                                             std::span<const Real> laplacians)
{
    FE_THROW_IF(field == INVALID_FIELD_ID, InvalidArgumentException,
                "AssemblyContext::setFieldSolutionScalar: invalid FieldId");
    if (!values.empty() && values.size() != static_cast<std::size_t>(n_qpts_)) {
        throw std::invalid_argument("AssemblyContext::setFieldSolutionScalar: values size does not match quadrature points");
    }
    if (!gradients.empty() && gradients.size() != static_cast<std::size_t>(n_qpts_)) {
        throw std::invalid_argument("AssemblyContext::setFieldSolutionScalar: gradients size does not match quadrature points");
    }
    if (!hessians.empty() && hessians.size() != static_cast<std::size_t>(n_qpts_)) {
        throw std::invalid_argument("AssemblyContext::setFieldSolutionScalar: hessians size does not match quadrature points");
    }
    if (!laplacians.empty() && laplacians.size() != static_cast<std::size_t>(n_qpts_)) {
        throw std::invalid_argument("AssemblyContext::setFieldSolutionScalar: laplacians size does not match quadrature points");
    }

    auto it = std::find_if(field_solution_data_.begin(), field_solution_data_.end(),
                           [&](const FieldSolutionData& d) { return d.id == field; });
    if (it == field_solution_data_.end()) {
        field_solution_data_.push_back(FieldSolutionData{});
        it = std::prev(field_solution_data_.end());
        it->id = field;
    }
    if (it->field_type != FieldType::Scalar) {
        throw std::logic_error("AssemblyContext::setFieldSolutionScalar: field is already bound as vector-valued");
    }
    it->value_dim = 1;

    it->values.assign(values.begin(), values.end());
    it->gradients.assign(gradients.begin(), gradients.end());
    it->hessians.assign(hessians.begin(), hessians.end());
    it->laplacians.assign(laplacians.begin(), laplacians.end());

    it->vector_values.clear();
    it->history_vector_values.clear();
    it->jacobians.clear();
    it->component_hessians.clear();
    it->component_laplacians.clear();

    rebuildJITFieldSolutionTable();
}

void AssemblyContext::setFieldSolutionVector(FieldId field,
                                             int value_dimension,
                                             std::span<const Vector3D> values,
                                             std::span<const Matrix3x3> jacobians,
                                             std::span<const Matrix3x3> component_hessians,
                                             std::span<const Real> component_laplacians)
{
    FE_THROW_IF(field == INVALID_FIELD_ID, InvalidArgumentException,
                "AssemblyContext::setFieldSolutionVector: invalid FieldId");
    FE_THROW_IF(value_dimension <= 0 || value_dimension > 3, InvalidArgumentException,
                "AssemblyContext::setFieldSolutionVector: value_dimension must be 1..3");

    if (!values.empty() && values.size() != static_cast<std::size_t>(n_qpts_)) {
        throw std::invalid_argument("AssemblyContext::setFieldSolutionVector: values size does not match quadrature points");
    }
    if (!jacobians.empty() && jacobians.size() != static_cast<std::size_t>(n_qpts_)) {
        throw std::invalid_argument("AssemblyContext::setFieldSolutionVector: jacobians size does not match quadrature points");
    }
    const auto expected_components = static_cast<std::size_t>(n_qpts_) * static_cast<std::size_t>(value_dimension);
    if (!component_hessians.empty() && component_hessians.size() != expected_components) {
        throw std::invalid_argument(
            "AssemblyContext::setFieldSolutionVector: component_hessians size does not match quadrature points * value_dimension");
    }
    if (!component_laplacians.empty() && component_laplacians.size() != expected_components) {
        throw std::invalid_argument(
            "AssemblyContext::setFieldSolutionVector: component_laplacians size does not match quadrature points * value_dimension");
    }

    auto it = std::find_if(field_solution_data_.begin(), field_solution_data_.end(),
                           [&](const FieldSolutionData& d) { return d.id == field; });
    const bool inserted = (it == field_solution_data_.end());
    if (it == field_solution_data_.end()) {
        field_solution_data_.push_back(FieldSolutionData{});
        it = std::prev(field_solution_data_.end());
        it->id = field;
        it->field_type = FieldType::Vector;
    }
    if (it->field_type != FieldType::Vector) {
        if (inserted) {
            throw std::logic_error("AssemblyContext::setFieldSolutionVector: internal field type mismatch after insertion");
        }
        throw std::logic_error("AssemblyContext::setFieldSolutionVector: field is already bound as scalar-valued");
    }
    it->value_dim = value_dimension;

    it->vector_values.assign(values.begin(), values.end());
    it->jacobians.assign(jacobians.begin(), jacobians.end());
    it->component_hessians.assign(component_hessians.begin(), component_hessians.end());
    it->component_laplacians.assign(component_laplacians.begin(), component_laplacians.end());

    it->values.clear();
    it->history_values.clear();
    it->gradients.clear();
    it->hessians.clear();
    it->laplacians.clear();

    rebuildJITFieldSolutionTable();
}

void AssemblyContext::setFieldPreviousSolutionScalarK(FieldId field, int k, std::span<const Real> values)
{
    FE_THROW_IF(field == INVALID_FIELD_ID, InvalidArgumentException,
                "AssemblyContext::setFieldPreviousSolutionScalarK: invalid FieldId");
    FE_THROW_IF(k <= 0, InvalidArgumentException,
                "AssemblyContext::setFieldPreviousSolutionScalarK: k must be >= 1");
    if (!values.empty() && values.size() != static_cast<std::size_t>(n_qpts_)) {
        throw std::invalid_argument(
            "AssemblyContext::setFieldPreviousSolutionScalarK: values size does not match quadrature points");
    }

    auto it = std::find_if(field_solution_data_.begin(), field_solution_data_.end(),
                           [&](const FieldSolutionData& d) { return d.id == field; });
    const bool inserted = (it == field_solution_data_.end());
    if (it == field_solution_data_.end()) {
        field_solution_data_.push_back(FieldSolutionData{});
        it = std::prev(field_solution_data_.end());
        it->id = field;
        it->field_type = FieldType::Scalar;
        it->value_dim = 1;
    }
    if (it->field_type != FieldType::Scalar) {
        if (inserted) {
            throw std::logic_error(
                "AssemblyContext::setFieldPreviousSolutionScalarK: internal field type mismatch after insertion");
        }
        throw std::logic_error("AssemblyContext::setFieldPreviousSolutionScalarK: field is not scalar-valued");
    }
    it->value_dim = 1;

    it->history_vector_values.clear();

    const auto nq = static_cast<std::size_t>(n_qpts_);
    const auto needed = static_cast<std::size_t>(k) * nq;
    if (it->history_values.size() < needed) {
        it->history_values.resize(needed, 0.0);
    }
    if (!values.empty()) {
        std::copy(values.begin(), values.end(), it->history_values.begin() + (static_cast<std::size_t>(k - 1) * nq));
    }

    rebuildJITFieldSolutionTable();
}

void AssemblyContext::setFieldPreviousSolutionVectorK(FieldId field,
                                                      int k,
                                                      int value_dimension,
                                                      std::span<const Vector3D> values)
{
    FE_THROW_IF(field == INVALID_FIELD_ID, InvalidArgumentException,
                "AssemblyContext::setFieldPreviousSolutionVectorK: invalid FieldId");
    FE_THROW_IF(k <= 0, InvalidArgumentException,
                "AssemblyContext::setFieldPreviousSolutionVectorK: k must be >= 1");
    FE_THROW_IF(value_dimension <= 0 || value_dimension > 3, InvalidArgumentException,
                "AssemblyContext::setFieldPreviousSolutionVectorK: value_dimension must be 1..3");
    if (!values.empty() && values.size() != static_cast<std::size_t>(n_qpts_)) {
        throw std::invalid_argument(
            "AssemblyContext::setFieldPreviousSolutionVectorK: values size does not match quadrature points");
    }

    auto it = std::find_if(field_solution_data_.begin(), field_solution_data_.end(),
                           [&](const FieldSolutionData& d) { return d.id == field; });
    const bool inserted = (it == field_solution_data_.end());
    if (it == field_solution_data_.end()) {
        field_solution_data_.push_back(FieldSolutionData{});
        it = std::prev(field_solution_data_.end());
        it->id = field;
        it->field_type = FieldType::Vector;
        it->value_dim = value_dimension;
    }
    if (it->field_type != FieldType::Vector) {
        if (inserted) {
            throw std::logic_error(
                "AssemblyContext::setFieldPreviousSolutionVectorK: internal field type mismatch after insertion");
        }
        throw std::logic_error("AssemblyContext::setFieldPreviousSolutionVectorK: field is not vector-valued");
    }
    it->value_dim = value_dimension;

    it->history_values.clear();

    const auto nq = static_cast<std::size_t>(n_qpts_);
    const auto needed = static_cast<std::size_t>(k) * nq;
    if (it->history_vector_values.size() < needed) {
        it->history_vector_values.resize(needed, Vector3D{0.0, 0.0, 0.0});
    }
    if (!values.empty()) {
        std::copy(values.begin(), values.end(),
                  it->history_vector_values.begin() + (static_cast<std::size_t>(k - 1) * nq));
    }

    rebuildJITFieldSolutionTable();
}

bool AssemblyContext::hasFieldSolutionData(FieldId field) const noexcept
{
    const auto it = std::find_if(field_solution_data_.begin(), field_solution_data_.end(),
                                 [&](const FieldSolutionData& d) { return d.id == field; });
    return it != field_solution_data_.end();
}

FieldType AssemblyContext::fieldSolutionFieldType(FieldId field) const
{
    const auto it = std::find_if(field_solution_data_.begin(), field_solution_data_.end(),
                                 [&](const FieldSolutionData& d) { return d.id == field; });
    if (it == field_solution_data_.end()) {
        throw std::logic_error("AssemblyContext::fieldSolutionFieldType: field solution data not set");
    }
    return it->field_type;
}

int AssemblyContext::fieldSolutionValueDimension(FieldId field) const
{
    const auto it = std::find_if(field_solution_data_.begin(), field_solution_data_.end(),
                                 [&](const FieldSolutionData& d) { return d.id == field; });
    if (it == field_solution_data_.end()) {
        throw std::logic_error("AssemblyContext::fieldSolutionValueDimension: field solution data not set");
    }
    return it->value_dim;
}

Real AssemblyContext::fieldValue(FieldId field, LocalIndex q) const
{
    const auto it = std::find_if(field_solution_data_.begin(), field_solution_data_.end(),
                                 [&](const FieldSolutionData& d) { return d.id == field; });
    if (it == field_solution_data_.end() || it->values.empty()) {
        throw std::logic_error("AssemblyContext::fieldValue: field value data not set");
    }
    if (it->field_type != FieldType::Scalar) {
        throw std::logic_error("AssemblyContext::fieldValue: field is not scalar-valued");
    }
    if (q >= n_qpts_) {
        throw std::out_of_range("AssemblyContext::fieldValue: index out of range");
    }
    return it->values[static_cast<std::size_t>(q)];
}

AssemblyContext::Vector3D AssemblyContext::fieldVectorValue(FieldId field, LocalIndex q) const
{
    const auto it = std::find_if(field_solution_data_.begin(), field_solution_data_.end(),
                                 [&](const FieldSolutionData& d) { return d.id == field; });
    if (it == field_solution_data_.end() || it->vector_values.empty()) {
        throw std::logic_error("AssemblyContext::fieldVectorValue: field vector value data not set");
    }
    if (it->field_type != FieldType::Vector) {
        throw std::logic_error("AssemblyContext::fieldVectorValue: field is not vector-valued");
    }
    if (q >= n_qpts_) {
        throw std::out_of_range("AssemblyContext::fieldVectorValue: index out of range");
    }
    return it->vector_values[static_cast<std::size_t>(q)];
}

AssemblyContext::Vector3D AssemblyContext::fieldGradient(FieldId field, LocalIndex q) const
{
    const auto it = std::find_if(field_solution_data_.begin(), field_solution_data_.end(),
                                 [&](const FieldSolutionData& d) { return d.id == field; });
    if (it == field_solution_data_.end() || it->gradients.empty()) {
        throw std::logic_error("AssemblyContext::fieldGradient: field gradient data not set");
    }
    if (it->field_type != FieldType::Scalar) {
        throw std::logic_error("AssemblyContext::fieldGradient: field is not scalar-valued");
    }
    if (q >= n_qpts_) {
        throw std::out_of_range("AssemblyContext::fieldGradient: index out of range");
    }
    return it->gradients[static_cast<std::size_t>(q)];
}

AssemblyContext::Matrix3x3 AssemblyContext::fieldJacobian(FieldId field, LocalIndex q) const
{
    const auto it = std::find_if(field_solution_data_.begin(), field_solution_data_.end(),
                                 [&](const FieldSolutionData& d) { return d.id == field; });
    if (it == field_solution_data_.end() || it->jacobians.empty()) {
        throw std::logic_error("AssemblyContext::fieldJacobian: field Jacobian data not set");
    }
    if (it->field_type != FieldType::Vector) {
        throw std::logic_error("AssemblyContext::fieldJacobian: field is not vector-valued");
    }
    if (q >= n_qpts_) {
        throw std::out_of_range("AssemblyContext::fieldJacobian: index out of range");
    }
    return it->jacobians[static_cast<std::size_t>(q)];
}

AssemblyContext::Matrix3x3 AssemblyContext::fieldHessian(FieldId field, LocalIndex q) const
{
    const auto it = std::find_if(field_solution_data_.begin(), field_solution_data_.end(),
                                 [&](const FieldSolutionData& d) { return d.id == field; });
    if (it == field_solution_data_.end() || it->hessians.empty()) {
        throw std::logic_error("AssemblyContext::fieldHessian: field Hessian data not set");
    }
    if (it->field_type != FieldType::Scalar) {
        throw std::logic_error("AssemblyContext::fieldHessian: field is not scalar-valued");
    }
    if (q >= n_qpts_) {
        throw std::out_of_range("AssemblyContext::fieldHessian: index out of range");
    }
    return it->hessians[static_cast<std::size_t>(q)];
}

Real AssemblyContext::fieldLaplacian(FieldId field, LocalIndex q) const
{
    const auto it = std::find_if(field_solution_data_.begin(), field_solution_data_.end(),
                                 [&](const FieldSolutionData& d) { return d.id == field; });
    if (it == field_solution_data_.end() || it->laplacians.empty()) {
        throw std::logic_error("AssemblyContext::fieldLaplacian: field Laplacian data not set");
    }
    if (it->field_type != FieldType::Scalar) {
        throw std::logic_error("AssemblyContext::fieldLaplacian: field is not scalar-valued");
    }
    if (q >= n_qpts_) {
        throw std::out_of_range("AssemblyContext::fieldLaplacian: index out of range");
    }
    return it->laplacians[static_cast<std::size_t>(q)];
}

AssemblyContext::Matrix3x3 AssemblyContext::fieldComponentHessian(FieldId field, LocalIndex q, int component) const
{
    const auto it = std::find_if(field_solution_data_.begin(), field_solution_data_.end(),
                                 [&](const FieldSolutionData& d) { return d.id == field; });
    if (it == field_solution_data_.end() || it->component_hessians.empty()) {
        throw std::logic_error("AssemblyContext::fieldComponentHessian: field component Hessian data not set");
    }
    if (it->field_type != FieldType::Vector) {
        throw std::logic_error("AssemblyContext::fieldComponentHessian: field is not vector-valued");
    }
    if (q >= n_qpts_) {
        throw std::out_of_range("AssemblyContext::fieldComponentHessian: index out of range");
    }
    if (component < 0 || component >= it->value_dim) {
        throw std::out_of_range("AssemblyContext::fieldComponentHessian: component index out of range");
    }
    const auto idx = static_cast<std::size_t>(q) * static_cast<std::size_t>(it->value_dim) +
                     static_cast<std::size_t>(component);
    if (idx >= it->component_hessians.size()) {
        throw std::out_of_range("AssemblyContext::fieldComponentHessian: component indexing out of range");
    }
    return it->component_hessians[idx];
}

Real AssemblyContext::fieldComponentLaplacian(FieldId field, LocalIndex q, int component) const
{
    const auto it = std::find_if(field_solution_data_.begin(), field_solution_data_.end(),
                                 [&](const FieldSolutionData& d) { return d.id == field; });
    if (it == field_solution_data_.end() || it->component_laplacians.empty()) {
        throw std::logic_error("AssemblyContext::fieldComponentLaplacian: field component Laplacian data not set");
    }
    if (it->field_type != FieldType::Vector) {
        throw std::logic_error("AssemblyContext::fieldComponentLaplacian: field is not vector-valued");
    }
    if (q >= n_qpts_) {
        throw std::out_of_range("AssemblyContext::fieldComponentLaplacian: index out of range");
    }
    if (component < 0 || component >= it->value_dim) {
        throw std::out_of_range("AssemblyContext::fieldComponentLaplacian: component index out of range");
    }
    const auto idx = static_cast<std::size_t>(q) * static_cast<std::size_t>(it->value_dim) +
                     static_cast<std::size_t>(component);
    if (idx >= it->component_laplacians.size()) {
        throw std::out_of_range("AssemblyContext::fieldComponentLaplacian: component indexing out of range");
    }
    return it->component_laplacians[idx];
}

Real AssemblyContext::fieldPreviousValue(FieldId field, LocalIndex q, int k) const
{
    if (k <= 0) {
        throw std::out_of_range("AssemblyContext::fieldPreviousValue: history index k must be >= 1");
    }
    const auto it = std::find_if(field_solution_data_.begin(), field_solution_data_.end(),
                                 [&](const FieldSolutionData& d) { return d.id == field; });
    if (it == field_solution_data_.end() || it->history_values.empty()) {
        throw std::logic_error("AssemblyContext::fieldPreviousValue: field previous value data not set");
    }
    if (it->field_type != FieldType::Scalar) {
        throw std::logic_error("AssemblyContext::fieldPreviousValue: field is not scalar-valued");
    }
    if (q >= n_qpts_) {
        throw std::out_of_range("AssemblyContext::fieldPreviousValue: index out of range");
    }

    const auto nq = static_cast<std::size_t>(n_qpts_);
    const auto idx = static_cast<std::size_t>(k - 1) * nq + static_cast<std::size_t>(q);
    if (idx >= it->history_values.size()) {
        throw std::out_of_range("AssemblyContext::fieldPreviousValue: history index out of range");
    }
    return it->history_values[idx];
}

AssemblyContext::Vector3D AssemblyContext::fieldPreviousVectorValue(FieldId field, LocalIndex q, int k) const
{
    if (k <= 0) {
        throw std::out_of_range("AssemblyContext::fieldPreviousVectorValue: history index k must be >= 1");
    }
    const auto it = std::find_if(field_solution_data_.begin(), field_solution_data_.end(),
                                 [&](const FieldSolutionData& d) { return d.id == field; });
    if (it == field_solution_data_.end() || it->history_vector_values.empty()) {
        throw std::logic_error("AssemblyContext::fieldPreviousVectorValue: field previous vector value data not set");
    }
    if (it->field_type != FieldType::Vector) {
        throw std::logic_error("AssemblyContext::fieldPreviousVectorValue: field is not vector-valued");
    }
    if (q >= n_qpts_) {
        throw std::out_of_range("AssemblyContext::fieldPreviousVectorValue: index out of range");
    }

    const auto nq = static_cast<std::size_t>(n_qpts_);
    const auto idx = static_cast<std::size_t>(k - 1) * nq + static_cast<std::size_t>(q);
    if (idx >= it->history_vector_values.size()) {
        throw std::out_of_range("AssemblyContext::fieldPreviousVectorValue: history index out of range");
    }
    return it->history_vector_values[idx];
}

// ============================================================================
// Material State
// ============================================================================

void AssemblyContext::setMaterialState(std::byte* cell_state_base,
                                       std::size_t bytes_per_qpt,
                                       std::size_t stride_bytes,
                                       std::size_t alignment_bytes) noexcept
{
    setMaterialState(cell_state_base, cell_state_base, bytes_per_qpt, stride_bytes, alignment_bytes);
}

void AssemblyContext::setMaterialState(std::byte* cell_state_old_base,
                                       std::byte* cell_state_work_base,
                                       std::size_t bytes_per_qpt,
                                       std::size_t stride_bytes,
                                       std::size_t alignment_bytes) noexcept
{
    material_state_old_base_ = (cell_state_old_base != nullptr) ? cell_state_old_base : cell_state_work_base;
    material_state_work_base_ = cell_state_work_base;
    material_state_bytes_per_qpt_ = bytes_per_qpt;
    material_state_stride_bytes_ = stride_bytes;
    material_state_alignment_bytes_ = alignment_bytes;
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

void AssemblyContext::rebuildInterleavedQPointGeometry()
{
    const auto n_qpts = static_cast<std::size_t>(n_qpts_);
    if (n_qpts == 0u) {
        interleaved_qpoint_geometry_.clear();
        return;
    }

    const std::size_t stride =
        static_cast<std::size_t>(AssemblyContext::kInterleavedQPointGeometryStride);
    interleaved_qpoint_geometry_.assign(n_qpts * stride, Real(0.0));

    const auto write_vec3 = [&](std::size_t base,
                                const std::array<Real, 3>& v,
                                std::size_t offset) {
        interleaved_qpoint_geometry_[base + offset + 0u] = v[0];
        interleaved_qpoint_geometry_[base + offset + 1u] = v[1];
        interleaved_qpoint_geometry_[base + offset + 2u] = v[2];
    };

    for (std::size_t q = 0; q < n_qpts; ++q) {
        const std::size_t base = q * stride;

        if (q < physical_points_.size()) {
            write_vec3(base,
                       physical_points_[q],
                       static_cast<std::size_t>(AssemblyContext::kInterleavedQPointPhysicalOffset));
        }

        if (q < jacobians_.size()) {
            const auto& J = jacobians_[q];
            const std::size_t off = base + static_cast<std::size_t>(AssemblyContext::kInterleavedQPointJacobianOffset);
            for (std::size_t r = 0; r < 3u; ++r) {
                for (std::size_t c = 0; c < 3u; ++c) {
                    interleaved_qpoint_geometry_[off + r * 3u + c] = J[r][c];
                }
            }
        }

        if (q < inverse_jacobians_.size()) {
            const auto& Jinv = inverse_jacobians_[q];
            const std::size_t off =
                base + static_cast<std::size_t>(AssemblyContext::kInterleavedQPointInverseJacobianOffset);
            for (std::size_t r = 0; r < 3u; ++r) {
                for (std::size_t c = 0; c < 3u; ++c) {
                    interleaved_qpoint_geometry_[off + r * 3u + c] = Jinv[r][c];
                }
            }
        }

        if (q < jacobian_dets_.size()) {
            interleaved_qpoint_geometry_[base + static_cast<std::size_t>(AssemblyContext::kInterleavedQPointDetOffset)] =
                jacobian_dets_[q];
        }

        if (q < normals_.size()) {
            write_vec3(base,
                       normals_[q],
                       static_cast<std::size_t>(AssemblyContext::kInterleavedQPointNormalOffset));
        }
    }
}

void AssemblyContext::setQuadratureData(
    std::span<const Point3D> points,
    std::span<const Real> weights)
{
    if (points.size() != weights.size()) {
        throw std::invalid_argument("AssemblyContext::setQuadratureData: size mismatch");
    }

    ensureArenaCapacity(std::max(n_test_dofs_, n_trial_dofs_),
                        static_cast<LocalIndex>(points.size()));

    n_qpts_ = static_cast<LocalIndex>(points.size());
    quad_points_.assign(points.begin(), points.end());
    quad_weights_.assign(weights.begin(), weights.end());
    rebuildInterleavedQPointGeometry();
}

void AssemblyContext::setPhysicalPoints(std::span<const Point3D> points)
{
    ensureArenaCapacity(std::max(n_test_dofs_, n_trial_dofs_),
                        static_cast<LocalIndex>(std::max<std::size_t>(points.size(),
                                                                      static_cast<std::size_t>(n_qpts_))));
    physical_points_.assign(points.begin(), points.end());
    rebuildInterleavedQPointGeometry();
}

void AssemblyContext::setJacobianData(
    std::span<const Matrix3x3> jacobians,
    std::span<const Matrix3x3> inverse_jacobians,
    std::span<const Real> determinants)
{
    const auto required_qpts = static_cast<LocalIndex>(
        std::max({static_cast<std::size_t>(n_qpts_),
                  jacobians.size(),
                  inverse_jacobians.size(),
                  determinants.size()}));
    ensureArenaCapacity(std::max(n_test_dofs_, n_trial_dofs_), required_qpts);

    jacobians_.assign(jacobians.begin(), jacobians.end());
    inverse_jacobians_.assign(inverse_jacobians.begin(), inverse_jacobians.end());
    jacobian_dets_.assign(determinants.begin(), determinants.end());
    rebuildInterleavedQPointGeometry();
}

void AssemblyContext::setIntegrationWeights(std::span<const Real> weights)
{
    ensureArenaCapacity(std::max(n_test_dofs_, n_trial_dofs_),
                        static_cast<LocalIndex>(std::max<std::size_t>(weights.size(),
                                                                      static_cast<std::size_t>(n_qpts_))));
    integration_weights_.assign(weights.begin(), weights.end());
}

void AssemblyContext::setTestBasisData(
    LocalIndex n_dofs,
    std::span<const Real> values,
    std::span<const Vector3D> gradients)
{
    if (test_is_vector_basis_) {
        throw std::logic_error("AssemblyContext::setTestBasisData: scalar basis data not valid for vector-basis spaces");
    }
    n_test_dofs_ = n_dofs;
    if (trial_is_test_) {
        n_trial_dofs_ = n_dofs;
    }
    ensureArenaCapacity(std::max(n_test_dofs_, n_trial_dofs_), n_qpts_);
    test_basis_values_.assign(values.begin(), values.end());
    test_ref_gradients_.assign(gradients.begin(), gradients.end());
    test_basis_vector_values_.clear();
    test_basis_curls_.clear();
    test_basis_divergences_.clear();
}

void AssemblyContext::setTrialBasisData(
    LocalIndex n_dofs,
    std::span<const Real> values,
    std::span<const Vector3D> gradients)
{
    if (trial_is_vector_basis_) {
        throw std::logic_error("AssemblyContext::setTrialBasisData: scalar basis data not valid for vector-basis spaces");
    }
    n_trial_dofs_ = n_dofs;
    trial_is_test_ = false;
    ensureArenaCapacity(std::max(n_test_dofs_, n_trial_dofs_), n_qpts_);
    trial_basis_values_.assign(values.begin(), values.end());
    trial_ref_gradients_.assign(gradients.begin(), gradients.end());
    trial_basis_vector_values_.clear();
    trial_basis_curls_.clear();
    trial_basis_divergences_.clear();
}

void AssemblyContext::setTestVectorBasisValues(LocalIndex n_dofs, std::span<const Vector3D> values)
{
    if (!test_is_vector_basis_) {
        throw std::logic_error("AssemblyContext::setTestVectorBasisValues: test space does not use a vector basis");
    }
    n_test_dofs_ = n_dofs;
    if (trial_is_test_) {
        n_trial_dofs_ = n_dofs;
    }
    ensureArenaCapacity(std::max(n_test_dofs_, n_trial_dofs_), n_qpts_);
    test_basis_vector_values_.assign(values.begin(), values.end());
    test_basis_values_.clear();
    test_ref_gradients_.clear();
    test_phys_gradients_.clear();
}

void AssemblyContext::setTrialVectorBasisValues(LocalIndex n_dofs, std::span<const Vector3D> values)
{
    if (!trial_is_vector_basis_) {
        throw std::logic_error("AssemblyContext::setTrialVectorBasisValues: trial space does not use a vector basis");
    }
    n_trial_dofs_ = n_dofs;
    trial_is_test_ = false;
    ensureArenaCapacity(std::max(n_test_dofs_, n_trial_dofs_), n_qpts_);
    trial_basis_vector_values_.assign(values.begin(), values.end());
    trial_basis_values_.clear();
    trial_ref_gradients_.clear();
    trial_phys_gradients_.clear();
}

void AssemblyContext::setTestBasisCurls(LocalIndex n_dofs, std::span<const Vector3D> curls)
{
    (void)n_dofs;
    if (!test_is_vector_basis_) {
        throw std::logic_error("AssemblyContext::setTestBasisCurls: test space does not use a vector basis");
    }
    ensureArenaCapacity(std::max(n_test_dofs_, n_trial_dofs_), n_qpts_);
    test_basis_curls_.assign(curls.begin(), curls.end());
}

void AssemblyContext::setTrialBasisCurls(LocalIndex n_dofs, std::span<const Vector3D> curls)
{
    (void)n_dofs;
    if (!trial_is_vector_basis_) {
        throw std::logic_error("AssemblyContext::setTrialBasisCurls: trial space does not use a vector basis");
    }
    trial_is_test_ = false;
    ensureArenaCapacity(std::max(n_test_dofs_, n_trial_dofs_), n_qpts_);
    trial_basis_curls_.assign(curls.begin(), curls.end());
}

void AssemblyContext::setTestBasisDivergences(LocalIndex n_dofs, std::span<const Real> divergences)
{
    (void)n_dofs;
    if (!test_is_vector_basis_) {
        throw std::logic_error("AssemblyContext::setTestBasisDivergences: test space does not use a vector basis");
    }
    ensureArenaCapacity(std::max(n_test_dofs_, n_trial_dofs_), n_qpts_);
    test_basis_divergences_.assign(divergences.begin(), divergences.end());
}

void AssemblyContext::setTrialBasisDivergences(LocalIndex n_dofs, std::span<const Real> divergences)
{
    (void)n_dofs;
    if (!trial_is_vector_basis_) {
        throw std::logic_error("AssemblyContext::setTrialBasisDivergences: trial space does not use a vector basis");
    }
    trial_is_test_ = false;
    ensureArenaCapacity(std::max(n_test_dofs_, n_trial_dofs_), n_qpts_);
    trial_basis_divergences_.assign(divergences.begin(), divergences.end());
}

void AssemblyContext::setTestBasisHessians(
    LocalIndex n_dofs,
    std::span<const Matrix3x3> hessians)
{
    (void)n_dofs;
    if (test_is_vector_basis_) {
        throw std::logic_error("AssemblyContext::setTestBasisHessians: scalar basis Hessians not valid for vector-basis spaces");
    }
    ensureArenaCapacity(std::max(n_test_dofs_, n_trial_dofs_), n_qpts_);
    test_ref_hessians_.assign(hessians.begin(), hessians.end());
}

void AssemblyContext::setTrialBasisHessians(
    LocalIndex n_dofs,
    std::span<const Matrix3x3> hessians)
{
    (void)n_dofs;
    if (trial_is_vector_basis_) {
        throw std::logic_error("AssemblyContext::setTrialBasisHessians: scalar basis Hessians not valid for vector-basis spaces");
    }
    ensureArenaCapacity(std::max(n_test_dofs_, n_trial_dofs_), n_qpts_);
    trial_ref_hessians_.assign(hessians.begin(), hessians.end());
}

void AssemblyContext::setPhysicalGradients(
    std::span<const Vector3D> test_gradients,
    std::span<const Vector3D> trial_gradients)
{
    const auto expected_test = static_cast<std::size_t>(n_test_dofs_) * static_cast<std::size_t>(n_qpts_);
    if (!test_gradients.empty() && test_gradients.size() != expected_test) {
        throw std::invalid_argument("AssemblyContext::setPhysicalGradients: test gradient size mismatch");
    }
    ensureArenaCapacity(std::max(n_test_dofs_, n_trial_dofs_), n_qpts_);
    test_phys_gradients_.assign(test_gradients.begin(), test_gradients.end());

    if (!trial_is_test_) {
        const auto expected_trial = static_cast<std::size_t>(n_trial_dofs_) * static_cast<std::size_t>(n_qpts_);
        if (!trial_gradients.empty() && trial_gradients.size() != expected_trial) {
            throw std::invalid_argument("AssemblyContext::setPhysicalGradients: trial gradient size mismatch");
        }
        ensureArenaCapacity(std::max(n_test_dofs_, n_trial_dofs_), n_qpts_);
        trial_phys_gradients_.assign(trial_gradients.begin(), trial_gradients.end());
    }
}

void AssemblyContext::setPhysicalHessians(
    std::span<const Matrix3x3> test_hessians,
    std::span<const Matrix3x3> trial_hessians)
{
    ensureArenaCapacity(std::max(n_test_dofs_, n_trial_dofs_), n_qpts_);
    test_phys_hessians_.assign(test_hessians.begin(), test_hessians.end());
    if (!trial_is_test_) {
        ensureArenaCapacity(std::max(n_test_dofs_, n_trial_dofs_), n_qpts_);
        trial_phys_hessians_.assign(trial_hessians.begin(), trial_hessians.end());
    }
}

void AssemblyContext::setNormals(std::span<const Vector3D> normals)
{
    ensureArenaCapacity(std::max(n_test_dofs_, n_trial_dofs_),
                        static_cast<LocalIndex>(std::max<std::size_t>(normals.size(),
                                                                      static_cast<std::size_t>(n_qpts_))));
    normals_.assign(normals.begin(), normals.end());
    rebuildInterleavedQPointGeometry();
}

void AssemblyContext::setSolutionValues(std::span<const Real> values)
{
    ensureArenaCapacity(std::max(n_test_dofs_, n_trial_dofs_),
                        static_cast<LocalIndex>(std::max<std::size_t>(values.size(),
                                                                      static_cast<std::size_t>(n_qpts_))));
    solution_values_.assign(values.begin(), values.end());
}

void AssemblyContext::setSolutionGradients(std::span<const Vector3D> gradients)
{
    ensureArenaCapacity(std::max(n_test_dofs_, n_trial_dofs_),
                        static_cast<LocalIndex>(std::max<std::size_t>(gradients.size(),
                                                                      static_cast<std::size_t>(n_qpts_))));
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
