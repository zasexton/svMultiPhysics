/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#include "RobinBC.h"

#include <cmath>

namespace svmp {
namespace FE {
namespace constraints {

// ============================================================================
// RobinBC - Construction
// ============================================================================

RobinBC::RobinBC() = default;

RobinBC::RobinBC(double alpha, double beta, double g, const RobinBCOptions& options)
    : options_(options)
{
    constant_data_.alpha = alpha;
    constant_data_.beta = beta;
    constant_data_.g = g;
}

RobinBC::RobinBC(RobinFunction func, const RobinBCOptions& options)
    : robin_func_(std::move(func)), options_(options) {}

RobinBC::~RobinBC() = default;

RobinBC::RobinBC(const RobinBC& other) = default;
RobinBC::RobinBC(RobinBC&& other) noexcept = default;
RobinBC& RobinBC::operator=(const RobinBC& other) = default;
RobinBC& RobinBC::operator=(RobinBC&& other) noexcept = default;

// ============================================================================
// Setup
// ============================================================================

void RobinBC::setConstant(double alpha, double beta, double g) {
    constant_data_.alpha = alpha;
    constant_data_.beta = beta;
    constant_data_.g = g;
    robin_func_ = nullptr;
}

void RobinBC::setFunction(RobinFunction func) {
    robin_func_ = std::move(func);
}

// ============================================================================
// Evaluation
// ============================================================================

RobinData RobinBC::evaluate(double x, double y, double z, double time) const {
    if (robin_func_) {
        return robin_func_(x, y, z, time);
    }
    return constant_data_;
}

double RobinBC::evaluateAlpha(double x, double y, double z, double time) const {
    return evaluate(x, y, z, time).alpha;
}

double RobinBC::evaluateG(double x, double y, double z, double time) const {
    return evaluate(x, y, z, time).g;
}

std::vector<RobinData> RobinBC::evaluateMultiple(
    std::span<const std::array<double, 3>> coords,
    double time) const
{
    std::vector<RobinData> results;
    results.reserve(coords.size());

    for (const auto& pt : coords) {
        results.push_back(evaluate(pt[0], pt[1], pt[2], time));
    }

    return results;
}

// ============================================================================
// Assembly helpers
// ============================================================================

void RobinBC::computeMatrixContribution(
    std::span<const double> shape_values,
    double alpha,
    double weight,
    std::span<double> local_matrix)
{
    std::size_t n_dofs = shape_values.size();

    if (local_matrix.size() < n_dofs * n_dofs) {
        CONSTRAINT_THROW("Local matrix too small");
    }

    // Add alpha * weight * N_i * N_j to matrix
    double alpha_w = alpha * weight;

    for (std::size_t i = 0; i < n_dofs; ++i) {
        for (std::size_t j = 0; j < n_dofs; ++j) {
            local_matrix[i * n_dofs + j] += alpha_w * shape_values[i] * shape_values[j];
        }
    }
}

void RobinBC::computeRhsContribution(
    std::span<const double> shape_values,
    double g,
    double weight,
    std::span<double> local_rhs)
{
    std::size_t n_dofs = shape_values.size();

    if (local_rhs.size() < n_dofs) {
        CONSTRAINT_THROW("Local RHS too small");
    }

    // Add g * weight * N_i to RHS
    double g_w = g * weight;

    for (std::size_t i = 0; i < n_dofs; ++i) {
        local_rhs[i] += g_w * shape_values[i];
    }
}

// ============================================================================
// Static factory methods
// ============================================================================

RobinBC RobinBC::convective(double heat_transfer_coeff,
                             double ambient_temperature,
                             const RobinBCOptions& options) {
    // Robin form: h*T - h*T_inf = -q_n (Neumann from flux)
    // In standard Robin: alpha*u + beta*du/dn = g
    // Here: h*T + (-1)*(-q) = h*T_inf
    // So: alpha = h, beta = -1, g = h*T_inf
    // Or equivalently for natural BC integration: alpha = h, g = h*T_inf
    return RobinBC(heat_transfer_coeff, 1.0, heat_transfer_coeff * ambient_temperature, options);
}

RobinBC RobinBC::absorbing(double wave_speed, const RobinBCOptions& options) {
    // First-order absorbing BC: du/dt + c * du/dn = 0
    // In frequency domain or after time discretization, this becomes
    // a Robin-type condition. For simplicity, we use: c*u + du/dn = 0
    return RobinBC(wave_speed, 1.0, 0.0, options);
}

RobinBC RobinBC::elasticFoundation(double spring_constant,
                                    const RobinBCOptions& options) {
    // Winkler foundation: k*u + sigma_n = 0
    // Robin form: k*u + 1*du/dn = 0
    return RobinBC(spring_constant, 1.0, 0.0, options);
}

RobinBC RobinBC::impedance(double impedance, double rhs,
                           const RobinBCOptions& options) {
    // Impedance BC: Z*u + du/dn = g
    return RobinBC(impedance, 1.0, rhs, options);
}

// ============================================================================
// VectorRobinBC - Construction
// ============================================================================

VectorRobinBC::VectorRobinBC() = default;

VectorRobinBC::VectorRobinBC(const VectorRobinData& data, const RobinBCOptions& options)
    : constant_data_(data), options_(options) {}

VectorRobinBC::VectorRobinBC(VectorRobinFunction func, const RobinBCOptions& options)
    : robin_func_(std::move(func)), options_(options) {}

VectorRobinBC::~VectorRobinBC() = default;

VectorRobinBC::VectorRobinBC(const VectorRobinBC& other) = default;
VectorRobinBC::VectorRobinBC(VectorRobinBC&& other) noexcept = default;
VectorRobinBC& VectorRobinBC::operator=(const VectorRobinBC& other) = default;
VectorRobinBC& VectorRobinBC::operator=(VectorRobinBC&& other) noexcept = default;

VectorRobinData VectorRobinBC::evaluate(double x, double y, double z, double time) const {
    if (robin_func_) {
        return robin_func_(x, y, z, time);
    }
    return constant_data_;
}

} // namespace constraints
} // namespace FE
} // namespace svmp
