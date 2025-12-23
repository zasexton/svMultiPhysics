#include "Legacy/AdaptivityStrategies.hpp"
#include <cmath>
#include <algorithm>
#include <numeric>
#include <Eigen/Dense>
#include <Eigen/Eigenvalues>

namespace mesh {

//=============================================================================
// GoalOrientedAdaptivity Implementation
//=============================================================================

GoalOrientedAdaptivity::GoalOrientedAdaptivity(const GoalOrientedConfig& config)
    : config_(config) {}

double GoalOrientedAdaptivity::compute_goal_functional(
    const MeshBase& mesh,
    const MeshFields& fields) const {

    double functional_value = 0.0;

    // Example: compute various goal functionals
    if (config_.goal_functional == "drag") {
        // Integrate pressure and viscous forces on boundary
        for (size_t i = 0; i < mesh.n_boundary_faces(); ++i) {
            // Get boundary normal and area
            auto normal = mesh.get_face_normal(i);
            double area = mesh.get_face_area(i);

            // Get pressure and shear stress
            double pressure = fields.get_field_value("pressure", i);
            auto shear_stress = fields.get_field_gradient("velocity", i);

            // Integrate force component
            functional_value += pressure * normal[0] * area;  // Pressure drag
            // Add viscous drag component
        }
    } else if (config_.goal_functional == "lift") {
        // Similar to drag but in perpendicular direction
        for (size_t i = 0; i < mesh.n_boundary_faces(); ++i) {
            auto normal = mesh.get_face_normal(i);
            double area = mesh.get_face_area(i);
            double pressure = fields.get_field_value("pressure", i);
            functional_value += pressure * normal[1] * area;  // Lift force
        }
    } else if (config_.goal_functional == "point_value") {
        // Evaluate solution at specific point
        // Find element containing target point and interpolate
        functional_value = fields.get_field_value("solution", 0);  // Simplified
    } else if (config_.goal_functional == "average") {
        // Average value over domain
        double total_volume = 0.0;
        for (size_t i = 0; i < mesh.n_elements(); ++i) {
            double volume = mesh.get_element_volume(i);
            double value = fields.get_field_value("solution", i);
            functional_value += value * volume;
            total_volume += volume;
        }
        functional_value /= total_volume;
    }

    return functional_value;
}

void GoalOrientedAdaptivity::solve_dual_problem(
    const MeshBase& mesh,
    const MeshFields& primal_fields,
    MeshFields& dual_fields) const {

    size_t n_dofs = mesh.n_nodes();

    // Initialize dual solution
    dual_fields.initialize("dual", n_dofs);

    // Set up dual problem: adjoint operator with goal functional as forcing
    // L*(u*) = dJ/du where L* is adjoint operator

    if (!config_.dual_config.solve_dual_problem) {
        // Use simplified dual solution (identity weights)
        for (size_t i = 0; i < n_dofs; ++i) {
            dual_fields.set_field_value("dual", i, 1.0);
        }
        return;
    }

    // Build adjoint system matrix (transpose of primal Jacobian)
    // This is problem-dependent - example for diffusion problem
    std::vector<std::vector<double>> adjoint_matrix(n_dofs,
                                                     std::vector<double>(n_dofs, 0.0));
    std::vector<double> adjoint_rhs(n_dofs, 0.0);

    // Compute adjoint forcing: derivative of functional w.r.t. solution
    for (size_t elem = 0; elem < mesh.n_elements(); ++elem) {
        auto nodes = mesh.get_element_nodes(elem);
        double volume = mesh.get_element_volume(elem);

        // Add contribution from functional derivative
        if (config_.goal_functional == "average") {
            double total_volume = mesh.get_total_volume();
            for (auto node : nodes) {
                adjoint_rhs[node] += volume / total_volume / nodes.size();
            }
        }

        // Build adjoint operator (transpose of primal operator)
        // For steady diffusion: -∇²u* = dJ/du
        // Assemble stiffness matrix transpose
        for (size_t i = 0; i < nodes.size(); ++i) {
            for (size_t j = 0; j < nodes.size(); ++j) {
                // Simplified stiffness matrix entry
                double stiffness = volume / (nodes.size() * nodes.size());
                adjoint_matrix[nodes[j]][nodes[i]] += stiffness;  // Note transpose
            }
        }
    }

    // Solve adjoint system using iterative method
    std::vector<double> dual_solution(n_dofs, 0.0);
    std::vector<double> residual(n_dofs);

    for (int iter = 0; iter < config_.dual_config.max_dual_iterations; ++iter) {
        // Compute residual: r = b - A*x
        for (size_t i = 0; i < n_dofs; ++i) {
            residual[i] = adjoint_rhs[i];
            for (size_t j = 0; j < n_dofs; ++j) {
                residual[i] -= adjoint_matrix[i][j] * dual_solution[j];
            }
        }

        // Check convergence
        double res_norm = 0.0;
        for (double r : residual) res_norm += r * r;
        res_norm = std::sqrt(res_norm);

        if (res_norm < config_.dual_config.dual_tolerance) {
            break;
        }

        // Simple Jacobi iteration
        std::vector<double> dual_new(n_dofs);
        for (size_t i = 0; i < n_dofs; ++i) {
            double diag = adjoint_matrix[i][i];
            if (std::abs(diag) > 1e-12) {
                dual_new[i] = (adjoint_rhs[i] -
                    std::inner_product(adjoint_matrix[i].begin(),
                                      adjoint_matrix[i].end(),
                                      dual_solution.begin(), 0.0) +
                    diag * dual_solution[i]) / diag;
            }
        }
        dual_solution = dual_new;
    }

    // Store dual solution
    for (size_t i = 0; i < n_dofs; ++i) {
        dual_fields.set_field_value("dual", i, dual_solution[i]);
    }
}

std::vector<double> GoalOrientedAdaptivity::compute_dwr_indicator(
    const MeshBase& mesh,
    const MeshFields& primal_fields,
    const MeshFields& dual_fields) const {

    size_t n_elements = mesh.n_elements();
    std::vector<double> dwr_indicator(n_elements, 0.0);

    // Dual-weighted residual: η_K = R_K(u_h) · (z - z_h)
    // where R_K is element residual, z is dual solution, z_h is approximate dual

    for (size_t elem = 0; elem < n_elements; ++elem) {
        // Compute element residual
        auto residual = compute_element_residual(mesh, primal_fields, elem);

        // Get dual weight
        double dual_weight = compute_adjoint_weight(mesh, dual_fields, elem);

        // Compute weighted residual
        double weighted_residual = 0.0;
        for (double r : residual) {
            weighted_residual += r * dual_weight;
        }

        dwr_indicator[elem] = std::abs(weighted_residual);

        // Scale by adjoint weight if configured
        if (config_.use_goal_contribution) {
            dwr_indicator[elem] *= std::abs(dual_weight) * config_.adjoint_weight;
        }
    }

    return dwr_indicator;
}

std::vector<double> GoalOrientedAdaptivity::compute_goal_contribution(
    const MeshBase& mesh,
    const MeshFields& primal_fields,
    const MeshFields& dual_fields) const {

    size_t n_elements = mesh.n_elements();
    std::vector<double> contribution(n_elements, 0.0);

    // Compute how much each element contributes to goal error
    auto dwr = compute_dwr_indicator(mesh, primal_fields, dual_fields);

    double total_error = std::accumulate(dwr.begin(), dwr.end(), 0.0);

    if (total_error > 1e-12) {
        for (size_t i = 0; i < n_elements; ++i) {
            contribution[i] = dwr[i] / total_error;
        }
    }

    return contribution;
}

double GoalOrientedAdaptivity::estimate_goal_error(
    const MeshBase& mesh,
    const MeshFields& primal_fields,
    const MeshFields& dual_fields) const {

    auto dwr = compute_dwr_indicator(mesh, primal_fields, dual_fields);
    return std::accumulate(dwr.begin(), dwr.end(), 0.0);
}

std::vector<size_t> GoalOrientedAdaptivity::mark_for_goal(
    const MeshBase& mesh,
    const std::vector<double>& dwr_indicator,
    double target_error) const {

    std::vector<size_t> marked;

    // Use Dorfler marking strategy for goal
    std::vector<std::pair<double, size_t>> sorted_indicators;
    for (size_t i = 0; i < dwr_indicator.size(); ++i) {
        sorted_indicators.push_back({dwr_indicator[i], i});
    }

    std::sort(sorted_indicators.begin(), sorted_indicators.end(),
              std::greater<std::pair<double, size_t>>());

    double total_error = std::accumulate(dwr_indicator.begin(),
                                        dwr_indicator.end(), 0.0);
    double marked_error = 0.0;
    double theta = 0.7;  // Dorfler parameter

    for (const auto& [indicator, elem] : sorted_indicators) {
        marked.push_back(elem);
        marked_error += indicator;

        if (marked_error >= theta * total_error ||
            total_error - marked_error < target_error) {
            break;
        }
    }

    return marked;
}

std::vector<double> GoalOrientedAdaptivity::compute_element_residual(
    const MeshBase& mesh,
    const MeshFields& fields,
    size_t element_id) const {

    // Compute strong residual for element
    auto nodes = mesh.get_element_nodes(element_id);
    std::vector<double> residual(nodes.size(), 0.0);

    // Example: for diffusion problem Lu = f, residual R = f - Lu_h
    // This is problem-dependent

    for (size_t i = 0; i < nodes.size(); ++i) {
        double value = fields.get_field_value("solution", nodes[i]);

        // Compute discrete operator application (simplified)
        double laplacian = 0.0;
        for (size_t j = 0; j < nodes.size(); ++j) {
            if (i != j) {
                double neighbor_value = fields.get_field_value("solution", nodes[j]);
                laplacian += neighbor_value - value;
            }
        }

        double forcing = fields.get_field_value("forcing", nodes[i]);
        residual[i] = forcing - laplacian;
    }

    return residual;
}

double GoalOrientedAdaptivity::compute_adjoint_weight(
    const MeshBase& mesh,
    const MeshFields& dual_fields,
    size_t element_id) const {

    auto nodes = mesh.get_element_nodes(element_id);

    // Average dual solution over element
    double dual_avg = 0.0;
    for (auto node : nodes) {
        dual_avg += dual_fields.get_field_value("dual", node);
    }
    dual_avg /= nodes.size();

    return dual_avg;
}

//=============================================================================
// FeatureBasedAdaptivity Implementation
//=============================================================================

FeatureBasedAdaptivity::FeatureBasedAdaptivity(const FeatureDetectionConfig& config)
    : config_(config) {}

std::map<FeatureDetectionConfig::FeatureType, std::vector<size_t>>
FeatureBasedAdaptivity::detect_features(
    const MeshBase& mesh,
    const MeshFields& fields) const {

    std::map<FeatureDetectionConfig::FeatureType, std::vector<size_t>> features;

    for (auto feature_type : config_.features_to_detect) {
        std::vector<size_t> elements;

        switch (feature_type) {
            case FeatureDetectionConfig::FeatureType::SHOCK:
                elements = detect_shocks(mesh, fields);
                break;
            case FeatureDetectionConfig::FeatureType::VORTEX:
                elements = detect_vortices(mesh, fields);
                break;
            case FeatureDetectionConfig::FeatureType::BOUNDARY_LAYER:
                elements = detect_boundary_layers(mesh, fields);
                break;
            case FeatureDetectionConfig::FeatureType::INTERFACE:
                elements = detect_interfaces(mesh, fields);
                break;
            default:
                // Compute generic feature indicator
                auto indicator = compute_feature_indicator(mesh, fields, feature_type);
                double threshold = config_.thresholds.gradient_threshold;
                for (size_t i = 0; i < indicator.size(); ++i) {
                    if (indicator[i] > threshold) {
                        elements.push_back(i);
                    }
                }
        }

        features[feature_type] = elements;
    }

    return features;
}

std::vector<double> FeatureBasedAdaptivity::compute_feature_indicator(
    const MeshBase& mesh,
    const MeshFields& fields,
    FeatureDetectionConfig::FeatureType feature_type) const {

    size_t n_elements = mesh.n_elements();
    std::vector<double> indicator(n_elements, 0.0);

    switch (feature_type) {
        case FeatureDetectionConfig::FeatureType::SHOCK:
            return compute_ducros_sensor(mesh, fields);

        case FeatureDetectionConfig::FeatureType::VORTEX:
            return compute_q_criterion(mesh, fields);

        case FeatureDetectionConfig::FeatureType::GRADIENT_MAXIMUM:
            // Compute solution gradient magnitude
            for (size_t elem = 0; elem < n_elements; ++elem) {
                auto gradient = fields.get_field_gradient("solution", elem);
                indicator[elem] = std::sqrt(gradient[0]*gradient[0] +
                                           gradient[1]*gradient[1] +
                                           gradient[2]*gradient[2]);
            }
            break;

        default:
            // Generic gradient-based indicator
            for (size_t elem = 0; elem < n_elements; ++elem) {
                auto gradient = fields.get_field_gradient("solution", elem);
                indicator[elem] = std::sqrt(gradient[0]*gradient[0] +
                                           gradient[1]*gradient[1] +
                                           gradient[2]*gradient[2]);
            }
    }

    return indicator;
}

std::vector<size_t> FeatureBasedAdaptivity::detect_shocks(
    const MeshBase& mesh,
    const MeshFields& fields) const {

    std::vector<size_t> shock_elements;

    // Use Ducros sensor for shock detection
    auto ducros = compute_ducros_sensor(mesh, fields);

    for (size_t i = 0; i < ducros.size(); ++i) {
        if (ducros[i] > config_.thresholds.shock_threshold) {
            shock_elements.push_back(i);
        }
    }

    return shock_elements;
}

std::vector<size_t> FeatureBasedAdaptivity::detect_vortices(
    const MeshBase& mesh,
    const MeshFields& fields) const {

    std::vector<size_t> vortex_elements;

    // Use Q-criterion for vortex detection
    auto q_criterion = compute_q_criterion(mesh, fields);

    for (size_t i = 0; i < q_criterion.size(); ++i) {
        if (q_criterion[i] > config_.thresholds.vorticity_threshold) {
            vortex_elements.push_back(i);
        }
    }

    return vortex_elements;
}

std::vector<size_t> FeatureBasedAdaptivity::detect_boundary_layers(
    const MeshBase& mesh,
    const MeshFields& fields) const {

    std::vector<size_t> bl_elements;

    // Detect elements near walls with high velocity gradients
    for (size_t elem = 0; elem < mesh.n_elements(); ++elem) {
        // Check if element is near boundary
        bool near_boundary = false;
        auto faces = mesh.get_element_faces(elem);
        for (auto face : faces) {
            if (mesh.is_boundary_face(face)) {
                near_boundary = true;
                break;
            }
        }

        if (!near_boundary) continue;

        // Compute velocity gradient normal to wall
        auto vel_gradient = fields.get_field_gradient("velocity", elem);
        double grad_magnitude = std::sqrt(vel_gradient[0]*vel_gradient[0] +
                                         vel_gradient[1]*vel_gradient[1] +
                                         vel_gradient[2]*vel_gradient[2]);

        if (grad_magnitude > config_.thresholds.boundary_layer_threshold) {
            bl_elements.push_back(elem);
        }
    }

    return bl_elements;
}

std::vector<size_t> FeatureBasedAdaptivity::detect_interfaces(
    const MeshBase& mesh,
    const MeshFields& fields) const {

    std::vector<size_t> interface_elements;

    // Detect interfaces using level set or volume fraction gradient
    for (size_t elem = 0; elem < mesh.n_elements(); ++elem) {
        auto gradient = fields.get_field_gradient("level_set", elem);
        double grad_magnitude = std::sqrt(gradient[0]*gradient[0] +
                                         gradient[1]*gradient[1] +
                                         gradient[2]*gradient[2]);

        // Also check curvature
        auto hessian = fields.get_field_hessian("level_set", elem);
        double curvature = std::abs(hessian[0] + hessian[3] + hessian[5]);

        if (grad_magnitude > config_.thresholds.gradient_threshold ||
            curvature > config_.thresholds.curvature_threshold) {
            interface_elements.push_back(elem);
        }
    }

    return interface_elements;
}

void FeatureBasedAdaptivity::track_features(
    const MeshBase& mesh,
    const MeshFields& fields_old,
    const MeshFields& fields_new,
    std::map<size_t, size_t>& feature_correspondence) const {

    // Simple feature tracking based on spatial proximity
    // In practice, more sophisticated tracking would be needed

    feature_correspondence.clear();

    // This is a simplified placeholder
    // Real implementation would track feature centroids, topology, etc.
}

std::vector<double> FeatureBasedAdaptivity::compute_ducros_sensor(
    const MeshBase& mesh,
    const MeshFields& fields) const {

    size_t n_elements = mesh.n_elements();
    std::vector<double> ducros(n_elements, 0.0);

    // Ducros sensor: χ = (∇ × v)² / [(∇ × v)² + (∇ · v)² + ε]
    // High values indicate vorticity-dominated regions (vortices)
    // Low values indicate dilatation-dominated regions (shocks)

    for (size_t elem = 0; elem < n_elements; ++elem) {
        // Get velocity gradient tensor
        auto grad_u = fields.get_field_gradient("velocity_x", elem);
        auto grad_v = fields.get_field_gradient("velocity_y", elem);
        auto grad_w = fields.get_field_gradient("velocity_z", elem);

        // Compute vorticity magnitude: |∇ × v|²
        double omega_x = grad_w[1] - grad_v[2];  // dw/dy - dv/dz
        double omega_y = grad_u[2] - grad_w[0];  // du/dz - dw/dx
        double omega_z = grad_v[0] - grad_u[1];  // dv/dx - du/dy
        double vorticity_sq = omega_x*omega_x + omega_y*omega_y + omega_z*omega_z;

        // Compute dilatation: (∇ · v)²
        double divergence = grad_u[0] + grad_v[1] + grad_w[2];
        double dilatation_sq = divergence * divergence;

        // Ducros sensor (inverted: 1 means shock, 0 means vortex)
        double epsilon = 1e-10;
        ducros[elem] = dilatation_sq / (vorticity_sq + dilatation_sq + epsilon);
    }

    return ducros;
}

std::vector<double> FeatureBasedAdaptivity::compute_jameson_sensor(
    const MeshBase& mesh,
    const MeshFields& fields) const {

    size_t n_elements = mesh.n_elements();
    std::vector<double> jameson(n_elements, 0.0);

    // Jameson sensor based on second derivative of pressure
    for (size_t elem = 0; elem < n_elements; ++elem) {
        double p_center = fields.get_field_value("pressure", elem);

        // Get neighboring elements
        auto neighbors = mesh.get_element_neighbors(elem);

        if (neighbors.size() < 2) continue;

        double sum_p = 0.0;
        double sum_abs_p = 0.0;

        for (auto neighbor : neighbors) {
            double p_neighbor = fields.get_field_value("pressure", neighbor);
            sum_p += p_neighbor;
            sum_abs_p += std::abs(p_neighbor);
        }

        // Second difference
        double second_diff = std::abs(sum_p - neighbors.size() * p_center);
        double epsilon = 1e-10;

        jameson[elem] = second_diff / (sum_abs_p + epsilon);
    }

    return jameson;
}

std::vector<double> FeatureBasedAdaptivity::compute_vorticity(
    const MeshBase& mesh,
    const MeshFields& fields) const {

    size_t n_elements = mesh.n_elements();
    std::vector<double> vorticity(n_elements, 0.0);

    for (size_t elem = 0; elem < n_elements; ++elem) {
        auto grad_u = fields.get_field_gradient("velocity_x", elem);
        auto grad_v = fields.get_field_gradient("velocity_y", elem);
        auto grad_w = fields.get_field_gradient("velocity_z", elem);

        double omega_x = grad_w[1] - grad_v[2];
        double omega_y = grad_u[2] - grad_w[0];
        double omega_z = grad_v[0] - grad_u[1];

        vorticity[elem] = std::sqrt(omega_x*omega_x + omega_y*omega_y +
                                   omega_z*omega_z);
    }

    return vorticity;
}

std::vector<double> FeatureBasedAdaptivity::compute_q_criterion(
    const MeshBase& mesh,
    const MeshFields& fields) const {

    size_t n_elements = mesh.n_elements();
    std::vector<double> q_criterion(n_elements, 0.0);

    // Q-criterion: Q = 0.5 * (||Ω||² - ||S||²)
    // where Ω is vorticity tensor and S is strain rate tensor

    for (size_t elem = 0; elem < n_elements; ++elem) {
        // Get velocity gradient tensor
        Eigen::Matrix3d grad_v;
        auto grad_u = fields.get_field_gradient("velocity_x", elem);
        auto grad_v_y = fields.get_field_gradient("velocity_y", elem);
        auto grad_w = fields.get_field_gradient("velocity_z", elem);

        grad_v << grad_u[0], grad_u[1], grad_u[2],
                  grad_v_y[0], grad_v_y[1], grad_v_y[2],
                  grad_w[0], grad_w[1], grad_w[2];

        // Vorticity tensor: Ω = 0.5 * (∇v - (∇v)ᵀ)
        Eigen::Matrix3d omega = 0.5 * (grad_v - grad_v.transpose());

        // Strain rate tensor: S = 0.5 * (∇v + (∇v)ᵀ)
        Eigen::Matrix3d strain = 0.5 * (grad_v + grad_v.transpose());

        // Q = 0.5 * (||Ω||² - ||S||²)
        double omega_norm_sq = omega.squaredNorm();
        double strain_norm_sq = strain.squaredNorm();

        q_criterion[elem] = 0.5 * (omega_norm_sq - strain_norm_sq);
    }

    return q_criterion;
}

std::vector<double> FeatureBasedAdaptivity::compute_lambda2(
    const MeshBase& mesh,
    const MeshFields& fields) const {

    size_t n_elements = mesh.n_elements();
    std::vector<double> lambda2(n_elements, 0.0);

    // λ₂-criterion: vortex core where second eigenvalue of S² + Ω² is negative

    for (size_t elem = 0; elem < n_elements; ++elem) {
        Eigen::Matrix3d grad_v;
        auto grad_u = fields.get_field_gradient("velocity_x", elem);
        auto grad_v_y = fields.get_field_gradient("velocity_y", elem);
        auto grad_w = fields.get_field_gradient("velocity_z", elem);

        grad_v << grad_u[0], grad_u[1], grad_u[2],
                  grad_v_y[0], grad_v_y[1], grad_v_y[2],
                  grad_w[0], grad_w[1], grad_w[2];

        Eigen::Matrix3d omega = 0.5 * (grad_v - grad_v.transpose());
        Eigen::Matrix3d strain = 0.5 * (grad_v + grad_v.transpose());

        // Compute S² + Ω²
        Eigen::Matrix3d M = strain * strain + omega * omega;

        // Get eigenvalues
        Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> solver(M);
        Eigen::Vector3d eigenvalues = solver.eigenvalues();

        // λ₂ is the second eigenvalue (middle one when sorted)
        lambda2[elem] = eigenvalues(1);
    }

    return lambda2;
}

//=============================================================================
// ShockCapturingAdaptivity Implementation
//=============================================================================

ShockCapturingAdaptivity::ShockCapturingAdaptivity(
    const ShockCapturingConfig& config)
    : config_(config) {}

std::vector<size_t> ShockCapturingAdaptivity::detect_shocks(
    const MeshBase& mesh,
    const MeshFields& fields) const {

    auto indicator = compute_shock_indicator(mesh, fields);

    std::vector<size_t> shock_elements;
    double threshold = 0.5;  // Shock detection threshold

    for (size_t i = 0; i < indicator.size(); ++i) {
        if (indicator[i] > threshold) {
            shock_elements.push_back(i);
        }
    }

    return shock_elements;
}

std::vector<double> ShockCapturingAdaptivity::compute_shock_indicator(
    const MeshBase& mesh,
    const MeshFields& fields) const {

    size_t n_elements = mesh.n_elements();
    std::vector<double> indicator(n_elements, 0.0);

    switch (config_.detector) {
        case ShockCapturingConfig::ShockDetector::GRADIENT_BASED:
            indicator = compute_pressure_gradient(mesh, fields);
            break;

        case ShockCapturingConfig::ShockDetector::MACH_NUMBER:
            indicator = compute_mach_number(mesh, fields);
            // Mark supersonic regions as potential shocks
            for (auto& mach : indicator) {
                mach = (mach > 1.0) ? 1.0 : 0.0;
            }
            break;

        case ShockCapturingConfig::ShockDetector::PRESSURE_JUMP:
            // Detect pressure discontinuities
            for (size_t elem = 0; elem < n_elements; ++elem) {
                double p_center = fields.get_field_value("pressure", elem);
                auto neighbors = mesh.get_element_neighbors(elem);

                double max_jump = 0.0;
                for (auto neighbor : neighbors) {
                    double p_neighbor = fields.get_field_value("pressure", neighbor);
                    double jump = std::abs(p_neighbor - p_center) /
                                 std::max(p_center, 1e-10);
                    max_jump = std::max(max_jump, jump);
                }
                indicator[elem] = max_jump;
            }
            break;

        case ShockCapturingConfig::ShockDetector::ENTROPY_VISCOSITY:
            indicator = compute_entropy_residual(mesh, fields);
            break;

        case ShockCapturingConfig::ShockDetector::DUCROS_SENSOR:
            // Inverted Ducros: high values indicate shocks
            indicator = compute_pressure_gradient(mesh, fields);
            break;

        default:
            indicator = compute_pressure_gradient(mesh, fields);
    }

    // Normalize indicator
    double max_indicator = *std::max_element(indicator.begin(), indicator.end());
    if (max_indicator > 1e-12) {
        for (auto& ind : indicator) {
            ind /= max_indicator;
        }
    }

    return indicator;
}

std::map<size_t, std::array<double, 3>>
ShockCapturingAdaptivity::compute_shock_normals(
    const MeshBase& mesh,
    const MeshFields& fields,
    const std::vector<size_t>& shock_elements) const {

    std::map<size_t, std::array<double, 3>> normals;

    for (auto elem : shock_elements) {
        std::array<double, 3> normal = {0.0, 0.0, 0.0};

        switch (config_.normal_method) {
            case ShockCapturingConfig::NormalComputation::GRADIENT_BASED: {
                // Normal from pressure gradient
                auto grad_p = fields.get_field_gradient("pressure", elem);
                double magnitude = std::sqrt(grad_p[0]*grad_p[0] +
                                           grad_p[1]*grad_p[1] +
                                           grad_p[2]*grad_p[2]);
                if (magnitude > 1e-12) {
                    normal[0] = grad_p[0] / magnitude;
                    normal[1] = grad_p[1] / magnitude;
                    normal[2] = grad_p[2] / magnitude;
                }
                break;
            }

            case ShockCapturingConfig::NormalComputation::EIGENVECTOR_BASED: {
                // Normal from smallest eigenvector of Hessian
                auto hessian = fields.get_field_hessian("pressure", elem);

                Eigen::Matrix3d H;
                H << hessian[0], hessian[1], hessian[2],
                     hessian[1], hessian[3], hessian[4],
                     hessian[2], hessian[4], hessian[5];

                Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> solver(H);
                Eigen::Vector3d eigenvec = solver.eigenvectors().col(0);

                normal[0] = eigenvec(0);
                normal[1] = eigenvec(1);
                normal[2] = eigenvec(2);
                break;
            }

            default:
                // Default to gradient-based
                auto grad_p = fields.get_field_gradient("pressure", elem);
                double magnitude = std::sqrt(grad_p[0]*grad_p[0] +
                                           grad_p[1]*grad_p[1] +
                                           grad_p[2]*grad_p[2]);
                if (magnitude > 1e-12) {
                    normal[0] = grad_p[0] / magnitude;
                    normal[1] = grad_p[1] / magnitude;
                    normal[2] = grad_p[2] / magnitude;
                }
        }

        normals[elem] = normal;
    }

    return normals;
}

std::map<size_t, std::array<double, 6>>
ShockCapturingAdaptivity::compute_shock_metrics(
    const MeshBase& mesh,
    const MeshFields& fields,
    const std::vector<size_t>& shock_elements) const {

    std::map<size_t, std::array<double, 6>> metrics;

    if (!config_.anisotropic_shock_refinement) {
        // Isotropic refinement
        for (auto elem : shock_elements) {
            std::array<double, 6> metric = {1.0, 0.0, 0.0, 1.0, 0.0, 1.0};
            metrics[elem] = metric;
        }
        return metrics;
    }

    // Anisotropic refinement aligned with shock
    auto normals = compute_shock_normals(mesh, fields, shock_elements);

    for (auto elem : shock_elements) {
        auto normal = normals[elem];

        // Create metric tensor with strong refinement across shock
        // and weak refinement along shock
        double h_normal = config_.target_shock_thickness;  // Small spacing across shock
        double h_tangent = 10.0 * h_normal;  // Larger spacing along shock

        // Metric in normal direction: λ_n = 1/h_n²
        // Metric in tangent direction: λ_t = 1/h_t²
        double lambda_n = 1.0 / (h_normal * h_normal);
        double lambda_t = 1.0 / (h_tangent * h_tangent);

        // Build metric tensor: M = λ_n * n⊗n + λ_t * (I - n⊗n)
        std::array<double, 6> metric;

        // M_xx, M_xy, M_xz, M_yy, M_yz, M_zz
        metric[0] = lambda_n * normal[0]*normal[0] +
                   lambda_t * (1.0 - normal[0]*normal[0]);
        metric[1] = (lambda_n - lambda_t) * normal[0]*normal[1];
        metric[2] = (lambda_n - lambda_t) * normal[0]*normal[2];
        metric[3] = lambda_n * normal[1]*normal[1] +
                   lambda_t * (1.0 - normal[1]*normal[1]);
        metric[4] = (lambda_n - lambda_t) * normal[1]*normal[2];
        metric[5] = lambda_n * normal[2]*normal[2] +
                   lambda_t * (1.0 - normal[2]*normal[2]);

        metrics[elem] = metric;
    }

    return metrics;
}

std::vector<size_t> ShockCapturingAdaptivity::mark_shock_zones(
    const MeshBase& mesh,
    const MeshFields& fields,
    const std::vector<size_t>& shock_elements) const {

    std::vector<size_t> zone_elements = shock_elements;

    // Add pre-shock and post-shock regions
    std::set<size_t> zone_set(shock_elements.begin(), shock_elements.end());

    // Expand zone by neighbor layers
    int n_layers = static_cast<int>(
        std::max(config_.pre_shock_refinement_width,
                config_.post_shock_refinement_width));

    for (int layer = 0; layer < n_layers; ++layer) {
        std::set<size_t> new_elements;

        for (auto elem : zone_set) {
            auto neighbors = mesh.get_element_neighbors(elem);
            for (auto neighbor : neighbors) {
                new_elements.insert(neighbor);
            }
        }

        zone_set.insert(new_elements.begin(), new_elements.end());
    }

    zone_elements.assign(zone_set.begin(), zone_set.end());
    return zone_elements;
}

std::vector<std::pair<size_t, size_t>>
ShockCapturingAdaptivity::detect_shock_interactions(
    const MeshBase& mesh,
    const std::vector<size_t>& shock_elements) const {

    std::vector<std::pair<size_t, size_t>> interactions;

    if (!config_.detect_shock_interactions) {
        return interactions;
    }

    // Detect where shock elements are neighbors (potential interaction)
    std::set<size_t> shock_set(shock_elements.begin(), shock_elements.end());

    for (auto elem1 : shock_elements) {
        auto neighbors = mesh.get_element_neighbors(elem1);
        for (auto elem2 : neighbors) {
            if (shock_set.count(elem2) && elem1 < elem2) {
                interactions.push_back({elem1, elem2});
            }
        }
    }

    return interactions;
}

std::vector<double> ShockCapturingAdaptivity::compute_pressure_gradient(
    const MeshBase& mesh,
    const MeshFields& fields) const {

    size_t n_elements = mesh.n_elements();
    std::vector<double> grad_magnitude(n_elements, 0.0);

    for (size_t elem = 0; elem < n_elements; ++elem) {
        auto grad = fields.get_field_gradient("pressure", elem);
        grad_magnitude[elem] = std::sqrt(grad[0]*grad[0] +
                                        grad[1]*grad[1] +
                                        grad[2]*grad[2]);
    }

    return grad_magnitude;
}

std::vector<double> ShockCapturingAdaptivity::compute_entropy_residual(
    const MeshBase& mesh,
    const MeshFields& fields) const {

    size_t n_elements = mesh.n_elements();
    std::vector<double> entropy_residual(n_elements, 0.0);

    // Entropy residual for shock detection
    // For ideal gas: s = cv*log(p/ρ^γ)

    for (size_t elem = 0; elem < n_elements; ++elem) {
        double rho = fields.get_field_value("density", elem);
        double p = fields.get_field_value("pressure", elem);
        double gamma = 1.4;  // Specific heat ratio for air

        // Compute entropy
        double entropy = std::log(p / std::pow(rho, gamma));

        // Entropy should be constant across domain
        // Large entropy jumps indicate shocks
        auto neighbors = mesh.get_element_neighbors(elem);
        double max_entropy_jump = 0.0;

        for (auto neighbor : neighbors) {
            double rho_n = fields.get_field_value("density", neighbor);
            double p_n = fields.get_field_value("pressure", neighbor);
            double entropy_n = std::log(p_n / std::pow(rho_n, gamma));

            double jump = std::abs(entropy - entropy_n);
            max_entropy_jump = std::max(max_entropy_jump, jump);
        }

        entropy_residual[elem] = max_entropy_jump;
    }

    return entropy_residual;
}

std::vector<double> ShockCapturingAdaptivity::compute_mach_number(
    const MeshBase& mesh,
    const MeshFields& fields) const {

    size_t n_elements = mesh.n_elements();
    std::vector<double> mach(n_elements, 0.0);

    double gamma = 1.4;  // Specific heat ratio

    for (size_t elem = 0; elem < n_elements; ++elem) {
        double rho = fields.get_field_value("density", elem);
        double p = fields.get_field_value("pressure", elem);

        auto velocity = fields.get_field_vector("velocity", elem);
        double vel_magnitude = std::sqrt(velocity[0]*velocity[0] +
                                        velocity[1]*velocity[1] +
                                        velocity[2]*velocity[2]);

        // Speed of sound: a = sqrt(γ*p/ρ)
        double sound_speed = std::sqrt(gamma * p / rho);

        // Mach number
        mach[elem] = vel_magnitude / sound_speed;
    }

    return mach;
}

//=============================================================================
// LoadBalancedAdaptivity Implementation
//=============================================================================

LoadBalancedAdaptivity::LoadBalancedAdaptivity(const LoadBalancingConfig& config)
    : config_(config) {}

std::vector<double> LoadBalancedAdaptivity::compute_partition_loads(
    const MeshBase& mesh,
    const std::vector<int>& partition_assignment) const {

    int n_partitions = *std::max_element(partition_assignment.begin(),
                                        partition_assignment.end()) + 1;

    std::vector<double> loads(n_partitions, 0.0);

    for (size_t elem = 0; elem < mesh.n_elements(); ++elem) {
        int partition = partition_assignment[elem];

        switch (config_.strategy) {
            case LoadBalancingConfig::BalancingStrategy::ELEMENTS_PER_PARTITION:
                loads[partition] += 1.0;
                break;

            case LoadBalancingConfig::BalancingStrategy::WORK_PER_PARTITION:
                loads[partition] += estimate_element_work(mesh, elem);
                break;

            case LoadBalancingConfig::BalancingStrategy::MEMORY_PER_PARTITION:
                loads[partition] += mesh.get_element_memory_footprint(elem);
                break;

            case LoadBalancingConfig::BalancingStrategy::HYBRID:
                loads[partition] += 0.4 * 1.0 +  // Element count
                                  0.4 * estimate_element_work(mesh, elem) +
                                  0.2 * mesh.get_element_memory_footprint(elem);
                break;
        }
    }

    return loads;
}

double LoadBalancedAdaptivity::compute_imbalance(
    const std::vector<double>& partition_loads) const {

    if (partition_loads.empty()) return 0.0;

    double max_load = *std::max_element(partition_loads.begin(),
                                       partition_loads.end());
    double avg_load = std::accumulate(partition_loads.begin(),
                                     partition_loads.end(), 0.0) /
                     partition_loads.size();

    if (avg_load < 1e-12) return 0.0;

    return (max_load - avg_load) / avg_load;
}

bool LoadBalancedAdaptivity::needs_repartitioning(
    const MeshBase& mesh,
    const std::vector<int>& current_partition) const {

    auto loads = compute_partition_loads(mesh, current_partition);
    double imbalance = compute_imbalance(loads);

    return imbalance > config_.imbalance_tolerance;
}

std::vector<double> LoadBalancedAdaptivity::compute_element_weights(
    const MeshBase& mesh) const {

    size_t n_elements = mesh.n_elements();
    std::vector<double> weights(n_elements, 1.0);

    if (config_.strategy == LoadBalancingConfig::BalancingStrategy::WORK_PER_PARTITION ||
        config_.strategy == LoadBalancingConfig::BalancingStrategy::HYBRID) {

        for (size_t elem = 0; elem < n_elements; ++elem) {
            weights[elem] = estimate_element_work(mesh, elem);
        }
    }

    return weights;
}

double LoadBalancedAdaptivity::estimate_migration_cost(
    const MeshBase& mesh,
    const std::vector<int>& old_partition,
    const std::vector<int>& new_partition) const {

    if (!config_.consider_migration_cost) {
        return 0.0;
    }

    double cost = 0.0;

    for (size_t elem = 0; elem < mesh.n_elements(); ++elem) {
        if (old_partition[elem] != new_partition[elem]) {
            // Cost proportional to element data size
            cost += mesh.get_element_memory_footprint(elem);
        }
    }

    return cost;
}

std::map<int, std::vector<size_t>> LoadBalancedAdaptivity::balance_refinement(
    const MeshBase& mesh,
    const std::vector<size_t>& marked_elements,
    const std::vector<int>& partition_assignment) const {

    std::map<int, std::vector<size_t>> balanced_marking;

    // Distribute marked elements to maintain load balance
    std::map<int, double> partition_loads;

    for (auto elem : marked_elements) {
        int partition = partition_assignment[elem];
        balanced_marking[partition].push_back(elem);
        partition_loads[partition] += estimate_element_work(mesh, elem);
    }

    return balanced_marking;
}

double LoadBalancedAdaptivity::estimate_element_work(
    const MeshBase& mesh,
    size_t element_id) const {

    double work = config_.work_estimation.element_base_cost;

    if (config_.work_estimation.use_timing_data) {
        // Use actual timing data if available
        // This is a placeholder
        work = 1.0;
    }

    // Account for refinement level
    int refinement_level = mesh.get_element_refinement_level(element_id);
    work *= std::pow(config_.work_estimation.refinement_cost_factor,
                    refinement_level);

    return work;
}

//=============================================================================
// TimeAdaptiveStrategy Implementation
//=============================================================================

TimeAdaptiveStrategy::TimeAdaptiveStrategy(const TimeAdaptivityConfig& config)
    : config_(config) {}

double TimeAdaptiveStrategy::compute_temporal_error(
    const MeshBase& mesh,
    const MeshFields& fields_old,
    const MeshFields& fields_new,
    double dt) const {

    if (!config_.temporal_control.enable) {
        return 0.0;
    }

    double error = 0.0;

    // Estimate temporal discretization error
    // Using simple difference between time steps
    for (size_t elem = 0; elem < mesh.n_elements(); ++elem) {
        double u_old = fields_old.get_field_value("solution", elem);
        double u_new = fields_new.get_field_value("solution", elem);

        double local_error = std::abs(u_new - u_old) / dt;
        error += local_error * local_error * mesh.get_element_volume(elem);
    }

    return std::sqrt(error);
}

double TimeAdaptiveStrategy::suggest_time_step(
    const MeshBase& mesh,
    const MeshFields& fields,
    double current_dt,
    double temporal_error) const {

    if (config_.strategy == TimeAdaptivityConfig::TimeSteppingStrategy::FIXED_TIME_STEP) {
        return current_dt;
    }

    double safety = config_.temporal_control.time_step_safety_factor;
    double tolerance = config_.temporal_control.temporal_tolerance;

    // PI controller for time step adaptation
    double suggested_dt = current_dt;

    if (temporal_error > 1e-12) {
        suggested_dt = current_dt * safety * std::sqrt(tolerance / temporal_error);
    }

    // Clamp to min/max
    suggested_dt = std::max(suggested_dt, config_.temporal_control.min_time_step);
    suggested_dt = std::min(suggested_dt, config_.temporal_control.max_time_step);

    // Also consider CFL condition if velocity field exists
    double cfl_dt = current_dt;
    for (size_t elem = 0; elem < mesh.n_elements(); ++elem) {
        auto velocity = fields.get_field_vector("velocity", elem);
        double vel_mag = std::sqrt(velocity[0]*velocity[0] +
                                  velocity[1]*velocity[1] +
                                  velocity[2]*velocity[2]);

        double elem_size = mesh.get_element_characteristic_length(elem);
        if (vel_mag > 1e-12) {
            cfl_dt = std::min(cfl_dt, 0.5 * elem_size / vel_mag);
        }
    }

    suggested_dt = std::min(suggested_dt, cfl_dt);

    return suggested_dt;
}

bool TimeAdaptiveStrategy::needs_spatial_adaptation(int time_step) const {
    return (time_step % config_.spatial_adapt_frequency) == 0;
}

std::map<size_t, size_t> TimeAdaptiveStrategy::track_moving_features(
    const MeshBase& mesh_old,
    const MeshBase& mesh_new,
    const MeshFields& fields_old,
    const MeshFields& fields_new) const {

    std::map<size_t, size_t> correspondence;

    if (!config_.track_moving_features) {
        return correspondence;
    }

    // Track features based on spatial proximity and field values
    // This is a simplified implementation

    return correspondence;
}

std::vector<size_t> TimeAdaptiveStrategy::predict_refinement_regions(
    const MeshBase& mesh,
    const std::vector<MeshFields>& fields_history) const {

    std::vector<size_t> predicted_regions;

    if (!config_.predictive_refinement.enable || fields_history.size() < 2) {
        return predicted_regions;
    }

    // Predict where refinement will be needed based on field evolution
    MeshFields predicted_field = fields_history.back();
    extrapolate_field(fields_history, predicted_field);

    // Mark regions with high predicted gradients
    for (size_t elem = 0; elem < mesh.n_elements(); ++elem) {
        auto gradient = predicted_field.get_field_gradient("solution", elem);
        double grad_mag = std::sqrt(gradient[0]*gradient[0] +
                                   gradient[1]*gradient[1] +
                                   gradient[2]*gradient[2]);

        if (grad_mag > 1.0) {  // Threshold
            predicted_regions.push_back(elem);
        }
    }

    return predicted_regions;
}

void TimeAdaptiveStrategy::apply_temporal_smoothing(
    MeshBase& mesh,
    const MeshBase& previous_mesh) const {

    if (!config_.temporal_mesh_smoothing) {
        return;
    }

    // Smooth mesh changes to avoid sudden topology changes
    // This is a placeholder for mesh smoothing logic
}

void TimeAdaptiveStrategy::extrapolate_field(
    const std::vector<MeshFields>& fields_history,
    MeshFields& predicted_field) const {

    size_t n_history = fields_history.size();
    if (n_history < 2) return;

    // Linear extrapolation
    if (config_.predictive_refinement.predictor_type == "linear") {
        const auto& f_n = fields_history[n_history - 1];
        const auto& f_nm1 = fields_history[n_history - 2];

        // predicted = f_n + (f_n - f_nm1)
        // This is a simplified version
    }
    // Quadratic extrapolation
    else if (config_.predictive_refinement.predictor_type == "quadratic" &&
             n_history >= 3) {
        // Use three previous time steps for quadratic extrapolation
    }
}

//=============================================================================
// AdaptivityStrategyManager Implementation
//=============================================================================

AdaptivityStrategyManager::AdaptivityStrategyManager() {}

void AdaptivityStrategyManager::register_goal_oriented(
    std::shared_ptr<GoalOrientedAdaptivity> strategy) {
    goal_oriented_ = strategy;
    strategies_["goal_oriented"] = strategy;
}

void AdaptivityStrategyManager::register_feature_based(
    std::shared_ptr<FeatureBasedAdaptivity> strategy) {
    feature_based_ = strategy;
    strategies_["feature_based"] = strategy;
}

void AdaptivityStrategyManager::register_shock_capturing(
    std::shared_ptr<ShockCapturingAdaptivity> strategy) {
    shock_capturing_ = strategy;
    strategies_["shock_capturing"] = strategy;
}

void AdaptivityStrategyManager::register_load_balancing(
    std::shared_ptr<LoadBalancedAdaptivity> strategy) {
    load_balancing_ = strategy;
    strategies_["load_balancing"] = strategy;
}

void AdaptivityStrategyManager::register_time_adaptive(
    std::shared_ptr<TimeAdaptiveStrategy> strategy) {
    time_adaptive_ = strategy;
    strategies_["time_adaptive"] = strategy;
}

std::vector<size_t> AdaptivityStrategyManager::combine_strategies(
    const MeshBase& mesh,
    const MeshFields& fields,
    const std::map<std::string, double>& strategy_weights) const {

    std::map<size_t, double> element_scores;

    // Execute each strategy and combine results
    auto all_results = execute_all_strategies(mesh, fields);

    for (const auto& [strategy_name, elements] : all_results) {
        double weight = 1.0;
        auto it = strategy_weights.find(strategy_name);
        if (it != strategy_weights.end()) {
            weight = it->second;
        }

        for (auto elem : elements) {
            element_scores[elem] += weight;
        }
    }

    // Collect elements with scores above threshold
    std::vector<size_t> combined_marking;
    double threshold = 0.5;

    for (const auto& [elem, score] : element_scores) {
        if (score >= threshold) {
            combined_marking.push_back(elem);
        }
    }

    return combined_marking;
}

std::map<std::string, std::vector<size_t>>
AdaptivityStrategyManager::execute_all_strategies(
    const MeshBase& mesh,
    const MeshFields& fields) const {

    std::map<std::string, std::vector<size_t>> results;

    // This is a simplified implementation
    // In practice, each strategy would be executed and return marked elements

    return results;
}

} // namespace mesh
