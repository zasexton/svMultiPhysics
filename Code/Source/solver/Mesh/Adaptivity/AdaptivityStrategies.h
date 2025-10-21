#ifndef ADAPTIVITY_STRATEGIES_H
#define ADAPTIVITY_STRATEGIES_H

#include <vector>
#include <array>
#include <memory>
#include <map>
#include <functional>
#include <string>

namespace mesh {

// Forward declarations
class MeshBase;
class MeshFields;
class ErrorEstimator;

/// @brief Goal-oriented adaptivity configuration
struct GoalOrientedConfig {
    /// Target functional to minimize (e.g., drag, lift, pressure at point)
    std::string goal_functional;

    /// Dual problem configuration
    struct DualConfig {
        bool solve_dual_problem;
        double dual_tolerance;
        int max_dual_iterations;
        std::string dual_solver;
    } dual_config;

    /// Error representation
    enum class ErrorRepresentation {
        RESIDUAL_BASED,      // Use residual-based dual weighted residual
        INTERPOLATION_BASED, // Use interpolation error estimates
        RECOVERY_BASED,      // Use recovered gradients
        HYBRID              // Combine multiple representations
    } error_representation;

    /// Goal tolerance
    double goal_tolerance;

    /// Whether to adapt based on goal contribution
    bool use_goal_contribution;

    /// Adjoint weighting factor
    double adjoint_weight;
};

/// @brief Feature-based adaptivity configuration
struct FeatureDetectionConfig {
    /// Feature types to detect
    enum class FeatureType {
        SHOCK,              // Discontinuities in solution
        CONTACT,            // Contact discontinuities
        BOUNDARY_LAYER,     // Thin viscous layers
        VORTEX,             // Vortical structures
        INTERFACE,          // Material interfaces
        GRADIENT_MAXIMUM    // Solution gradient maxima
    };

    std::vector<FeatureType> features_to_detect;

    /// Detection thresholds
    struct Thresholds {
        double shock_threshold;           // For shock detection
        double vorticity_threshold;       // For vortex detection
        double gradient_threshold;        // For gradient features
        double curvature_threshold;       // For interface detection
        double boundary_layer_threshold;  // For boundary layer detection
    } thresholds;

    /// Feature tracking
    bool track_features_in_time;

    /// Feature preservation
    bool preserve_feature_topology;

    /// Sensor types
    enum class SensorType {
        GRADIENT_BASED,     // Based on solution gradients
        HESSIAN_BASED,      // Based on solution Hessian
        VARIATIONAL,        // Based on variational principles
        DUCROS,             // Ducros sensor for shocks
        JAMESON,            // Jameson shock sensor
        TROUBLED_CELL       // Troubled cell indicator
    } sensor_type;
};

/// @brief Shock-capturing adaptivity configuration
struct ShockCapturingConfig {
    /// Shock detection method
    enum class ShockDetector {
        GRADIENT_BASED,     // Based on density/pressure gradients
        MACH_NUMBER,        // Based on local Mach number
        PRESSURE_JUMP,      // Based on pressure discontinuities
        ENTROPY_VISCOSITY,  // Based on entropy residual
        DUCROS_SENSOR,      // Ducros shock sensor (compressibility)
        JAMESON_SENSOR,     // Jameson shock sensor
        PERSSON_SENSOR      // Persson modal decay sensor
    } detector;

    /// Shock alignment
    bool align_with_shock;

    /// Shock thickness control
    double target_shock_thickness;  // In number of cells

    /// Anisotropic refinement along shock normal
    bool anisotropic_shock_refinement;

    /// Shock normal computation method
    enum class NormalComputation {
        GRADIENT_BASED,     // From pressure/density gradient
        EIGENVECTOR_BASED,  // From Hessian eigenvectors
        CHARACTERISTIC_BASED // From characteristic decomposition
    } normal_method;

    /// Pre-shock and post-shock refinement zones
    double pre_shock_refinement_width;
    double post_shock_refinement_width;

    /// Multi-dimensional shock detection
    bool detect_shock_interactions;
};

/// @brief Load-balancing adaptivity configuration
struct LoadBalancingConfig {
    /// Load balancing strategy
    enum class BalancingStrategy {
        ELEMENTS_PER_PARTITION,  // Balance element count
        WORK_PER_PARTITION,      // Balance computational work
        MEMORY_PER_PARTITION,    // Balance memory usage
        HYBRID                   // Combine multiple criteria
    } strategy;

    /// Work estimation
    struct WorkEstimation {
        bool use_timing_data;
        std::map<std::string, double> operation_costs;  // Cost per operation type
        double element_base_cost;
        double refinement_cost_factor;
    } work_estimation;

    /// Imbalance tolerance
    double imbalance_tolerance;

    /// Repartitioning frequency
    int repartition_frequency;

    /// Migration cost consideration
    bool consider_migration_cost;

    /// Graph partitioning method
    enum class PartitioningMethod {
        RECURSIVE_BISECTION,
        K_WAY,
        GEOMETRIC,
        HYPERGRAPH
    } partitioning_method;
};

/// @brief Time-dependent adaptivity configuration
struct TimeAdaptivityConfig {
    /// Time stepping strategy
    enum class TimeSteppingStrategy {
        FIXED_TIME_STEP,
        ADAPTIVE_TIME_STEP,
        SPACE_TIME_COUPLED
    } strategy;

    /// Error control in time
    struct TemporalErrorControl {
        bool enable;
        double temporal_tolerance;
        double min_time_step;
        double max_time_step;
        double time_step_safety_factor;
    } temporal_control;

    /// Spatial refinement frequency
    int spatial_adapt_frequency;

    /// Feature tracking
    bool track_moving_features;

    /// Predictive refinement
    struct PredictiveRefinement {
        bool enable;
        int prediction_steps;
        std::string predictor_type;  // "linear", "quadratic", "extrapolation"
    } predictive_refinement;

    /// Mesh smoothing in time
    bool temporal_mesh_smoothing;
};

/// @brief Goal-oriented adaptivity using dual-weighted residual method
class GoalOrientedAdaptivity {
public:
    GoalOrientedAdaptivity(const GoalOrientedConfig& config);

    /// Compute goal functional value
    double compute_goal_functional(const MeshBase& mesh,
                                   const MeshFields& fields) const;

    /// Solve dual (adjoint) problem
    void solve_dual_problem(const MeshBase& mesh,
                           const MeshFields& primal_fields,
                           MeshFields& dual_fields) const;

    /// Compute dual-weighted residual error indicator
    std::vector<double> compute_dwr_indicator(const MeshBase& mesh,
                                              const MeshFields& primal_fields,
                                              const MeshFields& dual_fields) const;

    /// Compute goal contribution for each element
    std::vector<double> compute_goal_contribution(const MeshBase& mesh,
                                                  const MeshFields& primal_fields,
                                                  const MeshFields& dual_fields) const;

    /// Estimate goal error
    double estimate_goal_error(const MeshBase& mesh,
                              const MeshFields& primal_fields,
                              const MeshFields& dual_fields) const;

    /// Mark elements for refinement based on goal
    std::vector<size_t> mark_for_goal(const MeshBase& mesh,
                                      const std::vector<double>& dwr_indicator,
                                      double target_error) const;

    const GoalOrientedConfig& get_config() const { return config_; }
    void set_config(const GoalOrientedConfig& config) { config_ = config; }

private:
    GoalOrientedConfig config_;

    /// Compute residual for element
    std::vector<double> compute_element_residual(const MeshBase& mesh,
                                                 const MeshFields& fields,
                                                 size_t element_id) const;

    /// Compute adjoint weight for element
    double compute_adjoint_weight(const MeshBase& mesh,
                                 const MeshFields& dual_fields,
                                 size_t element_id) const;
};

/// @brief Feature-based adaptivity using various detection methods
class FeatureBasedAdaptivity {
public:
    FeatureBasedAdaptivity(const FeatureDetectionConfig& config);

    /// Detect features in the solution
    std::map<FeatureDetectionConfig::FeatureType, std::vector<size_t>>
    detect_features(const MeshBase& mesh, const MeshFields& fields) const;

    /// Compute feature indicator for each element
    std::vector<double> compute_feature_indicator(
        const MeshBase& mesh,
        const MeshFields& fields,
        FeatureDetectionConfig::FeatureType feature_type) const;

    /// Detect shocks using various sensors
    std::vector<size_t> detect_shocks(const MeshBase& mesh,
                                      const MeshFields& fields) const;

    /// Detect vortical structures
    std::vector<size_t> detect_vortices(const MeshBase& mesh,
                                        const MeshFields& fields) const;

    /// Detect boundary layers
    std::vector<size_t> detect_boundary_layers(const MeshBase& mesh,
                                               const MeshFields& fields) const;

    /// Detect interfaces
    std::vector<size_t> detect_interfaces(const MeshBase& mesh,
                                          const MeshFields& fields) const;

    /// Track features in time
    void track_features(const MeshBase& mesh,
                       const MeshFields& fields_old,
                       const MeshFields& fields_new,
                       std::map<size_t, size_t>& feature_correspondence) const;

    /// Compute Ducros sensor (shock detector)
    std::vector<double> compute_ducros_sensor(const MeshBase& mesh,
                                             const MeshFields& fields) const;

    /// Compute Jameson sensor
    std::vector<double> compute_jameson_sensor(const MeshBase& mesh,
                                              const MeshFields& fields) const;

    const FeatureDetectionConfig& get_config() const { return config_; }
    void set_config(const FeatureDetectionConfig& config) { config_ = config; }

private:
    FeatureDetectionConfig config_;

    /// Compute vorticity magnitude
    std::vector<double> compute_vorticity(const MeshBase& mesh,
                                         const MeshFields& fields) const;

    /// Compute Q-criterion for vortex detection
    std::vector<double> compute_q_criterion(const MeshBase& mesh,
                                           const MeshFields& fields) const;

    /// Compute lambda2-criterion for vortex detection
    std::vector<double> compute_lambda2(const MeshBase& mesh,
                                       const MeshFields& fields) const;
};

/// @brief Shock-capturing adaptivity with alignment and anisotropic refinement
class ShockCapturingAdaptivity {
public:
    ShockCapturingAdaptivity(const ShockCapturingConfig& config);

    /// Detect shock elements
    std::vector<size_t> detect_shocks(const MeshBase& mesh,
                                      const MeshFields& fields) const;

    /// Compute shock indicator for each element
    std::vector<double> compute_shock_indicator(const MeshBase& mesh,
                                               const MeshFields& fields) const;

    /// Compute shock normal direction for each element
    std::map<size_t, std::array<double, 3>> compute_shock_normals(
        const MeshBase& mesh,
        const MeshFields& fields,
        const std::vector<size_t>& shock_elements) const;

    /// Determine anisotropic refinement directions for shocks
    std::map<size_t, std::array<double, 6>> compute_shock_metrics(
        const MeshBase& mesh,
        const MeshFields& fields,
        const std::vector<size_t>& shock_elements) const;

    /// Mark shock zone elements (pre-shock and post-shock regions)
    std::vector<size_t> mark_shock_zones(const MeshBase& mesh,
                                         const MeshFields& fields,
                                         const std::vector<size_t>& shock_elements) const;

    /// Detect shock interactions
    std::vector<std::pair<size_t, size_t>> detect_shock_interactions(
        const MeshBase& mesh,
        const std::vector<size_t>& shock_elements) const;

    const ShockCapturingConfig& get_config() const { return config_; }
    void set_config(const ShockCapturingConfig& config) { config_ = config; }

private:
    ShockCapturingConfig config_;

    /// Compute pressure gradient magnitude
    std::vector<double> compute_pressure_gradient(const MeshBase& mesh,
                                                  const MeshFields& fields) const;

    /// Compute entropy residual
    std::vector<double> compute_entropy_residual(const MeshBase& mesh,
                                                const MeshFields& fields) const;

    /// Compute Mach number
    std::vector<double> compute_mach_number(const MeshBase& mesh,
                                           const MeshFields& fields) const;
};

/// @brief Load-balancing adaptivity for parallel computations
class LoadBalancedAdaptivity {
public:
    LoadBalancedAdaptivity(const LoadBalancingConfig& config);

    /// Compute partition loads
    std::vector<double> compute_partition_loads(
        const MeshBase& mesh,
        const std::vector<int>& partition_assignment) const;

    /// Compute load imbalance
    double compute_imbalance(const std::vector<double>& partition_loads) const;

    /// Determine if repartitioning is needed
    bool needs_repartitioning(const MeshBase& mesh,
                             const std::vector<int>& current_partition) const;

    /// Compute element weights for partitioning
    std::vector<double> compute_element_weights(const MeshBase& mesh) const;

    /// Estimate migration cost
    double estimate_migration_cost(const MeshBase& mesh,
                                   const std::vector<int>& old_partition,
                                   const std::vector<int>& new_partition) const;

    /// Balance refinement across partitions
    std::map<int, std::vector<size_t>> balance_refinement(
        const MeshBase& mesh,
        const std::vector<size_t>& marked_elements,
        const std::vector<int>& partition_assignment) const;

    const LoadBalancingConfig& get_config() const { return config_; }
    void set_config(const LoadBalancingConfig& config) { config_ = config; }

private:
    LoadBalancingConfig config_;

    /// Estimate computational work for element
    double estimate_element_work(const MeshBase& mesh, size_t element_id) const;
};

/// @brief Time-dependent adaptivity with feature tracking
class TimeAdaptiveStrategy {
public:
    TimeAdaptiveStrategy(const TimeAdaptivityConfig& config);

    /// Compute temporal error estimate
    double compute_temporal_error(const MeshBase& mesh,
                                 const MeshFields& fields_old,
                                 const MeshFields& fields_new,
                                 double dt) const;

    /// Suggest next time step
    double suggest_time_step(const MeshBase& mesh,
                            const MeshFields& fields,
                            double current_dt,
                            double temporal_error) const;

    /// Determine if spatial adaptation is needed
    bool needs_spatial_adaptation(int time_step) const;

    /// Track moving features
    std::map<size_t, size_t> track_moving_features(
        const MeshBase& mesh_old,
        const MeshBase& mesh_new,
        const MeshFields& fields_old,
        const MeshFields& fields_new) const;

    /// Predict refinement regions for next time step
    std::vector<size_t> predict_refinement_regions(
        const MeshBase& mesh,
        const std::vector<MeshFields>& fields_history) const;

    /// Smooth mesh changes in time
    void apply_temporal_smoothing(MeshBase& mesh,
                                 const MeshBase& previous_mesh) const;

    const TimeAdaptivityConfig& get_config() const { return config_; }
    void set_config(const TimeAdaptivityConfig& config) { config_ = config; }

private:
    TimeAdaptivityConfig config_;

    /// Extrapolate field to predict future state
    void extrapolate_field(const std::vector<MeshFields>& fields_history,
                          MeshFields& predicted_field) const;
};

/// @brief Unified adaptivity strategy manager
class AdaptivityStrategyManager {
public:
    AdaptivityStrategyManager();

    /// Register goal-oriented strategy
    void register_goal_oriented(std::shared_ptr<GoalOrientedAdaptivity> strategy);

    /// Register feature-based strategy
    void register_feature_based(std::shared_ptr<FeatureBasedAdaptivity> strategy);

    /// Register shock-capturing strategy
    void register_shock_capturing(std::shared_ptr<ShockCapturingAdaptivity> strategy);

    /// Register load-balancing strategy
    void register_load_balancing(std::shared_ptr<LoadBalancedAdaptivity> strategy);

    /// Register time-adaptive strategy
    void register_time_adaptive(std::shared_ptr<TimeAdaptiveStrategy> strategy);

    /// Combine multiple strategies
    std::vector<size_t> combine_strategies(
        const MeshBase& mesh,
        const MeshFields& fields,
        const std::map<std::string, double>& strategy_weights) const;

    /// Execute all registered strategies
    std::map<std::string, std::vector<size_t>> execute_all_strategies(
        const MeshBase& mesh,
        const MeshFields& fields) const;

    /// Get strategy by name
    template<typename T>
    std::shared_ptr<T> get_strategy(const std::string& name) const {
        auto it = strategies_.find(name);
        if (it != strategies_.end()) {
            return std::dynamic_pointer_cast<T>(it->second);
        }
        return nullptr;
    }

private:
    std::map<std::string, std::shared_ptr<void>> strategies_;

    std::shared_ptr<GoalOrientedAdaptivity> goal_oriented_;
    std::shared_ptr<FeatureBasedAdaptivity> feature_based_;
    std::shared_ptr<ShockCapturingAdaptivity> shock_capturing_;
    std::shared_ptr<LoadBalancedAdaptivity> load_balancing_;
    std::shared_ptr<TimeAdaptiveStrategy> time_adaptive_;
};

} // namespace mesh

#endif // ADAPTIVITY_STRATEGIES_H
