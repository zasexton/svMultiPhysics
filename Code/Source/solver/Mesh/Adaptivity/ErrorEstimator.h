/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 *
 * Permission is hereby granted, free of charge, to any person obtaining
 * a copy of this software and associated documentation files (the
 * "Software"), to deal in the Software without restriction, including
 * without limitation the rights to use, copy, modify, merge, publish,
 * distribute, sublicense, and/or sell copies of the Software, and to
 * permit persons to whom the Software is furnished to do so, subject
 * to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included
 * in all copies or substantial portions of the Software.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS
 * IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED
 * TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
 * PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER
 * OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
 * LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
 * NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#ifndef SVMP_ERROR_ESTIMATOR_H
#define SVMP_ERROR_ESTIMATOR_H

#include "Options.h"
#include <memory>
#include <string>
#include <vector>
#include <functional>

namespace svmp {

// Forward declarations
class MeshBase;
class MeshFields;

/**
 * @brief Abstract base class for error estimation strategies
 *
 * Error estimators compute element-wise error indicators that drive
 * adaptive mesh refinement and coarsening decisions.
 */
class ErrorEstimator {
public:
  virtual ~ErrorEstimator() = default;

  /**
   * @brief Compute element-wise error indicators
   *
   * @param mesh The mesh to estimate errors on
   * @param fields Optional fields for computation
   * @param options Adaptivity options
   * @return Vector of error indicators (one per element)
   */
  virtual std::vector<double> estimate(
      const MeshBase& mesh,
      const MeshFields* fields,
      const AdaptivityOptions& options) const = 0;

  /**
   * @brief Get estimator name for diagnostics
   */
  virtual std::string name() const = 0;

  /**
   * @brief Check if estimator requires field data
   */
  virtual bool requires_fields() const { return false; }

  /**
   * @brief Get required field names
   */
  virtual std::vector<std::string> required_field_names() const { return {}; }
};

/**
 * @brief Gradient recovery error estimator (Zienkiewicz-Zhu)
 *
 * Estimates error by comparing gradient of solution with a recovered
 * (smoothed) gradient field.
 */
class GradientRecoveryEstimator : public ErrorEstimator {
public:
  /**
   * @brief Configuration for gradient recovery
   */
  struct Config {
    /** Field name to estimate gradients from */
    std::string field_name = "solution";

    /** Use superconvergent patch recovery */
    bool use_patch_recovery = true;

    /** Recovery polynomial order */
    int recovery_order = 1;

    /** Weight by element volume */
    bool volume_weighted = true;
	  };

	  GradientRecoveryEstimator() : GradientRecoveryEstimator(Config{}) {}
	  explicit GradientRecoveryEstimator(const Config& config);

  std::vector<double> estimate(
      const MeshBase& mesh,
      const MeshFields* fields,
      const AdaptivityOptions& options) const override;

  std::string name() const override { return "GradientRecovery"; }

  bool requires_fields() const override { return true; }

  std::vector<std::string> required_field_names() const override {
    return {config_.field_name};
  }

private:
  Config config_;

  /** Recover gradient at vertices using patch recovery */
  std::vector<std::vector<double>> recover_gradients(
      const MeshBase& mesh,
      const std::vector<double>& field_values) const;

  /** Compute element error from gradient difference */
  double compute_element_error(
      const MeshBase& mesh,
      size_t elem_id,
      const std::vector<double>& field_gradients,
      const std::vector<std::vector<double>>& recovered_gradients) const;
};

/**
 * @brief Jump-based error indicator
 *
 * Estimates error based on solution jumps across element interfaces.
 */
class JumpIndicatorEstimator : public ErrorEstimator {
public:
  /**
   * @brief Configuration for jump indicator
   */
  struct Config {
    /** Field name to compute jumps from */
    std::string field_name = "solution";

    /** Type of jump to compute */
    enum class JumpType {
      NORMAL_DERIVATIVE,  // Jump in normal derivative
      TANGENTIAL_DERIVATIVE, // Jump in tangential derivative
      VALUE,             // Jump in value
      FLUX               // Jump in flux
    };

    JumpType jump_type = JumpType::NORMAL_DERIVATIVE;

    /** Scale jump by face area */
    bool area_scaled = true;

    /** Power for jump norm */
    double norm_power = 2.0;
	  };

	  JumpIndicatorEstimator() : JumpIndicatorEstimator(Config{}) {}
	  explicit JumpIndicatorEstimator(const Config& config);

  std::vector<double> estimate(
      const MeshBase& mesh,
      const MeshFields* fields,
      const AdaptivityOptions& options) const override;

  std::string name() const override { return "JumpIndicator"; }

  bool requires_fields() const override { return true; }

  std::vector<std::string> required_field_names() const override {
    return {config_.field_name};
  }

private:
  Config config_;

  /** Compute jump across a face */
  double compute_face_jump(
      const MeshBase& mesh,
      size_t face_id,
      const std::vector<double>& field_values) const;
};

/**
 * @brief Residual-based error estimator
 *
 * Estimates error using element and edge residuals from the PDE.
 */
class ResidualBasedEstimator : public ErrorEstimator {
public:
  /**
   * @brief Configuration for residual-based estimation
   */
  struct Config {
    /** Function to compute element residual */
    using ResidualFunc = std::function<double(
        const MeshBase&, size_t elem_id, const MeshFields*)>;

    ResidualFunc element_residual;

    /** Function to compute edge/face residual */
    ResidualFunc edge_residual;

    /** Include edge residuals */
    bool include_edge_residuals = true;

    /** Scaling constant */
    double scaling_constant = 1.0;

    /** Use h-weighting */
    bool h_weighted = true;
  };

  explicit ResidualBasedEstimator(const Config& config);

  std::vector<double> estimate(
      const MeshBase& mesh,
      const MeshFields* fields,
      const AdaptivityOptions& options) const override;

  std::string name() const override { return "ResidualBased"; }

  bool requires_fields() const override { return true; }

private:
  Config config_;
};

/**
 * @brief User-provided field error indicator
 *
 * Uses a pre-computed error field provided by the user.
 */
class UserFieldEstimator : public ErrorEstimator {
public:
  /**
   * @brief Configuration for user field estimator
   */
  struct Config {
    /** Name of the error indicator field */
    std::string error_field_name = "error_indicator";

    /** Apply scaling or normalization */
    bool normalize = false;

    /** Scaling factor */
    double scale_factor = 1.0;
	  };

	  UserFieldEstimator() : UserFieldEstimator(Config{}) {}
	  explicit UserFieldEstimator(const Config& config);

  std::vector<double> estimate(
      const MeshBase& mesh,
      const MeshFields* fields,
      const AdaptivityOptions& options) const override;

  std::string name() const override { return "UserField"; }

  bool requires_fields() const override { return true; }

  std::vector<std::string> required_field_names() const override {
    return {config_.error_field_name};
  }

private:
  Config config_;
};

/**
 * @brief Multi-criteria error estimator
 *
 * Combines multiple error estimators with weighted aggregation.
 */
class MultiCriteriaEstimator : public ErrorEstimator {
public:
  /**
   * @brief Add an estimator with weight
   */
  void add_estimator(std::unique_ptr<ErrorEstimator> estimator, double weight = 1.0);

  /**
   * @brief Set aggregation method
   */
  enum class AggregationMethod {
    WEIGHTED_SUM,     // Weighted sum of indicators
    WEIGHTED_MAX,     // Weighted maximum
    WEIGHTED_L2,      // Weighted L2 norm
    WEIGHTED_LP       // Weighted Lp norm
  };

  void set_aggregation_method(AggregationMethod method, double p = 2.0) {
    aggregation_method_ = method;
    p_norm_ = p;
  }

  std::vector<double> estimate(
      const MeshBase& mesh,
      const MeshFields* fields,
      const AdaptivityOptions& options) const override;

  std::string name() const override { return "MultiCriteria"; }

  bool requires_fields() const override;

  std::vector<std::string> required_field_names() const override;

private:
  std::vector<std::pair<std::unique_ptr<ErrorEstimator>, double>> estimators_;
  AggregationMethod aggregation_method_ = AggregationMethod::WEIGHTED_SUM;
  double p_norm_ = 2.0;

  /** Aggregate multiple indicator vectors */
  std::vector<double> aggregate_indicators(
      const std::vector<std::vector<double>>& indicators,
      const std::vector<double>& weights) const;
};

/**
 * @brief Factory for creating error estimators
 */
class ErrorEstimatorFactory {
public:
  /**
   * @brief Create an error estimator based on options
   */
  static std::unique_ptr<ErrorEstimator> create(const AdaptivityOptions& options);

  /**
   * @brief Create a gradient recovery estimator
   */
  static std::unique_ptr<ErrorEstimator> create_gradient_recovery(
      const GradientRecoveryEstimator::Config& config = {});

  /**
   * @brief Create a jump indicator estimator
   */
  static std::unique_ptr<ErrorEstimator> create_jump_indicator(
      const JumpIndicatorEstimator::Config& config = {});

  /**
   * @brief Create a user field estimator
   */
  static std::unique_ptr<ErrorEstimator> create_user_field(
      const std::string& field_name);

  /**
   * @brief Create a multi-criteria estimator
   */
  static std::unique_ptr<MultiCriteriaEstimator> create_multi_criteria();
};

/**
 * @brief Utility functions for error estimation
 */
class ErrorEstimatorUtils {
public:
  /**
   * @brief Normalize error indicators
   */
  static void normalize_indicators(std::vector<double>& indicators);

  /**
   * @brief Compute global error norm
   */
  static double compute_global_error(
      const std::vector<double>& indicators,
      double p = 2.0);

  /**
   * @brief Compute error statistics
   */
  struct ErrorStats {
    double min_error;
    double max_error;
    double mean_error;
    double std_dev;
    double total_error;
    size_t num_elements;
  };

  static ErrorStats compute_statistics(const std::vector<double>& indicators);

  /**
   * @brief Write error indicators to field
   */
  static void write_to_field(
      MeshBase& mesh,
      const std::string& field_name,
      const std::vector<double>& indicators);
};

} // namespace svmp

#endif // SVMP_ERROR_ESTIMATOR_H
