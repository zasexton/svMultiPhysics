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

#ifndef SVMP_FIELD_TRANSFER_H
#define SVMP_FIELD_TRANSFER_H

#include "Marker.h"
#include "Options.h"
#include "RefinementRules.h"
#include <array>
#include <map>
#include <memory>
#include <string>
#include <vector>

namespace svmp {

// Forward declarations
class MeshBase;
class MeshFields;

/**
 * @brief Parent-child relationship for field transfer
 */
struct ParentChildMap {
  /** Map from child element to parent element */
  std::vector<size_t> child_to_parent;

  /** Map from parent element to children */
  std::map<size_t, std::vector<size_t>> parent_to_children;

  /** Map from child vertex to parent vertices (for interpolation) */
  std::map<size_t, std::vector<std::pair<size_t, double>>> child_vertex_weights;

  /** Map from parent vertex to child vertices */
  std::map<size_t, std::vector<size_t>> parent_vertex_to_children;

  /** Refinement pattern used for each parent */
  std::map<size_t, AdaptivityOptions::RefinementPattern> parent_patterns;
};

/**
 * @brief Field transfer statistics
 */
struct TransferStats {
  /** Number of fields transferred */
  size_t num_fields = 0;

  /** Number of prolongation operations */
  size_t num_prolongations = 0;

  /** Number of restriction operations */
  size_t num_restrictions = 0;

  /** Conservation error for each field */
  std::map<std::string, double> conservation_errors;

  /** Maximum interpolation error */
  double max_interpolation_error = 0.0;

  /** Total transfer time */
  double transfer_time = 0.0;
};

/**
 * @brief Abstract base class for field transfer strategies
 */
class FieldTransfer {
public:
  virtual ~FieldTransfer() = default;

  /**
   * @brief Transfer fields from old mesh to new mesh
   *
   * @param old_mesh Source mesh
   * @param new_mesh Target mesh
   * @param old_fields Source fields
   * @param new_fields Target fields (created/updated)
   * @param parent_child Parent-child relationships
   * @param options Transfer options
   * @return Transfer statistics
   */
  virtual TransferStats transfer(
      const MeshBase& old_mesh,
      const MeshBase& new_mesh,
      const MeshFields& old_fields,
      MeshFields& new_fields,
      const ParentChildMap& parent_child,
      const AdaptivityOptions& options) const = 0;

  /**
   * @brief Prolongate field from coarse to fine (refinement)
   *
   * @param old_mesh Coarse mesh
   * @param new_mesh Fine mesh
   * @param old_field Coarse field values
   * @param new_field Fine field values (output)
   * @param parent_child Parent-child relationships
   */
  virtual void prolongate(
      const MeshBase& old_mesh,
      const MeshBase& new_mesh,
      const std::vector<double>& old_field,
      std::vector<double>& new_field,
      const ParentChildMap& parent_child) const = 0;

  /**
   * @brief Restrict field from fine to coarse (coarsening)
   *
   * @param old_mesh Fine mesh
   * @param new_mesh Coarse mesh
   * @param old_field Fine field values
   * @param new_field Coarse field values (output)
   * @param parent_child Parent-child relationships
   */
  virtual void restrict(
      const MeshBase& old_mesh,
      const MeshBase& new_mesh,
      const std::vector<double>& old_field,
      std::vector<double>& new_field,
      const ParentChildMap& parent_child) const = 0;

  /**
   * @brief Get transfer method name
   */
  virtual std::string name() const = 0;
};

/**
 * @brief Linear interpolation field transfer
 *
 * Uses linear interpolation for prolongation and averaging for restriction.
 */
class LinearInterpolationTransfer : public FieldTransfer {
public:
  /**
   * @brief Configuration for linear interpolation
   */
  struct Config {
    /** Use area/volume weighting for restriction */
    bool use_volume_weighting = true;

    /** Preserve boundary values exactly */
    bool preserve_boundary = true;

    /** Minimum interpolation weight */
    double min_weight = 1e-10;

    /** Check interpolation quality */
    bool check_quality = false;
  };

  LinearInterpolationTransfer() : LinearInterpolationTransfer(Config{}) {}
  explicit LinearInterpolationTransfer(const Config& config);

  TransferStats transfer(
      const MeshBase& old_mesh,
      const MeshBase& new_mesh,
      const MeshFields& old_fields,
      MeshFields& new_fields,
      const ParentChildMap& parent_child,
      const AdaptivityOptions& options) const override;

  void prolongate(
      const MeshBase& old_mesh,
      const MeshBase& new_mesh,
      const std::vector<double>& old_field,
      std::vector<double>& new_field,
      const ParentChildMap& parent_child) const override;

  void restrict(
      const MeshBase& old_mesh,
      const MeshBase& new_mesh,
      const std::vector<double>& old_field,
      std::vector<double>& new_field,
      const ParentChildMap& parent_child) const override;

  std::string name() const override { return "LinearInterpolation"; }

private:
  Config config_;

  /** Interpolate value at child vertex */
  double interpolate_at_vertex(
      const std::vector<double>& old_field,
      const std::vector<std::pair<size_t, double>>& weights) const;

  /** Average values from children */
  double average_from_children(
      const std::vector<double>& old_field,
      const std::vector<size_t>& children,
      const MeshBase& mesh) const;
};

/**
 * @brief Conservative field transfer
 *
 * Preserves integral quantities during transfer.
 */
class ConservativeTransfer : public FieldTransfer {
public:
  /**
   * @brief Configuration for conservative transfer
   */
  struct Config {
    /** Field quantities to conserve */
    enum class ConservedQuantity {
      INTEGRAL,     // Preserve integral over domain
      MASS,         // Preserve mass (density * volume)
      MOMENTUM,     // Preserve momentum
      ENERGY        // Preserve energy
    };

    ConservedQuantity quantity = ConservedQuantity::INTEGRAL;

    /** Tolerance for conservation check */
    double conservation_tolerance = 1e-10;

    /** Use high-order reconstruction */
    bool high_order_reconstruction = false;

    /** Maximum iterations for conservation enforcement */
    size_t max_conservation_iterations = 10;
  };

  ConservativeTransfer() : ConservativeTransfer(Config{}) {}
  explicit ConservativeTransfer(const Config& config);

  TransferStats transfer(
      const MeshBase& old_mesh,
      const MeshBase& new_mesh,
      const MeshFields& old_fields,
      MeshFields& new_fields,
      const ParentChildMap& parent_child,
      const AdaptivityOptions& options) const override;

  void prolongate(
      const MeshBase& old_mesh,
      const MeshBase& new_mesh,
      const std::vector<double>& old_field,
      std::vector<double>& new_field,
      const ParentChildMap& parent_child) const override;

  void restrict(
      const MeshBase& old_mesh,
      const MeshBase& new_mesh,
      const std::vector<double>& old_field,
      std::vector<double>& new_field,
      const ParentChildMap& parent_child) const override;

  std::string name() const override { return "Conservative"; }

private:
  Config config_;

  /** Compute integral of field over mesh */
  double compute_integral(
      const MeshBase& mesh,
      const std::vector<double>& field) const;

  /** Enforce conservation constraint */
  void enforce_conservation(
      const MeshBase& old_mesh,
      const MeshBase& new_mesh,
      const std::vector<double>& old_field,
      std::vector<double>& new_field) const;

  /** Reconstruct field in parent element */
  std::vector<double> reconstruct_in_parent(
      const MeshBase& mesh,
      size_t parent_elem,
      const std::vector<double>& field) const;
};

/**
 * @brief High-order field transfer
 *
 * Uses high-order polynomial reconstruction for accurate transfer.
 */
class HighOrderTransfer : public FieldTransfer {
public:
  /**
   * @brief Configuration for high-order transfer
   */
  struct Config {
    /** Polynomial order for reconstruction */
    size_t polynomial_order = 2;

    /** Use least-squares reconstruction */
    bool use_least_squares = true;

    /** Limit gradients to avoid oscillations */
    bool limit_gradients = true;

    /** Minimum stencil size for reconstruction */
    size_t min_stencil_size = 6;

    /** Weight function for least squares */
    enum class WeightFunction {
      UNIFORM,
      INVERSE_DISTANCE,
      GAUSSIAN
    };

    WeightFunction weight_function = WeightFunction::INVERSE_DISTANCE;
  };

  HighOrderTransfer() : HighOrderTransfer(Config{}) {}
  explicit HighOrderTransfer(const Config& config);

  TransferStats transfer(
      const MeshBase& old_mesh,
      const MeshBase& new_mesh,
      const MeshFields& old_fields,
      MeshFields& new_fields,
      const ParentChildMap& parent_child,
      const AdaptivityOptions& options) const override;

  void prolongate(
      const MeshBase& old_mesh,
      const MeshBase& new_mesh,
      const std::vector<double>& old_field,
      std::vector<double>& new_field,
      const ParentChildMap& parent_child) const override;

  void restrict(
      const MeshBase& old_mesh,
      const MeshBase& new_mesh,
      const std::vector<double>& old_field,
      std::vector<double>& new_field,
      const ParentChildMap& parent_child) const override;

  std::string name() const override { return "HighOrder"; }

private:
  Config config_;

  /** Build polynomial reconstruction */
  std::vector<double> build_polynomial(
      const MeshBase& mesh,
      size_t elem_id,
      const std::vector<double>& field) const;

  /** Evaluate polynomial at point */
  double evaluate_polynomial(
      const std::vector<double>& coefficients,
      const std::array<double, 3>& point) const;

  /** Apply gradient limiter */
  void apply_limiter(
      std::vector<double>& gradients,
      const MeshBase& mesh,
      size_t elem_id) const;
};

/**
 * @brief Injection field transfer
 *
 * Simple direct copying/injection without interpolation.
 */
class InjectionTransfer : public FieldTransfer {
public:
  TransferStats transfer(
      const MeshBase& old_mesh,
      const MeshBase& new_mesh,
      const MeshFields& old_fields,
      MeshFields& new_fields,
      const ParentChildMap& parent_child,
      const AdaptivityOptions& options) const override;

  void prolongate(
      const MeshBase& old_mesh,
      const MeshBase& new_mesh,
      const std::vector<double>& old_field,
      std::vector<double>& new_field,
      const ParentChildMap& parent_child) const override;

  void restrict(
      const MeshBase& old_mesh,
      const MeshBase& new_mesh,
      const std::vector<double>& old_field,
      std::vector<double>& new_field,
      const ParentChildMap& parent_child) const override;

  std::string name() const override { return "Injection"; }
};

/**
 * @brief Factory for creating field transfer objects
 */
class FieldTransferFactory {
public:
  /**
   * @brief Create field transfer based on options
   */
  static std::unique_ptr<FieldTransfer> create(const AdaptivityOptions& options);

  /**
   * @brief Create linear interpolation transfer
   */
  static std::unique_ptr<FieldTransfer> create_linear(
      const LinearInterpolationTransfer::Config& config = LinearInterpolationTransfer::Config{});

  /**
   * @brief Create conservative transfer
   */
  static std::unique_ptr<FieldTransfer> create_conservative(
      const ConservativeTransfer::Config& config = ConservativeTransfer::Config{});

  /**
   * @brief Create high-order transfer
   */
  static std::unique_ptr<FieldTransfer> create_high_order(
      const HighOrderTransfer::Config& config = HighOrderTransfer::Config{});

  /**
   * @brief Create injection transfer
   */
  static std::unique_ptr<FieldTransfer> create_injection();
};

/**
 * @brief Utility functions for field transfer
 */
class FieldTransferUtils {
public:
  /**
   * @brief Build parent-child map from refinement information
   */
  static ParentChildMap build_parent_child_map(
      const MeshBase& old_mesh,
      const MeshBase& new_mesh,
      const std::vector<MarkType>& marks);

  /**
   * @brief Check field conservation
   */
  static double check_conservation(
      const MeshBase& old_mesh,
      const MeshBase& new_mesh,
      const std::vector<double>& old_field,
      const std::vector<double>& new_field);

  /**
   * @brief Compute interpolation error
   */
  static double compute_interpolation_error(
      const MeshBase& mesh,
      const std::vector<double>& exact_field,
      const std::vector<double>& interpolated_field);

  /**
   * @brief Project field between meshes (general case)
   */
  static void project_field(
      const MeshBase& source_mesh,
      const MeshBase& target_mesh,
      const std::vector<double>& source_field,
      std::vector<double>& target_field);

  /**
   * @brief Transfer all fields between meshes
   */
  static TransferStats transfer_all_fields(
      const MeshBase& old_mesh,
      const MeshBase& new_mesh,
      const MeshFields& old_fields,
      MeshFields& new_fields,
      const ParentChildMap& parent_child,
      const AdaptivityOptions& options);
};

} // namespace svmp

#endif // SVMP_FIELD_TRANSFER_H
