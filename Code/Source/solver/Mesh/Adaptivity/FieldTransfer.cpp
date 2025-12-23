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

#include "FieldTransfer.h"
#include <algorithm>
#include <cmath>
#include <numeric>
#include <stdexcept>

namespace svmp {

LinearInterpolationTransfer::LinearInterpolationTransfer(const Config& config)
    : config_(config) {}

TransferStats LinearInterpolationTransfer::transfer(
    const MeshBase& old_mesh,
    const MeshBase& new_mesh,
    const MeshFields& old_fields,
    MeshFields& new_fields,
    const ParentChildMap& parent_child,
    const AdaptivityOptions& options) const {
  (void)old_mesh;
  (void)new_mesh;
  (void)old_fields;
  (void)new_fields;
  (void)parent_child;
  (void)options;
  return {};
}

void LinearInterpolationTransfer::prolongate(
    const MeshBase& old_mesh,
    const MeshBase& new_mesh,
    const std::vector<double>& old_field,
    std::vector<double>& new_field,
    const ParentChildMap& parent_child) const {
  (void)old_mesh;
  (void)new_mesh;
  (void)parent_child;

  if (new_field.empty()) {
    new_field = old_field;
    return;
  }

  const size_t n = std::min(old_field.size(), new_field.size());
  std::fill(new_field.begin(), new_field.end(), 0.0);
  std::copy(old_field.begin(), old_field.begin() + static_cast<ptrdiff_t>(n), new_field.begin());
}

void LinearInterpolationTransfer::restrict(
    const MeshBase& old_mesh,
    const MeshBase& new_mesh,
    const std::vector<double>& old_field,
    std::vector<double>& new_field,
    const ParentChildMap& parent_child) const {
  prolongate(old_mesh, new_mesh, old_field, new_field, parent_child);
}

double LinearInterpolationTransfer::interpolate_at_vertex(
    const std::vector<double>& old_field,
    const std::vector<std::pair<size_t, double>>& weights) const {
  double value = 0.0;
  for (const auto& [idx, w] : weights) {
    if (idx < old_field.size()) {
      value += w * old_field[idx];
    }
  }
  return value;
}

double LinearInterpolationTransfer::average_from_children(
    const std::vector<double>& old_field,
    const std::vector<size_t>& children,
    const MeshBase& mesh) const {
  (void)mesh;
  if (children.empty()) {
    return 0.0;
  }
  double sum = 0.0;
  size_t count = 0;
  for (auto c : children) {
    if (c < old_field.size()) {
      sum += old_field[c];
      ++count;
    }
  }
  return count > 0 ? (sum / static_cast<double>(count)) : 0.0;
}

ConservativeTransfer::ConservativeTransfer(const Config& config)
    : config_(config) {}

TransferStats ConservativeTransfer::transfer(
    const MeshBase& old_mesh,
    const MeshBase& new_mesh,
    const MeshFields& old_fields,
    MeshFields& new_fields,
    const ParentChildMap& parent_child,
    const AdaptivityOptions& options) const {
  (void)old_mesh;
  (void)new_mesh;
  (void)old_fields;
  (void)new_fields;
  (void)parent_child;
  (void)options;
  return {};
}

void ConservativeTransfer::prolongate(
    const MeshBase& old_mesh,
    const MeshBase& new_mesh,
    const std::vector<double>& old_field,
    std::vector<double>& new_field,
    const ParentChildMap& parent_child) const {
  (void)parent_child;
  LinearInterpolationTransfer::Config cfg;
  LinearInterpolationTransfer linear(cfg);
  linear.prolongate(old_mesh, new_mesh, old_field, new_field, parent_child);
}

void ConservativeTransfer::restrict(
    const MeshBase& old_mesh,
    const MeshBase& new_mesh,
    const std::vector<double>& old_field,
    std::vector<double>& new_field,
    const ParentChildMap& parent_child) const {
  prolongate(old_mesh, new_mesh, old_field, new_field, parent_child);
}

double ConservativeTransfer::compute_integral(
    const MeshBase& mesh,
    const std::vector<double>& field) const {
  (void)mesh;
  return std::accumulate(field.begin(), field.end(), 0.0);
}

void ConservativeTransfer::enforce_conservation(
    const MeshBase& old_mesh,
    const MeshBase& new_mesh,
    const std::vector<double>& old_field,
    std::vector<double>& new_field) const {
  (void)old_mesh;
  (void)new_mesh;
  (void)old_field;
  (void)new_field;
}

std::vector<double> ConservativeTransfer::reconstruct_in_parent(
    const MeshBase& mesh,
    size_t parent_elem,
    const std::vector<double>& field) const {
  (void)mesh;
  (void)parent_elem;
  return field;
}

HighOrderTransfer::HighOrderTransfer(const Config& config)
    : config_(config) {}

TransferStats HighOrderTransfer::transfer(
    const MeshBase& old_mesh,
    const MeshBase& new_mesh,
    const MeshFields& old_fields,
    MeshFields& new_fields,
    const ParentChildMap& parent_child,
    const AdaptivityOptions& options) const {
  (void)old_mesh;
  (void)new_mesh;
  (void)old_fields;
  (void)new_fields;
  (void)parent_child;
  (void)options;
  return {};
}

void HighOrderTransfer::prolongate(
    const MeshBase& old_mesh,
    const MeshBase& new_mesh,
    const std::vector<double>& old_field,
    std::vector<double>& new_field,
    const ParentChildMap& parent_child) const {
  (void)parent_child;
  LinearInterpolationTransfer::Config cfg;
  LinearInterpolationTransfer linear(cfg);
  linear.prolongate(old_mesh, new_mesh, old_field, new_field, parent_child);
}

void HighOrderTransfer::restrict(
    const MeshBase& old_mesh,
    const MeshBase& new_mesh,
    const std::vector<double>& old_field,
    std::vector<double>& new_field,
    const ParentChildMap& parent_child) const {
  prolongate(old_mesh, new_mesh, old_field, new_field, parent_child);
}

std::vector<double> HighOrderTransfer::build_polynomial(
    const MeshBase& mesh,
    size_t elem_id,
    const std::vector<double>& field) const {
  (void)mesh;
  (void)elem_id;
  (void)field;
  return {};
}

double HighOrderTransfer::evaluate_polynomial(
    const std::vector<double>& coefficients,
    const std::array<double, 3>& point) const {
  (void)coefficients;
  (void)point;
  return 0.0;
}

void HighOrderTransfer::apply_limiter(
    std::vector<double>& gradients,
    const MeshBase& mesh,
    size_t elem_id) const {
  (void)gradients;
  (void)mesh;
  (void)elem_id;
}

TransferStats InjectionTransfer::transfer(
    const MeshBase& old_mesh,
    const MeshBase& new_mesh,
    const MeshFields& old_fields,
    MeshFields& new_fields,
    const ParentChildMap& parent_child,
    const AdaptivityOptions& options) const {
  (void)old_mesh;
  (void)new_mesh;
  (void)old_fields;
  (void)new_fields;
  (void)parent_child;
  (void)options;
  return {};
}

void InjectionTransfer::prolongate(
    const MeshBase& old_mesh,
    const MeshBase& new_mesh,
    const std::vector<double>& old_field,
    std::vector<double>& new_field,
    const ParentChildMap& parent_child) const {
  (void)old_mesh;
  (void)new_mesh;
  (void)parent_child;
  new_field = old_field;
}

void InjectionTransfer::restrict(
    const MeshBase& old_mesh,
    const MeshBase& new_mesh,
    const std::vector<double>& old_field,
    std::vector<double>& new_field,
    const ParentChildMap& parent_child) const {
  prolongate(old_mesh, new_mesh, old_field, new_field, parent_child);
}

std::unique_ptr<FieldTransfer> FieldTransferFactory::create(const AdaptivityOptions& options) {
  // Prefer the high-level selection when explicitly set (tests use this).
  if (options.field_transfer != FieldTransferType::LINEAR_INTERPOLATION) {
    switch (options.field_transfer) {
      case FieldTransferType::INJECTION:
        return create_injection();
      case FieldTransferType::HIGH_ORDER:
        return create_high_order();
      case FieldTransferType::CONSERVATIVE:
        return create_conservative();
      case FieldTransferType::LINEAR_INTERPOLATION:
      default:
        break;
    }
  }

  // Backward compatibility: select based on legacy prolongation choice.
  switch (options.prolongation_method) {
    case AdaptivityOptions::ProlongationMethod::COPY:
      return create_injection();
    case AdaptivityOptions::ProlongationMethod::HIGH_ORDER_INTERP:
      return create_high_order();
    case AdaptivityOptions::ProlongationMethod::CONSERVATIVE:
      return create_conservative();
    case AdaptivityOptions::ProlongationMethod::LINEAR_INTERP:
    default:
      return create_linear();
  }
}

std::unique_ptr<FieldTransfer> FieldTransferFactory::create_linear(
    const LinearInterpolationTransfer::Config& config) {
  return std::make_unique<LinearInterpolationTransfer>(config);
}

std::unique_ptr<FieldTransfer> FieldTransferFactory::create_conservative(
    const ConservativeTransfer::Config& config) {
  return std::make_unique<ConservativeTransfer>(config);
}

std::unique_ptr<FieldTransfer> FieldTransferFactory::create_high_order(
    const HighOrderTransfer::Config& config) {
  return std::make_unique<HighOrderTransfer>(config);
}

std::unique_ptr<FieldTransfer> FieldTransferFactory::create_injection() {
  return std::make_unique<InjectionTransfer>();
}

ParentChildMap FieldTransferUtils::build_parent_child_map(
    const MeshBase& old_mesh,
    const MeshBase& new_mesh,
    const std::vector<MarkType>& marks) {
  (void)old_mesh;
  (void)new_mesh;
  (void)marks;
  return {};
}

double FieldTransferUtils::check_conservation(
    const MeshBase& old_mesh,
    const MeshBase& new_mesh,
    const std::vector<double>& old_field,
    const std::vector<double>& new_field) {
  (void)old_mesh;
  (void)new_mesh;
  const double old_sum = std::accumulate(old_field.begin(), old_field.end(), 0.0);
  const double new_sum = std::accumulate(new_field.begin(), new_field.end(), 0.0);
  return std::abs(new_sum - old_sum);
}

double FieldTransferUtils::compute_interpolation_error(
    const MeshBase& mesh,
    const std::vector<double>& exact_field,
    const std::vector<double>& interpolated_field) {
  (void)mesh;
  const size_t n = std::min(exact_field.size(), interpolated_field.size());
  double max_err = 0.0;
  for (size_t i = 0; i < n; ++i) {
    max_err = std::max(max_err, std::abs(interpolated_field[i] - exact_field[i]));
  }
  return max_err;
}

void FieldTransferUtils::project_field(
    const MeshBase& source_mesh,
    const MeshBase& target_mesh,
    const std::vector<double>& source_field,
    std::vector<double>& target_field) {
  (void)source_mesh;
  (void)target_mesh;
  const size_t n = std::min(source_field.size(), target_field.size());
  std::copy(source_field.begin(), source_field.begin() + static_cast<ptrdiff_t>(n), target_field.begin());
  if (target_field.size() > n) {
    std::fill(target_field.begin() + static_cast<ptrdiff_t>(n), target_field.end(), 0.0);
  }
}

TransferStats FieldTransferUtils::transfer_all_fields(
    const MeshBase& old_mesh,
    const MeshBase& new_mesh,
    const MeshFields& old_fields,
    MeshFields& new_fields,
    const ParentChildMap& parent_child,
    const AdaptivityOptions& options) {
  (void)old_mesh;
  (void)new_mesh;
  (void)old_fields;
  (void)new_fields;
  (void)parent_child;
  (void)options;
  return {};
}

} // namespace svmp
