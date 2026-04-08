#ifndef SV_FE_LS_DISTRIBUTED_SPARSE_OPERATOR_H
#define SV_FE_LS_DISTRIBUTED_SPARSE_OPERATOR_H

#include "fsils_api.hpp"

namespace fe_fsi_linear_solver::distributed_sparse_operator {

enum class VectorState {
  owned_only,
  ghost_zeroed,
  ghost_synced,
};

[[nodiscard]] const char* to_string(VectorState state) noexcept;

struct UnaryOperatorContract {
  VectorState input_state{VectorState::ghost_synced};
  VectorState output_state{VectorState::ghost_synced};
};

struct FusedOperatorContract {
  VectorState input_state{VectorState::ghost_synced};
  VectorState first_output_state{VectorState::ghost_synced};
  VectorState second_output_state{VectorState::ghost_synced};
};

struct ScalarInput {
  const Vector<double>& values;
  VectorState state{VectorState::owned_only};
};

struct ScalarOutput {
  Vector<double>& values;
  VectorState state{VectorState::owned_only};
};

struct BlockInput {
  const Array<double>& values;
  int components{0};
  VectorState state{VectorState::owned_only};
};

struct BlockOutput {
  Array<double>& values;
  int components{0};
  VectorState state{VectorState::owned_only};
};

[[nodiscard]] inline ScalarInput scalar_input(const Vector<double>& values,
                                              VectorState state) noexcept
{
  return ScalarInput{values, state};
}

[[nodiscard]] inline ScalarOutput scalar_output(Vector<double>& values,
                                                VectorState state) noexcept
{
  return ScalarOutput{values, state};
}

[[nodiscard]] inline BlockInput block_input(int components,
                                            const Array<double>& values,
                                            VectorState state) noexcept
{
  return BlockInput{values, components, state};
}

[[nodiscard]] inline BlockOutput block_output(int components,
                                              Array<double>& values,
                                              VectorState state) noexcept
{
  return BlockOutput{values, components, state};
}

[[nodiscard]] inline ScalarInput ghost_synced_input(const Vector<double>& values) noexcept
{
  return scalar_input(values, VectorState::ghost_synced);
}

[[nodiscard]] inline ScalarOutput ghost_synced_output(Vector<double>& values) noexcept
{
  return scalar_output(values, VectorState::ghost_synced);
}

[[nodiscard]] inline BlockInput ghost_synced_input(int components,
                                                   const Array<double>& values) noexcept
{
  return block_input(components, values, VectorState::ghost_synced);
}

[[nodiscard]] inline BlockOutput ghost_synced_output(int components,
                                                     Array<double>& values) noexcept
{
  return block_output(components, values, VectorState::ghost_synced);
}

[[nodiscard]] inline ScalarInput owned_only_input(const Vector<double>& values) noexcept
{
  return scalar_input(values, VectorState::owned_only);
}

[[nodiscard]] inline ScalarOutput owned_only_output(Vector<double>& values) noexcept
{
  return scalar_output(values, VectorState::owned_only);
}

[[nodiscard]] inline BlockInput owned_only_input(int components,
                                                 const Array<double>& values) noexcept
{
  return block_input(components, values, VectorState::owned_only);
}

[[nodiscard]] inline BlockOutput owned_only_output(int components,
                                                   Array<double>& values) noexcept
{
  return block_output(components, values, VectorState::owned_only);
}

[[nodiscard]] inline ScalarOutput ghost_zeroed_output(Vector<double>& values) noexcept
{
  return scalar_output(values, VectorState::ghost_zeroed);
}

[[nodiscard]] inline BlockOutput ghost_zeroed_output(int components,
                                                     Array<double>& values) noexcept
{
  return block_output(components, values, VectorState::ghost_zeroed);
}

class ScalarToScalarOperator {
 public:
  ScalarToScalarOperator(const FSILS_lhsType& lhs,
                         const Array<fsils_int>& row_ptr,
                         const Vector<fsils_int>& col_ptr,
                         const Vector<double>& values) noexcept;

  [[nodiscard]] UnaryOperatorContract contract() const noexcept;
  void apply(ScalarInput input, ScalarOutput output) const;

 private:
  const FSILS_lhsType* lhs_{nullptr};
  const Array<fsils_int>* row_ptr_{nullptr};
  const Vector<fsils_int>* col_ptr_{nullptr};
  const Vector<double>* values_{nullptr};
};

class ScalarToVectorOperator {
 public:
  ScalarToVectorOperator(const FSILS_lhsType& lhs,
                         const Array<fsils_int>& row_ptr,
                         const Vector<fsils_int>& col_ptr,
                         int components,
                         const Array<double>& values) noexcept;

  [[nodiscard]] UnaryOperatorContract contract() const noexcept;
  void apply(ScalarInput input, BlockOutput output) const;

 private:
  const FSILS_lhsType* lhs_{nullptr};
  const Array<fsils_int>* row_ptr_{nullptr};
  const Vector<fsils_int>* col_ptr_{nullptr};
  int components_{0};
  const Array<double>* values_{nullptr};
};

class VectorToScalarOperator {
 public:
  VectorToScalarOperator(const FSILS_lhsType& lhs,
                         const Array<fsils_int>& row_ptr,
                         const Vector<fsils_int>& col_ptr,
                         int components,
                         const Array<double>& values) noexcept;

  [[nodiscard]] UnaryOperatorContract contract() const noexcept;
  void apply(BlockInput input, ScalarOutput output) const;

 private:
  const FSILS_lhsType* lhs_{nullptr};
  const Array<fsils_int>* row_ptr_{nullptr};
  const Vector<fsils_int>* col_ptr_{nullptr};
  int components_{0};
  const Array<double>* values_{nullptr};
};

class VectorToVectorOperator {
 public:
  VectorToVectorOperator(const FSILS_lhsType& lhs,
                         const Array<fsils_int>& row_ptr,
                         const Vector<fsils_int>& col_ptr,
                         int components,
                         const Array<double>& values) noexcept;

  [[nodiscard]] UnaryOperatorContract contract() const noexcept;
  void apply(BlockInput input, BlockOutput output) const;

 private:
  const FSILS_lhsType* lhs_{nullptr};
  const Array<fsils_int>* row_ptr_{nullptr};
  const Vector<fsils_int>* col_ptr_{nullptr};
  int components_{0};
  const Array<double>* values_{nullptr};
};

class RectangularOperator {
 public:
  RectangularOperator(const FSILS_lhsType& lhs,
                      const Array<fsils_int>& row_ptr,
                      const Vector<fsils_int>& col_ptr,
                      int out_components,
                      int in_components,
                      const Array<double>& values) noexcept;

  [[nodiscard]] UnaryOperatorContract contract() const noexcept;
  void apply(BlockInput input, BlockOutput output) const;

 private:
  const FSILS_lhsType* lhs_{nullptr};
  const Array<fsils_int>* row_ptr_{nullptr};
  const Vector<fsils_int>* col_ptr_{nullptr};
  int out_components_{0};
  int in_components_{0};
  const Array<double>* values_{nullptr};
};

class FusedScalarConstraintOperator {
 public:
  FusedScalarConstraintOperator(const FSILS_lhsType& lhs,
                                const Array<fsils_int>& row_ptr,
                                const Vector<fsils_int>& col_ptr,
                                int momentum_components,
                                const Array<double>& momentum_values,
                                const Vector<double>& constraint_values) noexcept;

  [[nodiscard]] FusedOperatorContract contract() const noexcept;
  void apply(ScalarInput input,
             BlockOutput momentum_output,
             ScalarOutput constraint_output) const;

 private:
  const FSILS_lhsType* lhs_{nullptr};
  const Array<fsils_int>* row_ptr_{nullptr};
  const Vector<fsils_int>* col_ptr_{nullptr};
  int momentum_components_{0};
  const Array<double>* momentum_values_{nullptr};
  const Vector<double>* constraint_values_{nullptr};
};

class FusedRectangularConstraintOperator {
 public:
  FusedRectangularConstraintOperator(const FSILS_lhsType& lhs,
                                     const Array<fsils_int>& row_ptr,
                                     const Vector<fsils_int>& col_ptr,
                                     int momentum_components,
                                     int constraint_components,
                                     const Array<double>& momentum_values,
                                     const Array<double>& constraint_values) noexcept;

  [[nodiscard]] FusedOperatorContract contract() const noexcept;
  void apply(BlockInput input,
             BlockOutput momentum_output,
             BlockOutput constraint_output) const;

 private:
  const FSILS_lhsType* lhs_{nullptr};
  const Array<fsils_int>* row_ptr_{nullptr};
  const Vector<fsils_int>* col_ptr_{nullptr};
  int momentum_components_{0};
  int constraint_components_{0};
  const Array<double>* momentum_values_{nullptr};
  const Array<double>* constraint_values_{nullptr};
};

class SparseOperatorBundle {
 public:
  SparseOperatorBundle(const FSILS_lhsType& lhs,
                       const Array<fsils_int>& row_ptr,
                       const Vector<fsils_int>& col_ptr) noexcept;

  [[nodiscard]] ScalarToScalarOperator scalar(const Vector<double>& values) const noexcept;
  [[nodiscard]] ScalarToVectorOperator scalar_to_vector(int components,
                                                        const Array<double>& values) const noexcept;
  [[nodiscard]] VectorToScalarOperator vector_to_scalar(int components,
                                                        const Array<double>& values) const noexcept;
  [[nodiscard]] VectorToVectorOperator vector(int components,
                                              const Array<double>& values) const noexcept;
  [[nodiscard]] RectangularOperator rectangular(int out_components,
                                                int in_components,
                                                const Array<double>& values) const noexcept;
  [[nodiscard]] FusedScalarConstraintOperator fused_scalar_constraint(
      int momentum_components,
      const Array<double>& momentum_values,
      const Vector<double>& constraint_values) const noexcept;
  [[nodiscard]] FusedRectangularConstraintOperator fused_rectangular_constraint(
      int momentum_components,
      int constraint_components,
      const Array<double>& momentum_values,
      const Array<double>& constraint_values) const noexcept;

 private:
  const FSILS_lhsType* lhs_{nullptr};
  const Array<fsils_int>* row_ptr_{nullptr};
  const Vector<fsils_int>* col_ptr_{nullptr};
};

}  // namespace fe_fsi_linear_solver::distributed_sparse_operator

#endif  // SV_FE_LS_DISTRIBUTED_SPARSE_OPERATOR_H
