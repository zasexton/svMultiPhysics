#ifndef SV_FE_LS_DISTRIBUTED_SOLVER_BUNDLES_H
#define SV_FE_LS_DISTRIBUTED_SOLVER_BUNDLES_H

#include "distributed_sparse_operator.h"

namespace fe_fsi_linear_solver::distributed_solver_bundles {

namespace dso = distributed_sparse_operator;

struct ScalarLinearSystem {
  ScalarLinearSystem(FSILS_lhsType& lhs_in,
                     dso::ScalarToScalarOperator A_in) noexcept
      : lhs(&lhs_in), A(A_in)
  {
  }

  FSILS_lhsType* lhs;
  dso::ScalarToScalarOperator A;
};

struct VectorLinearSystem {
  VectorLinearSystem(FSILS_lhsType& lhs_in,
                     int components_in,
                     dso::VectorToVectorOperator A_in) noexcept
      : lhs(&lhs_in), components(components_in), A(A_in)
  {
  }

  FSILS_lhsType* lhs;
  int components;
  dso::VectorToVectorOperator A;
};

struct ScalarBlockSchurSystem {
  ScalarBlockSchurSystem(FSILS_lhsType& lhs_in,
                         int momentum_components_in,
                         const Array<double>& momentum_values_in,
                         const Array<double>& D_values_in,
                         const Array<double>& G_values_in,
                         const Vector<double>& L_values_in,
                         VectorLinearSystem momentum_in,
                         dso::VectorToScalarOperator D_in,
                         dso::ScalarToVectorOperator G_in,
                         dso::ScalarToScalarOperator L_in,
                         dso::FusedScalarConstraintOperator GL_in) noexcept
      : lhs(&lhs_in),
        momentum_components(momentum_components_in),
        momentum_values(&momentum_values_in),
        D_values(&D_values_in),
        G_values(&G_values_in),
        L_values(&L_values_in),
        momentum(momentum_in),
        D(D_in),
        G(G_in),
        L(L_in),
        GL(GL_in)
  {
  }

  FSILS_lhsType* lhs;
  int momentum_components;
  const Array<double>* momentum_values;
  const Array<double>* D_values;
  const Array<double>* G_values;
  const Vector<double>* L_values;
  VectorLinearSystem momentum;
  dso::VectorToScalarOperator D;
  dso::ScalarToVectorOperator G;
  dso::ScalarToScalarOperator L;
  dso::FusedScalarConstraintOperator GL;
};

struct ScalarConstraintSchurSystem {
  ScalarConstraintSchurSystem(FSILS_lhsType& lhs_in,
                              int momentum_components_in,
                              const Array<double>& D_values_in,
                              const Array<double>& G_values_in,
                              const Vector<double>& L_values_in,
                              dso::VectorToScalarOperator D_in,
                              dso::ScalarToVectorOperator G_in,
                              dso::ScalarToScalarOperator L_in,
                              dso::FusedScalarConstraintOperator GL_in) noexcept
      : lhs(&lhs_in),
        momentum_components(momentum_components_in),
        D_values(&D_values_in),
        G_values(&G_values_in),
        L_values(&L_values_in),
        D(D_in),
        G(G_in),
        L(L_in),
        GL(GL_in)
  {
  }

  FSILS_lhsType* lhs;
  int momentum_components;
  const Array<double>* D_values;
  const Array<double>* G_values;
  const Vector<double>* L_values;
  dso::VectorToScalarOperator D;
  dso::ScalarToVectorOperator G;
  dso::ScalarToScalarOperator L;
  dso::FusedScalarConstraintOperator GL;
};

struct MultiConstraintSchurSystem {
  MultiConstraintSchurSystem(FSILS_lhsType& lhs_in,
                             int momentum_components_in,
                             int constraint_components_in,
                             const Array<double>& D_values_in,
                             const Array<double>& G_values_in,
                             const Array<double>& L_values_in,
                             dso::RectangularOperator D_in,
                             dso::RectangularOperator G_in,
                             dso::VectorToVectorOperator L_in,
                             dso::FusedRectangularConstraintOperator GL_in) noexcept
      : lhs(&lhs_in),
        momentum_components(momentum_components_in),
        constraint_components(constraint_components_in),
        D_values(&D_values_in),
        G_values(&G_values_in),
        L_values(&L_values_in),
        D(D_in),
        G(G_in),
        L(L_in),
        GL(GL_in)
  {
  }

  FSILS_lhsType* lhs;
  int momentum_components;
  int constraint_components;
  const Array<double>* D_values;
  const Array<double>* G_values;
  const Array<double>* L_values;
  dso::RectangularOperator D;
  dso::RectangularOperator G;
  dso::VectorToVectorOperator L;
  dso::FusedRectangularConstraintOperator GL;
};

struct MultiConstraintBlockSchurSystem {
  MultiConstraintBlockSchurSystem(FSILS_lhsType& lhs_in,
                                  int momentum_components_in,
                                  int constraint_components_in,
                                  const Array<double>& momentum_values_in,
                                  const Array<double>& D_values_in,
                                  const Array<double>& G_values_in,
                                  const Array<double>& L_values_in,
                                  VectorLinearSystem momentum_in,
                                  dso::RectangularOperator D_in,
                                  dso::RectangularOperator G_in,
                                  dso::VectorToVectorOperator L_in,
                                  dso::FusedRectangularConstraintOperator GL_in) noexcept
      : lhs(&lhs_in),
        momentum_components(momentum_components_in),
        constraint_components(constraint_components_in),
        momentum_values(&momentum_values_in),
        D_values(&D_values_in),
        G_values(&G_values_in),
        L_values(&L_values_in),
        momentum(momentum_in),
        D(D_in),
        G(G_in),
        L(L_in),
        GL(GL_in)
  {
  }

  FSILS_lhsType* lhs;
  int momentum_components;
  int constraint_components;
  const Array<double>* momentum_values;
  const Array<double>* D_values;
  const Array<double>* G_values;
  const Array<double>* L_values;
  VectorLinearSystem momentum;
  dso::RectangularOperator D;
  dso::RectangularOperator G;
  dso::VectorToVectorOperator L;
  dso::FusedRectangularConstraintOperator GL;
};

inline ScalarLinearSystem make_scalar_linear_system(FSILS_lhsType& lhs,
                                                    const Vector<double>& values)
{
  const auto ops = dso::SparseOperatorBundle(lhs, lhs.rowPtr, lhs.colPtr);
  return ScalarLinearSystem(lhs, ops.scalar(values));
}

inline VectorLinearSystem make_vector_linear_system(FSILS_lhsType& lhs,
                                                    int components,
                                                    const Array<double>& values)
{
  const auto ops = dso::SparseOperatorBundle(lhs, lhs.rowPtr, lhs.colPtr);
  return VectorLinearSystem(lhs, components, ops.vector(components, values));
}

inline ScalarConstraintSchurSystem make_scalar_constraint_schur_system(FSILS_lhsType& lhs,
                                                                       int momentum_components,
                                                                       const Array<double>& D,
                                                                       const Array<double>& G,
                                                                       const Vector<double>& L)
{
  const auto ops = dso::SparseOperatorBundle(lhs, lhs.rowPtr, lhs.colPtr);
  return ScalarConstraintSchurSystem(
      lhs, momentum_components, D, G, L,
      ops.vector_to_scalar(momentum_components, D),
      ops.scalar_to_vector(momentum_components, G),
      ops.scalar(L),
      ops.fused_scalar_constraint(momentum_components, G, L));
}

inline ScalarBlockSchurSystem make_scalar_block_schur_system(FSILS_lhsType& lhs,
                                                             int momentum_components,
                                                             const Array<double>& K,
                                                             const Array<double>& D,
                                                             const Array<double>& G,
                                                             const Vector<double>& L)
{
  const auto ops = dso::SparseOperatorBundle(lhs, lhs.rowPtr, lhs.colPtr);
  return ScalarBlockSchurSystem(
      lhs, momentum_components, K, D, G, L,
      make_vector_linear_system(lhs, momentum_components, K),
      ops.vector_to_scalar(momentum_components, D),
      ops.scalar_to_vector(momentum_components, G),
      ops.scalar(L),
      ops.fused_scalar_constraint(momentum_components, G, L));
}

inline MultiConstraintSchurSystem make_multi_constraint_schur_system(
    FSILS_lhsType& lhs,
    int momentum_components,
    int constraint_components,
    const Array<double>& D,
    const Array<double>& G,
    const Array<double>& L)
{
  const auto ops = dso::SparseOperatorBundle(lhs, lhs.rowPtr, lhs.colPtr);
  return MultiConstraintSchurSystem(
      lhs, momentum_components, constraint_components, D, G, L,
      ops.rectangular(constraint_components, momentum_components, D),
      ops.rectangular(momentum_components, constraint_components, G),
      ops.vector(constraint_components, L),
      ops.fused_rectangular_constraint(momentum_components, constraint_components, G, L));
}

inline MultiConstraintBlockSchurSystem make_multi_constraint_block_schur_system(
    FSILS_lhsType& lhs,
    int momentum_components,
    int constraint_components,
    const Array<double>& K,
    const Array<double>& D,
    const Array<double>& G,
    const Array<double>& L)
{
  const auto ops = dso::SparseOperatorBundle(lhs, lhs.rowPtr, lhs.colPtr);
  return MultiConstraintBlockSchurSystem(
      lhs, momentum_components, constraint_components, K, D, G, L,
      make_vector_linear_system(lhs, momentum_components, K),
      ops.rectangular(constraint_components, momentum_components, D),
      ops.rectangular(momentum_components, constraint_components, G),
      ops.vector(constraint_components, L),
      ops.fused_rectangular_constraint(momentum_components, constraint_components, G, L));
}

}  // namespace fe_fsi_linear_solver::distributed_solver_bundles

#endif  // SV_FE_LS_DISTRIBUTED_SOLVER_BUNDLES_H
