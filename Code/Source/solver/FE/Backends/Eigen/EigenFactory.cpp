/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#include "Backends/Eigen/EigenFactory.h"

#include "Backends/Eigen/EigenLinearSolver.h"
#include "Backends/Eigen/EigenMatrix.h"
#include "Backends/Eigen/EigenVector.h"

namespace svmp {
namespace FE {
namespace backends {

#if defined(FE_HAS_EIGEN)

std::unique_ptr<GenericMatrix>
EigenFactory::createMatrix(const sparsity::SparsityPattern& sparsity) const
{
    return std::make_unique<EigenMatrix>(sparsity);
}

std::unique_ptr<GenericVector> EigenFactory::createVector(GlobalIndex size) const
{
    return std::make_unique<EigenVector>(size);
}

std::unique_ptr<LinearSolver>
EigenFactory::createLinearSolver(const SolverOptions& options) const
{
    return std::make_unique<EigenLinearSolver>(options);
}

#endif // FE_HAS_EIGEN

} // namespace backends
} // namespace FE
} // namespace svmp

