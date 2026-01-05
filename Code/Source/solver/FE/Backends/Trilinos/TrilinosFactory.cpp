/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#include "Backends/Trilinos/TrilinosFactory.h"

#if defined(FE_HAS_TRILINOS)

#include "Backends/Trilinos/TrilinosLinearSolver.h"
#include "Backends/Trilinos/TrilinosMatrix.h"
#include "Backends/Trilinos/TrilinosVector.h"
#include "Sparsity/DistributedSparsityPattern.h"

namespace svmp {
namespace FE {
namespace backends {

std::unique_ptr<GenericMatrix>
TrilinosFactory::createMatrix(const sparsity::SparsityPattern& sparsity) const
{
    cached_ = CachedGhosting{};
    return std::make_unique<TrilinosMatrix>(sparsity);
}

std::unique_ptr<GenericMatrix>
TrilinosFactory::createMatrix(const sparsity::DistributedSparsityPattern& sparsity) const
{
    auto mat = std::make_unique<TrilinosMatrix>(sparsity);

    cached_.global_size = sparsity.globalRows();
    cached_.owned_first = sparsity.ownedRows().first;
    cached_.local_owned_size = sparsity.ownedRows().size();
    cached_.ghost_global_indices.assign(sparsity.getGhostColMap().begin(), sparsity.getGhostColMap().end());
    cached_.valid = true;

    return mat;
}

std::unique_ptr<GenericVector> TrilinosFactory::createVector(GlobalIndex size) const
{
    if (cached_.valid && cached_.global_size == size) {
        return std::make_unique<TrilinosVector>(cached_.owned_first,
                                                cached_.local_owned_size,
                                                cached_.global_size,
                                                cached_.ghost_global_indices);
    }
    return std::make_unique<TrilinosVector>(size);
}

std::unique_ptr<GenericVector>
TrilinosFactory::createVector(GlobalIndex local_size, GlobalIndex global_size) const
{
    if (cached_.valid && cached_.global_size == global_size && cached_.local_owned_size == local_size) {
        return std::make_unique<TrilinosVector>(cached_.owned_first,
                                                local_size,
                                                global_size,
                                                cached_.ghost_global_indices);
    }
    return std::make_unique<TrilinosVector>(local_size, global_size);
}

std::unique_ptr<LinearSolver>
TrilinosFactory::createLinearSolver(const SolverOptions& options) const
{
    return std::make_unique<TrilinosLinearSolver>(options);
}

} // namespace backends
} // namespace FE
} // namespace svmp

#endif // FE_HAS_TRILINOS
