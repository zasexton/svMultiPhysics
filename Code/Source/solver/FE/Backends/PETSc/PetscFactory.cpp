/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#include "Backends/PETSc/PetscFactory.h"

#if defined(FE_HAS_PETSC)

#include "Backends/PETSc/PetscLinearSolver.h"
#include "Backends/PETSc/PetscMatrix.h"
#include "Backends/PETSc/PetscVector.h"
#include "Sparsity/DistributedSparsityPattern.h"

namespace svmp {
namespace FE {
namespace backends {

std::unique_ptr<GenericMatrix>
PetscFactory::createMatrix(const sparsity::SparsityPattern& sparsity) const
{
    cached_ = CachedGhosting{};
    return std::make_unique<PetscMatrix>(sparsity);
}

std::unique_ptr<GenericMatrix>
PetscFactory::createMatrix(const sparsity::DistributedSparsityPattern& sparsity) const
{
    auto mat = std::make_unique<PetscMatrix>(sparsity);

    cached_.global_size = sparsity.globalRows();
    cached_.local_owned_size = sparsity.ownedRows().size();
    cached_.ghost_global_indices.assign(sparsity.getGhostColMap().begin(), sparsity.getGhostColMap().end());
    cached_.valid = true;

    return mat;
}

std::unique_ptr<GenericVector> PetscFactory::createVector(GlobalIndex size) const
{
    if (cached_.valid && cached_.global_size == size && cached_.local_owned_size >= 0) {
        return std::make_unique<PetscVector>(cached_.local_owned_size, cached_.global_size, cached_.ghost_global_indices);
    }
    return std::make_unique<PetscVector>(size);
}

std::unique_ptr<GenericVector>
PetscFactory::createVector(GlobalIndex local_size, GlobalIndex global_size) const
{
    if (cached_.valid && cached_.global_size == global_size && cached_.local_owned_size == local_size) {
        return std::make_unique<PetscVector>(local_size, global_size, cached_.ghost_global_indices);
    }
    return std::make_unique<PetscVector>(local_size, global_size);
}

std::unique_ptr<LinearSolver>
PetscFactory::createLinearSolver(const SolverOptions& options) const
{
    return std::make_unique<PetscLinearSolver>(options);
}

} // namespace backends
} // namespace FE
} // namespace svmp

#endif // FE_HAS_PETSC
