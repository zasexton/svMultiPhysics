/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#ifndef SVMP_FE_BACKENDS_BACKEND_FACTORY_H
#define SVMP_FE_BACKENDS_BACKEND_FACTORY_H

#include "Backends/Interfaces/BackendKind.h"
#include "Backends/Interfaces/GenericMatrix.h"
#include "Backends/Interfaces/GenericVector.h"
#include "Backends/Interfaces/LinearSolver.h"
#include "Backends/Utils/BackendOptions.h"

#include "Sparsity/SparsityPattern.h"

#include <memory>
#include <string_view>

namespace svmp {
namespace FE {
namespace sparsity {
class DistributedSparsityPattern;
} // namespace sparsity

namespace backends {

class BackendFactory {
public:
    struct CreateOptions {
        // Backend-specific options.
        // - FSILS: interpreted as dof-per-node block size for matrix storage.
        int dof_per_node{1};
    };

    virtual ~BackendFactory() = default;

    [[nodiscard]] virtual BackendKind backendKind() const noexcept = 0;

    [[nodiscard]] virtual std::unique_ptr<GenericMatrix>
    createMatrix(const sparsity::SparsityPattern& sparsity) const = 0;

    [[nodiscard]] virtual std::unique_ptr<GenericMatrix>
    createMatrix(const sparsity::DistributedSparsityPattern& sparsity) const;

    [[nodiscard]] virtual std::unique_ptr<GenericVector> createVector(GlobalIndex size) const = 0;

    [[nodiscard]] virtual std::unique_ptr<GenericVector>
    createVector(GlobalIndex local_size, GlobalIndex global_size) const;

    [[nodiscard]] virtual std::unique_ptr<LinearSolver>
    createLinearSolver(const SolverOptions& options) const = 0;

    [[nodiscard]] static std::unique_ptr<BackendFactory> create(BackendKind kind);
    [[nodiscard]] static std::unique_ptr<BackendFactory> create(std::string_view name);
    [[nodiscard]] static std::unique_ptr<BackendFactory> create(BackendKind kind, const CreateOptions& options);
    [[nodiscard]] static std::unique_ptr<BackendFactory> create(std::string_view name, const CreateOptions& options);
};

} // namespace backends
} // namespace FE
} // namespace svmp

#endif // SVMP_FE_BACKENDS_BACKEND_FACTORY_H
