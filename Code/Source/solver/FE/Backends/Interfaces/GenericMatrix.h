/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#ifndef SVMP_FE_BACKENDS_GENERIC_MATRIX_H
#define SVMP_FE_BACKENDS_GENERIC_MATRIX_H

#include "Backends/Interfaces/BackendKind.h"
#include "Backends/Interfaces/GenericVector.h"
#include "Core/Types.h"

#include "Assembly/GlobalSystemView.h"

#include <memory>

namespace svmp {
namespace FE {
namespace sparsity {
class SparsityPattern;
class DistributedSparsityPattern;
} // namespace sparsity

namespace backends {

class GenericMatrix {
public:
    virtual ~GenericMatrix() = default;

    [[nodiscard]] virtual BackendKind backendKind() const noexcept = 0;
    [[nodiscard]] virtual GlobalIndex numRows() const noexcept = 0;
    [[nodiscard]] virtual GlobalIndex numCols() const noexcept = 0;

    virtual void zero() = 0;
    virtual void finalizeAssembly() = 0;

    virtual void mult(const GenericVector& x, GenericVector& y) const = 0;
    virtual void multAdd(const GenericVector& x, GenericVector& y) const = 0;

    [[nodiscard]] virtual std::unique_ptr<assembly::GlobalSystemView> createAssemblyView() = 0;

    [[nodiscard]] virtual Real getEntry(GlobalIndex row, GlobalIndex col) const
    {
        (void)row;
        (void)col;
        return 0.0;
    }

    /**
     * @brief Rebuild matrix storage in place from a new sparsity pattern.
     *
     * Preserves object identity so long-lived references stay valid — needed
     * when interface-tracking constraints re-augment the system sparsity in
     * the middle of a nonlinear solve. Entries are zeroed. Returns false when
     * the backend does not support in-place reinitialization (callers must
     * then replace the matrix object instead).
     */
    virtual bool reinitFromPattern(const sparsity::SparsityPattern& pattern)
    {
        (void)pattern;
        return false;
    }

    /**
     * @brief Distributed-pattern counterpart of reinitFromPattern().
     *
     * Backends that retain layout metadata in their vectors may implement
     * this only when the new owned/ghost layout remains compatible.  A false
     * result tells the caller that an in-place refresh is unavailable.
     */
    virtual bool reinitFromPattern(
        const sparsity::DistributedSparsityPattern& pattern)
    {
        (void)pattern;
        return false;
    }
};

} // namespace backends
} // namespace FE
} // namespace svmp

#endif // SVMP_FE_BACKENDS_GENERIC_MATRIX_H
