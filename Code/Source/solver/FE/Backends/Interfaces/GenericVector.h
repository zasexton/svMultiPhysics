/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#ifndef SVMP_FE_BACKENDS_GENERIC_VECTOR_H
#define SVMP_FE_BACKENDS_GENERIC_VECTOR_H

#include "Backends/Interfaces/BackendKind.h"
#include "Core/Types.h"

#include "Assembly/GlobalSystemView.h"

#include <cstdint>
#include <memory>
#include <span>
#include <vector>

namespace svmp {
namespace FE {
namespace backends {

class GenericVector {
public:
    virtual ~GenericVector() = default;

    [[nodiscard]] virtual BackendKind backendKind() const noexcept = 0;
    [[nodiscard]] virtual GlobalIndex size() const noexcept = 0;
    [[nodiscard]] virtual std::uint64_t valueRevision() const noexcept = 0;
    virtual void markModified() noexcept = 0;

    virtual void zero() = 0;
    virtual void set(Real value) = 0;
    virtual void add(Real value) = 0;
    virtual void scale(Real alpha) = 0;

    virtual void copyFrom(const GenericVector& other) = 0;

    [[nodiscard]] virtual Real dot(const GenericVector& other) const = 0;
    [[nodiscard]] virtual Real norm() const = 0;

    /** Refresh owner-to-ghost values used by global-indexed read views. */
    virtual void updateGhosts() = 0;

    /**
     * @brief Whether updateGhosts() requires communicator-wide participation.
     *
     * Collective callers can include this property in their request preflight
     * so a rank-local layout mismatch is rejected before some ranks enter a
     * backend communication operation while others return locally.
     */
    [[nodiscard]] virtual bool ghostUpdateRequiresCollectiveParticipation()
        const noexcept
    {
        return false;
    }

    /** Create the backend's legacy global-indexed assembly/read view. */
    [[nodiscard]] virtual std::unique_ptr<assembly::GlobalSystemView> createAssemblyView() = 0;

    /**
     * @brief Create a locally relevant, global-numbered coefficient read view.
     *
     * After updateGhosts(), getVectorEntry/getVectorEntries must return the
     * exact value for every owned or represented ghost global row. A valid
     * global row outside the represented overlap must be rejected rather than
     * silently returned as zero. The default is sufficient for replicated
     * vectors; distributed backends override it without changing the legacy
     * assembly-view read behavior used by unrelated callers.
     */
    [[nodiscard]] virtual std::unique_ptr<assembly::GlobalSystemView>
    createGhostedReadView()
    {
        return createAssemblyView();
    }

    [[nodiscard]] virtual std::span<Real> localSpan() = 0;
    [[nodiscard]] virtual std::span<const Real> localSpan() const = 0;

    /**
     * @brief Return backend-certified owned rows in public global numbering.
     *
     * Rows must be strictly increasing, lie in `[0, size())`, and exclude
     * ghosts. Across the vector layout's communicator, the returned sets must
     * form an exact disjoint cover of `[0, size())`. A backend that cannot
     * certify that contract must throw rather than return a guessed layout.
     * An empty result is valid on a rank that owns no rows.
     */
    [[nodiscard]] virtual std::vector<GlobalIndex> ownedGlobalRows() const = 0;
};

} // namespace backends
} // namespace FE
} // namespace svmp

#endif // SVMP_FE_BACKENDS_GENERIC_VECTOR_H
