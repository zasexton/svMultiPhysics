/* Copyright (c) Stanford University, The Regents of the
 * University of California, and others.
 *
 * All Rights Reserved.
 *
 * See License file.
 */

#ifndef SVMP_FE_SPACES_GLOBALSPACE_H
#define SVMP_FE_SPACES_GLOBALSPACE_H

/**
 * @file GlobalSpace.h
 * @brief The space of a global scalar (one value for the whole domain)
 *
 * A GlobalSpace carries one value that is the same everywhere, for example the
 * Lagrange multiplier of a mean-value constraint.  On every cell its basis is
 * the order-0 L2 function 1 with gradient 0, so `lam * v` and `mu * u` integrate
 * with the existing measures.  A field on this space is registered with
 * `systems::FieldScope::Global`: every cell maps to the same DOF, one rank owns
 * it, and the other ranks hold it as a ghost.
 *
 * The rows of a global test function are integrated on each cell with the
 * quadrature of that cell type's element of this space.  By default the order
 * is automatic: FESystem::setup binds it to the largest quadrature order of the
 * system's volume fields, so the global rows use the rule of the highest-order
 * field's rows.  An explicit order fixes it instead.
 */

#include "Spaces/FunctionSpace.h"

#include <array>
#include <atomic>
#include <cstddef>
#include <memory>
#include <mutex>
#include <span>
#include <vector>

namespace svmp {
namespace FE {
namespace spaces {

class GlobalSpace final : public FunctionSpace {
public:
    /**
     * @param n Number of global scalars of the field (1; several scalars per
     *          field are planned, declare one global field per scalar)
     * @param quadrature_order Quadrature order of the global rows on every cell
     *          type, or 0 for the automatic order bound at setup
     */
    explicit GlobalSpace(int n = 1, int quadrature_order = 0);

    SpaceType space_type() const noexcept override { return SpaceType::Global; }
    FieldType field_type() const noexcept override { return FieldType::Scalar; }
    Continuity continuity() const noexcept override { return Continuity::L2; }

    int value_dimension() const noexcept override { return n_; }
    int topological_dimension() const noexcept override { return 0; }
    int polynomial_order() const noexcept override { return 0; }
    ElementType element_type() const noexcept override { return ElementType::Point1; }

    /// The order-0 element on a point (the space's prototype)
    const elements::Element& element() const noexcept override { return *prototype_; }
    std::shared_ptr<const elements::Element> element_ptr() const noexcept override { return prototype_; }

    /**
     * @brief The order-0 element of a cell type, with the global rows' quadrature
     *
     * A cell type whose quadrature is not bound yet gets the prototype element;
     * FESystem::setup binds every cell type of its mesh before any assembly.
     */
    const elements::Element& getElement(ElementType cell_type,
                                        GlobalIndex cell_id) const noexcept override;

    std::size_t dofs_per_element() const noexcept override { return static_cast<std::size_t>(n_); }

    /// The explicit quadrature order, or 0 when the order is automatic
    [[nodiscard]] int explicitQuadratureOrder() const noexcept { return explicit_quadrature_order_; }

    /// The order bound to the cell types so far (the explicit one, or the automatic one once bound; 0 before)
    [[nodiscard]] int quadratureOrder() const noexcept;

    /**
     * @brief Bind the quadrature of the global rows on these cell types
     *
     * Builds the order-0 element of each cell type with a rule of
     * @p automatic_order (or of the explicit order, which then wins).  A space
     * shared by systems whose automatic orders differ is refused: give it an
     * explicit order.
     */
    void bindQuadrature(std::span<const ElementType> cell_types, int automatic_order) const;

    /// Whether the quadrature of this cell type is bound
    [[nodiscard]] bool hasBoundQuadrature(ElementType cell_type) const noexcept;

private:
    static constexpr std::size_t kCellTypeSlots = 32u;

    int n_{1};
    int explicit_quadrature_order_{0};
    std::shared_ptr<const elements::Element> prototype_;

    mutable std::mutex bind_mutex_;
    mutable std::atomic<int> bound_order_{0};
    mutable std::vector<std::shared_ptr<const elements::Element>> owned_elements_;
    mutable std::array<std::atomic<const elements::Element*>, kCellTypeSlots> cell_elements_{};
};

} // namespace spaces
} // namespace FE
} // namespace svmp

#endif // SVMP_FE_SPACES_GLOBALSPACE_H
