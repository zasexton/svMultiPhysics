/* Copyright (c) Stanford University, The Regents of the
 * University of California, and others.
 *
 * All Rights Reserved.
 *
 * See License file.
 */

#include "Spaces/GlobalSpace.h"

#include "Basis/BasisFactory.h"
#include "Elements/DiscontinuousElement.h"
#include "Elements/GeneralBasisElement.h"
#include "Quadrature/QuadratureFactory.h"

#include <string>

namespace svmp {
namespace FE {
namespace spaces {

namespace {

[[nodiscard]] std::size_t cellTypeSlot(ElementType cell_type) noexcept
{
    return static_cast<std::size_t>(static_cast<std::uint8_t>(cell_type));
}

/// The linear cell of the same shape (the constant basis needs no higher-order nodes).
[[nodiscard]] ElementType linearCellType(ElementType cell_type) noexcept
{
    switch (cell_type) {
    case ElementType::Line3: return ElementType::Line2;
    case ElementType::Triangle6: return ElementType::Triangle3;
    case ElementType::Quad8:
    case ElementType::Quad9: return ElementType::Quad4;
    case ElementType::Tetra10: return ElementType::Tetra4;
    case ElementType::Hex20:
    case ElementType::Hex27: return ElementType::Hex8;
    case ElementType::Wedge15:
    case ElementType::Wedge18: return ElementType::Wedge6;
    case ElementType::Pyramid13:
    case ElementType::Pyramid14: return ElementType::Pyramid5;
    default: return cell_type;
    }
}

[[nodiscard]] std::shared_ptr<const elements::Element> makeCellElement(ElementType cell_type,
                                                                       int quadrature_order)
{
    const auto shape = linearCellType(cell_type);
    basis::BasisRequest request;
    request.element_type = shape;
    request.basis_type = BasisType::Lagrange;
    request.order = 0;
    request.continuity = Continuity::L2;
    request.field_type = FieldType::Scalar;
    auto basis = basis::basis_factory::create(request);
    FE_CHECK_NOT_NULL(basis.get(), "GlobalSpace: order-0 basis");
    auto rule = quadrature::QuadratureFactory::create(shape, quadrature_order);
    FE_CHECK_NOT_NULL(rule.get(), "GlobalSpace: quadrature rule");
    return std::make_shared<elements::GeneralBasisElement>(
        std::move(basis), std::move(rule), FieldType::Scalar, Continuity::L2);
}

} // namespace

GlobalSpace::GlobalSpace(int n, int quadrature_order)
    : n_(n)
    , explicit_quadrature_order_(quadrature_order)
{
    FE_THROW_IF(n != 1, NotImplementedException,
                "GlobalSpace: a global field carries one scalar (n = 1); several scalars per field are "
                "planned, so declare one global field per scalar");
    FE_THROW_IF(quadrature_order < 0, InvalidArgumentException,
                "GlobalSpace: the quadrature order must be >= 0 (0 is the automatic order)");
    prototype_ = std::make_shared<elements::DiscontinuousElement>(ElementType::Point1, 0);
    for (auto& slot : cell_elements_) {
        slot.store(nullptr, std::memory_order_relaxed);
    }
}

const elements::Element& GlobalSpace::getElement(ElementType cell_type,
                                                 GlobalIndex /*cell_id*/) const noexcept
{
    const auto slot = cellTypeSlot(cell_type);
    if (slot < kCellTypeSlots) {
        if (const auto* element = cell_elements_[slot].load(std::memory_order_acquire)) {
            return *element;
        }
    }
    return *prototype_;
}

int GlobalSpace::quadratureOrder() const noexcept
{
    return bound_order_.load(std::memory_order_acquire);
}

bool GlobalSpace::hasBoundQuadrature(ElementType cell_type) const noexcept
{
    const auto slot = cellTypeSlot(cell_type);
    return slot < kCellTypeSlots &&
           cell_elements_[slot].load(std::memory_order_acquire) != nullptr;
}

void GlobalSpace::bindQuadrature(std::span<const ElementType> cell_types, int automatic_order) const
{
    const int order = explicit_quadrature_order_ > 0 ? explicit_quadrature_order_ : automatic_order;
    FE_THROW_IF(order <= 0, InvalidArgumentException,
                "GlobalSpace::bindQuadrature: the quadrature order must be positive");

    std::lock_guard<std::mutex> lock(bind_mutex_);
    const int bound = bound_order_.load(std::memory_order_relaxed);
    FE_THROW_IF(bound > 0 && bound != order, InvalidArgumentException,
                "GlobalSpace::bindQuadrature: the space's global rows are integrated with order " +
                    std::to_string(bound) + " for another system, and this system asks for order " +
                    std::to_string(order) + "; give the GlobalSpace an explicit quadrature order");
    for (const auto cell_type : cell_types) {
        const auto slot = cellTypeSlot(cell_type);
        FE_THROW_IF(slot >= kCellTypeSlots, InvalidArgumentException,
                    "GlobalSpace::bindQuadrature: unsupported cell type");
        if (cell_elements_[slot].load(std::memory_order_relaxed) != nullptr) {
            continue;
        }
        auto element = makeCellElement(cell_type, order);
        owned_elements_.push_back(element);
        cell_elements_[slot].store(element.get(), std::memory_order_release);
    }
    bound_order_.store(order, std::memory_order_release);
}

} // namespace spaces
} // namespace FE
} // namespace svmp
