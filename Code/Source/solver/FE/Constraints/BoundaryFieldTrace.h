/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#ifndef SVMP_FE_CONSTRAINTS_BOUNDARY_FIELD_TRACE_H
#define SVMP_FE_CONSTRAINTS_BOUNDARY_FIELD_TRACE_H

#include "Spaces/FaceRestriction.h"
#include "Spaces/ProductSpace.h"

#include <map>

namespace svmp::FE::constraints::detail {

inline const spaces::FunctionSpace& scalarBoundarySpace(const spaces::FunctionSpace& space)
{
    const auto* product = dynamic_cast<const spaces::ProductSpace*>(&space);
    return product ? *product->base_space() : space;
}

/// The actual scalar field's trace layout on each visited cell. The per-call
/// cache also distinguishes hp levels whose geometry uses the same cell type.
class BoundaryFieldTraces {
public:
    const spaces::FaceRestriction& get(const spaces::FunctionSpace& scalar_space, ElementType cell_type)
    {
        const auto key = std::pair{cell_type, &scalar_space};
        const auto found = restrictions_.find(key);
        if (found != restrictions_.end()) return found->second;
        if (to_mesh_family(cell_type) == to_mesh_family(scalar_space.element_type()))
            return restrictions_.try_emplace(key, scalar_space).first->second;
        return restrictions_.try_emplace(key, cell_type, scalar_space.polynomial_order(), scalar_space.continuity(),
                                         scalar_space.element().basis().basis_type()).first->second;
    }

private:
    std::map<std::pair<ElementType, const spaces::FunctionSpace*>, spaces::FaceRestriction> restrictions_;
};

} // namespace svmp::FE::constraints::detail

#endif
