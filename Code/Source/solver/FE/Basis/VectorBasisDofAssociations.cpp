/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See License file.
 */

#include "VectorBasis.h"
#include "Basis/BasisTraits.h"
#include "Elements/ReferenceElement.h"
#include "VectorBasisEvaluationHelpers.h"

namespace svmp {
namespace FE {
namespace basis {
std::vector<DofAssociation> BDMBasis::dof_associations() const {
    std::vector<DofAssociation> result(size_);
    std::size_t idx = 0;

    if (element_type_ == ElementType::Triangle3) {
        for (int e = 0; e < 3; ++e) {
            for (int m = 0; m <= order_; ++m) {
                result[idx].entity_type = DofEntity::Edge;
                result[idx].entity_id = e;
                result[idx].moment_index = m;
                ++idx;
            }
        }
        while (idx < size_) {
            result[idx].entity_type = DofEntity::Interior;
            result[idx].entity_id = 0;
            result[idx].moment_index = static_cast<int>(idx - std::size_t(3 * (order_ + 1)));
            ++idx;
        }
    } else if (element_type_ == ElementType::Tetra4) {
        const std::size_t dofs_per_face =
            detail::vector_common::triangle_poly_dim(static_cast<std::size_t>(order_));
        for (int f = 0; f < 4; ++f) {
            for (std::size_t m = 0; m < dofs_per_face; ++m) {
                result[idx].entity_type = DofEntity::Face;
                result[idx].entity_id = f;
                result[idx].moment_index = static_cast<int>(m);
                ++idx;
            }
        }
        while (idx < size_) {
            result[idx].entity_type = DofEntity::Interior;
            result[idx].entity_id = 0;
            result[idx].moment_index = static_cast<int>(idx - 4u * dofs_per_face);
            ++idx;
        }
    } else {
        for (int e = 0; e < 4; ++e) {
            for (int m = 0; m < 2; ++m) {
                result[idx].entity_type = DofEntity::Edge;
                result[idx].entity_id = e;
                result[idx].moment_index = m;
                ++idx;
            }
        }
    }

    return result;
}

std::vector<DofAssociation> RaviartThomasBasis::dof_associations() const {
    std::vector<DofAssociation> result(size_);
    std::size_t idx = 0;
    const int k = order_;

    if (dimension_ == 2) {
        const std::size_t dofs_per_edge = static_cast<std::size_t>(k + 1);
        const std::size_t num_edges = is_triangle(element_type_) ? 3u : 4u;

        for (std::size_t e = 0; e < num_edges; ++e) {
            for (std::size_t m = 0; m < dofs_per_edge; ++m) {
                result[idx].entity_type = DofEntity::Edge;
                result[idx].entity_id = static_cast<int>(e);
                result[idx].moment_index = static_cast<int>(m);
                ++idx;
            }
        }

        while (idx < size_) {
            result[idx].entity_type = DofEntity::Interior;
            result[idx].entity_id = 0;
            result[idx].moment_index = static_cast<int>(idx - num_edges * dofs_per_edge);
            ++idx;
        }
    } else {
        const elements::ReferenceElement ref = elements::ReferenceElement::create(element_type_);
        const std::size_t num_faces = ref.num_faces();

        for (std::size_t f = 0; f < num_faces; ++f) {
            const auto& face_nodes = ref.face_nodes(f);
            const std::size_t dofs_per_face =
                (face_nodes.size() == 3u)
                    ? detail::vector_common::triangle_poly_dim(static_cast<std::size_t>(k))
                    : static_cast<std::size_t>((k + 1) * (k + 1));

            for (std::size_t m = 0; m < dofs_per_face; ++m) {
                if (idx >= size_) break;
                result[idx].entity_type = DofEntity::Face;
                result[idx].entity_id = static_cast<int>(f);
                result[idx].moment_index = static_cast<int>(m);
                ++idx;
            }
        }

        const std::size_t interior_start = idx;
        while (idx < size_) {
            result[idx].entity_type = DofEntity::Interior;
            result[idx].entity_id = 0;
            result[idx].moment_index = static_cast<int>(idx - interior_start);
            ++idx;
        }
    }

    return result;
}

std::vector<DofAssociation> NedelecBasis::dof_associations() const {
    std::vector<DofAssociation> result(size_);
    std::size_t idx = 0;
    const int k = order_;

    const elements::ReferenceElement ref = elements::ReferenceElement::create(element_type_);
    const std::size_t num_edges = ref.num_edges();
    const std::size_t dofs_per_edge = static_cast<std::size_t>(k + 1);

    for (std::size_t e = 0; e < num_edges; ++e) {
        for (std::size_t m = 0; m < dofs_per_edge; ++m) {
            if (idx >= size_) break;
            result[idx].entity_type = DofEntity::Edge;
            result[idx].entity_id = static_cast<int>(e);
            result[idx].moment_index = static_cast<int>(m);
            ++idx;
        }
    }

    if (dimension_ == 3 && k >= 1) {
        const std::size_t num_faces = ref.num_faces();
        for (std::size_t f = 0; f < num_faces; ++f) {
            const auto& face_nodes = ref.face_nodes(f);
            const std::size_t dofs_per_face =
                (face_nodes.size() == 3u)
                    ? static_cast<std::size_t>(k * (k + 1))
                    : static_cast<std::size_t>(2 * k * (k + 1));

            for (std::size_t m = 0; m < dofs_per_face; ++m) {
                if (idx >= size_) break;
                result[idx].entity_type = DofEntity::Face;
                result[idx].entity_id = static_cast<int>(f);
                result[idx].moment_index = static_cast<int>(m);
                ++idx;
            }
        }
    }

    const std::size_t interior_start = idx;
    while (idx < size_) {
        result[idx].entity_type = DofEntity::Interior;
        result[idx].entity_id = 0;
        result[idx].moment_index = static_cast<int>(idx - interior_start);
        ++idx;
    }

    return result;
}

} // namespace basis
} // namespace FE
} // namespace svmp
