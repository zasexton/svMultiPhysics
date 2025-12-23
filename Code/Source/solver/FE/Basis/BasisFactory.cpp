/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See License file.
 */

#include "BasisFactory.h"

namespace svmp {
namespace FE {
namespace basis {

std::shared_ptr<BasisFunction> BasisFactory::create(const BasisRequest& req) {
    // Vector-valued conforming bases take precedence
    if (req.continuity == Continuity::H_div) {
        // Allow explicit family selection via BasisType.
        if (req.basis_type == BasisType::RaviartThomas) {
            return std::make_shared<RaviartThomasBasis>(req.element_type, req.order);
        }
        if (req.basis_type == BasisType::BDM) {
            return std::make_shared<BDMBasis>(req.element_type, req.order);
        }

        // Default selection (BasisType::Lagrange): keep the historical choice of BDM(1) on 2D elements,
        // but fall back to Raviart-Thomas when BDM is not applicable (e.g., 3D tensor-product RT(1)+).
        const int dim = element_dimension(req.element_type);
        if (dim == 2 && req.order == 1) {
            return std::make_shared<BDMBasis>(req.element_type, 1);
        }
        return std::make_shared<RaviartThomasBasis>(req.element_type, req.order);
    }

    if (req.continuity == Continuity::H_curl) {
        if (req.basis_type != BasisType::Lagrange && req.basis_type != BasisType::Nedelec) {
            throw FEException("BasisFactory: H(curl) bases require BasisType::Lagrange or BasisType::Nedelec",
                              __FILE__, __LINE__, __func__, FEStatus::InvalidArgument);
        }
        return std::make_shared<NedelecBasis>(req.element_type, req.order);
    }

    // C¹ scalar bases (currently 1D Hermite on Line2)
    if (req.continuity == Continuity::C1) {
        if (req.field_type == FieldType::Scalar) {
            return std::make_shared<HermiteBasis>(req.element_type, req.order);
        }
        throw FEException("BasisFactory: C1 continuity currently supports scalar fields only",
                          __FILE__, __LINE__, __func__, FEStatus::InvalidArgument);
    }

    switch (req.basis_type) {
        case BasisType::Lagrange:
            return std::make_shared<LagrangeBasis>(req.element_type, req.order);
        case BasisType::Hierarchical:
            return std::make_shared<HierarchicalBasis>(req.element_type, req.order);
        case BasisType::Bernstein:
            return std::make_shared<BernsteinBasis>(req.element_type, req.order);
        case BasisType::Spectral:
            return std::make_shared<SpectralBasis>(req.element_type, req.order);
        case BasisType::Serendipity:
            return std::make_shared<SerendipityBasis>(req.element_type, req.order);
        default:
            throw FEException("Unsupported basis type in BasisFactory",
                              __FILE__, __LINE__, __func__, FEStatus::InvalidArgument);
    }
}

} // namespace basis
} // namespace FE
} // namespace svmp
