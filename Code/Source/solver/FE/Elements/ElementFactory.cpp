/* Copyright (c) Stanford University, The Regents of the University of
 * California, and others.
 *
 * All Rights Reserved.
 *
 * See License file.
 */

#include "Elements/ElementFactory.h"

#include "Core/FEException.h"
#include "Elements/LagrangeElement.h"
#include "Elements/DiscontinuousElement.h"
#include "Elements/VectorElement.h"
#include "Elements/SpectralElement.h"

namespace svmp {
namespace FE {
namespace elements {

std::shared_ptr<Element> ElementFactory::create(const ElementRequest& req) {
    if (req.element_type == ElementType::Unknown) {
        throw FEException("ElementFactory: ElementType::Unknown is not allowed",
                          __FILE__, __LINE__, __func__, FEStatus::InvalidArgument);
    }

    // Validate polynomial order early
    if (req.order < 0) {
        throw FEException("ElementFactory: negative polynomial order is not allowed",
                          __FILE__, __LINE__, __func__, FEStatus::InvalidArgument);
    }

    // Mixed, isogeometric, and composite elements are typically constructed
    // explicitly by higher-level code and are not created by this factory.

    // ---------------------------------------------------------------------
    // Vector-valued conforming elements (H(div) / H(curl))
    //
    // The VectorElement internally delegates basis selection to BasisFactory
    // using (continuity, order). For user convenience, allow callers to
    // specify either:
    //   - req.continuity = H_div / H_curl (req.basis_type may be Lagrange), or
    //   - req.basis_type = RaviartThomas / BDM / Nedelec (continuity inferred).
    //
    // For the vector-specific basis types, we validate that the requested
    // order matches the currently implemented families to avoid silently
    // returning a different basis than requested.
    // ---------------------------------------------------------------------

    Continuity effective_continuity = req.continuity;
    if (req.basis_type == BasisType::Nedelec) {
        if (req.continuity != Continuity::C0 && req.continuity != Continuity::H_curl) {
            throw FEException("ElementFactory: BasisType::Nedelec conflicts with requested continuity",
                              __FILE__, __LINE__, __func__, FEStatus::InvalidArgument);
        }
        effective_continuity = Continuity::H_curl;
    } else if (req.basis_type == BasisType::RaviartThomas) {
        if (req.continuity != Continuity::C0 && req.continuity != Continuity::H_div) {
            throw FEException("ElementFactory: BasisType::RaviartThomas conflicts with requested continuity",
                              __FILE__, __LINE__, __func__, FEStatus::InvalidArgument);
        }
        effective_continuity = Continuity::H_div;
    } else if (req.basis_type == BasisType::BDM) {
        if (req.continuity != Continuity::C0 && req.continuity != Continuity::H_div) {
            throw FEException("ElementFactory: BasisType::BDM conflicts with requested continuity",
                              __FILE__, __LINE__, __func__, FEStatus::InvalidArgument);
        }
        effective_continuity = Continuity::H_div;
    }

    if (effective_continuity == Continuity::H_div || effective_continuity == Continuity::H_curl) {
        if (req.field_type != FieldType::Vector) {
            throw FEException("ElementFactory: H(div)/H(curl) elements require FieldType::Vector",
                              __FILE__, __LINE__, __func__, FEStatus::InvalidArgument);
        }

        if (effective_continuity == Continuity::H_div) {
            const bool ok_basis =
                (req.basis_type == BasisType::Lagrange) ||
                (req.basis_type == BasisType::RaviartThomas) ||
                (req.basis_type == BasisType::BDM);
            if (!ok_basis) {
                throw FEException("ElementFactory: H(div) elements require BasisType::Lagrange/::RaviartThomas/::BDM",
                                  __FILE__, __LINE__, __func__, FEStatus::InvalidArgument);
            }
        } else {
            const bool ok_basis =
                (req.basis_type == BasisType::Lagrange) ||
                (req.basis_type == BasisType::Nedelec);
            if (!ok_basis) {
                throw FEException("ElementFactory: H(curl) elements require BasisType::Lagrange/::Nedelec",
                                  __FILE__, __LINE__, __func__, FEStatus::InvalidArgument);
            }
        }

        return std::make_shared<VectorElement>(req.element_type, req.order, effective_continuity, req.basis_type);
    }

    if (req.basis_type == BasisType::Lagrange) {
        if (req.continuity == Continuity::L2) {
            // DG scalar element
            if (req.field_type != FieldType::Scalar) {
                throw FEException("ElementFactory: DG elements currently support scalar fields only",
                                  __FILE__, __LINE__, __func__, FEStatus::InvalidArgument);
            }
            return std::make_shared<DiscontinuousElement>(req.element_type, req.order, req.field_type);
        }

        // Standard scalar H¹ Lagrange element
        if (req.field_type != FieldType::Scalar) {
            throw FEException("ElementFactory: scalar LagrangeElement requires FieldType::Scalar",
                              __FILE__, __LINE__, __func__, FEStatus::InvalidArgument);
        }
        return std::make_shared<LagrangeElement>(req.element_type, req.order, req.field_type, req.continuity);
    }

    if (req.basis_type == BasisType::Spectral) {
        if (req.field_type != FieldType::Scalar) {
            throw FEException("ElementFactory: SpectralElement currently supports scalar fields only",
                              __FILE__, __LINE__, __func__, FEStatus::InvalidArgument);
        }
        return std::make_shared<SpectralElement>(req.element_type, req.order, req.field_type);
    }

    throw FEException("ElementFactory: requested basis type not supported by factory",
                      __FILE__, __LINE__, __func__, FEStatus::NotImplemented);
}

} // namespace elements
} // namespace FE
} // namespace svmp
