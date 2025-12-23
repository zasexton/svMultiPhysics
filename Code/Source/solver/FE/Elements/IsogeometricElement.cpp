/* Copyright (c) Stanford University, The Regents of the University of
 * California, and others.
 *
 * All Rights Reserved.
 *
 * See License file.
 */

#include "Elements/IsogeometricElement.h"

#include "Core/FEException.h"

namespace svmp {
namespace FE {
namespace elements {

IsogeometricElement::IsogeometricElement(std::shared_ptr<basis::BasisFunction> basis,
                                         std::shared_ptr<const quadrature::QuadratureRule> quadrature,
                                         FieldType field_type,
                                         Continuity continuity)
    : basis_(std::move(basis)),
      quad_(std::move(quadrature)) {
    if (!basis_) {
        throw FEException("IsogeometricElement requires a valid basis pointer",
                          __FILE__, __LINE__, __func__, FEStatus::InvalidArgument);
    }
    if (!quad_) {
        throw FEException("IsogeometricElement requires a valid quadrature rule",
                          __FILE__, __LINE__, __func__, FEStatus::InvalidArgument);
    }

    // Validate basis/quadrature compatibility: dimensions must match
    if (basis_->dimension() != quad_->dimension()) {
        throw FEException(
            "IsogeometricElement: basis dimension does not match quadrature dimension",
            __FILE__, __LINE__, __func__, FEStatus::InvalidArgument);
    }

    // Validate cell family compatibility
    if (to_mesh_family(basis_->element_type()) != quad_->cell_family()) {
        throw FEException(
            "IsogeometricElement: basis element type does not match quadrature cell family",
            __FILE__, __LINE__, __func__, FEStatus::InvalidArgument);
    }

    info_.element_type = basis_->element_type();
    info_.field_type   = field_type;
    info_.continuity   = continuity;
    info_.order        = basis_->order();

    dimension_ = basis_->dimension();
    if (dimension_ <= 0) {
        throw FEException("IsogeometricElement: basis reports non-positive dimension",
                          __FILE__, __LINE__, __func__, FEStatus::InvalidElement);
    }

    // Validate that the provided field/continuity metadata is compatible with
    // the supplied basis. The library's Element/FunctionSpace semantics assume
    // one scalar coefficient per (possibly vector-valued) basis function.
    const bool vector_basis = basis_->is_vector_valued();

    if (vector_basis) {
        if (field_type != FieldType::Vector) {
            throw FEException(
                "IsogeometricElement: vector-valued basis requires FieldType::Vector",
                __FILE__, __LINE__, __func__, FEStatus::InvalidArgument);
        }
        if (continuity != Continuity::H_div && continuity != Continuity::H_curl) {
            throw FEException(
                "IsogeometricElement: vector-valued basis requires Continuity::H_div or Continuity::H_curl",
                __FILE__, __LINE__, __func__, FEStatus::InvalidArgument);
        }
    } else {
        if (field_type != FieldType::Scalar) {
            throw FEException(
                "IsogeometricElement: scalar basis currently supports FieldType::Scalar only (use Mixed/Product spaces for multi-component fields)",
                __FILE__, __LINE__, __func__, FEStatus::NotImplemented);
        }
        if (continuity == Continuity::H_div || continuity == Continuity::H_curl) {
            throw FEException(
                "IsogeometricElement: scalar basis cannot be tagged as H(div)/H(curl) continuity",
                __FILE__, __LINE__, __func__, FEStatus::InvalidArgument);
        }
    }

    // Each basis function carries one scalar DOF coefficient.
    num_dofs_ = basis_->size();
}

} // namespace elements
} // namespace FE
} // namespace svmp
