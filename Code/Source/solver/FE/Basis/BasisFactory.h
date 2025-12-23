/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See License file.
 */

#ifndef SVMP_FE_BASIS_BASISFACTORY_H
#define SVMP_FE_BASIS_BASISFACTORY_H

/**
 * @file BasisFactory.h
 * @brief Runtime creation of basis families
 */

#include "BasisFunction.h"
#include "LagrangeBasis.h"
#include "HierarchicalBasis.h"
#include "VectorBasis.h"
#include "TensorBasis.h"
#include "BernsteinBasis.h"
#include "SpectralBasis.h"
#include "SerendipityBasis.h"
#include "HermiteBasis.h"
#include <memory>

namespace svmp {
namespace FE {
namespace basis {

struct BasisRequest {
    ElementType element_type;
    BasisType basis_type;
    int order;
    Continuity continuity{Continuity::C0};
    FieldType field_type{FieldType::Scalar};
};

class BasisFactory {
public:
    static std::shared_ptr<BasisFunction> create(const BasisRequest& req);
};

} // namespace basis
} // namespace FE
} // namespace svmp

#endif // SVMP_FE_BASIS_BASISFACTORY_H
