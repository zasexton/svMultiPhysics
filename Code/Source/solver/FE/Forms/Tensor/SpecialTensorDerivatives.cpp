/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#include "Forms/Tensor/SpecialTensorDerivatives.h"

#include <algorithm>

namespace svmp {
namespace FE {
namespace forms {
namespace tensor {

FormExpr differentiateDelta(const TensorDiffContext& /*ctx*/)
{
    return FormExpr::constant(0.0);
}

FormExpr differentiateLeviCivita(const TensorDiffContext& /*ctx*/)
{
    return FormExpr::constant(0.0);
}

FormExpr differentiateMetricTensor(const TensorDiffContext& /*ctx*/)
{
    // Default metric is constant (identity) => derivative 0.
    return FormExpr::constant(0.0);
}

FormExpr differentiateInverseMetricTensor(const TensorDiffContext& /*ctx*/)
{
    // Default inverse metric is constant (identity) => derivative 0.
    return FormExpr::constant(0.0);
}

DeformationGradientDiffResult differentiateDeformationGradient(const FormExpr& displacement,
                                                              const TensorDiffContext& ctx)
{
    DeformationGradientDiffResult out;
    if (!displacement.isValid() || displacement.node() == nullptr) {
        return out;
    }

    int dim = 3;
    if (const auto* sig = displacement.node()->spaceSignature(); sig != nullptr) {
        dim = std::clamp(sig->topological_dimension, 1, 3);
    }

    out.F = FormExpr::identity(dim) + grad(displacement);
    out.dF = differentiateTensorResidual(out.F, ctx);
    return out;
}

} // namespace tensor
} // namespace forms
} // namespace FE
} // namespace svmp
