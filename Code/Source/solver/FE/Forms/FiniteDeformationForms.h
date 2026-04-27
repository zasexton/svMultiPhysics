/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#ifndef SVMP_FE_FORMS_FINITE_DEFORMATION_FORMS_H
#define SVMP_FE_FORMS_FINITE_DEFORMATION_FORMS_H

/**
 * @file FiniteDeformationForms.h
 * @brief Physics-neutral symbolic finite-deformation FE form helpers.
 */

#include "Forms/FormExpr.h"

#include <algorithm>

namespace svmp {
namespace FE {
namespace forms {
namespace finite_deformation {

struct FiniteDeformationExpressions {
    FormExpr F{};
    FormExpr J{};
    FormExpr Finv{};
    FormExpr FinvT{};
    FormExpr C{};
    FormExpr b{};
    FormExpr green_lagrange{};
    FormExpr almansi{};
};

struct FiniteDeformationLinearizationExpressions {
    FiniteDeformationExpressions primal{};
    FormExpr dF{};
    FormExpr dJ{};
    FormExpr dFinv{};
    FormExpr dFinvT{};
    FormExpr dC{};
    FormExpr db{};
    FormExpr dGreenLagrange{};
    FormExpr dAlmansi{};
};

[[nodiscard]] inline int finiteDeformationDimension(const FormExpr& displacement,
                                                    int requested_dim = 0)
{
    if (requested_dim > 0) {
        return std::clamp(requested_dim, 1, 3);
    }
    if (displacement.isValid() && displacement.node() != nullptr) {
        if (const auto* sig = displacement.node()->spaceSignature(); sig != nullptr) {
            return std::clamp(sig->topological_dimension, 1, 3);
        }
    }
    return 3;
}

[[nodiscard]] inline FormExpr deformationGradient(const FormExpr& displacement,
                                                  int dim = 0)
{
    const int d = finiteDeformationDimension(displacement, dim);
    return FormExpr::identity(d) + grad(displacement);
}

[[nodiscard]] inline FormExpr deformationGradientVariation(const FormExpr& variation)
{
    return grad(variation);
}

[[nodiscard]] inline FormExpr jacobian(const FormExpr& F)
{
    return det(F);
}

[[nodiscard]] inline FormExpr inverseDeformationGradient(const FormExpr& F)
{
    return inv(F);
}

[[nodiscard]] inline FormExpr inverseTransposeDeformationGradient(const FormExpr& F)
{
    return transpose(inv(F));
}

[[nodiscard]] inline FormExpr rightCauchyGreen(const FormExpr& F)
{
    return transpose(F) * F;
}

[[nodiscard]] inline FormExpr leftCauchyGreen(const FormExpr& F)
{
    return F * transpose(F);
}

[[nodiscard]] inline FormExpr greenLagrangeStrain(const FormExpr& F, int dim = 3)
{
    return (rightCauchyGreen(F) - FormExpr::identity(std::clamp(dim, 1, 3))) * Real(0.5);
}

[[nodiscard]] inline FormExpr almansiStrain(const FormExpr& F, int dim = 3)
{
    return (FormExpr::identity(std::clamp(dim, 1, 3)) - inv(leftCauchyGreen(F))) * Real(0.5);
}

[[nodiscard]] inline FiniteDeformationExpressions kinematics(const FormExpr& displacement,
                                                             int dim = 0)
{
    FiniteDeformationExpressions out;
    const int d = finiteDeformationDimension(displacement, dim);
    out.F = deformationGradient(displacement, d);
    out.J = jacobian(out.F);
    out.Finv = inverseDeformationGradient(out.F);
    out.FinvT = transpose(out.Finv);
    out.C = rightCauchyGreen(out.F);
    out.b = leftCauchyGreen(out.F);
    out.green_lagrange = (out.C - FormExpr::identity(d)) * Real(0.5);
    out.almansi = (FormExpr::identity(d) - inv(out.b)) * Real(0.5);
    return out;
}

[[nodiscard]] inline FormExpr jacobianVariation(const FormExpr& F, const FormExpr& dF)
{
    return cofactor(F).doubleContraction(dF);
}

[[nodiscard]] inline FormExpr inverseVariation(const FormExpr& F, const FormExpr& dF)
{
    const auto Finv = inv(F);
    return (Finv * dF * Finv) * Real(-1);
}

[[nodiscard]] inline FormExpr rightCauchyGreenVariation(const FormExpr& F,
                                                        const FormExpr& dF)
{
    return transpose(dF) * F + transpose(F) * dF;
}

[[nodiscard]] inline FormExpr leftCauchyGreenVariation(const FormExpr& F,
                                                       const FormExpr& dF)
{
    return dF * transpose(F) + F * transpose(dF);
}

[[nodiscard]] inline FormExpr greenLagrangeVariation(const FormExpr& F,
                                                     const FormExpr& dF)
{
    return rightCauchyGreenVariation(F, dF) * Real(0.5);
}

[[nodiscard]] inline FormExpr almansiVariation(const FormExpr& F,
                                               const FormExpr& dF)
{
    const auto b = leftCauchyGreen(F);
    const auto binv = inv(b);
    const auto db = leftCauchyGreenVariation(F, dF);
    return (binv * db * binv) * Real(0.5);
}

[[nodiscard]] inline FiniteDeformationLinearizationExpressions linearizeKinematics(
    const FormExpr& displacement,
    const FormExpr& displacement_variation,
    int dim = 0)
{
    FiniteDeformationLinearizationExpressions out;
    const int d = finiteDeformationDimension(displacement, dim);
    out.primal = kinematics(displacement, d);
    out.dF = deformationGradientVariation(displacement_variation);
    out.dJ = jacobianVariation(out.primal.F, out.dF);
    out.dFinv = inverseVariation(out.primal.F, out.dF);
    out.dFinvT = transpose(out.dFinv);
    out.dC = rightCauchyGreenVariation(out.primal.F, out.dF);
    out.db = leftCauchyGreenVariation(out.primal.F, out.dF);
    out.dGreenLagrange = out.dC * Real(0.5);
    out.dAlmansi = almansiVariation(out.primal.F, out.dF);
    return out;
}

[[nodiscard]] inline FormExpr scalarCurrentGradientFromReferenceGradient(
    const FormExpr& gradient_reference,
    const FormExpr& F)
{
    return transpose(inv(F)) * gradient_reference;
}

[[nodiscard]] inline FormExpr scalarReferenceGradientFromCurrentGradient(
    const FormExpr& gradient_current,
    const FormExpr& F)
{
    return transpose(F) * gradient_current;
}

[[nodiscard]] inline FormExpr vectorCurrentGradientFromReferenceGradient(
    const FormExpr& gradient_reference,
    const FormExpr& F)
{
    return gradient_reference * inv(F);
}

[[nodiscard]] inline FormExpr vectorReferenceGradientFromCurrentGradient(
    const FormExpr& gradient_current,
    const FormExpr& F)
{
    return gradient_current * F;
}

[[nodiscard]] inline FormExpr pushForwardVector(const FormExpr& vector_reference,
                                                const FormExpr& F)
{
    return F * vector_reference;
}

[[nodiscard]] inline FormExpr pullBackVector(const FormExpr& vector_current,
                                             const FormExpr& F)
{
    return inv(F) * vector_current;
}

[[nodiscard]] inline FormExpr contravariantPiolaPushForward(
    const FormExpr& vector_reference,
    const FormExpr& F)
{
    return (F * vector_reference) / det(F);
}

[[nodiscard]] inline FormExpr covariantPiolaPushForward(
    const FormExpr& vector_reference,
    const FormExpr& F)
{
    return transpose(inv(F)) * vector_reference;
}

[[nodiscard]] inline FormExpr nansonMeasureVector(const FormExpr& F,
                                                  const FormExpr& reference_normal)
{
    return cofactor(F) * reference_normal;
}

[[nodiscard]] inline FormExpr nansonMeasureVectorVariation(const FormExpr& F,
                                                           const FormExpr& dF,
                                                           const FormExpr& reference_normal)
{
    const auto Finv = inv(F);
    const auto dJ = jacobianVariation(F, dF);
    const auto dFinvT = transpose(inverseVariation(F, dF));
    return (transpose(Finv) * reference_normal) * dJ +
           (dFinvT * reference_normal) * det(F);
}

[[nodiscard]] inline FormExpr pk1InternalVirtualWorkDensity(const FormExpr& first_piola,
                                                            const FormExpr& test_displacement)
{
    return first_piola.doubleContraction(grad(test_displacement));
}

[[nodiscard]] inline FormExpr initialStressGeometricStiffnessDensity(
    const FormExpr& initial_stress,
    const FormExpr& trial_displacement,
    const FormExpr& test_displacement)
{
    return initial_stress.doubleContraction(transpose(grad(test_displacement)) *
                                            grad(trial_displacement));
}

} // namespace finite_deformation
} // namespace forms
} // namespace FE
} // namespace svmp

#endif // SVMP_FE_FORMS_FINITE_DEFORMATION_FORMS_H
