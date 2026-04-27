/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#ifndef SVMP_FE_CONSTITUTIVE_STRESS_TANGENT_CONTRACT_H
#define SVMP_FE_CONSTITUTIVE_STRESS_TANGENT_CONTRACT_H

/**
 * @file StressTangentContract.h
 * @brief Physics-neutral stress/tangent data contracts for finite-deformation forms.
 */

#include "Core/Types.h"

#include <cstdint>

namespace svmp {
namespace FE {
namespace constitutive {

enum class StressMeasure : std::uint8_t {
    FirstPiolaKirchhoff,
    SecondPiolaKirchhoff,
    Cauchy,
    Kirchhoff
};

enum class TangentMeasure : std::uint8_t {
    Material,
    Spatial,
    Mixed
};

enum class KinematicInputMeasure : std::uint8_t {
    DeformationGradient,
    GreenLagrangeStrain,
    AlmansiStrain,
    SmallStrain
};

enum class ConstitutiveUpdateFrame : std::uint8_t {
    Reference,
    Current
};

struct StressTangentContract {
    StressMeasure stress_measure{StressMeasure::FirstPiolaKirchhoff};
    TangentMeasure tangent_measure{TangentMeasure::Material};
    KinematicInputMeasure input_measure{KinematicInputMeasure::DeformationGradient};
    ConstitutiveUpdateFrame update_frame{ConstitutiveUpdateFrame::Reference};
    int dim{3};
    bool provides_stress{true};
    bool provides_consistent_tangent{false};
    bool history_update_is_trial{true};
};

[[nodiscard]] constexpr bool validFiniteDeformationDimension(int dim) noexcept
{
    return dim >= 1 && dim <= 3;
}

[[nodiscard]] constexpr bool isReferenceStress(StressMeasure measure) noexcept
{
    return measure == StressMeasure::FirstPiolaKirchhoff ||
           measure == StressMeasure::SecondPiolaKirchhoff;
}

[[nodiscard]] constexpr bool isSpatialStress(StressMeasure measure) noexcept
{
    return measure == StressMeasure::Cauchy ||
           measure == StressMeasure::Kirchhoff;
}

[[nodiscard]] constexpr ConstitutiveUpdateFrame naturalUpdateFrame(StressMeasure measure) noexcept
{
    return isReferenceStress(measure) ? ConstitutiveUpdateFrame::Reference
                                      : ConstitutiveUpdateFrame::Current;
}

[[nodiscard]] constexpr bool isMaterialTangent(TangentMeasure tangent) noexcept
{
    return tangent == TangentMeasure::Material;
}

[[nodiscard]] constexpr bool isSpatialTangent(TangentMeasure tangent) noexcept
{
    return tangent == TangentMeasure::Spatial;
}

[[nodiscard]] constexpr bool isFiniteDeformationInput(KinematicInputMeasure input) noexcept
{
    return input == KinematicInputMeasure::DeformationGradient ||
           input == KinematicInputMeasure::GreenLagrangeStrain ||
           input == KinematicInputMeasure::AlmansiStrain;
}

[[nodiscard]] constexpr bool isStressTangentContractValid(
    const StressTangentContract& contract) noexcept
{
    if (!validFiniteDeformationDimension(contract.dim)) {
        return false;
    }
    if (!contract.provides_stress && contract.provides_consistent_tangent) {
        return false;
    }
    if (isReferenceStress(contract.stress_measure) &&
        contract.tangent_measure == TangentMeasure::Spatial) {
        return false;
    }
    if (isSpatialStress(contract.stress_measure) &&
        contract.tangent_measure == TangentMeasure::Material) {
        return false;
    }
    return true;
}

[[nodiscard]] constexpr const char* stressMeasureName(StressMeasure measure) noexcept
{
    switch (measure) {
        case StressMeasure::FirstPiolaKirchhoff: return "FirstPiolaKirchhoff";
        case StressMeasure::SecondPiolaKirchhoff: return "SecondPiolaKirchhoff";
        case StressMeasure::Cauchy: return "Cauchy";
        case StressMeasure::Kirchhoff: return "Kirchhoff";
    }
    return "Unknown";
}

[[nodiscard]] constexpr const char* tangentMeasureName(TangentMeasure measure) noexcept
{
    switch (measure) {
        case TangentMeasure::Material: return "Material";
        case TangentMeasure::Spatial: return "Spatial";
        case TangentMeasure::Mixed: return "Mixed";
    }
    return "Unknown";
}

[[nodiscard]] constexpr const char* kinematicInputMeasureName(
    KinematicInputMeasure measure) noexcept
{
    switch (measure) {
        case KinematicInputMeasure::DeformationGradient: return "DeformationGradient";
        case KinematicInputMeasure::GreenLagrangeStrain: return "GreenLagrangeStrain";
        case KinematicInputMeasure::AlmansiStrain: return "AlmansiStrain";
        case KinematicInputMeasure::SmallStrain: return "SmallStrain";
    }
    return "Unknown";
}

[[nodiscard]] constexpr const char* constitutiveUpdateFrameName(
    ConstitutiveUpdateFrame frame) noexcept
{
    switch (frame) {
        case ConstitutiveUpdateFrame::Reference: return "Reference";
        case ConstitutiveUpdateFrame::Current: return "Current";
    }
    return "Unknown";
}

} // namespace constitutive
} // namespace FE
} // namespace svmp

#endif // SVMP_FE_CONSTITUTIVE_STRESS_TANGENT_CONTRACT_H
