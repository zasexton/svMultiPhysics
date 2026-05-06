/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#ifndef SVMP_FE_ANALYSIS_CONSTITUTIVE_LAW_METADATA_H
#define SVMP_FE_ANALYSIS_CONSTITUTIVE_LAW_METADATA_H

/**
 * @file ConstitutiveLawMetadata.h
 * @brief Residual-level metadata for constitutive laws used by Forms physics.
 */

#include "Core/Types.h"

#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace svmp {
namespace FE {

namespace forms {
class ConstitutiveModel;
} // namespace forms

namespace analysis {

enum class ConstitutiveLawRole : std::uint8_t {
    Unknown,
    DynamicViscosity,
};

enum class ConstitutiveLawInputMeasure : std::uint8_t {
    Unspecified,
    SymmetricGradientSecondInvariant,
};

struct ConstitutiveLawMetadata {
    std::string name;
    ConstitutiveLawRole role{ConstitutiveLawRole::Unknown};
    ConstitutiveLawInputMeasure input_measure{
        ConstitutiveLawInputMeasure::Unspecified};
    FieldId primary_field{INVALID_FIELD_ID};
    bool constant_value_available{false};
    Real constant_value{0.0};
    bool state_dependent{false};
    bool time_dependent{false};
    std::string tensor_rank;
    std::string symmetry_class;
    std::string positivity_class;
    Real positivity_tolerance{0.0};
    bool positivity_tolerance_present{false};
    std::string robustness_theorem_id;
    std::string robustness_norm_id;
    std::string robustness_parameter_range_scope;
    std::string robustness_mesh_family_scope;
    bool robustness_uniform_constant_present{false};
    Real robustness_uniform_constant{0.0};
    std::shared_ptr<const forms::ConstitutiveModel> model{};
    std::string source_operator_tag;
};

[[nodiscard]] inline bool sameConstitutiveLawIdentity(
    const ConstitutiveLawMetadata& lhs,
    const ConstitutiveLawMetadata& rhs) noexcept
{
    return lhs.name == rhs.name &&
           lhs.role == rhs.role &&
           lhs.input_measure == rhs.input_measure &&
           lhs.primary_field == rhs.primary_field &&
           lhs.model.get() == rhs.model.get();
}

inline void addConstitutiveLawIfAbsent(
    std::vector<ConstitutiveLawMetadata>& laws,
    ConstitutiveLawMetadata law)
{
    for (const auto& existing : laws) {
        if (sameConstitutiveLawIdentity(existing, law)) {
            return;
        }
    }
    laws.push_back(std::move(law));
}

[[nodiscard]] inline ConstitutiveLawMetadata dynamicViscosityMetadata(
    FieldId velocity_field,
    Real constant_viscosity,
    std::shared_ptr<const forms::ConstitutiveModel> model = {},
    std::string name = "dynamic_viscosity")
{
    return ConstitutiveLawMetadata{
        .name = std::move(name),
        .role = ConstitutiveLawRole::DynamicViscosity,
        .input_measure =
            ConstitutiveLawInputMeasure::SymmetricGradientSecondInvariant,
        .primary_field = velocity_field,
        .constant_value_available = model == nullptr,
        .constant_value = constant_viscosity,
        .model = std::move(model),
    };
}

} // namespace analysis
} // namespace FE
} // namespace svmp

#endif // SVMP_FE_ANALYSIS_CONSTITUTIVE_LAW_METADATA_H
