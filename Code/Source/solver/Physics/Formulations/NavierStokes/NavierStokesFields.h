/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#ifndef SVMP_PHYSICS_FORMULATIONS_NAVIERSTOKES_NAVIER_STOKES_FIELDS_H
#define SVMP_PHYSICS_FORMULATIONS_NAVIERSTOKES_NAVIER_STOKES_FIELDS_H

/**
 * @file NavierStokesFields.h
 * @brief Fields of a Navier-Stokes registration, and the node pressure constraints
 *
 * Moved from IncompressibleNavierStokesVMSModule.cpp: compatibility of a
 * requested field with an existing one, registration (or reuse) of unknowns
 * and prescribed data, distinct field names, the affine P1 trace-space
 * envelope of the generated-boundary certificates, and the node pressure
 * constraints of the options as vertex values.
 */

#include "Physics/Formulations/NavierStokes/IncompressibleNavierStokesVMSModule.h"
#include "Physics/Formulations/NavierStokes/NavierStokesPreflight.h"

#include "FE/Assembly/CutIntegrationContext.h"
#include "FE/Constraints/ConstraintPlan.h"
#include "FE/Core/Types.h"
#include "FE/Forms/BoundaryConditions.h"
#include "FE/Forms/FormExpr.h"
#include "FE/Forms/IntegrationDomain.h"
#include "FE/LevelSet/LevelSetGeometry.h"
#include "FE/Systems/ALEBinding.h"
#include "FE/Systems/FESystem.h"
#include "FE/Systems/FormsInstaller.h"
#include "Interfaces/GeneratedActiveBoundaryDomain.h"
#include "Interfaces/GeneratedInterfaceBoundaryIntersectionDomain.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "FE/Constraints/VertexDirichletConstraint.h"

namespace svmp {
namespace Physics {
namespace formulations {
namespace navier_stokes {

[[nodiscard]] bool spacesCompatible(const FE::spaces::FunctionSpace& lhs,
                                    const FE::spaces::FunctionSpace& rhs) noexcept;

[[nodiscard]] bool
isSupportedGeneratedBoundaryTraceSpace(
    const FE::spaces::FunctionSpace& space,
    int dimension) noexcept;

void validateCompatibleField(const FE::systems::FESystem& system,
                             const FE::systems::FieldSpec& spec,
                             FE::systems::FieldSourceKind expected_source,
                             bool allow_missing,
                             const char* context);

[[nodiscard]] FE::FieldId ensureCompatibleUnknownField(
    FE::systems::FESystem& system,
    FE::systems::FieldSpec spec,
    const char* context);

[[nodiscard]] FE::FieldId ensureCompatiblePrescribedField(
    FE::systems::FESystem& system,
    FE::systems::FieldSpec spec,
    bool auto_register,
    const char* context);

void requireDistinctFieldNames(
    std::initializer_list<std::pair<std::string_view, std::string_view>> fields);

/// The node pressure constraints of the options as vertex values, and their id mode.
[[nodiscard]] std::vector<FE::constraints::VertexDirichletValue> nodePressureValues(
    const IncompressibleNavierStokesVMSOptions& options);
[[nodiscard]] FE::constraints::VertexIdMode nodePressureIdMode(
    const IncompressibleNavierStokesVMSOptions& options);

struct NavierStokesFieldIds {
    FE::FieldId velocity{FE::INVALID_FIELD_ID};
    FE::FieldId pressure{FE::INVALID_FIELD_ID};
    FE::FieldId body_force{FE::INVALID_FIELD_ID};
};

/// Registers (or reuses) u and p, then the body-force data and the
/// semi-implicit reference velocity when requested: this order is part of
/// the DOF layout.
[[nodiscard]] NavierStokesFieldIds registerNavierStokesFields(
    FE::systems::FESystem& system,
    const IncompressibleNavierStokesVMSOptions& options,
    NavierStokesPreflight& preflight);

} // namespace navier_stokes
} // namespace formulations
} // namespace Physics
} // namespace svmp

#endif // SVMP_PHYSICS_FORMULATIONS_NAVIERSTOKES_NAVIER_STOKES_FIELDS_H
