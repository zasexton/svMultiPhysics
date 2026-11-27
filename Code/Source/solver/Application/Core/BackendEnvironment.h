/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 * All Rights Reserved. See Copyright-SimVascular.txt for details.
 */
#pragma once

#include "Physics/Formulations/NavierStokes/FreeSurface/CertificateEnvironment.h"

namespace application::core {

inline constexpr char legacyPrimaryApproximationControl[] = "SVMP_FSILS_BLOCKSCHUR_MOMENTUM_HAT";

/// The complete catalogue accepted by the application, including its legacy solver controls.
inline const svmp::FE::environment::EnvironmentNameRegistry& applicationEnvironmentRegistry()
{
    namespace env = svmp::FE::environment;
    static constexpr env::NameDeclaration legacy[] = {
        {legacyPrimaryApproximationControl, env::NameKind::NumericalOverride, "Application/Core/SimulationBuilder",
         env::ValueType::Text, "ilu-k", env::RunStatus::NonReference,
         "replaces the BlockSchur momentum approximation (diag-k, blockdiag-k, ilu-k, asm-k)"},
    };
    static const env::EnvironmentNameRegistry registry(
        svmp::Physics::formulations::navier_stokes::free_surface::environmentRegistry(), legacy);
    return registry;
}

} // namespace application::core
