/* Copyright (c) Stanford University, The Regents of the
 * University of California, and others.
 *
 * All Rights Reserved.
 *
 * See License file.
 */

#include "Physics/Formulations/NavierStokes/FreeSurface/FittedALEOperatorStage.h"

#include "FE/Backends/Interfaces/GenericVector.h"

namespace svmp::Physics::formulations::navier_stokes::free_surface {

svmp::FE::timestepping::MeshNormalStageContext meshNormalStageContext(
    const FreeSurfaceStepContext& context)
{
  svmp::FE::timestepping::MeshNormalStageContext stage;
  stage.system = context.sim.fe_system.get();
  stage.scheme = context.opts.scheme;
  stage.scheme_name = context.transient_scheme.canonical_name;
  stage.comm =
      detail::feCollectiveCommunicator(context.velocity_extension_artifact_comm);
  stage.state_fingerprint =
      [fingerprint = context.collectiveMeshBoundaryStateFingerprint,
       comm = context.velocity_extension_artifact_comm](
          const svmp::FE::backends::GenericVector& state) {
        return fingerprint(state, comm);
      };
  return stage;
}

FittedALEStageParticipant::FittedALEStageParticipant(
    const FreeSurfaceStepContext& context)
    : MeshNormalStageParticipant(meshNormalStageContext(context))
{
}

} // namespace svmp::Physics::formulations::navier_stokes::free_surface
