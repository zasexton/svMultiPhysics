/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#ifndef SVMP_PHYSICS_FORMULATIONS_NAVIERSTOKES_NAVIER_STOKES_SUPPORT_H
#define SVMP_PHYSICS_FORMULATIONS_NAVIERSTOKES_NAVIER_STOKES_SUPPORT_H

/**
 * @file NavierStokesSupport.h
 * @brief Everything the Navier-Stokes formulation calls that is not mathematics
 *
 * Option queries and capability rules, the registration preflight, field
 * binding and initial state, the free-surface geometry adapters and
 * contact-line laws, certificates and gauges, logs, the residual-work ledger,
 * the diagnostic operators and the configuration artifact.
 * IncompressibleNavierStokesVMSModule.cpp includes this one header so that it
 * reads as the formulation.
 */

#include "Physics/Formulations/NavierStokes/FreeSurface/FreeSurfaceConfigurationArtifact.h"
#include "Physics/Formulations/NavierStokes/FreeSurface/FreeSurfaceContactCertificate.h"
#include "Physics/Formulations/NavierStokes/FreeSurface/FreeSurfaceDiagnosticOperators.h"
#include "Physics/Formulations/NavierStokes/FreeSurface/FreeSurfaceDiagnostics.h"
#include "Physics/Formulations/NavierStokes/FreeSurface/FreeSurfaceEnergy.h"
#include "Physics/Formulations/NavierStokes/FreeSurface/FreeSurfaceFittedALE.h"
#include "Physics/Formulations/NavierStokes/FreeSurface/FreeSurfaceGeometry.h"
#include "Physics/Formulations/NavierStokes/FreeSurface/FreeSurfaceQueries.h"
#include "Physics/Formulations/NavierStokes/FreeSurface/FreeSurfaceResidualWork.h"
#include "Physics/Formulations/NavierStokes/FreeSurface/FreeSurfaceValidation.h"
#include "Physics/Formulations/NavierStokes/NavierStokesFields.h"
#include "Physics/Formulations/NavierStokes/NavierStokesGeneratedNitsche.h"
#include "Physics/Formulations/NavierStokes/NavierStokesInitialConditions.h"
#include "Physics/Formulations/NavierStokes/NavierStokesPreflight.h"

#endif // SVMP_PHYSICS_FORMULATIONS_NAVIERSTOKES_NAVIER_STOKES_SUPPORT_H
