#ifndef SVMP_FE_FORMS_AUTHORING_H
#define SVMP_FE_FORMS_AUTHORING_H

/**
 * @file Authoring.h
 * @brief The one include of a formulation
 *
 * Everything a module's registerOn uses in the canonical skeleton
 * (Forms/SYSTEMS_INTEGRATION.md section 2): the form language, typed
 * integration domains, boundary conditions and boundary functionals, field
 * binding, the formulation builder, the constraint plan, level-set geometry
 * handles, measured functionals, derived results, global multipliers and 0D
 * auxiliary models.  Test FE/Tests/Unit/Forms/test_Authoring.cpp compiles the
 * documented skeleton with this include alone.
 *
 * Expert layers stay separate includes: Forms/FormCompiler.h,
 * Forms/FormKernels.h, Forms/SymbolicDifferentiation.h, Forms/BlockForm.h,
 * Forms/Complex.h, Forms/Index.h and Forms/Einsum.h.
 */

// The form language and its integration domains.
#include "Forms/Vocabulary.h"
#include "Forms/IntegrationDomain.h"
#include "Forms/CutCellForms.h"
#include "Forms/FiniteDeformationForms.h"

// Boundary conditions and boundary functionals.
#include "Forms/BoundaryConditions.h"
#include "Forms/StandardBCs.h"
#include "Forms/NitscheBC.h"
#include "Forms/LagrangeDirichletBC.h"
#include "Forms/BoundaryFunctional.h"

// The FE services a formulation calls.
#include "Systems/FESystem.h"
#include "Systems/FieldBinding.h"
#include "Systems/FormulationBuilder.h"
#include "Systems/BoundaryConditionManager.h"
#include "Systems/FunctionalForms.h"
#include "Constraints/ConstraintPlan.h"
#include "LevelSet/LevelSetGeometry.h"
#include "PostProcessing/DerivedResultBuilder.h"
#include "Spaces/GlobalSpace.h"
#include "Auxiliary/AuxiliaryBindings.h"
#include "Auxiliary/AuxiliaryModelDSL.h"

#endif // SVMP_FE_FORMS_AUTHORING_H
