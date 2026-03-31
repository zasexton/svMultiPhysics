/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#include "Backends/Utils/BackendOptions.h"

namespace svmp {
namespace FE {
namespace backends {

std::string_view solverMethodToString(SolverMethod m) noexcept
{
    switch (m) {
        case SolverMethod::Direct: return "direct";
        case SolverMethod::CG: return "cg";
        case SolverMethod::BiCGSTAB: return "bicgstab";
        case SolverMethod::GMRES: return "gmres";
        case SolverMethod::PGMRES: return "pgmres";
        case SolverMethod::FGMRES: return "fgmres";
        case SolverMethod::BlockSchur: return "block-schur";
        default: return "unknown";
    }
}

std::string_view preconditionerToString(PreconditionerType pc) noexcept
{
    switch (pc) {
        case PreconditionerType::None: return "none";
        case PreconditionerType::Diagonal: return "diagonal";
        case PreconditionerType::ILU: return "ilu";
        case PreconditionerType::AMG: return "amg";
        case PreconditionerType::RowColumnScaling: return "row-column-scaling";
        case PreconditionerType::FieldSplit: return "field-split";
        default: return "unknown";
    }
}

std::string_view fieldSplitKindToString(FieldSplitKind kind) noexcept
{
    switch (kind) {
        case FieldSplitKind::Additive: return "additive";
        case FieldSplitKind::Multiplicative: return "multiplicative";
        case FieldSplitKind::Schur: return "schur";
        default: return "unknown";
    }
}

std::string_view fsilsBlockSchurPreconditionerToString(FsilsBlockSchurSchurPreconditioner pc) noexcept
{
    switch (pc) {
        case FsilsBlockSchurSchurPreconditioner::DiagL: return "diag-l";
        case FsilsBlockSchurSchurPreconditioner::BlockDiagL: return "blockdiag-l";
        case FsilsBlockSchurSchurPreconditioner::ILUL: return "ilu-l";
        case FsilsBlockSchurSchurPreconditioner::AlgebraicSchur: return "algebraic-shat";
        default: return "unknown";
    }
}

std::string_view
fsilsBlockSchurMomentumApproximationToString(FsilsBlockSchurMomentumApproximation approx) noexcept
{
    switch (approx) {
        case FsilsBlockSchurMomentumApproximation::DiagK: return "diag-k";
        case FsilsBlockSchurMomentumApproximation::BlockDiagK: return "blockdiag-k";
        case FsilsBlockSchurMomentumApproximation::ILUK: return "ilu-k";
        case FsilsBlockSchurMomentumApproximation::ASM: return "asm-k";
        default: return "unknown";
    }
}

} // namespace backends
} // namespace FE
} // namespace svmp
