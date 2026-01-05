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

} // namespace backends
} // namespace FE
} // namespace svmp
