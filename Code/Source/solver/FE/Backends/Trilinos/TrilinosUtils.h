/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#ifndef SVMP_FE_BACKENDS_TRILINOS_UTILS_H
#define SVMP_FE_BACKENDS_TRILINOS_UTILS_H

#include "Core/Types.h"

#if defined(FE_HAS_TRILINOS)

#include <Teuchos_RCP.hpp>
#include <Tpetra_Core.hpp>
#include <Tpetra_CrsMatrix.hpp>
#include <Tpetra_Import.hpp>
#include <Tpetra_Map.hpp>
#include <Tpetra_Vector.hpp>

namespace svmp {
namespace FE {
namespace backends {
namespace trilinos {

using Scalar = Real;
using LO = int;
using GO = GlobalIndex;
using Node = Tpetra::Map<LO, GO>::node_type;

using Map = Tpetra::Map<LO, GO, Node>;
using Vector = Tpetra::Vector<Scalar, LO, GO, Node>;
using CrsMatrix = Tpetra::CrsMatrix<Scalar, LO, GO, Node>;

using Comm = Teuchos::Comm<int>;

} // namespace trilinos
} // namespace backends
} // namespace FE
} // namespace svmp

#endif // FE_HAS_TRILINOS

#endif // SVMP_FE_BACKENDS_TRILINOS_UTILS_H
