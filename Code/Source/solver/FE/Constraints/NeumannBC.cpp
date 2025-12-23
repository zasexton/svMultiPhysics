/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#include "NeumannBC.h"

namespace svmp {
namespace FE {
namespace constraints {

// ============================================================================
// Construction - Scalar flux
// ============================================================================

NeumannBC::NeumannBC(int boundary_id, double flux) {
    spec_.boundary_id = boundary_id;
    spec_.constant_flux = flux;
    spec_.is_vector_valued = false;
    spec_.is_time_dependent = false;
}

NeumannBC::NeumannBC(int boundary_id, NeumannFluxFunction flux_func) {
    spec_.boundary_id = boundary_id;
    spec_.flux_function = std::move(flux_func);
    spec_.is_vector_valued = false;
    spec_.is_time_dependent = false;
}

NeumannBC::NeumannBC(int boundary_id, TimeDependentNeumannFunction flux_func) {
    spec_.boundary_id = boundary_id;
    spec_.time_flux_function = std::move(flux_func);
    spec_.is_vector_valued = false;
    spec_.is_time_dependent = true;
}

// ============================================================================
// Construction - Vector traction
// ============================================================================

NeumannBC::NeumannBC(int boundary_id, std::array<double, 3> traction) {
    spec_.boundary_id = boundary_id;
    spec_.constant_traction = traction;
    spec_.is_vector_valued = true;
    spec_.is_time_dependent = false;
}

NeumannBC::NeumannBC(int boundary_id, TractionFunction traction_func) {
    spec_.boundary_id = boundary_id;
    spec_.traction_function = std::move(traction_func);
    spec_.is_vector_valued = true;
    spec_.is_time_dependent = false;
}

NeumannBC::NeumannBC(int boundary_id, TimeDependentTractionFunction traction_func) {
    spec_.boundary_id = boundary_id;
    spec_.time_traction_function = std::move(traction_func);
    spec_.is_vector_valued = true;
    spec_.is_time_dependent = true;
}

NeumannBC::NeumannBC(NeumannSpec spec)
    : spec_(std::move(spec)) {}

// ============================================================================
// Copy/Move
// ============================================================================

NeumannBC::NeumannBC(const NeumannBC& other) = default;
NeumannBC::NeumannBC(NeumannBC&& other) noexcept = default;
NeumannBC& NeumannBC::operator=(const NeumannBC& other) = default;
NeumannBC& NeumannBC::operator=(NeumannBC&& other) noexcept = default;

} // namespace constraints
} // namespace FE
} // namespace svmp
