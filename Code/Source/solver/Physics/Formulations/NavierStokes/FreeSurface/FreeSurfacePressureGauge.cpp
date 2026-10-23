/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#include "Physics/Formulations/NavierStokes/FreeSurface/FreeSurfacePressureGauge.h"

namespace svmp {
namespace Physics {
namespace formulations {
namespace navier_stokes {

FE::constraints::InterfaceMeasureGaugeAnchor freeSurfacePressureAnchor(
    FE::FieldId p_id,
    const FreeSurfaceBoundary& boundary)
{
    FE::constraints::InterfaceMeasureGaugeAnchor anchor;
    anchor.field = p_id;
    anchor.interface_marker = boundary.interface_marker;
    anchor.level_set_field_name = boundary.level_set_field_name;
    anchor.domain_id = boundary.generated_interface_domain_id;
    auto& texts = anchor.texts;
    texts.owner = "IncompressibleNavierStokesVMSModule";
    texts.anchor_source =
        "Unfitted CutVolume embedded free-surface natural traction anchors absolute pressure";
    texts.kernel_name = "CutVolumePressureAnchorMeasureGuard";
    texts.callback_name = "navier_stokes_cut_volume_pressure_anchor_measure";
    texts.requirement =
        "CutVolume embedded free-surface pressure anchor requires a positive generated "
        "interface measure before a pressure solve";
    texts.preflight_phase = "embedded_free_surface_measure_preflight";
    texts.preflight_rejection =
        "another communicator rank rejected embedded free-surface measure preflight";
    texts.context_phase = "cut_volume_pressure_anchor_context";
    texts.null_context_accepted = "CutVolume pressure-anchor null context was unexpectedly accepted";
    texts.gauge_policy = "reject_zero_interface_no_dynamic_gauge_insertion";
    // The type label the kernel manifest has always shown for this anchor.
    texts.manifest_type_name =
        "svmp::Physics::formulations::navier_stokes::(anonymous namespace)::CutVolumePressureAnchorMeasureGuard";
    return anchor;
}

} // namespace navier_stokes
} // namespace formulations
} // namespace Physics
} // namespace svmp
