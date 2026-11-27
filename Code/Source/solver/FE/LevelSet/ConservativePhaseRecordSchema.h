/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 * All Rights Reserved. See Copyright-SimVascular.txt for details.
 */

#pragma once

namespace svmp::FE::level_set {

/// Field names of the conservative-phase record. Version 4 names the selected
/// phase in geometric terms. A formulation can provide a previously published
/// schema. Its labels must live as long as the reporting host.
struct ConservativePhaseRecordSchema {
    int version{4};
    const char* active_side{"active_side"};
    const char* measure{"active_measure"};
    const char* indicator{"phase_indicator"};
    const char* nonprimary_component{"nonprimary_component"};
};

inline constexpr ConservativePhaseRecordSchema defaultConservativePhaseRecordSchema{};

} // namespace svmp::FE::level_set
