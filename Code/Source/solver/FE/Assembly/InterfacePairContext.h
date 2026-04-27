/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#ifndef SVMP_FE_ASSEMBLY_INTERFACEPAIRCONTEXT_H
#define SVMP_FE_ASSEMBLY_INTERFACEPAIRCONTEXT_H

#include "Core/Types.h"

#if defined(SVMP_FE_WITH_MESH) && SVMP_FE_WITH_MESH

#include "Mesh/Search/MultiMeshInterface.h"

#include <array>
#include <vector>

namespace svmp {
namespace FE {
namespace assembly {

struct InterfaceQuadraturePair {
    MeshIndex source_face = static_cast<MeshIndex>(svmp::INVALID_INDEX);
    MeshIndex target_face = static_cast<MeshIndex>(svmp::INVALID_INDEX);
    MeshIndex source_cell = static_cast<MeshIndex>(svmp::INVALID_INDEX);
    MeshIndex target_cell = static_cast<MeshIndex>(svmp::INVALID_INDEX);
    std::array<Real, 3> source_point{{0.0, 0.0, 0.0}};
    std::array<Real, 3> target_point{{0.0, 0.0, 0.0}};
    std::array<Real, 3> source_face_xi{{0.0, 0.0, 0.0}};
    std::array<Real, 3> target_face_xi{{0.0, 0.0, 0.0}};
    std::array<Real, 3> source_cell_xi{{0.0, 0.0, 0.0}};
    std::array<Real, 3> target_cell_xi{{0.0, 0.0, 0.0}};
    std::array<Real, 3> source_normal{{0.0, 0.0, 0.0}};
    std::array<Real, 3> target_normal{{0.0, 0.0, 0.0}};
    Real weight = 0.0;
};

class InterfacePairContext {
public:
    InterfacePairContext() = default;
    explicit InterfacePairContext(const svmp::search::InterfaceMap& interface_map) {
        reset(interface_map);
    }

    void reset(const svmp::search::InterfaceMap& interface_map) {
        source_configuration_ = interface_map.source.configuration;
        target_configuration_ = interface_map.target.configuration;
        quadrature_pairs_.clear();
        quadrature_pairs_.reserve(interface_map.pairs.size());
        for (const auto& pair : interface_map.pairs) {
            InterfaceQuadraturePair qp;
            qp.source_face = static_cast<MeshIndex>(pair.source_face);
            qp.target_face = static_cast<MeshIndex>(pair.target_face);
            qp.source_cell = static_cast<MeshIndex>(pair.source_cell);
            qp.target_cell = static_cast<MeshIndex>(pair.target_cell);
            qp.source_point = pair.source_point;
            qp.target_point = pair.target_point;
            qp.source_face_xi = pair.source_face_xi;
            qp.target_face_xi = pair.target_face_xi;
            qp.source_cell_xi = pair.source_cell_xi;
            qp.target_cell_xi = pair.target_cell_xi;
            qp.source_normal = pair.source_normal;
            qp.target_normal = pair.target_normal;
            qp.weight = pair.source_measure > 0.0 ? pair.source_measure : pair.target_measure;
            quadrature_pairs_.push_back(qp);
        }
    }

    [[nodiscard]] const std::vector<InterfaceQuadraturePair>& quadraturePairs() const noexcept {
        return quadrature_pairs_;
    }

    [[nodiscard]] svmp::Configuration sourceConfiguration() const noexcept {
        return source_configuration_;
    }

    [[nodiscard]] svmp::Configuration targetConfiguration() const noexcept {
        return target_configuration_;
    }

private:
    svmp::Configuration source_configuration_ = svmp::Configuration::Reference;
    svmp::Configuration target_configuration_ = svmp::Configuration::Reference;
    std::vector<InterfaceQuadraturePair> quadrature_pairs_;
};

} // namespace assembly
} // namespace FE
} // namespace svmp

#endif // defined(SVMP_FE_WITH_MESH) && SVMP_FE_WITH_MESH

#endif // SVMP_FE_ASSEMBLY_INTERFACEPAIRCONTEXT_H
