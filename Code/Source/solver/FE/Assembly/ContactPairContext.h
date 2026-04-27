/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#ifndef SVMP_FE_ASSEMBLY_CONTACTPAIRCONTEXT_H
#define SVMP_FE_ASSEMBLY_CONTACTPAIRCONTEXT_H

#include "Core/Types.h"

#if defined(SVMP_FE_WITH_MESH) && SVMP_FE_WITH_MESH

#include "Mesh/Search/ContactProximity.h"

#include <array>
#include <map>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

namespace svmp {
namespace FE {
namespace assembly {

enum class ContactPairSelection {
    AllPairs,
    ActivePairsOnly,
    ActiveAndProjectedPairs
};

struct ContactProjectionOperatorMetadata {
    bool available = false;
    std::uint64_t pair_id = 0;
    std::string operator_family;
    std::string generation_policy;
    bool conservative = false;
};

struct ContactQuadraturePair {
    MeshIndex source_entity = static_cast<MeshIndex>(svmp::INVALID_INDEX);
    MeshIndex target_entity = static_cast<MeshIndex>(svmp::INVALID_INDEX);
    svmp::gid_t source_gid = svmp::INVALID_GID;
    svmp::gid_t target_gid = svmp::INVALID_GID;
    std::string source_surface_name;
    std::string target_surface_name;
    svmp::rank_t canonical_owner_rank = 0;
    svmp::label_t source_label = svmp::INVALID_LABEL;
    svmp::label_t target_label = svmp::INVALID_LABEL;
    svmp::search::ContactEntityKind source_kind = svmp::search::ContactEntityKind::Face;
    svmp::search::ContactEntityKind target_kind = svmp::search::ContactEntityKind::Face;
    svmp::Configuration source_configuration = svmp::Configuration::Reference;
    svmp::Configuration target_configuration = svmp::Configuration::Reference;
    svmp::search::ContactPairState state = svmp::search::ContactPairState::Candidate;
    std::uint64_t pair_id = 0;
    Real time_level = 0.0;
    std::array<Real, 3> source_point{{0.0, 0.0, 0.0}};
    std::array<Real, 3> target_point{{0.0, 0.0, 0.0}};
    std::array<Real, 3> source_local_coordinates{{0.0, 0.0, 0.0}};
    std::array<Real, 3> target_local_coordinates{{0.0, 0.0, 0.0}};
    std::array<Real, 3> normal{{0.0, 0.0, 1.0}};
    std::array<Real, 3> target_normal{{0.0, 0.0, -1.0}};
    std::array<Real, 3> tangent0{{1.0, 0.0, 0.0}};
    std::array<Real, 3> tangent1{{0.0, 1.0, 0.0}};
    std::array<Real, 3> tangential_reference0{{1.0, 0.0, 0.0}};
    std::array<Real, 3> tangential_reference1{{0.0, 1.0, 0.0}};
    Real unsigned_gap = 0.0;
    Real signed_gap = 0.0;
    Real tangential_slip_magnitude = 0.0;
    Real shell_thickness_offset = 0.0;
    bool projection_valid = false;
    bool tangential_frame_valid = false;
    bool wrong_side_projection = false;
    std::string side;
    std::string generation_policy;
    ContactProjectionOperatorMetadata projection_operator;
};

struct ContactPatch {
    svmp::label_t source_label = svmp::INVALID_LABEL;
    svmp::label_t target_label = svmp::INVALID_LABEL;
    svmp::search::ContactEntityKind source_kind = svmp::search::ContactEntityKind::Face;
    svmp::search::ContactEntityKind target_kind = svmp::search::ContactEntityKind::Face;
    svmp::Configuration source_configuration = svmp::Configuration::Reference;
    svmp::Configuration target_configuration = svmp::Configuration::Reference;
    svmp::rank_t canonical_owner_rank = 0;
    std::vector<std::size_t> pair_indices;
};

class ContactPairContext {
public:
    using PairIterator = std::vector<ContactQuadraturePair>::const_iterator;
    using PatchIterator = std::vector<ContactPatch>::const_iterator;

    ContactPairContext() = default;

    explicit ContactPairContext(
        const svmp::search::ContactProximityMap& contact_map,
        ContactPairSelection selection = ContactPairSelection::AllPairs) {
        reset(contact_map, selection);
    }

    void reset(
        const svmp::search::ContactProximityMap& contact_map,
        ContactPairSelection selection = ContactPairSelection::AllPairs) {
        source_configuration_ = contact_map.source.configuration;
        target_configuration_ = contact_map.target.configuration;
        contact_revision_key_ = contact_map.revision_key();
        pairs_.clear();
        pairs_.reserve(contact_map.pairs.size());
        for (const auto& pair : contact_map.pairs) {
            if (!selected(pair.state, selection)) {
                continue;
            }
            ContactQuadraturePair qp;
            qp.source_entity = static_cast<MeshIndex>(pair.provenance.source_entity);
            qp.target_entity = static_cast<MeshIndex>(pair.provenance.target_entity);
            qp.source_gid = pair.provenance.source_gid;
            qp.target_gid = pair.provenance.target_gid;
            qp.source_surface_name = pair.provenance.source_surface_name;
            qp.target_surface_name = pair.provenance.target_surface_name;
            qp.canonical_owner_rank = pair.provenance.canonical_owner_rank;
            qp.source_label = pair.provenance.source_label;
            qp.target_label = pair.provenance.target_label;
            qp.source_kind = pair.provenance.source_kind;
            qp.target_kind = pair.provenance.target_kind;
            qp.source_configuration = pair.provenance.source_configuration;
            qp.target_configuration = pair.provenance.target_configuration;
            qp.state = pair.state;
            qp.pair_id = pair.provenance.pair_id;
            qp.time_level = pair.provenance.time_level;
            qp.source_point = pair.projection.source_point;
            qp.target_point = pair.projection.target_point;
            qp.source_local_coordinates = pair.projection.source_local_coordinates;
            qp.target_local_coordinates = pair.projection.target_local_coordinates;
            qp.normal = pair.projection.source_normal;
            qp.target_normal = pair.projection.target_normal;
            qp.tangent0 = pair.projection.tangent0;
            qp.tangent1 = pair.projection.tangent1;
            qp.tangential_reference0 = pair.projection.tangential_reference0;
            qp.tangential_reference1 = pair.projection.tangential_reference1;
            qp.unsigned_gap = pair.projection.unsigned_gap;
            qp.signed_gap = pair.projection.signed_gap;
            qp.tangential_slip_magnitude = pair.projection.tangential_slip_magnitude;
            qp.shell_thickness_offset = pair.projection.shell_thickness_offset;
            qp.projection_valid = pair.projection.valid;
            qp.tangential_frame_valid = pair.projection.tangential_frame_valid;
            qp.wrong_side_projection = pair.projection.wrong_side_projection;
            qp.side = pair.projection.side;
            qp.generation_policy = pair.provenance.generation_policy;
            qp.projection_operator.available = pair.projection.valid;
            qp.projection_operator.pair_id = pair.provenance.pair_id;
            qp.projection_operator.operator_family = "closest-point";
            qp.projection_operator.generation_policy = pair.provenance.generation_policy;
            qp.projection_operator.conservative = false;
            pairs_.push_back(qp);
        }
        rebuildPatches();
    }

    [[nodiscard]] const std::vector<ContactQuadraturePair>& quadraturePairs() const noexcept {
        return pairs_;
    }

    [[nodiscard]] const std::vector<ContactPatch>& contactPatches() const noexcept {
        return patches_;
    }

    [[nodiscard]] PairIterator begin() const noexcept {
        return pairs_.begin();
    }

    [[nodiscard]] PairIterator end() const noexcept {
        return pairs_.end();
    }

    [[nodiscard]] PatchIterator patchesBegin() const noexcept {
        return patches_.begin();
    }

    [[nodiscard]] PatchIterator patchesEnd() const noexcept {
        return patches_.end();
    }

    [[nodiscard]] const ContactQuadraturePair& pair(std::size_t index) const {
        return pairs_.at(index);
    }

    [[nodiscard]] svmp::Configuration sourceConfiguration() const noexcept {
        return source_configuration_;
    }

    [[nodiscard]] svmp::Configuration targetConfiguration() const noexcept {
        return target_configuration_;
    }

    [[nodiscard]] std::uint64_t contactRevisionKey() const noexcept {
        return contact_revision_key_;
    }

private:
    void rebuildPatches() {
        patches_.clear();
        using PatchKey = std::tuple<svmp::label_t, svmp::label_t, int, int, svmp::rank_t>;
        std::map<PatchKey, std::size_t> patch_indices;
        for (std::size_t i = 0; i < pairs_.size(); ++i) {
            const auto& qp = pairs_[i];
            const PatchKey key{qp.source_label,
                               qp.target_label,
                               static_cast<int>(qp.source_kind),
                               static_cast<int>(qp.target_kind),
                               qp.canonical_owner_rank};
            auto it = patch_indices.find(key);
            if (it == patch_indices.end()) {
                ContactPatch patch;
                patch.source_label = qp.source_label;
                patch.target_label = qp.target_label;
                patch.source_kind = qp.source_kind;
                patch.target_kind = qp.target_kind;
                patch.source_configuration = qp.source_configuration;
                patch.target_configuration = qp.target_configuration;
                patch.canonical_owner_rank = qp.canonical_owner_rank;
                patch.pair_indices.push_back(i);
                const std::size_t patch_index = patches_.size();
                patches_.push_back(std::move(patch));
                patch_indices.emplace(key, patch_index);
            } else {
                patches_[it->second].pair_indices.push_back(i);
            }
        }
    }

    [[nodiscard]] static bool selected(
        svmp::search::ContactPairState state,
        ContactPairSelection selection) noexcept {
        switch (selection) {
            case ContactPairSelection::AllPairs:
                return true;
            case ContactPairSelection::ActivePairsOnly:
                return state == svmp::search::ContactPairState::Active;
            case ContactPairSelection::ActiveAndProjectedPairs:
                return state == svmp::search::ContactPairState::Active ||
                       state == svmp::search::ContactPairState::Projected;
        }
        return false;
    }

    svmp::Configuration source_configuration_ = svmp::Configuration::Reference;
    svmp::Configuration target_configuration_ = svmp::Configuration::Reference;
    std::uint64_t contact_revision_key_ = 0;
    std::vector<ContactQuadraturePair> pairs_;
    std::vector<ContactPatch> patches_;
};

} // namespace assembly
} // namespace FE
} // namespace svmp

#endif // defined(SVMP_FE_WITH_MESH) && SVMP_FE_WITH_MESH

#endif // SVMP_FE_ASSEMBLY_CONTACTPAIRCONTEXT_H
