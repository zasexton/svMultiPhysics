#include "Interfaces/FreeSurfaceGeometrySnapshot.h"

#include "Assembly/Assembler.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <map>
#include <numbers>
#include <set>
#include <stdexcept>
#include <tuple>
#include <unordered_set>
#include <utility>

namespace svmp::FE::interfaces {
namespace {

constexpr std::uint64_t kHashOffset = 1469598103934665603ull;
constexpr std::uint64_t kHashPrime = 1099511628211ull;

void mix(std::uint64_t& hash, std::uint64_t value) noexcept
{
    hash ^= value;
    hash *= kHashPrime;
}

void mix(std::uint64_t& hash, std::string_view value) noexcept
{
    for (const char character : value) {
        mix(hash, static_cast<unsigned char>(character));
    }
}

[[nodiscard]] std::uint64_t canonicalRealBits(Real value) noexcept
{
    // IEEE signed zero is numerically identical geometry.  Canonicalize it so
    // rank-local orientation arithmetic cannot give owner and ghost copies
    // different content keys solely through +0 versus -0.
    if (value == Real{0.0}) {
        value = Real{0.0};
    }
    std::uint64_t bits{0};
    static_assert(sizeof(value) <= sizeof(bits));
    std::memcpy(&bits, &value, sizeof(value));
    return bits;
}

void mix(std::uint64_t& hash, Real value) noexcept
{
    mix(hash, canonicalRealBits(value));
}

[[nodiscard]] std::uint64_t stringDigest(std::string_view value) noexcept
{
    std::uint64_t hash = kHashOffset;
    mix(hash, value);
    return hash == 0u ? 1u : hash;
}

[[nodiscard]] bool finitePoint(const std::array<Real, 3>& point) noexcept
{
    return std::isfinite(point[0]) && std::isfinite(point[1]) &&
           std::isfinite(point[2]);
}

[[nodiscard]] Real dot(const std::array<Real, 3>& a,
                       const std::array<Real, 3>& b) noexcept
{
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

[[nodiscard]] Real norm(const std::array<Real, 3>& point) noexcept
{
    return std::sqrt(dot(point, point));
}

[[nodiscard]] std::array<Real, 3> mapReferenceGradient(
    const geometry::CutGeometryJacobian& inverse,
    const std::array<Real, 3>& gradient) noexcept
{
    std::array<Real, 3> result{{0.0, 0.0, 0.0}};
    for (std::size_t i = 0; i < 3u; ++i) {
        for (std::size_t j = 0; j < 3u; ++j) {
            result[i] += inverse[j][i] * gradient[j];
        }
    }
    const Real magnitude = norm(result);
    if (magnitude > Real{0.0}) {
        for (auto& value : result) {
            value /= magnitude;
        }
    }
    return result;
}

[[nodiscard]] bool insideReferenceCell(ElementType type,
                                       const std::array<Real, 3>& point,
                                       Real tolerance) noexcept
{
    const auto in = [tolerance](Real value, Real lower, Real upper) {
        return value >= lower - tolerance && value <= upper + tolerance;
    };
    switch (type) {
    case ElementType::Line2:
    case ElementType::Line3:
        return in(point[0], Real{-1.0}, Real{1.0});
    case ElementType::Quad4:
    case ElementType::Quad8:
    case ElementType::Quad9:
        return in(point[0], Real{-1.0}, Real{1.0}) &&
               in(point[1], Real{-1.0}, Real{1.0});
    case ElementType::Hex8:
    case ElementType::Hex20:
    case ElementType::Hex27:
        return in(point[0], Real{-1.0}, Real{1.0}) &&
               in(point[1], Real{-1.0}, Real{1.0}) &&
               in(point[2], Real{-1.0}, Real{1.0});
    case ElementType::Triangle3:
    case ElementType::Triangle6:
        return point[0] >= -tolerance && point[1] >= -tolerance &&
               point[0] + point[1] <= Real{1.0} + tolerance;
    case ElementType::Tetra4:
    case ElementType::Tetra10:
        return point[0] >= -tolerance && point[1] >= -tolerance &&
               point[2] >= -tolerance &&
               point[0] + point[1] + point[2] <= Real{1.0} + tolerance;
    case ElementType::Wedge6:
    case ElementType::Wedge15:
    case ElementType::Wedge18:
        return point[0] >= -tolerance && point[1] >= -tolerance &&
               point[0] + point[1] <= Real{1.0} + tolerance &&
               in(point[2], Real{-1.0}, Real{1.0});
    case ElementType::Pyramid5:
    case ElementType::Pyramid13:
    case ElementType::Pyramid14: {
        if (!in(point[2], Real{0.0}, Real{1.0})) {
            return false;
        }
        const Real width = Real{1.0} - point[2];
        return std::abs(point[0]) <= width + tolerance &&
               std::abs(point[1]) <= width + tolerance;
    }
    default:
        return false;
    }
}

[[nodiscard]] FreeSurfaceGeometryRuleRole roleFor(
    const geometry::CutQuadratureRule& rule,
    FreeSurfaceGeometryRuleRole nonvolume_role)
{
    if (rule.kind != geometry::CutQuadratureKind::Volume) {
        return nonvolume_role;
    }
    if (rule.side == geometry::CutIntegrationSide::Negative) {
        return FreeSurfaceGeometryRuleRole::NegativeVolume;
    }
    if (rule.side == geometry::CutIntegrationSide::Positive) {
        return FreeSurfaceGeometryRuleRole::PositiveVolume;
    }
    throw std::invalid_argument(
        "free-surface snapshot volume rule cannot have Interface side");
}

[[nodiscard]] bool negativeRole(FreeSurfaceGeometryRuleRole role) noexcept
{
    return role == FreeSurfaceGeometryRuleRole::NegativeVolume ||
           role == FreeSurfaceGeometryRuleRole::NegativeExteriorBoundary;
}

[[nodiscard]] bool positiveRole(FreeSurfaceGeometryRuleRole role) noexcept
{
    return role == FreeSurfaceGeometryRuleRole::PositiveVolume ||
           role == FreeSurfaceGeometryRuleRole::PositiveExteriorBoundary;
}

[[nodiscard]] bool volumeRole(FreeSurfaceGeometryRuleRole role) noexcept
{
    return role == FreeSurfaceGeometryRuleRole::NegativeVolume ||
           role == FreeSurfaceGeometryRuleRole::PositiveVolume;
}

using OwnershipRuleIdentity = std::array<std::uint64_t, 5>;

struct OwnedRuleDigest {
    OwnershipRuleIdentity identity{};
    std::uint64_t content_digest{0};
};

[[nodiscard]] OwnershipRuleIdentity ownershipRuleIdentity(
    const FreeSurfaceGeometryRuleRecord& record) noexcept
{
    const auto& provenance = record.reference_rule.provenance;
    return {{
        static_cast<std::uint64_t>(record.role),
        static_cast<std::uint64_t>(provenance.marker),
        provenance.cut_topology_revision,
        static_cast<std::uint64_t>(provenance.parent_entity_global_id),
        static_cast<std::uint64_t>(
            provenance.parent_boundary_entity_global_id)}};
}

void mixRuleContent(std::uint64_t& hash,
                    const FreeSurfaceGeometryRuleRecord& record) noexcept
{
    mix(hash, static_cast<std::uint64_t>(record.role));
    mix(hash, static_cast<std::uint64_t>(record.retention));
    mix(hash, static_cast<std::uint64_t>(
                  record.physical_boundary_marker + 1));
    mix(hash, record.reference_rule.provenance.cut_topology_revision);
    mix(hash, static_cast<std::uint64_t>(
                  record.reference_rule.provenance.marker));
    mix(hash, static_cast<std::uint64_t>(
                  record.reference_rule.provenance.parent_entity_global_id));
    mix(hash, static_cast<std::uint64_t>(
                  record.reference_rule.provenance
                      .parent_boundary_entity_global_id));
    mix(hash, static_cast<std::uint64_t>(
                  record.reference_rule.provenance.owner_rank + 1));
    mix(hash, record.reference_rule.provenance.cut_topology_id);
    mix(hash,
        record.reference_rule.provenance
            .selected_implicit_quadrature_backend);
    mix(hash, record.reference_rule.provenance.implicit_fallback_status);
    mix(hash, static_cast<std::uint64_t>(record.reference_rule.kind));
    mix(hash, static_cast<std::uint64_t>(record.reference_rule.side));
    mix(hash, static_cast<std::uint64_t>(
                  record.reference_rule.geometric_dimension + 1));
    mix(hash, record.reference_rule.measure);
    mix(hash, record.reference_rule.parent_measure);
    mix(hash, record.reference_rule.volume_fraction);
    mix(hash, static_cast<std::uint64_t>(
                  record.reference_rule.exact_polynomial_order + 1));
    for (const auto& point : record.reference_rule.points) {
        for (const auto value : point.parent_coordinate) {
            mix(hash, value);
        }
        for (const auto value : point.normal) {
            mix(hash, value);
        }
        for (const auto value : point.boundary_normal) {
            mix(hash, value);
        }
        for (const auto value : point.tangent) {
            mix(hash, value);
        }
        mix(hash, point.weight);
        mix(hash, point.reference_measure_factor);
        mix(hash, point.level_set_residual);
        mix(hash, point.gradient_norm);
    }
    mix(hash, record.physical_rule.physical_measure);
    for (const auto& point : record.physical_rule.points) {
        for (const auto value : point.reference_point) {
            mix(hash, value);
        }
        for (const auto value : point.physical_point) {
            mix(hash, value);
        }
        for (const auto value : point.normal) {
            mix(hash, value);
        }
        for (const auto value : point.boundary_normal) {
            mix(hash, value);
        }
        for (const auto value : point.tangent) {
            mix(hash, value);
        }
        mix(hash, point.absolute_jacobian_determinant);
        mix(hash, point.reference_weight);
        mix(hash, point.physical_weight);
    }
    for (const auto id : record.source_fragment_stable_ids) {
        mix(hash, id);
    }
}

[[nodiscard]] std::uint64_t ruleContentDigest(
    const FreeSurfaceGeometryRuleRecord& record) noexcept
{
    std::uint64_t hash = kHashOffset;
    mixRuleContent(hash, record);
    return hash == 0u ? 1u : hash;
}

[[nodiscard]] std::vector<OwnedRuleDigest> validateUniqueRuleOwnership(
    const std::vector<FreeSurfaceGeometryRuleRecord>& records,
    const assembly::IMeshAccess& mesh,
    const FreeSurfaceGeometryOwnershipCollective& collective,
    FreeSurfaceGeometryValidationLedger& ledger)
{
    constexpr std::size_t identity_width =
        std::tuple_size_v<OwnershipRuleIdentity>;
    constexpr std::size_t width = identity_width + 1u;
    if (collective.rank != mesh.parallelRank() ||
        collective.size != mesh.parallelSize() || collective.rank < 0 ||
        collective.size < 1 || collective.rank >= collective.size ||
        (collective.size > 1 &&
         !collective.all_gather_owned_rule_identity_values)) {
        throw std::invalid_argument(
            "free-surface snapshot ownership collective does not match the mesh communicator");
    }

    std::vector<std::uint64_t> local_owned_values;
    local_owned_values.reserve(ledger.owned_rule_count * width);
    for (const auto& record : records) {
        if (!record.locally_owned) {
            continue;
        }
        const auto identity = ownershipRuleIdentity(record);
        local_owned_values.insert(local_owned_values.end(),
                                  identity.begin(),
                                  identity.end());
        local_owned_values.push_back(ruleContentDigest(record));
    }

    std::vector<std::uint64_t> global_owned_values = local_owned_values;
    if (collective.size > 1) {
        global_owned_values =
            collective.all_gather_owned_rule_identity_values(
                local_owned_values);
    }
    if (global_owned_values.size() % width != 0u) {
        throw std::invalid_argument(
            "free-surface snapshot ownership collective returned a malformed identity stream");
    }

    std::map<OwnershipRuleIdentity, std::uint64_t> globally_owned_by_identity;
    for (std::size_t offset = 0; offset < global_owned_values.size();
         offset += width) {
        OwnershipRuleIdentity identity{};
        std::copy_n(global_owned_values.begin() +
                        static_cast<std::ptrdiff_t>(offset),
                    static_cast<std::ptrdiff_t>(width),
                    identity.begin());
        const auto content_digest =
            global_owned_values[offset + identity_width];
        if (!globally_owned_by_identity.emplace(identity, content_digest)
                 .second) {
            ++ledger.duplicate_rule_identity_count;
            throw std::invalid_argument(
                "free-surface snapshot found a rule owned by more than one rank");
        }
    }
    for (const auto& record : records) {
        const auto found = globally_owned_by_identity.find(
            ownershipRuleIdentity(record));
        if (found == globally_owned_by_identity.end()) {
            ++ledger.invalid_global_identity_count;
            throw std::invalid_argument(
                "free-surface snapshot found a local rule without one global owner");
        }
        if (found->second != ruleContentDigest(record)) {
            ++ledger.invalid_global_identity_count;
            throw std::invalid_argument(
                "free-surface snapshot found local rule content that differs "
                "from its global owner: rank=" +
                std::to_string(collective.rank) + " role=" +
                std::to_string(static_cast<int>(record.role)) +
                " marker=" +
                std::to_string(record.reference_rule.provenance.marker) +
                " parent_global_id=" +
                std::to_string(
                    record.reference_rule.provenance
                        .parent_entity_global_id) +
                " boundary_parent_global_id=" +
                std::to_string(
                    record.reference_rule.provenance
                        .parent_boundary_entity_global_id) +
                " owner_digest=" + std::to_string(found->second) +
                " local_digest=" +
                std::to_string(ruleContentDigest(record)));
        }
    }
    ledger.global_owned_rule_count = globally_owned_by_identity.size();
    std::vector<OwnedRuleDigest> globally_owned;
    globally_owned.reserve(globally_owned_by_identity.size());
    for (const auto& [identity, content_digest] :
         globally_owned_by_identity) {
        globally_owned.push_back(OwnedRuleDigest{
            .identity = identity,
            .content_digest = content_digest,
        });
    }
    return globally_owned;
}

[[nodiscard]] FreeSurfaceGeometryRevision makeRevision(
    const LevelSetInterfaceDomain& domain,
    const assembly::IMeshAccess& mesh,
    std::string domain_id)
{
    const auto& request = domain.request();
    FreeSurfaceGeometryRevision revision;
    revision.source_id = request.source.identifier();
    revision.domain_id = domain_id.empty() ? revision.source_id
                                           : std::move(domain_id);
    revision.interface_marker = request.interface_marker;
    revision.isovalue = request.isovalue;
    revision.source_layout_revision = request.source.layout_revision;
    revision.source_value_revision = request.source.value_revision;
    revision.mesh_geometry_revision = request.mesh_geometry_revision;
    revision.mesh_topology_revision = request.mesh_topology_revision;
    revision.ownership_revision = request.ownership_revision;
    revision.numbering_revision = mesh.numberingRevision();
    revision.quadrature_policy_key = request.quadrature_policy_key;
    return revision;
}

void canonicalizeDistributedRevision(
    FreeSurfaceGeometryRevision& revision,
    const FreeSurfaceGeometrySnapshotPolicy& policy,
    const FreeSurfaceGeometryOwnershipCollective& collective)
{
    if (collective.size == 1) {
        return;
    }
    if (!collective.all_gather_revision_values) {
        throw std::invalid_argument(
            "distributed free-surface snapshot requires a revision collective");
    }

    constexpr std::size_t common_value_count = 11u;
    constexpr std::size_t value_count = 15u;
    const std::array<std::uint64_t, value_count> local_values{{
        stringDigest(revision.source_id),
        stringDigest(revision.domain_id),
        static_cast<std::uint64_t>(revision.interface_marker),
        canonicalRealBits(revision.isovalue),
        revision.source_layout_revision,
        revision.source_value_revision,
        revision.quadrature_policy_key,
        canonicalRealBits(policy.tolerance),
        canonicalRealBits(policy.minimum_retained_volume_fraction),
        static_cast<std::uint64_t>(
            policy.minimum_achieved_quadrature_order),
        static_cast<std::uint64_t>(
            policy.require_complete_exterior_boundary_partition),
        revision.mesh_geometry_revision,
        revision.mesh_topology_revision,
        revision.ownership_revision,
        revision.numbering_revision,
    }};
    const auto gathered =
        collective.all_gather_revision_values(local_values);
    const auto expected_count =
        static_cast<std::size_t>(collective.size) * value_count;
    if (gathered.size() != expected_count) {
        throw std::invalid_argument(
            "free-surface snapshot revision collective returned a malformed value stream");
    }
    for (int rank = 1; rank < collective.size; ++rank) {
        const auto offset = static_cast<std::size_t>(rank) * value_count;
        if (!std::equal(gathered.begin(),
                        gathered.begin() +
                            static_cast<std::ptrdiff_t>(common_value_count),
                        gathered.begin() +
                            static_cast<std::ptrdiff_t>(offset))) {
            throw std::invalid_argument(
                "free-surface snapshot source or policy differs across ranks");
        }
    }

    const auto distributed_key = [&](std::size_t field,
                                     std::uint64_t tag) {
        std::uint64_t hash = kHashOffset;
        mix(hash, tag);
        mix(hash, static_cast<std::uint64_t>(collective.size));
        for (int rank = 0; rank < collective.size; ++rank) {
            mix(hash, static_cast<std::uint64_t>(rank));
            mix(hash,
                gathered[static_cast<std::size_t>(rank) * value_count +
                         field]);
        }
        return hash == 0u ? std::uint64_t{1u} : hash;
    };
    revision.mesh_geometry_revision = distributed_key(11u, 1u);
    revision.mesh_topology_revision = distributed_key(12u, 2u);
    revision.ownership_revision = distributed_key(13u, 3u);
    revision.numbering_revision = distributed_key(14u, 4u);
}

void requireContactRevision(
    const GeneratedInterfaceBoundaryIntersectionDomain& contact,
    const FreeSurfaceGeometryRevision& revision)
{
    const auto& request = contact.request();
    if (request.source.identifier() != revision.source_id ||
        request.source.layout_revision != revision.source_layout_revision ||
        request.source_value_revision != revision.source_value_revision ||
        request.interface_marker != revision.interface_marker ||
        request.isovalue != revision.isovalue ||
        request.mesh_geometry_revision != revision.mesh_geometry_revision ||
        request.mesh_topology_revision != revision.mesh_topology_revision ||
        request.ownership_revision != revision.ownership_revision ||
        request.quadrature_policy_key != revision.quadrature_policy_key) {
        throw std::invalid_argument(
            "contact domain does not match the free-surface snapshot revision");
    }
}

void requireActiveBoundaryRevision(
    const GeneratedActiveBoundaryDomain& active,
    const FreeSurfaceGeometryRevision& revision)
{
    const auto& request = active.request();
    if (request.source.identifier() != revision.source_id ||
        request.source.layout_revision != revision.source_layout_revision ||
        request.source_value_revision != revision.source_value_revision ||
        request.interface_marker != revision.interface_marker ||
        request.isovalue != revision.isovalue ||
        request.mesh_geometry_revision != revision.mesh_geometry_revision ||
        request.mesh_topology_revision != revision.mesh_topology_revision ||
        request.ownership_revision != revision.ownership_revision ||
        request.quadrature_policy_key != revision.quadrature_policy_key) {
        throw std::invalid_argument(
            "active-boundary domain does not match the free-surface snapshot revision");
    }
}

[[nodiscard]] std::vector<std::uint64_t> sourceIdsForActiveRule(
    const GeneratedActiveBoundaryDomain& domain,
    std::uint64_t stable_id)
{
    for (const auto& fragment : domain.fragments()) {
        if (fragment.stable_id == stable_id) {
            auto ids = fragment.source_interface_stable_ids;
            ids.insert(ids.end(),
                       fragment.source_contact_stable_ids.begin(),
                       fragment.source_contact_stable_ids.end());
            std::sort(ids.begin(), ids.end());
            ids.erase(std::unique(ids.begin(), ids.end()), ids.end());
            return ids;
        }
    }
    return {};
}

void completeAndValidateRuleIdentity(
    geometry::CutQuadratureRule& rule,
    const assembly::IMeshAccess& mesh,
    FreeSurfaceGeometryValidationLedger& ledger)
{
    const auto local_cell =
        static_cast<GlobalIndex>(rule.provenance.parent_entity);
    if (local_cell < 0 || local_cell >= mesh.numCells()) {
        ++ledger.invalid_global_identity_count;
        throw std::invalid_argument(
            "retained free-surface rule has an invalid local parent cell");
    }
    if (mesh.parallelSize() > 1 && !mesh.globalEntityIdsAvailable()) {
        ++ledger.invalid_global_identity_count;
        throw std::invalid_argument(
            "distributed free-surface snapshot requires globally unique mesh entity ids");
    }
    const auto expected_cell_id = mesh.globalEntityIdsAvailable()
                                      ? mesh.getCellGlobalId(local_cell)
                                      : local_cell;
    if (rule.provenance.parent_entity_global_id == INVALID_GLOBAL_INDEX) {
        rule.provenance.parent_entity_global_id = expected_cell_id;
    }
    const bool has_boundary_parent =
        rule.provenance.parent_boundary_entity >= 0;
    GlobalIndex expected_boundary_id = INVALID_GLOBAL_INDEX;
    int expected_owner = mesh.getCellOwnerRank(local_cell);
    if (has_boundary_parent) {
        const auto local_face = static_cast<GlobalIndex>(
            rule.provenance.parent_boundary_entity);
        expected_boundary_id = mesh.globalEntityIdsAvailable()
                                   ? mesh.getBoundaryFaceGlobalId(local_face)
                                   : local_face;
        expected_owner =
            mesh.getBoundaryFaceOwnerRank(local_face, local_cell);
        if (rule.provenance.parent_boundary_entity_global_id ==
            INVALID_GLOBAL_INDEX) {
            rule.provenance.parent_boundary_entity_global_id =
                expected_boundary_id;
        }
    }
    if (rule.provenance.owner_rank < 0 && mesh.parallelSize() == 1) {
        rule.provenance.owner_rank = expected_owner;
    }
    const bool invalid_owner = expected_owner < 0 ||
                               expected_owner >= mesh.parallelSize() ||
                               rule.provenance.owner_rank != expected_owner;
    const bool invalid_boundary =
        has_boundary_parent
            ? rule.provenance.parent_boundary_entity_global_id !=
                  expected_boundary_id
            : rule.provenance.parent_boundary_entity_global_id !=
                  INVALID_GLOBAL_INDEX;
    if (rule.provenance.parent_entity_global_id != expected_cell_id ||
        invalid_boundary || invalid_owner ||
        (mesh.isOwnedCell(local_cell) !=
         (expected_owner == mesh.parallelRank()))) {
        ++ledger.invalid_global_identity_count;
        throw std::invalid_argument(
            "retained free-surface rule has stale global identity or ownership metadata");
    }
}

void addRule(std::vector<FreeSurfaceGeometryRuleRecord>& records,
             FreeSurfaceGeometryValidationLedger& ledger,
             geometry::CutQuadratureRule rule,
             FreeSurfaceGeometryRuleRole role,
             const assembly::IMeshAccess& mesh,
             const FreeSurfaceGeometrySnapshotPolicy& policy,
             int physical_boundary_marker = -1,
             std::vector<std::uint64_t> source_ids = {})
{
    completeAndValidateRuleIdentity(rule, mesh, ledger);
    FreeSurfaceGeometryRuleRecord record;
    record.role = roleFor(rule, role);
    const bool boundary_rule =
        record.role == FreeSurfaceGeometryRuleRole::Contact ||
        record.role == FreeSurfaceGeometryRuleRole::NegativeExteriorBoundary ||
        record.role == FreeSurfaceGeometryRuleRole::PositiveExteriorBoundary;
    if (boundary_rule != (physical_boundary_marker >= 0)) {
        throw std::invalid_argument(
            "free-surface rule has invalid physical-boundary provenance");
    }
    record.physical_boundary_marker = physical_boundary_marker;
    record.retention =
        volumeRole(record.role) && !rule.full_cell_equivalent &&
                std::isfinite(rule.volume_fraction) &&
                rule.volume_fraction > Real{0.0} &&
                rule.volume_fraction < policy.minimum_retained_volume_fraction
            ? FreeSurfaceGeometryRetention::PrunedSmallVolume
            : FreeSurfaceGeometryRetention::Retained;
    record.locally_owned =
        rule.provenance.owner_rank == mesh.parallelRank();
    record.source_fragment_stable_ids = std::move(source_ids);
    if (record.source_fragment_stable_ids.empty() &&
        rule.provenance.source_stable_id != 0u) {
        record.source_fragment_stable_ids.push_back(
            rule.provenance.source_stable_id);
    }
    record.topology_id = rule.provenance.cut_topology_id;
    record.physical_rule =
        geometry::mapCutQuadratureRuleToPhysical(mesh, rule);
    record.reference_rule = std::move(rule);
    ++ledger.rule_count;
    ledger.quadrature_point_count += record.reference_rule.points.size();
    if (record.locally_owned) {
        ++ledger.owned_rule_count;
    }
    if (record.retention == FreeSurfaceGeometryRetention::Retained) {
        ++ledger.retained_rule_count;
    } else {
        ++ledger.pruned_rule_count;
    }
    records.push_back(std::move(record));
}

void validateRule(
    const FreeSurfaceGeometryRuleRecord& record,
    const assembly::IMeshAccess& mesh,
    const FreeSurfaceGeometryRevision& revision,
    const FreeSurfaceGeometrySnapshotPolicy& policy,
    const FreeSurfaceGeometryScalarEvaluator& scalar,
    FreeSurfaceGeometryValidationLedger& ledger)
{
    const auto& rule = record.reference_rule;
    const int geometric_dimension =
        rule.geometric_dimension >= 0
            ? rule.geometric_dimension
            : (rule.kind == geometry::CutQuadratureKind::Volume
                   ? mesh.dimension()
                   : std::max(0, mesh.dimension() - 1));
    // A positive-weight rule that is exact for all quadratic polynomials on a
    // nondegenerate m-dimensional set needs at least m+1 points: its centered
    // second-moment matrix has rank m, while n points have centered rank at
    // most n-1.  This is a necessary (not sufficient) fail-closed check that
    // catches impossible backend order claims before assembly.
    const bool impossible_quadratic_point_count =
        rule.exact_polynomial_order >= 2 && geometric_dimension > 0 &&
        rule.points.size() <
            static_cast<std::size_t>(geometric_dimension + 1);
    if (rule.provenance.implicit_geometry_mode.empty() ||
        rule.provenance.implicit_quadrature_backend.empty() ||
        rule.provenance.selected_implicit_quadrature_backend.empty() ||
        rule.provenance.selected_implicit_quadrature_backend == "Auto" ||
        rule.provenance.selected_implicit_quadrature_backend == "Unknown" ||
        rule.provenance.implicit_quadrature_backend !=
            rule.provenance.selected_implicit_quadrature_backend) {
        throw std::invalid_argument(
            "retained free-surface rule has incomplete or ambiguous represented-backend provenance");
    }
    if (rule.provenance.source_value_revision != revision.source_value_revision ||
        rule.provenance.predicate_policy_key != revision.quadrature_policy_key) {
        ++ledger.stale_revision_count;
        throw std::invalid_argument(
            "retained free-surface rule has a stale source revision");
    }
    if (rule.provenance.achieved_quadrature_order <
            policy.minimum_achieved_quadrature_order ||
        rule.exact_polynomial_order !=
            rule.provenance.achieved_quadrature_order ||
        impossible_quadratic_point_count ||
        (rule.provenance.requested_quadrature_order >= 0 &&
         rule.provenance.achieved_quadrature_order >
             rule.provenance.requested_quadrature_order)) {
        ++ledger.false_achieved_order_count;
        throw std::invalid_argument(
            "retained free-surface rule has a false or insufficient achieved order");
    }
    if (rule.points.size() != record.physical_rule.points.size()) {
        throw std::invalid_argument(
            "retained free-surface rule lost points during physical mapping");
    }
    Real reference_weight_sum{0.0};
    for (std::size_t q = 0; q < rule.points.size(); ++q) {
        const auto& point = rule.points[q];
        const auto& physical = record.physical_rule.points[q];
        reference_weight_sum += point.weight;
        if (!finitePoint(physical.reference_point) ||
            !finitePoint(physical.physical_point) ||
            !std::isfinite(point.weight) || !(point.weight > Real{0.0}) ||
            !std::isfinite(physical.physical_weight) ||
            !(physical.physical_weight > Real{0.0})) {
            ++ledger.invalid_weight_count;
            throw std::invalid_argument(
                "retained free-surface rule has an invalid mapped point or weight");
        }
        const auto parent = static_cast<GlobalIndex>(rule.provenance.parent_entity);
        if (!insideReferenceCell(mesh.getCellType(parent),
                                 physical.reference_point,
                                 Real{64.0} * policy.tolerance)) {
            ++ledger.outside_parent_point_count;
            throw std::invalid_argument(
                "retained free-surface point lies outside its parent reference cell");
        }
        if (record.role == FreeSurfaceGeometryRuleRole::Interface ||
            record.role == FreeSurfaceGeometryRuleRole::Contact) {
            if (!std::isfinite(point.level_set_residual)) {
                throw std::invalid_argument(
                    "retained interface/contact point has a non-finite represented root residual");
            }
            ledger.maximum_root_residual = std::max(
                ledger.maximum_root_residual,
                std::abs(point.level_set_residual));
            if (std::abs(point.level_set_residual) >
                Real{128.0} * policy.tolerance) {
                throw std::invalid_argument(
                    "retained interface/contact point fails its represented-backend root residual");
            }
        }
        if (!scalar.canEvaluateValue()) {
            continue;
        }
        const Real level_set =
            scalar.value(parent,
                         physical.reference_point,
                         rule.provenance) -
            revision.isovalue;
        if (!std::isfinite(level_set)) {
            throw std::invalid_argument(
                "free-surface snapshot scalar evaluator returned a non-finite value");
        }
        const Real scaled_tolerance = Real{128.0} * policy.tolerance;
        if ((negativeRole(record.role) && level_set > scaled_tolerance) ||
            (positiveRole(record.role) && level_set < -scaled_tolerance)) {
            ++ledger.invalid_phase_point_count;
            throw std::invalid_argument(
                "retained free-surface point has the wrong declared phase sign");
        }
        if (record.role == FreeSurfaceGeometryRuleRole::Interface ||
            record.role == FreeSurfaceGeometryRuleRole::Contact) {
            ledger.maximum_root_residual = std::max(
                ledger.maximum_root_residual, std::abs(level_set));
            if (std::abs(level_set) > scaled_tolerance) {
                throw std::invalid_argument(
                    "retained interface/contact point is off the represented zero set");
            }
            if (scalar.reference_gradient) {
                const auto gradient = scalar.reference_gradient(
                    parent, physical.reference_point, rule.provenance);
                const auto mapped_gradient = mapReferenceGradient(
                    physical.inverse_jacobian, gradient);
                if (norm(mapped_gradient) > Real{0.0} &&
                    norm(physical.normal) > Real{0.0}) {
                    const Real cosine = std::clamp(
                        dot(mapped_gradient, physical.normal),
                        Real{-1.0}, Real{1.0});
                    const Real angle = std::acos(cosine);
                    ledger.maximum_normal_angular_error =
                        std::max(ledger.maximum_normal_angular_error, angle);
                    if (angle > Real{1.0e-7}) {
                        throw std::invalid_argument(
                            "retained interface/contact normal disagrees with the represented scalar gradient");
                    }
                }
            }
        }
    }
    const Real moment_tolerance = Real{512.0} *
                                  std::numeric_limits<Real>::epsilon() *
                                  std::max(Real{1.0}, std::abs(rule.measure));
    const Real moment_error = std::abs(reference_weight_sum - rule.measure);
    ledger.maximum_constant_moment_error =
        std::max(ledger.maximum_constant_moment_error, moment_error);
    if (moment_error > moment_tolerance) {
        throw std::invalid_argument(
            "retained free-surface quadrature does not integrate constants to its declared measure");
    }
}

void accumulateLedger(const FreeSurfaceGeometryRuleRecord& record,
                      FreeSurfaceGeometryValidationLedger& ledger) noexcept
{
    if (record.retention != FreeSurfaceGeometryRetention::Retained) {
        return;
    }
    const Real reference = record.reference_rule.measure;
    const Real physical = record.physical_rule.physical_measure;
    switch (record.role) {
    case FreeSurfaceGeometryRuleRole::NegativeVolume:
        ledger.retained_negative_reference_volume += reference;
        ledger.retained_negative_physical_volume += physical;
        if (record.locally_owned) {
            ledger.owned_retained_negative_reference_volume += reference;
            ledger.owned_retained_negative_physical_volume += physical;
        }
        break;
    case FreeSurfaceGeometryRuleRole::PositiveVolume:
        ledger.retained_positive_reference_volume += reference;
        ledger.retained_positive_physical_volume += physical;
        if (record.locally_owned) {
            ledger.owned_retained_positive_reference_volume += reference;
            ledger.owned_retained_positive_physical_volume += physical;
        }
        break;
    case FreeSurfaceGeometryRuleRole::Interface:
        ledger.interface_reference_measure += reference;
        ledger.interface_physical_measure += physical;
        break;
    case FreeSurfaceGeometryRuleRole::Contact:
        ledger.contact_reference_measure += reference;
        ledger.contact_physical_measure += physical;
        break;
    default:
        break;
    }
}

void validateVolumePartition(
    const std::vector<FreeSurfaceGeometryRuleRecord>& records,
    const FreeSurfaceGeometrySnapshotPolicy& policy,
    FreeSurfaceGeometryValidationLedger& ledger)
{
    struct Measures {
        Real negative{0.0};
        Real positive{0.0};
        Real parent{0.0};
    };
    std::map<GlobalIndex, Measures> by_cell;
    for (const auto& record : records) {
        if (!volumeRole(record.role)) {
            continue;
        }
        auto& measures =
            by_cell[record.reference_rule.provenance.parent_entity_global_id];
        if (record.role == FreeSurfaceGeometryRuleRole::NegativeVolume) {
            measures.negative += record.reference_rule.measure;
        } else {
            measures.positive += record.reference_rule.measure;
        }
        measures.parent =
            std::max(measures.parent, record.reference_rule.parent_measure);
    }
    for (const auto& [cell, measures] : by_cell) {
        (void)cell;
        const Real error =
            std::abs(measures.negative + measures.positive - measures.parent);
        ledger.maximum_volume_partition_error =
            std::max(ledger.maximum_volume_partition_error, error);
        const Real tolerance = Real{512.0} *
                               std::numeric_limits<Real>::epsilon() *
                               std::max(Real{1.0}, measures.parent) +
                               policy.tolerance;
        if (error > tolerance) {
            throw std::invalid_argument(
                "positive and negative cut volumes do not partition their parent cell");
        }
    }
}

[[nodiscard]] std::uint64_t finalizeRevisionKey(
    const FreeSurfaceGeometryRevision& revision,
    const FreeSurfaceGeometrySnapshotPolicy& policy,
    std::span<const OwnedRuleDigest> globally_owned_rules) noexcept
{
    std::uint64_t hash = kHashOffset;
    mix(hash, revision.source_id);
    mix(hash, revision.domain_id);
    mix(hash, static_cast<std::uint64_t>(revision.interface_marker));
    mix(hash, revision.isovalue);
    mix(hash, revision.source_layout_revision);
    mix(hash, revision.source_value_revision);
    mix(hash, revision.mesh_geometry_revision);
    mix(hash, revision.mesh_topology_revision);
    mix(hash, revision.ownership_revision);
    mix(hash, revision.numbering_revision);
    mix(hash, revision.quadrature_policy_key);
    mix(hash, policy.tolerance);
    mix(hash, policy.minimum_retained_volume_fraction);
    mix(hash, static_cast<std::uint64_t>(
                  policy.minimum_achieved_quadrature_order));
    mix(hash, static_cast<std::uint64_t>(
                  policy.require_complete_exterior_boundary_partition));
    for (const auto& rule : globally_owned_rules) {
        for (const auto value : rule.identity) {
            mix(hash, value);
        }
        mix(hash, rule.content_digest);
    }
    return hash == 0u ? 1u : hash;
}

} // namespace

bool FreeSurfaceGeometryRevision::complete() const noexcept
{
    return !source_id.empty() && !domain_id.empty() && interface_marker >= 0 &&
           source_value_revision != 0u && snapshot_revision_key != 0u;
}

bool FreeSurfaceGeometryRevision::sameSourceState(
    const FreeSurfaceGeometryRevision& other) const noexcept
{
    return source_id == other.source_id && domain_id == other.domain_id &&
           interface_marker == other.interface_marker &&
           isovalue == other.isovalue &&
           source_layout_revision == other.source_layout_revision &&
           source_value_revision == other.source_value_revision &&
           mesh_geometry_revision == other.mesh_geometry_revision &&
           mesh_topology_revision == other.mesh_topology_revision &&
           ownership_revision == other.ownership_revision &&
           numbering_revision == other.numbering_revision &&
           quadrature_policy_key == other.quadrature_policy_key;
}

FreeSurfaceGeometrySnapshot::FreeSurfaceGeometrySnapshot(
    FreeSurfaceGeometryRevision revision,
    FreeSurfaceGeometrySnapshotPolicy policy,
    LevelSetInterfaceDomain interface_domain,
    std::vector<GeneratedInterfaceBoundaryIntersectionDomain> contact_domains,
    std::vector<GeneratedActiveBoundaryDomain> active_boundary_domains,
    std::vector<FreeSurfaceGeometryRuleRecord> rules,
    FreeSurfaceGeometryValidationLedger ledger)
    : revision_(std::move(revision))
    , policy_(policy)
    , interface_domain_(std::move(interface_domain))
    , contact_domains_(std::move(contact_domains))
    , active_boundary_domains_(std::move(active_boundary_domains))
    , rules_(std::move(rules))
    , ledger_(ledger)
{
}

const FreeSurfaceGeometryRevision&
FreeSurfaceGeometrySnapshot::revision() const noexcept
{
    return revision_;
}

const FreeSurfaceGeometrySnapshotPolicy&
FreeSurfaceGeometrySnapshot::policy() const noexcept
{
    return policy_;
}

const LevelSetInterfaceDomain&
FreeSurfaceGeometrySnapshot::interfaceDomain() const noexcept
{
    return interface_domain_;
}

const std::vector<GeneratedInterfaceBoundaryIntersectionDomain>&
FreeSurfaceGeometrySnapshot::contactDomains() const noexcept
{
    return contact_domains_;
}

const std::vector<GeneratedActiveBoundaryDomain>&
FreeSurfaceGeometrySnapshot::activeBoundaryDomains() const noexcept
{
    return active_boundary_domains_;
}

const std::vector<FreeSurfaceGeometryRuleRecord>&
FreeSurfaceGeometrySnapshot::rules() const noexcept
{
    return rules_;
}

const FreeSurfaceGeometryValidationLedger&
FreeSurfaceGeometrySnapshot::ledger() const noexcept
{
    return ledger_;
}

FreeSurfaceDiscreteFunctionalState evaluateFreeSurfaceDiscreteFunctional(
    const FreeSurfaceGeometrySnapshot& snapshot,
    const FreeSurfaceDiscreteFunctionalParameters& parameters)
{
    if (!snapshot.revision().complete()) {
        throw std::invalid_argument(
            "free-surface functional requires a revision-complete snapshot");
    }
    if (!std::isfinite(parameters.surface_tension) ||
        parameters.surface_tension < Real{0.0}) {
        throw std::invalid_argument(
            "free-surface functional requires finite nonnegative surface tension");
    }
    if (!std::isfinite(parameters.volume_multiplier)) {
        throw std::invalid_argument(
            "free-surface functional requires a finite volume multiplier");
    }
    std::map<int, FreeSurfaceDiscreteWallFunctionalState> walls;
    for (const auto& coefficient : parameters.young_wall_coefficients) {
        if (coefficient.boundary_marker < 0 ||
            !std::isfinite(
                coefficient.equilibrium_contact_angle_radians) ||
            !(coefficient.equilibrium_contact_angle_radians > Real{0.0}) ||
            !(coefficient.equilibrium_contact_angle_radians <
              std::numbers::pi_v<Real>)) {
            throw std::invalid_argument(
                "free-surface functional wall coefficient requires a nonnegative marker and an angle strictly between zero and pi");
        }
        FreeSurfaceDiscreteWallFunctionalState wall;
        wall.boundary_marker = coefficient.boundary_marker;
        wall.equilibrium_contact_angle_radians =
            coefficient.equilibrium_contact_angle_radians;
        if (!walls.emplace(coefficient.boundary_marker, wall).second) {
            throw std::invalid_argument(
                "free-surface functional has duplicate coefficients for one physical boundary");
        }
    }
    const auto ensure_geometry_wall = [&walls](int boundary_marker) {
        auto& wall = walls[boundary_marker];
        wall.boundary_marker = boundary_marker;
    };
    for (const auto& contact : snapshot.contactDomains()) {
        ensure_geometry_wall(contact.boundaryMarker());
    }
    for (const auto& active : snapshot.activeBoundaryDomains()) {
        ensure_geometry_wall(active.request().boundary_marker);
    }

    const auto volume_role =
        parameters.liquid_side == geometry::CutIntegrationSide::Negative
            ? FreeSurfaceGeometryRuleRole::NegativeVolume
            : FreeSurfaceGeometryRuleRole::PositiveVolume;
    const auto wall_role =
        parameters.liquid_side == geometry::CutIntegrationSide::Negative
            ? FreeSurfaceGeometryRuleRole::NegativeExteriorBoundary
            : FreeSurfaceGeometryRuleRole::PositiveExteriorBoundary;

    FreeSurfaceDiscreteFunctionalState state;
    state.snapshot_revision_key =
        snapshot.revision().snapshot_revision_key;
    state.liquid_side = parameters.liquid_side;
    state.surface_tension = parameters.surface_tension;
    state.volume_multiplier = parameters.volume_multiplier;

    for (const auto& record : snapshot.rules()) {
        if (!record.locally_owned ||
            record.retention != FreeSurfaceGeometryRetention::Retained) {
            continue;
        }
        const Real measure = record.physical_rule.physical_measure;
        if (!std::isfinite(measure) || measure < Real{0.0}) {
            throw std::invalid_argument(
                "free-surface functional encountered an invalid owned physical measure");
        }
        if (record.role == volume_role) {
            state.owned_liquid_volume += measure;
        } else if (record.role == FreeSurfaceGeometryRuleRole::Interface) {
            state.owned_liquid_gas_area += measure;
        } else if (record.role == wall_role) {
            if (record.physical_boundary_marker < 0) {
                throw std::invalid_argument(
                    "free-surface functional encountered exterior-boundary geometry without a physical marker");
            }
            state.owned_wetted_wall_area += measure;
            auto& wall = walls[record.physical_boundary_marker];
            wall.boundary_marker = record.physical_boundary_marker;
            wall.owned_wetted_wall_area += measure;
        } else if (record.role == FreeSurfaceGeometryRuleRole::Contact) {
            if (record.physical_boundary_marker < 0) {
                throw std::invalid_argument(
                    "free-surface functional encountered contact geometry without a physical marker");
            }
            state.owned_contact_measure += measure;
            auto& wall = walls[record.physical_boundary_marker];
            wall.boundary_marker = record.physical_boundary_marker;
            wall.owned_contact_measure += measure;
        }
    }

    state.liquid_gas_surface_energy =
        parameters.surface_tension * state.owned_liquid_gas_area;
    state.walls.reserve(walls.size());
    for (auto& [boundary_marker, wall] : walls) {
        (void)boundary_marker;
        if (wall.equilibrium_contact_angle_radians.has_value()) {
            wall.young_wall_energy =
                -parameters.surface_tension *
                std::cos(*wall.equilibrium_contact_angle_radians) *
                wall.owned_wetted_wall_area;
            state.young_wall_energy += wall.young_wall_energy;
        }
        state.walls.push_back(wall);
    }
    state.volume_constraint_potential =
        parameters.volume_multiplier * state.owned_liquid_volume;
    state.total_potential = state.liquid_gas_surface_energy +
                            state.young_wall_energy +
                            state.volume_constraint_potential;
    if (!std::isfinite(state.total_potential)) {
        throw std::invalid_argument(
            "free-surface functional produced a non-finite potential");
    }
    return state;
}

std::vector<const FreeSurfaceGeometryRuleRecord*>
FreeSurfaceGeometrySnapshot::retainedRules(
    FreeSurfaceGeometryRuleRole role) const
{
    std::vector<const FreeSurfaceGeometryRuleRecord*> result;
    for (const auto& rule : rules_) {
        if (rule.role == role &&
            rule.retention == FreeSurfaceGeometryRetention::Retained) {
            result.push_back(&rule);
        }
    }
    return result;
}

std::size_t FreeSurfaceGeometrySnapshot::residentBytes() const noexcept
{
    std::size_t bytes = sizeof(*this);
    bytes += revision_.source_id.capacity() + revision_.domain_id.capacity();
    const auto add_source = [&bytes](const LevelSetInterfaceSource& source) {
        bytes += source.evaluator_id.capacity();
    };
    const auto& interface_request = interface_domain_.request();
    add_source(interface_request.source);
    bytes += interface_request.implicit_geometry_mode.capacity();
    bytes += interface_request.implicit_quadrature_backend.capacity();
    bytes += interface_request.implicit_fallback_policy.capacity();
    bytes += interface_request.implicit_fallback_status.capacity();
    bytes += interface_request.required_implicit_cut_backend_qualification.capacity();
    bytes += interface_request.geometry_tangent_policy.capacity();
    bytes += interface_domain_.fragments().capacity() *
             sizeof(CutInterfaceFragment);
    for (const auto& fragment : interface_domain_.fragments()) {
        bytes += fragment.topology_id.capacity();
        bytes += fragment.implicit_quadrature_backend.capacity();
        bytes += fragment.implicit_fallback_status.capacity();
        bytes += fragment.branch_id.capacity();
        bytes += fragment.conditioning_diagnostic.capacity();
        bytes += fragment.vertices.capacity() * sizeof(CutInterfaceVertex);
        bytes += fragment.quadrature_points.capacity() *
                 sizeof(CutInterfaceQuadraturePoint);
    }
    bytes += interface_domain_.volumeRegions().capacity() *
             sizeof(CutInterfaceVolumeRegion);
    for (const auto& region : interface_domain_.volumeRegions()) {
        bytes += region.topology_id.capacity();
        bytes += region.implicit_quadrature_backend.capacity();
        bytes += region.implicit_fallback_status.capacity();
        bytes += region.quadrature_points.capacity() *
                 sizeof(geometry::CutQuadraturePoint);
    }
    bytes += interface_domain_.sensitivityRecords().capacity() *
             sizeof(GeneratedInterfaceSensitivityRecord);
    for (const auto& record : interface_domain_.sensitivityRecords()) {
        bytes += record.target_kind.capacity();
        bytes += record.construction_policy.capacity();
        bytes += record.provenance_id.capacity();
        bytes += record.parent_geometry_dofs.capacity() * sizeof(MeshIndex);
        bytes += record.samples.capacity() *
                 sizeof(GeneratedInterfaceSensitivitySample);
        for (const auto& sample : record.samples) {
            bytes += sample.influencing_parent_geometry_dofs.capacity() *
                     sizeof(MeshIndex);
            bytes += sample.shape_values.capacity() * sizeof(Real);
            bytes += sample.shape_gradients.capacity() *
                     sizeof(std::array<Real, 3>);
        }
    }
    bytes += contact_domains_.capacity() *
             sizeof(GeneratedInterfaceBoundaryIntersectionDomain);
    for (const auto& domain : contact_domains_) {
        add_source(domain.request().source);
        bytes += domain.request().generated_domain_id.capacity();
        bytes += domain.fragments().capacity() *
                 sizeof(GeneratedInterfaceBoundaryIntersectionFragment);
        for (const auto& fragment : domain.fragments()) {
            bytes += fragment.represented_implicit_geometry_mode.capacity();
            bytes += fragment.represented_implicit_quadrature_backend.capacity();
            bytes += fragment.represented_implicit_fallback_status.capacity();
            bytes += fragment.topology_id.capacity();
            bytes += fragment.diagnostic.capacity();
            bytes += fragment.vertices.capacity() *
                     sizeof(std::array<Real, 3>);
            bytes += fragment.quadrature_points.capacity() *
                     sizeof(GeneratedInterfaceBoundaryIntersectionQuadraturePoint);
        }
    }
    bytes += active_boundary_domains_.capacity() *
             sizeof(GeneratedActiveBoundaryDomain);
    for (const auto& domain : active_boundary_domains_) {
        add_source(domain.request().source);
        bytes += domain.request().generated_domain_id.capacity();
        bytes += domain.fragments().capacity() *
                 sizeof(GeneratedActiveBoundaryFragment);
        for (const auto& fragment : domain.fragments()) {
            bytes += fragment.source_contact_stable_ids.capacity() *
                     sizeof(std::uint64_t);
            bytes += fragment.source_interface_stable_ids.capacity() *
                     sizeof(std::uint64_t);
            bytes += fragment.represented_implicit_geometry_mode.capacity();
            bytes += fragment.represented_implicit_quadrature_backend.capacity();
            bytes += fragment.represented_implicit_fallback_status.capacity();
            bytes += fragment.topology_id.capacity();
            bytes += fragment.vertices.capacity() *
                     sizeof(std::array<Real, 3>);
            bytes += fragment.quadrature_points.capacity() *
                     sizeof(geometry::CutQuadraturePoint);
        }
    }
    bytes += rules_.capacity() * sizeof(FreeSurfaceGeometryRuleRecord);
    for (const auto& rule : rules_) {
        bytes += rule.reference_rule.points.capacity() *
                 sizeof(geometry::CutQuadraturePoint);
        bytes += rule.physical_rule.points.capacity() *
                 sizeof(geometry::MappedCutQuadraturePoint);
        bytes += rule.source_fragment_stable_ids.capacity() *
                 sizeof(std::uint64_t);
        bytes += rule.topology_id.capacity();
        bytes += rule.reference_rule.policy.name.capacity();
        bytes += rule.reference_rule.provenance.embedded_geometry_id.capacity();
        bytes += rule.reference_rule.provenance.cut_topology_id.capacity();
        bytes += rule.reference_rule.provenance.implicit_geometry_mode.capacity();
        bytes += rule.reference_rule.provenance.implicit_quadrature_backend.capacity();
        bytes += rule.reference_rule.provenance.selected_implicit_quadrature_backend.capacity();
        bytes += rule.reference_rule.provenance.implicit_fallback_policy.capacity();
        bytes += rule.reference_rule.provenance.implicit_fallback_status.capacity();
        bytes += rule.reference_rule.provenance.geometry_tangent_policy.capacity();
        bytes += rule.reference_rule.provenance_id.capacity();
    }
    return bytes;
}

std::shared_ptr<const FreeSurfaceGeometrySnapshot>
buildFreeSurfaceGeometrySnapshot(
    LevelSetInterfaceDomain interface_domain,
    std::vector<GeneratedInterfaceBoundaryIntersectionDomain> contact_domains,
    std::vector<GeneratedActiveBoundaryDomain> active_boundary_domains,
    const assembly::IMeshAccess& mesh,
    FreeSurfaceGeometrySnapshotPolicy policy,
    FreeSurfaceGeometryScalarEvaluator scalar,
    std::string domain_id,
    FreeSurfaceGeometryOwnershipCollective ownership_collective)
{
    if (!interface_domain.request().valid() || !(policy.tolerance > Real{0.0}) ||
        !(policy.minimum_retained_volume_fraction > Real{0.0}) ||
        !(policy.minimum_retained_volume_fraction < Real{1.0}) ||
        policy.minimum_achieved_quadrature_order < 0) {
        throw std::invalid_argument(
            "free-surface geometry snapshot received an invalid domain or policy");
    }
    if (mesh.parallelSize() < 1 || mesh.parallelRank() < 0 ||
        mesh.parallelRank() >= mesh.parallelSize() ||
        (mesh.parallelSize() > 1 && !mesh.globalEntityIdsAvailable())) {
        throw std::invalid_argument(
            "free-surface geometry snapshot requires valid communicator metadata and distributed global entity ids");
    }
    auto revision = makeRevision(interface_domain, mesh, std::move(domain_id));
    FreeSurfaceGeometryValidationLedger ledger;
    std::vector<FreeSurfaceGeometryRuleRecord> records;

    auto volume_rules = interface_domain.volumeQuadratureRules();
    auto interface_rules = interface_domain.interfaceQuadratureRules();
    records.reserve(volume_rules.size() + interface_rules.size());
    for (auto& rule : volume_rules) {
        addRule(records,
                ledger,
                std::move(rule),
                FreeSurfaceGeometryRuleRole::Interface,
                mesh,
                policy);
    }
    for (auto& rule : interface_rules) {
        addRule(records,
                ledger,
                std::move(rule),
                FreeSurfaceGeometryRuleRole::Interface,
                mesh,
                policy);
    }

    std::map<int, const GeneratedInterfaceBoundaryIntersectionDomain*>
        contact_by_boundary;
    for (const auto& contact : contact_domains) {
        requireContactRevision(contact, revision);
        const auto [iterator, inserted] = contact_by_boundary.emplace(
            contact.boundaryMarker(), &contact);
        (void)iterator;
        if (!inserted) {
            throw std::invalid_argument(
                "free-surface snapshot has duplicate contact domains for one boundary marker");
        }
        const auto provenance =
            validateGeneratedInterfaceBoundaryProvenance(contact,
                                                         interface_domain);
        ledger.contact_fragment_count +=
            provenance.active_contact_fragment_count;
        ledger.referenced_surface_fragment_count +=
            provenance.referenced_source_surface_fragment_count;
        ledger.orphan_contact_fragment_count +=
            provenance.orphan_contact_fragment_count;
        ledger.stale_revision_count += provenance.stale_revision_count;
        auto rules = contact.intersectionQuadratureRules();
        for (auto& rule : rules) {
            addRule(records,
                    ledger,
                    std::move(rule),
                    FreeSurfaceGeometryRuleRole::Contact,
                    mesh,
                    policy,
                    contact.boundaryMarker());
        }
    }
    if (ledger.orphan_contact_fragment_count != 0u ||
        ledger.stale_revision_count != 0u) {
        throw std::invalid_argument(
            "free-surface snapshot rejected stale or orphan contact geometry");
    }

    struct ActivePair {
        const GeneratedActiveBoundaryDomain* negative{nullptr};
        const GeneratedActiveBoundaryDomain* positive{nullptr};
    };
    std::map<int, ActivePair> active_by_boundary;
    for (const auto& active : active_boundary_domains) {
        requireActiveBoundaryRevision(active, revision);
        auto& pair = active_by_boundary[active.request().boundary_marker];
        auto*& slot = active.request().side ==
                              geometry::CutIntegrationSide::Negative
                          ? pair.negative
                          : pair.positive;
        if (slot != nullptr) {
            throw std::invalid_argument(
                "free-surface snapshot has duplicate active-boundary phase domains");
        }
        slot = &active;
        auto rules = active.boundaryQuadratureRules();
        for (auto& rule : rules) {
            const auto role = active.request().side ==
                                      geometry::CutIntegrationSide::Negative
                                  ? FreeSurfaceGeometryRuleRole::
                                        NegativeExteriorBoundary
                                  : FreeSurfaceGeometryRuleRole::
                                        PositiveExteriorBoundary;
            const auto stable_id = rule.provenance.cut_topology_revision;
            addRule(records,
                    ledger,
                    std::move(rule),
                    role,
                    mesh,
                    policy,
                    active.request().boundary_marker,
                    sourceIdsForActiveRule(active, stable_id));
        }
    }
    for (const auto& [boundary, contact] : contact_by_boundary) {
        const auto found = active_by_boundary.find(boundary);
        if (found == active_by_boundary.end() ||
            found->second.negative == nullptr ||
            found->second.positive == nullptr) {
            if (policy.require_complete_exterior_boundary_partition) {
                throw std::invalid_argument(
                    "free-surface snapshot is missing a positive/negative exterior-boundary partition");
            }
            continue;
        }
        const auto partition = validateGeneratedActiveBoundaryPartition(
            *found->second.negative,
            *found->second.positive,
            interface_domain,
            *contact,
            mesh);
        ledger.maximum_boundary_partition_error = std::max(
            ledger.maximum_boundary_partition_error,
            partition.max_partition_error);
    }
    if (policy.require_complete_exterior_boundary_partition &&
        active_by_boundary.size() != contact_by_boundary.size()) {
        throw std::invalid_argument(
            "free-surface snapshot active-boundary/contact marker sets differ");
    }

    std::set<std::tuple<FreeSurfaceGeometryRuleRole,
                        int,
                        GlobalIndex,
                        GlobalIndex,
                        std::uint64_t>>
        unique_rules;
    for (const auto& record : records) {
        const auto identity = std::make_tuple(
            record.role,
            record.reference_rule.provenance.marker,
            record.reference_rule.provenance.parent_entity_global_id,
            record.reference_rule.provenance
                .parent_boundary_entity_global_id,
            record.reference_rule.provenance.cut_topology_revision);
        if (!unique_rules.insert(identity).second) {
            ++ledger.duplicate_rule_identity_count;
            throw std::invalid_argument(
                "free-surface snapshot contains a duplicate retained rule identity");
        }
        validateRule(record,
                     mesh,
                     revision,
                     policy,
                     scalar,
                     ledger);
        accumulateLedger(record, ledger);
    }
    validateVolumePartition(records, policy, ledger);
    const auto globally_owned_rule_digests = validateUniqueRuleOwnership(
        records, mesh, ownership_collective, ledger);
    canonicalizeDistributedRevision(
        revision, policy, ownership_collective);
    revision.snapshot_revision_key =
        finalizeRevisionKey(revision, policy, globally_owned_rule_digests);
    if (!revision.complete()) {
        throw std::invalid_argument(
            "free-surface geometry snapshot revision is incomplete");
    }
    return std::shared_ptr<const FreeSurfaceGeometrySnapshot>(
        new FreeSurfaceGeometrySnapshot(
            std::move(revision),
            policy,
            std::move(interface_domain),
            std::move(contact_domains),
            std::move(active_boundary_domains),
            std::move(records),
            ledger));
}

std::shared_ptr<const FreeSurfaceGeometrySnapshot>
FreeSurfaceGeometrySnapshotCache::find(std::uint64_t revision_key)
{
    evictExpired();
    const auto found = snapshots_.find(revision_key);
    if (found == snapshots_.end()) {
        ++statistics_.miss_count;
        return {};
    }
    auto snapshot = found->second.lock();
    if (!snapshot) {
        snapshots_.erase(found);
        ++statistics_.expired_eviction_count;
        ++statistics_.miss_count;
        return {};
    }
    ++statistics_.hit_count;
    return snapshot;
}

void FreeSurfaceGeometrySnapshotCache::insert(
    std::shared_ptr<const FreeSurfaceGeometrySnapshot> snapshot)
{
    if (!snapshot || !snapshot->revision().complete()) {
        throw std::invalid_argument(
            "free-surface snapshot cache requires a complete immutable snapshot");
    }
    evictExpired();
    snapshots_[snapshot->revision().snapshot_revision_key] = snapshot;
    (void)statistics();
}

void FreeSurfaceGeometrySnapshotCache::evictExpired()
{
    for (auto iterator = snapshots_.begin(); iterator != snapshots_.end();) {
        if (iterator->second.expired()) {
            iterator = snapshots_.erase(iterator);
            ++statistics_.expired_eviction_count;
        } else {
            ++iterator;
        }
    }
}

FreeSurfaceGeometrySnapshotCacheStatistics
FreeSurfaceGeometrySnapshotCache::statistics()
{
    evictExpired();
    statistics_.live_snapshot_count = 0u;
    statistics_.live_resident_bytes = 0u;
    for (const auto& [key, weak] : snapshots_) {
        (void)key;
        if (const auto snapshot = weak.lock()) {
            ++statistics_.live_snapshot_count;
            statistics_.live_resident_bytes += snapshot->residentBytes();
        }
    }
    statistics_.peak_live_snapshot_count = std::max(
        statistics_.peak_live_snapshot_count,
        statistics_.live_snapshot_count);
    statistics_.peak_live_resident_bytes = std::max(
        statistics_.peak_live_resident_bytes,
        statistics_.live_resident_bytes);
    return statistics_;
}

} // namespace svmp::FE::interfaces
