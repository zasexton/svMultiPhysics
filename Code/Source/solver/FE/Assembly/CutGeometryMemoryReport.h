/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#ifndef SVMP_FE_ASSEMBLY_CUTGEOMETRYMEMORYREPORT_H
#define SVMP_FE_ASSEMBLY_CUTGEOMETRYMEMORYREPORT_H

/**
 * @file CutGeometryMemoryReport.h
 * @brief Approximate heap accounting of generated cut-geometry storage.
 *
 * The estimates count container capacities, heap strings beyond the
 * small-string buffer and a 16-byte malloc chunk rounding.  They are meant
 * for comparing storage layouts, not for exact allocator statistics.
 */

#include "Assembly/CutIntegrationContext.h"
#include "Interfaces/FreeSurfaceGeometrySnapshot.h"
#include "Interfaces/LevelSetInterfaceDomain.h"

#include <array>
#include <cstddef>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

namespace svmp::FE::assembly::memory_report {

[[nodiscard]] inline std::size_t allocationBytes(std::size_t bytes) noexcept
{
    if (bytes == 0u) {
        return 0u;
    }
    const std::size_t chunk = (bytes + 8u + 15u) & ~std::size_t{15u};
    return chunk < 32u ? 32u : chunk;
}

template <typename T>
[[nodiscard]] std::size_t vectorBytes(const std::vector<T>& values) noexcept
{
    return allocationBytes(values.capacity() * sizeof(T));
}

// A shared element array is split evenly between the copies sharing it, so
// the copies of one domain add up to a single array.
template <typename T>
[[nodiscard]] std::size_t vectorBytes(
    const interfaces::CopyOnWriteVector<T>& values) noexcept
{
    const auto share_count = values.shareCount();
    if (share_count <= 0) {
        return 0u;
    }
    return allocationBytes(values.capacity() * sizeof(T)) /
           static_cast<std::size_t>(share_count);
}

[[nodiscard]] inline std::size_t stringBytes(const std::string& value) noexcept
{
    return value.capacity() > 15u ? allocationBytes(value.capacity() + 1u) : 0u;
}

/** Count, quadrature points and bytes of one storage class. */
struct StorageClass {
    std::size_t count{0u};
    std::size_t points{0u};
    std::size_t bytes{0u};

    void add(std::size_t point_count, std::size_t byte_count) noexcept
    {
        ++count;
        points += point_count;
        bytes += byte_count;
    }
};

/** Volume storage split by side and full-cell equivalence. */
struct VolumeStorage {
    // [side][full]: side 0 negative, 1 positive; full 0 cut, 1 full cell.
    std::array<std::array<StorageClass, 2>, 2> classes{};

    StorageClass& at(geometry::CutIntegrationSide side, bool full) noexcept
    {
        return classes[side == geometry::CutIntegrationSide::Negative ? 0u : 1u]
                      [full ? 1u : 0u];
    }

    [[nodiscard]] std::size_t bytes() const noexcept
    {
        std::size_t total = 0u;
        for (const auto& side : classes) {
            for (const auto& entry : side) {
                total += entry.bytes;
            }
        }
        return total;
    }
};

[[nodiscard]] inline std::size_t provenanceHeapBytes(
    const geometry::CutQuadratureProvenance& provenance) noexcept
{
    return stringBytes(provenance.embedded_geometry_id) +
           stringBytes(provenance.cut_topology_id) +
           stringBytes(provenance.implicit_geometry_mode) +
           stringBytes(provenance.implicit_quadrature_backend) +
           stringBytes(provenance.selected_implicit_quadrature_backend) +
           stringBytes(provenance.implicit_fallback_policy) +
           stringBytes(provenance.implicit_fallback_status) +
           stringBytes(provenance.geometry_tangent_policy);
}

[[nodiscard]] inline std::size_t ruleHeapBytes(
    const geometry::CutQuadratureRule& rule) noexcept
{
    return vectorBytes(rule.points) + stringBytes(rule.policy.name) +
           provenanceHeapBytes(rule.provenance) +
           stringBytes(rule.provenance_id);
}

[[nodiscard]] inline std::size_t regionHeapBytes(
    const interfaces::CutInterfaceVolumeRegion& region) noexcept
{
    return stringBytes(region.topology_id) +
           stringBytes(region.implicit_quadrature_backend) +
           stringBytes(region.implicit_fallback_status) +
           vectorBytes(region.reference_subcells) +
           vectorBytes(region.quadrature_points);
}

[[nodiscard]] inline std::size_t fragmentHeapBytes(
    const interfaces::CutInterfaceFragment& fragment) noexcept
{
    return stringBytes(fragment.topology_id) +
           stringBytes(fragment.implicit_quadrature_backend) +
           stringBytes(fragment.implicit_fallback_status) +
           stringBytes(fragment.branch_id) +
           stringBytes(fragment.conditioning_diagnostic) +
           vectorBytes(fragment.vertices) +
           vectorBytes(fragment.quadrature_points) +
           vectorBytes(fragment.moment_certificate_points);
}

struct DomainStorage {
    VolumeStorage regions{};
    StorageClass fragments{};
    std::size_t sensitivity_bytes{0u};
    std::size_t container_slack_bytes{0u};

    [[nodiscard]] std::size_t bytes() const noexcept
    {
        return regions.bytes() + fragments.bytes + sensitivity_bytes +
               container_slack_bytes;
    }
};

inline void addRegion(VolumeStorage& storage,
                      const interfaces::CutInterfaceVolumeRegion& region)
{
    storage.at(region.side, region.full_cell_equivalent)
        .add(region.quadrature_points.size(),
             sizeof(region) + regionHeapBytes(region));
}

[[nodiscard]] inline DomainStorage domainStorage(
    const interfaces::LevelSetInterfaceDomain& domain)
{
    DomainStorage storage;
    for (const auto& region : domain.volumeRegions()) {
        addRegion(storage.regions, region);
    }
    for (const auto& fragment : domain.fragments()) {
        storage.fragments.add(fragment.quadrature_points.size(),
                              sizeof(fragment) + fragmentHeapBytes(fragment));
    }
    for (const auto& record : domain.sensitivityRecords()) {
        storage.sensitivity_bytes += sizeof(record) +
                                     stringBytes(record.target_kind) +
                                     stringBytes(record.construction_policy) +
                                     stringBytes(record.provenance_id) +
                                     vectorBytes(record.parent_geometry_dofs) +
                                     vectorBytes(record.samples);
        for (const auto& sample : record.samples) {
            storage.sensitivity_bytes +=
                vectorBytes(sample.influencing_parent_geometry_dofs) +
                vectorBytes(sample.shape_values) +
                vectorBytes(sample.shape_gradients);
        }
    }
    storage.container_slack_bytes =
        (domain.volumeRegions().capacity() - domain.volumeRegions().size()) *
            sizeof(interfaces::CutInterfaceVolumeRegion) +
        (domain.fragments().capacity() - domain.fragments().size()) *
            sizeof(interfaces::CutInterfaceFragment);
    // Domain copies share their arrays: split the bytes between them.
    const auto share_count = domain.volumeRegionShareCount();
    if (share_count > 1) {
        const auto divisor = static_cast<std::size_t>(share_count);
        for (auto& side : storage.regions.classes) {
            for (auto& entry : side) {
                entry.bytes /= divisor;
            }
        }
        storage.fragments.bytes /= divisor;
        storage.sensitivity_bytes /= divisor;
        storage.container_slack_bytes /= divisor;
    }
    return storage;
}

struct SnapshotStorage {
    std::size_t resident_bytes{0u};
    VolumeStorage volume_records{};
    StorageClass interface_records{};
    StorageClass boundary_records{};
    StorageClass contact_records{};
    DomainStorage domain{};
};

[[nodiscard]] inline std::size_t recordBytes(
    const interfaces::FreeSurfaceGeometryRuleRecord& record) noexcept
{
    return sizeof(record) + ruleHeapBytes(record.reference_rule) +
           vectorBytes(record.physical_rule.points) +
           vectorBytes(record.source_fragment_stable_ids) +
           stringBytes(record.topology_id) +
           vectorBytes(record.moment_certificate.moments);
}

[[nodiscard]] inline SnapshotStorage snapshotStorage(
    const interfaces::FreeSurfaceGeometrySnapshot& snapshot)
{
    using Role = interfaces::FreeSurfaceGeometryRuleRole;
    SnapshotStorage storage;
    storage.resident_bytes = snapshot.residentBytes();
    for (const auto& record : snapshot.rules()) {
        const auto points = record.reference_rule.points.size();
        const auto bytes = recordBytes(record);
        switch (record.role) {
        case Role::NegativeVolume:
        case Role::PositiveVolume:
            storage.volume_records
                .at(record.role == Role::NegativeVolume
                        ? geometry::CutIntegrationSide::Negative
                        : geometry::CutIntegrationSide::Positive,
                    record.reference_rule.full_cell_equivalent)
                .add(points, bytes);
            break;
        case Role::Interface:
            storage.interface_records.add(points, bytes);
            break;
        case Role::Contact:
            storage.contact_records.add(points, bytes);
            break;
        default:
            storage.boundary_records.add(points, bytes);
            break;
        }
    }
    storage.domain = domainStorage(snapshot.interfaceDomain());
    return storage;
}

struct ContextStorage {
    VolumeStorage volume_rules{};
    StorageClass interface_rules{};
    StorageClass facet_set_rules{};
    std::size_t metadata_bytes{0u};
    std::size_t binding_bytes{0u};
    std::size_t sensitivity_bytes{0u};
    std::size_t facet_handle_bytes{0u};
    std::size_t two_sided_binding_bytes{0u};
    std::size_t container_slack_bytes{0u};

    [[nodiscard]] std::size_t bytes() const noexcept
    {
        return volume_rules.bytes() + interface_rules.bytes +
               facet_set_rules.bytes + metadata_bytes + binding_bytes +
               sensitivity_bytes + facet_handle_bytes +
               two_sided_binding_bytes + container_slack_bytes;
    }
};

[[nodiscard]] inline ContextStorage contextStorage(
    const CutIntegrationContext& context)
{
    ContextStorage storage;
    for (const auto& rule : context.volumeRules()) {
        storage.volume_rules.at(rule.side, rule.full_cell_equivalent)
            .add(rule.points.size(), sizeof(rule) + ruleHeapBytes(rule));
    }
    for (const auto& rule : context.interfaceRules()) {
        storage.interface_rules.add(rule.points.size(),
                                    sizeof(rule) + ruleHeapBytes(rule));
    }
    for (const auto& rule : context.facetSetRules()) {
        storage.facet_set_rules.add(rule.points.size(),
                                    sizeof(rule) + ruleHeapBytes(rule));
    }
    for (const auto& metadata : context.metadata()) {
        storage.metadata_bytes += sizeof(metadata) +
                                  stringBytes(metadata.provenance_id) +
                                  stringBytes(metadata.cut_topology_id);
    }
    for (const auto& binding : context.bindings()) {
        storage.binding_bytes +=
            sizeof(binding) + vectorBytes(binding.visible_to_paths);
    }
    for (const auto& metadata : context.sensitivityMetadata()) {
        storage.sensitivity_bytes += sizeof(metadata) +
                                     vectorBytes(metadata.parent_geometry_dofs) +
                                     vectorBytes(metadata.samples) +
                                     vectorBytes(metadata.visible_to_paths);
    }
    for (const auto& handle : context.facetSetHandles()) {
        storage.facet_handle_bytes += sizeof(handle) +
                                      vectorBytes(handle.facets) +
                                      vectorBytes(handle.facet_metadata);
    }
    for (const int marker : context.generatedLevelSetInterfaceMarkers()) {
        const auto& bindings =
            context.generatedInterfaceTwoSidedBindingsForMarker(marker);
        storage.two_sided_binding_bytes += vectorBytes(bindings);
        for (const auto& binding : bindings) {
            storage.two_sided_binding_bytes +=
                stringBytes(binding.interface_topology_id) +
                vectorBytes(binding.negative_volume_region_stable_ids) +
                vectorBytes(binding.positive_volume_region_stable_ids);
        }
    }
    storage.container_slack_bytes =
        vectorBytes(context.volumeRules()) -
        context.volumeRules().size() * sizeof(geometry::CutQuadratureRule) +
        vectorBytes(context.metadata()) -
        context.metadata().size() * sizeof(CutCellAssemblyMetadata);
    return storage;
}

[[nodiscard]] inline double megabytes(std::size_t bytes) noexcept
{
    return static_cast<double>(bytes) / (1024.0 * 1024.0);
}

inline void appendVolume(std::ostringstream& out,
                         const std::string& prefix,
                         const VolumeStorage& storage)
{
    static constexpr std::array<const char*, 2> side_names{{"neg", "pos"}};
    static constexpr std::array<const char*, 2> full_names{{"cut", "full"}};
    for (std::size_t side = 0u; side < 2u; ++side) {
        for (std::size_t full = 0u; full < 2u; ++full) {
            const auto& entry = storage.classes[side][full];
            out << ' ' << prefix << '_' << side_names[side] << '_'
                << full_names[full] << "=" << entry.count << '/'
                << entry.points << '/' << megabytes(entry.bytes);
        }
    }
}

inline void appendClass(std::ostringstream& out,
                        const std::string& name,
                        const StorageClass& entry)
{
    out << ' ' << name << '=' << entry.count << '/' << entry.points << '/'
        << megabytes(entry.bytes);
}

/** Key=value text: classes are count/points/MB, totals are MB. */
[[nodiscard]] inline std::string formatDomain(const std::string& prefix,
                                              const DomainStorage& storage)
{
    std::ostringstream out;
    out << ' ' << prefix << "_mb=" << megabytes(storage.bytes());
    appendVolume(out, prefix + "_regions", storage.regions);
    appendClass(out, prefix + "_fragments", storage.fragments);
    return out.str();
}

[[nodiscard]] inline std::string formatSnapshot(const std::string& prefix,
                                                const SnapshotStorage& storage)
{
    std::ostringstream out;
    out << ' ' << prefix << "_resident_mb="
        << megabytes(storage.resident_bytes);
    appendVolume(out, prefix + "_records", storage.volume_records);
    appendClass(out, prefix + "_interface_records", storage.interface_records);
    appendClass(out, prefix + "_boundary_records", storage.boundary_records);
    appendClass(out, prefix + "_contact_records", storage.contact_records);
    out << formatDomain(prefix + "_domain", storage.domain);
    return out.str();
}

[[nodiscard]] inline std::string formatContext(const std::string& prefix,
                                               const ContextStorage& storage)
{
    std::ostringstream out;
    out << ' ' << prefix << "_mb=" << megabytes(storage.bytes());
    appendVolume(out, prefix + "_volume_rules", storage.volume_rules);
    appendClass(out, prefix + "_interface_rules", storage.interface_rules);
    appendClass(out, prefix + "_facet_set_rules", storage.facet_set_rules);
    out << ' ' << prefix << "_metadata_mb=" << megabytes(storage.metadata_bytes)
        << ' ' << prefix << "_bindings_mb=" << megabytes(storage.binding_bytes)
        << ' ' << prefix << "_two_sided_mb="
        << megabytes(storage.two_sided_binding_bytes) << ' ' << prefix
        << "_facet_handles_mb=" << megabytes(storage.facet_handle_bytes)
        << ' ' << prefix << "_slack_mb="
        << megabytes(storage.container_slack_bytes);
    return out.str();
}

} // namespace svmp::FE::assembly::memory_report

#endif // SVMP_FE_ASSEMBLY_CUTGEOMETRYMEMORYREPORT_H
