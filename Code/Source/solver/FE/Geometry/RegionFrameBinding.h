/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#ifndef SVMP_FE_GEOMETRY_REGIONFRAMEBINDING_H
#define SVMP_FE_GEOMETRY_REGIONFRAMEBINDING_H

#include "Core/Types.h"
#include "Geometry/MovingFrame.h"

#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

namespace svmp {
namespace FE {
namespace geometry {

enum class FrameRegionKind : std::uint8_t {
    WholeDomain,
    NamedRegion,
    BoundaryMarker,
    CellSet,
    MaterialIdSet,
    InterfaceMarker
};

struct FrameRegionDescriptor {
    FrameRegionKind kind{FrameRegionKind::WholeDomain};
    std::string name;
    int marker{-1};
    std::vector<GlobalIndex> entity_ids;
};

struct RegionFrameBinding {
    FrameRegionDescriptor region{};
    std::string frame_name;
    bool required{true};
};

struct RegionFrameValidationResult {
    bool ok{true};
    std::vector<std::string> messages;
};

class RegionFrameRegistry {
public:
    void addFrame(CoordinateFrameDescriptor frame);
    void bindRegion(RegionFrameBinding binding);
    void clear();

    [[nodiscard]] bool hasFrame(std::string_view frame_name) const noexcept;
    [[nodiscard]] const CoordinateFrameDescriptor* frame(
        std::string_view frame_name) const noexcept;

    [[nodiscard]] const CoordinateFrameDescriptor* findFrameForRegion(
        const FrameRegionDescriptor& region) const noexcept;

    [[nodiscard]] const CoordinateFrameDescriptor& requireFrameForRegion(
        const FrameRegionDescriptor& region) const;

    [[nodiscard]] RegionFrameValidationResult validateBindings() const;

    [[nodiscard]] const std::vector<CoordinateFrameDescriptor>& frames() const noexcept
    {
        return frames_;
    }

    [[nodiscard]] const std::vector<RegionFrameBinding>& bindings() const noexcept
    {
        return bindings_;
    }

private:
    std::vector<CoordinateFrameDescriptor> frames_{};
    std::vector<RegionFrameBinding> bindings_{};
};

[[nodiscard]] bool sameFrameRegion(const FrameRegionDescriptor& lhs,
                                   const FrameRegionDescriptor& rhs) noexcept;

[[nodiscard]] std::string describeFrameRegion(const FrameRegionDescriptor& region);

} // namespace geometry
} // namespace FE
} // namespace svmp

#endif // SVMP_FE_GEOMETRY_REGIONFRAMEBINDING_H
