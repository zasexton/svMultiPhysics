/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#include "Geometry/RegionFrameBinding.h"

#include "Core/FEException.h"

#include <algorithm>
#include <sstream>
#include <utility>

namespace svmp {
namespace FE {
namespace geometry {

namespace {

[[nodiscard]] bool wholeDomain(const FrameRegionDescriptor& region) noexcept
{
    return region.kind == FrameRegionKind::WholeDomain;
}

[[nodiscard]] const RegionFrameBinding* exactBinding(
    const std::vector<RegionFrameBinding>& bindings,
    const FrameRegionDescriptor& region) noexcept
{
    const auto it = std::find_if(
        bindings.begin(), bindings.end(),
        [&region](const RegionFrameBinding& binding) {
            return sameFrameRegion(binding.region, region);
        });
    return it == bindings.end() ? nullptr : &(*it);
}

[[nodiscard]] const RegionFrameBinding* wholeDomainBinding(
    const std::vector<RegionFrameBinding>& bindings) noexcept
{
    const auto it = std::find_if(
        bindings.begin(), bindings.end(),
        [](const RegionFrameBinding& binding) {
            return wholeDomain(binding.region);
        });
    return it == bindings.end() ? nullptr : &(*it);
}

} // namespace

void RegionFrameRegistry::addFrame(CoordinateFrameDescriptor frame)
{
    FE_THROW_IF(frame.name.empty(), InvalidArgumentException,
                "RegionFrameRegistry::addFrame: frame name is empty");
    const auto validation = MovingFrameTransform::validate(frame);
    FE_THROW_IF(!validation.ok, InvalidArgumentException,
                "RegionFrameRegistry::addFrame: invalid frame '" + frame.name + "'");

    for (auto& existing : frames_) {
        if (existing.name == frame.name) {
            existing = std::move(frame);
            return;
        }
    }
    frames_.push_back(std::move(frame));
}

void RegionFrameRegistry::bindRegion(RegionFrameBinding binding)
{
    FE_THROW_IF(binding.frame_name.empty(), InvalidArgumentException,
                "RegionFrameRegistry::bindRegion: frame name is empty");

    for (auto& existing : bindings_) {
        if (sameFrameRegion(existing.region, binding.region)) {
            existing = std::move(binding);
            return;
        }
    }
    bindings_.push_back(std::move(binding));
}

void RegionFrameRegistry::clear()
{
    frames_.clear();
    bindings_.clear();
}

bool RegionFrameRegistry::hasFrame(std::string_view frame_name) const noexcept
{
    return frame(frame_name) != nullptr;
}

const CoordinateFrameDescriptor* RegionFrameRegistry::frame(
    std::string_view frame_name) const noexcept
{
    for (const auto& candidate : frames_) {
        if (candidate.name == frame_name) {
            return &candidate;
        }
    }
    return nullptr;
}

const CoordinateFrameDescriptor* RegionFrameRegistry::findFrameForRegion(
    const FrameRegionDescriptor& region) const noexcept
{
    const RegionFrameBinding* binding = exactBinding(bindings_, region);
    if (binding == nullptr && !wholeDomain(region)) {
        binding = wholeDomainBinding(bindings_);
    }
    return binding == nullptr ? nullptr : frame(binding->frame_name);
}

const CoordinateFrameDescriptor& RegionFrameRegistry::requireFrameForRegion(
    const FrameRegionDescriptor& region) const
{
    const RegionFrameBinding* binding = exactBinding(bindings_, region);
    if (binding == nullptr && !wholeDomain(region)) {
        binding = wholeDomainBinding(bindings_);
    }
    FE_THROW_IF(binding == nullptr, InvalidArgumentException,
                "RegionFrameRegistry::requireFrameForRegion: no coordinate frame is bound for " +
                    describeFrameRegion(region));
    const auto* out = frame(binding->frame_name);
    FE_THROW_IF(out == nullptr, InvalidArgumentException,
                "RegionFrameRegistry::requireFrameForRegion: coordinate frame '" +
                    binding->frame_name + "' is not registered for " +
                    describeFrameRegion(region));
    return *out;
}

RegionFrameValidationResult RegionFrameRegistry::validateBindings() const
{
    RegionFrameValidationResult result;
    for (const auto& binding : bindings_) {
        if (frame(binding.frame_name) == nullptr && binding.required) {
            result.ok = false;
            result.messages.push_back("required coordinate frame '" +
                                      binding.frame_name + "' is missing for " +
                                      describeFrameRegion(binding.region));
        }
    }
    return result;
}

bool sameFrameRegion(const FrameRegionDescriptor& lhs,
                     const FrameRegionDescriptor& rhs) noexcept
{
    return lhs.kind == rhs.kind &&
           lhs.name == rhs.name &&
           lhs.marker == rhs.marker &&
           lhs.entity_ids == rhs.entity_ids;
}

std::string describeFrameRegion(const FrameRegionDescriptor& region)
{
    std::ostringstream os;
    switch (region.kind) {
        case FrameRegionKind::WholeDomain:
            os << "whole domain";
            break;
        case FrameRegionKind::NamedRegion:
            os << "named region '" << region.name << "'";
            break;
        case FrameRegionKind::BoundaryMarker:
            os << "boundary marker " << region.marker;
            break;
        case FrameRegionKind::CellSet:
            os << "cell set with " << region.entity_ids.size() << " entries";
            break;
        case FrameRegionKind::MaterialIdSet:
            os << "material-id set with " << region.entity_ids.size() << " entries";
            break;
        case FrameRegionKind::InterfaceMarker:
            os << "interface marker " << region.marker;
            break;
    }
    return os.str();
}

} // namespace geometry
} // namespace FE
} // namespace svmp
