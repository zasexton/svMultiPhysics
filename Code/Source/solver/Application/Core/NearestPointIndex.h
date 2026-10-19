#pragma once

// Forwarding header: the nearest-point search lives in
// FE/Geometry/NearestPointIndex.h (namespace svmp::FE::geometry).

#include "FE/Geometry/NearestPointIndex.h"

namespace application {
namespace core {

using svmp::FE::geometry::NearestPointIndex;
using svmp::FE::geometry::NearestPointQueryResult;
using svmp::FE::geometry::NearestPointRecord;

} // namespace core
} // namespace application
