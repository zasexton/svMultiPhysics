#include "Search/MultiMeshInterface.h"

#include <gtest/gtest.h>

using namespace svmp;
using namespace svmp::search;

TEST(InterfaceProvenance, AcceptsConsistentLogicalIdentity)
{
  InterfaceMap map;
  map.source.logical_region = LogicalInterfaceRegionId{
      LogicalInterfaceRegionKind::RotatingRegion, "rotor-a", "rotor", 4, 7};
  map.source.boundary_label = 4;
  map.target.logical_region = LogicalInterfaceRegionId{
      LogicalInterfaceRegionKind::StationaryRegion, "stator-a", "stator", 5, 7};
  map.target.boundary_label = 5;

  InterfacePair pair;
  pair.source_logical_region = map.source.logical_region;
  pair.target_logical_region = map.target.logical_region;
  map.pairs.push_back(pair);

  const auto diagnostic = validate_interface_provenance(map);
  EXPECT_TRUE(diagnostic.ok);
  EXPECT_TRUE(diagnostic.messages.empty());
}

TEST(InterfaceProvenance, ReportsMismatchedPhysicalLabelAndPersistentIdentity)
{
  InterfaceMap map;
  map.source.logical_region = LogicalInterfaceRegionId{
      LogicalInterfaceRegionKind::SlidingInterface, "slide-a", "slide", 4, 1};
  map.source.boundary_label = 99;

  InterfacePair pair;
  pair.source_logical_region = LogicalInterfaceRegionId{
      LogicalInterfaceRegionKind::SlidingInterface, "slide-b", "slide", 4, 1};
  map.pairs.push_back(pair);

  const auto diagnostic = validate_interface_provenance(map);
  EXPECT_FALSE(diagnostic.ok);
  ASSERT_GE(diagnostic.messages.size(), 2u);
}
