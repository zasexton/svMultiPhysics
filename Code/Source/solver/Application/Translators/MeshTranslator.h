#pragma once

#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "Mesh/Mesh.h"

class MeshParameters;
class FaceParameters;

namespace svmp::FE::level_set {
struct ActiveCutVolumeRequest;
}

namespace application {
namespace translators {

// Startup partition weights from the sign of a vertex scalar field of the
// mesh file (the initial level set of a free surface): cells whose vertex
// values all lie below the isovalue weigh `negative`, all above `positive`,
// otherwise `cut`.  ParMETIS then balances the cell count and the weighted
// work per rank.  Empty `vertex_field` disables the weighting.  With
// `fallback_to_unweighted` a vertex field that is missing or unusable at
// startup gives the unweighted partition (and one rank-0 line) instead of an
// error; the default weighting of unfitted free-surface runs sets it.
struct MeshPartitionWeights {
  std::string vertex_field{};
  double isovalue{0.0};
  double negative{1.0};
  double positive{1.0};
  double cut{1.0};
  bool fallback_to_unweighted{false};
};

// <Partition_weighting>: 'none', 'free_surface', or 'auto' (the default when
// the key is absent): free-surface weighting for a multi-rank run with an
// unfitted level-set free surface, none otherwise.
enum class PartitionWeightingMode { None, FreeSurface, Auto };

// The level set of a run's unfitted level-set free surface.
struct FreeSurfaceLevelSet {
  std::string level_set_field{};
  double isovalue{0.0};
  bool negative_side_active{true};
};

class MeshTranslator {
public:
  // `minimum_ghost_layers` applies only when the deck leaves <Ghost_layers>
  // unset; an explicit value is always honored.
  static std::shared_ptr<svmp::Mesh> loadMesh(const MeshParameters& params,
                                              int minimum_ghost_layers = 0,
                                              const MeshPartitionWeights& partition_weights = {});

  // The <Partition_weighting> mode of `params` ('auto' when the key is
  // absent); throws for an unknown value.
  static PartitionWeightingMode partitionWeightingMode(const MeshParameters& params);

  // The unfitted level-set free surface among a run's cut-volume requests:
  // the first free-surface boundary request.  Fitted (ALE) free surfaces,
  // two-fluid material interfaces and equation-level cut domains make no
  // such request, so they give std::nullopt.
  static std::optional<FreeSurfaceLevelSet> unfittedFreeSurfaceLevelSet(
      const std::vector<svmp::FE::level_set::ActiveCutVolumeRequest>& requests);

  // The startup partition weights of a mesh of a run on `world_size` ranks
  // whose unfitted free surface is `free_surface`:
  //   - none on one rank (nothing is partitioned) and for 'none';
  //   - 'free_surface': the free-surface weights; throws without an
  //     unfitted free surface, and a missing level-set field is an error;
  //   - 'auto': the free-surface weights when the run has an unfitted free
  //     surface, falling back to the unweighted partition when the
  //     level-set field is not available at startup; none otherwise.
  static MeshPartitionWeights startupPartitionWeights(
      const MeshParameters& params,
      const std::optional<FreeSurfaceLevelSet>& free_surface,
      int world_size);

  // The startup partition weights of <Partition_weighting>free_surface:
  // <Partition_weight_active>, <_inactive> and <_cut> mapped onto the
  // negative, positive and cut cells of the level-set vertex field
  // `level_set_field` at `isovalue`; the active (liquid) side is the
  // negative one when `negative_side_active`.  Throws for a weight that is
  // negative, non-finite or above 1e4.
  static MeshPartitionWeights freeSurfacePartitionWeights(const MeshParameters& params,
                                                          const std::string& level_set_field,
                                                          double isovalue,
                                                          bool negative_side_active);

private:
  static std::string detectFormat(const std::string& file_path);

  // Face files must be disjoint; `allow_overlapping_face_files` keeps the
  // legacy last-listed labeling of decks written with overlapping files.
  static void applyFaceLabels(svmp::Mesh& mesh,
                              const std::vector<FaceParameters*>& face_params,
                              const std::string& mesh_name,
                              bool allow_overlapping_face_files);

  static void applyDomainLabels(svmp::Mesh& mesh, const MeshParameters& params);
};

} // namespace translators
} // namespace application
