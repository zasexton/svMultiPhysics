#pragma once

#include <memory>
#include <string>
#include <vector>

#include "Mesh/Mesh.h"

class MeshParameters;
class FaceParameters;

namespace application {
namespace translators {

// Startup partition weights from the sign of a vertex scalar field of the
// mesh file (the initial level set of a free surface): cells whose vertex
// values all lie below the isovalue weigh `negative`, all above `positive`,
// otherwise `cut`.  ParMETIS then balances the cell count and the weighted
// work per rank.  Empty `vertex_field` disables the weighting.
struct MeshPartitionWeights {
  std::string vertex_field{};
  double isovalue{0.0};
  double negative{1.0};
  double positive{1.0};
  double cut{1.0};
};

class MeshTranslator {
public:
  // `minimum_ghost_layers` applies only when the deck leaves <Ghost_layers>
  // unset; an explicit value is always honored.
  static std::shared_ptr<svmp::Mesh> loadMesh(const MeshParameters& params,
                                              int minimum_ghost_layers = 0,
                                              const MeshPartitionWeights& partition_weights = {});

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
