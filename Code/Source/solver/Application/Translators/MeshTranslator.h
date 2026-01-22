#pragma once

#include <memory>
#include <string>
#include <vector>

#include "Mesh/Mesh.h"

class MeshParameters;
class FaceParameters;

namespace application {
namespace translators {

class MeshTranslator {
public:
  static std::shared_ptr<svmp::Mesh> loadMesh(const MeshParameters& params);

private:
  static std::string detectFormat(const std::string& file_path);

  static void applyFaceLabels(svmp::Mesh& mesh,
                              const std::vector<FaceParameters*>& face_params);

  static void applyDomainLabels(svmp::Mesh& mesh, const MeshParameters& params);
};

} // namespace translators
} // namespace application
