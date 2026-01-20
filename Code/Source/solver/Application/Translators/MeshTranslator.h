#pragma once

#include <memory>
#include <string>
#include <vector>

class MeshParameters;
class FaceParameters;

namespace svmp {
class MeshBase;
}

namespace application {
namespace translators {

class MeshTranslator {
public:
  static std::shared_ptr<svmp::MeshBase> loadMesh(const MeshParameters& params);

private:
  static std::string detectFormat(const std::string& file_path);

  static void applyFaceLabels(svmp::MeshBase& mesh,
                              const std::vector<FaceParameters*>& face_params);

  static void applyDomainLabels(svmp::MeshBase& mesh, const MeshParameters& params);
};

} // namespace translators
} // namespace application

