#include "Application/Translators/FreeSurfaceInputAdapter.h"

#include "Mesh/Mesh.h"

#include <algorithm>
#include <cctype>
#include <initializer_list>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace application::translators::free_surface_input {
namespace {

std::string trim_copy(std::string s)
{
  auto not_space = [](unsigned char ch) { return !std::isspace(ch); };
  s.erase(s.begin(), std::find_if(s.begin(), s.end(), not_space));
  s.erase(std::find_if(s.rbegin(), s.rend(), not_space).base(), s.end());
  return s;
}

std::string normalized_token(std::string raw)
{
  raw = trim_copy(std::move(raw));
  std::transform(raw.begin(), raw.end(), raw.begin(),
                 [](unsigned char ch) { return static_cast<char>(std::tolower(ch)); });
  raw.erase(std::remove_if(raw.begin(), raw.end(), [](unsigned char ch) {
              return ch == '_' || ch == '-' || std::isspace(ch);
            }),
            raw.end());
  return raw;
}

std::string defined_param_token(const svmp::Physics::ParameterMap& params,
                                std::string_view key)
{
  const auto it = params.find(std::string(key));
  if (it == params.end() || !it->second.defined) {
    return {};
  }
  return normalized_token(it->second.value);
}

} // namespace

bool usesLegacyUnfittedFaceBinding(const svmp::Physics::ParameterMap& params)
{
  if (defined_param_token(params, "Type") != "freesurface") {
    return false;
  }
  const auto implementation = defined_param_token(params, "Implementation");
  return implementation == "unfitted" ||
         implementation == "unfittedlevelset" ||
         implementation == "levelset" ||
         implementation == "embeddedlevelset";
}

namespace {

const svmp::Physics::ParameterValue* first_defined_param(
    const svmp::Physics::ParameterMap& params,
    std::initializer_list<std::string_view> keys)
{
  for (const auto key : keys) {
    const auto it = params.find(std::string(key));
    if (it != params.end() && it->second.defined &&
        !trim_copy(it->second.value).empty()) {
      return &it->second;
    }
  }
  return nullptr;
}

std::vector<std::string> split_face_entries(std::string_view raw)
{
  std::vector<std::string> entries;
  std::string current;
  for (const char ch : raw) {
    if (ch == ';' || ch == ',') {
      auto entry = trim_copy(std::move(current));
      if (!entry.empty()) {
        entries.push_back(std::move(entry));
      }
      current.clear();
      continue;
    }
    current.push_back(ch);
  }
  auto entry = trim_copy(std::move(current));
  if (!entry.empty()) {
    entries.push_back(std::move(entry));
  }
  return entries;
}

} // namespace

void resolveLegacyContactWallFaces(svmp::Physics::ParameterMap& params,
                                     const svmp::Mesh& mesh)
{
  const auto* faces = first_defined_param(
      params,
      {"Contact_line_wall_faces", "ContactLineWallFaces",
       "Contact_line_wall_face", "ContactLineWallFace",
       "Wall_boundary_faces", "WallBoundaryFaces",
       "Wall_boundary_face", "WallBoundaryFace"});
  if (faces == nullptr) {
    return;
  }
  if (first_defined_param(
          params,
          {"Contact_line_wall_markers", "ContactLineWallMarkers",
           "Contact_line_wall_marker", "ContactLineWallMarker",
           "Wall_boundary_markers", "WallBoundaryMarkers",
           "Wall_boundary_marker", "WallBoundaryMarker"}) != nullptr) {
    throw std::runtime_error(
        "[svMultiPhysics::Application] Free-surface contact-line wall input must use either wall face names or wall markers, not both.");
  }

  const auto entries = split_face_entries(faces->value);
  if (entries.empty()) {
    throw std::runtime_error(
        "[svMultiPhysics::Application] Contact_line_wall_face(s) is defined but empty.");
  }

  std::ostringstream markers;
  for (std::size_t i = 0; i < entries.size(); ++i) {
    const auto label = mesh.label_from_name(entries[i]);
    if (label == svmp::INVALID_LABEL) {
      throw std::runtime_error(
          "[svMultiPhysics::Application] Contact-line wall face '" +
          entries[i] +
          "' is not registered in the mesh labels.");
    }
    if (i != 0u) {
      markers << ';';
    }
    markers << static_cast<int>(label);
  }
  params["Contact_line_wall_markers"] =
      svmp::Physics::ParameterValue{true, markers.str()};

  // Face names are an application-level convenience.  The physics module
  // intentionally accepts only resolved marker ids, so remove every face
  // alias after translating it instead of forwarding both representations.
  for (const std::string_view key : {
           "Contact_line_wall_faces", "ContactLineWallFaces",
           "Contact_line_wall_face", "ContactLineWallFace",
           "Wall_boundary_faces", "WallBoundaryFaces",
           "Wall_boundary_face", "WallBoundaryFace"}) {
    params.erase(std::string(key));
  }
}

} // namespace application::translators::free_surface_input
