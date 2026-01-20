#include "Application/Core/ApplicationDriver.h"

#include "Application/Core/SimulationBuilder.h"

#include "Mesh/Core/MeshBase.h"
#include "Parameters.h"
#include "tinyxml2.h"

#include <algorithm>
#include <cctype>
#include <iostream>

namespace {

std::string trim_copy(std::string s)
{
  auto not_space = [](unsigned char ch) { return !std::isspace(ch); };
  s.erase(s.begin(), std::find_if(s.begin(), s.end(), not_space));
  s.erase(std::find_if(s.rbegin(), s.rend(), not_space).base(), s.end());
  return s;
}

std::string lower_copy(std::string s)
{
  std::transform(s.begin(), s.end(), s.begin(),
                 [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
  return s;
}

bool parse_bool_relaxed(const std::string& raw)
{
  const auto v = lower_copy(trim_copy(raw));
  if (v == "true" || v == "1" || v == "yes" || v == "on") {
    return true;
  }
  if (v == "false" || v == "0" || v == "no" || v == "off") {
    return false;
  }
  return false;
}

} // namespace

namespace application {
namespace core {

bool ApplicationDriver::shouldUseNewSolver(const std::string& xml_file)
{
  tinyxml2::XMLDocument doc;
  if (doc.LoadFile(xml_file.c_str()) != tinyxml2::XML_SUCCESS) {
    return false;
  }

  auto* root = doc.FirstChildElement(Parameters::FSI_FILE.c_str());
  if (!root) {
    return false;
  }

  auto* general = root->FirstChildElement("GeneralSimulationParameters");
  if (!general) {
    return false;
  }

  auto* flag_elem = general->FirstChildElement("Use_new_OOP_solver");
  if (!flag_elem || !flag_elem->GetText()) {
    return false;
  }

  return parse_bool_relaxed(flag_elem->GetText());
}

void ApplicationDriver::run(const std::string& xml_file)
{
  Parameters params;
  params.read_xml(xml_file);
  runWithParameters(params);
}

void ApplicationDriver::runWithParameters(const Parameters& params)
{
  std::cout << "[svMultiPhysics::Application] <Use_new_OOP_solver>=true; running new OOP solver path."
            << std::endl;
  std::cout << "[svMultiPhysics::Application] Supported (initial): equation types heatS/heatF (Poisson), "
               "single <Add_mesh>, steady BCs (Dir/Neu/Robin) with constant <Value> on named <Add_face>."
            << std::endl;
  std::cout << "[svMultiPhysics::Application] Not supported yet: Domain_file_path, multiple domains, "
               "spatial/temporal BC files, transient solves, fluid/FSI/etc. "
               "Set <Use_new_OOP_solver>false</Use_new_OOP_solver> to use the legacy solver."
            << std::endl;

  SimulationBuilder builder(params);
  auto sim = builder.build();

  if (sim.primary_mesh) {
    std::cout << "[svMultiPhysics::Application] Loaded mesh '" << sim.primary_mesh_name << "': "
              << sim.primary_mesh->n_vertices() << " vertices, " << sim.primary_mesh->n_cells()
              << " cells, " << sim.primary_mesh->n_faces() << " faces." << std::endl;
  } else {
    std::cout << "[svMultiPhysics::Application] No meshes were loaded from <Add_mesh>." << std::endl;
  }

  const int num_steps = params.general_simulation_parameters.number_of_time_steps.value();
  if (num_steps <= 1) {
    runSteadyState(sim, params);
  } else {
    runTransient(sim, params);
  }
}

void ApplicationDriver::runSteadyState(SimulationComponents& /*sim*/, const Parameters& /*params*/)
{
  std::cout << "[svMultiPhysics::Application] Steady-state solve is not implemented yet (requires Phases 5+ and 8+)."
            << std::endl;
}

void ApplicationDriver::runTransient(SimulationComponents& /*sim*/, const Parameters& /*params*/)
{
  std::cout << "[svMultiPhysics::Application] Transient solve is not implemented yet (requires Phases 5+ and 11+)."
            << std::endl;
}

void ApplicationDriver::outputResults(const SimulationComponents& /*sim*/, const Parameters& /*params*/,
                                      int /*step*/, double /*time*/)
{
}

} // namespace core
} // namespace application
