#pragma once

#include <string>

class Parameters;

namespace application {
namespace core {

struct SimulationComponents;

class ApplicationDriver {
public:
  static bool shouldUseNewSolver(const std::string& xml_file);
  static void run(const std::string& xml_file);

private:
  static void runWithParameters(const Parameters& params);
  static void runSteadyState(SimulationComponents& sim, const Parameters& params);
  static void runTransient(SimulationComponents& sim, const Parameters& params);
  static void outputResults(const SimulationComponents& sim, const Parameters& params, int step,
                            double time);
};

} // namespace core
} // namespace application

