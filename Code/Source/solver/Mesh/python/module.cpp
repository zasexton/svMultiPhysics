// Main Python module that organizes submodules mirroring the Mesh folder

#include <pybind11/pybind11.h>

#include "Core/bind_core.h"
#include "Topology/bind_topology.h"
#include "Geometry/bind_geometry.h"
#include "Fields/bind_fields.h"
#include "Labels/bind_labels.h"
#include "Search/bind_search.h"
#include "Boundary/bind_boundary.h"
#include "IO/bind_io.h"

namespace py = pybind11;

PYBIND11_MODULE(svmesh_py, m) {
  m.doc() = "Python bindings for svMultiPhysics mesh (svmesh) with submodules";

  auto core = m.def_submodule("core", "Core mesh bindings");
  svmp_pybind::bind_core(core);

  auto topology = m.def_submodule("topology", "Topology utilities");
  svmp_pybind::bind_topology(topology);

  auto geometry = m.def_submodule("geometry", "Geometry utilities");
  svmp_pybind::bind_geometry(geometry);

  auto fields = m.def_submodule("fields", "Field helpers");
  svmp_pybind::bind_fields(fields);

  auto labels = m.def_submodule("labels", "Label helpers");
  svmp_pybind::bind_labels(labels);

  auto search = m.def_submodule("search", "Search utilities");
  svmp_pybind::bind_search(search);

  auto boundary = m.def_submodule("boundary", "Boundary utilities");
  svmp_pybind::bind_boundary(boundary);

  auto io = m.def_submodule("io", "I/O helpers");
  svmp_pybind::bind_io(io);
}
