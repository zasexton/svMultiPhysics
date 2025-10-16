// IO bindings: VTK helpers and registry

#include "bind_io.h"

#include <pybind11/pybind11.h>
#include <string>

#include "../../Core/MeshBase.h"

#ifdef MESH_HAS_VTK
#include "../../IO/VTKReader.h"
#include "../../IO/VTKWriter.h"
#endif

namespace py = pybind11;
using namespace svmp;

namespace svmp_pybind {

void bind_io(py::module_& m) {
#ifdef MESH_HAS_VTK
  // Ensure VTK is registered with MeshBase on import of io submodule
  VTKReader::register_with_mesh();
  VTKWriter::register_with_mesh();

  m.def("read_vtk", [](const std::string& filename) {
    MeshIOOptions opts; opts.path = filename; opts.format = "vtk";
    return MeshBase::load(opts);
  });
  m.def("read_vtu", [](const std::string& filename) {
    MeshIOOptions opts; opts.path = filename; opts.format = "vtu";
    return MeshBase::load(opts);
  });
  m.def("read_vtp", [](const std::string& filename) {
    MeshIOOptions opts; opts.path = filename; opts.format = "vtp";
    return MeshBase::load(opts);
  });

  m.def("write_vtk", [](const MeshBase& mesh, const std::string& filename, bool binary, bool compress) {
    MeshIOOptions opts; opts.path = filename; opts.format = "vtk";
    opts.kv["binary"] = binary ? "true" : "false";
    opts.kv["compress"] = compress ? "true" : "false";
    mesh.save(opts);
  }, py::arg("mesh"), py::arg("filename"), py::arg("binary") = false, py::arg("compress") = false);

  m.def("write_vtu", [](const MeshBase& mesh, const std::string& filename, bool binary, bool compress) {
    MeshIOOptions opts; opts.path = filename; opts.format = "vtu";
    opts.kv["binary"] = binary ? "true" : "false";
    opts.kv["compress"] = compress ? "true" : "false";
    mesh.save(opts);
  }, py::arg("mesh"), py::arg("filename"), py::arg("binary") = false, py::arg("compress") = true);

  m.def("write_vtp", [](const MeshBase& mesh, const std::string& filename, bool binary, bool compress) {
    MeshIOOptions opts; opts.path = filename; opts.format = "vtp";
    opts.kv["binary"] = binary ? "true" : "false";
    opts.kv["compress"] = compress ? "true" : "false";
    mesh.save(opts);
  }, py::arg("mesh"), py::arg("filename"), py::arg("binary") = false, py::arg("compress") = true);

  m.attr("has_vtk") = true;
#else
  m.attr("has_vtk") = false;
#endif
}

} // namespace svmp_pybind

