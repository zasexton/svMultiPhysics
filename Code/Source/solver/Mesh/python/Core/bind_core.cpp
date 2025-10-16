// Core bindings: enums and Mesh class

#include "bind_core.h"

#include <pybind11/stl.h>
#include <pybind11/numpy.h>

#include "../../Core/MeshBase.h"
#include "../../Core/MeshTypes.h"
#include "../../Topology/CellShape.h"

namespace py = pybind11;
using namespace svmp;

namespace {
// Helper: flatten coordinates from numpy array (N,d) or (N*dim,)
static std::vector<real_t> flatten_coords(const py::array& arr, int spatial_dim) {
  if (arr.ndim() == 2) {
    if (arr.shape(1) != spatial_dim) {
      throw std::runtime_error("X_ref has wrong second dimension; expected dim = " + std::to_string(spatial_dim));
    }
    std::vector<real_t> out;
    out.reserve(static_cast<size_t>(arr.shape(0) * arr.shape(1)));
    auto buf = arr.cast<py::array_t<real_t, py::array::c_style | py::array::forcecast>>();
    auto r = buf.unchecked<2>();
    for (ssize_t i = 0; i < r.shape(0); ++i) {
      for (ssize_t j = 0; j < r.shape(1); ++j) out.push_back(r(i,j));
    }
    return out;
  } else if (arr.ndim() == 1) {
    if (arr.shape(0) % spatial_dim != 0) {
      throw std::runtime_error("Flat X_ref length must be a multiple of spatial_dim");
    }
    auto buf = arr.cast<py::array_t<real_t, py::array::c_style | py::array::forcecast>>();
    std::vector<real_t> out(buf.size());
    std::memcpy(out.data(), buf.data(), buf.size() * sizeof(real_t));
    return out;
  }
  throw std::runtime_error("X_ref must be 1D or 2D numpy array");
}

// Helper: convert sequence to vector<T>
template <typename T> static std::vector<T> to_vector(const py::object& obj) {
  std::vector<T> out;
  for (auto item : obj) out.push_back(item.cast<T>());
  return out;
}
} // namespace

namespace svmp_pybind {

void bind_core(py::module_& m) {
  // Enums
  py::enum_<EntityKind>(m, "EntityKind")
      .value("Vertex", EntityKind::Vertex)
      .value("Line", EntityKind::Line)
      .value("Edge", EntityKind::Edge)
      .value("Face", EntityKind::Face)
      .value("Volume", EntityKind::Volume);

  py::enum_<CellFamily>(m, "CellFamily")
      .value("Line", CellFamily::Line)
      .value("Triangle", CellFamily::Triangle)
      .value("Quad", CellFamily::Quad)
      .value("Tetra", CellFamily::Tetra)
      .value("Hex", CellFamily::Hex)
      .value("Wedge", CellFamily::Wedge)
      .value("Pyramid", CellFamily::Pyramid)
      .value("Polygon", CellFamily::Polygon)
      .value("Polyhedron", CellFamily::Polyhedron);

  py::enum_<Configuration>(m, "Configuration")
      .value("Reference", Configuration::Reference)
      .value("Current", Configuration::Current);

  py::class_<CellShape>(m, "CellShape")
      .def(py::init<>())
      .def_readwrite("family", &CellShape::family)
      .def_readwrite("num_corners", &CellShape::num_corners)
      .def_readwrite("order", &CellShape::order)
      .def_readwrite("is_mixed_order", &CellShape::is_mixed_order)
      .def_readwrite("num_faces_hint", &CellShape::num_faces_hint)
      .def("expected_nodes", &CellShape::expected_nodes)
      .def("topo_kind", &CellShape::topo_kind);

  // MeshBase bindings
  py::class_<MeshBase>(m, "Mesh")
      .def(py::init<>())
      .def(py::init<int>())
      .def("clear", &MeshBase::clear)
      .def("finalize", &MeshBase::finalize)
      .def("dim", &MeshBase::dim)
      .def("n_vertices", &MeshBase::n_vertices)
      .def("n_cells", &MeshBase::n_cells)
      .def("n_faces", &MeshBase::n_faces)
      .def("n_edges", &MeshBase::n_edges)
      .def("bounding_box", [](const MeshBase& mesh) {
          auto bb = mesh.bounding_box();
          py::array_t<real_t> vmin(3), vmax(3);
          auto rmin = vmin.mutable_unchecked<1>();
          auto rmax = vmax.mutable_unchecked<1>();
          for (int i = 0; i < 3; ++i) { rmin(i) = bb.min[i]; rmax(i) = bb.max[i]; }
          return py::make_tuple(vmin, vmax);
      })
      .def("cell_center", &MeshBase::cell_center, py::arg("cell"), py::arg("cfg") = Configuration::Reference)
      .def("cell_measure", &MeshBase::cell_measure, py::arg("cell"), py::arg("cfg") = Configuration::Reference)
      .def("cell_vertices", [](const MeshBase& mesh, index_t c) {
          auto span = mesh.cell_vertices_span(c);
          std::vector<index_t> vertices(span.second);
          for (size_t i = 0; i < span.second; ++i) vertices[i] = span.first[i];
          return vertices;
      })
      .def("save",
           [](const MeshBase& mesh, const std::string& filename, const std::string& format, const std::map<std::string,std::string>& kv) {
             MeshIOOptions opts; opts.path = filename; opts.format = format; opts.kv = {kv.begin(), kv.end()};
             mesh.save(opts);
           },
           py::arg("filename"), py::arg("format") = std::string(), py::arg("options") = std::map<std::string,std::string>{})
      // Labels & sets
      .def("set_region_label", &MeshBase::set_region_label, py::arg("cell"), py::arg("label"))
      .def("region_label", &MeshBase::region_label, py::arg("cell"))
      .def("cells_with_label", &MeshBase::cells_with_label, py::arg("label"))
      .def("cell_region_ids", &MeshBase::cell_region_ids)
      .def("set_boundary_label", &MeshBase::set_boundary_label, py::arg("face"), py::arg("label"))
      .def("boundary_label", &MeshBase::boundary_label, py::arg("face"))
      .def("faces_with_label", &MeshBase::faces_with_label, py::arg("label"))
      .def("add_to_set", &MeshBase::add_to_set, py::arg("kind"), py::arg("name"), py::arg("id"))
      .def("get_set", &MeshBase::get_set, py::arg("kind"), py::arg("name"), py::return_value_policy::reference)
      .def("has_set", &MeshBase::has_set, py::arg("kind"), py::arg("name"))
      .def("register_label", &MeshBase::register_label, py::arg("name"), py::arg("label"))
      .def("label_name", &MeshBase::label_name, py::arg("label"))
      .def("label_from_name", &MeshBase::label_from_name, py::arg("name"))
      // Build from arrays
      .def("build_from_arrays",
           [](MeshBase& mesh,
              int spatial_dim,
              const py::array& X_ref,
              const py::object& cell2node_offsets,
              const py::object& cell2node,
              const py::object& families,
              const py::object& orders_opt,
              const py::object& num_corners_opt) {
              std::vector<real_t> X = flatten_coords(X_ref, spatial_dim);
              std::vector<offset_t> offsets = to_vector<offset_t>(cell2node_offsets);
              std::vector<index_t>  conn    = to_vector<index_t>(cell2node);
              std::vector<CellShape> shapes;
              shapes.reserve(offsets.size() ? offsets.size() - 1 : 0);
              std::vector<int> orders;
              std::vector<int> ncorner;
              if (!orders_opt.is_none()) orders = to_vector<int>(orders_opt);
              if (!num_corners_opt.is_none()) ncorner = to_vector<int>(num_corners_opt);

              size_t n_cells = (offsets.size() ? offsets.size() - 1 : 0);
              auto fam_iter = py::iter(families);
              size_t idx = 0;
              for (auto item : fam_iter) {
                if (idx >= n_cells) break;
                CellShape cs;
                std::string s = item.cast<std::string>();
                if (s == "Line" || s == "line") cs.family = CellFamily::Line;
                else if (s == "Triangle" || s == "tri") cs.family = CellFamily::Triangle;
                else if (s == "Quad" || s == "quad") cs.family = CellFamily::Quad;
                else if (s == "Tetra" || s == "tet") cs.family = CellFamily::Tetra;
                else if (s == "Hex" || s == "hex") cs.family = CellFamily::Hex;
                else if (s == "Wedge" || s == "prism" || s == "wedge") cs.family = CellFamily::Wedge;
                else if (s == "Pyramid" || s == "pyr") cs.family = CellFamily::Pyramid;
                else if (s == "Polygon" || s == "polygon") cs.family = CellFamily::Polygon;
                else if (s == "Polyhedron" || s == "polyhedron") cs.family = CellFamily::Polyhedron;
                else throw std::runtime_error("Unknown CellFamily: " + s);

                cs.order = (orders.empty() ? 1 : orders[std::min(idx, orders.size()-1)]);
                if (!ncorner.empty()) {
                  cs.num_corners = ncorner[std::min(idx, ncorner.size()-1)];
                } else {
                  auto start = static_cast<size_t>(offsets[idx]);
                  auto end   = static_cast<size_t>(offsets[idx+1]);
                  cs.num_corners = static_cast<int>(end - start);
                }
                shapes.push_back(cs);
                ++idx;
              }
              if (shapes.size() != n_cells) throw std::runtime_error("families length mismatch");
              mesh.build_from_arrays(spatial_dim, X, offsets, conn, shapes);
           },
           py::arg("spatial_dim"), py::arg("X_ref"), py::arg("cell2node_offsets"), py::arg("cell2node"),
           py::arg("families"), py::arg("orders") = py::none(), py::arg("num_corners") = py::none());
}

} // namespace svmp_pybind
