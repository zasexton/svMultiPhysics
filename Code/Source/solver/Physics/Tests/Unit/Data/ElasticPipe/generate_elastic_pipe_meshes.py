#!/usr/bin/env python3
"""Generate elastic pipe FSI test meshes with PyVista and TetGen."""

from __future__ import annotations

import argparse
import json
import math
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Tuple

import numpy as np
import pyvista as pv
import tetgen


Point = Tuple[float, float, float]
Face = Tuple[int, int, int]


@dataclass(frozen=True)
class MeshVariant:
    name: str
    length: float
    inner_radius: float
    outer_radius: float
    circumferential_segments: int
    axial_segments: int
    fluid_radial_segments: int
    solid_max_volume: float


VARIANTS = (
    MeshVariant(
        name="coarse",
        length=2.0,
        inner_radius=0.40,
        outer_radius=0.52,
        circumferential_segments=16,
        axial_segments=4,
        fluid_radial_segments=4,
        solid_max_volume=0.02,
    ),
    MeshVariant(
        name="refined",
        length=2.0,
        inner_radius=0.40,
        outer_radius=0.52,
        circumferential_segments=24,
        axial_segments=8,
        fluid_radial_segments=5,
        solid_max_volume=0.008,
    ),
)


SURFACE_IDS = {
    "fluid": {
        "inlet": 1,
        "outlet": 2,
        "fsi_interface": 3,
    },
    "solid": {
        "solid_inlet": 1,
        "solid_outlet": 2,
        "fsi_interface": 3,
        "outer_wall": 4,
    },
}


COARSE_SOLVER_XML = """<?xml version="1.0" encoding="UTF-8" ?>
<svMultiPhysicsFile version="0.1">

<GeneralSimulationParameters>
  <Continue_previous_simulation> false </Continue_previous_simulation>
  <Use_new_OOP_solver> false </Use_new_OOP_solver>
  <Number_of_spatial_dimensions> 3 </Number_of_spatial_dimensions>
  <Number_of_time_steps> 2 </Number_of_time_steps>
  <Time_step_size> 1.0e-4 </Time_step_size>
  <Spectral_radius_of_infinite_time_step> 0.50 </Spectral_radius_of_infinite_time_step>
  <Searched_file_name_to_trigger_stop> STOP_SIM </Searched_file_name_to_trigger_stop>
  <Save_results_to_VTK_format> true </Save_results_to_VTK_format>
  <Name_prefix_of_saved_VTK_files> result </Name_prefix_of_saved_VTK_files>
  <Increment_in_saving_VTK_files> 1 </Increment_in_saving_VTK_files>
  <Start_saving_after_time_step> 1 </Start_saving_after_time_step>
  <Increment_in_saving_restart_files> 1 </Increment_in_saving_restart_files>
  <Convert_BIN_to_VTK_format> 0 </Convert_BIN_to_VTK_format>
  <Verbose> 1 </Verbose>
  <Warning> 0 </Warning>
  <Debug> 0 </Debug>
</GeneralSimulationParameters>

<Add_mesh name="lumen" >
  <Mesh_file_path> fluid/mesh/mesh-complete.mesh.vtu </Mesh_file_path>
  <Add_face name="lumen_inlet">
    <Face_file_path> fluid/mesh/mesh-surfaces/inlet.vtp </Face_file_path>
  </Add_face>
  <Add_face name="lumen_outlet">
    <Face_file_path> fluid/mesh/mesh-surfaces/outlet.vtp </Face_file_path>
  </Add_face>
  <Add_face name="lumen_wall">
    <Face_file_path> fluid/mesh/mesh-surfaces/fsi_interface.vtp </Face_file_path>
  </Add_face>
  <Domain> 0 </Domain>
</Add_mesh>

<Add_mesh name="wall" >
  <Mesh_file_path> solid/mesh/mesh-complete.mesh.vtu </Mesh_file_path>
  <Add_face name="wall_inlet">
    <Face_file_path> solid/mesh/mesh-surfaces/solid_inlet.vtp </Face_file_path>
  </Add_face>
  <Add_face name="wall_outlet">
    <Face_file_path> solid/mesh/mesh-surfaces/solid_outlet.vtp </Face_file_path>
  </Add_face>
  <Add_face name="wall_inner">
    <Face_file_path> solid/mesh/mesh-surfaces/fsi_interface.vtp </Face_file_path>
  </Add_face>
  <Add_face name="wall_outer">
    <Face_file_path> solid/mesh/mesh-surfaces/outer_wall.vtp </Face_file_path>
  </Add_face>
  <Domain> 1 </Domain>
</Add_mesh>

<Add_projection name="wall_inner" >
  <Project_from_face> lumen_wall </Project_from_face>
</Add_projection>

<Add_equation type="FSI" >
  <Coupled> true </Coupled>
  <Min_iterations> 1 </Min_iterations>
  <Max_iterations> 5 </Max_iterations>
  <Tolerance> 1.0e-10 </Tolerance>

  <Domain id="0" >
    <Equation> fluid </Equation>
    <Density> 1.0 </Density>
    <Viscosity model="Constant" >
      <Value> 0.04 </Value>
    </Viscosity>
    <Backflow_stabilization_coefficient> 0.2 </Backflow_stabilization_coefficient>
  </Domain>

  <Domain id="1" >
    <Equation> struct </Equation>
    <Constitutive_model type="neoHookean"> </Constitutive_model>
    <Dilational_penalty_model> M94 </Dilational_penalty_model>
    <Density> 1.0 </Density>
    <Elasticity_modulus> 1.0e7 </Elasticity_modulus>
    <Poisson_ratio> 0.3 </Poisson_ratio>
  </Domain>

  <LS type="GMRES" >
    <Linear_algebra type="fsils" >
      <Preconditioner> fsils </Preconditioner>
    </Linear_algebra>
    <Tolerance> 1.0e-10 </Tolerance>
    <Max_iterations> 100 </Max_iterations>
    <Krylov_space_dimension> 50 </Krylov_space_dimension>
  </LS>

  <Output type="Spatial" >
    <Displacement> true </Displacement>
    <Velocity> true </Velocity>
    <Pressure> true </Pressure>
    <VonMises_stress> true </VonMises_stress>
  </Output>

  <Output type="Alias" >
    <Displacement> FS_Displacement </Displacement>
  </Output>

  <Add_BC name="lumen_inlet" >
    <Type> Neu </Type>
    <Value> 1.0e3 </Value>
  </Add_BC>

  <Add_BC name="lumen_outlet" >
    <Type> Neu </Type>
    <Value> 0.0 </Value>
  </Add_BC>

  <Add_BC name="wall_inlet" >
    <Type> Dir </Type>
    <Value> 0.0 </Value>
    <Impose_on_state_variable_integral> true </Impose_on_state_variable_integral>
    <Zero_out_perimeter> false </Zero_out_perimeter>
    <Effective_direction> (0, 0, 1) </Effective_direction>
  </Add_BC>

  <Add_BC name="wall_outlet" >
    <Type> Dir </Type>
    <Value> 0.0 </Value>
    <Impose_on_state_variable_integral> true </Impose_on_state_variable_integral>
    <Zero_out_perimeter> false </Zero_out_perimeter>
    <Effective_direction> (0, 0, 1) </Effective_direction>
  </Add_BC>
</Add_equation>

<Add_equation type="mesh" >
  <Coupled> true </Coupled>
  <Min_iterations> 1 </Min_iterations>
  <Max_iterations> 5 </Max_iterations>
  <Tolerance> 1.0e-10 </Tolerance>
  <Poisson_ratio> 0.3 </Poisson_ratio>

  <LS type="CG" >
    <Linear_algebra type="fsils" >
      <Preconditioner> fsils </Preconditioner>
    </Linear_algebra>
    <Tolerance> 1.0e-10 </Tolerance>
    <Max_iterations> 100 </Max_iterations>
  </LS>

  <Output type="Spatial" >
    <Displacement> true </Displacement>
  </Output>

  <Add_BC name="lumen_inlet" >
    <Type> Dir </Type>
    <Value> 0.0 </Value>
  </Add_BC>

  <Add_BC name="lumen_outlet" >
    <Type> Dir </Type>
    <Value> 0.0 </Value>
  </Add_BC>
</Add_equation>

</svMultiPhysicsFile>
"""


def add_point(points: List[Point], point: Iterable[float]) -> int:
    points.append(tuple(float(x) for x in point))
    return len(points) - 1


def ring(points: List[Point], radius: float, z: float, segments: int) -> List[int]:
    ids: List[int] = []
    for i in range(segments):
        theta = 2.0 * math.pi * float(i) / float(segments)
        ids.append(add_point(points, (radius * math.cos(theta),
                                      radius * math.sin(theta),
                                      z)))
    return ids


def polydata_from_faces(points: np.ndarray, faces: np.ndarray) -> pv.PolyData:
    cells = np.column_stack(
        [np.full(faces.shape[0], 3, dtype=np.int64), faces.astype(np.int64)]
    )
    return pv.PolyData(points, cells.ravel())


def build_fluid_mesh(
    variant: MeshVariant,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, List[Face]]]:
    points: List[Point] = []
    tets: List[Tuple[int, int, int, int]] = []
    faces: Dict[str, List[Face]] = {
        "inlet": [],
        "outlet": [],
        "fsi_interface": [],
    }

    z_values = np.linspace(0.0, variant.length, variant.axial_segments + 1)
    centerline: List[int] = []
    rings: List[List[List[int]]] = []
    for z in z_values:
        centerline.append(add_point(points, (0.0, 0.0, float(z))))
        radial_rings: List[List[int]] = []
        for ir in range(1, variant.fluid_radial_segments + 1):
            radius = (
                variant.inner_radius * float(ir) /
                float(variant.fluid_radial_segments)
            )
            radial_rings.append(
                ring(points, radius, float(z), variant.circumferential_segments)
            )
        rings.append(radial_rings)

    def node(iz: int, ir: int, itheta: int) -> int:
        if ir == 0:
            return centerline[iz]
        return rings[iz][ir - 1][itheta % variant.circumferential_segments]

    def cross_section_triangles(iz: int) -> List[Face]:
        triangles: List[Face] = []
        for i in range(variant.circumferential_segments):
            j = (i + 1) % variant.circumferential_segments
            triangles.append((node(iz, 0, 0), node(iz, 1, i), node(iz, 1, j)))
            for ir in range(1, variant.fluid_radial_segments):
                inner_i = node(iz, ir, i)
                inner_j = node(iz, ir, j)
                outer_i = node(iz, ir + 1, i)
                outer_j = node(iz, ir + 1, j)
                triangles.append((inner_i, outer_i, outer_j))
                triangles.append((inner_i, outer_j, inner_j))
        return triangles

    section_triangles = [cross_section_triangles(iz)
                         for iz in range(variant.axial_segments + 1)]

    for tri in section_triangles[0]:
        faces["inlet"].append((tri[0], tri[2], tri[1]))
    faces["outlet"].extend(section_triangles[-1])

    for iz in range(variant.axial_segments):
        lower_outer = rings[iz][-1]
        upper_outer = rings[iz + 1][-1]
        for i in range(variant.circumferential_segments):
            j = (i + 1) % variant.circumferential_segments
            a, b, c, d = (
                lower_outer[i],
                lower_outer[j],
                upper_outer[j],
                upper_outer[i],
            )
            faces["fsi_interface"].append((a, b, c))
            faces["fsi_interface"].append((a, c, d))

        for lower_tri, upper_tri in zip(section_triangles[iz],
                                        section_triangles[iz + 1]):
            a, b, c = lower_tri
            top_a, top_b, top_c = upper_tri
            tets.append((a, b, c, top_a))
            tets.append((b, top_b, top_c, top_a))
            tets.append((b, top_c, c, top_a))

    return (
        np.asarray(points, dtype=np.float64),
        np.asarray(tets, dtype=np.int32),
        faces,
    )


def build_solid_surface(variant: MeshVariant) -> Tuple[np.ndarray, Dict[str, List[Face]]]:
    points: List[Point] = []
    faces: Dict[str, List[Face]] = {
        "solid_inlet": [],
        "solid_outlet": [],
        "fsi_interface": [],
        "outer_wall": [],
    }

    z_values = np.linspace(0.0, variant.length, variant.axial_segments + 1)
    inner_rings: List[List[int]] = []
    outer_rings: List[List[int]] = []
    for z in z_values:
        inner_rings.append(
            ring(points, variant.inner_radius, float(z), variant.circumferential_segments)
        )
        outer_rings.append(
            ring(points, variant.outer_radius, float(z), variant.circumferential_segments)
        )

    for iz in range(variant.axial_segments):
        inner_lower = inner_rings[iz]
        inner_upper = inner_rings[iz + 1]
        outer_lower = outer_rings[iz]
        outer_upper = outer_rings[iz + 1]
        for i in range(variant.circumferential_segments):
            j = (i + 1) % variant.circumferential_segments

            ia, ib, ic, id_ = inner_lower[i], inner_lower[j], inner_upper[j], inner_upper[i]
            faces["fsi_interface"].append((ia, ic, ib))
            faces["fsi_interface"].append((ia, id_, ic))

            oa, ob, oc, od = outer_lower[i], outer_lower[j], outer_upper[j], outer_upper[i]
            faces["outer_wall"].append((oa, ob, oc))
            faces["outer_wall"].append((oa, oc, od))

    for i in range(variant.circumferential_segments):
        j = (i + 1) % variant.circumferential_segments

        ia, ib = inner_rings[0][i], inner_rings[0][j]
        oa, ob = outer_rings[0][i], outer_rings[0][j]
        faces["solid_inlet"].append((ia, ib, ob))
        faces["solid_inlet"].append((ia, ob, oa))

        ia, ib = inner_rings[-1][i], inner_rings[-1][j]
        oa, ob = outer_rings[-1][i], outer_rings[-1][j]
        faces["solid_outlet"].append((ia, ob, ib))
        faces["solid_outlet"].append((ia, oa, ob))

    return np.asarray(points, dtype=np.float64), faces


def all_faces(surface_faces: Mapping[str, List[Face]]) -> np.ndarray:
    return np.asarray(
        [face for faces in surface_faces.values() for face in faces],
        dtype=np.int32,
    )


def tetrahedralize(points: np.ndarray,
                   faces: np.ndarray,
                   max_volume: float) -> pv.UnstructuredGrid:
    surface = polydata_from_faces(points, faces)
    if surface.n_open_edges != 0:
        raise RuntimeError(f"surface is not closed; open edges={surface.n_open_edges}")

    generator = tetgen.TetGen(points, faces)
    nodes, elements, _, _ = generator.tetrahedralize(
        plc=True,
        quality=True,
        nobisect=True,
        fixedvolume=True,
        maxvolume=max_volume,
        minratio=1.5,
        mindihedral=5.0,
        quiet=True,
    )

    return grid_from_tetrahedra(nodes, elements)


def grid_from_tetrahedra(points: np.ndarray,
                         elements: np.ndarray) -> pv.UnstructuredGrid:
    cells = np.column_stack(
        [np.full(elements.shape[0], 4, dtype=np.int64), elements.astype(np.int64)]
    )
    cell_types = np.full(elements.shape[0], pv.CellType.TETRA, dtype=np.uint8)
    grid = pv.UnstructuredGrid(cells.ravel(), cell_types, points)
    grid.point_data["GlobalNodeID"] = np.arange(grid.n_points, dtype=np.int32)
    grid.cell_data["GlobalElementID"] = np.arange(grid.n_cells, dtype=np.int32)
    return grid


def write_surface(path: Path,
                  points: np.ndarray,
                  faces: np.ndarray,
                  surface_id: int) -> None:
    used_ids = sorted({int(index) for face in faces for index in face})
    local_id = {global_id: i for i, global_id in enumerate(used_ids)}
    local_points = points[np.asarray(used_ids, dtype=np.int64)]
    local_faces = np.asarray(
        [[local_id[int(index)] for index in face] for face in faces],
        dtype=np.int32,
    )
    surface = polydata_from_faces(local_points, local_faces)
    surface.point_data["GlobalNodeID"] = np.arange(surface.n_points, dtype=np.int32)
    surface.cell_data["GlobalElementID"] = np.arange(surface.n_cells, dtype=np.int32)
    surface.cell_data["ModelFaceID"] = np.full(surface.n_cells, surface_id, dtype=np.int32)
    surface.save(path, binary=True)


def write_participant_mesh(base_dir: Path,
                           participant: str,
                           points: np.ndarray,
                           surface_faces: Mapping[str, List[Face]],
                           max_volume: float) -> Dict[str, object]:
    faces = all_faces(surface_faces)
    grid = tetrahedralize(points, faces, max_volume)
    return write_participant_grid(base_dir, participant, points, surface_faces, grid)


def write_participant_grid(base_dir: Path,
                           participant: str,
                           points: np.ndarray,
                           surface_faces: Mapping[str, List[Face]],
                           grid: pv.UnstructuredGrid) -> Dict[str, object]:
    participant_dir = base_dir / participant / "mesh"
    surfaces_dir = participant_dir / "mesh-surfaces"
    surfaces_dir.mkdir(parents=True, exist_ok=True)

    grid.save(participant_dir / "mesh-complete.mesh.vtu", binary=True)

    for name, grouped_faces in surface_faces.items():
        write_surface(
            surfaces_dir / f"{name}.vtp",
            points,
            np.asarray(grouped_faces, dtype=np.int32),
            SURFACE_IDS[participant][name],
        )

    validate_surface_faces(grid, points, surface_faces)
    return {
        "points": int(grid.n_points),
        "tetrahedra": int(grid.n_cells),
        "surfaces": {name: len(grouped_faces)
                     for name, grouped_faces in surface_faces.items()},
    }


def rounded_key(point: np.ndarray, scale: float = 1.0e12) -> Tuple[int, int, int]:
    return tuple(int(round(float(x) * scale)) for x in point)


def validate_surface_faces(grid: pv.UnstructuredGrid,
                           source_points: np.ndarray,
                           surface_faces: Mapping[str, List[Face]]) -> None:
    volume_point_ids = {
        rounded_key(point): i
        for i, point in enumerate(np.asarray(grid.points))
    }

    boundary_faces: Dict[Tuple[int, int, int], int] = {}
    for tet in np.asarray(grid.cells).reshape((-1, 5))[:, 1:]:
        candidates = (
            (tet[0], tet[1], tet[2]),
            (tet[0], tet[1], tet[3]),
            (tet[0], tet[2], tet[3]),
            (tet[1], tet[2], tet[3]),
        )
        for face in candidates:
            key = tuple(sorted(int(x) for x in face))
            boundary_faces[key] = boundary_faces.get(key, 0) + 1

    exterior = {key for key, count in boundary_faces.items() if count == 1}
    for surface_name, faces in surface_faces.items():
        for face in faces:
            mapped = []
            for source_index in face:
                key = rounded_key(source_points[source_index])
                if key not in volume_point_ids:
                    raise RuntimeError(
                        f"{surface_name} references a point not retained by TetGen"
                    )
                mapped.append(volume_point_ids[key])
            if tuple(sorted(mapped)) not in exterior:
                raise RuntimeError(
                    f"{surface_name} contains a triangle that is not a volume boundary face"
                )


def normalized_surface_signature(points: np.ndarray,
                                 faces: Iterable[Face]) -> List[Tuple[Tuple[int, int, int],
                                                                     Tuple[int, int, int],
                                                                     Tuple[int, int, int]]]:
    signature = []
    for face in faces:
        coords = sorted(rounded_key(points[i]) for i in face)
        signature.append(tuple(coords))
    return sorted(signature)


def validate_shared_interface(fluid_points: np.ndarray,
                              fluid_faces: Mapping[str, List[Face]],
                              solid_points: np.ndarray,
                              solid_faces: Mapping[str, List[Face]]) -> None:
    fluid_signature = normalized_surface_signature(
        fluid_points,
        fluid_faces["fsi_interface"],
    )
    solid_signature = normalized_surface_signature(
        solid_points,
        solid_faces["fsi_interface"],
    )
    if fluid_signature != solid_signature:
        raise RuntimeError("fluid and solid interface triangulations do not match")


def write_coarse_solver_xml(variant_dir: Path) -> None:
    with (variant_dir / "solver.xml").open("w", encoding="utf-8") as handle:
        handle.write(COARSE_SOLVER_XML)


def generate_variant(root: Path, variant: MeshVariant) -> Dict[str, object]:
    variant_dir = root / variant.name
    if variant_dir.exists():
        shutil.rmtree(variant_dir)
    variant_dir.mkdir(parents=True)

    fluid_points, fluid_elements, fluid_faces = build_fluid_mesh(variant)
    solid_points, solid_faces = build_solid_surface(variant)
    validate_shared_interface(fluid_points, fluid_faces, solid_points, solid_faces)
    fluid_grid = grid_from_tetrahedra(fluid_points, fluid_elements)
    if variant.name == "coarse":
        write_coarse_solver_xml(variant_dir)

    summary = {
        "geometry": {
            "length": variant.length,
            "inner_radius": variant.inner_radius,
            "outer_radius": variant.outer_radius,
            "circumferential_segments": variant.circumferential_segments,
            "axial_segments": variant.axial_segments,
        },
        "mesh_controls": {
            "fluid_radial_segments": variant.fluid_radial_segments,
            "solid_max_volume": variant.solid_max_volume,
        },
        "participants": {
            "fluid": write_participant_grid(
                variant_dir,
                "fluid",
                fluid_points,
                fluid_faces,
                fluid_grid,
            ),
            "solid": write_participant_mesh(
                variant_dir,
                "solid",
                solid_points,
                solid_faces,
                variant.solid_max_volume,
            ),
        },
    }
    if variant.name == "coarse":
        summary["solver_input"] = "solver.xml"
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate elastic pipe FSI meshes for physics tests."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="Directory where the mesh variants will be written.",
    )
    args = parser.parse_args()

    root = args.output_dir.resolve()
    root.mkdir(parents=True, exist_ok=True)

    manifest = {
        "description": "Elastic pipe FSI test meshes generated with PyVista and TetGen.",
        "layout": {
            "fluid": "lumen volume mesh",
            "solid": "elastic wall volume mesh",
        },
        "surface_ids": SURFACE_IDS,
        "variants": {},
    }
    for variant in VARIANTS:
        manifest["variants"][variant.name] = generate_variant(root, variant)

    with (root / "manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
        handle.write("\n")


if __name__ == "__main__":
    main()
