from .conftest import add_test_boundary_face, make_interior_node_case, run_by_name, run_with_reference
import copy
import numpy as np
import meshio
import pytest
import xml.etree.ElementTree as ET
import os
import subprocess

# Common folder for all tests in this file
base_folder = "linear-elasticity"

# Fields to test
fields = [
    "Displacement",
    "Jacobian",
    "Strain",
    "Stress", 
    "VonMises_stress",
]

def test_beam(n_proc):
    test_folder = "beam"
    t_max = 1
    run_with_reference(base_folder, test_folder, fields, n_proc, t_max)


def make_interior_structural_case(tmp_path, physics):
    root, points, cells, ids = make_interior_node_case(tmp_path, 3)
    root.find("GeneralSimulationParameters/Spectral_radius_of_infinite_time_step").text = "0.5"
    equation = root.find("Add_equation")
    equation.set("type", physics)
    for child in list(equation):
        if child.tag.startswith("Darcy_") or child.tag in ("Fluid_density", "Source_term"):
            equation.remove(child)
    for tag, value in [("Density", "1"), ("Elasticity_modulus", "10"),
                       ("Poisson_ratio", "0.3"), ("Force_y", "1")]:
        ET.SubElement(equation, tag).text = value
    if physics == "ustruct":
        ET.SubElement(equation, "Constitutive_model", type="nHK")
    equation.find("Max_iterations").text = "20"
    equation.find("LS").set("type", "GMRES")
    equation.find("LS/Linear_algebra/Preconditioner").text = "fsils"
    ET.SubElement(equation.find("LS"), "Absolute_tolerance").text = "1e-13"
    equation.find("Output").clear()
    equation.find("Output").set("type", "Spatial")
    for field in ["Displacement", "Velocity"]:
        ET.SubElement(equation.find("Output"), field).text = "true"
    return root, points, cells, ids


@pytest.mark.parametrize("physics", ["lElas", "ustruct"])
@pytest.mark.parametrize("integral", [False, True])
@pytest.mark.parametrize("prescription", ["components", "all", "general", "history"])
def test_interior_node_components(tmp_path, n_proc, physics, integral, prescription):
    root, points, _, ids = make_interior_structural_case(tmp_path, physics)
    slope = 0.01 if prescription == "history" else 0
    equation = root.find("Add_equation")
    bc = equation.find("Add_BC")
    bc.remove(bc.find("Temporal_and_spatial_values_file_path"))
    bc.find("Time_dependence").text = "Steady"
    ET.SubElement(bc, "Value").text = "0.02"
    ET.SubElement(bc, "Effective_direction").text = "1 0 0"
    ET.SubElement(bc, "Impose_on_state_variable_integral").text = str(integral).lower()
    # A second prescription on the same nodes constrains a disjoint component.
    other = copy.deepcopy(bc)
    other.set("name", "z_component")
    other.find("Effective_direction").text = "0 0 1"
    other.find("Value").text = "0.03"
    if prescription == "components":
        equation.append(other)
        expected = np.array([[0.02, 0.03]] * 2)
        components = [0, 2]
    else:
        bc.remove(bc.find("Effective_direction"))
        components = [0, 1, 2]
        expected = np.full((2, 3), 0.02)
        if prescription in ("general", "history"):
            bc.remove(bc.find("Value"))
            bc.find("Time_dependence").text = "General"
            ET.SubElement(bc, "Temporal_and_spatial_values_file_path").text = "vector.dat"
            expected = np.array([[0.01, 0.02, 0.03], [0.04, 0.05, 0.06]])
            (tmp_path / "vector.dat").write_text(
                "3 2 2\n0 1\n" + "".join(
                    f"{ids[i]} " + " ".join(map(str, np.r_[expected[i], expected[i] + slope])) + "\n"
                    for i in [1, 0]))
            expected += slope * 0.2
    ET.ElementTree(root).write(tmp_path / "solver.xml")
    result = run_by_name(tmp_path, "solver.xml", 2, n_proc)
    order = [np.argmin(np.linalg.norm(result.points - points[i-1], axis=1)) for i in ids]
    for step in [0, 1, 2]:
        sample = meshio.read(tmp_path / f"{n_proc}-procs/result_{step:03d}.vtu")
        selected = [np.argmin(np.linalg.norm(sample.points - points[i-1], axis=1)) for i in ids]
        constrained = sample.point_data["Displacement" if integral else "Velocity"][selected]
        np.testing.assert_allclose(constrained[:, components], expected + slope * (step * 0.1 - 0.2), atol=1e-10, rtol=0)
        if integral:
            np.testing.assert_allclose(sample.point_data["Velocity"][selected][:, components], slope, atol=1e-10)
    # A mask that accidentally fixes every component suppresses this response.
    if prescription == "components":
        assert np.all(result.point_data["Displacement"][order, 1] > 1e-5)
    assert np.isfinite(result.point_data["Displacement"]).all()


@pytest.mark.parametrize("face_first", [False, True])
@pytest.mark.parametrize("slope", [0, 0.01])
@pytest.mark.parametrize("components", [1, 3])
def test_interior_node_structural_overlap(tmp_path, n_proc, face_first, slope, components):
    root, points, cells, ids = make_interior_structural_case(tmp_path, "ustruct")
    surface = add_test_boundary_face(root, tmp_path, points, cells)
    ids[0] = surface[0]
    root.find("Add_mesh/Add_node_set/Node_IDs").text = " ".join(map(str, ids))
    equation = root.find("Add_equation")
    node_bc = equation.find("Add_BC")
    if components == 1:
        ET.SubElement(node_bc, "Effective_direction").text = "1 0 0"
    face_bc = copy.deepcopy(node_bc)
    face_bc.set("name", "surface")
    face_bc.remove(face_bc.find("Mesh_name"))
    face_bc.remove(face_bc.find("Node_set"))
    face_bc.find("Temporal_and_spatial_values_file_path").text = "face-values.dat"
    for name, selected in [("values.dat", ids), ("face-values.dat", surface)]:
        history = " ".join(map(str, [0.02] * components + [0.02 + slope] * components))
        (tmp_path / name).write_text(f"{components} 2 {len(selected)}\n0 1\n" + "".join(
            f"{node} {history}\n" for node in selected))
    equation.remove(node_bc)
    equation.extend([face_bc, node_bc] if face_first else [node_bc, face_bc])
    ET.ElementTree(root).write(tmp_path / "solver.xml")
    run_by_name(tmp_path, "solver.xml", 2, n_proc)
    for step in [0, 1, 2]:
        sample = meshio.read(tmp_path / f"{n_proc}-procs/result_{step:03d}.vtu")
        selected = [np.argmin(np.linalg.norm(sample.points - points[i-1], axis=1)) for i in ids]
        time = step * 0.1
        np.testing.assert_allclose(sample.point_data["Velocity"][selected, :components], 0.02 + slope*time, atol=1e-10)
        # Generalized-alpha integration with gamma=2/3 (spectral radius 0.5),
        # starting from zero displacement even when the initial velocity is nonzero.
        displacement = 0.02*time + slope*(0.5*time**2 + time*0.1/6)
        np.testing.assert_allclose(sample.point_data["Displacement"][selected, :components], displacement, atol=1e-10)


def test_interior_node_with_coupled_face(tmp_path, n_proc):
    root, points, cells, ids = make_interior_node_case(tmp_path, 3)
    add_test_boundary_face(root, tmp_path, points, cells)
    equation = root.find("Add_equation")
    equation.set("type", "fluid")
    for child in list(equation):
        if child.tag.startswith("Darcy_") or child.tag in ("Fluid_density", "Source_term"):
            equation.remove(child)
    ET.SubElement(equation, "Density").text = "1"
    ET.SubElement(ET.SubElement(equation, "Viscosity", model="Constant"), "Value").text = "1"
    equation.find("LS").set("type", "GMRES")
    equation.find("LS/Linear_algebra/Preconditioner").text = "fsils"
    equation.find("Output/Darcy_pressure").tag = "Velocity"
    bc = equation.find("Add_BC")
    bc.remove(bc.find("Temporal_and_spatial_values_file_path"))
    bc.find("Time_dependence").text = "Steady"
    ET.SubElement(bc, "Value").text = "0.02"
    outlet = ET.SubElement(equation, "Add_BC", name="surface")
    ET.SubElement(outlet, "Type").text = "Neumann"
    ET.SubElement(outlet, "Time_dependence").text = "RCR"
    rcr = ET.SubElement(outlet, "RCR_values")
    for tag, value in [("Capacitance", "1"), ("Proximal_resistance", "1"),
                       ("Distal_resistance", "1"), ("Distal_pressure", "0"), ("Initial_pressure", "0")]:
        ET.SubElement(rcr, tag).text = value
    ET.ElementTree(root).write(tmp_path / "solver.xml")
    result = run_by_name(tmp_path, "solver.xml", 2, n_proc)
    order = [np.argmin(np.linalg.norm(result.points - points[i-1], axis=1)) for i in ids]
    np.testing.assert_allclose(result.point_data["Velocity"][order], 0.02, atol=1e-10)
    assert np.isfinite(result.point_data["Velocity"]).all()
