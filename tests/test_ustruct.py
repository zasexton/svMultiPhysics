import os
import pytest
import subprocess
import xml.etree.ElementTree as ET

from .conftest import make_interior_node_case, run_by_name, run_with_reference, skip_if_no_trilinos

# Common folder for all tests in this file
base_folder = "ustruct"

# Fields to test
fields = [
    "Displacement",
    "Velocity",
    "Pressure",
    "VonMises_stress",
    "Cauchy_stress",
    "Strain",
    "Jacobian",
]


@pytest.mark.parametrize("ele", ["P1P1_VMS"])
def test_block_compression(ele, n_proc):
    test_folder = os.path.join("block_compression", ele)
    run_with_reference(base_folder, test_folder, fields, n_proc)


def test_tensile_adventitia_HGO(n_proc):
    test_folder = "tensile_adventitia_HGO"
    run_with_reference(base_folder, test_folder, fields, n_proc)


def test_LV_Guccione_active(n_proc):
    test_folder = "LV_Guccione_active"
    run_with_reference(base_folder, test_folder, fields, n_proc)

def test_LV_HolzapfelOgden_active(n_proc):
    test_folder = "LV_HolzapfelOgden_active"
    run_with_reference(base_folder, test_folder, fields, n_proc)

def test_LV_HolzapfelOgden_passive(n_proc):
    test_folder = "LV_HolzapfelOgden_passive"
    run_with_reference(base_folder, test_folder, fields, n_proc)

def test_LV_HolzapfelOgdenModifiedAnisotropy_passive(n_proc):
    test_folder = "LV_HolzapfelOgdenModifiedAnisotropy_passive"
    run_with_reference(base_folder, test_folder, fields, n_proc)

@skip_if_no_trilinos
def test_LV_HolzapfelOgdenModifiedAnisotropy_passive_trilinos(n_proc):
    test_folder = "LV_HolzapfelOgdenModifiedAnisotropy_passive_trilinos"
    run_with_reference(base_folder, test_folder, fields, n_proc)

def test_LV_NeoHookean_passive_genBC(n_proc):
    test_folder = "LV_NeoHookean_passive_genBC"

     # Remove old genBC output
    os.chdir(os.path.join("cases", base_folder, test_folder))
    for name in ["AllData", "InitialData", "GenBC.int"]:
        if os.path.isfile(name):
            os.remove(name)

    # Compile genBC
    os.chdir("genBC_svMultiPhysics")
    subprocess.run(["make", "clean"], check=True)
    subprocess.run(["make"], check=True)

    # Change back to original directory
    os.chdir("../../../../")

    run_with_reference(base_folder, test_folder, fields, n_proc, t_max=3)

def test_LV_NeoHookean_passive_sv0D(n_proc):
    test_folder = "LV_NeoHookean_passive_sv0D"

    run_with_reference(base_folder, test_folder, fields, n_proc, t_max=3)

def test_spatially_variable_robin(n_proc):
    test_folder = "spatially_variable_robin"
    run_with_reference(base_folder, test_folder, fields, n_proc, t_max=2)

def test_tensile_adventitia_Newtonian_viscosity(n_proc):
    test_folder = "tensile_adventitia_Newtonian_viscosity"
    run_with_reference(base_folder, test_folder, fields, n_proc, t_max=1)

def test_tensile_adventitia_Potential_viscosity(n_proc):
    test_folder = "tensile_adventitia_Potential_viscosity"
    run_with_reference(base_folder, test_folder, fields, n_proc, t_max=1)

def test_interior_node_set_rejected(tmp_path):
    root, _, _, _ = make_interior_node_case(tmp_path, 3)
    equation = root.find("Add_equation")
    equation.set("type", "ustruct")
    for child in list(equation):
        if child.tag.startswith("Darcy_") or child.tag in ("Fluid_density", "Source_term"):
            equation.remove(child)
    for tag, value in [("Density", "1"), ("Elasticity_modulus", "10"), ("Poisson_ratio", "0.3")]:
        ET.SubElement(equation, tag).text = value
    ET.SubElement(equation, "Constitutive_model", type="nHK")
    equation.find("Output/Darcy_pressure").tag = "Displacement"
    ET.ElementTree(root).write(tmp_path / "solver.xml")
    run_by_name(tmp_path, "solver.xml", 2, expected_error="node set.*ustruct")
