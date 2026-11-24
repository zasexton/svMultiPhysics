from .conftest import run_with_reference, cpp_exec
import os
import re
import shutil
import subprocess

# Common folder for all tests in this file
base_folder = "fluid"

# Fields to test
fields = ["Velocity", "Pressure", "Traction", "WSS", "Vorticity", "Divergence"]


def test_pipe_RCR_3d(n_proc):
    test_folder = "pipe_RCR_3d"
    t_max = 2
    run_with_reference(base_folder, test_folder, fields, n_proc, t_max)

def test_pipe_RCR_3d_fourier_coeff(n_proc):
    test_folder = "pipe_RCR_3d_fourier_coeff"
    t_max = 2
    run_with_reference(base_folder, test_folder, fields, n_proc, t_max)

def test_pipe_RCR_3d_petsc(n_proc):
    test_folder = "pipe_RCR_3d_petsc"
    t_max = 2
    run_with_reference(base_folder, test_folder, fields, n_proc, t_max)

def test_pipe_RCR_3d_trilinos_ilut(n_proc):
    test_folder = "pipe_RCR_3d_trilinos_ilut"
    t_max = 2
    run_with_reference(base_folder, test_folder, fields, n_proc, t_max)

def test_pipe_RCR_3d_trilinos_bj(n_proc):
    test_folder = "pipe_RCR_3d_trilinos_bj"
    t_max = 2
    run_with_reference(base_folder, test_folder, fields, n_proc, t_max)

def test_pipe_RCR_weak_dir_3d(n_proc):
    test_folder = "pipe_RCR_weak_dir_3d"
    t_max = 2
    run_with_reference(base_folder, test_folder, fields, n_proc, t_max)

def test_pipe_RCR_weak_dir_3d_new_solver(n_proc, tmp_path):
    # The new solver on the weak-wall deck (Penalty_parameter as its Nitsche
    # gamma) converges at every step. Its results are not compared with the
    # legacy reference result_002.vtu here.
    case = tmp_path / "pipe_RCR_weak_dir_3d"
    shutil.copytree(os.path.join("cases", base_folder, "pipe_RCR_weak_dir_3d"), case,
                    ignore=shutil.ignore_patterns("*-procs"))
    cmd = ["mpirun"] + (["--oversubscribe"] if n_proc > 1 else []) + ["-np", str(n_proc), cpp_exec, "solver_new.xml"]
    run = subprocess.run(cmd, cwd=case, capture_output=True, text=True)
    assert run.returncode == 0, run.stdout[-4000:] + run.stderr[-4000:]
    steps = re.findall(r"nonlinear_done step=(\d+) time=\S+ converged=(\d)", run.stdout)
    assert [int(step) for step, _ in steps] == [0, 1], run.stdout[-4000:]
    assert all(converged == "1" for _, converged in steps), run.stdout[-4000:]

def test_pipe_RCR_genBC(n_proc):
    test_folder = "pipe_RCR_genBC"
    t_max = 2

    # Remove old genBC output
    os.chdir(os.path.join("cases", base_folder, test_folder))
    for name in ["AllData", "InitialData", "GenBC.int"]:
        if os.path.isfile(name):
            os.remove(name)

    # Compile genBC
    os.chdir("genBC")
    subprocess.run(["make", "clean"], check=True)
    subprocess.run(["make"], check=True)

    # Change back to original directory
    os.chdir("../../../..")

    run_with_reference(base_folder, test_folder, fields, n_proc, t_max)

def test_pipe_RCR_sv0D(n_proc):
    test_folder = "pipe_RCR_sv0D"
    t_max = 2
    run_with_reference(base_folder, test_folder, fields, n_proc, t_max)


def test_driven_cavity_2d(n_proc):
    test_folder = "driven_cavity_2d"
    t_max = 2
    run_with_reference(base_folder, test_folder, fields, n_proc, t_max)


def test_driven_cavity_2d_porous(n_proc):
    test_folder = "driven_cavity_2d_porous"
    t_max = 2
    run_with_reference(base_folder, test_folder, fields, n_proc, t_max)


def test_dye_AD(n_proc):
    test_folder = "dye_AD"
    run_with_reference(base_folder, test_folder, fields, n_proc)

def test_precomputed_dye_AD(n_proc):
    test_folder = "precomputed_dye_AD"
    run_with_reference(base_folder, test_folder, ['Velocity', 'Concentration'], n_proc)

def test_newtonian(n_proc):
    test_folder = "newtonian"
    run_with_reference(base_folder, test_folder, fields, n_proc)


def test_casson(n_proc):
    test_folder = "casson"
    run_with_reference(base_folder, test_folder, fields, n_proc)


def test_carreau_yasuda(n_proc):
    test_folder = "carreau_yasuda"
    run_with_reference(base_folder, test_folder, fields, n_proc)


def test_iliac_artery(n_proc):
    test_folder = "iliac_artery"
    run_with_reference(base_folder, test_folder, fields, n_proc)

def test_iliac_artery_trilinos_gmres_ilut(n_proc):
    test_folder = "iliac_artery_trilinos_gmres_ilut"
    run_with_reference(base_folder, test_folder, fields, n_proc)

def test_quadratic_tet10(n_proc):
    test_folder = "quadratic_tet10"
    t_max = 3
    fields = ["Velocity", "Pressure", "Vorticity", "Divergence"]
    run_with_reference(base_folder, test_folder, fields, n_proc, t_max) 
