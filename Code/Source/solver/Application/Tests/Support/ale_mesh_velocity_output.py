"""Check accepted ALE rates in the VTU files of a harmonically moving square."""

import argparse
import math
from pathlib import Path
import subprocess
import tempfile
import xml.etree.ElementTree as ET

from free_surface_rank_consistency import read_vtu


def write_case(work, reference_mesh, scheme):
    # Copy the 5x5 square, seeding an intentionally stale three-component array.
    volume = (reference_mesh / 'mesh-complete.mesh.vtu').read_bytes()
    seed = '<PointData><DataArray type="Float64" Name="grid_speed" NumberOfComponents="3" format="ascii">' + '17 -21 32 ' * 25 + '</DataArray></PointData>\n'
    assert volume.count(b'<Points>') == 1 and b'NumberOfPoints="25"' in volume
    (work / 'seed.vtu').write_bytes(volume.replace(b'<Points>', seed.encode() + b'<Points>'))
    faces = '\n'.join(f'<Add_face name="{name}"><Face_file_path>{reference_mesh}/mesh-surfaces/{name}.vtp</Face_file_path></Add_face>'
                      for name in ('left', 'right', 'bottom', 'top'))
    slip = '\n'.join(f'<Add_BC name="{name}"><Type>Dir</Type><Value>0</Value><Effective_direction>{direction}</Effective_direction></Add_BC>'
                     for name, direction in [('left', '1 0'), ('right', '1 0'), ('bottom', '0 1')])
    (work / 'solver.xml').write_text(f'''<svMultiPhysicsFile version="0.1">
  <GeneralSimulationParameters>
    <Use_new_OOP_solver>true</Use_new_OOP_solver><Number_of_spatial_dimensions>2</Number_of_spatial_dimensions>
    <Continue_previous_simulation>false</Continue_previous_simulation>
    <Number_of_time_steps>4</Number_of_time_steps><Time_step_size>0.05</Time_step_size>
    <Transient_time_integration_scheme>{scheme}</Transient_time_integration_scheme>
    <Spectral_radius_of_infinite_time_step>0.5</Spectral_radius_of_infinite_time_step>
    <Generalized_alpha_initial_rate>Zero</Generalized_alpha_initial_rate>
    <Save_results_to_VTK_format>true</Save_results_to_VTK_format><Combine_time_series>true</Combine_time_series>
    <Name_prefix_of_saved_VTK_files>result</Name_prefix_of_saved_VTK_files>
    <Increment_in_saving_VTK_files>1</Increment_in_saving_VTK_files>
    <Start_saving_after_time_step>1</Start_saving_after_time_step>
    <Increment_in_saving_restart_files>4</Increment_in_saving_restart_files>
    <Searched_file_name_to_trigger_stop>STOP_SIM</Searched_file_name_to_trigger_stop>
    <Convert_BIN_to_VTK_format>0</Convert_BIN_to_VTK_format><Verbose>1</Verbose><Warning>0</Warning><Debug>0</Debug>
  </GeneralSimulationParameters>
  <Add_mesh name="square"><Mesh_file_path>seed.vtu</Mesh_file_path>{faces}</Add_mesh>
  <Add_equation type="fluid">
    <Coupled>true</Coupled><Min_iterations>1</Min_iterations><Max_iterations>12</Max_iterations>
    <Tolerance>1e-6</Tolerance><Enable_ALE>true</Enable_ALE>
    <Mesh_velocity_field>grid_speed</Mesh_velocity_field><Mesh_displacement_field>grid_map</Mesh_displacement_field>
    <Auto_register_mesh_displacement_field>true</Auto_register_mesh_displacement_field>
    <Density>1</Density><Viscosity model="Constant"><Value>0.02</Value></Viscosity>
    <Output type="Spatial"><Mesh_velocity>true</Mesh_velocity><Mesh_displacement>true</Mesh_displacement></Output>
    <LS type="Direct"><Linear_algebra type="eigen"><Preconditioner>none</Preconditioner></Linear_algebra></LS>
    {slip}<Add_BC name="top"><Type>Neu</Type><Value>0</Value></Add_BC>
  </Add_equation>
  <Add_equation type="mesh_motion"><Model>Harmonic</Model><Field_name>grid_map</Field_name>
    <Harmonic_quantity>velocity</Harmonic_quantity><Kappa>1</Kappa>{slip}
    <Add_BC name="top"><Type>Neu</Type><Value>0.2</Value><Effective_direction>0 1</Effective_direction></Add_BC>
  </Add_equation>
</svMultiPhysicsFile>''')


def check_case(work, scheme):
    steps = ET.parse(work / 'result.pvd').findall('.//DataSet')
    assert len(steps) == 4, f'{scheme} saved {len(steps)} snapshots'
    # rho_inf=1/2 gives gamma=2/3. Backward Euler uses gamma=1.
    gamma = 2. / 3. if scheme == 'GeneralizedAlpha' else 1.
    previous = [0.] * 50
    rate = [0.] * 50
    old_time = peak_speed = peak_error = peak_secant_gap = 0.
    for step, dataset in enumerate(steps, 1):
        t = float(dataset.get('timestep'))
        assert math.isclose(t, step * .05, rel_tol=1e-12)
        data = read_vtu(work / dataset.get('file'), ('grid_map', 'grid_speed'))
        displacement = [v for row in data['grid_map'] for v in row]
        velocity = [v for row in data['grid_speed'] for v in row[:2]]
        assert len(displacement) == len(velocity) == len(rate)
        secant = [(d - old) / (t - old_time) for d, old in zip(displacement, previous)]
        rate = [(s - (1. - gamma) * r) / gamma for s, r in zip(secant, rate)]
        errors = [abs(v - r) for v, r in zip(velocity, rate)]
        assert all(math.isfinite(e) for e in errors)
        peak_error = max(peak_error, *errors)
        peak_speed = max(peak_speed, *map(abs, rate))
        peak_secant_gap = max(peak_secant_gap, *(abs(s - r) for s, r in zip(secant, rate)))
        previous, old_time = displacement, t
    assert peak_speed > .1, f'{scheme} generated no prescribed motion'
    if scheme == 'GeneralizedAlpha':
        assert peak_secant_gap > .02, 'the prescribed motion did not distinguish the endpoint rate'
    print(f'{scheme}: endpoint speed {peak_speed:.9g}, max VTU error {peak_error:.3g}', flush=True)
    assert peak_error < 1e-11, f'{scheme} wrote an inconsistent mesh velocity'


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--solver', type=Path, required=True)
    parser.add_argument('--mesh', type=Path, required=True)
    args = parser.parse_args()
    for scheme in ('GeneralizedAlpha', 'BackwardEuler'):
        with tempfile.TemporaryDirectory(prefix='ale_velocity_output_') as directory:
            work = Path(directory)
            write_case(work, args.mesh.resolve(), scheme)
            run = subprocess.run([args.solver.resolve(), 'solver.xml'], cwd=work, stdout=subprocess.PIPE,
                                 stderr=subprocess.STDOUT, text=True)
            if run.returncode:
                raise RuntimeError(f'{scheme} solver failed ({run.returncode})\n{run.stdout[-5000:]}')
            check_case(work, scheme)
