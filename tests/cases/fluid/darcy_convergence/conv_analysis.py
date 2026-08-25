import pyvista as pv
import numpy as np
import matplotlib.pyplot as plt


# Analytical Solutions:
def exact_pressure(x, y, z):
    r = np.sqrt(x ** 2 + y ** 2)
    return 1.0 - (np.log(r) / np.log(2.0))


def exact_velocity(x, y, z):
    K = 1e-11  # Permeability
    mu = 1.0  # Viscosity

    r = np.sqrt(x ** 2 + y ** 2)

    # Apply the Darcy material scale to the pressure gradient
    vel_r = (K / mu) * (1.0 / (r * np.log(2.0)))

    theta = np.arctan2(y, x)
    u_x = vel_r * np.cos(theta)
    u_y = vel_r * np.sin(theta)
    u_z = np.zeros_like(r)
    return np.column_stack((u_x, u_y, u_z))



# Compute L2 Norms:
def compute_l2_errors(result_file, pressure_array="MBF", velocity_array="MBF_flux"):
    mesh = pv.read(result_file)
    mesh = mesh.compute_cell_sizes()
    volumes = np.abs(mesh.cell_data["Volume"])

    centers = mesh.cell_centers().points
    x, y, z = centers[:, 0], centers[:, 1], centers[:, 2]

    if pressure_array in mesh.point_data:
        mesh = mesh.ptc()

    num_p = mesh.cell_data[pressure_array]
    num_v = mesh.cell_data[velocity_array]

    ana_p = exact_pressure(x, y, z)
    ana_v = exact_velocity(x, y, z)

    p_err_sq = (num_p - ana_p) ** 2
    L2_p = np.sqrt(np.sum(p_err_sq * volumes))

    v_err_sq = np.sum((num_v - ana_v) ** 2, axis=1)
    L2_v = np.sqrt(np.sum(v_err_sq * volumes))

    total_volume = np.sum(volumes)
    h = (total_volume / mesh.n_cells) ** (1 / 3.0)

    return h, L2_p, L2_v


if __name__ == "__main__":
    base_dir = "."
    meshes = [
        f"{base_dir}/coarse-mesh/16-procs/result_020.vtu",
        f"{base_dir}/med-mesh/16-procs/result_020.vtu",
        f"{base_dir}/fine-mesh/16-procs/result_020.vtu"
    ]

    h_vals, p_errs, v_errs = [], [], []

    print(f"{'Mesh Level':<15} | {'h (Elem Size)':<15} | {'L2 Pressure':<15} | {'L2 Velocity':<15}")
    print("-" * 65)

    for mesh_file in meshes:
        mesh_name = mesh_file.split('/')[1]
        h, L2_p, L2_v = compute_l2_errors(mesh_file)
        h_vals.append(h)
        p_errs.append(L2_p)
        v_errs.append(L2_v)
        print(f"{mesh_name:<15} | {h:<15.6e} | {L2_p:<15.6e} | {L2_v:<15.6e}")


    # Plot the Convergence Rates
    h_vals = np.array(h_vals)
    p_errs = np.array(p_errs)
    v_errs = np.array(v_errs)

    # Create theoretical reference lines starting from the coarse mesh error
    p_ref = p_errs[0] * (h_vals / h_vals[0]) ** 2  # O(h^2) slope
    v_ref = v_errs[0] * (h_vals / h_vals[0]) ** 1  # O(h^1) slope

    plt.figure(figsize=(9, 7))

    # Plot actual calculated errors
    plt.loglog(h_vals, p_errs, 'o-', linewidth=2, markersize=8, label='Pressure $L_2$ Error', color='blue')
    plt.loglog(h_vals, v_errs, 's-', linewidth=2, markersize=8, label='Velocity $L_2$ Error', color='red')

    # Plot reference slopes (dashed)
    plt.loglog(h_vals, p_ref, '--', color='lightblue', label='Expected $O(h^2)$')
    plt.loglog(h_vals, v_ref, '--', color='lightcoral', label='Expected $O(h)$')

    plt.xlabel('Element Size ($h$)', fontsize=12)
    plt.ylabel('$L_2$ Norm Error', fontsize=12)
    plt.title('Mesh Convergence: Darcy Flow', fontsize=14)
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.legend(fontsize=11)

    # Invert X-axis so finer meshes (smaller h) are on the right
    plt.gca().invert_xaxis()

    plt.tight_layout()
    plt.show()
