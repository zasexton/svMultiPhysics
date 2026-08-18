#!/usr/bin/env python3
"""Generate OOP Ustruct affine analytical patch benchmark XML files.

The generated cases intentionally live beside this script so reviewers can
inspect or run individual solver XMLs directly. Re-run this script after
editing CASES or material constants.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from textwrap import indent


THIS_DIR = Path(__file__).resolve().parent

YOUNGS_MODULUS = 1.0e6
POISSON_RATIO = 0.45
DENSITY = 1.0
SHEAR_MODULUS = 0.5 * YOUNGS_MODULUS / (1.0 + POISSON_RATIO)
BULK_MODULUS = YOUNGS_MODULUS / (3.0 * (1.0 - 2.0 * POISSON_RATIO))

THREE_D_FACES = [
    ("X0", "mesh/cube-mesh-complete/mesh-surfaces/X0.vtp"),
    ("X1", "mesh/cube-mesh-complete/mesh-surfaces/X1.vtp"),
    ("Y0", "mesh/cube-mesh-complete/mesh-surfaces/Y0.vtp"),
    ("Y1", "mesh/cube-mesh-complete/mesh-surfaces/Y1.vtp"),
    ("Z0", "mesh/cube-mesh-complete/mesh-surfaces/Z0.vtp"),
    ("Z1", "mesh/cube-mesh-complete/mesh-surfaces/Z1.vtp"),
]

SINGLE_HEX_FACES = [
    ("X0", "mesh/single-hex/mesh-surfaces/X0.vtp"),
    ("X1", "mesh/single-hex/mesh-surfaces/X1.vtp"),
    ("Y0", "mesh/single-hex/mesh-surfaces/Y0.vtp"),
    ("Y1", "mesh/single-hex/mesh-surfaces/Y1.vtp"),
    ("Z0", "mesh/single-hex/mesh-surfaces/Z0.vtp"),
    ("Z1", "mesh/single-hex/mesh-surfaces/Z1.vtp"),
]

TWO_D_FACES = [
    ("left", "../Square/mesh/mesh-surfaces/left.vtp"),
    ("right", "../Square/mesh/mesh-surfaces/right.vtp"),
    ("bottom", "../Square/mesh/mesh-surfaces/bottom.vtp"),
    ("top", "../Square/mesh/mesh-surfaces/top.vtp"),
]

TWO_D_SMALL_FACES = [
    ("left", "mesh/quad-2x2/mesh-surfaces/left.vtp"),
    ("right", "mesh/quad-2x2/mesh-surfaces/right.vtp"),
    ("bottom", "mesh/quad-2x2/mesh-surfaces/bottom.vtp"),
    ("top", "mesh/quad-2x2/mesh-surfaces/top.vtp"),
]


def pressure_for_model(model: str, jacobian: float) -> float:
    model_lc = model.lower()
    if model_lc == "st91":
        return 0.5 * BULK_MODULUS * (1.0 / jacobian - jacobian)
    if model_lc == "quadratic":
        return BULK_MODULUS * (1.0 - jacobian)
    if model_lc == "m94":
        return BULK_MODULUS * (1.0 / jacobian - 1.0)
    if model_lc == "none":
        return 0.0
    raise ValueError(f"unsupported volumetric model {model!r}")


def matmul(a: list[list[float]], b: list[list[float]]) -> list[list[float]]:
    n = len(a)
    return [
        [sum(a[i][k] * b[k][j] for k in range(n)) for j in range(n)]
        for i in range(n)
    ]


def transpose(a: list[list[float]]) -> list[list[float]]:
    n = len(a)
    return [[a[j][i] for j in range(n)] for i in range(n)]


def identity(dim: int) -> list[list[float]]:
    return [[1.0 if i == j else 0.0 for j in range(dim)] for i in range(dim)]


def det(a: list[list[float]]) -> float:
    if len(a) == 2:
        return a[0][0] * a[1][1] - a[0][1] * a[1][0]
    return (
        a[0][0] * (a[1][1] * a[2][2] - a[1][2] * a[2][1])
        - a[0][1] * (a[1][0] * a[2][2] - a[1][2] * a[2][0])
        + a[0][2] * (a[1][0] * a[2][1] - a[1][1] * a[2][0])
    )


def inv(a: list[list[float]]) -> list[list[float]]:
    d = det(a)
    if abs(d) < 1.0e-30:
        raise ValueError("singular matrix")
    if len(a) == 2:
        return [
            [a[1][1] / d, -a[0][1] / d],
            [-a[1][0] / d, a[0][0] / d],
        ]
    return [
        [
            (a[1][1] * a[2][2] - a[1][2] * a[2][1]) / d,
            (a[0][2] * a[2][1] - a[0][1] * a[2][2]) / d,
            (a[0][1] * a[1][2] - a[0][2] * a[1][1]) / d,
        ],
        [
            (a[1][2] * a[2][0] - a[1][0] * a[2][2]) / d,
            (a[0][0] * a[2][2] - a[0][2] * a[2][0]) / d,
            (a[0][2] * a[1][0] - a[0][0] * a[1][2]) / d,
        ],
        [
            (a[1][0] * a[2][1] - a[1][1] * a[2][0]) / d,
            (a[0][1] * a[2][0] - a[0][0] * a[2][1]) / d,
            (a[0][0] * a[1][1] - a[0][1] * a[1][0]) / d,
        ],
    ]


def sub(a: list[list[float]], b: list[list[float]]) -> list[list[float]]:
    return [[a[i][j] - b[i][j] for j in range(len(a))] for i in range(len(a))]


def scale(a: list[list[float]], factor: float) -> list[list[float]]:
    return [[factor * a[i][j] for j in range(len(a))] for i in range(len(a))]


def trace(a: list[list[float]]) -> float:
    return sum(a[i][i] for i in range(len(a)))


def double_contraction(a: list[list[float]], b: list[list[float]]) -> float:
    n = len(a)
    return sum(a[i][j] * b[i][j] for i in range(n) for j in range(n))


def isochoric_neohookean_pk1(f: list[list[float]]) -> list[list[float]]:
    dim = len(f)
    c = matmul(transpose(f), f)
    c_inv = inv(c)
    jacobian = det(f)
    tr_c = trace(c)
    j_m2d = jacobian ** (-2.0 / float(dim))
    s_iso = [
        [
            SHEAR_MODULUS
            * j_m2d
            * ((1.0 if i == j else 0.0) - (tr_c / float(dim)) * c_inv[i][j])
            for j in range(dim)
        ]
        for i in range(dim)
    ]
    return matmul(f, s_iso)


def total_pk1(f: list[list[float]], pressure: float) -> list[list[float]]:
    pdev = isochoric_neohookean_pk1(f)
    dim = len(f)
    jacobian = det(f)
    finv_t = transpose(inv(f))
    return [
        [
            pdev[i][j] - pressure * jacobian * finv_t[i][j]
            for j in range(dim)
        ]
        for i in range(dim)
    ]


def second_piola(f: list[list[float]], pressure: float) -> list[list[float]]:
    return matmul(inv(f), total_pk1(f, pressure))


def cauchy_stress(f: list[list[float]], pressure: float) -> list[list[float]]:
    return scale(matmul(total_pk1(f, pressure), transpose(f)), 1.0 / det(f))


def green_lagrange_strain(f: list[list[float]]) -> list[list[float]]:
    return scale(sub(matmul(transpose(f), f), identity(len(f))), 0.5)


def symmetric_voigt(a: list[list[float]]) -> list[float]:
    if len(a) == 2:
        return [a[0][0], a[1][1], a[0][1]]
    return [a[0][0], a[1][1], a[2][2], a[0][1], a[1][2], a[2][0]]


def flatten_matrix(a: list[list[float]]) -> list[float]:
    return [value for row in a for value in row]


def von_mises_stress(sigma: list[list[float]]) -> float:
    dim = len(sigma)
    mean = trace(sigma) / float(dim)
    dev = [
        [sigma[i][j] - (mean if i == j else 0.0) for j in range(dim)]
        for i in range(dim)
    ]
    return math.sqrt(1.5 * double_contraction(dev, dev))


def fmt(value: float) -> str:
    if abs(value) < 5.0e-16:
        value = 0.0
    return f"{value:.16g}"


def direction_text(direction: tuple[int, ...]) -> str:
    return "(" + ", ".join(str(v) for v in direction) + ")"


def displacement_bc(face: str, value: float, direction: tuple[int, ...]) -> str:
    return f"""    <Add_BC name="{face}">
      <Type>Dir</Type>
      <Time_dependence>Steady</Time_dependence>
      <Value>{fmt(value)}</Value>
      <Effective_direction>{direction_text(direction)}</Effective_direction>
      <Impose_on_state_variable_integral>true</Impose_on_state_variable_integral>
    </Add_BC>"""


def traction_bc(face: str, value: float, direction: tuple[int, ...]) -> str:
    return f"""    <Add_BC name="{face}">
      <Type>Neu</Type>
      <Time_dependence>Steady</Time_dependence>
      <Value>{fmt(value)}</Value>
      <Effective_direction>{direction_text(direction)}</Effective_direction>
    </Add_BC>"""


def pressure_dirichlet_bc(face: str, value: float) -> str:
    return f"""    <Add_BC name="{face}">
      <Type>pressure_dirichlet</Type>
      <Time_dependence>Steady</Time_dependence>
      <Value>{fmt(value)}</Value>
    </Add_BC>"""


def unit_directions(dim: int) -> list[tuple[int, ...]]:
    return [
        tuple(1 if i == j else 0 for i in range(dim))
        for j in range(dim)
    ]


def displacement_vector_bcs(face: str, values: list[float]) -> list[str]:
    directions = unit_directions(len(values))
    return [displacement_bc(face, values[i], directions[i]) for i in range(len(values))]


def traction_vector_bcs(face: str, values: list[float]) -> list[str]:
    directions = unit_directions(len(values))
    return [
        traction_bc(face, values[i], directions[i])
        for i in range(len(values))
        if abs(values[i]) > 1.0e-14
    ]


def diagonal_displacement_bcs(f: list[list[float]], faces: dict[str, str]) -> list[str]:
    if len(f) == 2:
        return [
            displacement_bc(faces["x0"], 0.0, (1, 0)),
            displacement_bc(faces["x1"], f[0][0] - 1.0, (1, 0)),
            displacement_bc(faces["y0"], 0.0, (0, 1)),
            displacement_bc(faces["y1"], f[1][1] - 1.0, (0, 1)),
        ]
    return [
        displacement_bc("X0", 0.0, (1, 0, 0)),
        displacement_bc("X1", f[0][0] - 1.0, (1, 0, 0)),
        displacement_bc("Y0", 0.0, (0, 1, 0)),
        displacement_bc("Y1", f[1][1] - 1.0, (0, 1, 0)),
        displacement_bc("Z0", 0.0, (0, 0, 1)),
        displacement_bc("Z1", f[2][2] - 1.0, (0, 0, 1)),
    ]


def default_faces(dim: int) -> dict[str, str]:
    if dim == 2:
        return {"x0": "left", "x1": "right", "y0": "bottom", "y1": "top"}
    return {"x0": "X0", "x1": "X1", "y0": "Y0", "y1": "Y1", "z0": "Z0", "z1": "Z1"}


def xml_text(case: dict[str, object]) -> str:
    bcs = "\n\n".join(case["boundary_conditions"])  # type: ignore[index]
    model = case["volumetric_model"]
    prefix = case["prefix"]
    newton_tolerance = fmt(float(case.get("newton_tolerance", 1.0e-10)))
    newton_max_iterations = int(case.get("newton_max_iterations", 8))
    linear_solver_type = str(case.get("linear_solver_type", "GMRES"))
    linear_tolerance = fmt(float(case.get("linear_tolerance", 1.0e-12)))
    linear_absolute_tolerance = fmt(float(case.get("linear_absolute_tolerance", 1.0e-10)))
    linear_max_iterations = int(case.get("linear_max_iterations", 250))
    dim = int(case.get("dim", 3))
    mesh_file_path = str(case.get("mesh_file_path", "mesh/cube-mesh-complete/mesh-complete.mesh.vtu"))
    face_entries = case.get("face_entries", THREE_D_FACES)
    face_xml = "\n\n".join(
        f"""    <Add_face name="{name}">
      <Face_file_path>{path}</Face_file_path>
    </Add_face>"""
        for name, path in face_entries  # type: ignore[union-attr]
    )
    return f"""<?xml version="1.0" encoding="UTF-8" ?>
<svMultiPhysicsFile version="0.1">

  <GeneralSimulationParameters>
    <Use_new_OOP_solver>true</Use_new_OOP_solver>
    <Continue_previous_simulation>false</Continue_previous_simulation>
    <Number_of_spatial_dimensions>{dim}</Number_of_spatial_dimensions>
    <Number_of_time_steps>0</Number_of_time_steps>
    <Time_step_size>0.01</Time_step_size>
    <Spectral_radius_of_infinite_time_step>0.50</Spectral_radius_of_infinite_time_step>
    <Searched_file_name_to_trigger_stop>STOP_SIM</Searched_file_name_to_trigger_stop>

    <Save_results_to_VTK_format>true</Save_results_to_VTK_format>
    <Name_prefix_of_saved_VTK_files>{prefix}</Name_prefix_of_saved_VTK_files>
    <Increment_in_saving_VTK_files>1</Increment_in_saving_VTK_files>
    <Start_saving_after_time_step>1</Start_saving_after_time_step>

    <Increment_in_saving_restart_files>100</Increment_in_saving_restart_files>
    <Convert_BIN_to_VTK_format>false</Convert_BIN_to_VTK_format>

    <Verbose>false</Verbose>
    <Warning>false</Warning>
    <Debug>false</Debug>
  </GeneralSimulationParameters>

  <Add_mesh name="msh">
    <Mesh_file_path>{mesh_file_path}</Mesh_file_path>

{face_xml}

    <Mesh_scale_factor>1.0</Mesh_scale_factor>
  </Add_mesh>

  <Add_equation type="ustruct">
    <Coupled>true</Coupled>
    <Module_options>quasi_static=true</Module_options>
    <Min_iterations>1</Min_iterations>
    <Max_iterations>{newton_max_iterations}</Max_iterations>
    <Tolerance>{newton_tolerance}</Tolerance>

    <Constitutive_model type="nHK"></Constitutive_model>
    <Density>{fmt(DENSITY)}</Density>
    <Elasticity_modulus>{fmt(YOUNGS_MODULUS)}</Elasticity_modulus>
    <Poisson_ratio>{fmt(POISSON_RATIO)}</Poisson_ratio>
    <Dilational_penalty_model>{model}</Dilational_penalty_model>

    <Momentum_stabilization_coefficient>1e-3</Momentum_stabilization_coefficient>
    <Continuity_stabilization_coefficient>1e-3</Continuity_stabilization_coefficient>

    <Output type="Spatial">
      <Divergence>true</Divergence>
      <Pressure>true</Pressure>
      <Displacement>true</Displacement>
      <Velocity>true</Velocity>
      <Def_grad>true</Def_grad>
      <Jacobian>true</Jacobian>
      <Stress>true</Stress>
      <Strain>true</Strain>
      <Cauchy_stress>true</Cauchy_stress>
      <VonMises_stress>true</VonMises_stress>
    </Output>

    <Output type="Volume_integral">
      <Pressure>true</Pressure>
    </Output>

    <LS type="{linear_solver_type}">
      <Linear_algebra type="fsils">
        <Preconditioner>row-column-scaling</Preconditioner>
      </Linear_algebra>
      <Tolerance>{linear_tolerance}</Tolerance>
      <Absolute_tolerance>{linear_absolute_tolerance}</Absolute_tolerance>
      <Max_iterations>{linear_max_iterations}</Max_iterations>
      <Krylov_space_dimension>250</Krylov_space_dimension>
    </LS>

{bcs}
  </Add_equation>

</svMultiPhysicsFile>
"""


def case_exact_fields(f: list[list[float]], pressure: float) -> dict[str, object]:
    strain = green_lagrange_strain(f)
    stress = second_piola(f, pressure)
    cauchy = cauchy_stress(f, pressure)
    return {
        "Def_grad": flatten_matrix(f),
        "Jacobian": [det(f)],
        "Divergence": [0.0],
        "Strain": symmetric_voigt(strain),
        "Stress": symmetric_voigt(stress),
        "Cauchy_stress": symmetric_voigt(cauchy),
        "VonMises_stress": [von_mises_stress(cauchy)],
    }


def diagonal_case(
    case_id: str,
    description: str,
    stretch: tuple[float, ...],
    model: str,
    *,
    dim: int = 3,
    mesh_file_path: str = "mesh/cube-mesh-complete/mesh-complete.mesh.vtu",
    face_entries: list[tuple[str, str]] = THREE_D_FACES,
    faces: dict[str, str] | None = None,
    newton_max_iterations: int = 8,
    linear_solver_type: str = "GMRES",
    linear_tolerance: float = 1.0e-12,
    linear_max_iterations: int = 250,
) -> dict[str, object]:
    f = [
        [stretch[i] if i == j else 0.0 for j in range(dim)]
        for i in range(dim)
    ]
    jacobian = det(f)
    pressure = pressure_for_model(model, jacobian)
    face_names = faces or default_faces(dim)
    boundary_conditions = diagonal_displacement_bcs(f, face_names)
    boundary_conditions.append(pressure_dirichlet_bc(face_names["x0"], pressure))
    return {
        "id": case_id,
        "description": description,
        "xml": f"{case_id}_oop.xml",
        "prefix": case_id,
        "dim": dim,
        "mesh_file_path": mesh_file_path,
        "face_entries": face_entries,
        "volumetric_model": model,
        "newton_max_iterations": newton_max_iterations,
        "linear_solver_type": linear_solver_type,
        "linear_tolerance": linear_tolerance,
        "linear_max_iterations": linear_max_iterations,
        "deformation_gradient": f,
        "jacobian": jacobian,
        "pressure": pressure,
        "exact_fields": case_exact_fields(f, pressure),
        "boundary_conditions": boundary_conditions,
    }


def shear_case() -> dict[str, object]:
    gamma = 0.001
    f = [
        [1.0, gamma, 0.0],
        [0.0, 1.0 - gamma, 0.0],
        [0.0, 0.0, 1.0],
    ]
    jacobian = det(f)
    pressure = pressure_for_model("ST91", jacobian)
    pk1 = total_pk1(f, pressure)
    x_traction = [pk1[i][0] for i in range(3)]
    z_traction = [pk1[i][2] for i in range(3)]
    boundary_conditions = [
        displacement_bc("Y0", 0.0, (1, 1, 1)),
        displacement_bc("Y1", gamma, (1, -1, 0)),
        *traction_vector_bcs("X0", [-v for v in x_traction]),
        *traction_vector_bcs("X1", x_traction),
        *traction_vector_bcs("Z0", [-v for v in z_traction]),
        *traction_vector_bcs("Z1", z_traction),
        pressure_dirichlet_bc("X0", pressure),
    ]
    return {
        "id": "ustruct_simple_shear",
        "description": "Affine shear-compression patch with exact natural-face tractions",
        "xml": "ustruct_simple_shear_oop.xml",
        "prefix": "ustruct_simple_shear",
        "dim": 3,
        "mesh_file_path": "mesh/cube-mesh-complete/mesh-complete.mesh.vtu",
        "face_entries": THREE_D_FACES,
        "volumetric_model": "ST91",
        "deformation_gradient": f,
        "jacobian": jacobian,
        "pressure": pressure,
        "exact_fields": case_exact_fields(f, pressure),
        "gamma": gamma,
        "pk1": pk1,
        "newton_tolerance": 1.0e-6,
        "linear_absolute_tolerance": 1.0e-5,
        "boundary_conditions": boundary_conditions,
    }


def mixed_volumetric_deviatoric_case() -> dict[str, object]:
    case = diagonal_case(
        "ustruct_mixed_volumetric_deviatoric_st91",
        "Mixed volumetric and deviatoric affine stretch, ST91 pressure law",
        (1.04, 0.99, 0.97),
        "ST91",
    )
    case["newton_max_iterations"] = 12
    return case


def rigid_rotation_case() -> dict[str, object]:
    return diagonal_case(
        "ustruct_rigid_rotation_z_180",
        "Rigid 180 degree rotation about z on a single Hex8 element",
        (-1.0, -1.0, 1.0),
        "ST91",
        mesh_file_path="mesh/single-hex/mesh-complete.mesh.vtu",
        face_entries=SINGLE_HEX_FACES,
    )


def two_dimensional_volumetric_case() -> dict[str, object]:
    return diagonal_case(
        "ustruct_2d_volumetric_dilation_st91",
        "2D uniform volumetric dilation, ST91 pressure law",
        (1.02, 1.02),
        "ST91",
        dim=2,
        mesh_file_path="../Square/mesh/mesh-complete.mesh.vtu",
        face_entries=TWO_D_FACES,
        newton_max_iterations=14,
        linear_solver_type="BICGS",
        linear_tolerance=1.0e-8,
        linear_max_iterations=1000,
    )


def two_dimensional_shear_case() -> dict[str, object]:
    gamma = 0.01
    f = [
        [1.0, gamma],
        [0.0, 1.0],
    ]
    jacobian = det(f)
    pressure = pressure_for_model("ST91", jacobian)
    pk1 = total_pk1(f, pressure)
    x_traction = [pk1[i][0] for i in range(2)]
    boundary_conditions = [
        *displacement_vector_bcs("bottom", [0.0, 0.0]),
        *displacement_vector_bcs("top", [gamma, 0.0]),
        *traction_vector_bcs("left", [-v for v in x_traction]),
        *traction_vector_bcs("right", x_traction),
        pressure_dirichlet_bc("left", pressure),
    ]
    return {
        "id": "ustruct_2d_isochoric_shear",
        "description": "2D isochoric affine shear patch with exact side tractions",
        "xml": "ustruct_2d_isochoric_shear_oop.xml",
        "prefix": "ustruct_2d_isochoric_shear",
        "dim": 2,
        "mesh_file_path": "mesh/quad-2x2/mesh-complete.mesh.vtu",
        "face_entries": TWO_D_SMALL_FACES,
        "volumetric_model": "ST91",
        "newton_max_iterations": 14,
        "linear_solver_type": "BICGS",
        "linear_tolerance": 1.0e-8,
        "linear_max_iterations": 1000,
        "deformation_gradient": f,
        "jacobian": jacobian,
        "pressure": pressure,
        "exact_fields": case_exact_fields(f, pressure),
        "gamma": gamma,
        "pk1": pk1,
        "boundary_conditions": boundary_conditions,
    }


def build_cases() -> list[dict[str, object]]:
    cases: list[dict[str, object]] = []
    for model in ("ST91", "quadratic", "M94"):
        model_id = model.lower()
        cases.append(
            diagonal_case(
                f"ustruct_volumetric_dilation_{model_id}",
                f"Uniform volumetric dilation, {model} pressure law",
                (1.02, 1.02, 1.02),
                model,
            )
        )
        cases.append(
            diagonal_case(
                f"ustruct_volumetric_compression_{model_id}",
                f"Uniform volumetric compression, {model} pressure law",
                (0.98, 0.98, 0.98),
                model,
            )
        )

    lambda_z_tension = 1.02
    lambda_xy_tension = lambda_z_tension ** -0.5
    cases.append(
        diagonal_case(
            "ustruct_isochoric_tension_z",
            "Isochoric affine axial tension with lateral compression",
            (lambda_xy_tension, lambda_xy_tension, lambda_z_tension),
            "ST91",
        )
    )

    lambda_z_compression = 0.98
    lambda_xy_compression = lambda_z_compression ** -0.5
    cases.append(
        diagonal_case(
            "ustruct_isochoric_compression_z",
            "Isochoric affine axial compression with lateral expansion",
            (lambda_xy_compression, lambda_xy_compression, lambda_z_compression),
            "ST91",
        )
    )

    cases.append(shear_case())
    cases.append(mixed_volumetric_deviatoric_case())
    cases.append(rigid_rotation_case())
    cases.append(two_dimensional_volumetric_case())
    cases.append(two_dimensional_shear_case())
    return cases


def manifest_case(case: dict[str, object]) -> dict[str, object]:
    exact_fields = case["exact_fields"]
    return {
        "description": case["description"],
        "xml": case["xml"],
        "prefix": case["prefix"],
        "dim": case["dim"],
        "volumetric_model": case["volumetric_model"],
        "youngs_modulus": YOUNGS_MODULUS,
        "poisson_ratio": POISSON_RATIO,
        "density": DENSITY,
        "shear_modulus": SHEAR_MODULUS,
        "bulk_modulus": BULK_MODULUS,
        "deformation_gradient": case["deformation_gradient"],
        "jacobian": case["jacobian"],
        "pressure": case["pressure"],
        "exact_fields": exact_fields,
        "default_tolerances": {
            "Displacement": {"linf_abs": 1.0e-7},
            "Velocity": {"linf_abs": 1.0e-9},
            "Pressure": {"linf_abs": 1.0},
            "Jacobian": {"linf_abs": 1.0e-9},
            "Def_grad": {"linf_abs": 1.0e-8},
            "Divergence": {"linf_abs": 1.0e-8},
            "Strain": {"linf_abs": 1.0e-8},
            "Stress": {"linf_abs": 1.0},
            "Cauchy_stress": {"linf_abs": 1.0},
            "VonMises_stress": {"linf_abs": 1.0},
        },
    }


def main() -> int:
    cases = build_cases()
    manifest = {
        "notes": "Generated by generate_ustruct_affine_patch_cases.py.",
        "cases": {},
    }
    for case in cases:
        xml_path = THIS_DIR / str(case["xml"])
        xml_path.write_text(xml_text(case), encoding="utf-8")
        manifest["cases"][str(case["id"])] = manifest_case(case)
        print(f"wrote {xml_path.name}")

    manifest_path = THIS_DIR / "affine_patch_cases.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {manifest_path.name}")

    print("\nGenerated cases:")
    for case in cases:
        print(indent(f"{case['id']}: {case['description']}", "  "))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
