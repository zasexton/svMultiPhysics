import importlib.util
import math
import tempfile
from pathlib import Path

import pytest


def _load_runner():
    repo = Path(__file__).resolve().parents[1]
    path = (
        repo / "tests/cases/fluid/open_vessel_free_surface"
        / "run_test05_velocity_growth_smoke.py"
    )
    spec = importlib.util.spec_from_file_location("fs16_physical_runner", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _load_matrix_runner():
    repo = Path(__file__).resolve().parents[1]
    path = (
        repo / "tests/cases/fluid/open_vessel_free_surface"
        / "run_fs16_physical_matrix.py"
    )
    spec = importlib.util.spec_from_file_location("fs16_physical_matrix", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _load_validation_grade_runner():
    repo = Path(__file__).resolve().parents[1]
    path = (
        repo / "tests/cases/fluid/open_vessel_free_surface"
        / "run_test05_validation_grade.py"
    )
    spec = importlib.util.spec_from_file_location(
        "fs16_validation_grade", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _load_wall_advection_runner():
    repo = Path(__file__).resolve().parents[1]
    path = (
        repo / "tests/cases/fluid/open_vessel_free_surface"
        / "run_impermeable_wall_advection_qualification.py"
    )
    spec = importlib.util.spec_from_file_location(
        "fs16_wall_advection_qualification", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_application_driver_seeds_wet_volume_before_time_loop_callbacks():
    repo = Path(__file__).resolve().parents[1]
    source = (
        repo / "Code/Source/solver/Application/Core/ApplicationDriver.cpp"
    ).read_text(encoding="utf-8")
    transient_setup = source.index(
        "opts.newton.accepted_state_sync_invalidates_residual =")
    baseline = source.index(
        "std::map<std::string, svmp::FE::Real> initial_wet_volume_by_key;",
        transient_setup,
    )
    initial_refresh = source.index(
        "refreshActiveCutIntegrationContextCached(", baseline)
    initial_projection = source.index(
        "projectLevelSetCurvatureFieldsFromState(", initial_refresh)
    initial_diagnostic = source.index(
        "logWetVolumeDiagnostics(", initial_projection)
    callbacks = source.index(
        "svmp::FE::timestepping::TimeLoopCallbacks callbacks{};",
        initial_diagnostic,
    )
    accepted_callback = source.index(
        "callbacks.on_step_accepted", callbacks)
    accepted_diagnostic = source.index(
        "logWetVolumeDiagnostics(", accepted_callback)

    assert (
        baseline < initial_refresh < initial_projection < initial_diagnostic
        < callbacks < accepted_callback < accepted_diagnostic
    )
    assert "initial_wet_volume_by_key);" in source[
        initial_diagnostic:callbacks]
    assert "initial_wet_volume_by_key);" in source[
        accepted_diagnostic:accepted_diagnostic + 600]


def test_validation_grade_requires_and_gates_false_wall_wet_history():
    runner = _load_validation_grade_runner()
    physical_runner = _load_runner()
    physical_args = physical_runner.argparse.Namespace(
        enable_physical_history_instrumentation=True)
    missing = runner.false_wall_wet_failures({})
    assert len(missing) == 1
    physical_missing = physical_runner.false_wall_wet_history_errors(
        {}, physical_args)
    assert len(physical_missing) == 2
    assert any("cell-interior stencils" in error for error in physical_missing)
    assert any("history is unavailable" in error for error in physical_missing)
    clean = runner.false_wall_wet_failures({
        "wall_only_false_wet_history": [{"step": 0, "time": 0.0, "count": 0}],
        "first_wall_only_false_wet": None,
    })
    assert clean == []
    assert physical_runner.false_wall_wet_history_errors({
        "wall_only_false_wet_history": [{"step": 0, "time": 0.0, "count": 0}],
        "first_wall_only_false_wet": None,
        "wall_inward_cell_centroid_stencil_complete": True,
    }, physical_args) == []
    closed_interface = {
        "wall_only_false_wet_applicability": (
            "not_applicable_closed_interface"),
        "wall_only_false_wet_closed_interface_certified": True,
        "wall_only_false_wet_history": [],
        "first_wall_only_false_wet": None,
    }
    assert runner.false_wall_wet_failures(closed_interface) == []
    assert physical_runner.false_wall_wet_history_errors(
        closed_interface, physical_args) == []
    uncertified = dict(closed_interface)
    uncertified["wall_only_false_wet_closed_interface_certified"] = False
    assert "without a valid" in runner.false_wall_wet_failures(
        uncertified)[0]
    assert "without a valid" in physical_runner.false_wall_wet_history_errors(
        uncertified, physical_args)[0]
    detected = runner.false_wall_wet_failures({
        "wall_only_false_wet_history": [{"step": 470, "time": 0.235, "count": 1}],
        "first_wall_only_false_wet": {
            "step": 470,
            "time": 0.235,
            "global_node_id": 17,
        },
    })
    assert len(detected) == 1
    assert "0.235" in detected[0]
    physical_detected = physical_runner.false_wall_wet_history_errors({
        "wall_only_false_wet_history": [
            {"step": 470, "time": 0.235, "count": 1}],
        "wall_inward_cell_centroid_stencil_complete": True,
        "first_wall_only_false_wet": {
            "step": 470,
            "time": 0.235,
            "wall": "wall_right",
            "global_node_id": 17,
        },
    }, physical_args)
    assert len(physical_detected) == 1
    assert "0.235" in physical_detected[0]


def test_test05_defaults_span_historical_false_wet_window():
    repo = Path(__file__).resolve().parents[1]
    case_root = (
        repo / "tests/cases/fluid/open_vessel_free_surface/unfitted_level_set"
    )
    runner = _load_runner()

    for case in (
        "spheric_test05_wet_bed_d18",
        "spheric_test05_wet_bed_d38",
    ):
        root = runner.ET.parse(case_root / case / "solver.xml").getroot()
        general = root.find("GeneralSimulationParameters")
        assert general is not None
        assert general.findtext("Number_of_time_steps") == "562"
        assert general.findtext("Time_step_size") == "0.0005"
        assert general.findtext("Increment_in_saving_restart_files") == "562"
        free_surface = runner.free_surface_bc(root)
        assert free_surface.findtext("Use_cut_metadata_scale") == "false"

    args = runner.argparse.Namespace(
        high_order_3d_benchmark_profile_qualification=True,
        high_order_mpi_motion_smoke=False,
        high_order_curved_3d_simplex_smoke=False,
        case=None,
        steps=None,
        timeout_seconds=None,
        surface_tension=None,
        min_max_speed=1.0e-2,
        min_wet_mean_speed=2.5e-4,
        min_gate_mean_ux=1.0e-4,
        min_front_mean_ux=1.0e-4,
        implicit_cut_quadrature_backend=None,
        enable_physical_history_instrumentation=False,
        enable_adaptive_time_loop=False,
        newton_line_search_fail_on_no_reduction=False,
    )
    runner.apply_high_order_3d_benchmark_profile_qualification_defaults(args)
    assert args.steps == 562
    assert args.disable_cut_metadata_scale is True
    assert args.enable_physical_history_instrumentation is True
    assert args.require_level_set_mass_correction_histories is True
    assert args.final_output_only is False


def test_ren_e_speed_sign_does_not_treat_rest_as_advancing_or_receding():
    runner = _load_runner()
    assert runner.ren_e_speed_sign_agrees(0.25, 0.5)
    assert runner.ren_e_speed_sign_agrees(-0.25, -0.5)
    assert not runner.ren_e_speed_sign_agrees(-0.25, 0.5)
    assert not runner.ren_e_speed_sign_agrees(0.0, 0.5)
    assert not runner.ren_e_speed_sign_agrees(0.5, 0.0)
    assert runner.ren_e_speed_sign_agrees(0.0, 0.0)


def test_sessile_circle_postprocessor_resolves_60_90_120_degree_states():
    runner = _load_runner()
    with tempfile.TemporaryDirectory() as temp_dir:
        for angle in (60.0, 90.0, 120.0):
            case_dir = Path(temp_dir) / f"theta_{int(angle)}"
            runner.write_sessile2d_case(
                case_dir,
                steps=1,
                nx=16,
                ny=16,
                initial_angle_degrees=angle,
                equilibrium_angle_degrees=angle,
                radius=0.3,
                surface_tension=1.0,
                time_step_size=0.001,
                mobility=1.0,
                slip_length=0.1,
                dynamic=False,
            )
            grid = runner.pv.read(
                case_dir / "mesh/background/mesh-complete.mesh.vtu")
            fitted = runner.fit_sessile_interface(grid)
            assert fitted["available"] is True
            state = runner.sessile_state_metrics(
                grid, runner.load_benchmark(case_dir))
            assert state["operator_contact_geometry_available"] is True
            # The static physical benchmark must retain the globally sampled
            # circular cap.  Its generated LinearCorner chord angle is a P1
            # representation error for the refinement gate to measure; it is
            # not overwritten locally to manufacture an exact initial angle.
            benchmark = runner.load_benchmark(case_dir)
            contact = benchmark["sessile_contact"]
            assert contact["discrete_contact_initialization_local_overwrite"] is False
            assert contact["discrete_contact_initialization_cell_ids"] == []
            points = runner.np.asarray(grid.points, dtype=float)
            center = contact["circle_center"]
            radius = contact["circle_radius"]
            expected_phi = runner.np.sqrt(
                (points[:, 0] - float(center[0])) ** 2 +
                (points[:, 1] - float(center[1])) ** 2
            ) - float(radius)
            assert runner.np.allclose(
                runner.np.asarray(grid.point_data["phi"]), expected_phi,
                rtol=0.0, atol=2.0e-15)
            generated_angle_error = abs(
                state["operator_dynamic_angle_degrees_mean"] - angle)
            assert 1.0e-3 < generated_angle_error < 7.0
            assert abs(fitted["contact_angle_degrees"] - angle) < 0.5
            assert fitted["circle_fit_rmse"] < 7.5e-4


def test_sessile_manufactured_contact_initialization_sets_discrete_angle():
    runner = _load_runner()
    with tempfile.TemporaryDirectory() as temp_dir:
        case_dir = Path(temp_dir) / "manufactured_contact"
        runner.write_sessile2d_case(
            case_dir,
            steps=1,
            nx=8,
            ny=8,
            initial_angle_degrees=60.0,
            equilibrium_angle_degrees=60.0,
            radius=0.3,
            surface_tension=1.0,
            time_step_size=0.001,
            mobility=1.0,
            slip_length=0.1,
            dynamic=False,
            contact_line_model="prescribed",
            initialize_discrete_static_contact_geometry=True,
        )

        benchmark = runner.load_benchmark(case_dir)
        contact = benchmark["sessile_contact"]
        assert contact["discrete_contact_initialization_local_overwrite"] is True
        assert contact["discrete_static_contact_initialization"] is True
        assert len(contact["discrete_contact_initialization_cell_ids"]) == 2

        mesh = runner.pv.read(
            case_dir / "mesh/background/mesh-complete.mesh.vtu")
        state = runner.sessile_state_metrics(mesh, benchmark)
        assert state["operator_contact_geometry_available"] is True
        assert math.isclose(
            state["operator_dynamic_angle_degrees_mean"],
            60.0,
            rel_tol=0.0,
            abs_tol=2.0e-12,
        )


def test_stationary_sessile_cap_rotates_to_every_wall_and_is_scale_invariant():
    runner = _load_runner()
    wall_contracts = {
        "wall_bottom": (1, 0.0, (0.0, -1.0, 0.0), "0 1"),
        "wall_left": (0, 0.0, (-1.0, 0.0, 0.0), "1 0"),
        "wall_right": (0, 1.0, (1.0, 0.0, 0.0), "1 0"),
        "wall_top": (1, 1.0, (0.0, 1.0, 0.0), "0 1"),
    }
    with tempfile.TemporaryDirectory() as temp_dir:
        for wall, (axis, coordinate, normal, direction) in wall_contracts.items():
            states = []
            for scale in (0.25, 4.0):
                case_dir = Path(temp_dir) / f"{wall}_{scale}"
                runner.write_sessile2d_case(
                    case_dir,
                    steps=1,
                    nx=16,
                    ny=16,
                    initial_angle_degrees=60.0,
                    equilibrium_angle_degrees=60.0,
                    radius=0.3,
                    surface_tension=1.0,
                    time_step_size=0.001,
                    mobility=1.0,
                    slip_length=0.1,
                    dynamic=False,
                    wall_face=wall,
                    contact_line_model="prescribed",
                    level_set_positive_scale=scale,
                )

                benchmark = runner.load_benchmark(case_dir)
                contact = benchmark["sessile_contact"]
                assert contact["wall"] == wall
                assert contact["wall_axis"] == axis
                assert contact["wall_coordinate"] == coordinate
                assert tuple(contact["wall_normal"]) == normal
                assert contact["level_set_positive_scale"] == scale

                root = runner.ET.parse(case_dir / "solver.xml").getroot()
                fluid = runner.fluid_equation(root)
                wall_bc = next(
                    bc for bc in fluid.findall("Add_BC")
                    if bc.attrib.get("name") == wall
                )
                assert wall_bc.findtext("Effective_direction") == direction
                free_surface = runner.free_surface_bc(root)
                assert free_surface.findtext("Contact_line_wall_face") == wall
                assert tuple(float(value) for value in free_surface.findtext(
                    "Contact_line_wall_normal").split()) == normal

                mesh = runner.pv.read(
                    case_dir / "mesh/background/mesh-complete.mesh.vtu")
                points = runner.np.asarray(mesh.points, dtype=float)
                center = runner.np.asarray(contact["circle_center"][:2])
                radius = float(contact["circle_radius"])
                expected_phi = scale * (
                    runner.np.linalg.norm(points[:, :2] - center, axis=1) -
                    radius
                )
                assert runner.np.allclose(
                    runner.np.asarray(mesh.point_data["phi"]),
                    expected_phi,
                    rtol=0.0,
                    atol=8.0e-15,
                )
                state = runner.sessile_state_metrics(mesh, benchmark)
                assert state["operator_contact_geometry_available"] is True
                samples = state["operator_contact_geometry_samples"]
                assert len(samples) == 2
                assert all(
                    abs(float(sample["point"][axis]) - coordinate) < 1.0e-13
                    for sample in samples
                )
                states.append(state)

            for key in (
                    "operator_dynamic_angle_degrees_mean",
                    "operator_dynamic_cos_mean",
                    "operator_young_gap_mean",
                    "contact_fluid_outward_speed"):
                assert math.isclose(
                    float(states[0][key]),
                    float(states[1][key]),
                    rel_tol=0.0,
                    abs_tol=2.0e-12,
                )


def test_stationary_sessile_active_side_and_tangent_offset_are_invariant():
    runner = _load_runner()
    offset = 0.05
    grids = []
    states = []
    with tempfile.TemporaryDirectory() as temp_dir:
        for label, active_domain in (
                ("negative", "LevelSetNegative"),
                ("positive", "LevelSetPositive")):
            case_dir = Path(temp_dir) / label
            runner.write_sessile2d_case(
                case_dir,
                steps=1,
                nx=16,
                ny=16,
                initial_angle_degrees=60.0,
                equilibrium_angle_degrees=60.0,
                radius=0.3,
                surface_tension=1.0,
                time_step_size=0.001,
                mobility=1.0,
                slip_length=0.1,
                dynamic=False,
                wall_face="wall_left",
                contact_line_model="prescribed",
                active_domain=active_domain,
                tangent_center_offset=offset,
            )
            benchmark = runner.load_benchmark(case_dir)
            contact = benchmark["sessile_contact"]
            grid = runner.pv.read(
                case_dir / "mesh/background/mesh-complete.mesh.vtu")
            phi = runner.np.asarray(grid.point_data["phi"], dtype=float)
            gauge_node = int(benchmark["pressure_gauge"]["node_id"])

            assert benchmark["active_domain"] == active_domain
            assert benchmark["tangent_center_offset"] == offset
            assert contact["active_domain"] == active_domain
            assert contact["circle_center"][1] == pytest.approx(0.5 + offset)
            assert runner.active_signed_level_set(
                phi, active_domain)[gauge_node] < 0.0
            root = runner.ET.parse(case_dir / "solver.xml").getroot()
            assert runner.free_surface_bc(root).findtext(
                "Active_domain") == active_domain
            runner.configure_solver(
                case_dir / "solver.xml",
                steps=1,
                active_domain=active_domain,
            )

            state = runner.sessile_state_metrics(grid, benchmark)
            assert state["available"] is True
            assert state["active_domain"] == active_domain
            grids.append(grid)
            states.append(state)

    assert runner.np.allclose(
        grids[0].point_data["phi"], -grids[1].point_data["phi"])
    for metric in (
            "circle_radius",
            "circle_fit_rmse",
            "contact_angle_degrees",
            "operator_dynamic_angle_degrees_mean",
            "operator_dynamic_cos_mean",
            "max_liquid_speed"):
        assert float(states[0][metric]) == pytest.approx(
            float(states[1][metric]), abs=1.0e-8)


def test_closed_droplet_active_side_and_center_offset_are_invariant():
    runner = _load_runner()
    offset = (0.04, -0.03)
    grids = []
    metrics = []
    with tempfile.TemporaryDirectory() as temp_dir:
        for label, active_domain in (
                ("negative", "LevelSetNegative"),
                ("positive", "LevelSetPositive")):
            case_dir = Path(temp_dir) / label
            runner.write_capillary_droplet2d_case(
                case_dir,
                steps=1,
                pressure_jump=2.0,
                nx=12,
                ny=12,
                simplex_mesh=True,
                active_domain=active_domain,
                center_offset=offset,
            )
            benchmark = runner.load_benchmark(case_dir)
            mesh_path = (
                case_dir / "mesh/background/mesh-complete.mesh.vtu")
            grid = runner.pv.read(mesh_path)
            phi = runner.np.asarray(grid.point_data["phi"], dtype=float)
            gauge_node = int(benchmark["pressure_gauge"]["node_id"])

            assert benchmark["active_domain"] == active_domain
            assert benchmark["circle_center_offset"] == pytest.approx(offset)
            assert benchmark["circle_center"] == pytest.approx(
                (0.54, 0.47))
            assert runner.active_signed_level_set(
                phi, active_domain)[gauge_node] < 0.0
            root = runner.ET.parse(case_dir / "solver.xml").getroot()
            assert runner.free_surface_bc(root).findtext(
                "Active_domain") == active_domain
            state = runner.compute_metrics(
                "droplet2d", case_dir, mesh_path)
            assert state["active_domain"] == active_domain
            grids.append(grid)
            metrics.append(state)

    assert runner.np.allclose(
        grids[0].point_data["phi"], -grids[1].point_data["phi"])
    for metric in (
            "wet_nodes",
            "max_speed",
            "wet_mean_speed",
            "interface_peak_height",
            "interface_front_x"):
        assert float(metrics[0][metric]) == pytest.approx(
            float(metrics[1][metric]), abs=1.0e-7)


@pytest.mark.parametrize("scale", [0.0, -1.0, math.inf, math.nan])
def test_stationary_sessile_rejects_nonpositive_or_nonfinite_scale(scale):
    runner = _load_runner()
    with tempfile.TemporaryDirectory() as temp_dir:
        with pytest.raises(ValueError, match="positive scale"):
            runner.write_sessile2d_case(
                Path(temp_dir) / "invalid_scale",
                steps=1,
                nx=8,
                ny=8,
                initial_angle_degrees=90.0,
                equilibrium_angle_degrees=90.0,
                radius=0.3,
                surface_tension=1.0,
                time_step_size=0.001,
                mobility=1.0,
                slip_length=0.1,
                dynamic=False,
                level_set_positive_scale=scale,
            )


def test_sessile_energy_density_matches_generated_solver_deck():
    runner = _load_runner()
    with tempfile.TemporaryDirectory() as temp_dir:
        case_dir = Path(temp_dir) / "sessile"
        runner.write_sessile2d_case(
            case_dir,
            steps=1,
            nx=8,
            ny=8,
            initial_angle_degrees=90.0,
            equilibrium_angle_degrees=90.0,
            radius=0.3,
            surface_tension=1.0,
            time_step_size=0.001,
            mobility=1.0,
            slip_length=0.1,
            dynamic=False,
        )
        benchmark_density = float(runner.load_benchmark(case_dir)["density"])
        root = runner.ET.parse(case_dir / "solver.xml").getroot()
        solver_density = float(runner.fluid_equation(root).findtext("Density"))
        assert benchmark_density == solver_density == 1.0


def test_stationary_sessile_case_supports_prescribed_contact_angle_ownership():
    runner = _load_runner()
    with tempfile.TemporaryDirectory() as temp_dir:
        case_dir = Path(temp_dir) / "sessile_prescribed"
        runner.write_sessile2d_case(
            case_dir,
            steps=1,
            nx=8,
            ny=8,
            initial_angle_degrees=60.0,
            equilibrium_angle_degrees=60.0,
            radius=0.3,
            surface_tension=1.0,
            time_step_size=0.001,
            mobility=1.0,
            slip_length=0.1,
            dynamic=False,
            contact_line_model="prescribed",
        )

        root = runner.ET.parse(case_dir / "solver.xml").getroot()
        free_surface = runner.free_surface_bc(root)
        level_set = runner.level_set_equation(root)
        assert free_surface.findtext(
            "Contact_line_model") == "PrescribedContactAngle"
        assert level_set.findtext("Enable_bound_preserving_limiter") == "false"
        assert free_surface.find("Contact_line_mobility") is None
        assert free_surface.find("Wall_slip_model") is None
        assert free_surface.find("Wall_slip_length") is None

        benchmark = runner.load_benchmark(case_dir)
        contact = benchmark["sessile_contact"]
        assert contact["contact_line_model"] == "PrescribedContactAngle"
        assert contact["level_set_geometry_owner"] == (
            "accepted_state_wall_aware_repair")
        assert contact["momentum_owner"] == "young_wall_energy"
        assert "mobility" not in contact
        assert "ren_e_relation" not in contact

        mesh = runner.pv.read(
            case_dir / "mesh/background/mesh-complete.mesh.vtu")
        state = runner.sessile_state_metrics(mesh, benchmark)
        assert state["operator_contact_geometry_available"] is True
        assert state["operator_contact_geometry_sample_count"] == 2
        assert state["operator_contact_geometry_source"] == (
            "LinearCorner_generated_fragment_normal_at_phi_zero_wall_roots")
        assert "operator_predicted_contact_line_speed" not in state


def test_dynamic_sessile_pair_preserves_one_reference_liquid_area():
    runner = _load_runner()
    reference_radius = 0.3
    _center, reference_half, reference_area = runner.sessile_circle_geometry(
        90.0, reference_radius)
    assert reference_half == reference_radius
    with tempfile.TemporaryDirectory() as temp_dir:
        records = {}
        for label, angle in (("advancing", 95.0), ("receding", 85.0)):
            case_dir = Path(temp_dir) / label
            runner.write_sessile2d_case(
                case_dir,
                steps=1,
                nx=16,
                ny=16,
                initial_angle_degrees=angle,
                equilibrium_angle_degrees=90.0,
                radius=reference_radius,
                surface_tension=1.0,
                time_step_size=0.001,
                mobility=1.0,
                slip_length=0.1,
                dynamic=True,
            )
            benchmark = runner.load_benchmark(case_dir)
            contact = benchmark["sessile_contact"]
            initial_radius = contact["circle_radius"]
            records[label] = contact
            assert contact["liquid_area_contract"] == (
                "fixed_at_equilibrium_reference_cap")
            assert contact["equilibrium_reference_radius"] == reference_radius
            assert math.isclose(
                contact["expected_initial_liquid_area"], reference_area,
                rel_tol=1.0e-14)
            assert math.isclose(
                runner.sessile_circle_geometry(angle, initial_radius)[2],
                reference_area,
                rel_tol=1.0e-14,
            )
            mesh = runner.pv.read(
                case_dir / "mesh/background/mesh-complete.mesh.vtu")
            assert runner.np.allclose(
                runner.np.asarray(mesh.point_data["Pressure"]),
                1.0 / initial_radius,
            )

        assert records["advancing"]["expected_initial_half_footprint"] < reference_half
        assert records["receding"]["expected_initial_half_footprint"] > reference_half
        assert records["advancing"]["circle_radius"] < reference_radius
        assert records["receding"]["circle_radius"] > reference_radius


def test_sessile_contact_fluid_speed_interpolates_the_ren_e_velocity_observable():
    runner = _load_runner()
    with tempfile.TemporaryDirectory() as temp_dir:
        case_dir = Path(temp_dir) / "sessile"
        runner.write_sessile2d_case(
            case_dir, 1, 16, 16, 90.0, 90.0, 0.3, 1.0, 0.001,
            1.0, 0.1, True)
        mesh = runner.pv.read(
            case_dir / "mesh/background/mesh-complete.mesh.vtu")
        points = runner.np.asarray(mesh.points, dtype=float)
        velocity = runner.np.zeros((mesh.n_points, 3), dtype=float)
        velocity[:, 0] = 2.0 * (points[:, 0] - 0.5)
        mesh.point_data["Velocity"] = velocity
        state = runner.sessile_state_metrics(
            mesh, runner.load_benchmark(case_dir))

        assert state["available"] is True
        contact_x = state["contact_fluid_evaluation_contact_x"]
        expected = contact_x[1] - contact_x[0]
        assert math.isclose(
            state["contact_fluid_outward_speed"], expected,
            rel_tol=1.0e-12, abs_tol=1.0e-12)
        assert abs(state["contact_fluid_symmetry_defect"]) < 1.0e-12
        assert state["contact_fluid_evaluation_source"] == (
            "phi_zero_wall_intersections")
        assert math.isclose(
            state["operator_contact_fluid_speed"], expected,
            rel_tol=1.0e-12, abs_tol=1.0e-12)
        assert state["operator_contact_fluid_evaluation_source"] == (
            "Q1_velocity_and_generated_fragment_normal_at_phi_zero_wall_roots")
        assert math.isclose(
            state["max_generated_contact_fluid_speed"], expected,
            rel_tol=1.0e-12, abs_tol=1.0e-12)
        assert math.isclose(
            state["max_liquid_speed"], expected,
            rel_tol=1.0e-12, abs_tol=1.0e-12)
        assert state["max_liquid_speed_source"] == (
            "generated_contact_fluid_interpolation")


def test_sessile_operator_contact_geometry_uses_generated_fragment_normal():
    runner = _load_runner()
    with tempfile.TemporaryDirectory() as temp_dir:
        case_dir = Path(temp_dir) / "sessile"
        runner.write_sessile2d_case(
            case_dir, 1, 16, 16, 90.0, 90.0, 0.3, 1.0, 0.001,
            1.0, 0.1, True)
        mesh = runner.pv.read(
            case_dir / "mesh/background/mesh-complete.mesh.vtu")
        state = runner.sessile_state_metrics(
            mesh, runner.load_benchmark(case_dir))

        assert state["operator_contact_geometry_available"] is True
        assert state["operator_contact_geometry_sample_count"] == 2
        assert state["operator_contact_geometry_source"] == (
            "LinearCorner_generated_fragment_normal_at_phi_zero_wall_roots")
        assert math.isclose(state["operator_dynamic_cos_mean"], 0.0,
                            rel_tol=0.0, abs_tol=1.0e-12)
        assert math.isclose(state["operator_dynamic_angle_degrees_mean"],
                            90.0, rel_tol=0.0, abs_tol=1.0e-12)
        assert math.isclose(
            state["diagnostic_q1_dynamic_angle_degrees_mean"],
            90.0, rel_tol=0.0, abs_tol=1.0e-12)
        assert math.isclose(
            state["operator_young_gap_mean"],
            0.0, rel_tol=0.0, abs_tol=1.0e-12)
        assert math.isclose(
            state["operator_predicted_contact_line_speed"],
            state["operator_young_gap_mean"],
            rel_tol=1.0e-12, abs_tol=1.0e-12)
        assert state["operator_wall_tangential_normal_norm_min"] > 0.99


def test_sessile_angle_and_ren_e_metrics_gate_generated_surface_geometry():
    runner = _load_runner()
    with tempfile.TemporaryDirectory() as temp_dir:
        case_dir = Path(temp_dir) / "sessile"
        runner.write_sessile2d_case(
            case_dir, 1, 16, 16, 95.0, 90.0, 0.3, 1.0, 0.001,
            1.0, 0.1, True)
        benchmark = runner.load_benchmark(case_dir)
        initial = runner.pv.read(
            case_dir / "mesh/background/mesh-complete.mesh.vtu")
        output = initial.copy()
        points = runner.np.asarray(output.points, dtype=float)
        expected_prediction = benchmark["sessile_contact"][
            "predicted_initial_contact_line_speed"]
        rho_inf = 0.5
        alpha_f = 1.0 / (1.0 + rho_inf)
        half_footprint = benchmark["sessile_contact"][
            "expected_initial_half_footprint"]
        velocity = runner.np.zeros((output.n_points, 3), dtype=float)
        # The saved endpoint is the generalized-alpha extrapolation.  With a
        # zero initial velocity, scaling it by 1/alpha_f makes the reconstructed
        # stage velocity (and only that stage velocity) satisfy Ren--E exactly.
        velocity[:, 0] = (
            (expected_prediction / (alpha_f * half_footprint)) *
            (points[:, 0] - 0.5))
        output.point_data["Velocity"] = velocity
        output.save(case_dir / "result_001.vtu")

        metrics = {}
        runner.add_physical_time_history_metrics(
            metrics, case_dir, benchmark, initial,
            accepted_steps=[{"step": 1, "time": 0.001, "dt": 0.001}],
            transient_solve={
                "scheme": "GeneralizedAlpha",
                "rho_inf": rho_inf,
            },
        )

        assert metrics["sessile_final_contact_angle_source"] == (
            "same_state_LinearCorner_generated_fragment_normal_at_phi_zero_wall_roots")
        assert math.isclose(
            metrics["sessile_final_contact_angle_degrees"],
            95.0, rel_tol=1.0e-12, abs_tol=1.0e-12)
        assert metrics["ren_e_prediction_source"] == (
            runner.GENERALIZED_ALPHA_REN_E_PREDICTION_SOURCE)
        assert metrics["ren_e_contact_fluid_evaluation_source"] == (
            runner.GENERALIZED_ALPHA_REN_E_VELOCITY_SOURCE)
        assert metrics["ren_e_stage_reconstruction_available"] is True
        assert metrics["ren_e_stage_state_source"] == (
            runner.GENERALIZED_ALPHA_STAGE_STATE_SOURCE)
        assert metrics["ren_e_generalized_alpha_parameter_source"] == (
            "parsed_solver_transient_diagnostics")
        assert math.isclose(metrics["ren_e_generalized_alpha_rho_inf"], rho_inf)
        assert math.isclose(metrics["ren_e_generalized_alpha_alpha_f"], alpha_f)
        assert math.isclose(
            metrics["ren_e_constitutive_stage_time"], alpha_f * 0.001,
            rel_tol=0.0, abs_tol=1.0e-15)
        assert math.isclose(
            metrics["ren_e_predicted_final_contact_line_speed"],
            expected_prediction, rel_tol=1.0e-12, abs_tol=1.0e-12)
        assert math.isclose(
            metrics["ren_e_measured_final_contact_fluid_speed"],
            expected_prediction, rel_tol=1.0e-12, abs_tol=1.0e-12)
        assert metrics["ren_e_contact_fluid_speed_relative_error"] < 1.0e-12


def test_vertical_left_wall_dynamic_contact_uses_rotated_wall_frame_and_gates():
    runner = _load_runner()
    with tempfile.TemporaryDirectory() as temp_dir:
        case_dir = Path(temp_dir) / "dynamiccontact_left_wall"
        dt = 0.001
        runner.write_sessile2d_case(
            case_dir, 1, 16, 16, 95.0, 90.0, 0.3, 1.0, dt,
            1.0, 0.1, True, "wall_left")

        root = runner.ET.parse(case_dir / "solver.xml").getroot()
        fluid = runner.fluid_equation(root)
        walls = {
            bc.attrib.get("name"): bc for bc in fluid.findall("Add_BC")
        }
        assert walls["wall_left"].findtext("Effective_direction") == "1 0"
        assert walls["wall_bottom"].find("Effective_direction") is None
        free_surface = runner.free_surface_bc(root)
        assert free_surface.findtext("Contact_line_wall_face") == "wall_left"
        assert free_surface.findtext("Contact_line_wall_normal") == "-1.0 0.0 0.0"

        benchmark = runner.load_benchmark(case_dir)
        contact = benchmark["sessile_contact"]
        assert contact["wall"] == "wall_left"
        assert contact["wall_axis"] == 0
        assert contact["wall_coordinate"] == 0.0
        assert contact["wall_tangent_axis"] == 1
        assert contact["wall_normal"] == [-1.0, 0.0, 0.0]
        assert contact["wall_tangent"] == [0.0, 1.0, 0.0]
        assert contact["circle_center"][1] == 0.5

        initial = runner.pv.read(
            case_dir / "mesh/background/mesh-complete.mesh.vtu")
        initial_state = runner.sessile_state_metrics(initial, benchmark)
        assert initial_state["operator_contact_geometry_available"] is True
        assert math.isclose(
            initial_state["operator_dynamic_angle_degrees_mean"],
            95.0, rel_tol=0.0, abs_tol=1.0e-12)
        samples = initial_state["operator_contact_geometry_samples"]
        assert len(samples) == 2
        assert all(abs(sample["point"][0]) < 1.0e-14 for sample in samples)
        assert samples[0]["point"][1] < 0.5 < samples[1]["point"][1]
        assert samples[0]["generated_fragment_normal"][1] < 0.0
        assert samples[1]["generated_fragment_normal"][1] > 0.0

        prediction = float(contact["predicted_initial_contact_line_speed"])
        initial_half = float(contact["expected_initial_half_footprint"])
        endpoint_half = initial_half + prediction * dt
        rho_inf = 0.5
        alpha_f = 1.0 / (1.0 + rho_inf)
        stage_half = initial_half + alpha_f * prediction * dt
        output = initial.copy(deep=True)
        points = runner.np.asarray(output.points, dtype=float)
        endpoint_phi = runner.np.asarray(
            output.point_data["phi"], dtype=float).copy()
        theta = math.radians(95.0)
        wall_normal = runner.np.asarray([-1.0, 0.0])
        wall_tangent = runner.np.asarray([0.0, 1.0])
        for cell_id in contact["discrete_contact_initialization_cell_ids"]:
            point_ids = runner.np.asarray(
                output.get_cell(int(cell_id)).point_ids, dtype=int)
            side = -1.0 if float(runner.np.mean(points[point_ids, 1])) < 0.5 else 1.0
            contact_point = runner.np.asarray([
                0.0, 0.5 + side * endpoint_half])
            outward_normal = (
                side * math.sin(theta) * wall_tangent -
                math.cos(theta) * wall_normal)
            endpoint_phi[point_ids] = (
                points[point_ids, :2] - contact_point) @ outward_normal
        output.point_data["phi"] = endpoint_phi
        velocity = runner.np.zeros((output.n_points, 3), dtype=float)
        velocity[:, 1] = (
            prediction / (alpha_f * stage_half) * (points[:, 1] - 0.5))
        output.point_data["Velocity"] = velocity
        output.save(case_dir / "result_001.vtu")

        metrics = {}
        runner.add_physical_time_history_metrics(
            metrics, case_dir, benchmark, initial,
            accepted_steps=[{"step": 1, "time": dt, "dt": dt}],
            transient_solve={
                "scheme": "GeneralizedAlpha",
                "rho_inf": rho_inf,
            },
        )
        assert metrics["ren_e_stage_reconstruction_available"] is True
        assert math.isclose(
            metrics["ren_e_predicted_final_contact_line_speed"], prediction,
            rel_tol=1.0e-11, abs_tol=1.0e-12)
        assert math.isclose(
            metrics["ren_e_measured_final_contact_fluid_speed"], prediction,
            rel_tol=1.0e-11, abs_tol=1.0e-12)
        assert metrics["ren_e_contact_fluid_speed_sign_agrees"] is True
        assert metrics["ren_e_contact_fluid_speed_relative_error"] < 1.0e-10
        assert math.isclose(
            metrics["ren_e_measured_mean_geometric_contact_line_speed"],
            prediction, rel_tol=1.0e-9, abs_tol=1.0e-11)
        assert math.isclose(
            metrics["ren_e_final_interval_geometric_contact_line_speed"],
            prediction, rel_tol=1.0e-9, abs_tol=1.0e-11)
        assert metrics["ren_e_geometric_speed_sign_agrees"] is True


def test_generalized_alpha_stage_reconstruction_uses_global_ids_and_both_endpoints():
    runner = _load_runner()
    points = runner.np.asarray([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [2.0, 0.0, 0.0],
    ])
    gids = runner.np.asarray([10, 11, 12], dtype=runner.np.int64)
    initial = runner.pv.PolyData(points)
    initial.point_data["GlobalNodeID"] = gids
    initial.point_data["phi"] = runner.np.asarray([-1.0, 0.0, 1.0])
    initial.point_data["Velocity"] = runner.np.zeros((3, 3))

    previous = initial.copy(deep=True)
    previous.point_data["phi"] = runner.np.asarray([1.0, 2.0, 3.0])
    previous.point_data["Velocity"] = runner.np.asarray([
        [1.0, 10.0, 100.0],
        [2.0, 20.0, 200.0],
        [3.0, 30.0, 300.0],
    ])

    permutation = runner.np.asarray([2, 0, 1])
    current = runner.pv.PolyData(points[permutation])
    current.point_data["GlobalNodeID"] = gids[permutation]
    current_phi = runner.np.asarray([5.0, 6.0, 7.0])
    current_velocity = runner.np.asarray([
        [5.0, 50.0, 500.0],
        [6.0, 60.0, 600.0],
        [7.0, 70.0, 700.0],
    ])
    current.point_data["phi"] = current_phi[permutation]
    current.point_data["Velocity"] = current_velocity[permutation]

    alpha_f = 0.25
    stage = runner.reconstruct_generalized_alpha_first_order_stage(
        initial, previous, current, alpha_f)
    assert runner.np.allclose(
        stage.point_data["phi"],
        (1.0 - alpha_f) * runner.np.asarray([1.0, 2.0, 3.0]) +
        alpha_f * current_phi)
    assert runner.np.allclose(
        stage.point_data["Velocity"],
        (1.0 - alpha_f) * previous.point_data["Velocity"] +
        alpha_f * current_velocity)


def test_generalized_alpha_stage_reconstruction_trims_only_zero_2d_storage_tail():
    runner = _load_runner()
    points = runner.np.asarray([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
    ])
    initial = runner.pv.PolyData(points)
    initial.point_data["GlobalNodeID"] = runner.np.asarray(
        [10, 11], dtype=runner.np.int64)
    initial.point_data["phi"] = runner.np.asarray([-1.0, 1.0])
    initial.point_data["Velocity"] = runner.np.asarray([
        [1.0, 2.0, 0.0],
        [3.0, 4.0, 0.0],
    ])
    current = initial.copy(deep=True)
    current.point_data["Velocity"] = runner.np.asarray([
        [5.0, 6.0],
        [7.0, 8.0],
    ])

    stage = runner.reconstruct_generalized_alpha_first_order_stage(
        initial, initial, current, 0.25)
    assert runner.np.asarray(stage.point_data["Velocity"]).shape == (2, 2)
    assert runner.np.allclose(
        stage.point_data["Velocity"],
        runner.np.asarray([[2.0, 3.0], [4.0, 5.0]]),
    )

    nonzero_tail = initial.copy(deep=True)
    nonzero_tail.point_data["Velocity"] = runner.np.asarray([
        [1.0, 2.0, 9.0],
        [3.0, 4.0, 0.0],
    ])
    with pytest.raises(ValueError, match="shapes differ"):
        runner.reconstruct_generalized_alpha_first_order_stage(
            initial, nonzero_tail, current, 0.25)


def test_incomplete_solve_postprocesses_accepted_dynamic_contact_stage():
    runner = _load_runner()
    with tempfile.TemporaryDirectory() as temp_dir:
        case_dir = Path(temp_dir) / "dynamiccontact2d"
        runner.write_sessile2d_case(
            case_dir, 1, 16, 16, 95.0, 90.0, 0.3, 1.0, 0.001,
            1.0, 0.1, True)
        benchmark = runner.load_benchmark(case_dir)
        initial = runner.pv.read(
            case_dir / "mesh/background/mesh-complete.mesh.vtu")
        output = initial.copy(deep=True)
        points = runner.np.asarray(output.points, dtype=float)
        alpha_f = 2.0 / 3.0
        prediction = benchmark["sessile_contact"][
            "predicted_initial_contact_line_speed"]
        half_footprint = benchmark["sessile_contact"][
            "expected_initial_half_footprint"]
        velocity = runner.np.zeros((output.n_points, 2), dtype=float)
        velocity[:, 0] = (
            prediction / (alpha_f * half_footprint) *
            (points[:, 0] - 0.5))
        output.point_data["Velocity"] = velocity
        output.save(case_dir / "result_001.vtu")

        diagnostics = {
            "solver_controls": {
                "transient_solve": {
                    "scheme": "GeneralizedAlpha",
                    "rho_inf": 0.5,
                },
            },
            "time_loop": {
                "accepted_steps": [
                    {"step": 1, "time": 0.001, "dt": 0.001},
                ],
            },
        }
        args = runner.argparse.Namespace(
            disable_vtk_output=False,
            enable_physical_history_instrumentation=True,
            require_level_set_mass_correction_histories=False,
        )
        metrics = {}
        runner.add_incomplete_solve_output_metrics(
            metrics, "dynamiccontact2d", case_dir, diagnostics, args)

        assert metrics["incomplete_solve_output_metrics_available"] is True
        assert metrics["result_step"] == 1
        assert metrics["ren_e_stage_reconstruction_available"] is True
        assert metrics["ren_e_contact_fluid_evaluation_source"] == (
            runner.GENERALIZED_ALPHA_REN_E_VELOCITY_SOURCE)
        assert metrics["ren_e_contact_fluid_speed_relative_error"] < 1.0e-12


def test_generalized_alpha_stage_parameters_are_fail_closed():
    runner = _load_runner()
    parameters = runner.generalized_alpha_first_order_stage_parameters(
        Path("unused"),
        {"scheme": "GeneralizedAlpha", "rho_inf": 0.5},
    )
    assert math.isclose(parameters["alpha_f"], 2.0 / 3.0)
    assert parameters["parameter_source"] == "parsed_solver_transient_diagnostics"

    with pytest.raises(ValueError, match="requires GeneralizedAlpha"):
        runner.generalized_alpha_first_order_stage_parameters(
            Path("unused"), {"scheme": "BDF2", "rho_inf": 0.5})
    with pytest.raises(ValueError, match="rho_inf must be finite"):
        runner.generalized_alpha_first_order_stage_parameters(
            Path("unused"),
            {"scheme": "GeneralizedAlpha", "rho_inf": 1.5})


def test_capillary_wave_frequency_is_fitted_from_saved_solution_history():
    runner = _load_runner()
    surface_tension = 50.0
    with tempfile.TemporaryDirectory() as temp_dir:
        case_dir = Path(temp_dir) / "wave"
        runner.write_capillary_wave2d_case(
            case_dir,
            steps=100,
            surface_tension=surface_tension,
            time_step_size=0.001,
        )
        initial_path = case_dir / "mesh/background/mesh-complete.mesh.vtu"
        initial = runner.pv.read(initial_path)
        points = runner.np.asarray(initial.points, dtype=float)
        accepted_steps = [
            {"step": 1, "time": 0.0230, "dt": 0.0230},
            {"step": 2, "time": 0.0485, "dt": 0.0255},
            {"step": 3, "time": 0.0730, "dt": 0.0245},
            {"step": 4, "time": 0.1000, "dt": 0.0270},
        ]
        for accepted in accepted_steps:
            step = accepted["step"]
            output = initial.copy()
            height = runner.capillary_wave_height(
                points[:, 0], accepted["time"], surface_tension)
            output.point_data["phi"] = points[:, 1] - height
            output.save(case_dir / f"result_{step:03d}.vtu")

        metrics = {}
        runner.add_physical_time_history_metrics(
            metrics,
            case_dir,
            runner.load_benchmark(case_dir),
            initial,
            accepted_steps=accepted_steps,
        )

        expected = runner.capillary_wave_omega(surface_tension)
        k = 2.0 * runner.math.pi / runner.CAPILLARY_WAVE_WAVELENGTH
        finite_depth_expected = runner.math.sqrt(
            surface_tension * k ** 3 *
            runner.math.tanh(k * runner.CAPILLARY_WAVE_DEPTH) /
            runner.CAPILLARY_WAVE_DENSITY
        )
        assert abs(expected - finite_depth_expected) < 1.0e-14
        observed = metrics["capillary_wave_observed_omega"]
        assert len(metrics["capillary_wave_amplitude_history"]) == 5
        assert metrics["capillary_wave_frequency_observed_phase_span"] > 0.25
        assert abs(observed - expected) / expected < 1.0e-3
        expected_clock = [
            (accepted["step"], accepted["time"], accepted["dt"])
            for accepted in accepted_steps
        ]
        assert [
            (record["step"], record["time"], record["dt"])
            for record in metrics["physical_liquid_measure_history"]
        ] == expected_clock
        assert [
            (record["step"], record["time"], record["dt"])
            for record in metrics["capillary_wave_amplitude_history"][1:]
        ] == expected_clock
        assert [
            (record["step"], record["time"], record["dt"])
            for record in metrics["wall_only_false_wet_history"]
        ] == expected_clock
        assert metrics["physical_time_history_clock_source"] == (
            "solver_accepted_steps"
        )
        assert metrics["physical_time_history_clock_complete"] is True
        assert metrics["physical_time_history_missing_accepted_step_ids"] == []
        assert metrics["physical_time_history_output_step_identity_complete"] is True
        assert metrics["physical_time_history_accepted_step_ids"] == [1, 2, 3, 4]
        assert metrics["physical_time_history_output_step_ids"] == [1, 2, 3, 4]
        assert metrics["physical_time_history_missing_output_step_ids"] == []
        assert metrics["physical_time_history_unexpected_output_step_ids"] == []


def test_physical_history_clock_fails_when_saved_step_has_no_accepted_record():
    runner = _load_runner()
    clock, parse_errors = runner.accepted_step_clock([
        {"step": 2, "time": 0.00125, "dt": 0.00025},
    ])
    assert parse_errors == []
    stamp, exact = runner.physical_history_stamp(2, clock, 0.001)
    assert stamp == {"step": 2, "time": 0.00125, "dt": 0.00025}
    assert exact is True
    fallback, exact = runner.physical_history_stamp(3, clock, 0.001)
    assert fallback == {"step": 3, "time": 0.003, "dt": 0.001}
    assert exact is False

    args = runner.argparse.Namespace(
        enable_physical_history_instrumentation=True)
    errors = runner.physical_history_clock_errors({
        "physical_time_history_clock_source": "solver_accepted_steps",
        "physical_time_history_clock_errors": [],
        "physical_time_history_missing_accepted_step_ids": [3],
        "physical_time_history_missing_output_step_ids": [],
        "physical_time_history_unexpected_output_step_ids": [],
        "physical_time_history_output_step_identity_complete": True,
        "physical_time_history_clock_complete": False,
    }, args)
    assert len(errors) == 1
    assert "output step(s) 3" in errors[0]


def test_physical_history_clock_requires_exact_accepted_output_identity():
    runner = _load_runner()
    args = runner.argparse.Namespace(
        enable_physical_history_instrumentation=True)
    metrics = {
        "physical_time_history_clock_source": "solver_accepted_steps",
        "physical_time_history_clock_errors": [],
        "physical_time_history_missing_accepted_step_ids": [],
        "physical_time_history_accepted_step_ids": [1, 2, 3],
        "physical_time_history_output_step_ids": [1, 3],
        "physical_time_history_missing_output_step_ids": [2],
        "physical_time_history_unexpected_output_step_ids": [],
        "physical_time_history_output_step_identity_complete": False,
        "physical_time_history_clock_complete": False,
    }
    errors = runner.physical_history_clock_errors(metrics, args)
    assert errors == [
        "physical history is missing VTK output for accepted step(s) 2"
    ]

    metrics.update({
        "physical_time_history_output_step_ids": [1, 2, 3],
        "physical_time_history_missing_output_step_ids": [],
        "physical_time_history_output_step_identity_complete": True,
        "physical_time_history_clock_complete": True,
    })
    assert runner.physical_history_clock_errors(metrics, args) == []


def test_production_physical_volume_history_is_shared_and_fail_closed():
    runner = _load_runner()

    def record(step, time, volume):
        return {
            "step": step,
            "time": time,
            "field": "phi",
            "domain_id": "open_vessel_surface",
            "marker": 1484991,
            "active_side": "LevelSetNegative",
            "isovalue": 0.0,
            "wet_volume": volume,
            "wet_volume_frame": "physical",
            "physical_wet_volume": volume,
            "initial_wet_volume": 0.5,
            "volume_rule_count": 36,
            "physical_volume_rule_count": 36,
            "skipped_physical_volume_rule_count": 0,
        }

    metrics = {
        "solver_controls": {"transient_solve": {"t0": 0.0}},
        "time_loop": {"accepted_steps": [
            {"step": 1, "time": 0.002, "dt": 0.002},
            {"step": 2, "time": 0.003, "dt": 0.001},
        ]},
        "production_wet_volume_diagnostic_history": [
            record(0, 0.0, 0.5),
            record(1, 0.002, 0.499999),
            record(2, 0.003, 0.5),
        ],
    }
    runner.add_production_physical_liquid_volume_metrics(metrics)
    args = runner.argparse.Namespace(
        enable_physical_history_instrumentation=True)
    assert runner.production_physical_liquid_volume_history_errors(
        metrics, args) == []
    assert metrics["production_physical_liquid_volume_available"] is True
    history = metrics["production_physical_liquid_volume_history"]
    assert [(item["step"], item["time"], item["dt"]) for item in history] == [
        (0, 0.0, 0.0),
        (1, 0.002, 0.002),
        (2, 0.003, 0.001),
    ]
    assert history[0]["state_stage"] == "initialized"
    assert history[-1]["state_stage"] == (
        "accepted_post_level_set_maintenance")

    missing = dict(metrics)
    missing["production_wet_volume_diagnostic_history"] = [
        record(0, 0.0, 0.5), record(2, 0.003, 0.5)]
    runner.add_production_physical_liquid_volume_metrics(missing)
    errors = runner.production_physical_liquid_volume_history_errors(
        missing, args)
    assert len(errors) == 1
    assert "initial and every accepted state" in errors[0]


def test_mass_correction_histories_are_separate_clocked_and_physical():
    runner = _load_runner()

    def production_record(step, time):
        return {
            "step": step,
            "time": time,
            "field": "phi",
            "domain_id": "open_vessel_surface",
            "marker": 1484991,
            "active_side": "LevelSetNegative",
            "isovalue": 0.0,
            "wet_volume": 0.5,
            "wet_volume_frame": "physical",
            "physical_wet_volume": 0.5,
            "initial_wet_volume": 0.5,
            "volume_rule_count": 36,
            "physical_volume_rule_count": 36,
            "skipped_physical_volume_rule_count": 0,
        }

    accepted = [
        {"step": 1, "time": 0.002, "dt": 0.002},
        {"step": 2, "time": 0.003, "dt": 0.001},
    ]
    corrections = [
        {
            "field": "phi", "step": 1,
            "target_negative_volume": 0.5,
            "initial_negative_volume": 0.51,
            "initial_volume_error": 0.01,
            "corrected_negative_volume": 0.5,
            "achieved_volume_error": 0.0,
            "volume_measure_source": "generated_interface_quadrature",
            "correction_triggered": True,
            "correction_applied": True,
            "applied_shift": 1.0e-4,
        },
        {
            "field": "phi", "step": 2,
            "target_negative_volume": 0.5,
            "initial_negative_volume": 0.49,
            "initial_volume_error": -0.01,
            "corrected_negative_volume": 0.5,
            "achieved_volume_error": 0.0,
            "volume_measure_source": "generated_interface_quadrature",
            "correction_triggered": True,
            "correction_applied": True,
            "applied_shift": -1.0e-4,
        },
    ]
    metrics = {
        "solver_controls": {"transient_solve": {"t0": 0.0}},
        "time_loop": {"accepted_steps": accepted},
        "diagnostics": {"level_set_volume_corrections": corrections},
        "production_wet_volume_diagnostic_history": [
            production_record(0, 0.0),
            production_record(1, 0.002),
            production_record(2, 0.003),
        ],
    }
    runner.add_production_physical_liquid_volume_metrics(metrics)
    runner.add_level_set_mass_correction_history_metrics(metrics, density=2.0)
    args = runner.argparse.Namespace(
        require_level_set_mass_correction_histories=True)
    assert runner.level_set_mass_correction_history_errors(metrics, args) == []
    assert metrics["level_set_mass_correction_history_available"] is True
    uncorrected = metrics["level_set_uncorrected_mass_history"]
    corrected = metrics["level_set_corrected_mass_history"]
    assert [(item["step"], item["time"], item["dt"])
            for item in corrected] == [
                (1, 0.002, 0.002), (2, 0.003, 0.001)]
    assert [item["liquid_mass"] for item in uncorrected] == [1.02, 0.98]
    assert [item["liquid_mass"] for item in corrected] == [1.0, 1.0]
    assert all(item["state_stage"] == "accepted_pre_volume_correction"
               for item in uncorrected)
    assert all(item["state_stage"] == "accepted_post_volume_correction"
               for item in corrected)

    incomplete = dict(metrics)
    incomplete["diagnostics"] = {
        "level_set_volume_corrections": corrections[:1]}
    runner.add_level_set_mass_correction_history_metrics(
        incomplete, density=2.0)
    errors = runner.level_set_mass_correction_history_errors(incomplete, args)
    assert len(errors) == 1
    assert "every accepted step" in errors[0]


def test_corner_and_thin_extrusion_use_interior_tetra_centroid_stencils():
    runner = _load_runner()
    points = runner.np.asarray([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.2, 0.2, 0.2],
    ])
    grid = runner.pv.UnstructuredGrid(
        runner.np.asarray([4, 0, 1, 2, 3]),
        runner.np.asarray([runner.pv.CellType.TETRA], dtype=runner.np.uint8),
        points,
    )
    grid.point_data["GlobalNodeID"] = runner.np.arange(4, dtype=runner.np.int64)
    walls = {
        "wall_left": runner.np.asarray([0, 2]),
        "wall_bottom": runner.np.asarray([0, 1]),
    }
    stencils, errors = runner.inward_cell_centroid_stencils_by_wall(grid, walls)
    assert errors == []
    left_corner = stencils[("wall_left", 0)]
    bottom_corner = stencils[("wall_bottom", 0)]
    assert len(left_corner) == len(bottom_corner) == 1
    assert left_corner[0]["point_indices"] == [0, 1, 2, 3]
    assert left_corner[0]["weights"] == [0.25, 0.25, 0.25, 0.25]
    assert runner.np.allclose(left_corner[0]["centroid"], [0.3, 0.3, 0.05])

    # A one-element-thick extrusion can place every tetra vertex on at least
    # one boundary.  The volume-cell centroid remains a valid interior sample.
    all_boundary = dict(walls)
    all_boundary["wall_other"] = runner.np.asarray([3])
    stencils, errors = runner.inward_cell_centroid_stencils_by_wall(
        grid, all_boundary)
    assert errors == []
    assert len(stencils[("wall_left", 0)]) == 1
    assert len(stencils[("wall_other", 3)]) == 1


def test_wall_centroid_stencil_rejects_unsupported_incident_volume_cell():
    runner = _load_runner()
    points = runner.np.asarray([
        [0.0, 0.0, 0.0], [1.0, 0.0, 0.0],
        [1.0, 1.0, 0.0], [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0], [1.0, 0.0, 1.0],
        [1.0, 1.0, 1.0], [0.0, 1.0, 1.0],
    ])
    grid = runner.pv.UnstructuredGrid(
        runner.np.asarray([8, 0, 1, 2, 3, 4, 5, 6, 7]),
        runner.np.asarray([runner.pv.CellType.HEXAHEDRON],
                          dtype=runner.np.uint8),
        points,
    )
    grid.point_data["GlobalNodeID"] = runner.np.arange(8, dtype=runner.np.int64)
    stencils, errors = runner.inward_cell_centroid_stencils_by_wall(
        grid, {"wall_left": runner.np.asarray([0])})
    assert stencils == {}
    assert any(error["reason"] == "unsupported_incident_cell"
               for error in errors)


def test_wall_centroid_stencil_supports_quad4_q1_reference_center():
    runner = _load_runner()
    points = runner.np.asarray([
        [0.0, 0.0, 0.0],
        [2.0, 0.0, 0.0],
        [1.8, 1.0, 0.0],
        [0.1, 1.0, 0.0],
    ])
    grid = runner.pv.UnstructuredGrid(
        runner.np.asarray([4, 0, 1, 2, 3]),
        runner.np.asarray([runner.pv.CellType.QUAD], dtype=runner.np.uint8),
        points,
    )
    grid.point_data["GlobalNodeID"] = runner.np.asarray(
        [40, 41, 42, 43], dtype=runner.np.int64)
    stencils, errors = runner.inward_cell_centroid_stencils_by_wall(
        grid, {"wall_bottom": runner.np.asarray([0, 1])})
    assert errors == []
    stencil = stencils[("wall_bottom", 0)][0]
    assert stencil["cell_type"] == "Quad4"
    assert stencil["point_indices"] == [0, 1, 2, 3]
    assert stencil["weights"] == [0.25, 0.25, 0.25, 0.25]
    assert runner.np.allclose(stencil["centroid"], points.mean(axis=0))


def test_wall_centroid_stencil_rejects_folded_quad4_mapping():
    runner = _load_runner()
    points = runner.np.asarray([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [1.0, 1.0, 0.0],
    ])
    grid = runner.pv.UnstructuredGrid(
        runner.np.asarray([4, 0, 1, 2, 3]),
        runner.np.asarray([runner.pv.CellType.QUAD], dtype=runner.np.uint8),
        points,
    )
    grid.point_data["GlobalNodeID"] = runner.np.arange(
        4, dtype=runner.np.int64)
    stencils, errors = runner.inward_cell_centroid_stencils_by_wall(
        grid, {"wall_bottom": runner.np.asarray([0])})
    assert stencils == {}
    assert any(error["reason"] == "nondegenerate_cell_interior_unavailable"
               for error in errors)


def test_point_scalar_mapping_accepts_identical_owned_and_ghost_copies():
    runner = _load_runner()
    initial = runner.pv.PolyData(runner.np.asarray([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
    ]))
    initial.point_data["GlobalNodeID"] = runner.np.asarray(
        [10, 11], dtype=runner.np.int64)

    # Put the ghost first to exercise selection of the later owned copy.
    output = runner.pv.PolyData(runner.np.asarray([
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
    ]))
    output.point_data["GlobalNodeID"] = runner.np.asarray(
        [10, 10, 11], dtype=runner.np.int64)
    output.point_data["vtkGhostType"] = runner.np.asarray(
        [1, 0, 0], dtype=runner.np.uint8)
    output.point_data["phi"] = runner.np.asarray([0.25, 0.25, -0.5])

    mapped = runner.point_scalar_in_initial_gid_order(initial, output, "phi")
    assert runner.np.array_equal(mapped, [0.25, -0.5])


def test_point_scalar_mapping_accepts_identical_shared_non_ghost_copies():
    runner = _load_runner()
    initial = runner.pv.PolyData(runner.np.asarray([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
    ]))
    initial.point_data["GlobalNodeID"] = runner.np.asarray(
        [20, 21], dtype=runner.np.int64)

    output = runner.pv.PolyData(runner.np.asarray([
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
    ]))
    output.point_data["GlobalNodeID"] = runner.np.asarray(
        [20, 20, 21], dtype=runner.np.int64)
    output.point_data["vtkGhostType"] = runner.np.zeros(3, dtype=runner.np.uint8)
    output.point_data["phi"] = runner.np.asarray([0.125, 0.125, -0.75])

    mapped = runner.point_scalar_in_initial_gid_order(initial, output, "phi")
    assert runner.np.array_equal(mapped, [0.125, -0.75])


@pytest.mark.parametrize(
    ("duplicate_point", "duplicate_value", "expected_error"),
    [
        ([1.0e-5, 0.0, 0.0], 0.25, "inconsistent coordinates"),
        ([0.0, 0.0, 0.0], 0.5, "inconsistent scalar"),
    ],
)
def test_point_scalar_mapping_rejects_inconsistent_duplicate_copies(
        duplicate_point, duplicate_value, expected_error):
    runner = _load_runner()
    initial = runner.pv.PolyData(runner.np.asarray([[0.0, 0.0, 0.0]]))
    initial.point_data["GlobalNodeID"] = runner.np.asarray(
        [30], dtype=runner.np.int64)

    output = runner.pv.PolyData(runner.np.asarray([
        [0.0, 0.0, 0.0],
        duplicate_point,
    ]))
    output.point_data["GlobalNodeID"] = runner.np.asarray(
        [30, 30], dtype=runner.np.int64)
    output.point_data["vtkGhostType"] = runner.np.asarray(
        [0, 1], dtype=runner.np.uint8)
    output.point_data["phi"] = runner.np.asarray([0.25, duplicate_value])

    with pytest.raises(ValueError, match=expected_error):
        runner.point_scalar_in_initial_gid_order(initial, output, "phi")


def test_point_scalar_mapping_rejects_missing_or_ghost_only_coverage():
    runner = _load_runner()
    initial = runner.pv.PolyData(runner.np.asarray([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
    ]))
    initial.point_data["GlobalNodeID"] = runner.np.asarray(
        [40, 41], dtype=runner.np.int64)

    missing = runner.pv.PolyData(runner.np.asarray([[0.0, 0.0, 0.0]]))
    missing.point_data["GlobalNodeID"] = runner.np.asarray(
        [40], dtype=runner.np.int64)
    missing.point_data["phi"] = runner.np.asarray([0.25])
    with pytest.raises(ValueError, match="output omits"):
        runner.point_scalar_in_initial_gid_order(initial, missing, "phi")

    ghost_only = runner.pv.PolyData(runner.np.asarray([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
    ]))
    ghost_only.point_data["GlobalNodeID"] = runner.np.asarray(
        [40, 41], dtype=runner.np.int64)
    ghost_only.point_data["vtkGhostType"] = runner.np.asarray(
        [1, 0], dtype=runner.np.uint8)
    ghost_only.point_data["phi"] = runner.np.asarray([0.25, -0.5])
    with pytest.raises(ValueError, match="owned coverage is ambiguous"):
        runner.point_scalar_in_initial_gid_order(initial, ghost_only, "phi")


def test_closed_interface_certificate_reads_vtu_wall_surfaces():
    runner = _load_runner()
    with tempfile.TemporaryDirectory() as temp_dir:
        case_dir = Path(temp_dir)
        surface_dir = case_dir / "mesh/background/mesh-surfaces"
        surface_dir.mkdir(parents=True)
        initial = runner.pv.ImageData(
            dimensions=(3, 3, 3),
            spacing=(0.5, 0.5, 0.5),
        )
        points = runner.np.asarray(initial.points, dtype=float)
        gids = runner.np.arange(
            1, initial.n_points + 1, dtype=runner.np.int64)
        initial.point_data["GlobalNodeID"] = gids
        initial.point_data["phi"] = (
            runner.np.linalg.norm(points - 0.5, axis=1) - 0.25)
        wall_masks = {
            "wall_left": runner.np.isclose(points[:, 0], 0.0),
            "wall_right": runner.np.isclose(points[:, 0], 1.0),
            "wall_bottom": runner.np.isclose(points[:, 1], 0.0),
            "wall_top": runner.np.isclose(points[:, 1], 1.0),
            "wall_back": runner.np.isclose(points[:, 2], 0.0),
            "wall_front": runner.np.isclose(points[:, 2], 1.0),
        }
        for name, mask in wall_masks.items():
            surface = runner.pv.PolyData(points[mask])
            surface.point_data["GlobalNodeID"] = gids[mask]
            surface.cast_to_unstructured_grid().save(
                surface_dir / f"{name}.vtu")
        declarations = "".join(
            f'<Add_face name="{name}" />' for name in wall_masks)
        (case_dir / "solver.xml").write_text(
            f"<svMultiPhysicsFile><Add_mesh>{declarations}"
            "</Add_mesh></svMultiPhysicsFile>",
            encoding="utf-8",
        )

        wall_indices = runner.boundary_face_point_indices(case_dir, initial)
        assert set(wall_indices) == set(wall_masks)
        assert len(runner.np.unique(runner.np.concatenate(
            list(wall_indices.values())))) == 26
        evidence = runner.wall_false_wet_applicability(
            case_dir, initial, wall_indices)
        assert evidence["wall_only_false_wet_applicability"] == (
            "not_applicable_closed_interface")
        assert evidence[
            "wall_only_false_wet_closed_interface_certified"] is True
        assert evidence[
            "wall_only_false_wet_boundary_coverage_complete"] is True
        assert evidence["wall_only_false_wet_initial_domain_phi_min"] < 0.0
        assert evidence["wall_only_false_wet_initial_boundary_phi_min"] > 0.0

        incomplete = dict(wall_indices)
        del incomplete["wall_top"]
        evidence = runner.wall_false_wet_applicability(
            case_dir, initial, incomplete)
        assert evidence["wall_only_false_wet_applicability"] == "indeterminate"
        assert evidence[
            "wall_only_false_wet_closed_interface_certified"] is False
        assert evidence["wall_only_false_wet_missing_boundary_names"] == [
            "wall_top"]


def test_wall_centroid_false_wet_uses_global_ids_for_permuted_output():
    runner = _load_runner()
    with tempfile.TemporaryDirectory() as temp_dir:
        case_dir = Path(temp_dir)
        mesh_dir = case_dir / "mesh/background"
        surface_dir = mesh_dir / "mesh-surfaces"
        surface_dir.mkdir(parents=True)
        points = runner.np.asarray([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.2, 0.2, 0.2],
        ])
        cells = runner.np.asarray([4, 0, 1, 2, 3])
        cell_types = runner.np.asarray(
            [runner.pv.CellType.TETRA], dtype=runner.np.uint8)
        initial = runner.pv.UnstructuredGrid(cells, cell_types, points)
        initial.point_data["GlobalNodeID"] = runner.np.asarray(
            [10, 11, 12, 13], dtype=runner.np.int64)
        initial.point_data["phi"] = runner.np.full(4, 0.2)
        initial.save(mesh_dir / "mesh-complete.mesh.vtu")
        wall = runner.pv.PolyData(points[[0]])
        wall.point_data["GlobalNodeID"] = runner.np.asarray(
            [10], dtype=runner.np.int64)
        wall.save(surface_dir / "wall_left.vtp")
        (case_dir / "solver.xml").write_text(
            "<svMultiPhysicsFile><GeneralSimulationParameters>"
            "<Time_step_size>0.1</Time_step_size>"
            "</GeneralSimulationParameters></svMultiPhysicsFile>",
            encoding="utf-8",
        )

        permutation = runner.np.asarray([2, 0, 3, 1])
        inverse = runner.np.empty(4, dtype=runner.np.int64)
        inverse[permutation] = runner.np.arange(4)
        output = runner.pv.UnstructuredGrid(
            runner.np.asarray([4, *inverse.tolist()]),
            cell_types,
            points[permutation],
        )
        output.point_data["GlobalNodeID"] = runner.np.asarray(
            [10, 11, 12, 13], dtype=runner.np.int64)[permutation]
        output_phi = runner.np.asarray([-0.2, 0.4, 0.4, 0.4])
        output.point_data["phi"] = output_phi[permutation]
        output.save(case_dir / "result_001.vtu")

        metrics = {}
        runner.add_physical_time_history_metrics(
            metrics,
            case_dir,
            {},
            initial,
            accepted_steps=[{"step": 1, "time": 0.1, "dt": 0.1}],
        )
        assert metrics["wall_inward_cell_centroid_stencil_complete"] is True
        event = metrics["first_wall_only_false_wet"]
        assert event["global_node_id"] == 10
        assert event["inward_cell_centroid_candidate_count"] == 1
        assert math.isclose(event["inward_cell_centroid_phi_min"], 0.25)
        assert math.isclose(event["inward_cell_centroid_phi_max"], 0.25)
        assert metrics["wall_only_false_wet_history"] == [
            {"step": 1, "time": 0.1, "dt": 0.1, "count": 1}]

        broken = output.copy()
        broken.point_data["GlobalNodeID"] = runner.np.asarray(
            [10, 11, 12, 99], dtype=runner.np.int64)
        with pytest.raises(ValueError, match="output omits"):
            runner.point_scalar_in_initial_gid_order(initial, broken, "phi")


def test_fs16_p1_decks_retain_transport_safety_controls():
    runner = _load_runner()
    expected = {
        "SUPG_transient_scale": "2.0",
        "Enable_discontinuity_capturing": "true",
        "Discontinuity_capturing_scale": "0.1",
        "Discontinuity_capturing_gradient_epsilon": "1.0e-12",
        "Discontinuity_capturing_max_courant": "0.5",
        "Enable_bound_preserving_limiter": "true",
        "Bound_preserving_bound_tolerance": (
            f"{runner.DEFAULT_BOUND_REPRESENTABILITY_SLACK:.16g}"
        ),
        "Bound_preserving_sign_tolerance": (
            f"{runner.BOUND_SIGN_TOLERANCE:.16g}"
        ),
        "Bound_preserving_maximum_courant": "1.0",
        "Bound_preserving_enforce_courant_limit": "true",
        "Bound_preserving_enforce_impermeable_boundaries": "true",
        "Bound_preserving_impermeable_normal_velocity_tolerance": "1.0e-10",
    }
    with tempfile.TemporaryDirectory() as temp_dir:
        base = Path(temp_dir)
        sessile = base / "sessile"
        runner.write_sessile2d_case(
            sessile, 1, 8, 8, 90.0, 90.0, 0.3, 1.0, 0.001,
            1.0, 0.1, False)
        wave = base / "wave"
        runner.write_capillary_wave2d_case(wave, 1, 50.0, 0.001, 8, 8)
        runner.configure_solver(
            wave / "solver.xml",
            steps=1,
            time_step_size=0.001,
            wet_extension_advection_velocity_method="wall_compatible_normal",
        )
        for case_dir in (sessile, wave):
            root = runner.ET.parse(case_dir / "solver.xml").getroot()
            level_set = runner.level_set_equation(root)
            assert {
                name: level_set.findtext(name)
                for name in expected
            } == expected
        assert (
            runner.DEFAULT_BOUND_REPRESENTABILITY_SLACK /
            runner.LEVEL_SET_CHARACTERISTIC_LENGTH
            == runner.MAX_BOUND_REPRESENTABILITY_SLACK_OVER_LENGTH
            == 1.0e-6
        )
        assert runner.BOUND_SIGN_TOLERANCE == 1.0e-12
        sessile_root = runner.ET.parse(sessile / "solver.xml").getroot()
        sessile_level_set = runner.level_set_equation(sessile_root)
        assert sessile_level_set.findtext("Level_set_source") == "prescribed_data"
        assert sessile_level_set.findtext("Velocity_source") == "coupled_field"
        assert sessile_level_set.find("Constant_velocity") is None
        assert sessile_level_set.find("Enable_curvature_projection") is None
        assert sessile_level_set.find("Projected_curvature_field") is None
        assert sessile_level_set.find(
            "Curvature_projection_narrow_band_width") is None

        sessile_fluid = runner.fluid_equation(sessile_root)
        bottom = next(
            bc for bc in sessile_fluid.findall("Add_BC")
            if bc.attrib.get("name") == "wall_bottom"
        )
        assert bottom.findtext("Effective_direction") == "0 1"
        sessile_free_surface = runner.free_surface_bc(sessile_root)
        assert (
            sessile_free_surface.findtext("Contact_line_model")
            == "DynamicContactAngle"
        )
        assert sessile_free_surface.findtext("Wall_slip_model") == "Navier"
        assert sessile_free_surface.findtext("Contact_line_mobility") == "1"
        assert sessile_free_surface.findtext("Wall_slip_length") == "0.1"
        assert sessile_free_surface.findtext(
            "Active_domain_smoothing_width"
        ) == "0"
        assert sessile_free_surface.find("Contact_angle_penalty") is None

        sessile_mesh = runner.pv.read(
            sessile / "mesh/background/mesh-complete.mesh.vtu")
        assert runner.np.allclose(
            runner.np.asarray(sessile_mesh.point_data["Pressure"]),
            1.0 / 0.3,
        )
        sessile_benchmark = runner.load_benchmark(sessile)
        assert (
            sessile_benchmark["sessile_contact"]["contact_line_model"]
            == "DynamicContactAngle"
        )
        assert (
            sessile_benchmark["sessile_contact"]
            ["curvature_projection_narrow_band_width"]
            == 0.125
        )
        contract = runner.capillary_wave_boundary_contract_metrics(wave)
        assert contract["capillary_wave_boundary_contract_valid"] is True
        assert contract["capillary_wave_bottom_effective_direction"] == [0.0, 1.0]
        assert contract["capillary_wave_dry_top_boundary_type"] == (
            "LevelSetOutflow"
        )
        assert (
            contract["capillary_wave_wet_extension_method"]
            == "wall_compatible_normal"
        )

        wave_mesh = runner.pv.read(
            wave / "mesh/background/mesh-complete.mesh.vtu")
        points = runner.np.asarray(wave_mesh.points, dtype=float)
        pressure = runner.np.asarray(
            wave_mesh.point_data["Pressure"], dtype=float)
        surface_nodes = runner.np.flatnonzero(
            runner.np.isclose(points[:, 1], runner.CAPILLARY_WAVE_DEPTH)
        )
        k = runner.capillary_wave_wavenumber()
        expected_surface_pressure = (
            50.0 * runner.CAPILLARY_WAVE_AMPLITUDE * k ** 2 *
            runner.np.cos(k * points[surface_nodes, 0])
        )
        assert runner.np.allclose(
            pressure[surface_nodes], expected_surface_pressure,
            rtol=1.0e-13, atol=1.0e-13)
        wave_benchmark = runner.load_benchmark(wave)["capillary_wave"]
        assert wave_benchmark["depth"] == runner.CAPILLARY_WAVE_DEPTH
        assert runner.math.isclose(
            wave_benchmark["finite_depth_factor"],
            runner.math.tanh(k * runner.CAPILLARY_WAVE_DEPTH),
            rel_tol=0.0,
            abs_tol=1.0e-15,
        )
        benchmark = runner.load_benchmark(wave)
        gauge = benchmark["pressure_gauge"]
        gauge_node = int(gauge["node_id"])
        assert runner.math.isclose(
            gauge["expected_initial_capillary_pressure"],
            float(pressure[gauge_node]),
            rel_tol=0.0,
            abs_tol=1.0e-14,
        )
        assert gauge["expected_initial_hydrostatic_pressure_component"] == 0.0


def test_surface_stress_droplet_omits_curvature_traction_state_by_default():
    runner = _load_runner()

    def droplet_args(projected_field=None, band_width=None):
        return runner.argparse.Namespace(
            high_order_capillary_droplet_equilibrium_smoke=True,
            high_order_3d_benchmark_qualification=False,
            high_order_3d_benchmark_profile_qualification=False,
            high_order_curved_3d_simplex_smoke=False,
            high_order_mpi_motion_smoke=False,
            high_order_capillary_projection_smoke=False,
            high_order_capillary_response_smoke=False,
            high_order_capillary_balance_smoke=False,
            case=None,
            steps=None,
            timeout_seconds=None,
            synthetic_nx=32,
            synthetic_ny=16,
            projected_curvature_field=projected_field,
            curvature_projection_narrow_band_width=band_width,
            min_max_speed=1.0e-2,
            min_wet_mean_speed=2.5e-4,
            min_gate_mean_ux=1.0e-4,
            min_front_mean_ux=1.0e-4,
        )

    args = droplet_args()
    runner.apply_high_order_capillary_droplet_equilibrium_smoke_defaults(args)

    assert args.projected_curvature_field is None
    assert args.require_curvature_projection_diagnostics is False
    assert args.require_curvature_projection_newton_freshness is False
    assert args.curvature_projection_narrow_band_width is None
    assert getattr(args, "max_capillary_curvature_relative_error", None) is None

    # A caller may still request kappa as an output-only diagnostic.  Its
    # narrow band remains mesh scaled, but it never becomes a SurfaceStress
    # Newton-freshness or force-accuracy gate.
    diagnostic = droplet_args(projected_field="kappa_projected")
    runner.apply_high_order_capillary_droplet_equilibrium_smoke_defaults(diagnostic)
    assert diagnostic.curvature_projection_narrow_band_width == 1.0 / 32.0
    assert diagnostic.require_curvature_projection_diagnostics is True
    assert diagnostic.require_curvature_projection_newton_freshness is False
    assert getattr(diagnostic, "max_capillary_curvature_relative_error", None) is None

    overridden = droplet_args(
        projected_field="kappa_projected", band_width=0.2)
    runner.apply_high_order_capillary_droplet_equilibrium_smoke_defaults(overridden)
    assert overridden.curvature_projection_narrow_band_width == 0.2

    with tempfile.TemporaryDirectory() as temp_dir:
        case_dir = Path(temp_dir) / "droplet2d"
        runner.write_capillary_droplet2d_case(
            case_dir,
            steps=diagnostic.steps,
            pressure_jump=(
                diagnostic.surface_tension / runner.CAPILLARY_DROPLET_RADIUS),
            nx=diagnostic.synthetic_nx,
            ny=diagnostic.synthetic_ny,
        )
        runner.configure_solver(
            case_dir / "solver.xml",
            diagnostic.steps,
            surface_tension=diagnostic.surface_tension,
            projected_curvature_field=diagnostic.projected_curvature_field,
            curvature_projection_narrow_band_width=(
                diagnostic.curvature_projection_narrow_band_width
            ),
        )
        root = runner.ET.parse(case_dir / "solver.xml").getroot()
        level_set = runner.level_set_equation(root)
        free_surface = runner.free_surface_bc(root)
        assert level_set.findtext(
            "Curvature_projection_narrow_band_width"
        ) == "0.03125"
        assert free_surface.findtext("Surface_tension_form") == "SurfaceStress"
        assert free_surface.find("Curvature_field") is None


def test_generated_curvature_traction_configuration_is_explicit():
    runner = _load_runner()
    with tempfile.TemporaryDirectory() as temp_dir:
        case_dir = Path(temp_dir) / "droplet2d"
        runner.write_capillary_droplet2d_case(
            case_dir,
            steps=1,
            pressure_jump=2.0,
            nx=8,
            ny=8,
        )
        solver = case_dir / "solver.xml"
        runner.configure_solver(
            solver,
            steps=1,
            surface_tension=0.5,
            capillary_force_form="generated_curvature_traction",
            prescribed_capillary_curvature=4.0,
        )
        root = runner.ET.parse(solver).getroot()
        free_surface = runner.free_surface_bc(root)
        assert free_surface.findtext(
            "Surface_tension_form") == "GeneratedCurvatureTraction"
        assert free_surface.findtext("Curvature") == "4"
        assert free_surface.findtext("Use_level_set_curvature") == "false"
        assert free_surface.find("Curvature_field") is None


def test_projected_curvature_configuration_exposes_recovery_controls():
    runner = _load_runner()
    with tempfile.TemporaryDirectory() as temp_dir:
        case_dir = Path(temp_dir) / "droplet2d"
        runner.write_capillary_droplet2d_case(
            case_dir,
            steps=1,
            pressure_jump=2.0,
            nx=8,
            ny=8,
        )
        solver = case_dir / "solver.xml"
        runner.configure_solver(
            solver,
            steps=1,
            surface_tension=0.5,
            capillary_force_form="generated_curvature_traction",
            projected_curvature_field="kappa_projected",
            curvature_projection_supplemental_sample_weight=0.125,
            curvature_projection_recovery_mode="generated_interface_patch",
        )

        root = runner.ET.parse(solver).getroot()
        level_set = runner.level_set_equation(root)
        free_surface = runner.free_surface_bc(root)
        assert level_set.findtext(
            "Curvature_projection_supplemental_sample_weight") == "0.125"
        assert level_set.findtext(
            "Curvature_projection_recovery_mode") == "generated_interface_patch"
        assert free_surface.findtext("Curvature_field") == "kappa_projected"


def test_kinematic_area_gradient_traction_configuration_is_explicit_and_unfiltered():
    runner = _load_runner()
    with tempfile.TemporaryDirectory() as temp_dir:
        case_dir = Path(temp_dir) / "droplet2d"
        runner.write_capillary_droplet2d_case(
            case_dir,
            steps=1,
            pressure_jump=2.0,
            nx=8,
            ny=8,
        )
        solver = case_dir / "solver.xml"
        runner.configure_solver(
            solver,
            steps=1,
            surface_tension=0.5,
            capillary_force_form="kinematic_area_gradient_traction",
            projected_curvature_field="kappa_area_gradient",
            curvature_projection_recovery_mode="kinematic_area_gradient",
            curvature_projection_kinematic_area_gradient_filter_coefficient=0.0,
            curvature_projection_smoothing_iterations=0,
            cut_cell_pressure_stabilization_policy="incremental",
            static_capillary_finite_difference_relative_step=2.0e-5,
            static_capillary_limited_memory_history_size=6,
            static_capillary_limited_memory_curvature_tolerance=3.0e-11,
            static_capillary_max_topology_epoch_transitions=24,
        )

        root = runner.ET.parse(solver).getroot()
        level_set = runner.level_set_equation(root)
        free_surface = runner.free_surface_bc(root)
        assert free_surface.findtext(
            "Surface_tension_form") == "KinematicAreaGradientTraction"
        assert free_surface.findtext("Interface_quadrature_order") == "2"
        assert free_surface.findtext(
            "Curvature_field") == "kappa_area_gradient"
        assert free_surface.findtext("Use_level_set_curvature") == "false"
        assert free_surface.findtext(
            "Cut_cell_pressure_stabilization_policy") == "Incremental"
        assert level_set.findtext(
            "Curvature_projection_recovery_mode") == "kinematic_area_gradient"
        assert level_set.findtext(
            "Curvature_projection_kinematic_area_gradient_filter_coefficient"
        ) == "0"
        assert level_set.findtext(
            "Curvature_projection_smoothing_iterations") == "0"
        assert level_set.findtext(
            "Static_capillary_finite_difference_relative_step") == "2e-05"
        assert level_set.findtext(
            "Static_capillary_limited_memory_history_size") == "6"
        assert level_set.findtext(
            "Static_capillary_limited_memory_curvature_tolerance") == "3e-11"
        assert level_set.findtext(
            "Static_capillary_max_topology_epoch_transitions") == "24"

        with pytest.raises(ValueError, match="quadrature order at least two"):
            runner.configure_solver(
                solver,
                steps=1,
                surface_tension=0.5,
                capillary_force_form="kinematic_area_gradient_traction",
                interface_quadrature_order=1,
                projected_curvature_field="kappa_area_gradient",
                curvature_projection_recovery_mode="kinematic_area_gradient",
                curvature_projection_kinematic_area_gradient_filter_coefficient=0.0,
            )


def test_static_capillary_minimizer_configuration_rejects_invalid_controls():
    runner = _load_runner()

    def configure(**options):
        with tempfile.TemporaryDirectory() as temp_dir:
            case_dir = Path(temp_dir) / "droplet2d"
            runner.write_capillary_droplet2d_case(
                case_dir,
                steps=1,
                pressure_jump=2.0,
                nx=8,
                ny=8,
            )
            runner.configure_solver(
                case_dir / "solver.xml",
                steps=1,
                **options,
            )

    with pytest.raises(ValueError, match="relative step must be positive"):
        configure(static_capillary_finite_difference_relative_step=0.0)
    with pytest.raises(ValueError, match="history size must be a nonnegative integer"):
        configure(static_capillary_limited_memory_history_size=-1)
    with pytest.raises(ValueError, match="history size must be a nonnegative integer"):
        configure(static_capillary_limited_memory_history_size=2.5)
    with pytest.raises(ValueError, match="history size must be a nonnegative integer"):
        configure(static_capillary_limited_memory_history_size=True)
    with pytest.raises(ValueError, match="curvature tolerance must be positive"):
        configure(static_capillary_limited_memory_curvature_tolerance=0.0)
    with pytest.raises(ValueError, match="topology epoch transitions"):
        configure(static_capillary_max_topology_epoch_transitions=-1)
    with pytest.raises(ValueError, match="topology epoch transitions"):
        configure(static_capillary_max_topology_epoch_transitions=2.5)
    with pytest.raises(ValueError, match="topology epoch transitions"):
        configure(static_capillary_max_topology_epoch_transitions=True)
    with pytest.raises(ValueError, match="pressure stabilization policy"):
        configure(cut_cell_pressure_stabilization_policy="unsupported")


def test_kinematic_area_gradient_static_initialization_requires_exact_derivatives():
    runner = _load_runner()
    record = {
        "active_coefficients": 81,
        "functional_evaluations": 326,
        "acceptance_certificate_evaluations": 1,
        "finite_difference_fourth_order_components": 0,
        "analytic_derivative_evaluations": 135,
        "derivative_resolution_step_acceptances": 17,
        "topology_epoch_transitions": 3,
        "max_topology_epoch_transitions": 24,
        "iterations": 134,
        "target_liquid_volume": 0.141,
        "initial_physical_potential_energy": 0.471,
        "final_physical_potential_energy": 0.470,
        "final_volume_error": -2.8e-17,
        "final_projected_gradient_norm": 8.1e-12,
        "pressure_representability_available": 1,
        "pressure_representability_converged": 1,
        "pressure_representability_breakdown": 0,
        "pressure_representability_residual_norm": 1.4e-11,
        "pressure_representability_relative_distance": 1.8e-11,
        "production_force_projection_applied": 0,
        "production_residual_norm": 2.4e-11,
        "constant_pressure_kkt_required": 0,
        "constant_pressure_kkt_available": 0,
        "qualification": "prerequisite_only",
    }
    args = runner.argparse.Namespace(
        initialize_discrete_static_capillary_equilibrium=True,
        capillary_force_form="kinematic_area_gradient_traction",
        static_capillary_max_iterations=200,
        static_capillary_max_topology_epoch_transitions=24,
    )

    metrics = {
        "diagnostics": {
            "static_capillary_equilibrium_initializations": [record],
        },
    }
    assert runner.static_capillary_equilibrium_initialization_errors(
        metrics, args) == []

    extracted = {}
    runner.add_diagnostic_metrics(extracted, metrics["diagnostics"])
    assert extracted["static_capillary_analytic_derivative_evaluations"] == 135
    assert extracted[
        "static_capillary_finite_difference_fourth_order_components"] == 0
    assert extracted[
        "static_capillary_derivative_resolution_step_acceptances"] == 17
    assert extracted["static_capillary_topology_epoch_transitions"] == 3
    assert extracted["static_capillary_max_topology_epoch_transitions"] == 24

    missing_exact = dict(record, analytic_derivative_evaluations=0)
    metrics["diagnostics"][
        "static_capillary_equilibrium_initializations"] = [missing_exact]
    errors = runner.static_capillary_equilibrium_initialization_errors(
        metrics, args)
    assert any("did not use exact functional derivatives" in error
               for error in errors)

    used_differences = dict(
        record, finite_difference_fourth_order_components=1)
    metrics["diagnostics"][
        "static_capillary_equilibrium_initializations"] = [used_differences]
    errors = runner.static_capillary_equilibrium_initialization_errors(
        metrics, args)
    assert any("used finite-difference derivative components" in error
               for error in errors)


def test_kinematic_area_gradient_sessile_mesh_uses_affine_triangles():
    runner = _load_runner()
    with tempfile.TemporaryDirectory() as temp_dir:
        case_dir = Path(temp_dir) / "sessile2d"
        runner.write_sessile2d_case(
            case_dir,
            steps=1,
            nx=8,
            ny=8,
            initial_angle_degrees=90.0,
            equilibrium_angle_degrees=90.0,
            radius=0.3,
            surface_tension=0.5,
            time_step_size=1.0e-3,
            mobility=1.0,
            slip_length=0.1,
            dynamic=False,
            contact_line_model="prescribed",
            simplex_mesh=True,
        )

        mesh = runner.pv.read(
            case_dir / "mesh/background/mesh-complete.mesh.vtu")
        assert mesh.n_cells == 2 * 8 * 8
        assert runner.np.all(
            runner.np.asarray(mesh.celltypes) == runner.pv.CellType.TRIANGLE)
        assert runner.np.array_equal(
            runner.np.asarray(mesh.cell_data["GlobalElementID"]),
            runner.np.arange(mesh.n_cells),
        )
        mesh_record = runner.load_benchmark(case_dir)["mesh_resolution"]
        assert mesh_record["cell_type"] == "Triangle3"
        assert mesh_record["cell_count"] == mesh.n_cells
        state = runner.sessile_state_metrics(
            mesh, runner.load_benchmark(case_dir))
        assert state["operator_contact_geometry_available"] is True
        assert state["operator_contact_geometry_sample_count"] == 2
        assert {
            sample["cell_type"]
            for sample in state["operator_contact_geometry_samples"]
        } == {"Triangle3"}


def test_kinematic_area_gradient_traction_rejects_incompatible_projection_controls():
    runner = _load_runner()

    def configure(**overrides):
        with tempfile.TemporaryDirectory() as temp_dir:
            case_dir = Path(temp_dir) / "droplet2d"
            runner.write_capillary_droplet2d_case(
                case_dir,
                steps=1,
                pressure_jump=2.0,
                nx=8,
                ny=8,
            )
            options = {
                "steps": 1,
                "surface_tension": 0.5,
                "capillary_force_form": "kinematic_area_gradient_traction",
                "projected_curvature_field": "kappa_area_gradient",
                "curvature_projection_recovery_mode":
                    "kinematic_area_gradient",
                "curvature_projection_kinematic_area_gradient_filter_coefficient":
                    0.0,
                "curvature_projection_smoothing_iterations": 0,
            }
            options.update(overrides)
            runner.configure_solver(case_dir / "solver.xml", **options)

    with pytest.raises(ValueError, match="explicit projected curvature field"):
        configure(projected_curvature_field=None)
    with pytest.raises(ValueError, match="requires the kinematic_area_gradient"):
        configure(curvature_projection_recovery_mode="level_set_quadratic")
    with pytest.raises(ValueError, match="explicit zero"):
        configure(
            curvature_projection_kinematic_area_gradient_filter_coefficient=0.25
        )
    with pytest.raises(ValueError, match="does not admit separate"):
        configure(curvature_projection_smoothing_iterations=1)


def test_reinitialization_controls_are_explicit_and_fail_closed():
    runner = _load_runner()
    with tempfile.TemporaryDirectory() as temp_dir:
        case_dir = Path(temp_dir) / "droplet2d"
        runner.write_capillary_droplet2d_case(
            case_dir,
            steps=4,
            pressure_jump=2.0,
            nx=8,
            ny=8,
        )
        solver_xml = case_dir / "solver.xml"
        runner.configure_solver(
            solver_xml,
            steps=4,
            enable_reinitialization=True,
            reinitialization_cadence_steps=2,
        )
        level_set = runner.level_set_equation(
            runner.ET.parse(solver_xml).getroot())
        assert level_set.findtext("Enable_reinitialization") == "true"
        assert level_set.findtext("Reinitialization_cadence_steps") == "2"

        with pytest.raises(ValueError, match="cadence must be a positive integer"):
            runner.configure_solver(
                solver_xml,
                steps=4,
                enable_reinitialization=True,
                reinitialization_cadence_steps=0,
            )
        with pytest.raises(ValueError, match="enable control must be boolean"):
            runner.configure_solver(
                solver_xml,
                steps=4,
                enable_reinitialization="true",
            )

        controls = {}

        class ControlArguments:
            enable_level_set_reinitialization = False
            reinitialization_cadence_steps = 3

            def __getattr__(self, _name):
                return None

        runner.add_solver_control_overrides(
            controls,
            ControlArguments(),
        )
        assert controls["enable_level_set_reinitialization"] is False
        assert controls["reinitialization_cadence_steps"] == 3


def test_projected_curvature_recovery_diagnostics_are_aggregated_and_gated():
    runner = _load_runner()
    diagnostics = runner.parse_solver_diagnostics(
        "[svMultiPhysics::Application] Level-set curvature projected "
        "field='phi' curvature_field='kappa_projected' reason=initial "
        "cache=miss cut_signature_cache=miss recovery_mode=generated_interface_patch "
        "generated_interface_geometry_samples=12 "
        "generated_interface_patch_fitted_vertices=7 "
        "generated_interface_patch_expanded_vertices=2 fitted_vertices=7"
    )
    metrics = {}
    runner.add_diagnostic_metrics(metrics, diagnostics)

    assert metrics[
        "diagnostic_curvature_projection_recovery_mode_counts"
    ] == {"generated_interface_patch": 1}
    assert metrics[
        "diagnostic_curvature_projection_max_interface_geometry_samples"
    ] == 12
    assert metrics[
        "diagnostic_curvature_projection_max_interface_patch_fitted_vertices"
    ] == 7
    assert metrics[
        "diagnostic_curvature_projection_max_interface_patch_expanded_vertices"
    ] == 2

    class OptionalArguments:
        def __getattr__(self, _name):
            return None

    args = OptionalArguments()
    args.require_curvature_projection_diagnostics = True
    args.expect_curvature_projection_recovery_mode = (
        "generated_interface_patch"
    )
    args.min_diagnostic_curvature_projection_interface_geometry_samples = 12
    args.min_diagnostic_curvature_projection_interface_patch_fitted_vertices = 7
    assert runner.curvature_projection_errors(metrics, args) == []

    args.expect_curvature_projection_recovery_mode = "level_set_quadratic"
    args.min_diagnostic_curvature_projection_interface_geometry_samples = 13
    args.min_diagnostic_curvature_projection_interface_patch_fitted_vertices = 8
    errors = runner.curvature_projection_errors(metrics, args)
    assert any("does not include level_set_quadratic" in error for error in errors)
    assert any("geometry samples 12 is below 13" in error for error in errors)
    assert any("fitted vertices 7 is below 8" in error for error in errors)


def test_generated_curvature_traction_requires_one_curvature_source():
    runner = _load_runner()
    with tempfile.TemporaryDirectory() as temp_dir:
        case_dir = Path(temp_dir) / "droplet2d"
        runner.write_capillary_droplet2d_case(
            case_dir,
            steps=1,
            pressure_jump=2.0,
            nx=8,
            ny=8,
        )
        solver = case_dir / "solver.xml"
        with pytest.raises(ValueError, match="requires either"):
            runner.configure_solver(
                solver,
                steps=1,
                surface_tension=0.5,
                capillary_force_form="generated_curvature_traction",
            )
        with pytest.raises(ValueError, match="exactly one curvature source"):
            runner.configure_solver(
                solver,
                steps=1,
                surface_tension=0.5,
                capillary_force_form="generated_curvature_traction",
                prescribed_capillary_curvature=4.0,
                projected_curvature_field="kappa_projected",
            )


def test_sessile_case_uses_monolithic_fsils_gmres_linear_budget():
    runner = _load_runner()
    args = runner.argparse.Namespace(
        high_order_mpi_production_qualification=False,
        steps=None,
        time_step_size=None,
        timeout_seconds=None,
        surface_tension=None,
        min_max_speed=1.0e-2,
        min_wet_mean_speed=2.5e-4,
        min_gate_mean_ux=1.0e-4,
        min_front_mean_ux=1.0e-4,
        enable_physical_history_instrumentation=False,
        require_time_loop_convergence=False,
        disable_velocity_extension=False,
        linear_solver_type=None,
        linear_algebra_backend=None,
        linear_preconditioner=None,
        linear_max_iterations=None,
        linear_krylov_space_dimension=None,
        linear_relative_tolerance=None,
        linear_absolute_tolerance=None,
    )

    configured = runner.case_args_for_run("sessile2d", args)

    assert configured.linear_solver_type == "gmres"
    assert configured.linear_algebra_backend == "fsils"
    assert configured.linear_preconditioner == "rcs"
    assert configured.linear_max_iterations == 100
    assert configured.linear_krylov_space_dimension == 50
    assert configured.linear_relative_tolerance == 1.0e-8
    assert configured.linear_absolute_tolerance == 1.0e-10

    area_gradient_args = runner.argparse.Namespace(**vars(args))
    area_gradient_args.capillary_force_form = (
        "kinematic_area_gradient_traction")
    area_gradient_args.projected_curvature_field = "kappa_area_gradient"
    area_gradient_args.initialize_discrete_static_capillary_equilibrium = True
    area_gradient = runner.case_args_for_run(
        "sessile2d", area_gradient_args)
    assert area_gradient.curvature_projection_recovery_mode == (
        "kinematic_area_gradient")
    assert (
        area_gradient
        .curvature_projection_kinematic_area_gradient_filter_coefficient
        == 0.0
    )
    assert area_gradient.curvature_projection_smoothing_iterations == 0
    assert area_gradient.cut_cell_pressure_stabilization_policy == "incremental"
    assert area_gradient.require_curvature_projection_diagnostics is True
    assert area_gradient.require_curvature_projection_newton_freshness is True
    assert area_gradient.require_free_surface_conservative_balance is True
    assert (
        area_gradient.require_free_surface_pressure_representability_diagnostic
        is True
    )
    assert area_gradient.initialize_static_compatible_pressure is False

    matrix_args = runner.argparse.Namespace(**vars(area_gradient_args))
    matrix_args.defer_static_physical_gates_to_matrix = True
    matrix_owned = runner.case_args_for_run("sphere3d", matrix_args)
    assert matrix_owned.max_capillary_pressure_jump_relative_error is None
    assert matrix_owned.max_capillary_parasitic_capillary_number is None

    conflicting_matrix_args = runner.argparse.Namespace(**vars(matrix_args))
    conflicting_matrix_args.max_capillary_pressure_jump_relative_error = 0.01
    with pytest.raises(ValueError, match="conflict with per-run thresholds"):
        runner.case_args_for_run("sphere3d", conflicting_matrix_args)
    with pytest.raises(ValueError, match="require a static capillary case"):
        runner.case_args_for_run("dynamiccontact2d", matrix_args)

    droplet_area_gradient = runner.case_args_for_run(
        "droplet2d", area_gradient_args)
    assert droplet_area_gradient.curvature_projection_recovery_mode == (
        "kinematic_area_gradient")
    assert droplet_area_gradient.curvature_projection_smoothing_iterations == 0
    assert (
        droplet_area_gradient.cut_cell_pressure_stabilization_policy ==
        "incremental"
    )
    assert droplet_area_gradient.require_curvature_projection_diagnostics is True
    assert droplet_area_gradient.require_curvature_projection_newton_freshness is True

    high_order_area_gradient_args = runner.argparse.Namespace(
        **vars(area_gradient_args))
    high_order_area_gradient_args.use_high_order_implicit_cuts = True
    with pytest.raises(ValueError, match="affine P1 simplex mesh"):
        runner.case_args_for_run(
            "droplet2d", high_order_area_gradient_args)

    candidate_args = runner.argparse.Namespace(**vars(args))
    candidate_args.capillary_force_form = "generated_curvature_traction"
    candidate_args.prescribed_capillary_curvature = 1.0 / 0.3
    candidate_args.initialize_discrete_static_capillary_equilibrium = None
    candidate = runner.case_args_for_run("sessile2d", candidate_args)
    assert candidate.require_free_surface_conservative_balance is False
    assert (
        candidate.require_free_surface_pressure_representability_diagnostic
        is False
    )
    assert candidate.initialize_static_compatible_pressure is False
    assert (
        candidate.max_free_surface_pressure_representability_relative_distance
        is None
    )

    candidate_args.initialize_discrete_static_capillary_equilibrium = True
    with pytest.raises(ValueError, match="surface-energy initialization"):
        runner.case_args_for_run("sessile2d", candidate_args)

    candidate_args.initialize_discrete_static_capillary_equilibrium = None
    candidate_args.require_free_surface_conservative_balance = True
    with pytest.raises(ValueError, match="conservative-balance controls"):
        runner.case_args_for_run("sessile2d", candidate_args)

    candidate_args.require_free_surface_conservative_balance = False
    candidate_args.max_free_surface_pressure_representability_relative_distance = 0.1
    with pytest.raises(ValueError, match="pressure-representability controls"):
        runner.case_args_for_run("sessile2d", candidate_args)

    candidate_args.max_free_surface_pressure_representability_relative_distance = None
    candidate_args.initialize_discrete_static_contact_geometry = True
    with pytest.raises(ValueError, match="stationary two-dimensional sessile"):
        runner.case_args_for_run("sphere3d", candidate_args)

    # These controls are defaults, not a hard gate: diagnostic runs may
    # still request the legacy BlockSchur route explicitly.
    args.linear_solver_type = "ns"
    args.linear_algebra_backend = "fsils"
    args.linear_preconditioner = "fsils"
    args.linear_krylov_space_dimension = 17
    overridden = runner.case_args_for_run("sessile2d", args)
    assert overridden.linear_solver_type == "ns"
    assert overridden.linear_algebra_backend == "fsils"
    assert overridden.linear_preconditioner == "fsils"
    assert overridden.linear_krylov_space_dimension == 17

    area_gradient_args.cut_cell_pressure_stabilization_policy = "enabled"
    absolute_pressure_penalty = runner.case_args_for_run(
        "sphere3d", area_gradient_args)
    assert (
        absolute_pressure_penalty.cut_cell_pressure_stabilization_policy ==
        "enabled"
    )

    # Qualification JSON copies this parsed control record verbatim under
    # solver_controls.linear_solver, including the actual method selected by
    # SimulationBuilder rather than only the requested CLI spelling.
    diagnostics = runner.parse_solver_diagnostics(
        "SimulationBuilder: linear solver method=gmres "
        "preconditioner=row-column-scaling rel_tol=1e-8 abs_tol=1e-10 "
        "max_iter=100 block_layout=[phi(0:1), Velocity(1:2), Pressure(3:1)]"
    )
    actual = diagnostics["solver_controls"]["linear_solver"]
    assert actual["method"] == "gmres"
    assert actual["preconditioner"] == "row-column-scaling"


def test_surface_stress_sessile_deck_has_no_unused_projected_curvature_state():
    runner = _load_runner()
    with tempfile.TemporaryDirectory() as temp_dir:
        case_dir = Path(temp_dir) / "sessile2d"
        runner.write_sessile2d_case(
            case_dir,
            steps=1,
            nx=8,
            ny=8,
            initial_angle_degrees=90.0,
            equilibrium_angle_degrees=90.0,
            radius=0.3,
            surface_tension=1.0,
            time_step_size=1.0e-3,
            mobility=1.0,
            slip_length=0.1,
            dynamic=False,
        )
        root = runner.ET.parse(case_dir / "solver.xml").getroot()
        level_set = runner.level_set_equation(root)
        free_surface = runner.free_surface_bc(root)

        assert level_set.find("Enable_curvature_projection") is None
        assert level_set.find("Projected_curvature_field") is None
        assert free_surface.findtext("Surface_tension_form") == "SurfaceStress"
        assert free_surface.find("Curvature_field") is None


def test_capillary_wave_uses_monolithic_fsils_gmres_linear_budget():
    runner = _load_runner()
    args = runner.argparse.Namespace(
        high_order_capillary_wave_smoke=True,
        high_order_3d_benchmark_qualification=False,
        high_order_3d_benchmark_profile_qualification=False,
        high_order_curved_3d_simplex_smoke=False,
        high_order_mpi_motion_smoke=False,
        high_order_capillary_projection_smoke=False,
        high_order_capillary_response_smoke=False,
        high_order_capillary_balance_smoke=False,
        high_order_capillary_droplet_equilibrium_smoke=False,
        case=None,
        steps=None,
        time_step_size=None,
        timeout_seconds=None,
        surface_tension=None,
        min_max_speed=1.0e-2,
        min_wet_mean_speed=2.5e-4,
        min_gate_mean_ux=1.0e-4,
        min_front_mean_ux=1.0e-4,
    )

    runner.apply_high_order_capillary_wave_smoke_defaults(args)

    assert args.linear_solver_type == "gmres"
    assert args.linear_algebra_backend == "fsils"
    assert args.linear_preconditioner == "rcs"
    assert args.linear_max_iterations == 100
    assert args.linear_krylov_space_dimension == 50
    assert args.linear_relative_tolerance == 1.0e-8
    assert args.linear_absolute_tolerance == 1.0e-10
    assert args.max_fsils_accepted_true_residual_norm == 1.0e-9


def test_fs16_refinement_acceptance_rejects_a_nonconvergent_physical_metric():
    matrix = _load_matrix_runner()
    required = {
        "capillary_equilibrium": (
            "capillary_pressure_jump_relative_error",
            "max_speed",
        ),
        "capillary_wave": (
            "capillary_wave_omega_relative_error",
            "capillary_wave_profile_relative_error",
        ),
        "dynamic_contact_advancing": (
            "ren_e_contact_fluid_speed_relative_error",
            "ren_e_geometric_speed_relative_error",
        ),
        "dynamic_contact_receding": (
            "ren_e_contact_fluid_speed_relative_error",
            "ren_e_geometric_speed_relative_error",
        ),
        "dynamic_contact_sidewall_advancing": (
            "ren_e_contact_fluid_speed_relative_error",
            "ren_e_geometric_speed_relative_error",
        ),
        "dynamic_contact_sidewall_receding": (
            "ren_e_contact_fluid_speed_relative_error",
            "ren_e_geometric_speed_relative_error",
        ),
    }
    for angle in (60, 90, 120):
        required[f"static_sessile_theta{angle}"] = (
            "sessile_final_contact_angle_absolute_error_degrees",
            "sessile_final_pressure_jump_relative_error",
            "sessile_final_liquid_area_relative_error",
            "sessile_final_parasitic_capillary_number",
        )
    convergence = {}
    for group, metrics in required.items():
        convergence[group] = {}
        for metric in metrics:
            convergence[group][metric] = {
                "samples": [
                    {"resolution": 16, "absolute_value": 0.1},
                    {"resolution": 32, "absolute_value": 0.05},
                ],
                "rates": [{
                    "coarse_resolution": 16,
                    "fine_resolution": 32,
                    "observed_rate": 1.0,
                }],
            }

    accepted = matrix.refinement_acceptance(convergence)
    assert accepted["passed"] is True

    convergence["capillary_wave"]["capillary_wave_omega_relative_error"][
        "rates"
    ][0]["observed_rate"] = -0.5
    rejected = matrix.refinement_acceptance(convergence)
    assert rejected["passed"] is False
    assert any("observed rate -0.5" in error for error in rejected["errors"])


def test_fs16_dynamic_contact_liveness_timeout_scales_with_cell_count():
    matrix = _load_matrix_runner()
    assert matrix.dynamic_contact_timeout_seconds(8) == 300
    assert matrix.dynamic_contact_timeout_seconds(16) == 1200
    assert matrix.dynamic_contact_timeout_seconds(32) == 4800
    with pytest.raises(ValueError, match="resolution must be positive"):
        matrix.dynamic_contact_timeout_seconds(0)


def test_fs16_matrix_contains_separate_vertical_sidewall_dynamic_qualifications():
    matrix = _load_matrix_runner()
    specs = matrix.matrix_specs([16], quick=False)
    by_name = {spec["name"]: spec for spec in specs}
    for motion in ("advancing", "receding"):
        bottom = by_name[f"dynamic_{motion}_n16"]
        side = by_name[f"dynamic_left_wall_{motion}_n16"]
        assert bottom["kind"] == "dynamic_contact"
        assert "--dynamic-contact-wall" not in bottom["args"]
        assert side["kind"] == "dynamic_contact_sidewall"
        wall_option = side["args"].index("--dynamic-contact-wall")
        assert side["args"][wall_option + 1] == "wall_left"
        assert side["args"][side["args"].index("--steps") + 1] == 20
        assert side["args"][side["args"].index("--timeout-seconds") + 1] == 1200


def test_symmetric_dynamic_triad_separates_equilibrium_bias_from_odd_response():
    matrix = _load_matrix_runner()

    def probe(initial, speed, predicted=None):
        value = {
            "benchmark": {
                "mesh_resolution": {"nx": 16},
                "sessile_contact": {
                    "initial_contact_angle_degrees": initial,
                    "equilibrium_contact_angle_degrees": 90.0,
                    "expected_initial_liquid_area": 0.125,
                },
            },
            "sessile_final_contact_fluid_outward_speed": speed,
        }
        if predicted is not None:
            value["ren_e_predicted_final_contact_line_speed"] = predicted
        return value

    results = [
        {"kind": "static_sessile", "probe": probe(90.0, -0.06)},
        {"kind": "dynamic_contact", "probe": probe(95.0, -0.01, 0.08)},
        {"kind": "dynamic_contact", "probe": probe(85.0, -0.11, -0.08)},
    ]
    records = matrix.symmetric_dynamic_response_records(results)

    assert len(records) == 1
    record = records[0]
    assert record["resolution"] == 16
    assert math.isclose(record["pair_even_speed"], -0.06, abs_tol=1.0e-15)
    assert math.isclose(
        record["pair_even_vs_equilibrium_absolute_defect"], 0.0,
        abs_tol=1.0e-15)
    assert math.isclose(record["centered_odd_response"], 0.05, abs_tol=1.0e-15)
    assert math.isclose(
        record["centered_odd_response_relative_error"], 0.375,
        abs_tol=1.0e-15)
    assert math.isclose(
        record["centered_antisymmetry_relative_defect"], 0.0,
        abs_tol=1.0e-15)
    assert record["maximum_reference_area_relative_mismatch"] == 0.0


def test_impermeable_wall_advection_gate_uses_nonperiodic_contact_distance():
    runner = _load_wall_advection_runner()
    assert runner.math.isclose(
        runner.contact_error([0.95], [0.05]), 0.9,
        rel_tol=0.0, abs_tol=1.0e-15)
    assert runner.contact_error([0.35], [0.35]) == 0.0
    assert runner.contact_error([], [0.35]) is None
    bottom = runner.exact_horizontal_wall_contacts(
        runner.FINAL_TIME, 0.0)[0]
    top = runner.exact_horizontal_wall_contacts(
        runner.FINAL_TIME, runner.TANK_HEIGHT)[0]
    assert bottom > runner.FRONT_INITIAL_X
    assert top < runner.FRONT_INITIAL_X


def test_impermeable_wall_advection_gate_emits_fixed_production_contract():
    runner = _load_wall_advection_runner()
    with tempfile.TemporaryDirectory() as temp_dir:
        base = Path(temp_dir)
        case_dir = base / "wall_advection_n4"
        generation = runner.generate_case(case_dir, 4)
        assert generation.returncode == 0, generation.stdout
        contract = runner.deck_contract(case_dir)
        assert contract["passed"] is True, contract["errors"]
        assert contract["boundaries"] == {}
        assert contract["maximum_nodal_wall_normal_velocity"] == 0.0
        assert contract["actual"]["Velocity_source"] == "prescribed_data"
        assert contract["actual"][
            "Bound_preserving_enforce_impermeable_boundaries"
        ] == "true"
        assert float(contract["actual"][
            "Bound_preserving_bound_tolerance"
        ]) == runner.DEFAULT_BOUND_REPRESENTABILITY_SLACK
        assert float(contract["actual"][
            "Bound_preserving_sign_tolerance"
        ]) == runner.BOUND_SIGN_TOLERANCE == 1.0e-12
        assert (
            runner.DEFAULT_BOUND_REPRESENTABILITY_SLACK / runner.LENGTH
            == runner.MAX_BOUND_REPRESENTABILITY_SLACK_OVER_LENGTH
            == 1.0e-6
        )
        assert contract["fixed_time_steps"] == runner.FIXED_TIME_STEPS == 512
        assert contract["fixed_time_step_size"] == (
            runner.FINAL_TIME / runner.FIXED_TIME_STEPS)

        default_case = base / "default_mms"
        default_case.mkdir()
        runner.shutil.copy2(runner.GENERATOR, default_case / "generate_case.py")
        default_generation = runner.subprocess.run(
            [
                runner.sys.executable,
                "generate_case.py",
                "--nx", "2",
                "--ny", "2",
                "--time-steps", "1",
                "--final-time", "0.01",
            ],
            cwd=default_case,
            text=True,
            stdout=runner.subprocess.PIPE,
            stderr=runner.subprocess.STDOUT,
            check=False,
        )
        assert default_generation.returncode == 0, default_generation.stdout
        default_root = runner.ET.parse(default_case / "solver.xml").getroot()
        default_level_set = next(
            equation for equation in default_root.findall("Add_equation")
            if equation.attrib.get("type") == "level_set"
        )
        assert default_level_set.find("Enable_bound_preserving_limiter") is None
        default_fluid = next(
            equation for equation in default_root.findall("Add_equation")
            if equation.attrib.get("type") == "fluid"
        )
        free_surface = next(
            bc for bc in default_fluid.findall("Add_BC")
            if (bc.findtext("Type") or "").strip() == "Free_surface"
        )
        assert free_surface.findtext(
            "Cut_cell_velocity_gradient_penalty") == "0.1"
        assert free_surface.findtext(
            "Cut_cell_velocity_max_derivative_order") == "1"


def test_impermeable_wall_advection_bound_slack_is_numerical_only():
    runner = _load_wall_advection_runner()
    runner.validate_bound_representability_slack(
        runner.DEFAULT_BOUND_REPRESENTABILITY_SLACK, [8, 16])

    for invalid in (-1.0, float("nan"), 1.0001e-6):
        try:
            runner.validate_bound_representability_slack(invalid, [8, 16])
        except ValueError:
            pass
        else:
            raise AssertionError(f"accepted invalid slack {invalid!r}")

    # At very fine h the caller must reduce the numerical slack so it remains
    # two orders of magnitude below the physical dry-wall sign margin.
    try:
        runner.validate_bound_representability_slack(
            runner.DEFAULT_BOUND_REPRESENTABILITY_SLACK, [2048])
    except ValueError as error:
        assert "dry sign margin" in str(error)
    else:
        raise AssertionError("accepted slack too close to the dry sign margin")

    with tempfile.TemporaryDirectory() as temp_dir:
        case_dir = Path(temp_dir) / "wall_advection_n4"
        generation = runner.generate_case(case_dir, 4)
        assert generation.returncode == 0, generation.stdout
        mesh = runner.pv.read(
            case_dir / "mesh/background/mesh-complete.mesh.vtu")
        previous = runner.np.asarray(
            mesh.point_data["phi"], dtype=float).reshape(-1)

        candidate = previous.copy()
        right_corner = int(runner.np.argmax(runner.np.asarray(
            mesh.points)[:, 0] + runner.np.asarray(mesh.points)[:, 1]))
        candidate[right_corner] += 0.5 * (
            runner.DEFAULT_BOUND_REPRESENTABILITY_SLACK)
        within = runner.one_ring_bound_metrics(
            mesh,
            previous,
            candidate,
            runner.DEFAULT_BOUND_REPRESENTABILITY_SLACK,
            runner.BOUND_SIGN_TOLERANCE,
        )
        assert within["strict_one_ring_bound_violation_dofs"] == 1
        assert within["representability_slack_exceeded_dofs"] == 0
        assert within["positive_patch_sign_flips"] == 0
        assert within["negative_patch_sign_flips"] == 0

        candidate[right_corner] = -2.0 * runner.BOUND_SIGN_TOLERANCE
        flipped = runner.one_ring_bound_metrics(
            mesh,
            previous,
            candidate,
            runner.DEFAULT_BOUND_REPRESENTABILITY_SLACK,
            runner.BOUND_SIGN_TOLERANCE,
        )
        assert flipped["representability_slack_exceeded_dofs"] == 1
        assert flipped["positive_patch_sign_flips"] == 1
