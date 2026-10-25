import collections
import importlib.util
from pathlib import Path

MODULE_PATH = Path(__file__).resolve().parent / "cases/fluid/free_surface_log_records.py"
SPEC = importlib.util.spec_from_file_location("free_surface_log_records", MODULE_PATH)
L = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(L)


def test_map_entries_are_complete_and_their_words_unique():
    logmap = L.load_map()
    words = collections.Counter(e["after"]["diagnostic"] for e in logmap.entries)
    assert [w for w, n in words.items() if n > 1] == []
    for e in logmap.entries:
        before = e["before"]
        assert before.get("head") or before.get("head_regex") or before.get("line_regex") or before.get("diagnostic"), e["id"]
        assert e["after"]["owner"] and e["after"]["diagnostic"], e["id"]
        assert set(L.levels(e)) <= {"INFO", "WARNING"}, e["id"]
        assert e["after"]["scope"] in ("root", "every"), e["id"]


def test_reads_a_line_selected_by_its_text_in_both_formats():
    old = ("[svMultiPhysics::Application] Wet volume diagnostic step=0 time=0.0000000000000000e+00 field='phi' "
           "wet_volume=5.0707106781186573e-01 volume_rule_count=272")
    new = ("[12:00:00.000] [R0] [INFO] ApplicationDriver: wet volume diagnostic=wet_volume step=0 time=0 field='phi' "
           "wet_volume=0.5070710678118657 volume_rule_count=272")
    assert L.line_diagnostic(old) == L.line_diagnostic(new) == "wet_volume"
    a, b = L.line_fields(old), L.line_fields(new)
    assert list(a) == list(b) == ["step", "time", "field", "wet_volume", "volume_rule_count"]
    assert all(L.values_equal(a[k], b[k]) for k in a)
    # A line that no entry covers keeps its own word and fields.
    assert L.line_diagnostic("[R0] [INFO] NewtonSolver: x diagnostic=newton_assembly a=1") == "newton_assembly"
    assert L.line_fields("[R0] [INFO] NewtonSolver: x diagnostic=newton_assembly a=1") == {"a": "1"}


def test_repeated_groups_are_child_lines_in_both_formats():
    old = ("[R0] [INFO] FESystem: accepted free-surface discrete functional "
           "diagnostic=free_surface_discrete_functional_history accepted_step=3 interface_marker=7 "
           "liquid_volume=0.26000000000000001 wall_count=2 wall_marker=1 wall_wetted_area=0.5 wall_young_energy=0 "
           "wall_marker=4 wall_wetted_area=0.25 wall_young_energy=-0.1 contact_stage_time=0.5 "
           "contact_wall_marker=4 contact_law=DynamicRenE contact_measure=1")
    new = [
        "[R0] [INFO] FunctionalRecords: accepted free-surface functionals "
        "diagnostic=free_surface_discrete_functional_history accepted_step=3 interface_marker=7 liquid_volume=0.26 "
        "wall_count=2 contact_stage_time=0.5",
        "[R0] [INFO] FunctionalRecords: accepted free-surface functionals diagnostic=free_surface_wall_history "
        "accepted_step=3 interface_marker=7 wall_marker=1 wall_wetted_area=0.5 wall_young_energy=0",
        "[R0] [INFO] FunctionalRecords: accepted free-surface functionals diagnostic=free_surface_wall_history "
        "accepted_step=3 interface_marker=7 wall_marker=4 wall_wetted_area=0.25 wall_young_energy=-0.1",
        "[R0] [INFO] FunctionalRecords: accepted free-surface functionals "
        "diagnostic=free_surface_contact_stage_wall_history accepted_step=3 interface_marker=7 contact_wall_marker=4 "
        "contact_law=DynamicRenE contact_measure=1",
    ]
    (a,) = L.records([old], "free_surface_discrete_functional_history")
    (b,) = L.records(new, "free_surface_discrete_functional_history")
    assert [k for k, _ in a.fields] == [k for k, _ in b.fields]
    assert [c.diagnostic for c in a.children] == [c.diagnostic for c in b.children]
    assert [c.fields for c in a.children] == [c.fields for c in b.children]
    assert L.values_equal(a.get("liquid_volume"), b.get("liquid_volume"))
    # The old shape of the new lines, for comparators written against the old lines.
    (legacy,) = L.legacy_view(new)
    assert legacy.startswith("[R0] [INFO] FESystem: accepted free-surface discrete functional "
                             "diagnostic=free_surface_discrete_functional_history accepted_step=3")
    assert legacy.count("wall_marker=") == 3 and legacy.endswith("contact_law=DynamicRenE contact_measure=1")


def test_reads_the_special_shapes():
    mass = ("[R0] [INFO] FESystem: free-surface functional absolute mass diagnostic=functional_absolute_mass "
            "accepted_step=0 interface_marker=7 free_surface.7.liquid_volume=0.5 free_surface.7.wall.3.wetted_area=2")
    (r,) = L.records([mass])
    assert r.as_dict() == {"accepted_step": "0", "interface_marker": "7", "liquid_volume": "0.5"}
    assert r.children[0].diagnostic == "functional_absolute_mass_wall"
    assert r.children[0].as_dict() == {"accepted_step": "0", "interface_marker": "7", "wall_marker": "3",
                                       "wetted_area": "2"}
    assert L.legacy_line(r).endswith("free_surface.7.liquid_volume=0.5 free_surface.7.wall.3.wetted_area=2")

    points = ("[R0] [INFO] ContactPointVelocityConstraint 'ren_e_contact_point:3': diagnostic=contact_point_velocity "
              "points=2 [x=-0.8 y=0 slave=50] [x=0.8 y=0 slave=686]")
    (r,) = L.records([points])
    assert r.as_dict() == {"constraint": "'ren_e_contact_point:3'", "points": "2"}
    assert [c.as_dict() for c in r.children] == [{"point": "0", "x": "-0.8", "y": "0", "slave": "50"},
                                                 {"point": "1", "x": "0.8", "y": "0", "slave": "686"}]

    warning = ("[svMultiPhysics::Application] WARNING: 'Use_level_set_curvature=false' (Add_equation type=fluid "
               "Add_BC name=free_surface) restates a retired setting (decision D47); it has no effect and will be "
               "removed.")
    assert L.line_diagnostic(warning) == "input_deprecation"
    (r,) = L.records([warning])
    assert r.level == "WARNING"
    assert r.as_dict() == {"setting": "'Use_level_set_curvature=false'",
                           "location": "'Add_equation type=fluid Add_BC name=free_surface'",
                           "decision": "D47", "effect": "none"}


def test_value_comparison_and_shortest_reals():
    assert L.values_equal("0.26000000000000001", "0.26")
    assert L.values_equal("129.81", "129.81019335983763")          # stream default, 6 significant digits
    assert L.values_equal("1.000000", "1")                           # std::to_string
    assert L.values_equal("5.50265214346803738e-04", "0.0005502652143468037")
    assert L.values_equal("1e-8", "1e-08")
    assert not L.values_equal("129.81", "129.82")
    assert not L.values_equal("1.000000", "1.0000005")
    assert not L.values_equal("true", "1")
    for x, text in ((0.26, "0.26"), (1e-05, "1e-05"), (1e20, "1e+20"), (2.0, "2"), (-0.0, "-0"),
                    (0.0005502652143468037, "0.0005502652143468037"), (123456789012345678.0, "123456789012345680"),
                    (1e16, "1e+16"), (0.1 + 0.2, "0.30000000000000004")):
        assert L.cxx_shortest(x) == text
    assert L.lint_line("[R0] [INFO] Owner: m diagnostic=w a=0.26 b=2") == []
    problems = L.lint_line("[R0] [INFO] Owner: m a=0.26000000000000001 diagnostic=w Bad=1 a=2")
    assert "diagnostic is not the first field" in problems
    assert any("not snake_case" in p for p in problems) and any("repeats" in p for p in problems)
    assert any("shortest" in p for p in problems)


def test_params_records_read_the_same_in_both_formats():
    pairs = [
        ("[R0] [INFO] IncompressibleNavierStokesVMSModule: validated complete DynamicContactAngle planar wall "
         "marker wall_boundary_marker=3 global_owned_faces=162 minimum_alignment=1.000000 alignment_tolerance=1e-8 "
         "geometry_contract=first_order_planar_current_physical "
         "diagnostic=dynamic_contact_complete_wall_marker_validation",
         "[12:00:00.000] [R0] [INFO] IncompressibleNavierStokesVMSModule: validated complete DynamicContactAngle "
         "planar wall marker diagnostic=dynamic_contact_complete_wall_marker_validation wall_boundary_marker=3 "
         "global_owned_faces=162 minimum_alignment=1 alignment_tolerance=1e-08 "
         "geometry_contract=first_order_planar_current_physical"),
        ("[svMultiPhysics::Application] Semi-implicit capillary term: surface_tension_semi_implicit=NormalIncrement "
         "reference_field=ns_free_surface_semi_implicit_reference_velocity velocity_field=Velocity "
         "reference_refresh=projected_outer|endpoint|restored|before_solve "
         "diagnostic=free_surface_semi_implicit_reference",
         "[12:00:00.000] [R0] [INFO] ApplicationDriver: semi-implicit capillary term "
         "diagnostic=free_surface_semi_implicit_reference surface_tension_semi_implicit=NormalIncrement "
         "reference_field=ns_free_surface_semi_implicit_reference_velocity velocity_field=Velocity "
         "reference_refresh=projected_outer|endpoint|restored|before_solve"),
        ("[svMultiPhysics::Application] WARNING: 'Use_level_set_curvature=false' (Add_equation type=fluid Add_BC "
         "name=free_surface) restates a retired setting (decision D47); it has no effect and will be removed.",
         "[12:00:00.000] [R0] [WARN] ApplicationDriver: retired input setting diagnostic=input_deprecation "
         "setting='Use_level_set_curvature=false' location='Add_equation type=fluid Add_BC name=free_surface' "
         "decision=D47 effect=none"),
        ("[svMultiPhysics::Application] WARNING: 'Use_ALE' (Add_equation type=fluid) is deprecated (decision D47) "
         "and will be removed; use 'Enable_ALE'.",
         "[12:00:00.000] [R0] [WARN] ApplicationDriver: deprecated input setting diagnostic=input_deprecation "
         "setting='Use_ALE' location='Add_equation type=fluid' decision=D47 replacement='Enable_ALE'"),
        ("[R1] [WARN] IncompressibleNavierStokesVMSModule: skipping active-domain inactive velocity/pressure "
         "constraints because native mesh support is unavailable in this FESystem",
         "[12:00:00.000] [R1] [WARN] IncompressibleNavierStokesVMSModule: active-domain inactive constraints "
         "skipped diagnostic=active_domain_inactive_constraints_skipped reason=native_mesh_unavailable"),
        ("[svMultiPhysics::Application] Wrote effective configuration: /tmp/run/effective_configuration.json",
         "[12:00:00.000] [R0] [INFO] ApplicationDriver: wrote effective configuration "
         "diagnostic=effective_configuration path='/tmp/run/effective_configuration.json'"),
    ]
    logmap = L.load_map()
    for old, new in pairs:
        (a,) = L.records([old])
        (b,) = L.records([new])
        assert a.entry["id"] == b.entry["id"] and a.entry["batch"] == "params", new
        assert a.diagnostic == b.diagnostic, new
        assert [k for k, _ in a.fields] == [k for k, _ in b.fields], new
        assert all(L.values_equal(va, vb) for (_, va), (_, vb) in zip(a.fields, b.fields)), new
        assert a.level == b.level, new
        assert L.lint_line(new) == [], new
        entry, role = logmap.match(L.Line(new))
        assert L._is_new_format(L.Line(new), entry, role), new
        assert not L._is_new_format(L.Line(old), *logmap.match(L.Line(old))), old
    # The deprecated spelling shares the word and keeps its own message.
    (deprecated,) = L.records([pairs[3][0]])
    assert deprecated.message == "deprecated input setting"
