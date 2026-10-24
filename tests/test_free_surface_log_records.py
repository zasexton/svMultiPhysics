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
