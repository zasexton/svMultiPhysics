#!/usr/bin/env python3
"""Convert free-surface decks to the profile form (decision D47, step P4).

A Free_surface boundary condition may name a method profile
(``<Free_surface_profile>Capillary@1</Free_surface_profile>``).  The profile
states its values on the boundary and on the level-set equation of the
boundary's field for every key that element does not state itself
(Physics/Options/FreeSurfaceProfiles.cpp).  This script rewrites a deck that
writes those values out into the profile form that resolves to the same
input:

  - it drops every key whose value the profile states;
  - it keeps every key whose value differs (an override);
  - for a profile value whose key the deck leaves at its code default, it
    writes the default as an override, or the velocity source that keeps the
    profile's advection velocity out;
  - it rejects a profile that would state a key whose mere presence changes
    the resolved input (the geometry tangent policy and the cut quadrature
    orders of the cut request).

It picks the profile that leaves the fewest keys among those that resolve to
the same input.  The proof of identity is the resolve-only comparison of the
effective configurations (``svmultiphysics --resolve-only``) of the original
and the converted deck; the provenance of the profile's values is the only
expected difference.

The profile table below mirrors FreeSurfaceProfiles.cpp.

explicit_form() writes a profile deck out again, for tools that check or
patch the method keys of a deck as text.

Usage:
  free_surface_profile_decks.py convert <solver.xml> [-o <out.xml>] [--profile NAME]
  free_surface_profile_decks.py plan <solver.xml>...     (JSON report, no writes)
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

# ---------------------------------------------------------------------------
# Profiles (mirror of Physics/Options/FreeSurfaceProfiles.cpp)
# ---------------------------------------------------------------------------
BC = "boundary"
LS = "level_set"
VELOCITY = ("Velocity_source", "Constant_velocity")


@dataclass(frozen=True)
class Value:
    element: str
    key: str
    value: str
    yields_to: tuple = ()
    contact_line_model: str | None = None


def _unfitted() -> list:
    return [
        Value(LS, "Level_set_source", "prescribed_data"),
        Value(LS, "Velocity_source", "prescribed_data", VELOCITY),
        Value(LS, "Velocity_field_name", "LevelSetAdvectionVelocity", VELOCITY),
        Value(LS, "Auto_register_velocity_field", "true", VELOCITY),
        Value(LS, "Source_velocity_field_name", "Velocity", VELOCITY),
        Value(LS, "Advection_velocity_extension_method", "pde_harmonic", VELOCITY),
        Value(LS, "Advection_velocity_extension_coupling", "monolithic", VELOCITY),
        Value(LS, "Enable_SUPG", "true"),
        Value(LS, "SUPG_tau_scale", "0.5"),
        Value(LS, "SUPG_transient_scale", "2.0"),
        Value(LS, "Enable_kinematic_reconciliation", "true"),
        Value(BC, "Level_set_isovalue", "0.0"),
        Value(BC, "Generated_interface_geometry", "LinearCorner"),
        Value(BC, "Enable_cut_cell_stabilization", "true"),
        Value(BC, "Cut_cell_pressure_gradient_penalty", "1.0", ("Enable_cut_cell_stabilization",)),
        Value(BC, "Small_cut_aggregation", "true"),
    ]


_CAPILLARY_GEOMETRY = [
    Value(BC, "Geometry_tangent_policy", "RefreshedFrozenQuadrature"),
    Value(BC, "Interface_quadrature_order", "2"),
]

PROFILES = {
    "Capillary@1": ("UnfittedLevelSet", _unfitted() + _CAPILLARY_GEOMETRY + [
        Value(BC, "Surface_tension_semi_implicit", "NormalIncrement")]),
    "Wetting@1": ("UnfittedLevelSet", _unfitted() + [
        Value(LS, "Enable_sign_definite_patch_bounds", "true"),
        Value(LS, "Enable_interface_preserving_redistancing", "true"),
    ] + _CAPILLARY_GEOMETRY + [
        Value(BC, "Surface_tension_semi_implicit", "None"),
        Value(BC, "Contact_line_law_enforcement", "Strong", (), "dynamicrene")]),
    "Gravity_wave@1": ("UnfittedLevelSet", _unfitted() + [
        Value(BC, "Surface_tension_semi_implicit", "None")]),
    "Fitted_ALE@1": ("FittedALE", [
        Value(BC, "Tangential_mesh_policy", "Free",
              ("Prescribed_tangential_mesh_velocity", "Tangential_mesh_penalty")),
        Value(BC, "Kinematic_nitsche_gamma", "10"),
        Value(BC, "Surface_tension_semi_implicit", "None")]),
}

# Code defaults of the profile keys (None: unset, and no text states "unset").
DEFAULTS = {
    "Level_set_source": "unknown",
    "Velocity_source": "coupled_field",
    "Velocity_field_name": "Velocity",
    "Auto_register_velocity_field": "false",
    "Source_velocity_field_name": None,
    "Advection_velocity_extension_method": None,
    "Advection_velocity_extension_coupling": None,
    "Enable_SUPG": "false",
    "SUPG_tau_scale": "0.5",
    "SUPG_transient_scale": "2.0",
    "Enable_kinematic_reconciliation": "false",
    "Enable_sign_definite_patch_bounds": "false",
    "Enable_interface_preserving_redistancing": "false",
    "Level_set_isovalue": "0.0",
    "Generated_interface_geometry": "LinearCorner",
    "Geometry_tangent_policy": "RefreshedFrozenQuadrature",
    "Interface_quadrature_order": None,
    "Enable_cut_cell_stabilization": "false",
    "Cut_cell_pressure_gradient_penalty": "1.0",
    "Small_cut_aggregation": "true",
    "Surface_tension_semi_implicit": "None",
    "Contact_line_law_enforcement": "Weak",
    "Tangential_mesh_policy": "SmoothingOnly",
    "Kinematic_nitsche_gamma": "10",
}

# Keys whose presence the cut request records (an explicit tangent policy,
# optional quadrature orders): a profile must not state them for a deck that
# leaves them out.
PRESENCE_OBSERVED = {"Geometry_tangent_policy", "Interface_quadrature_order"}

REAL_KEYS = {"SUPG_tau_scale", "SUPG_transient_scale", "Level_set_isovalue",
             "Cut_cell_pressure_gradient_penalty", "Kinematic_nitsche_gamma"}
INT_KEYS = {"Interface_quadrature_order"}
BOOL_KEYS = {"Auto_register_velocity_field", "Enable_SUPG", "Enable_kinematic_reconciliation",
             "Enable_sign_definite_patch_bounds", "Enable_interface_preserving_redistancing",
             "Enable_cut_cell_stabilization", "Small_cut_aggregation"}
# Enumeration synonyms of the schemas (normalized: lower case, no '_', '-', ' ').
ENUMS = {
    "Level_set_source": {"unknown": "unknown", "solution": "unknown",
                         "prescribeddata": "prescribed_data", "prescribed": "prescribed_data",
                         "data": "prescribed_data"},
    "Velocity_source": {"coupledfield": "coupled_field", "coupled": "coupled_field",
                        "unknown": "coupled_field", "prescribeddata": "prescribed_data",
                        "prescribed": "prescribed_data", "data": "prescribed_data",
                        "constantvector": "constant_vector", "constant": "constant_vector",
                        "materialinterfacephasepair": "material_interface_phase_pair"},
    "Advection_velocity_extension_method": {"pdeharmonic": "pde_harmonic",
                                            "pdenormal": "pde_normal",
                                            "pdeleastsquaresnormal": "pde_normal"},
    "Advection_velocity_extension_coupling": {"monolithic": "monolithic",
                                              "prescribed": "prescribed"},
    "Generated_interface_geometry": {"linearcorner": "LinearCorner", "cornerlinear": "LinearCorner",
                                     "linear": "LinearCorner", "legacy": "LinearCorner",
                                     "highorderimplicit": "HighOrderImplicit",
                                     "highorder": "HighOrderImplicit",
                                     "curvedimplicit": "HighOrderImplicit",
                                     "implicitcurved": "HighOrderImplicit"},
    "Geometry_tangent_policy": {"refreshedfrozenquadrature": "RefreshedFrozenQuadrature",
                                "refreshedfrozen": "RefreshedFrozenQuadrature",
                                "frozenquadrature": "RefreshedFrozenQuadrature",
                                "quasinewton": "RefreshedFrozenQuadrature",
                                "quasinewtongeometry": "RefreshedFrozenQuadrature",
                                "differentiatedquadrature": "DifferentiatedQuadrature",
                                "differentiated": "DifferentiatedQuadrature",
                                "exactgeometrytangent": "DifferentiatedQuadrature",
                                "exactsensitivities": "DifferentiatedQuadrature",
                                "shapederivative": "DifferentiatedQuadrature"},
    "Surface_tension_semi_implicit": {"none": "None", "off": "None", "disabled": "None",
                                      "normalincrement": "NormalIncrement",
                                      "laggednormalincrement": "NormalIncrement"},
    "Contact_line_law_enforcement": {"weak": "Weak", "strong": "Strong"},
    "Tangential_mesh_policy": {"free": "Free", "freetangential": "Free", "unconstrained": "Free",
                               "smoothingonly": "SmoothingOnly", "smoothing": "SmoothingOnly",
                               "meshsmoothing": "SmoothingOnly", "smooth": "SmoothingOnly",
                               "prescribed": "Prescribed", "prescribedtangential": "Prescribed",
                               "prescribedvelocity": "Prescribed"},
}
CONTACT_MODELS = {"none": "none", "disabled": "none", "off": "none", "pinned": "pinned",
                  "fixed": "pinned", "fixedposition": "pinned", "prescribedangle": "prescribedangle",
                  "prescribedcontactangle": "prescribedangle", "contactangle": "prescribedangle",
                  "dynamicrene": "dynamicrene", "dynamiccontactangle": "dynamicrene",
                  "dynamicangle": "dynamicrene", "rene": "dynamicrene"}
# Keys of a level-set equation that are not free-surface keys (deck length).
GENERIC_EQUATION_KEYS = {"Coupled", "Min_iterations", "Max_iterations", "Tolerance",
                         "Module_options", "Element_order", "Initialize", "Prestress",
                         "Use_taylor_hood_type_basis"}


def token(text: str) -> str:
    return re.sub(r"[\s_\-]", "", text).lower()


def canonical(key: str, text: str | None):
    """The value a schema reads from the text (for comparisons)."""
    if text is None:
        return None
    text = text.strip()
    try:
        if key in REAL_KEYS:
            return ("real", float(text))
        if key in INT_KEYS:
            return ("int", int(text))
    except ValueError:
        return ("text", text)
    if key in BOOL_KEYS:
        return ("bool", text.lower() in ("true", "1", "yes", "on"))
    if key in ENUMS:
        return ("enum", ENUMS[key].get(token(text), "?" + text))
    return ("text", text)


def same(key: str, a: str | None, b: str | None) -> bool:
    return canonical(key, a) == canonical(key, b)


# ---------------------------------------------------------------------------
# Deck text: direct keys of the free-surface boundaries and level-set equations
# ---------------------------------------------------------------------------
_OPEN = re.compile(r"<([A-Za-z_][\w.]*)(\s[^>]*)?(?<!/)>")
_CLOSE = re.compile(r"</([A-Za-z_][\w.]*)\s*>")
_SIMPLE = re.compile(r"^(\s*)<([A-Za-z_][\w.]*)>([^<]*)</\2>\s*$")
LEVEL_SET_TYPES = {"levelset", "levelsettransport"}


@dataclass
class Block:
    start: int          # line of the opening tag
    end: int            # line of the closing tag
    indent: str = ""     # indentation of the direct keys
    keys: dict = field(default_factory=dict)   # key -> (line index, text)

    def value(self, key):
        item = self.keys.get(key)
        return item[1] if item else None


def _blocks(lines, opener: re.Pattern, element: str):
    """Blocks whose opening line matches `opener`, with their direct simple keys."""
    out = []
    i = 0
    while i < len(lines):
        if not opener.search(lines[i]):
            i += 1
            continue
        depth = 0
        block = Block(start=i, end=i)
        for j in range(i, len(lines)):
            line = lines[j]
            simple = _SIMPLE.match(line)
            if simple and depth == 1:
                block.keys.setdefault(simple.group(2), (j, simple.group(3)))
                block.indent = block.indent or simple.group(1)
                continue
            if simple:
                continue
            depth += len(_OPEN.findall(line)) - len(_CLOSE.findall(line))
            if depth == 0 and j > i:
                block.end = j
                break
        out.append(block)
        i = block.end + 1
    return out


@dataclass
class Deck:
    path: Path
    lines: list
    level_sets: list
    surfaces: list


def read_deck(path: Path) -> Deck:
    return parse_deck(path.read_text(), path)


def parse_deck(text: str, path: Path | None = None) -> Deck:
    lines = text.splitlines(keepends=True)
    level_sets = [b for b in _blocks(lines, re.compile(r'<Add_equation\s+type="([^"]+)"'), "Add_equation")
                  if token(re.search(r'type="([^"]+)"', lines[b.start]).group(1)) in LEVEL_SET_TYPES]
    surfaces = [b for b in _blocks(lines, re.compile(r"<Add_BC\b"), "Add_BC")
                if token(b.value("Type") or "") == "freesurface"]
    return Deck(path, lines, level_sets, surfaces)


def profile_label(name: str) -> str:
    """The profile a Free_surface_profile value names ("Capillary" is its
    latest version)."""
    base, _, version = name.strip().partition("@")
    for label in PROFILES:
        if token(label.split("@")[0]) == token(base) and (not version or label.endswith("@" + version.strip())):
            return label
    if token(name) == "none":
        return "None"
    raise ValueError(f"unknown free-surface profile {name!r}")


def effective_deck_keys(text: str, boundary: str | None = None):
    """The keys of a deck's free-surface boundary (the first one, or the one
    named) and of the level-set equation of its field, after the profile the
    boundary names states its values: what the solver reads.  Values are the
    texts of the deck or of the profile."""
    deck = parse_deck(text)
    surfaces = [s for s in deck.surfaces
                if boundary is None or re.search(r'name="%s"' % re.escape(boundary), deck.lines[s.start])]
    if not surfaces:
        raise ValueError("no free-surface boundary condition")
    bc = {k: v[1] for k, v in surfaces[0].keys.items()}
    field_name = (bc.get("Level_set_field_name") or "level_set").strip()
    level_set = next((b for b in deck.level_sets
                      if (b.value("Level_set_field_name") or "level_set").strip() == field_name), None)
    ls = {k: v[1] for k, v in level_set.keys.items()} if level_set else {}
    name = bc.pop("Free_surface_profile", None)
    if name is None or profile_label(name) == "None":
        return bc, ls
    return effective(profile_label(name), bc, ls)


# ---------------------------------------------------------------------------
# Planning
# ---------------------------------------------------------------------------
def applicable(profile: str, element: str, stated: dict, boundary: dict):
    """FreeSurfaceProfiles.cpp: applicableFreeSurfaceProfileValues."""
    model = CONTACT_MODELS.get(token(boundary.get("Contact_line_model") or ""))
    out = {}
    for value in PROFILES[profile][1]:
        if value.element != element or value.key in stated or \
                any(key in stated for key in value.yields_to):
            continue
        if value.contact_line_model is not None and model != value.contact_line_model:
            continue
        out[value.key] = value.value
    return out


def effective(profile, bc: dict, ls: dict):
    """The entries the input holds after the profile states its values."""
    bc_all = dict(bc)
    bc_all.update(applicable(profile, BC, bc, bc))
    ls_all = dict(ls)
    ls_all.update(applicable(profile, LS, ls, bc))
    return bc_all, ls_all


def equivalent(original: dict, result: dict):
    """Whether `result` (entries after the profile) reads like `original`:
    the same values, and extra keys only at a default whose presence is not
    observed."""
    for key, text in original.items():
        if key not in result or not same(key, text, result[key]):
            return False, f"{key}: {text!r} -> {result.get(key)!r}"
    for key, text in result.items():
        if key in original:
            continue
        if key in PRESENCE_OBSERVED:
            return False, f"{key}: stated by the profile, absent from the deck"
        default = DEFAULTS.get(key, "<no default>")
        if default is None or not same(key, text, default):
            return False, f"{key}: profile value {text!r} differs from the default {default!r}"
    return True, ""


def plan_profile(profile: str, bc: dict, ls: dict):
    """The keys to drop and to add for one profile, or None with the reason."""
    implementation = PROFILES[profile][0]
    if token(bc.get("Implementation", "")) != token(implementation):
        return None, f"written for Implementation={implementation}"
    values = PROFILES[profile][1]
    bc_keys, ls_keys = dict(bc), dict(ls)

    def keys_of(element):
        return bc_keys if element == BC else ls_keys

    # Overrides for profile keys the deck leaves at the default.
    for value in values:
        stated = keys_of(value.element)
        if value.key in stated or any(k in stated for k in value.yields_to):
            continue
        if value.contact_line_model is not None and \
                CONTACT_MODELS.get(token(bc.get("Contact_line_model") or "")) != value.contact_line_model:
            continue
        default = DEFAULTS[value.key]
        if same(value.key, value.value, default) and value.key not in PRESENCE_OBSERVED:
            continue
        if value.yields_to == VELOCITY:
            stated.setdefault("Velocity_source", DEFAULTS["Velocity_source"])
        elif default is not None and value.key not in PRESENCE_OBSERVED:
            stated[value.key] = default
        else:
            return None, f"{value.key}: the deck leaves it out; the profile would state it"

    # Drop what the profile restates: the advection velocity as one block,
    # then single keys, repeated while a drop lets another value apply.
    group = [v for v in values if v.yields_to == VELOCITY]
    changed = True
    while changed:
        changed = False
        candidates = []
        if group and all(v.key not in ls_keys or same(v.key, ls_keys[v.key], v.value) for v in group) \
                and "Constant_velocity" not in ls_keys:
            candidates.append([v for v in group if v.key in ls_keys])
        candidates += [[v] for v in values if v.yields_to != VELOCITY]
        for candidate in candidates:
            if not candidate:
                continue
            trial_bc, trial_ls = dict(bc_keys), dict(ls_keys)
            for v in candidate:
                keys = trial_bc if v.element == BC else trial_ls
                if v.key not in keys or not same(v.key, keys[v.key], v.value):
                    break
                del keys[v.key]
            else:
                bc_all, ls_all = effective(profile, trial_bc, trial_ls)
                if equivalent(bc, bc_all)[0] and equivalent(ls, ls_all)[0]:
                    bc_keys, ls_keys = trial_bc, trial_ls
                    changed = True
    bc_all, ls_all = effective(profile, bc_keys, ls_keys)
    for original, result in ((bc, bc_all), (ls, ls_all)):
        ok, reason = equivalent(original, result)
        if not ok:
            return None, reason
    return (bc_keys, ls_keys), ""


def deck_length(bc: dict, ls: dict) -> int:
    return len([k for k in bc if k != "Type"]) + len([k for k in ls if k not in GENERIC_EQUATION_KEYS])


def plan(deck: Deck, forced: str | None = None):
    """One entry per free-surface boundary: the chosen profile and the edits."""
    entries = []
    for surface in deck.surfaces:
        bc = {k: v[1] for k, v in surface.keys.items()}
        if "Free_surface_profile" in bc:
            entries.append({"boundary": surface, "profile": None, "reason": "already names a profile"})
            continue
        field_name = (bc.get("Level_set_field_name") or "level_set").strip()
        level_set = next((b for b in deck.level_sets
                          if (b.value("Level_set_field_name") or "level_set").strip() == field_name), None)
        ls = {k: v[1] for k, v in level_set.keys.items()} if level_set else {}
        options = []
        for profile in ([forced] if forced else PROFILES):
            result, reason = plan_profile(profile, bc, ls)
            if result is None:
                continue
            if level_set is None and any(v.element == LS for v in PROFILES[profile][1]):
                continue
            options.append((deck_length(*result) + 1, list(PROFILES).index(profile), profile, result))
        before = deck_length(bc, ls)
        if not options:
            entries.append({"boundary": surface, "level_set": level_set, "profile": None,
                            "before": before, "reason": "no profile resolves to the same input"})
            continue
        length, _, profile, (bc_keys, ls_keys) = min(options)
        if length >= before and not forced:
            entries.append({"boundary": surface, "level_set": level_set, "profile": None,
                            "before": before, "reason": "no profile shortens the deck"})
            continue
        entries.append({"boundary": surface, "level_set": level_set, "profile": profile,
                        "before": before, "after": length, "bc": bc_keys, "ls": ls_keys,
                        "bc_original": bc, "ls_original": ls})
    return entries


def profile_form(text: str, profile: str | None = None) -> str:
    """The profile form of a deck's text (the text itself when no profile
    resolves to the same input with fewer keys).  Generators that write the
    explicit form pass their text through this function."""
    deck = parse_deck(text)
    entries = plan(deck, profile)
    if not any(entry.get("profile") for entry in entries):
        return text
    return rewrite(deck, entries)


def rewrite(deck: Deck, entries) -> str:
    lines = list(deck.lines)
    drops, inserts = set(), {}
    for entry in entries:
        if entry.get("profile") is None:
            continue
        surface, level_set = entry["boundary"], entry["level_set"]
        for block, original, result in ((surface, entry["bc_original"], entry["bc"]),
                                        (level_set, entry["ls_original"], entry["ls"])):
            if block is None:
                continue
            for key, (line, _) in block.keys.items():
                if key not in result:
                    drops.add(line)
            added = [k for k in result if k not in original]
            anchor = max(line for line, _ in block.keys.values()) if block.keys else block.start
            inserts.setdefault(anchor, []).extend(
                f"{block.indent}<{k}>{result[k]}</{k}>\n" for k in added)
        anchor = surface.keys["Implementation"][0]
        inserts.setdefault(anchor, []).insert(
            0, f"{surface.indent}<Free_surface_profile>{entry['profile']}</Free_surface_profile>\n")
    out = []
    for index, line in enumerate(lines):
        if index not in drops:
            out.append(line)
        out.extend(inserts.get(index, []))
    return "".join(out)


def explicit_form(text: str) -> str:
    """The explicit form of a deck's text: the values each named profile
    states are written out (after the element's last key) and the
    Free_surface_profile key goes.  Tools that check or patch the method keys
    of a deck as text read it through this function."""
    deck = parse_deck(text)
    drops, inserts = set(), {}
    for surface in deck.surfaces:
        bc = {k: v[1] for k, v in surface.keys.items()}
        name = bc.pop("Free_surface_profile", None)
        if name is None:
            continue
        drops.add(surface.keys["Free_surface_profile"][0])
        label = profile_label(name)
        if label == "None":
            continue
        field_name = (bc.get("Level_set_field_name") or "level_set").strip()
        level_set = next((b for b in deck.level_sets
                          if (b.value("Level_set_field_name") or "level_set").strip() == field_name), None)
        ls = {k: v[1] for k, v in level_set.keys.items()} if level_set else {}
        for block, element, stated in ((surface, BC, bc), (level_set, LS, ls)):
            values = applicable(label, element, stated, bc)
            if block is None or not values:
                continue
            anchor = max(line for line, _ in block.keys.values())
            inserts.setdefault(anchor, []).extend(
                f"{block.indent}<{key}>{value}</{key}>\n" for key, value in values.items())
    out = []
    for index, line in enumerate(deck.lines):
        if index not in drops:
            out.append(line)
        out.extend(inserts.get(index, []))
    return "".join(out)


def summary(entries):
    return [{k: v for k, v in entry.items() if k in ("profile", "before", "after", "reason")}
            for entry in entries]


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    convert = sub.add_parser("convert", help="write the profile form of one deck")
    convert.add_argument("deck", type=Path)
    convert.add_argument("-o", "--output", type=Path, default=None, help="default: in place")
    convert.add_argument("--profile", choices=tuple(PROFILES), default=None)
    report = sub.add_parser("plan", help="print the chosen profiles and deck lengths as JSON")
    report.add_argument("decks", type=Path, nargs="+")
    args = parser.parse_args(argv)

    if args.command == "plan":
        out = {}
        for path in args.decks:
            out[str(path)] = summary(plan(read_deck(path)))
        json.dump(out, sys.stdout, indent=2)
        print()
        return 0

    deck = read_deck(args.deck)
    entries = plan(deck, args.profile)
    for entry in entries:
        if entry.get("profile") is None and entry.get("reason") != "already names a profile":
            print(f"{args.deck}: {entry.get('reason')}", file=sys.stderr)
    if not any(entry.get("profile") for entry in entries):
        return 1
    (args.output or args.deck).write_text(rewrite(deck, entries))
    print(json.dumps({str(args.deck): summary(entries)}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
