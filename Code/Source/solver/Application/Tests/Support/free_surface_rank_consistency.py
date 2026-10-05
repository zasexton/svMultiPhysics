#!/usr/bin/env python3
"""End-to-end rank-count check of unfitted free-surface (CutFEM) runs.

Generates small benchmark decks with the repository generators (a static
drop at R/h = 8, a capillary wave at lambda/h = 32 and a sessile drop at
R/h = 16 with per-step volume correction), runs each for a few steps with
svmultiphysics through mpiexec on several rank counts, and checks that

  * every run exits cleanly (a rank whose partition holds no interface cell,
    as in the capillary wave on 4 ranks, must not abort the run, and the
    post-accept volume-correction transaction must reach consensus),
  * the Newton iteration count of every nonlinear solve agrees with the
    1-rank run, and
  * phi, Velocity and Pressure at the last step agree with the 1-rank run to
    round-off (matched by GlobalNodeID; shared vertices of partitioned
    output are also checked for agreement between their copies).

Only the Python standard library is used to read results; the generators
need numpy.  Exit code 77 means the prerequisites are missing (CTest SKIP).

Usage:
  free_surface_rank_consistency.py --solver BIN --mpiexec MPIEXEC
      --numproc-flag=-n --benchmarks DIR --work-dir DIR [--ranks 1 2 4]
      [--steps 2] [--mpiexec-preflags="..."]
"""

import argparse
import math
import re
import shlex
import shutil
import struct
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

SKIP = 77
FIELDS = ("phi", "Velocity", "Pressure")
# Relative to the field's scale (max |value| over the 1-rank run).  Serial
# and distributed solves see the same linear system; their differences come
# from reduction order and the GMRES stopping point only.
RELATIVE_TOLERANCE = 1.0e-9
NEWTON_RE = re.compile(r"Total Newton time:\s+\S+ s\s+\((\d+) Newton iters")

# name -> (generator directory, generator arguments, deck edits)
CASES = {
    "static_drop_L8": ("static_drop_2d",
                       ["--level", "8", "--capillary-form", "surface_stress",
                        "--laplace-number", "12"], []),
    "capillary_wave_L32": ("capillary_wave_2d", ["--level", "32"], []),
    "sessile_drop_L16_60_volume_correction": (
        "sessile_drop_2d",
        ["--level", "16", "--contact-angle", "60", "--transport", "pde_extension"],
        [(r"<Enable_volume_correction>\s*false\s*</Enable_volume_correction>",
          "<Enable_volume_correction>true</Enable_volume_correction>"
          "<Volume_correction_cadence_steps>1</Volume_correction_cadence_steps>"
          "<Volume_correction_minimum_relative_error>1.0e-12"
          "</Volume_correction_minimum_relative_error>")]),
}


def generate(benchmarks, bench, args, edits, out, steps):
    script = benchmarks / bench / "generate_case.py"
    cmd = [sys.executable, str(script), *args, "--output-dir", str(out),
           "--max-steps", str(steps), "--force"]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        if "No module named" in res.stderr:
            print(res.stderr)
            sys.exit(SKIP)
        raise RuntimeError(f"generator failed: {' '.join(cmd)}\n{res.stdout}\n{res.stderr}")
    deck = out / "solver.xml"
    text = deck.read_text()
    text = re.sub(r"<Number_of_time_steps>\s*\d+\s*</Number_of_time_steps>",
                  f"<Number_of_time_steps>{steps}</Number_of_time_steps>", text)
    text = re.sub(r"<Increment_in_saving_VTK_files>\s*\d+\s*</Increment_in_saving_VTK_files>",
                  "<Increment_in_saving_VTK_files>1</Increment_in_saving_VTK_files>", text)
    for pattern, replacement in edits:
        text, count = re.subn(pattern, replacement, text)
        if count != 1:
            raise RuntimeError(f"deck edit {pattern!r} matched {count} times in {deck}")
    deck.write_text(text)


def read_vtu(path):
    """Point arrays of an appended-raw VTU written by svmultiphysics."""
    data = path.read_bytes()
    marker = data.find(b"<AppendedData")
    start = data.index(b"_", marker) + 1
    prefix = data[:marker].decode()
    if "</UnstructuredGrid>" not in prefix:
        prefix += "</UnstructuredGrid>"
    header = ET.fromstring(prefix + "</VTKFile>")
    grid = header.find("UnstructuredGrid")
    header_type = header.get("header_type", "UInt32")
    hfmt, hsize = ("<Q", 8) if header_type == "UInt64" else ("<I", 4)
    types = {"Float64": ("d", 8), "Float32": ("f", 4), "Int64": ("q", 8), "Int32": ("i", 4),
             "UInt64": ("Q", 8), "UInt32": ("I", 4), "Int8": ("b", 1), "UInt8": ("B", 1)}
    out = {}
    for array in grid.find("Piece").find("PointData").findall("DataArray"):
        name = array.get("Name")
        if name not in FIELDS and name != "GlobalNodeID":
            continue
        code, size = types[array.get("type")]
        offset = start + int(array.get("offset"))
        nbytes = struct.unpack_from(hfmt, data, offset)[0]
        values = struct.unpack_from(f"<{nbytes // size}{code}", data, offset + hsize)
        ncomp = int(array.get("NumberOfComponents", "1"))
        out[name] = [values[i * ncomp:(i + 1) * ncomp] for i in range(len(values) // ncomp)]
    return out


def last_step_fields(run, steps):
    stem = f"result_{steps:03d}"
    pieces = sorted(run.glob(f"{stem}_p*.vtu")) or [run / f"{stem}.vtu"]
    merged = {}
    spread = 0.0
    for piece in pieces:
        arrays = read_vtu(piece)
        ids = [int(v[0]) for v in arrays["GlobalNodeID"]]
        for k, node in enumerate(ids):
            values = {f: tuple(arrays[f][k][:2] if f == "Velocity" else arrays[f][k]) for f in FIELDS}
            if node in merged:
                for f in FIELDS:
                    spread = max(spread, max(abs(a - b) for a, b in zip(values[f], merged[node][f])))
            else:
                merged[node] = values
    if spread != 0.0:
        raise AssertionError(f"{run}: shared vertex copies disagree by {spread:.3e}")
    return merged


def newton_counts(log):
    return [int(m.group(1)) for m in NEWTON_RE.finditer(log)]


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--solver", type=Path, required=True)
    p.add_argument("--mpiexec", required=True)
    p.add_argument("--numproc-flag", default="-n")
    p.add_argument("--mpiexec-preflags", default="")
    p.add_argument("--benchmarks", type=Path, required=True)
    p.add_argument("--work-dir", type=Path, required=True)
    p.add_argument("--ranks", type=int, nargs="+", default=[1, 2, 4])
    p.add_argument("--steps", type=int, default=2)
    p.add_argument("--timeout", type=int, default=900)
    a = p.parse_args()

    if not a.solver.is_file() or not (a.benchmarks / "static_drop_2d").is_dir():
        print(f"missing solver {a.solver} or benchmarks {a.benchmarks}")
        return SKIP
    if a.work_dir.exists():
        shutil.rmtree(a.work_dir)
    a.work_dir.mkdir(parents=True)

    failures = []
    for name, (bench, gen_args, edits) in CASES.items():
        base = a.work_dir / name / "deck"
        generate(a.benchmarks, bench, gen_args, edits, base, a.steps)
        reference = None
        ref_newton = None
        scales = {}
        for ranks in a.ranks:
            run = a.work_dir / name / f"np{ranks}"
            shutil.copytree(base, run)
            cmd = [a.mpiexec, a.numproc_flag, str(ranks), *shlex.split(a.mpiexec_preflags),
                   str(a.solver), "solver.xml"]
            try:
                res = subprocess.run(cmd, cwd=run, capture_output=True, text=True, timeout=a.timeout)
            except subprocess.TimeoutExpired:
                failures.append(f"{name} np{ranks}: timed out after {a.timeout} s")
                continue
            (run / "solver_run.log").write_text(res.stdout + res.stderr)
            if res.returncode != 0:
                tail = "\n".join((res.stdout + res.stderr).splitlines()[-15:])
                failures.append(f"{name} np{ranks}: exit {res.returncode}\n{tail}")
                continue
            newton = newton_counts(res.stdout + res.stderr)
            fields = last_step_fields(run, a.steps)
            if reference is None:
                reference, ref_newton = fields, newton
                for f in FIELDS:
                    scales[f] = max(max(abs(x) for x in v[f]) for v in fields.values()) or 1.0
                print(f"{name} np{ranks}: reference, {len(fields)} vertices, Newton {newton}")
                continue
            if newton != ref_newton:
                failures.append(f"{name} np{ranks}: Newton iterations {newton} != np{a.ranks[0]} {ref_newton}")
            if set(fields) != set(reference):
                failures.append(f"{name} np{ranks}: vertex set differs from np{a.ranks[0]}")
                continue
            worst = {}
            for f in FIELDS:
                diff = max(max(abs(x - y) for x, y in zip(fields[n][f], reference[n][f])) for n in reference)
                worst[f] = diff / scales[f]
                if not (worst[f] <= RELATIVE_TOLERANCE) or math.isnan(worst[f]):
                    failures.append(f"{name} np{ranks}: {f} differs from np{a.ranks[0]} by "
                                    f"{worst[f]:.3e} of its scale (tolerance {RELATIVE_TOLERANCE:g})")
            print(f"{name} np{ranks}: Newton {newton}, relative differences "
                  + " ".join(f"{f}={worst[f]:.2e}" for f in FIELDS))

    for failure in failures:
        print("FAIL:", failure)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
