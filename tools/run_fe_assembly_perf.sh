#!/usr/bin/env bash
set -euo pipefail

# Perf harness focused on FE assembly costs.
#
# Defaults target pipe_simple because it is assembly dominated, but the script is
# intentionally case- and mode-configurable so findings can generalize across
# the FE library.
#
# Examples:
#   tools/run_fe_assembly_perf.sh
#   CASES="pipe_simple:solver_perf_oop.xml Channel2D:solver_perf_oop.xml" REPS=5 tools/run_fe_assembly_perf.sh
#   MODES="serial mpi2 mpi4" RUN_RECORD=0 tools/run_fe_assembly_perf.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -z "${REPO:-}" ]]; then
  if repo_root="$(git -C "$SCRIPT_DIR/.." rev-parse --show-toplevel 2>/dev/null)"; then
    REPO="$repo_root"
  else
    REPO="$(cd "$SCRIPT_DIR/.." && pwd)"
  fi
fi
BUILD_DIR="${BUILD_DIR:-$REPO/build}"
BINARY="${BINARY:-$BUILD_DIR/svMultiPhysics-build/bin/svmultiphysics}"
TESTS_DIR="${TESTS_DIR:-$REPO/tests/cases/fluid}"
RESULTS_DIR="${RESULTS_DIR:-/tmp/svmp_fe_assembly_perf_$(date +%Y%m%d_%H%M%S)}"

CASES="${CASES:-pipe_simple:solver_perf_oop.xml}"
MODES="${MODES:-serial}"
REPS="${REPS:-3}"
WARMUP="${WARMUP:-1}"
BUILD="${BUILD:-1}"
RUN_RECORD="${RUN_RECORD:-1}"
RECORD_REPS="${RECORD_REPS:-1}"
CHECK_PERF="${CHECK_PERF:-1}"
CLEAR_JIT="${CLEAR_JIT:-0}"
PARSE_ONLY="${PARSE_ONLY:-0}"

PERF_EVENTS="${PERF_EVENTS:-task-clock,context-switches,cpu-migrations,page-faults,cycles,instructions,branches,branch-misses,cache-references,cache-misses,L1-dcache-loads,L1-dcache-load-misses,L1-dcache-stores,L1-icache-load-misses,dTLB-loads,dTLB-load-misses,iTLB-loads,iTLB-load-misses}"
PERF_RECORD_EVENTS="${PERF_RECORD_EVENTS:-cycles:u,instructions:u,cache-misses:u,L1-dcache-load-misses:u,branch-misses:u}"
PERF_FREQ="${PERF_FREQ:-997}"
PERF_CALLGRAPH="${PERF_CALLGRAPH:-dwarf}"
PERF_REPORT_PERCENT_LIMIT="${PERF_REPORT_PERCENT_LIMIT:-0.25}"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export SVMP_ASSEMBLY_TIMING="${SVMP_ASSEMBLY_TIMING:-1}"
export SVMP_FSILS_GMRES_PROFILE="${SVMP_FSILS_GMRES_PROFILE:-0}"
export SVMP_FSILS_MATRIX_LOCALITY_PROFILE="${SVMP_FSILS_MATRIX_LOCALITY_PROFILE:-0}"
export SVMP_FSILS_GMRES_BASIS_PANEL="${SVMP_FSILS_GMRES_BASIS_PANEL:-0}"
export SVMP_FSILS_GMRES_REORTH="${SVMP_FSILS_GMRES_REORTH:-off}"

mkdir -p "$RESULTS_DIR"/{logs,perf,record,work}

metadata_file="$RESULTS_DIR/run_metadata.txt"
if [[ "$PARSE_ONLY" == "1" && -e "$metadata_file" ]]; then
  metadata_file="$RESULTS_DIR/parse_metadata_$(date +%Y%m%d_%H%M%S).txt"
fi

cat > "$metadata_file" <<EOF
date=$(date --iso-8601=seconds)
repo=$REPO
build_dir=$BUILD_DIR
binary=$BINARY
cases=$CASES
modes=$MODES
reps=$REPS
warmup=$WARMUP
run_record=$RUN_RECORD
record_reps=$RECORD_REPS
parse_only=$PARSE_ONLY
perf_events=$PERF_EVENTS
perf_record_events=$PERF_RECORD_EVENTS
perf_freq=$PERF_FREQ
perf_callgraph=$PERF_CALLGRAPH
OMP_NUM_THREADS=$OMP_NUM_THREADS
SVMP_ASSEMBLY_TIMING=$SVMP_ASSEMBLY_TIMING
SVMP_FSILS_GMRES_PROFILE=$SVMP_FSILS_GMRES_PROFILE
SVMP_FSILS_MATRIX_LOCALITY_PROFILE=$SVMP_FSILS_MATRIX_LOCALITY_PROFILE
SVMP_FSILS_GMRES_BASIS_PANEL=$SVMP_FSILS_GMRES_BASIS_PANEL
SVMP_FSILS_GMRES_REORTH=$SVMP_FSILS_GMRES_REORTH
EOF

echo "results_dir=$RESULTS_DIR"
echo "repo=$REPO"
echo "binary=$BINARY"
echo "cases=$CASES"
echo "modes=$MODES"
echo "reps=$REPS"
echo "run_record=$RUN_RECORD"
echo "parse_only=$PARSE_ONLY"

if [[ "$PARSE_ONLY" != "1" && "$BUILD" == "1" ]]; then
  cmake --build "$BUILD_DIR" --target svMultiPhysics --parallel
  if [[ -d "$BUILD_DIR/svMultiPhysics-build" ]]; then
    cmake --build "$BUILD_DIR/svMultiPhysics-build" --target svmultiphysics --parallel
  fi
fi

if [[ "$PARSE_ONLY" != "1" && ! -x "$BINARY" ]]; then
  echo "error: binary not executable: $BINARY" >&2
  exit 1
fi

if [[ "$PARSE_ONLY" != "1" && "$CHECK_PERF" == "1" ]]; then
  if ! perf stat true >/dev/null 2>"$RESULTS_DIR/logs/perf_check.log"; then
    echo "error: perf stat failed; see $RESULTS_DIR/logs/perf_check.log" >&2
    exit 1
  fi
fi

if [[ "$PARSE_ONLY" != "1" && "$CLEAR_JIT" == "1" ]]; then
  rm -rf "${HOME}/.cache/svMultiPhysics/jit_cache" 2>/dev/null || true
fi

case_specs=()
mode_specs=()
read -r -a case_specs <<< "${CASES//,/ }"
read -r -a mode_specs <<< "${MODES//,/ }"

np_for_mode() {
  local mode="$1"
  if [[ "$mode" == "serial" || "$mode" == "seq" || "$mode" == "1" ]]; then
    echo 1
  elif [[ "$mode" =~ ^mpi([0-9]+)$ ]]; then
    echo "${BASH_REMATCH[1]}"
  elif [[ "$mode" =~ ^np([0-9]+)$ ]]; then
    echo "${BASH_REMATCH[1]}"
  else
    echo "error: unsupported mode '$mode' (use serial, mpi2, mpi4, npN)" >&2
    exit 1
  fi
}

clean_outputs() {
  local dir="$1"
  rm -rf "$dir"/1-procs "$dir"/[0-9]*-procs "$dir"/*.vtu "$dir"/histor.dat \
         "$dir"/STOP_SIM "$dir"/restart.* "$dir"/geombc.dat.* "$dir"/restart[0-9]* 2>/dev/null || true
}

run_solver() {
  local np="$1"
  local xml="$2"
  if [[ "$np" == "1" ]]; then
    "$BINARY" "$xml"
  else
    mpirun -np "$np" "$BINARY" "$xml"
  fi
}

run_one() {
  local mode="$1"
  local np="$2"
  local case_name="$3"
  local xml="$4"
  local rep="$5"
  local src_dir="$TESTS_DIR/$case_name"
  local work_dir="$RESULTS_DIR/work/${mode}_${case_name}_rep${rep}"
  local stem="$RESULTS_DIR/perf/${mode}_${case_name}_rep${rep}"
  local record_stem="$RESULTS_DIR/record/${mode}_${case_name}_rep${rep}"

  if [[ ! -d "$src_dir" ]]; then
    echo "error: missing case directory: $src_dir" >&2
    exit 1
  fi
  if [[ ! -f "$src_dir/$xml" ]]; then
    echo "error: missing XML '$xml' in $src_dir" >&2
    exit 1
  fi

  rm -rf "$work_dir"
  cp -a "$src_dir" "$work_dir"
  clean_outputs "$work_dir"

  if [[ "$WARMUP" == "1" && "$rep" == "1" ]]; then
    echo "warmup mode=$mode case=$case_name xml=$xml"
    (
      cd "$work_dir"
      run_solver "$np" "$xml"
    ) >"${stem}.warmup.stdout.log" 2>"${stem}.warmup.stderr.log" || {
      echo "error: warmup failed for mode=$mode case=$case_name; see ${stem}.warmup.stderr.log" >&2
      exit 1
    }
    clean_outputs "$work_dir"
  fi

  echo "stat mode=$mode case=$case_name rep=$rep xml=$xml"
  (
    cd "$work_dir"
    if [[ "$np" == "1" ]]; then
      perf stat -x, -e "$PERF_EVENTS" -o "${stem}.perf-stat.csv" \
        "$BINARY" "$xml"
    else
      perf stat -x, -e "$PERF_EVENTS" -o "${stem}.perf-stat.csv" \
        mpirun -np "$np" "$BINARY" "$xml"
    fi
  ) >"${stem}.stdout.log" 2>"${stem}.stderr.log"

  if [[ "$RUN_RECORD" == "1" && "$rep" -le "$RECORD_REPS" ]]; then
    clean_outputs "$work_dir"
    echo "record mode=$mode case=$case_name rep=$rep xml=$xml"
    (
      cd "$work_dir"
      if [[ "$np" == "1" ]]; then
        perf record -F "$PERF_FREQ" -g --call-graph "$PERF_CALLGRAPH" \
          -e "$PERF_RECORD_EVENTS" -o "${record_stem}.perf.data" -- \
          "$BINARY" "$xml"
      else
        perf record -F "$PERF_FREQ" -g --call-graph "$PERF_CALLGRAPH" \
          -e "$PERF_RECORD_EVENTS" -o "${record_stem}.perf.data" -- \
          mpirun -np "$np" "$BINARY" "$xml"
      fi
    ) >"${record_stem}.stdout.log" 2>"${record_stem}.stderr.log"

    perf report --stdio --no-children --sort comm,dso,symbol -n \
      --percent-limit "$PERF_REPORT_PERCENT_LIMIT" \
      -i "${record_stem}.perf.data" > "${record_stem}.perf-report-flat.txt" || true
    perf report --stdio --children --sort comm,dso,symbol -n \
      --percent-limit "$PERF_REPORT_PERCENT_LIMIT" \
      -i "${record_stem}.perf.data" > "${record_stem}.perf-report-children.txt" || true
    perf annotate --stdio -i "${record_stem}.perf.data" \
      > "${record_stem}.perf-annotate.txt" 2>"${record_stem}.perf-annotate.stderr.log" || true
  fi
}

if [[ "$PARSE_ONLY" != "1" ]]; then
  for spec in "${case_specs[@]}"; do
    [[ -n "$spec" ]] || continue
    case_name="${spec%%:*}"
    if [[ "$spec" == *:* ]]; then
      xml="${spec#*:}"
    else
      xml="solver_perf_oop.xml"
    fi

    for mode in "${mode_specs[@]}"; do
      [[ -n "$mode" ]] || continue
      np="$(np_for_mode "$mode")"
      if [[ "$np" != "1" ]]; then
        mpirun -np "$np" /bin/true >/dev/null
      fi
      for rep in $(seq 1 "$REPS"); do
        run_one "$mode" "$np" "$case_name" "$xml" "$rep"
      done
    done
  done
fi

python3 - "$RESULTS_DIR" <<'PY'
from pathlib import Path
import csv
import math
import re
import statistics
import sys

root = Path(sys.argv[1])
perf_dir = root / "perf"
rows = []

def fnum(s):
    try:
        return float(str(s).replace(",", ""))
    except Exception:
        return None

def first_float(line):
    m = re.search(r"[-+]?[0-9]+(?:\.[0-9]+)?", line)
    return float(m.group(0)) if m else None

def parse_perf_stat(path):
    data = {}
    with path.open(errors="ignore") as fh:
        for line in fh:
            if not line.strip() or line.startswith("#"):
                continue
            parts = line.rstrip("\n").split(",")
            if len(parts) < 3:
                continue
            value = fnum(parts[0])
            if value is None:
                continue
            unit = parts[1].strip()
            event = parts[2].strip()
            data[event] = value
            if event == "task-clock" and unit == "msec":
                data["task_clock_ms"] = value
    instr = data.get("instructions")
    cycles = data.get("cycles")
    if instr and cycles:
        data["ipc"] = instr / cycles
    if instr:
        for key in ("cache-misses", "L1-dcache-load-misses", "dTLB-load-misses", "branch-misses"):
            if key in data:
                data[key + "_mpki"] = 1000.0 * data[key] / instr
    if data.get("cache-references") and data.get("cache-misses"):
        data["cache_miss_pct"] = 100.0 * data["cache-misses"] / data["cache-references"]
    if data.get("L1-dcache-loads") and data.get("L1-dcache-load-misses"):
        data["l1d_load_miss_pct"] = 100.0 * data["L1-dcache-load-misses"] / data["L1-dcache-loads"]
    if data.get("branches") and data.get("branch-misses"):
        data["branch_miss_pct"] = 100.0 * data["branch-misses"] / data["branches"]
    return data

def parse_solver_stderr(path):
    data = {
        "wall_time_loop_s": 0.0,
        "wall_solve_s": 0.0,
        "wall_vtk_s": 0.0,
        "assembly_blocks": 0,
        "assembly_total_s": 0.0,
        "assembly_cell_s": 0.0,
        "assembly_boundary_s": 0.0,
        "assembly_other_s": 0.0,
        "fsils_solve_blocks": 0,
        "fsils_solve_total_s": 0.0,
        "gmres_blocks": 0,
        "gmres_total_s": 0.0,
    }
    context = None
    with path.open(errors="ignore") as fh:
        for line in fh:
            if "TOP-LEVEL TIMING SUMMARY" in line:
                context = "top"
                continue
            if "=== assembleOperator TIMING" in line:
                context = "assembly"
                data["assembly_blocks"] += 1
                continue
            if "--- fsils_solve TIMING" in line:
                context = "fsils"
                data["fsils_solve_blocks"] += 1
                continue
            if "=== GMRES_V TIMING PROFILE" in line:
                context = "gmres"
                data["gmres_blocks"] += 1
                continue
            if "====" in line or "----" in line:
                context = None
                continue

            val = first_float(line)
            if val is None:
                continue
            if context == "top":
                if "Total time loop:" in line:
                    data["wall_time_loop_s"] += val
                elif "Solve (Newton+linear):" in line:
                    data["wall_solve_s"] += val
                elif "VTK output:" in line:
                    data["wall_vtk_s"] += val
            elif context == "assembly":
                if "Total:" in line:
                    data["assembly_total_s"] += val
                elif "Cell terms:" in line:
                    data["assembly_cell_s"] += val
                elif "Boundary terms:" in line:
                    data["assembly_boundary_s"] += val
                elif "Other (DG+global):" in line:
                    data["assembly_other_s"] += val
            elif context == "fsils":
                if "Total:" in line:
                    data["fsils_solve_total_s"] += val
            elif context == "gmres":
                if "Total GMRES time:" in line:
                    data["gmres_total_s"] += val
    return data

def parse_solver_stdout(path):
    data = {
        "loop_success": "",
        "steps_taken": 0,
        "nonlinear_steps": 0,
        "nonlinear_all_converged": "",
        "newton_iters_total": 0,
        "linear_iters_total": 0,
        "linear_all_converged": "",
        "max_residual": 0.0,
        "max_linear_rel": 0.0,
    }
    nonlinear_seen = False
    nonlinear_all = True
    linear_all = True
    seen_nonlinear_records = set()
    nonlinear_re = re.compile(
        r"TimeLoop: nonlinear_done step=([0-9]+).*? converged=([01]) iters=([0-9]+) "
        r"\|\|r\|\|=([-+0-9.eE]+).*?"
        r"\(linear: converged=([01]) iters=([0-9]+) rel=([-+0-9.eE]+)\)"
    )
    loop_re = re.compile(r"TimeLoop: loop\.run\(\) returned success=([01]) steps_taken=([0-9]+)")
    with path.open(errors="ignore") as fh:
        for line in fh:
            m = nonlinear_re.search(line)
            if m:
                # Trace mode logs identical TimeLoop summaries on every MPI rank.  Count each
                # nonlinear solve once while preserving distinct steps/attempts.
                record_key = m.groups()
                if record_key in seen_nonlinear_records:
                    continue
                seen_nonlinear_records.add(record_key)
                nonlinear_seen = True
                nonlinear_converged = (m.group(2) == "1")
                linear_converged = (m.group(5) == "1")
                nonlinear_all = nonlinear_all and nonlinear_converged
                linear_all = linear_all and linear_converged
                data["nonlinear_steps"] += 1
                data["newton_iters_total"] += int(m.group(3))
                data["linear_iters_total"] += int(m.group(6))
                data["max_residual"] = max(data["max_residual"], abs(float(m.group(4))))
                data["max_linear_rel"] = max(data["max_linear_rel"], abs(float(m.group(7))))
                continue
            m = loop_re.search(line)
            if m:
                data["loop_success"] = int(m.group(1))
                data["steps_taken"] = int(m.group(2))
    if nonlinear_seen:
        data["nonlinear_all_converged"] = int(nonlinear_all)
        data["linear_all_converged"] = int(linear_all)
    return data

for stat_path in sorted(perf_dir.glob("*.perf-stat.csv")):
    stem = stat_path.name.removesuffix(".perf-stat.csv")
    m = re.match(r"(.+?)_(.+)_rep([0-9]+)$", stem)
    if not m:
        continue
    mode, case_name, rep = m.group(1), m.group(2), int(m.group(3))
    row = {"mode": mode, "case": case_name, "rep": rep}
    row.update(parse_perf_stat(stat_path))
    stderr = perf_dir / f"{stem}.stderr.log"
    if stderr.exists():
        row.update(parse_solver_stderr(stderr))
    stdout = perf_dir / f"{stem}.stdout.log"
    if stdout.exists():
        row.update(parse_solver_stdout(stdout))
    rows.append(row)

fields = [
    "mode", "case", "rep", "task_clock_ms", "wall_time_loop_s", "wall_solve_s",
    "wall_vtk_s", "cycles", "instructions", "ipc",
    "cache_miss_pct", "l1d_load_miss_pct", "branch_miss_pct",
    "cache-misses_mpki", "L1-dcache-load-misses_mpki", "dTLB-load-misses_mpki",
    "assembly_blocks", "assembly_total_s", "assembly_cell_s", "assembly_boundary_s",
    "assembly_other_s", "fsils_solve_total_s", "gmres_total_s",
    "loop_success", "steps_taken", "nonlinear_steps", "nonlinear_all_converged",
    "newton_iters_total", "linear_iters_total", "linear_all_converged",
    "max_residual", "max_linear_rel",
]

summary = root / "summary.csv"
with summary.open("w", newline="") as fh:
    writer = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
    writer.writeheader()
    for row in rows:
        writer.writerow(row)

def median(vals):
    vals = [v for v in vals if isinstance(v, (int, float)) and not math.isnan(v)]
    return statistics.median(vals) if vals else ""

groups = {}
for row in rows:
    groups.setdefault((row["mode"], row["case"]), []).append(row)

med_fields = ["mode", "case", "runs"] + [f"{f}_median" for f in fields if f not in ("mode", "case", "rep")]
med_path = root / "summary_medians.csv"
with med_path.open("w", newline="") as fh:
    writer = csv.DictWriter(fh, fieldnames=med_fields)
    writer.writeheader()
    for (mode, case_name), group in sorted(groups.items()):
        out = {"mode": mode, "case": case_name, "runs": len(group)}
        for field in fields:
            if field in ("mode", "case", "rep"):
                continue
            out[f"{field}_median"] = median([r.get(field) for r in group])
        writer.writerow(out)

print("")
print(f"summary_csv={summary}")
print(f"summary_medians_csv={med_path}")
print("")
print("Median assembly summary:")
print("mode,case,runs,wall_s,task_s,assembly_s,assembly_pct,cell_pct,cache_miss_pct,l1d_miss_pct,cache_mpki,nonlinear_ok,linear_ok,newton_iters,linear_iters")
for (mode, case_name), group in sorted(groups.items()):
    task_ms = median([r.get("task_clock_ms") for r in group])
    wall_s = median([r.get("wall_time_loop_s") for r in group])
    asm_s = median([r.get("assembly_total_s") for r in group])
    cell_s = median([r.get("assembly_cell_s") for r in group])
    cache_pct = median([r.get("cache_miss_pct") for r in group])
    l1_pct = median([r.get("l1d_load_miss_pct") for r in group])
    cache_mpki = median([r.get("cache-misses_mpki") for r in group])
    nonlinear_ok = median([r.get("nonlinear_all_converged") for r in group])
    linear_ok = median([r.get("linear_all_converged") for r in group])
    newton_iters = median([r.get("newton_iters_total") for r in group])
    linear_iters = median([r.get("linear_iters_total") for r in group])
    task_s = task_ms / 1000.0 if task_ms != "" else 0.0
    asm_pct = 100.0 * asm_s / task_s if task_s else 0.0
    cell_pct = 100.0 * cell_s / asm_s if asm_s else 0.0
    print(f"{mode},{case_name},{len(group)},{wall_s:.3f},{task_s:.3f},{asm_s:.3f},{asm_pct:.1f},{cell_pct:.1f},{cache_pct:.2f},{l1_pct:.2f},{cache_mpki:.2f},{nonlinear_ok},{linear_ok},{newton_iters},{linear_iters}")
PY

echo "complete: $RESULTS_DIR"
