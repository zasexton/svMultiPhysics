#!/usr/bin/env bash
# Run perf_microbench_one wrapped with perf stat for each (element, op) pair.
# Parses output and writes CSV(s) consumed by the plot scripts:
#   data/perf/perf_hw_counters.csv  — raw counters per (element, op)
#
# Designed to be safe even when some perf events are unsupported on the host:
# unsupported events are simply omitted from the counter set.

set -u

BIN="${PERF_MICROBENCH_BIN:-/home/zack/Downloads/svMultiPhysics/build-unit/svMultiPhysics-build/bin/perf_microbench_one}"
if [[ -n "${SVMP_BASIS_COMPARE_OUT:-}" ]]; then
    OUT_DIR="${SVMP_BASIS_COMPARE_OUT}/perf"
else
    OUT_DIR="$(cd "$(dirname "$0")" && pwd)/data/perf"
fi
mkdir -p "$OUT_DIR"

ITERATIONS=${ITERATIONS:-20000000}
ELEMENTS=(TRI3 TRI6 TET4 TET10 HEX8 HEX27)
OPS=(values gradients hessians all)

EVENTS="cycles,instructions,L1-dcache-loads,L1-dcache-load-misses,LLC-loads,LLC-load-misses,branches,branch-misses,task-clock"

# Force single-threaded execution so per-cycle counters reflect a single core.
export OMP_NUM_THREADS=1
# Pin to a single core via taskset to avoid migration-induced cycle inflation.
TASKSET="taskset -c 0"

CSV="$OUT_DIR/perf_hw_counters.csv"
echo "elem_type,op,iterations,seconds,ns_per_call,cycles,instructions,l1d_loads,l1d_load_misses,llc_loads,llc_load_misses,branches,branch_misses,task_clock_ms" > "$CSV"

for elem in "${ELEMENTS[@]}"; do
    for op in "${OPS[@]}"; do
        echo "[perf-stat] $elem $op iterations=$ITERATIONS"
        out=$(perf stat -e "$EVENTS" -- $TASKSET "$BIN" "$elem" "$op" "$ITERATIONS" 2>&1) || {
            echo "  perf stat failed for $elem $op"
            continue
        }

        # Parse the binary's own '#' line for ns_per_call/seconds.
        bench_line=$(echo "$out" | grep -E '^#' | head -n1)
        seconds=$(echo "$bench_line" | sed -nE 's/.*seconds=([0-9.eE+-]+).*/\1/p')
        ns_per_call=$(echo "$bench_line" | sed -nE 's/.*ns_per_call=([0-9.eE+-]+).*/\1/p')

        # Parse perf-stat output for each event. perf stat uses comma separators
        # for thousands; strip them.
        get_counter() {
            local label="$1"
            echo "$out" | grep -E "^[[:space:]]*[0-9.,<not supported>]+[[:space:]]+$label" | \
                head -n1 | awk '{print $1}' | tr -d ','
        }

        cycles=$(get_counter "cycles")
        instructions=$(get_counter "instructions")
        l1d_loads=$(get_counter "L1-dcache-loads")
        l1d_load_misses=$(get_counter "L1-dcache-load-misses")
        llc_loads=$(get_counter "LLC-loads")
        llc_load_misses=$(get_counter "LLC-load-misses")
        branches=$(get_counter "branches")
        branch_misses=$(get_counter "branch-misses")
        task_clock=$(echo "$out" | grep -E '^[[:space:]]*[0-9.,]+[[:space:]]+msec[[:space:]]+task-clock' | \
            awk '{print $1}' | tr -d ',')

        # Default unsupported counters to empty.
        for var in cycles instructions l1d_loads l1d_load_misses llc_loads llc_load_misses branches branch_misses task_clock; do
            v="${!var}"
            if [[ -z "$v" || "$v" == "<not" ]]; then
                eval "$var="
            fi
        done

        echo "$elem,$op,$ITERATIONS,$seconds,$ns_per_call,$cycles,$instructions,$l1d_loads,$l1d_load_misses,$llc_loads,$llc_load_misses,$branches,$branch_misses,$task_clock" >> "$CSV"
    done
done

echo "Wrote $CSV"
