#!/bin/bash

# ============================================================
#  Compile & run all reduction_op*.cu kernels
#  Results are stored under ./final_log/
# ============================================================

set -euo pipefail

# ---------- Configuration ----------
SRC_DIR="."                       # directory that contains the .cu files
LOG_DIR="./final_log"
NVCC="nvcc"
NVCC_FLAGS="-O3 -std=c++14 -arch=sm_75"   # adjust -arch to your GPU if needed
# -----------------------------------

mkdir -p "${LOG_DIR}"

echo "============================================================"
echo "  Compiling & running all reduction kernels"
echo "  Log directory : ${LOG_DIR}"
echo "============================================================"
echo

# Find all matching source files (sorted)
mapfile -t SOURCES < <(find "${SRC_DIR}" -maxdepth 1 -name 'reduction_op*.cu' | sort)

if [ ${#SOURCES[@]} -eq 0 ]; then
    echo "ERROR: No reduction_op*.cu files found in ${SRC_DIR}"
    exit 1
fi

for src in "${SOURCES[@]}"; do
    base=$(basename "${src}" .cu)          # e.g. reduction_op5
    bin="./${base}"                        # binary name
    log="${LOG_DIR}/${base}.log"           # log file

    echo "--------------------------------------------------------"
    echo "  Source  : ${src}"
    echo "  Binary  : ${bin}"
    echo "  Log     : ${log}"
    echo "--------------------------------------------------------"

    # ---- Compile ----
    echo "[COMPILE] ${src} ..."
    if ! ${NVCC} ${NVCC_FLAGS} "${src}" -o "${bin}" 2>&1 | tee -a "${log}"; then
        echo "  ✗ Compilation failed – see ${log}"
        echo
        continue
    fi
    echo "  ✓ Compilation succeeded"

    # ---- Run ----
    echo "[RUN]     ${bin} ..."
    {
        echo "===== RUN OUTPUT ====="
        echo "Command : ${bin}"
        echo "Date    : $(date)"
        echo "---------------------"
        ./"${bin}"
        echo "---------------------"
        echo "Exit code: $?"
    } >> "${log}" 2>&1

    echo "  ✓ Finished – output stored in ${log}"
    echo
done

echo "============================================================"
echo "  All done.  Logs are in ${LOG_DIR}/"
echo "============================================================"
ls -l "${LOG_DIR}"