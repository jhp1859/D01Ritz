#!/bin/bash
set -euo pipefail
export OPENBLAS_NUM_THREADS="${ANALYSIS_THREADS:-1}" OMP_NUM_THREADS="${ANALYSIS_THREADS:-1}" MKL_NUM_THREADS="${ANALYSIS_THREADS:-1}"
cd /mnt/ceph/home/park687/D01Ritz_fermi_41bd5b5
m=$1
round=$2
extra_args=()
if [[ -n "${3:-}" ]]; then extra_args=(--frozen-fit "$3"); fi
/usr/bin/python3 scripts/analyze_ritz.py --root "work/$round/M$m" --out "reports/$round/M$m" "${extra_args[@]}"
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 /usr/bin/python3 scripts/estimate_norm.py --m "$m" --result "reports/$round/M$m" --samples "${NORM_SAMPLES:-4096}" --seed "${NORM_SEED:-6306100}"
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 /usr/bin/python3 scripts/audit_residual_uncertainty.py --m "$m" --round "$round"
