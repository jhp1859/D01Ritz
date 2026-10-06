#!/bin/bash
set -euo pipefail
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
cd /mnt/ceph/home/park687/D01Ritz_fermi_41bd5b5
m=$1
round=$2
/usr/bin/python3 scripts/analyze_ritz.py --root "work/$round/M$m" --out "reports/$round/M$m"
/usr/bin/python3 scripts/estimate_norm.py --m "$m" --result "reports/$round/M$m" --samples 4096
