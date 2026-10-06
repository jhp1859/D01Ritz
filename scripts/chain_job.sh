#!/bin/bash
set -euo pipefail
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
cd /mnt/ceph/home/park687/D01Ritz_fermi_41bd5b5
exec /usr/bin/python3 scripts/run_chain.py "$@"
