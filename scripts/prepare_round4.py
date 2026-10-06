#!/usr/bin/env python3
"""Prepare additional independent Fermi-origin samples after round-3 audit."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ROUND = ROOT / "work/round4"
JOB = ROOT / "reports/round4_sampling.sub"
specs = (
    # M, old corrected training round, old seeds, new train chains, new holdout chains,
    # samples per new chain, train seed base, holdout seed base
    (100, "round2", range(420201, 420209), 24, 32, 4096, 620200, 720200),
    (1000, "round2", range(420301, 420309), 48, 48, 2048, 620300, 720300),
)
rows = []
for m, old_round, old_seeds, ntrain, nhold, samples, tbase, hbase in specs:
    dest = ROUND / f"M{m}"
    dest.mkdir(parents=True, exist_ok=True)
    for seed in old_seeds:
        source = ROOT / f"work/{old_round}/M{m}/train_{seed}"
        assert (source / "manifest.json").is_file(), source
        link = dest / f"train_{seed}"
        if link.is_symlink():
            assert link.resolve() == source.resolve()
        else:
            assert not link.exists(), link
            link.symlink_to(source, target_is_directory=True)
    for stage, n, base in (("train", ntrain, tbase), ("holdout", nhold, hbase)):
        for seed in range(base + 1, base + n + 1):
            out = dest / f"{stage}_{seed}"
            assert not out.exists(), out
            rows.append(f"{m} {stage} {seed} {samples}")

content = """universe = vanilla
executable = /mnt/ceph/home/park687/D01Ritz_fermi_41bd5b5/scripts/chain_job.sh
arguments = --m $(m) --seed $(seed) --samples $(samples) --burn 4000 --stride 20 --stage $(stage) --out /mnt/ceph/home/park687/D01Ritz_fermi_41bd5b5/work/round4/M$(m)/$(stage)_$(seed)
initialdir = /mnt/ceph/home/park687/D01Ritz_fermi_41bd5b5
should_transfer_files = YES
when_to_transfer_output = ON_EXIT
transfer_output_files = ""
getenv = False
requirements = (TARGET.IsOttenComputer =?= true) && ((TARGET.Machine == "qis1.hep.wisc.edu") || (TARGET.Machine == "qis3.hep.wisc.edu"))
+ProjectName = "UWMadison_Physics_Otten"
+ChtcProjects = "UWMadison_Physics_Otten"
HEP_VO = "otten"
request_cpus = 1
request_memory = 1GB
request_disk = 16MB
output = work/scheduler/$(Cluster).$(Process).out
error = work/scheduler/$(Cluster).$(Process).err
log = work/scheduler/$(Cluster).log
+JobBatchName = "D01Ritz-FERMI-round4-precision"
queue m,stage,seed,samples from (
""" + "\n".join(rows) + "\n)\n"
if JOB.exists():
    assert JOB.read_text() == content, JOB
else:
    JOB.write_text(content)
print(f"Prepared {len(rows)} new chains and preserved corrected prior training banks")
