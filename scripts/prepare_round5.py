#!/usr/bin/env python3
"""Prepare an enlarged M10000 bank directly on cluster scratch."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRATCH = Path("/nfs_scratch/park687/D01Ritz_fermi_41bd5b5/round5")
BANK = SCRATCH / "M10000"
JOB = ROOT / "reports/round5_sampling.sub"
BANK.mkdir(parents=True, exist_ok=True)
(SCRATCH / "scheduler").mkdir(parents=True, exist_ok=True)
link = ROOT / "work/round5"
if link.is_symlink():
    assert link.resolve() == SCRATCH
else:
    assert not link.exists(), link
    link.symlink_to(SCRATCH, target_is_directory=True)
for seed in range(420401, 420417):
    source = ROOT / f"work/round2/M10000/train_{seed}"
    assert (source / "manifest.json").is_file()
    old = BANK / f"train_{seed}"
    if old.is_symlink():
        assert old.resolve() == source.resolve()
    else:
        assert not old.exists(), old
        old.symlink_to(source, target_is_directory=True)
rows = []
for stage, count, base in (("train", 112, 620400), ("holdout", 64, 720400)):
    for seed in range(base + 1, base + count + 1):
        assert not (BANK / f"{stage}_{seed}").exists()
        rows.append(f"10000 {stage} {seed} 1024")
content = """universe = vanilla
executable = /mnt/ceph/home/park687/D01Ritz_fermi_41bd5b5/scripts/chain_job.sh
arguments = --m $(m) --seed $(seed) --samples $(samples) --burn 4000 --stride 20 --stage $(stage) --out /nfs_scratch/park687/D01Ritz_fermi_41bd5b5/round5/M$(m)/$(stage)_$(seed)
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
output = /nfs_scratch/park687/D01Ritz_fermi_41bd5b5/round5/scheduler/$(Cluster).$(Process).out
error = /nfs_scratch/park687/D01Ritz_fermi_41bd5b5/round5/scheduler/$(Cluster).$(Process).err
log = /nfs_scratch/park687/D01Ritz_fermi_41bd5b5/round5/scheduler/$(Cluster).log
+JobBatchName = "D01Ritz-FERMI-round5-M10000-precision"
queue m,stage,seed,samples from (
""" + "\n".join(rows) + "\n)\n"
if JOB.exists():
    assert JOB.read_text() == content
else:
    JOB.write_text(content)
print(f"Prepared {len(rows)} new M10000 chains on {SCRATCH}")
