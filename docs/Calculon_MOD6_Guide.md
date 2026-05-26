# Running MOD6 on FHNW Calculon — Step-by-Step Guide

**Project:** Symbolic MOGP for FNN Weight Initialisation
**Author:** John M. Arellano
**Goal:** Execute the full 54-run MOD6 sweep in ~24–36 hours wall-clock instead of ~23 days.

---

## Strategy

Submit a Slurm **job array** of 18 tasks. Each task is one (topology, activation) pair running 3 random seeds, on its own compute node. All 18 tasks run concurrently. Because MOD6 already accepts `--activations` and `--topologies` filters as CLI args, no code changes are required.

| Item | Value |
|------|-------|
| Cluster | Calculon (FHNW HSI) |
| Partition | `top6` (newest Threadripper PRO 5955WX, Zen 3) |
| Tasks in array | 18 (= 6 activations × 3 topologies) |
| CPUs per task | 4 |
| Memory per task | 16 GB |
| Time limit per task | 3 days 12 hours (generous; should finish in 1–2 days) |
| Cluster hard time limit | 7 days 1 hour |

---

## Phase 1 — Get on Calculon

### 1.1 You must be on the FHNW network or VPN

Calculon's login node is only reachable from FHNW. If you're at home, connect via FHNW VPN first.

### 1.2 SSH from Windows

Win 11 has built-in OpenSSH. Open PowerShell or Git Bash and run:

```bash
ssh -i ~/.ssh/id_ed25519 john_arellano_students_fhnw@calculon.informatik.fhnw.ch
```

The first time you connect, you'll be asked to verify the host fingerprint — accept it.

If you don't have an SSH key yet, generate one on Windows:

```powershell
ssh-keygen -t ed25519 -C "john.arellano@students.fhnw.ch"
```

Then send the **public** key (`~/.ssh/id_ed25519.pub`) to the Calculon admin team via the FHNW HPC Teams channel.

### 1.3 Add an SSH shortcut (recommended)

Edit `~/.ssh/config` on your Windows machine (`C:\Users\John Arellano\.ssh\config`) and add:

```
Host calculon
    Hostname calculon.informatik.fhnw.ch
    Port 22
    User john_arellano_students_fhnw
    IdentityFile ~/.ssh/id_ed25519
```

Now you can just type `ssh calculon` instead of the full command.

---

## Phase 2 — Get your project onto Calculon

You have two options. **Git is strongly preferred** — it makes iterating on code painless.

### Option A — Git (recommended)

If your project is already in a git repo, just clone it:

```bash
# On Calculon
cd ~
git clone <your_repo_url> Applied_Comp_Intel_Project
cd Applied_Comp_Intel_Project
```

If it's not in git yet, this is a good moment to put it in one (GitHub, GitLab, or FHNW's internal GitLab). You can then `git pull` whenever you fix something locally.

### Option B — SCP (fallback)

From Windows PowerShell on your local machine:

```powershell
# Zip the project (excluding heavy caches)
cd "C:\Users\John Arellano\PycharmProjects"
Compress-Archive -Path Applied_Comp_Intel_Project -DestinationPath project.zip

# Transfer to Calculon
scp project.zip calculon:~/
```

Then on Calculon:

```bash
cd ~
unzip project.zip
mv Applied_Comp_Intel_Project Applied_Comp_Intel_Project  # rename if needed
rm project.zip
```

### A note on storage

- `/home2/$USER` is your home directory — fine for the project itself
- **No automatic backups** — keep your authoritative copy on your local machine (or in git)
- Large datasets (>1 GB) should go in `/mnt/nas05/data01/` (request via the FHNW HPC Teams channel)
- The OpenML cache MOD3 builds will be downloaded fresh on first run; that's fine

---

## Phase 3 — Set up the Python environment

Calculon has conda available. Use it.

```bash
# On Calculon, from your project directory
cd ~/Applied_Comp_Intel_Project

# Create a conda env matching your Windows setup
conda create -n aci_project python=3.12 -y

# Install your project dependencies
conda run -n aci_project pip install \
    torch==2.5.0 \
    deap==1.4.3 \
    scikit-learn==1.8 \
    openml \
    numpy pandas scipy sympy matplotlib \
    pydantic

# OR if you have a requirements.txt
conda run -n aci_project pip install -r requirements.txt
```

**Verify the install** by importing the heavy packages:

```bash
conda run -n aci_project python -c "
import torch, deap, sklearn, openml, sympy
print('PyTorch :', torch.__version__)
print('DEAP    :', deap.__version__)
print('sklearn :', sklearn.__version__)
print('OpenML  :', openml.__version__)
print('SymPy   :', sympy.__version__)
"
```

Adjust the version pins above to whatever your project actually uses. If you have any trouble matching versions, your MOD0 bootstrap script can validate the environment:

```bash
conda run -n aci_project python experiment_modules/MOD0_dependency_bootstrap.py
```

---

## Phase 4 — Pre-flight: run the prerequisite pipeline + smoke test

### 4.1 Run MOD1 → MOD2 → MOD3 once on the login node

**Important:** MOD6 has a pre-flight check that refuses to start if MOD2's normalisation parameters are missing on disk. Running these three modules creates the OpenML dataset selection, computes the 8 meta-features, builds the z-score normalisation params, and warms the tensor cache — all of which MOD6 reads at startup.

These three modules are mostly I/O bound (network + disk), not CPU heavy, so they're fine to run on the login node. Each takes a few minutes.

```bash
cd ~/Applied_Comp_Intel_Project/experiment_modules

conda run --no-capture-output -n aci_project python MOD1_pipeline_selector.py
conda run --no-capture-output -n aci_project python MOD2_pipeline_meta_extractor.py
conda run --no-capture-output -n aci_project python MOD3_pm_dataset_manager.py
```

If any of these fail, fix them before submitting the sweep. The error messages should be clear (missing packages → re-check your conda env; network errors → check VPN; permission errors → check directory ownership).

### 4.2 Submit the smoke test

The smoke test runs one tiny GP search (population 20, 1 seed, `rectification` × `shallow` only) to verify everything works end-to-end on a compute node before you commit to the full sweep.

```bash
# From the project root
cd ~/Applied_Comp_Intel_Project
mkdir -p logs

# Submit
sbatch scripts/mod6_smoke_test.sbatch
```

You'll get back `Submitted batch job 12345`. Watch it:

```bash
squeue -u $USER                    # check status
tail -f logs/smoke_12345.log       # tail the log as it runs
```

If the smoke test finishes with `Exit code: 0` and produces a rule file in `experiment_modules/generated_files/GA_rule_files/`, you're ready for the full sweep.

---

## Phase 5 — Launch the full sweep

```bash
sbatch scripts/mod6_sweep.sbatch
```

This submits the 18-task array. You'll get one job ID, and Slurm will internally manage the 18 tasks.

### Check it's running

```bash
# Show your jobs
squeue -u $USER

# Show all 18 array tasks
squeue -u $USER -t all

# Estimate start time if anything is pending
squeue --start -u $USER
```

You can also see node-level load:

```bash
spartition
```

---

## Phase 6 — Monitor the sweep

### Tail individual task logs

Each array task writes to `logs/mod6_<ARRAY_JOB_ID>_<TASK_ID>.log`. To follow one:

```bash
tail -f logs/mod6_12345_0.log   # task 0 = (linear, shallow)
tail -f logs/mod6_12345_7.log   # task 7 = (squashing, deep_narrow)
```

The task index decodes as:

| Task ID | Activation | Topology |
|--------:|-----------|----------|
| 0  | linear | shallow |
| 1  | linear | deep_narrow |
| 2  | linear | funnel |
| 3  | rectification | shallow |
| 4  | rectification | deep_narrow |
| 5  | rectification | funnel |
| 6  | squashing | shallow |
| 7  | squashing | deep_narrow |
| 8  | squashing | funnel |
| 9  | smooth | shallow |
| 10 | smooth | deep_narrow |
| 11 | smooth | funnel |
| 12 | aggregation | shallow |
| 13 | aggregation | deep_narrow |
| 14 | aggregation | funnel |
| 15 | trigonometric | shallow |
| 16 | trigonometric | deep_narrow |
| 17 | trigonometric | funnel |

### Check resource efficiency

While the sweep runs, you can confirm that you're actually using the CPUs you asked for:

```bash
sstat -j 12345_0 --format=JobID,AveCPU,MaxRSS
```

After a task finishes:

```bash
sacct -j 12345_0 --format=JobID,State,Elapsed,CPUEff,MemEff,MaxRSS
```

If `CPUEff` is below ~80%, the cores aren't being utilised — meaning DEAP isn't parallelising. For now this is fine because you're running 18 jobs in parallel anyway. If you later want each task to use more cores, you'd add `multiprocessing.Pool` to the DEAP toolbox in MOD6 (separate exercise).

### Email notifications

The SBATCH script has `--mail-type=END,FAIL` set, so you'll get an email at `john.arellano@students.fhnw.ch` when each task finishes or fails.

---

## Phase 7 — Retrieve results

When all 18 tasks finish (you'll know because `squeue -u $USER` shows nothing), copy the output files back to your local machine. MOD6 writes its rule files to `experiment_modules/generated_files/GA_rule_files/`:

```powershell
# From Windows PowerShell on your local machine
scp -r calculon:~/Applied_Comp_Intel_Project/experiment_modules/generated_files `
    "C:\Users\John Arellano\PycharmProjects\Applied_Comp_Intel_Project\generated_files_calculon"
```

You'll get 18 `Final_Discovered_Rules_*.txt` files (one per topology-activation pair) plus a `per_run_archive/` directory containing the per-seed intermediates.

---

## Troubleshooting

### "Pending" jobs that won't start

```bash
squeue -j 12345 --start
```

Common reason: the cluster is busy. The `top6` partition only has 6 nodes; if all are occupied, some of your 18 tasks will queue. They'll start as soon as nodes free up. The sweep elapsed time is set by the slowest task, not the queue wait — so this rarely blocks completion.

### Out of memory (exit code 137)

If a task dies with exit code 137, it ran out of memory:

```bash
sacct -j 12345_5 --format=JobID,MaxRSS,ReqMem
```

If `MaxRSS` is close to `ReqMem`, bump `--mem=16G` to `--mem=32G` in the SBATCH script and resubmit only the failed task:

```bash
sbatch --array=5 scripts/mod6_sweep.sbatch
```

### Time limit exceeded

If a task hits the 3.5-day wall (very unlikely on Calculon hardware), increase `--time` and resubmit:

```bash
sbatch --array=<failed_task_id> --time=7-00:00:00 scripts/mod6_sweep.sbatch
```

### Conda env not activating in the job

If you see `conda: command not found` in the logs, conda isn't on the PATH in non-interactive shells. Add this line to the top of the SBATCH script (right after the `#SBATCH` directives):

```bash
source /opt/miniconda3/etc/profile.d/conda.sh   # or wherever conda lives on Calculon
```

Or just use the full path: `~/miniconda3/bin/conda run -n aci_project python ...`

### "Module not found" inside the job but works on login

The compute node may have a slightly different environment. Always use `conda run --no-capture-output -n aci_project python ...` inside the job script — never rely on `conda activate` in non-interactive shells (the docs explicitly warn against this).

---

## Etiquette

- Calculon is shared. The `--array=0-17` line runs all 18 tasks concurrently. If the cluster is busy, change it to `--array=0-17%6` to throttle to 6 concurrent tasks at a time (still finishes in roughly 2x the time of the unthrottled version).
- Don't run intensive work on the login node. Always go through Slurm.
- Clean up large temp files in `/home2/$USER` when you're done.

---

## Quick command reference

```bash
sbatch scripts/mod6_smoke_test.sbatch        # validate setup
sbatch scripts/mod6_sweep.sbatch             # launch full sweep
squeue -u $USER                              # what's running
squeue --start -u $USER                      # when will pending jobs start
sacct -j 12345 --format=JobID,State,Elapsed  # results summary
scancel 12345                                # cancel a job
scancel -u $USER                             # cancel ALL my jobs (use carefully)
spartition                                   # cluster-wide node status
tail -f logs/mod6_12345_0.log                # follow a task
```
