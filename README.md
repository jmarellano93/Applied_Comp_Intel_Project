# Applied Computational Intelligence Project — Software Documentation

**Project:** Symbolic Discovery and Optimization of Data-Aware Weight-Initialization Rules for Feed-Forward Neural Network Classification — A Multi-Objective Genetic Programming Approach

**Author:** John M. Arellano

---

## 1. Overview

This project frames neural-network weight initialization as a multi-objective symbolic-regression problem. Genetic programming under NSGA-II selection (implemented with DEAP) discovers compact symbolic rules that map eight standardized dataset meta-features to an initialization variance, separately for each (topology, activation) cell. The discovered rules are then validated against six established baselines (Xavier-Glorot, He-Kaiming, LeCun, Orthogonal, FAVI, Laor) on held-out OpenML CC-18 datasets, with pooled and cluster-stratified statistical testing and a qualitative analysis of the discovered symbolic forms.

The codebase is organized as a linear pipeline of eleven numbered modules (MOD0 through MOD10). Each module is a self-contained, runnable Python script with a Pydantic configuration object, a clearly defined input contract, and a defined output artifact. Modules communicate through files on disk rather than through in-process calls, which makes each stage independently runnable, restartable, and auditable, and makes the heavy stages (discovery and validation) straightforward to dispatch to a high-performance-computing cluster.

The experimental grid consists of fifteen cells: five activation families (linear, rectification, squashing, smooth, aggregation) crossed with three topologies (shallow, deep_narrow, funnel). Genetic-programming discovery is replicated across three independent seeds per cell. A trigonometric activation family is present in the code as a sixth family but was excluded from the reported grid for scheduling reasons; the scripts retain it as an available option.

### 1.1 Pipeline at a glance

The modules execute in numerical order. The table below summarizes the role of each and its primary input and output.

| Module | Role | Primary input | Primary output |
|---|---|---|---|
| MOD0 | Dependency bootstrap and environment verification | None (reads the running interpreter) | A verified, dependency-complete Python environment |
| MOD1 | OpenML CC-18 dataset selection and download | OpenML API | Cleaned dataset CSVs plus a download log |
| MOD2 | Meta-feature extraction and Phase A/B partition | Dataset CSVs | Per-dataset meta-features, Phase A/B split, normalization parameters |
| MOD3 | Dataset ingestion and tensor cache | Dataset CSVs, normalization parameters | Leakage-free train/validation tensors per dataset |
| MOD4 | FNN landscape (problem model) | Tensors from MOD3 | Trained-network accuracy, loss, and epochs for a given initialization |
| MOD5 | GP engine — prototype (scaled) | Phase A tensors and meta-features | Small-scale consensus rule artifacts for a fast end-to-end check |
| MOD6 | GP engine — production | Phase A tensors and meta-features | Consensus Pareto rule artifacts per cell |
| MOD7 | Validation matrix (and driver) | Consensus rules, Phase B tensors | Per-cell JSON validation results |
| MOD8 | Statistical reporter | MOD7 JSON results | Pooled and cluster-stratified tables and figures |
| MOD9 | Qualitative analyzer | Consensus rules, Phase A meta-features | Symbolic derivative analyses and sensitivity plots |
| MOD10 | Post-hoc analyzer | MOD7 results and MOD9 rules | Effect sizes, He-distance, taxonomy, and Pareto-rank diagnostics |

---

## 2. Per-Script Reference

For each module below: **(A)** what it is, **(B)** what it does, **(C)** selected methodological choices, and where relevant the unit-test and harness files associated with it.

### MOD0 — Dependency Bootstrap and Environment Verification
*File:* `MOD0_dependency_bootstrap.py` · *Tests:* none

**(A) What it is.** An idempotent installer and verifier for every third-party package the pipeline requires.

**(B) What it does.** It checks the running interpreter version against a pinned minimum, then iterates a manifest of packages, installing or upgrading any that are missing or version-mismatched, and prints a summary of the resulting environment. It can be re-run safely; already-correct packages are reported and skipped.

**(C) Methodological choices.** Packages are installed with per-package, atomic pip calls rather than a single bundled install. This is required because the PyTorch wheel is served from a dedicated index URL, and applying that index at the batch level would contaminate resolution of the other packages. The manifest decouples the pip distribution name from the import name (for example, scikit-learn versus sklearn) so that the installed-versus-required check is reliable.

### MOD1 — OpenML CC-18 Pipeline Selector
*File:* `MOD1_pipeline_selector.py` · *Tests:* `MOD1_unit_test.py`

**(A) What it is.** The data-acquisition stage that selects and downloads the candidate dataset pool from the OpenML CC-18 benchmark suite.

**(B) What it does.** It fetches the suite metadata, filters datasets by instance-count and feature-count bounds, downloads each surviving dataset, removes rows with missing values and constant (zero-variance) columns, and writes a cleaned CSV per dataset together with a download log.

**(C) Methodological choices.** Datasets are constrained to a moderate feature count so that no single dataset forces an unusually wide first hidden layer, which keeps the network family comparable across datasets. Cleaning drops NaNs and zero-variance columns up front so that downstream meta-feature extraction and training are numerically stable. The OpenML API key is read from configuration; the committed default is a placeholder that the user supplies at runtime (see Section 4.2).

### MOD2 — Pipeline Meta-Extractor
*File:* `MOD2_pipeline_meta_extractor.py` · *Tests:* `MOD2_unit_test.py`

**(A) What it is.** The feature-engineering stage that computes the eight dataset meta-features and defines the Phase A / Phase B split.

**(B) What it does.** For each cleaned dataset it extracts the eight meta-features (the n-to-d ratio, feature kurtosis, an IQR-based dispersion measure, a principal-eigenvalue summary, target entropy, the Hopkins statistic, the silhouette score, and the Davies-Bouldin index), partitions the datasets into a Phase A discovery set and a Phase B validation set, and persists the per-feature normalization parameters.

**(C) Methodological choices.** Normalization parameters (per-feature mean and standard deviation) are fit on Phase A only, never on the full pool, so that no information from the validation datasets leaks into the discovery process. A near-zero standard deviation is floored to a small epsilon so the downstream z-score transform can never divide by zero. The Hopkins statistic is computed with a vectorized sampling procedure rather than a Python loop for efficiency.

### MOD3 — Dataset Ingestion and Tensor Cache Manager
*File:* `MOD3_pm_dataset_manager.py` · *Tests:* `MOD3_unit_test.py`

**(A) What it is.** The class that turns a cleaned dataset CSV into leakage-free PyTorch tensors and caches the result.

**(B) What it does.** It reads a dataset offline, builds a preprocessing pipeline that is fit on training data only, applies the stored Phase A normalization parameters, encodes targets to integer class indices, and returns train and validation tensors plus the dataset's meta-feature vector. Results are memoized per dataset id.

**(C) Methodological choices.** The preprocessing pipeline is fit strictly on the training partition and only then applied to the validation partition, enforcing separation between the two. Features are cast to 32-bit floats and targets to 64-bit integers to match the requirements of the network's linear layers and the cross-entropy loss. When the normalization-parameters file is absent the manager emits a warning and falls back to raw values, which preserves compatibility with rule artifacts produced before normalization was introduced.

### MOD4 — FNN Landscape (Problem Model)
*File:* `MOD4_pm_fnn_landscape.py` · *Tests:* `MOD4_unit_test.py`

**(A) What it is.** The problem model: the feed-forward-network definitions and the trainer that evaluates a given initialization variance.

**(B) What it does.** It defines the three network topologies and an activation router, then trains a network under a specified initialization variance and reports balanced accuracy, terminal loss, and the number of epochs needed to reach a target accuracy. An amortized trainer reuses one network per dataset across many evaluations.

**(C) Methodological choices.** Network architectures are fixed per topology so that the initialization variance is the only independent variable affecting the training trajectory, giving strict experimental control. The trainer validates input tensor dimensions before entering the PyTorch execution graph so that malformed inputs fail fast with a clear error. A single-shot evaluator is retained alongside the amortized trainer to support isolated validation and testing contexts.

### MOD5 — MOGP Engine (Prototype)
*File:* `MOD5_om_mogp_engine_prototype.py` · *Tests:* `MOD5_unit_test.py`

**(A) What it is.** A scaled-down copy of the production genetic-programming engine intended for a fast end-to-end smoke test.

**(B) What it does.** It runs the full discovery workflow — primitive set, multi-objective evolution, multi-seed consensus aggregation, and artifact export — but with small population, generation, and dataset-per-rule settings so the entire pipeline contract can be exercised quickly before committing to a long production run.

**(C) Methodological choices.** Only the outer-loop budget is reduced; the inner training depth (epoch budget) is unchanged, so the prototype still exercises the real epochs-to-target dynamic. The prototype writes to a separate testing output directory so its small-scale artifacts never commingle with production output. Its objective contract and all behavioral switches mirror the production engine exactly, so a regression that breaks one would break both.

### MOD6 — MOGP Meta-Grid Search Engine (Production)
*File:* `MOD6_om_mogp_engine_final.py` · *Tests:* `MOD6_unit_test.py`

**(A) What it is.** The production genetic-programming engine that discovers the symbolic initialization rules.

**(B) What it does.** For each (topology, activation) cell it evolves a population of symbolic trees under NSGA-II selection against three objectives — maximize balanced accuracy, minimize epochs-to-target, and minimize tree size — across the Phase A discovery datasets, repeats this for the configured seeds, aggregates the per-seed Pareto fronts into a single consensus front, and writes the ranked consensus rules to a text artifact. It also supports reconstructing the consensus from previously written per-seed files without re-running discovery.

**(C) Methodological choices.** Selection uses NSGA-II non-dominated sorting with crowding distance, so the three objectives are traded off by Pareto dominance rather than by a fragile fixed weighting. The function set uses protected operators (guarded division, square root, logarithm, and a clipped exponential) so that no candidate rule raises on degenerate input. A static tree-height limit controls bloat. The consensus aggregator deduplicates rules by their exact string form and averages the objective values of duplicates, so a rule found by several seeds is represented once and its fitness reflects all of them. The optional single-objective mode is disabled by default; the reported results use the multi-objective Pareto configuration.

### MOD7 — Framework Validation Matrix and Driver
*Files:* `MOD7_framework_validation_matrix.py`, `MOD7_pipeline_driver.py`, `MOD7_framework_validation_matrix_prototype.py` · *Tests:* `MOD7_unit_test.py`, `MOD7_pipeline_driver_unit_test.py`, `MOD7_integration_verification.py`

**(A) What it is.** The out-of-sample validation stage. The matrix script evaluates rules against the baselines on the Phase B datasets; the driver orchestrates the matrix across all cells.

**(B) What it does.** For each cell the matrix recompiles the discovered rule strings, applies the resulting initialization variance and each of the six baseline schemes to freshly trained networks across the Phase B datasets and trial seeds, and emits a per-cell JSON artifact summarizing aggregates, raw per-trial distributions, and a binomial win-matrix. The driver scans the consensus rule directory, selects the correct artifact per (activation, topology) pair, and invokes the matrix once per cell, forwarding a quick-test flag for fast pipeline checks.

**(C) Methodological choices.** The rule-recompilation primitives reproduce the exact operator semantics used at discovery time, so a serialized rule evaluates identically in validation. The six baselines span the variance, orthogonality, and data-aware families, which is sufficient for a representative comparison; a non-canonical single-pass scheme that had been considered is not part of the roster. Significance per cell uses a one-tailed binomial test of the hypothesis that the discovered rule wins a majority of paired trials. An amortized trainer is reused per Phase B dataset across all rules, baselines, and seeds, mirroring the efficiency pattern in MOD6.

### MOD8 — Framework Statistical Reporter
*File:* `MOD8_framework_statistical_reporter.py` · *Tests:* `MOD8_and_MOD9_unit_test.py`

**(A) What it is.** The reporting stage that turns MOD7's JSON artifacts into publication-ready tables and figures.

**(B) What it does.** It produces pooled Wilcoxon tables per cell, distribution figures, and a cluster-stratified report in which the Phase B datasets are grouped by K-means on their normalized meta-features and the rule-versus-baseline comparisons are repeated within each cluster with Holm-Bonferroni correction. It also writes cluster diagnostics and a PCA visualization.

**(C) Methodological choices.** The Wilcoxon helper short-circuits identical or empty input to a neutral result rather than raising, so degenerate cells are handled gracefully. The cluster count is selected by silhouette score over a small range of candidate values. Multiple-comparison control within each cluster uses the Holm-Bonferroni step-down procedure. The clustering is fit on the same normalized meta-feature space used throughout the project, so the cluster definitions are consistent with discovery.

### MOD9 — Framework Qualitative Analyzer
*File:* `MOD9_framework_qualitative_analyzer.py` · *Tests:* `MOD8_and_MOD9_unit_test.py`

**(A) What it is.** The qualitative stage that inspects the symbolic structure of the discovered rules.

**(B) What it does.** For every consensus rule it parses the equation into a symbolic expression, computes partial-derivative summaries, and renders one-dimensional sensitivity curves or two-dimensional surfaces in which non-dominant features are pinned at their Phase A empirical means and the two most influential features become the free axes.

**(C) Methodological choices.** The symbolic parser registers the trigonometric and protected operators directly by name because the rule files serialize primitives by their function name; a direct symbol mapping is therefore required. Dominant features are chosen by the magnitude of the partial derivative evaluated at the empirical mean. Sensitivity is plotted over a standardized range centered on zero, reflecting the z-scored meta-feature space. All ranks of each consensus front are analyzed, not only the top rank.

### MOD10 — Post-Hoc Analyzer
*File:* `MOD10_post_hoc_analyzer.py` · *Tests:* `MOD10_unit_test.py`, `MOD10_validate.py`

**(A) What it is.** The post-hoc stage that quantifies practical significance and characterizes the discovered rules structurally.

**(B) What it does.** It loads MOD7 results and MOD9 rules and computes paired effect sizes (Cliff's delta with bootstrap confidence intervals), the distance of each rule's variance from its He target, a structural failure-mode taxonomy of the rules, a Pareto-rank cross-comparison that checks whether the selected top rank is the best loss representative of its cell, and a cluster-stratified accuracy analysis. An orchestrator runs the enabled analyses end to end and degrades gracefully when inputs are absent.

**(C) Methodological choices.** Statistical primitives are isolated and reused: a zero-variance-safe Wilcoxon test, Cliff's delta for effect magnitude, a paired bootstrap for the median-difference interval, and Holm-Bonferroni for multiple-comparison control. The He-distance analysis expresses each rule's median variance as a ratio to the He target for its fan-in, locating it relative to the canonical scaling. The taxonomy classifies each simplified expression by inspecting its free symbols, operator shapes, and leading-coefficient scale. Loaders declare their output schema explicitly so that an empty input directory yields a well-formed empty table rather than an error.

### Test and harness files

The unit tests (`MOD*_unit_test.py`, `MOD8_and_MOD9_unit_test.py`, `MOD7_pipeline_driver_unit_test.py`) validate each module's public contract — configuration defaults and validation, numerical correctness, return shapes, and error handling — and are intended to be run with pytest. `MOD7_integration_verification.py` checks the command-line contract between the driver and the validation matrix by invoking them as subprocesses. `MOD10_validate.py` is a standalone sanity harness that runs the MOD10 numeric helpers and taxonomy without requiring pytest or pydantic; it is a quick check, not a substitute for the unit tests.

---

## 3. Environment and Installation

### 3.1 Prerequisites

A recent CPython interpreter (the bootstrap pins a minimum and a recommended version) and network access to PyPI and OpenML. PyTorch is used in CPU mode throughout; no GPU is required. On the HPC cluster the environment is managed with conda under the environment name `aci_project`.

### 3.2 Local installation

From the project root:

```
# 1. (recommended) create and activate an isolated environment
python -m venv .venv
# Windows:  .venv\Scripts\activate
# Linux/macOS:  source .venv/bin/activate

# 2. install and verify all dependencies
python experiment_modules/MOD0_dependency_bootstrap.py

# to force a clean re-install of everything:
python experiment_modules/MOD0_dependency_bootstrap.py --upgrade
```

MOD0 installs the scientific stack (NumPy, SciPy, pandas), the machine-learning stack (scikit-learn, PyTorch), the evolutionary-computation library (DEAP), symbolic mathematics (SymPy), data ingestion (OpenML), validation (Pydantic v2), visualization (Matplotlib, Seaborn), progress reporting (tqdm), and the test framework (pytest). A non-zero exit code indicates the interpreter is below the minimum version or a package failed to install.

### 3.3 Conda environment for the cluster

The Slurm scripts assume a conda environment named `aci_project` and invoke Python through it:

```
conda create -n aci_project python=3.12
conda activate aci_project
python experiment_modules/MOD0_dependency_bootstrap.py
```
### 3.4 Placeholders for User-Specific Information:

Search for these placeholders in the project and insert your user specific information before beginning to ensure the Modules run smoothly.

OPENML_API_KEY_HERE

FHNW_EMAIL_ADDRESS_HERE

---

## 4. Running the Pipeline

### 4.1 Directory expectations

Scripts resolve their input and output directories relative to their own location. The data-preparation stages create and populate `generated_files/` (download log, Phase A and Phase B dataset lists, normalization parameters) and `openml_cc18_datasets/` (cleaned dataset CSVs). Discovery writes consensus rule artifacts to `generated_files/GA_rule_files/` (production) or `generated_files/GA_rule_files_testing/` (prototype). Validation, reporting, and post-hoc analysis write under `experimental_results_analysis_visualizations/` in `reports/` and `visualizations/` subfolders named per module.

### 4.2 Supplying the OpenML API key

MOD1 reads the OpenML API key from its configuration. The committed default is a placeholder. Provide your own key by editing the configuration default locally, or by setting it in your OpenML configuration file, before running MOD1. Do not commit a real key to version control.

### 4.3 Local end-to-end run (small scale)

The following sequence runs the whole pipeline locally using the prototype discovery engine, which is sized for a single sitting:

```
cd experiment_modules

python MOD1_pipeline_selector.py            # download + clean datasets
python MOD2_pipeline_meta_extractor.py      # meta-features, Phase A/B split, norm params
python MOD5_om_mogp_engine_prototype.py     # scaled discovery -> GA_rule_files_testing/
python MOD7_pipeline_driver.py \
    --rule_directory generated_files/GA_rule_files_testing \
    --quick_test                            # fast validation pass
python MOD8_framework_statistical_reporter.py
python MOD9_framework_qualitative_analyzer.py
python MOD10_post_hoc_analyzer.py
```

MOD3 and MOD4 are libraries consumed by the discovery and validation stages; they are exercised through their unit tests rather than run directly.

### 4.4 Running the unit tests

```
cd unit_tests
python -m pytest -v                         # all tests
python -m pytest MOD10_unit_test.py -v      # a single module's tests
python MOD10_validate.py                    # standalone MOD10 sanity harness
```

---

## 5. Running Modules 1–7 on the Supercomputer (FHNW Calculon)

The discovery (MOD6) and validation (MOD7) stages are the computationally heavy parts of the experiment and are run on **Calculon**, the high-performance-computing cluster of the School of Computer Science (HSI) at FHNW, operated by the Institute for Data Science (i4Ds). Calculon uses the **Slurm** workload manager to queue jobs and allocate CPUs, memory, and (optionally) GPUs across its compute nodes. This experiment is CPU-only; no GPU is required.

This section is intended to be followed end to end by someone who has never used the cluster before. It covers, in order: obtaining an account, connecting, preparing the software environment, transferring the project, and submitting each stage of the pipeline. Light stages (MOD1, MOD2, MOD8, MOD9, MOD10) are also described so the whole experiment can be reproduced on the cluster.

> The authoritative reference is the official Calculon User Guide at `https://fhnw-hpc.pages.fhnw.ch/docs/`. The account, connection, partition, and environment details below summarize that guide as of early 2026; if anything has changed, the guide is the source of truth.

### 5.1 Getting a Calculon account

Access is not automatic and must be requested from the Calculon team. The cluster is reachable **only from inside the FHNW network**, so an account and either an on-campus connection or the FHNW VPN are both prerequisites. The procedure has three parts.

**Step 1 — Generate an ed25519 SSH key.** Calculon authenticates with SSH keys and accepts **only the ed25519 key type** (RSA keys are rejected). On your own machine:

```
ssh-keygen -t ed25519 -C "FHNW_EMAIL_ADDRESS"
cat ~/.ssh/id_ed25519.pub        # this PUBLIC key is what you submit
```

Keep the private key (`~/.ssh/id_ed25519`) secure and never share it; you submit only the `.pub` public key.

**Step 2 — Request access through the Calculon Teams channel.** Join the FHNW Calculon Teams channel (linked from the "Access to Calculon" page of the user guide). In the **Requests** channel, open the **public_ssh_keys** tab and add a new entry containing your ed25519 **public** key together with a short justification for access (for example, that it is for a Master's module project). The team then provisions the account; this can take several days, so request access well before you need to run.

**Step 3 — Receive your username.** Once approved you are given a Calculon username, which you use in the connection commands below. If you have never worked on a cluster, the guide's "Cluster101" course is a useful primer.

### 5.2 Connecting to Calculon

Connection requires being on the FHNW network. From outside, first start the FHNW VPN (profiles and instructions are on the FHNW intranet network/VPN page), then connect with SSH:

```
ssh -i ~/.ssh/id_ed25519 <username>@calculon.informatik.fhnw.ch
```

On the first connection you are asked to verify the host-key fingerprint; accept it only if it matches one published in the user guide. To avoid retyping the key path and host every time, add a stanza to your local `~/.ssh/config`:

```
Host calculon
    Hostname calculon.informatik.fhnw.ch
    Port 22
    User <username>
    IdentityFile ~/.ssh/id_ed25519
```

You can then connect simply with `ssh calculon`. On Windows, the guide also documents MobaXterm (configure an SSH session with the login node `calculon.informatik.fhnw.ch`, your username, and your private-key path); VS Code Remote-SSH can be configured similarly for editing on the cluster. If a connection fails, re-run the SSH command with `-vvv` for verbose diagnostics and include that text output when contacting the Calculon team.

> **Login node vs compute nodes.** The machine you land on after `ssh` is the **login node**. It is for editing files, managing the environment, and submitting jobs — never for running the actual workload. All computation must go through Slurm (`srun`/`sbatch`) so that it executes on the **compute nodes**. Running a heavy job directly on the login node is against the cluster's code of conduct and will slow the system for everyone.

### 5.3 Preparing the software environment on Calculon

The pipeline runs inside a conda environment named `aci_project`. The official guidance is to use conda for scientific-computing stacks (it provides optimized numerical binaries and handles complex dependencies), and — importantly for batch jobs — to invoke it with `conda run -n <env>` rather than `conda activate`, because `conda activate` only works in interactive shells. The provided Slurm scripts already follow this convention.

Create the environment once, on the login node:

```
# one-time environment creation
conda create -n aci_project python=3.12
conda activate aci_project          # interactive use on the login node is fine

# install and verify every dependency via the bootstrap module
python experiment_modules/MOD0_dependency_bootstrap.py
conda deactivate
```

If `conda activate` reports that the shell is not initialized, run `conda init bash` once and re-open the shell. You can confirm the environment exists with `conda env list`. For reproducibility you may export it with `conda env export > environment.yml`.

### 5.4 Transferring the project to the cluster

Copy the project tree to your Calculon home directory. The Slurm scripts assume the project lives at `~/Applied_Comp_Intel_Project`. From your local machine:

```
# using rsync (recommended; resumable and incremental)
rsync -av --exclude '.venv' --exclude '__pycache__' \
    ./Applied_Comp_Intel_Project/ calculon:~/Applied_Comp_Intel_Project/

# or a one-off copy with scp
scp -r ./Applied_Comp_Intel_Project <username>@calculon.informatik.fhnw.ch:~/
```

Then, on the cluster, create the directory the job scripts write their logs into (they will fail to start without it):

```
cd ~/Applied_Comp_Intel_Project
mkdir -p logs
```

### 5.5 Understanding the partitions used

A *partition* is a named queue of compute nodes with its own hardware and limits. This experiment uses two CPU partitions:

| Partition | Hardware | Used by | Why |
|---|---|---|---|
| `top6` | The six newest nodes (AMD Ryzen Threadripper PRO, Zen 3), high clock speed | MOD6 discovery array, MOD6 fixup, MOD7 smoke test and full sweep | CPU-bound genetic programming and network training benefit from the fastest cores |
| `performance` | General-purpose nodes | MOD6 consensus aggregation, MOD8 reporting | Lightweight, mostly file I/O; does not need the newest CPUs |

All partitions allow a maximum wall-clock time of seven days and one hour, with a short default (30 minutes) if `--time` is not specified, so every long job sets `--time` explicitly. Memory is allocated per CPU by default; the scripts request a total with `--mem`. Job priority is currently based on how long a job has waited plus backfill of small jobs, so there is no separate quality-of-service tier to select.

### 5.6 One-time data preparation (MOD1, MOD2)

Data preparation is light and can be run interactively. Either run it inside a short interactive Slurm session (preferred, to keep the login node clear) or, if it is very quick, directly. To request an interactive session:

```
srun --partition=performance --cpus-per-task=4 --mem=8G --time=1:00:00 --pty bash

# inside the interactive session, on a compute node:
cd ~/Applied_Comp_Intel_Project
conda activate aci_project
python experiment_modules/MOD1_pipeline_selector.py     # download + clean datasets
python experiment_modules/MOD2_pipeline_meta_extractor.py  # meta-features, Phase A/B split, norm params
conda deactivate
exit
```

Recall that MOD1 needs your OpenML API key supplied as described in Section 4.2, and that MOD2 must be run before any discovery so the Phase A normalization parameters exist.

### 5.7 Discovery sweep (MOD6) — per-seed array plus dependent aggregation

Discovery is split into two Slurm jobs. The first is a **job array** in which each task runs one seed for one (topology, activation) pair and writes only its per-seed file; the array deliberately excludes any pair whose consensus artifact already exists. The second job runs **after** the array completes and aggregates those per-seed files into the consensus artifacts, performing no genetic programming itself. The two are linked with a Slurm dependency so aggregation runs only if every array task succeeds.

```
cd ~/Applied_Comp_Intel_Project

# 1. launch the per-seed array and capture its job id
ARRAY_JOB=$(sbatch --parsable mod6_sweep.sbatch)
echo "Array job: $ARRAY_JOB"

# 2. launch the aggregation job to run only if the whole array succeeds
sbatch --dependency=afterok:${ARRAY_JOB} mod6_aggregate.sbatch

# 3. monitor
squeue -u $USER -t all
tail -f logs/mod6_${ARRAY_JOB}_0.log
```

Each array task requests the `top6` partition with four CPUs and 16 GB of memory under a generous wall-clock limit; the aggregation job is lightweight and runs on `performance`. Internally each task invokes the engine through `conda run -n aci_project python -u experiment_modules/MOD6_om_mogp_engine_final.py ...` with the single (activation, topology, seed) it was assigned, and the aggregation job re-runs the engine with `--aggregate_from_disk` over all cells, skipping any that already have a consensus file. Because both the array and the aggregation step skip work that is already on disk, the sweep is safe to re-submit after an interruption.

If a small number of pair-seed tasks fail to produce their per-seed file (for example, because of a transient scheduler issue), `mod6_targeted.sbatch` re-runs just an explicitly listed set of triples; edit its list to match the missing combinations, submit it the same way, and then re-run the aggregation job. A single combination can also be submitted directly with the helper script:

```
./mod6_single_job.sh <activation> <topology> <seed>
# example:
./mod6_single_job.sh squashing deep_narrow 42
```

### 5.8 Validation (MOD7) — smoke test then full sweep

Validate the pipeline on one topology before committing to the full matrix:

```
# fast check: one topology, quick-test mode (a few minutes)
sbatch mod7_smoke_test.sbatch

# full validation across all cells (several hours)
sbatch mod7_full_sweep.sbatch
```

Both jobs run on `top6`, verify that consensus rule files exist before launching, invoke the MOD7 driver (through `conda run -n aci_project`) against the production rule directory, and then confirm that the expected per-cell JSON artifacts were written. The smoke test restricts itself to the shallow topology under the quick-test flag, which collapses the work to one trial seed and a handful of datasets so the end-to-end path can be checked in minutes. The full sweep covers every cell that has a consensus artifact and skips any that do not.

### 5.9 Reporting, qualitative, and post-hoc analysis (MOD8, MOD9, MOD10)

After the full validation sweep completes, generate the statistical report and figures as a batch job:

```
sbatch mod8_run.sbatch
```

This job (on `performance`) checks that MOD7 JSON artifacts are present, warns if the Phase B dataset list or normalization parameters are missing (in which case the cluster-stratified analysis degrades), runs MOD8, and verifies that the pooled tables, distribution plots, cluster-stratified tables, cluster master CSV, diagnostics, and PCA figure were produced.

MOD9 and MOD10 are light enough to run in a short interactive session once the consensus rules and MOD7 results exist:

```
srun --partition=performance --cpus-per-task=2 --mem=8G --time=1:00:00 --pty bash
cd ~/Applied_Comp_Intel_Project
conda activate aci_project
python experiment_modules/MOD9_framework_qualitative_analyzer.py
python experiment_modules/MOD10_post_hoc_analyzer.py
conda deactivate
exit
```

### 5.10 Monitoring, retrieving results, and troubleshooting

Useful Slurm commands while jobs run:

| Command | Purpose |
|---|---|
| `squeue -u $USER` | List your queued and running jobs |
| `squeue -j <id> --start` | Estimated start time of a pending job |
| `sacct -u $USER` | History of finished jobs |
| `sacct -j <id> --format=JobID,State,ExitCode,MaxRSS,Elapsed` | Final state, exit code, peak memory, and runtime of a job |
| `scancel <id>` | Cancel a job (or `scancel -u $USER` to cancel all of yours) |
| `spartition` | Node and partition status overview |
| `tail -f logs/<logfile>` | Watch a job's live output |

Common issues and their checks: a job stuck in the **PD (pending)** state usually means the requested resources are busy — inspect the reason with `squeue -j <id> --start` and consider fewer resources or the `performance` partition; an **out-of-memory** failure shows as exit code 137 and is fixed by raising `--mem` after inspecting `MaxRSS`; a **time-limit** termination is fixed by raising `--time` (the discovery and validation engines also checkpoint, so an interrupted run can be resumed rather than restarted).

When the runs are complete, copy the results back to your machine. The artifacts live under `experiment_modules/experimental_results_analysis_visualizations/` (reports and visualizations) and `experiment_modules/generated_files/GA_rule_files/` (the consensus rules):

```
# from your local machine
rsync -av calculon:~/Applied_Comp_Intel_Project/experiment_modules/experimental_results_analysis_visualizations/ \
    ./results_from_calculon/
```

### 5.11 Recommended end-to-end order on the cluster

1. Obtain an account and connect (Sections 5.1–5.2).
2. Create the `aci_project` conda environment and run MOD0 (Section 5.3).
3. Transfer the project and create the `logs/` directory (Section 5.4).
4. Prepare data: MOD1, then MOD2 (Section 5.6).
5. Discover: submit `mod6_sweep.sbatch`, then `mod6_aggregate.sbatch` with an `afterok` dependency (Section 5.7); run `mod6_targeted.sbatch` only if some triples are missing.
6. Validate: `mod7_smoke_test.sbatch`, then `mod7_full_sweep.sbatch` (Section 5.8).
7. Report and analyze: `mod8_run.sbatch`, then MOD9 and MOD10 (Section 5.9).
8. Retrieve results to your local machine (Section 5.10).

## 6. Reproducibility Notes

Random number generation is seeded from a single value in every stage that involves randomness, covering the Python, NumPy, and PyTorch generators, and discovery is replicated across a fixed set of seeds. PyTorch runs in CPU mode with a configurable intra-op thread count. Normalization parameters are fit on Phase A only and persisted, so the same affine transform is applied at discovery and validation time. Each consensus artifact records the environment and seeds used to produce it. Because the stages communicate through files, any stage can be re-run independently against the artifacts already on disk.

---
