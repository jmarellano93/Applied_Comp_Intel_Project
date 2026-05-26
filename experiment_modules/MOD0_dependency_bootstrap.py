"""
Module 0: Dependency Bootstrap & Environment Verification.

Idempotent installer for all Python dependencies required by the Applied
Computational Intelligence project. Reads from a pinned manifest derived
directly from Section VI Q27 of the methodology document and verifies each
package post-install.

Design constraints:
    * STDLIB ONLY at module top-level. MOD0 must run on a fresh Python
      3.12.6 interpreter with nothing else installed, including pydantic.
      All non-stdlib imports below this docstring would create a chicken-
      and-egg dependency, so they are forbidden.
    * Idempotent. Re-running on a fully-configured machine results in all
      packages reporting [OK] without performing any installs.
    * Per-package atomic pip calls. PyTorch's +cu121 wheel needs a custom
      --index-url that the other packages must NOT inherit, so bundling
      installs into a single pip call is unsafe.

CLI:
    python MOD0_dependency_bootstrap.py            # install missing/mismatched
    python MOD0_dependency_bootstrap.py --dry_run  # preview only
    python MOD0_dependency_bootstrap.py --upgrade  # force re-install all

Exit codes:
    0  All dependencies satisfied.
    1  Python version below minimum, or one or more packages failed to install.

Usage notes:
    * Recommended invocation: from the project root with the venv interpreter
      active. The script installs into ``sys.executable``, whichever Python
      that is, so activating the correct virtualenv beforehand is the user's
      responsibility.
    * The script does NOT create a virtualenv. PyCharm typically manages
      venvs project-side; doing it here would risk silently nesting venvs.
    * The script does NOT download any OpenML datasets. MOD1 handles that
      on first run; MOD0's responsibility ends at Python-package availability.
"""

from __future__ import annotations

import argparse
import importlib
import importlib.metadata
import platform
import subprocess
import sys
from dataclasses import dataclass, field
from typing import List, Optional, Tuple


# =============================================================================
# FUNCTIONAL BLOCK: Methodology-Pinned Environment Constants
# 4A) WHAT IT DOES: Captures the Python interpreter version and the PyTorch
#     wheel-index URL specified in Section VI Q27 of the methodology document.
# 4B) PARAMETERS: REQUIRED_PYTHON (3.12.6), MINIMUM_PYTHON (3.12.0),
#     PYTORCH_INDEX_URL (the official PyTorch +cu121 wheel index).
# 4C) METHODOLOGICAL JUSTIFICATION: Pinning the interpreter version below
#     3.12 risks DEAP/PyTorch C-extension ABI breaks; pinning above 3.12
#     risks not-yet-released scikit-learn wheels. The +cu121 PyTorch wheel
#     includes CUDA binaries that are dormant on CPU-only execution but are
#     required to match the exact PyTorch build identifier reported in the
#     experiment provenance log.
# =============================================================================

REQUIRED_PYTHON: Tuple[int, int, int] = (3, 12, 6)
MINIMUM_PYTHON: Tuple[int, int, int] = (3, 12, 0)
PYTORCH_INDEX_URL: str = "https://download.pytorch.org/whl/cu121"


# =============================================================================
# FUNCTIONAL BLOCK: Package Specification Dataclass
# 4A) WHAT IT DOES: Encapsulates the metadata needed to install, identify,
#     and verify a single Python package.
# 4B) PARAMETERS:
#     pip_name     - The distribution name passed to ``pip install``.
#     import_name  - The top-level import name (often, but not always, the
#                    same as pip_name; e.g. 'sklearn' vs 'scikit-learn').
#     version      - Optional exact version pin. None means "latest".
#     index_url    - Optional custom wheel index (PyTorch only).
# 4C) METHODOLOGICAL JUSTIFICATION: Decoupling pip_name from import_name is
#     essential because Python's distribution names and module names diverge
#     for several major scientific packages.
# =============================================================================

@dataclass(frozen=True)
class PackageSpec:
    pip_name: str
    import_name: str
    version: Optional[str] = None
    index_url: Optional[str] = None

    def pip_install_command(self, force: bool = False) -> List[str]:
        """Returns the ``pip install`` argument list for this package.

        Args:
            force: If True, append ``--force-reinstall`` and
                ``--no-deps`` to override any cached version.

        Returns:
            A list of CLI tokens suitable for ``subprocess.run``.
        """
        spec = self.pip_name + (f"=={self.version}" if self.version else "")
        cmd: List[str] = [sys.executable, "-m", "pip", "install", spec]
        if self.index_url:
            cmd.extend(["--index-url", self.index_url])
        if force:
            cmd.append("--force-reinstall")
        return cmd


# =============================================================================
# FUNCTIONAL BLOCK: Dependency Manifest
# 4A) WHAT IT DOES: Enumerates the full set of project dependencies, in the
#     order they should be installed.
# 4B) PARAMETERS: One PackageSpec per dependency. Version pins follow the
#     methodology document; ``None`` versions mean "latest stable".
# 4C) METHODOLOGICAL JUSTIFICATION: Pinned versions (PyTorch, DEAP,
#     scikit-learn, Pydantic-major) appear in the methodology's software
#     stack table and must match for exact result reproducibility.
#     Unpinned versions (numpy, scipy, sympy, etc.) are documented as
#     "latest" in the methodology and are expected to behave consistently
#     across minor-version drift.
# =============================================================================

MANIFEST: Tuple[PackageSpec, ...] = (
    # --- Core scientific stack ---
    PackageSpec(pip_name="numpy", import_name="numpy"),
    PackageSpec(pip_name="scipy", import_name="scipy"),
    PackageSpec(pip_name="pandas", import_name="pandas"),

    # --- ML stack (pinned per methodology) ---
    PackageSpec(pip_name="scikit-learn", import_name="sklearn", version="1.8.0"),
    PackageSpec(
        pip_name="torch", import_name="torch",
        version="2.5.1+cu121", index_url=PYTORCH_INDEX_URL,
    ),

    # --- Evolutionary computation (pinned per methodology) ---
    PackageSpec(pip_name="deap", import_name="deap", version="1.4"),

    # --- Symbolic math ---
    PackageSpec(pip_name="sympy", import_name="sympy"),

    # --- Data ingest ---
    PackageSpec(pip_name="openml", import_name="openml"),

    # --- Validation (Pydantic v2; minor versions vary but major must be 2) ---
    PackageSpec(pip_name="pydantic", import_name="pydantic"),

    # --- Visualisation ---
    PackageSpec(pip_name="matplotlib", import_name="matplotlib"),
    PackageSpec(pip_name="seaborn", import_name="seaborn"),

    # --- Progress reporting ---
    PackageSpec(pip_name="tqdm", import_name="tqdm"),

    # --- Unit test framework ---
    PackageSpec(pip_name="pytest", import_name="pytest"),
)


# =============================================================================
# FUNCTIONAL BLOCK: Resolution + Installation Helpers
# 4A) WHAT IT DOES: Provides three small helpers that (a) verify interpreter
#     version, (b) detect whether a given package is already installed at the
#     correct version, and (c) shell out to pip to install one package.
# 4B) PARAMETERS: PackageSpec instances; current sys.version_info.
# 4C) METHODOLOGICAL JUSTIFICATION: Per-package atomic pip calls (rather than
#     a bundled requirements.txt install) are mandatory because PyTorch's
#     +cu121 wheel index would contaminate non-PyTorch package resolution if
#     applied at the batch level.
# =============================================================================

def verify_python_version() -> bool:
    """Checks the interpreter against the methodology-pinned Python version.

    Returns:
        True if the interpreter is at or above MINIMUM_PYTHON; False
        otherwise. Mismatches between the running interpreter and
        REQUIRED_PYTHON trigger a non-fatal warning.
    """
    current = sys.version_info[:3]
    cur_str = ".".join(str(x) for x in current)
    req_str = ".".join(str(x) for x in REQUIRED_PYTHON)
    min_str = ".".join(str(x) for x in MINIMUM_PYTHON)

    if current < MINIMUM_PYTHON:
        print(
            f"  [FATAL]  Python {min_str}+ required. Current: {cur_str}.",
            file=sys.stderr,
        )
        return False

    if current != REQUIRED_PYTHON:
        print(
            f"  [WARN]   Methodology pins Python {req_str} for exact "
            f"reproducibility. Current: {cur_str}. Proceeding."
        )
    else:
        print(f"  [OK]     Python {cur_str} matches methodology pin.")
    return True


def check_installed(spec: PackageSpec) -> Optional[str]:
    """Detects whether ``spec`` is already importable in the current interpreter.

    Args:
        spec: The package specification to check.

    Returns:
        The installed version string if the package is importable, or
        ``None`` if it is not. Returns the literal string ``"unknown"`` if
        the package is importable but its distribution metadata cannot
        be resolved.
    """
    try:
        importlib.import_module(spec.import_name)
    except ImportError:
        return None
    try:
        return importlib.metadata.version(spec.pip_name)
    except importlib.metadata.PackageNotFoundError:
        return "unknown"


def install_one(spec: PackageSpec, force: bool, dry_run: bool) -> bool:
    """Runs ``pip install`` for a single PackageSpec.

    Args:
        spec: The package to install.
        force: If True, passes ``--force-reinstall`` to pip.
        dry_run: If True, prints the command but does not execute it.

    Returns:
        True on successful install (or on a dry-run preview); False if
        pip returns a non-zero exit code.
    """
    cmd = spec.pip_install_command(force=force)
    print(f"  [RUN]    {' '.join(cmd[2:])}")

    if dry_run:
        print(f"  [DRY]    (skipped)")
        return True

    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
        return True
    except subprocess.CalledProcessError as err:
        tail = (err.stderr or "").strip().splitlines()[-3:]
        for line in tail:
            print(f"  [ERR]    {line}", file=sys.stderr)
        return False


# =============================================================================
# FUNCTIONAL BLOCK: Manifest Sweep Orchestrator
# 4A) WHAT IT DOES: Iterates the manifest, detects status per package, calls
#     install_one() as needed, and accumulates a summary table.
# 4B) PARAMETERS: dry_run (preview-only), upgrade (force-reinstall every package).
# 4C) METHODOLOGICAL JUSTIFICATION: A single-pass linear sweep keeps the
#     log output strictly ordered by manifest position, which makes the
#     final summary aligned with the install order and easy to audit.
# =============================================================================

def _version_matches(installed: str, wanted: Optional[str]) -> bool:
    """Returns True if ``installed`` satisfies ``wanted``.

    Tolerates PyTorch's local-suffix style ("2.5.1+cu121") by stripping
    the ``+...`` portion before comparison if the wanted version itself
    contains no local suffix.
    """
    if wanted is None:
        return True  # unpinned -> any present version is acceptable
    inst_base = installed.split("+", 1)[0]
    want_base = wanted.split("+", 1)[0]
    if "+" in wanted:
        return installed == wanted
    return inst_base == want_base


def sweep_manifest(
    dry_run: bool, upgrade: bool,
) -> List[Tuple[str, str, str]]:
    """Iterates MANIFEST, installs missing/mismatched packages, returns summary.

    Args:
        dry_run: If True, no pip commands actually execute.
        upgrade: If True, every package is force-reinstalled regardless
            of current state.

    Returns:
        A list of ``(pip_name, status, version)`` triples in manifest order.
        ``status`` is one of {"OK", "INSTALLED", "UPGRADED", "FAILED"}.
    """
    rows: List[Tuple[str, str, str]] = []

    for spec in MANIFEST:
        installed = check_installed(spec)
        print()  # blank line before each package report

        if upgrade:
            print(f"--- {spec.pip_name} (forced re-install)")
            ok = install_one(spec, force=True, dry_run=dry_run)
            status = "UPGRADED" if ok else "FAILED"
            installed = check_installed(spec) or "?"
        elif installed is None:
            print(f"--- {spec.pip_name} (MISSING)")
            ok = install_one(spec, force=False, dry_run=dry_run)
            status = "INSTALLED" if ok else "FAILED"
            installed = check_installed(spec) or "?"
        elif not _version_matches(installed, spec.version):
            print(
                f"--- {spec.pip_name} "
                f"(have {installed}, want {spec.version})"
            )
            ok = install_one(spec, force=True, dry_run=dry_run)
            status = "UPGRADED" if ok else "FAILED"
            installed = check_installed(spec) or "?"
        else:
            print(f"--- {spec.pip_name}: {installed}")
            print(f"  [OK]     already satisfied")
            status = "OK"

        rows.append((spec.pip_name, status, installed or "?"))

    return rows


# =============================================================================
# FUNCTIONAL BLOCK: Summary Reporter
# 4A) WHAT IT DOES: Prints a fixed-width aligned summary table of the
#     manifest sweep results, then returns the count of failures.
# 4B) PARAMETERS: A list of (pip_name, status, version) rows.
# 4C) METHODOLOGICAL JUSTIFICATION: A single summary table at the end of
#     execution gives the user one place to verify the final state of
#     their environment, independent of the verbose per-package log above.
# =============================================================================

def print_summary(rows: List[Tuple[str, str, str]]) -> int:
    """Prints the summary table; returns the number of FAILED entries."""
    print()
    print("=" * 72)
    print("DEPENDENCY MANIFEST SUMMARY")
    print("=" * 72)
    print(f"{'Package':<22} {'Status':<12} {'Version':<35}")
    print("-" * 72)
    for name, status, ver in rows:
        print(f"{name:<22} {status:<12} {ver:<35}")
    print("=" * 72)

    return sum(1 for _, status, _ in rows if status == "FAILED")


# =============================================================================
# FUNCTIONAL BLOCK: CLI Entry Point
# 4A) WHAT IT DOES: Parses CLI args, prints the banner, orchestrates the
#     verify -> sweep -> summary pipeline, and sets a meaningful exit code.
# 4B) PARAMETERS: --dry_run, --upgrade.
# 4C) METHODOLOGICAL JUSTIFICATION: Exit code 0 on success / 1 on failure
#     allows CI pipelines and bash scripts to chain MOD0 with downstream
#     experiment launchers via the standard ``&&`` operator.
# =============================================================================

def main() -> None:
    """CLI entry point. See module docstring for usage details."""
    parser = argparse.ArgumentParser(
        description="MOD0: Bootstrap all project Python dependencies."
    )
    parser.add_argument(
        "--dry_run", action="store_true",
        help="Print what would be installed without invoking pip.",
    )
    parser.add_argument(
        "--upgrade", action="store_true",
        help="Force re-install every package in the manifest.",
    )
    args = parser.parse_args()

    print("=" * 72)
    print("MOD0: DEPENDENCY BOOTSTRAP")
    print("=" * 72)
    print(f"  Platform:    {platform.platform()}")
    print(f"  Interpreter: {sys.executable}")
    print(f"  Mode:        "
          f"{'DRY-RUN' if args.dry_run else 'INSTALL'}"
          f"{' / FORCE-UPGRADE' if args.upgrade else ''}")
    print("-" * 72)

    if not verify_python_version():
        sys.exit(1)

    # Best-effort pip upgrade. Failures here are non-fatal.
    print()
    print("--- pip self-upgrade (best-effort) ---")
    if not args.dry_run:
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "--upgrade", "pip"],
            check=False, capture_output=True, text=True,
        )
        print("  [OK]     pip upgrade attempted")
    else:
        print("  [DRY]    skipped in dry-run mode")

    # Sweep
    print()
    print(f"--- Installing/verifying {len(MANIFEST)} dependencies ---")
    rows = sweep_manifest(dry_run=args.dry_run, upgrade=args.upgrade)

    # Report
    n_failed = print_summary(rows)
    print()
    if n_failed:
        print(f"RESULT: {n_failed} package(s) FAILED. Resolve manually.")
        sys.exit(1)
    print("RESULT: All dependencies satisfied. Project is ready to run.")
    print("        Next step: ``python controller_modules/MSTR1_master_data_orchestrator.py``")


if __name__ == "__main__":
    main()