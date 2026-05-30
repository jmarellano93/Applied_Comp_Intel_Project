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
    python MOD0_dependency_bootstrap.py             # install missing/mismatched
    python MOD0_dependency_bootstrap.py --dry_run   # preview only
    python MOD0_dependency_bootstrap.py --upgrade   # force re-install all
    python MOD0_dependency_bootstrap.py --configure # set FHNW e-mail + OpenML key

Credential configuration:
    The committed project ships with two placeholder tokens, one for the
    OpenML API key (in MOD1) and one for the FHNW e-mail address (in the
    Slurm batch scripts). Running with ``--configure`` -- or confirming the
    prompt shown after a normal run -- replaces every occurrence of those
    tokens, in place, across the project tree with the values you supply.
    The documentation files (README.md and PROJECT_DOCUMENTATION.md) are
    deliberately left untouched: they describe the placeholders to the
    reader, so they must keep the literal tokens. The rewrite preserves each
    file's exact bytes and line endings, and is idempotent: once the tokens
    are gone, re-running reports that there is nothing to do. The values can
    also be supplied non-interactively with ``--email`` and ``--openml_key``.

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
import getpass
import importlib
import importlib.metadata
import os
import platform
import re
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple


# =============================================================================
# FUNCTIONAL BLOCK: Methodology-Pinned Environment Constants
# WHAT IT DOES: Captures the Python interpreter version and the PyTorch
#  wheel-index URL specified in Section VI Q27 of the methodology document.
# PARAMETERS: REQUIRED_PYTHON (3.12.6), MINIMUM_PYTHON (3.12.0),
#  PYTORCH_INDEX_URL (the official PyTorch +cu121 wheel index).
# METHODOLOGICAL JUSTIFICATION: Pinning the interpreter version below
#  3.12 risks DEAP/PyTorch C-extension ABI breaks; pinning above 3.12
#  risks not-yet-released scikit-learn wheels. The +cu121 PyTorch wheel
#  includes CUDA binaries that are dormant on CPU-only execution but are
#  required to match the exact PyTorch build identifier reported in the
#  experiment provenance log.
# =============================================================================

REQUIRED_PYTHON: Tuple[int, int, int] = (3, 12, 6)
MINIMUM_PYTHON: Tuple[int, int, int] = (3, 12, 0)
PYTORCH_INDEX_URL: str = "https://download.pytorch.org/whl/cu121"


# =============================================================================
# FUNCTIONAL BLOCK: Package Specification Dataclass
# WHAT IT DOES: Encapsulates the metadata needed to install, identify,
#  and verify a single Python package.
# PARAMETERS:
#  pip_name     - The distribution name passed to ``pip install``.
#  import_name  - The top-level import name (often, but not always, the
#                same as pip_name; e.g. 'sklearn' vs 'scikit-learn').
#  version      - Optional exact version pin. None means "latest".
#  index_url    - Optional custom wheel index (PyTorch only).
# METHODOLOGICAL JUSTIFICATION: Decoupling pip_name from import_name is
#  essential because Python's distribution names and module names diverge
#  for several major scientific packages.
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
# WHAT IT DOES: Enumerates the full set of project dependencies, in the
#  order they should be installed.
# PARAMETERS: One PackageSpec per dependency. Version pins follow the
#  methodology document; ``None`` versions mean "latest stable".
# METHODOLOGICAL JUSTIFICATION: Pinned versions (PyTorch, DEAP,
#  scikit-learn, Pydantic-major) appear in the methodology's software
#  stack table and must match for exact result reproducibility.
#  Unpinned versions (numpy, scipy, sympy, etc.) are documented as
#  "latest" in the methodology and are expected to behave consistently
#  across minor-version drift.
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
# WHAT IT DOES: Provides three small helpers that (a) verify interpreter
#  version, (b) detect whether a given package is already installed at the
#  correct version, and (c) shell out to pip to install one package.
# PARAMETERS: PackageSpec instances; current sys.version_info.
# METHODOLOGICAL JUSTIFICATION: Per-package atomic pip calls (rather than
#  a bundled requirements.txt install) are mandatory because PyTorch's
#  +cu121 wheel index would contaminate non-PyTorch package resolution if
#  applied at the batch level.
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
# WHAT IT DOES: Iterates the manifest, detects status per package, calls
#  install_one() as needed, and accumulates a summary table.
# PARAMETERS: dry_run (preview-only), upgrade (force-reinstall every package).
# METHODOLOGICAL JUSTIFICATION: A single-pass linear sweep keeps the
#  log output strictly ordered by manifest position, which makes the
#  final summary aligned with the install order and easy to audit.
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
# WHAT IT DOES: Prints a fixed-width aligned summary table of the
#  manifest sweep results, then returns the count of failures.
# PARAMETERS: A list of (pip_name, status, version) rows.
# METHODOLOGICAL JUSTIFICATION: A single summary table at the end of
#  execution gives the user one place to verify the final state of
#  their environment, independent of the verbose per-package log above.
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
# FUNCTIONAL BLOCK: User-Credential Placeholder Configuration
# WHAT IT DOES: Replaces the two committed placeholder tokens -- one for the
#  OpenML API key and one for the FHNW e-mail address -- with the user's real
#  values, in place, across every text file in the project tree.
# PARAMETERS:
#  PLACEHOLDER_API_KEY / PLACEHOLDER_EMAIL - the exact tokens to find.
#  CONFIGURABLE_SUFFIXES - the text file extensions that are rewritten.
#  EXCLUDED_DIR_NAMES    - directory names never descended into.
#  EXCLUDED_FILE_NAMES   - file basenames never rewritten (the documentation
#                          files, which intentionally PRINT the placeholder
#                          tokens as user instructions and must keep them).
# METHODOLOGICAL JUSTIFICATION: The project is committed without secrets; the
#  placeholders let it be shared safely while pointing the reader at exactly
#  what must be supplied (see the documentation's "Placeholders" section).
#  Centralising the substitution in MOD0 lets a user configure the whole
#  project in one step instead of hand-editing the source and script files.
#  The documentation (README.md and PROJECT_DOCUMENTATION.md) is deliberately
#  excluded: those files describe the placeholders to the reader, so rewriting
#  the tokens there would corrupt the very instructions that explain them.
#  The two token constants are built by implicit string concatenation so this
#  file's own source never contains the contiguous token, which -- together
#  with the explicit self-skip in the file walk -- guarantees MOD0 can never
#  rewrite its own search strings. Reads/writes are byte-level so original
#  line endings (CRLF or LF) are preserved exactly.
# =============================================================================

PLACEHOLDER_API_KEY: str = "OPENML_API_KEY_" "HERE"
PLACEHOLDER_EMAIL: str = "FHNW_EMAIL_ADDRESS_" "HERE"

CONFIGURABLE_SUFFIXES: Tuple[str, ...] = (
    ".py", ".md", ".sbatch", ".sh", ".txt", ".cfg", ".ini",
    ".yml", ".yaml", ".toml", ".rst",
)
EXCLUDED_DIR_NAMES: frozenset = frozenset({
    ".git", "__pycache__", ".venv", "venv", ".idea", ".mypy_cache",
    ".pytest_cache", "node_modules",
})
# Documentation files that must never be rewritten: they intentionally contain
# the placeholder tokens as instructions to the reader. Matched by basename.
EXCLUDED_FILE_NAMES: frozenset = frozenset({
    "README.md", "PROJECT_DOCUMENTATION.md",
})


def _validate_email(email: str) -> Optional[str]:
    """Returns a trimmed e-mail if it looks structurally valid, else None."""
    candidate = email.strip()
    if re.fullmatch(r"[^@\s]+@[^@\s]+\.[^@\s]+", candidate):
        return candidate
    return None


def _looks_like_openml_key(key: str) -> bool:
    """True if ``key`` matches the canonical 32-character hex OpenML format."""
    return re.fullmatch(r"[0-9a-fA-F]{32}", key.strip()) is not None


def _mask_secret(secret: str) -> str:
    """Returns a display-safe masked rendering of a secret string."""
    s = secret.strip()
    if len(s) <= 8:
        return "*" * len(s)
    return f"{s[:4]}{'*' * (len(s) - 8)}{s[-4:]}"


def _relpath(path: Path, root: Path) -> str:
    """Best-effort path relative to ``root`` for tidy logging."""
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def _iter_project_text_files(root: Path, skip_path: Path) -> List[Path]:
    """Collects every configurable text file under ``root``.

    Args:
        root: Project root to walk.
        skip_path: A resolved file path to exclude (this script itself), so
            MOD0 never rewrites its own placeholder search tokens.

    Returns:
        A sorted list of candidate file paths whose suffix is in
        CONFIGURABLE_SUFFIXES, skipping EXCLUDED_DIR_NAMES directories and
        EXCLUDED_FILE_NAMES files (the documentation).
    """
    found: List[Path] = []
    for dirpath, dirnames, filenames in os.walk(root):
        # Prune excluded directories in place so os.walk does not descend.
        dirnames[:] = [d for d in dirnames if d not in EXCLUDED_DIR_NAMES]
        for name in filenames:
            if name in EXCLUDED_FILE_NAMES:
                continue  # never touch the documentation files
            p = Path(dirpath) / name
            if p.suffix.lower() not in CONFIGURABLE_SUFFIXES:
                continue
            try:
                if p.resolve() == skip_path:
                    continue
            except OSError:
                pass
            found.append(p)
    return sorted(found)


def _replace_tokens_in_file(
    path: Path, replacements: Dict[str, str], dry_run: bool,
) -> int:
    """Replaces tokens in one file, preserving its exact bytes and line endings.

    The file is read and written in binary so no newline translation occurs;
    only the placeholder tokens themselves change.

    Args:
        path: File to rewrite.
        replacements: Mapping of placeholder token -> replacement value.
        dry_run: If True, count occurrences but do not write.

    Returns:
        The number of token occurrences replaced (0 if none, or if the file
        is not valid UTF-8 / is unreadable, in which case it is skipped).
    """
    try:
        text = path.read_bytes().decode("utf-8")
    except (OSError, UnicodeDecodeError):
        return 0  # unreadable or non-UTF-8 (binary) -> skip silently

    count = sum(text.count(tok) for tok in replacements)
    if count == 0:
        return 0

    new_text = text
    for tok, value in replacements.items():
        new_text = new_text.replace(tok, value)

    if not dry_run:
        path.write_bytes(new_text.encode("utf-8"))
    return count


def configure_user_placeholders(
    email: str, api_key: str, project_root: Path, dry_run: bool,
) -> int:
    """Replaces both placeholder tokens across the whole project tree.

    Args:
        email: The FHNW e-mail address to substitute for the e-mail token.
        api_key: The OpenML API key to substitute for the key token.
        project_root: Root directory to walk.
        dry_run: If True, report what would change without writing.

    Returns:
        0 on success (including "nothing to do"); 1 if no text files could be
        located under ``project_root``.
    """
    replacements: Dict[str, str] = {
        PLACEHOLDER_API_KEY: api_key,
        PLACEHOLDER_EMAIL: email,
    }

    print()
    print("=" * 72)
    print("CREDENTIAL CONFIGURATION" + (" (DRY-RUN)" if dry_run else ""))
    print("=" * 72)
    print(f"  Project root: {project_root}")
    print(f"  FHNW e-mail:  {email}")
    print(f"  OpenML key:   {_mask_secret(api_key)}")
    print("-" * 72)

    skip_path = Path(__file__).resolve()
    files = _iter_project_text_files(project_root, skip_path)
    if not files:
        print(f"  [ERR]    No text files found under {project_root}.",
              file=sys.stderr)
        return 1

    changed: List[Tuple[str, int]] = []
    total = 0
    for p in files:
        n = _replace_tokens_in_file(p, replacements, dry_run)
        if n:
            rel = _relpath(p, project_root)
            verb = "would replace" if dry_run else "replaced"
            tag = "DRY" if dry_run else "SET"
            print(f"  [{tag}]    {verb} {n} in {rel}")
            changed.append((rel, n))
            total += n

    print("-" * 72)
    if total == 0:
        print("  [OK]     No placeholder tokens found -- nothing to do "
              "(already configured?).")
    else:
        occ = "occurrence" + ("" if total == 1 else "s")
        fls = "file" + ("" if len(changed) == 1 else "s")
        action = "Would update" if dry_run else "Updated"
        print(f"  [OK]     {action} {total} {occ} across {len(changed)} {fls}.")
    print("=" * 72)
    return 0


def _prompt_for_credentials() -> Optional[Tuple[str, str]]:
    """Interactively prompts for the FHNW e-mail and OpenML API key.

    Returns:
        An ``(email, api_key)`` tuple, or ``None`` if no controlling terminal
        is available or the user supplies invalid input three times.
    """
    if not sys.stdin.isatty():
        print("  [WARN]   Not an interactive terminal; cannot prompt. "
              "Pass --email and --openml_key instead.", file=sys.stderr)
        return None

    print()
    print("Enter the values to write into the project (Ctrl-C to abort).")

    # E-mail (visible input is fine; it is not a secret).
    email: Optional[str] = None
    for _ in range(3):
        email = _validate_email(input("  FHNW e-mail address: "))
        if email is None:
            print("  [WARN]   That does not look like a valid e-mail address.")
            continue
        if "fhnw" not in email.lower():
            print("  [WARN]   E-mail does not contain 'fhnw'; using it anyway.")
        break
    if email is None:
        print("  [ERR]    No valid e-mail entered.", file=sys.stderr)
        return None

    # OpenML API key (treated as a secret -> hidden entry, typed twice).
    api_key: Optional[str] = None
    for _ in range(3):
        first = getpass.getpass("  OpenML API key (hidden): ").strip()
        if not first:
            print("  [WARN]   Empty key.")
            continue
        if getpass.getpass("  Re-enter OpenML API key: ").strip() != first:
            print("  [WARN]   Keys did not match.")
            continue
        if not _looks_like_openml_key(first):
            print("  [WARN]   Key is not the usual 32-character hex format; "
                  "using it anyway.")
        api_key = first
        break
    if api_key is None:
        print("  [ERR]    No API key entered.", file=sys.stderr)
        return None

    return email, api_key


def _resolve_project_root(override: Optional[str]) -> Path:
    """Determines the project root to operate on.

    Args:
        override: An explicit --project_root value, or None to autodetect.

    Returns:
        The override if given; otherwise the parent of the directory holding
        this script (i.e. ``experiment_modules/`` -> the project root).
    """
    if override:
        return Path(override).expanduser().resolve()
    return Path(__file__).resolve().parent.parent


def run_configuration(
    email_arg: Optional[str], key_arg: Optional[str],
    project_root: Path, dry_run: bool,
) -> int:
    """Resolves credentials (from flags or an interactive prompt) and rewrites.

    Args:
        email_arg: Value of --email, or None to prompt.
        key_arg: Value of --openml_key, or None to prompt.
        project_root: Root directory to rewrite.
        dry_run: If True, preview without writing.

    Returns:
        A process exit code: 0 on success, 1 on failure or user abort.
    """
    if email_arg or key_arg:
        # Non-interactive path: both values must be supplied together.
        email = _validate_email(email_arg or "")
        api_key = (key_arg or "").strip()
        if email is None or not api_key:
            print("  [ERR]    Non-interactive configuration requires BOTH a "
                  "valid --email and a non-empty --openml_key.",
                  file=sys.stderr)
            return 1
        if not _looks_like_openml_key(api_key):
            print("  [WARN]   --openml_key is not the usual 32-character hex "
                  "format; using it anyway.")
    else:
        creds = _prompt_for_credentials()
        if creds is None:
            return 1
        email, api_key = creds

    return configure_user_placeholders(email, api_key, project_root, dry_run)


def maybe_offer_configuration(
    project_root: Path, args: argparse.Namespace,
) -> None:
    """After a normal run, detects remaining placeholders and guides the user.

    If placeholder tokens are still present anywhere in the project and the
    session is interactive, offers to set the credentials now; in a
    non-interactive session it prints a one-line reminder instead. The whole
    step is suppressed by ``--no_configure``.

    Args:
        project_root: Root directory to scan.
        args: Parsed CLI namespace (for --no_configure and --dry_run).
    """
    if args.no_configure:
        return

    skip_path = Path(__file__).resolve()
    tokens = (PLACEHOLDER_API_KEY, PLACEHOLDER_EMAIL)
    remaining = 0
    for p in _iter_project_text_files(project_root, skip_path):
        try:
            text = p.read_bytes().decode("utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        remaining += sum(text.count(t) for t in tokens)
    if remaining == 0:
        return

    occ = "occurrence" + ("" if remaining == 1 else "s")
    print()
    print(f"--- Credential placeholders detected ({remaining} {occ}) ---")
    if not sys.stdin.isatty():
        print("  [NOTE]   Set them with:  python MOD0_dependency_bootstrap.py "
              "--configure   (or pass --email and --openml_key).")
        return
    try:
        answer = input(
            "  Configure FHNW e-mail and OpenML key now? [y/N]: "
        ).strip().lower()
    except EOFError:
        return
    if answer in ("y", "yes"):
        run_configuration(None, None, project_root, args.dry_run)
    else:
        print("  [NOTE]   Skipped. Re-run with --configure when ready.")


# =============================================================================
# FUNCTIONAL BLOCK: CLI Entry Point
# WHAT IT DOES: Parses CLI args, prints the banner, orchestrates the
#  verify -> sweep -> summary pipeline, and sets a meaningful exit code.
# PARAMETERS: --dry_run, --upgrade.
# METHODOLOGICAL JUSTIFICATION: Exit code 0 on success / 1 on failure
#  allows CI pipelines and bash scripts to chain MOD0 with downstream
#  experiment launchers via the standard ``&&`` operator.
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
    parser.add_argument(
        "--configure", action="store_true",
        help="Set the FHNW e-mail and OpenML API key across the project, "
             "then exit (does not run the dependency sweep).",
    )
    parser.add_argument(
        "--email", default=None,
        help="FHNW e-mail for non-interactive configuration "
             "(use with --configure and --openml_key).",
    )
    parser.add_argument(
        "--openml_key", default=None,
        help="OpenML API key for non-interactive configuration "
             "(use with --configure and --email).",
    )
    parser.add_argument(
        "--project_root", default=None,
        help="Override the project root that configuration rewrites "
             "(defaults to the parent of experiment_modules/).",
    )
    parser.add_argument(
        "--no_configure", action="store_true",
        help="After a normal run, do NOT offer to set credentials even if "
             "placeholder tokens are still present.",
    )
    args = parser.parse_args()

    project_root = _resolve_project_root(args.project_root)

    # Dedicated configuration mode: set credentials and exit without a sweep.
    if args.configure or args.email or args.openml_key:
        sys.exit(run_configuration(
            args.email, args.openml_key, project_root, args.dry_run,
        ))

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

    # Credential placeholder check (after deps, so the project is otherwise ready).
    maybe_offer_configuration(project_root, args)

    print()
    if n_failed:
        print(f"RESULT: {n_failed} package(s) FAILED. Resolve manually.")
        sys.exit(1)
    print("RESULT: All dependencies satisfied. Project is ready to run.")

if __name__ == "__main__":
    main()

