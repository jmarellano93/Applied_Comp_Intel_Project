"""
Master Module 3: Validation & Analysis Orchestrator.

Sequentially executes the post-discovery analysis pipeline:

    1. path_structural_mapping_config.py        (pytest — path integrity)
    2. integration_verification.py              (pytest — Driver↔MOD7 CLI contract)
    3. MOD7_pipeline_driver.py                  (dispatches MOD7 per activation)
    4. MOD8_framework_statistical_reporter.py   (Wilcoxon tables + distribution plots)
    5. MOD9_framework_qualitative_analyzer.py   (sympy derivatives + topography surfaces)

Each step is isolated in its own subprocess so that Pandas / SciPy / Matplotlib
RAM is released between phases. Fails fast on any non-zero exit code from
steps 1-2 (structural verifications) — there is no point burning compute on
MOD7-9 if the directory contract is broken. Steps 3-5 also fail fast but log
which phase died so post-mortem is straightforward.

Mathematical Notes:
    Subprocess isolation is functionally equivalent to a sequential CPS chain
    over the pipeline's monad-like state, with the OS process boundary serving
    as the implicit garbage-collection sweep between phases.
"""

from __future__ import annotations

import argparse
import logging
import subprocess
import sys
from pathlib import Path
from typing import List, Optional

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - ACI-MSTR3 - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def _resolve_script(module_name: str) -> Path:
    """Resolves a target script's absolute path inside experiment_modules/.

    Args:
        module_name: The basename of the script (e.g. ``"MOD7_pipeline_driver.py"``).

    Returns:
        Absolute Path object pointing to the script.

    Raises:
        SystemExit: If the target script cannot be located.
    """
    controller_dir = Path(__file__).resolve().parent
    project_root = controller_dir.parent
    script_path = project_root / "experiment_modules" / module_name

    if not script_path.exists():
        logger.critical(
            f"Execution Failed: '{module_name}' missing at expected path: {script_path}"
        )
        sys.exit(1)
    return script_path


def _execute_subprocess(module_name: str, extra_args: Optional[List[str]] = None) -> None:
    """Runs a Python script as an isolated subprocess.

    Args:
        module_name: Basename of the target script.
        extra_args: Optional list of CLI arguments to forward.

    Raises:
        SystemExit: If the subprocess returns a non-zero exit code.
    """
    script_path = _resolve_script(module_name)
    extra_args = extra_args or []

    logger.info(f"========== INITIATING: {module_name} {' '.join(extra_args)} ==========")

    cmd = [sys.executable, str(script_path), *extra_args]
    try:
        result = subprocess.run(cmd, check=True, text=True)
        if result.returncode == 0:
            logger.info(f"========== SUCCESS: {module_name} completed cleanly. ==========\n")
    except subprocess.CalledProcessError as exc:
        logger.error(f"========== CATASTROPHIC FAILURE IN {module_name} ==========")
        logger.error(f"Exit Code: {exc.returncode}")
        sys.exit(1)


def _execute_pytest(test_module: str) -> None:
    """Runs a pytest test file as an isolated subprocess.

    Args:
        test_module: Basename of the pytest test file.

    Raises:
        SystemExit: On pytest failure (non-zero exit code). A failure here
            halts the pipeline before any MOD7-9 compute is consumed.
    """
    script_path = _resolve_script(test_module)
    logger.info(f"========== PYTEST: {test_module} ==========")

    cmd = [sys.executable, "-m", "pytest", str(script_path), "-v", "--no-header"]
    try:
        result = subprocess.run(cmd, check=True, text=True)
        if result.returncode == 0:
            logger.info(f"========== PYTEST PASSED: {test_module}. ==========\n")
    except subprocess.CalledProcessError as exc:
        logger.error(f"========== PYTEST FAILED: {test_module} ==========")
        logger.error(f"Exit Code: {exc.returncode}")
        logger.error("Halting pipeline — structural integrity violated.")
        sys.exit(1)


def run_validation_analysis_pipeline(quick_test: bool = False) -> None:
    """Executes the full validation + analysis chain.

    Args:
        quick_test: If True, propagates ``--quick_test`` to MOD7_pipeline_driver,
            which in turn forwards it to MOD7. Reduces trial count to 1 seed
            × 5 Phase B datasets per (rule, baseline) pair for pipeline
            validation runs.

    Returns:
        None. Exits with code 1 on any phase failure.
    """
    logger.info("Initializing Validation & Analysis Pipeline...")
    if quick_test:
        logger.info("MODE: QUICK_TEST — reduced trial count for pipeline validation.")

    # Phase 1: Structural path integrity (pytest)
    _execute_pytest("path_structural_mapping_config.py")

    # Phase 2: Driver↔MOD7 CLI contract verification (pytest)
    _execute_pytest("integration_verification.py")

    # Phase 3: GP rule validation matrix (subprocess fans out to MOD7 per activation)
    driver_args = ["--quick_test"] if quick_test else []
    _execute_subprocess("MOD7_pipeline_driver.py", driver_args)

    # Phase 4: Statistical aggregation + LaTeX + distribution plots
    _execute_subprocess("MOD8_framework_statistical_reporter.py")

    # Phase 5: Symbolic analysis + topography surfaces
    _execute_subprocess("MOD9_framework_qualitative_analyzer.py")

    logger.info("ALL VALIDATION & ANALYSIS ARTIFACTS GENERATED SUCCESSFULLY.")


def main() -> None:
    """CLI entry point. Parses ``--quick_test`` and dispatches the pipeline."""
    parser = argparse.ArgumentParser(
        description="MSTR3: Validation & Analysis Orchestrator"
    )
    parser.add_argument(
        "--quick_test",
        action="store_true",
        help="Reduce MOD7 trial count to 1 seed × 5 Phase B datasets per (rule, baseline)."
    )
    args = parser.parse_args()
    run_validation_analysis_pipeline(quick_test=args.quick_test)


if __name__ == "__main__":
    main()