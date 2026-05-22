"""
Master Module 1: Master Data Orchestrator

Sequentially controls and executes the foundational data ingestion pipeline (MOD1 -> MOD2 -> MOD3).
Utilizes isolated subprocess execution to ensure that highly intensive data-scraping and
clustering matrices (Pandas/SciPy) are completely flushed from system RAM before PyTorch
allocates tensor locks in the final caching phase.
"""

import sys
import os
import subprocess
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s - ACI-ORCHESTRATOR - %(message)s")
logger = logging.getLogger(__name__)


def execute_subprocess(module_name: str) -> None:
    """Executes a target module as an isolated Python process to prevent memory fragmentation.

    Args:
        module_name (str): The filename of the target module to execute.

    Raises:
        SystemExit: If the target script is missing or encounters a runtime error.
    """
    logger.info(f"========== INITIATING: {module_name} ==========")

    # DYNAMIC CROSS-DIRECTORY ROUTING ENGINE
    # 1. Locate current directory: Project_Root/controller_modules
    controller_dir = Path(__file__).resolve().parent

    # 2. Step up to Project_Root
    project_root = controller_dir.parent

    # 3. Step down into the sibling experiment_modules directory
    script_path = project_root / "experiment_modules" / module_name

    if not script_path.exists():
        logger.critical(f"Execution Failed: Core module '{module_name}' missing from directory template: {script_path}")
        sys.exit(1)

    try:
        result = subprocess.run([sys.executable, str(script_path)], check=True, text=True)
        if result.returncode == 0:
            logger.info(f"========== SUCCESS: {module_name} completed cleanly. ==========\n")
    except subprocess.CalledProcessError as e:
        logger.error(f"========== CATASTROPHIC FAILURE IN {module_name} ==========")
        logger.error(f"Exit Code: {e.returncode}")
        sys.exit(1)


def run_master_pipeline() -> None:
    """Orchestrates the sequential cross-directory pipeline execution."""
    logger.info("Initializing Top-Level ACI Data Architecture Pipeline...")

    # Phase 1: Retrieve and format raw matrices from OpenML
    execute_subprocess("MOD1_pipeline_selector.py")

    # Phase 2: Compute topological meta-features and partition clustering
    execute_subprocess("MOD2_pipeline_meta_extractor.py")

    # Phase 3: Validate that the pipeline successfully feeds into the PyTorch Tensor Caching engine
    execute_subprocess("MOD3_pm_dataset_manager.py")

    logger.info("ALL FOUNDATIONAL DATA ARCHITECTURES SECURED AND VERIFIED.")


if __name__ == "__main__":
    run_master_pipeline()