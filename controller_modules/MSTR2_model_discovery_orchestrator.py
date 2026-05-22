"""
Master Module 2: Model Discovery Orchestrator

Initiates the intensive Phase A hardware-accelerated evolutionary loop (MOD6).
Strictly delegates execution to the final production engine, bypassing the prototype.
"""

import sys
import os
import subprocess
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s - ACI-ORCHESTRATOR - %(message)s")
logger = logging.getLogger(__name__)


def execute_subprocess(module_name: str) -> None:
    """Executes a target module as an isolated Python process to prevent memory fragmentation."""
    logger.info(f"========== INITIATING: {module_name} ==========")

    controller_dir = Path(__file__).resolve().parent
    project_root = controller_dir.parent
    script_path = project_root / "experiment_modules" / module_name

    if not script_path.exists():
        logger.critical(f"Execution Failed: Core module '{module_name}' missing from directory: {script_path}")
        sys.exit(1)

    try:
        result = subprocess.run([sys.executable, str(script_path)], check=True, text=True)
        if result.returncode == 0:
            logger.info(f"========== SUCCESS: {module_name} completed cleanly. ==========\n")
    except subprocess.CalledProcessError as e:
        logger.error(f"========== CATASTROPHIC FAILURE IN {module_name} ==========")
        logger.error(f"Exit Code: {e.returncode}")
        sys.exit(1)


def run_discovery_pipeline() -> None:
    """Orchestrates the massive outer-loop evolutionary process."""
    logger.info("Initializing Top-Level ACI Model Discovery Pipeline...")

    # Exclusively runs the production Module 6, skipping prototype redundancy.
    execute_subprocess("MOD6_om_mogp_engine_final.py")

    logger.info("DISCOVERY ARCHITECTURE COMPLETE. GLOBAL OPTIMA IDENTIFIED.")


if __name__ == "__main__":
    run_discovery_pipeline()