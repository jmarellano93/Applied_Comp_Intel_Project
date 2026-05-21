"""
Module 7 Driver: Pipeline Matrix Orchestrator

Dynamically scans the explicit /rules sub-namespace within the visualization
directory for the latest MOD6 artifacts. Extracts equations via Regex, and
pipelines them into the statistical evaluation matrix.
"""

import os
import re
import glob
import sys
import logging
import subprocess
from pathlib import Path
from typing import List
from pydantic import BaseModel, Field

logging.basicConfig(level=logging.INFO, format="%(asctime)s - ACI-DRIVER - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

class DriverMatrixConfig(BaseModel):
    topology: str = Field(default="shallow")
    target_script: str = Field(default="MOD7_framework_validation_matrix.py")
    module_directory: Path = Field(default_factory=lambda: Path(__file__).resolve().parent)

    # Architectural Partitioning: Anchors strictly to the /rules subdirectory
    rule_directory: Path = Field(
        default_factory=lambda: Path(__file__).resolve().parent / "generated_files" / "experimental_results_analysis_visualizations" / "rules"
    )

    activation_targets: List[str] = Field(
        default=["rectification", "squashing", "smooth", "aggregation", "trigonometric", "linear"]
    )

class PipelineDriver:
    def __init__(self, config: DriverMatrixConfig):
        self.cfg = config
        self.interpreter: str = sys.executable

        if not self.cfg.rule_directory.exists():
            logger.warning(f"Rule directory not found: {self.cfg.rule_directory}. Creating partitioned taxonomy now.")
            self.cfg.rule_directory.mkdir(parents=True, exist_ok=True)

    def extract_rules_from_artifact(self, activation: str) -> List[str]:
        """Scans the targeted rule_directory for the latest artifact and extracts equations."""
        pattern = self.cfg.rule_directory / f"Final_Discovered_Rules_{activation}_*.txt"
        files = sorted(glob.glob(str(pattern)), key=os.path.getmtime, reverse=True)

        if not files:
            logger.warning(f"No rule artifacts found for {activation} in {self.cfg.rule_directory.name}. Skipping.")
            return []

        latest_file = files[0]
        with open(latest_file, 'r', encoding="utf-8") as f:
            content = f.read()

        equations = re.findall(r"Equation:\s*(.+)", content)

        if not equations:
            logger.warning(f"File {latest_file} contained no parseable equations.")
            return []

        return equations[:5]

    def execute_matrix_sweep(self) -> None:
        script_target = str(self.cfg.module_directory / self.cfg.target_script)

        logger.info(f"Initializing Automated Regex Cross-Validation Pipeline from {self.cfg.rule_directory.parent.name}/rules...")

        for activation in self.cfg.activation_targets:
            rules = self.extract_rules_from_artifact(activation)
            if not rules:
                continue

            logger.info(f"Loaded {len(rules)} Pareto Rules for [{activation.upper()}]")

            cmd_tokens = [
                self.interpreter, script_target,
                "--topology", self.cfg.topology,
                "--activation", activation,
                "--rule_strs"
            ] + rules

            try:
                res = subprocess.run(cmd_tokens, check=True, text=True)
                if res.returncode == 0:
                    logger.info(f"Node [{activation.upper()}] Statistical Matrix complete.")
            except subprocess.CalledProcessError as sub_err:
                logger.error(f"Catastrophic Process Collapse on {activation.upper()}: {sub_err}")

        logger.info("--- ALL AUTOMATED EXPERIMENTAL SECTORS COMPLETE ---")

if __name__ == "__main__":
    driver = PipelineDriver(DriverMatrixConfig())
    driver.execute_matrix_sweep()