"""
Module 7 Driver: Pipeline Matrix Orchestrator.

Scans the configured rule directory for ``Final_Discovered_Rules_*.txt``
artifacts (most-recent timestamp per activation), extracts equations via
regex, and pipelines them into ``MOD7_framework_validation_matrix.py`` as
subprocess invocations.

Default rule directory: ``generated_files/GA_rule_files_testing/``. Override
with ``--rule_directory PATH`` when running against real production rules
(``GA_rule_files/``) or any other location.

Forwards ``--quick_test`` to MOD7 so MSTR3 can request a fast pipeline check.
"""

from __future__ import annotations

import argparse
import glob
import logging
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import List, Optional

from pydantic import BaseModel, Field

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - ACI-DRIVER - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# =============================================================================
# 1. CONFIGURATION
# =============================================================================

def _default_rule_directory() -> Path:
    """Returns the default rule artifact directory (testing mode).

    Note:
        To switch to production rules, change the path returned here to
        ``generated_files/GA_rule_files`` (no trailing 'testing'), or pass
        ``--rule_directory`` at the CLI.
    """
    return Path(__file__).resolve().parent / "generated_files" / "GA_rule_files_testing"


class DriverMatrixConfig(BaseModel):
    """Runtime config for the MOD7 driver.

    Attributes:
        topology: FNN topology to pass through to MOD7.
        target_script: Filename of the MOD7 validation matrix script.
        module_directory: Directory containing ``target_script``.
        rule_directory: Where to scan for rule artifacts.
        activation_targets: Activation tokens to iterate.
        quick_test: If True, forwards ``--quick_test`` to MOD7.
    """

    topology: str = Field(default="shallow")
    target_script: str = Field(default="MOD7_framework_validation_matrix.py")
    module_directory: Path = Field(
        default_factory=lambda: Path(__file__).resolve().parent
    )
    rule_directory: Path = Field(default_factory=_default_rule_directory)
    activation_targets: List[str] = Field(
        default=["rectification", "squashing", "smooth", "aggregation", "trigonometric", "linear"]
    )
    quick_test: bool = Field(default=False)

    class Config:
        arbitrary_types_allowed = True


# =============================================================================
# 2. DRIVER
# =============================================================================

class PipelineDriver:
    """Orchestrates the per-activation matrix sweep.

    Attributes:
        cfg: A validated ``DriverMatrixConfig``.
        interpreter: Absolute path of the active Python interpreter.
    """

    def __init__(self, config: DriverMatrixConfig) -> None:
        self.cfg = config
        self.interpreter: str = sys.executable

        if not self.cfg.rule_directory.exists():
            logger.warning(
                f"Rule directory not found: {self.cfg.rule_directory}. Creating now."
            )
            self.cfg.rule_directory.mkdir(parents=True, exist_ok=True)

    # -------------------------------------------------------------------------
    # Rule extraction
    # -------------------------------------------------------------------------

    def extract_rules_from_artifact(self, activation: str) -> List[str]:
        """Returns up to 5 equations from the most recent artifact for ``activation``.

        Args:
            activation: One of the 6 canonical activation tokens.

        Returns:
            List of DEAP-format equation strings (top 5 by rank order in file).
            Empty list if no matching artifact is present.
        """
        pattern = self.cfg.rule_directory / f"Final_Discovered_Rules_{activation}_*.txt"
        files = sorted(glob.glob(str(pattern)), key=os.path.getmtime, reverse=True)

        if not files:
            logger.warning(
                f"No rule artifacts found for {activation} in {self.cfg.rule_directory.name}. Skipping."
            )
            return []

        latest_file = files[0]
        with open(latest_file, "r", encoding="utf-8") as fh:
            content = fh.read()

        equations = re.findall(r"Equation:\s*(.+)", content)
        if not equations:
            logger.warning(f"File {latest_file} contained no parseable equations.")
            return []
        return equations[:5]

    # -------------------------------------------------------------------------
    # Subprocess sweep
    # -------------------------------------------------------------------------

    def execute_matrix_sweep(self) -> None:
        """Iterates activations, dispatches MOD7 subprocesses with rule sets.

        Returns:
            None. Logs progress and failures per activation.
        """
        script_target = str(self.cfg.module_directory / self.cfg.target_script)

        logger.info(
            f"Reading rules from: {self.cfg.rule_directory}"
        )
        if self.cfg.quick_test:
            logger.info("MODE: QUICK_TEST — propagating --quick_test to MOD7.")

        for activation in self.cfg.activation_targets:
            rules = self.extract_rules_from_artifact(activation)
            if not rules:
                continue

            logger.info(f"Loaded {len(rules)} Pareto Rules for [{activation.upper()}]")

            cmd_tokens: List[str] = [
                self.interpreter, script_target,
                "--topology", self.cfg.topology,
                "--activation", activation,
            ]
            if self.cfg.quick_test:
                cmd_tokens.append("--quick_test")
            cmd_tokens.append("--rule_strs")
            cmd_tokens.extend(rules)

            try:
                res = subprocess.run(cmd_tokens, check=True, text=True)
                if res.returncode == 0:
                    logger.info(f"Node [{activation.upper()}] Statistical Matrix complete.")
            except subprocess.CalledProcessError as sub_err:
                logger.error(
                    f"Process collapse on {activation.upper()} "
                    f"(exit code {sub_err.returncode})."
                )

        logger.info("--- ALL ACTIVATION SECTORS COMPLETE ---")


# =============================================================================
# 3. CLI
# =============================================================================

def main() -> None:
    """CLI entry point. Parses ``--rule_directory``, ``--quick_test``, ``--topology``."""
    parser = argparse.ArgumentParser(
        description="MOD7 Driver: dispatches the framework validation matrix per activation."
    )
    parser.add_argument(
        "--rule_directory", type=str, default=None,
        help="Directory containing Final_Discovered_Rules_*.txt artifacts. "
             "Defaults to generated_files/GA_rule_files_testing.",
    )
    parser.add_argument(
        "--topology", type=str, default="shallow",
        choices=["shallow", "deep_narrow", "funnel"],
        help="FNN topology under test.",
    )
    parser.add_argument(
        "--quick_test", action="store_true",
        help="Forwarded to MOD7: collapse to 1 seed × 5 Phase B datasets.",
    )
    args = parser.parse_args()

    overrides = {"topology": args.topology, "quick_test": args.quick_test}
    if args.rule_directory is not None:
        overrides["rule_directory"] = Path(args.rule_directory).resolve()

    config = DriverMatrixConfig(**overrides)
    driver = PipelineDriver(config)
    driver.execute_matrix_sweep()


if __name__ == "__main__":
    main()