"""
Module 7 Driver: Pipeline Matrix Orchestrator.

Scans the configured rule directory for ``Final_Discovered_Rules_*.txt``
artifacts (most-recent timestamp per activation), extracts equations via
regex, and pipelines them into ``MOD7_framework_validation_matrix.py`` as
subprocess invocations.

Default rule directory: ``generated_files/GA_rule_files/`` (production
consensus artifacts). Override with ``--rule_directory PATH`` to point at
``GA_rule_files_testing/`` or any other location during development.

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
    """Returns the default rule artifact directory (production consensus).

    Note:
        Override with ``--rule_directory`` to point at a testing directory
        (e.g. ``generated_files/GA_rule_files_testing``) when developing
        without disturbing the production artifacts.
    """
    return Path(__file__).resolve().parent / "generated_files" / "GA_rule_files"


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

    topology: str = Field(
        default="shallow",
        description="Backward-compat scalar default. New code should use topology_targets.",
    )
    topology_targets: List[str] = Field(
        default=["shallow", "deep_narrow", "funnel"],
        description="Topologies to sweep. Each (topology, activation) pair gets one MOD7 subprocess.",
    )
    target_script: str = Field(default="MOD7_framework_validation_matrix.py")
    module_directory: Path = Field(
        default_factory=lambda: Path(__file__).resolve().parent
    )
    rule_directory: Path = Field(default_factory=_default_rule_directory)
    activation_targets: List[str] = Field(
        default=["rectification", "smooth", "aggregation", "squashing", "linear", "trigonometric"],
        description="Order: best-performing activations first, partial-coverage (trigonometric) last "
                    "so its expected 'no artifacts' warnings cluster at the end of the log.",
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

    def extract_rules_from_artifact(self, activation: str, topology: str = None) -> List[str]:
        """Returns up to 5 equations from the most recent artifact for one (activation, topology) pair.

        Args:
            activation: One of the 6 canonical activation tokens.
            topology: One of 'shallow', 'deep_narrow', 'funnel'. If None, falls
                back to ``self.cfg.topology`` (legacy single-topology behaviour).

        Returns:
            List of DEAP-format equation strings (top 5 by rank order in the
            most recently modified matching artifact). Empty list if no
            artifact is present for the (activation, topology) pair.
        """
        if topology is None:
            topology = self.cfg.topology

        pattern = (
            self.cfg.rule_directory
            / f"Final_Discovered_Rules_{activation}_{topology}_*.txt"
        )
        files = sorted(glob.glob(str(pattern)), key=os.path.getmtime, reverse=True)

        if not files:
            logger.warning(
                f"No rule artifacts found for ({activation}, {topology}) "
                f"in {self.cfg.rule_directory.name}. Skipping."
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
        """Iterates (topology, activation) pairs, dispatches one MOD7 subprocess per pair.

        Returns:
            None. Logs progress and failures per (topology, activation) cell.
        """
        script_target = str(self.cfg.module_directory / self.cfg.target_script)

        logger.info(f"Reading rules from: {self.cfg.rule_directory}")
        logger.info(f"Topologies sweep:   {', '.join(self.cfg.topology_targets)}")
        logger.info(f"Activations sweep:  {', '.join(self.cfg.activation_targets)}")
        if self.cfg.quick_test:
            logger.info("MODE: QUICK_TEST — propagating --quick_test to MOD7.")

        for topology in self.cfg.topology_targets:
            for activation in self.cfg.activation_targets:
                rules = self.extract_rules_from_artifact(activation, topology)
                if not rules:
                    continue

                logger.info(
                    f"Loaded {len(rules)} Pareto Rules for "
                    f"[{topology.upper()} / {activation.upper()}]"
                )

                cmd_tokens: List[str] = [
                    self.interpreter, script_target,
                    "--topology", topology,
                    "--activation", activation,
                ]
                if self.cfg.quick_test:
                    cmd_tokens.append("--quick_test")
                cmd_tokens.append("--rule_strs")
                cmd_tokens.extend(rules)

                try:
                    res = subprocess.run(cmd_tokens, check=True, text=True)
                    if res.returncode == 0:
                        logger.info(
                            f"Node [{topology.upper()} / {activation.upper()}] complete."
                        )
                except subprocess.CalledProcessError as sub_err:
                    logger.error(
                        f"Process collapse on ({topology}, {activation}) "
                        f"(exit code {sub_err.returncode})."
                    )

        logger.info("--- ALL (TOPOLOGY x ACTIVATION) SECTORS COMPLETE ---")


# =============================================================================
# 3. CLI
# =============================================================================

def main() -> None:
    """CLI entry point. Parses ``--rule_directory``, ``--quick_test``, ``--topologies``."""
    parser = argparse.ArgumentParser(
        description="MOD7 Driver: dispatches the framework validation matrix per "
                    "(topology, activation) pair.",
    )
    parser.add_argument(
        "--rule_directory", type=str, default=None,
        help="Directory containing Final_Discovered_Rules_*.txt artifacts. "
             "Defaults to generated_files/GA_rule_files (production).",
    )
    parser.add_argument(
        "--topologies", type=str, nargs="+", default=None,
        choices=["shallow", "deep_narrow", "funnel"],
        help="Subset of topologies to sweep. Default: all three.",
    )
    parser.add_argument(
        "--quick_test", action="store_true",
        help="Forwarded to MOD7: collapse to 1 seed × 5 Phase B datasets.",
    )
    args = parser.parse_args()

    overrides = {"quick_test": args.quick_test}
    if args.topologies is not None:
        overrides["topology_targets"] = args.topologies
    if args.rule_directory is not None:
        overrides["rule_directory"] = Path(args.rule_directory).resolve()

    config = DriverMatrixConfig(**overrides)
    driver = PipelineDriver(config)
    driver.execute_matrix_sweep()


if __name__ == "__main__":
    main()