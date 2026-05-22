"""
Module 7 Driver: Pipeline Matrix Orchestrator - **PROTOTYPE**.

A scaled-down derivative of MOD7_pipeline_driver.py used for pre-flight
pipeline validation. Differs from the production driver in only two
respects:

    1. Target script: invokes ``MOD7_framework_validation_matrix_prototype.py``
       (always quick: 1 seed x 5 Phase B datasets, ~10-20 min total) instead
       of the production framework.

    2. No ``--quick_test`` flag is exposed or forwarded — the prototype
       framework is unconditionally quick by design.

Default rule directory: ``generated_files/GA_rule_files_testing/`` (same as
production driver, since MOD5 prototype writes there). Override with
``--rule_directory PATH`` if testing against alternate fixture sets.

The (topology x activation) sweep iterates the same 18 cells as the
production driver. Output JSONs are written to
``MOD7_validation_matrix_prototype/`` rather than the production reports
directory, keeping prototype artifacts cleanly separated.
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

    topology: str = Field(
        default="shallow",
        description="Backward-compat scalar default. New code should use topology_targets.",
    )
    topology_targets: List[str] = Field(
        default=["shallow", "deep_narrow", "funnel"],
        description="Topologies to sweep. Each (topology, activation) pair gets one subprocess.",
    )
    target_script: str = Field(
        default="MOD7_framework_validation_matrix_prototype.py",
        description="The PROTOTYPE framework, always quick (1 seed x 5 datasets).",
    )
    module_directory: Path = Field(
        default_factory=lambda: Path(__file__).resolve().parent
    )
    rule_directory: Path = Field(default_factory=_default_rule_directory)
    activation_targets: List[str] = Field(
        default=["rectification", "squashing", "smooth", "aggregation", "trigonometric", "linear"]
    )
    # quick_test field intentionally removed: the prototype framework is
    # unconditionally quick. Use the production driver if a configurable
    # scale is needed.

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
        """Iterates (topology, activation) pairs, dispatches one PROTOTYPE
        framework subprocess per pair.

        Returns:
            None. Logs progress and failures per (topology, activation) cell.
        """
        script_target = str(self.cfg.module_directory / self.cfg.target_script)

        logger.info(f"--- MOD7 PROTOTYPE DRIVER ---")
        logger.info(f"Reading rules from: {self.cfg.rule_directory}")
        logger.info(f"Target framework:   {self.cfg.target_script}")
        logger.info(f"Topologies sweep:   {', '.join(self.cfg.topology_targets)}")
        logger.info(f"Activations sweep:  {', '.join(self.cfg.activation_targets)}")
        logger.info("Scale:              PROTOTYPE (1 seed x 5 Phase B datasets per cell)")

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
                # No --quick_test forwarding: prototype framework is always quick.
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

        logger.info("--- ALL (TOPOLOGY x ACTIVATION) PROTOTYPE SECTORS COMPLETE ---")


# =============================================================================
# 3. CLI
# =============================================================================

def main() -> None:
    """CLI entry point. Parses ``--rule_directory`` and ``--topologies``."""
    parser = argparse.ArgumentParser(
        description="MOD7 PROTOTYPE Driver: dispatches the prototype framework "
                    "(1 seed x 5 datasets) per (topology, activation) pair. "
                    "Use the production driver for paper-grade validation runs.",
    )
    parser.add_argument(
        "--rule_directory", type=str, default=None,
        help="Directory containing Final_Discovered_Rules_*.txt artifacts. "
             "Defaults to generated_files/GA_rule_files_testing.",
    )
    parser.add_argument(
        "--topologies", type=str, nargs="+", default=None,
        choices=["shallow", "deep_narrow", "funnel"],
        help="Subset of topologies to sweep. Default: all three.",
    )
    # No --quick_test flag: this driver always runs the prototype framework,
    # which is unconditionally quick.
    args = parser.parse_args()

    overrides = {}
    if args.topologies is not None:
        overrides["topology_targets"] = args.topologies
    if args.rule_directory is not None:
        overrides["rule_directory"] = Path(args.rule_directory).resolve()

    config = DriverMatrixConfig(**overrides)
    driver = PipelineDriver(config)
    driver.execute_matrix_sweep()


if __name__ == "__main__":
    main()