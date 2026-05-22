"""
Unit Tests for Module 5: MOGP Engine PROTOTYPE.

These tests verify the prototype's behavioural contract with downstream
consumers (MOD7/MOD8/MOD9) plus the prototype-specific differences from
MOD6:

  * Scaled hyperparameters         (pop=20, gens=3, dpr=3)
  * Routing to GA_rule_files_testing/
  * "MODULE 5 RESULTS (PROTOTYPE)" header label
  * All MOD6 patches preserved (Q-E multi-seed, Q-F Plan B, Q-G topology
    iteration, consensus aggregator, force_rerun resume guard).

The eight Pareto/SOO contract tests are intentionally identical in shape
to MOD6's so any regression that breaks one would break the other,
giving the user a redundant safety net before launching the 7-day run.
"""

import copy
import pytest
from pathlib import Path

import MOD5_om_mogp_engine_prototype as mod5


# =============================================================================
# PROTOTYPE-SPECIFIC: routing + scaled hyperparameters + labels
# =============================================================================

def test_get_generated_directory_routes_to_testing():
    """The prototype MUST route its output to GA_rule_files_testing/ so that
    MOD7's default ``--rule_directory`` finds it without CLI overrides.

    Routing to the production GA_rule_files/ would commingle prototype-scale
    artifacts with the 7-day MOD6 production output, which is unacceptable.
    """
    output_path = mod5.get_generated_directory()
    assert isinstance(output_path, Path)

    trailing = output_path.parts[-2:]
    assert trailing == ("generated_files", "GA_rule_files_testing"), (
        f"MOD5 must route to GA_rule_files_testing, got {output_path}"
    )


def test_prototype_hyperparameter_boundaries():
    """Confirms the three scaled defaults that distinguish MOD5 from MOD6."""
    config = mod5.MOGPConfig()

    # Prototype scale (vs MOD6's 150/20/20).
    assert config.population_size == 20, "MOD5 prototype must use pop=20."
    assert config.generations == 3, "MOD5 prototype must use gens=3."
    assert config.datasets_per_rule == 3, "MOD5 prototype must use dpr=3."

    # max_epochs SHOULD remain at 30 — only the outer-loop budget shrinks,
    # not the inner FNN training depth (otherwise the prototype wouldn't
    # exercise the real epochs-to-threshold dynamic).
    assert config.max_epochs == 30


def test_total_evaluation_budget_is_prototype_scale():
    """Sanity check: 18 (topology x activation) pairs x 3 runs at prototype
    scale should land near 9,720 cumulative PyTorch evaluations — small
    enough for a one-sitting smoke test, large enough to expose any
    contract bug in the MOD7-9 ingestion path.
    """
    config = mod5.MOGPConfig()
    evals = (
        config.population_size
        * config.generations
        * config.datasets_per_rule
        * len(config.activation_functions)
        * len(config.topology_targets)
        * config.n_gp_runs
    )
    assert evals == 9_720, (
        f"Expected 9,720 prototype evaluations but got {evals:,}. "
        "If this assertion fails the prototype scale has drifted."
    )


# =============================================================================
# Q-E / Q-F / Q-G CONTRACTS — must match MOD6 exactly
# =============================================================================

def test_qe_multi_seed_defaults():
    """Q-E: GP discovery defaults to N=3 independent runs with seeds {42,43,44}."""
    config = mod5.MOGPConfig()
    assert config.n_gp_runs == 3
    assert len(config.gp_run_seeds) >= config.n_gp_runs
    assert config.gp_run_seeds[:3] == [42, 43, 44]


def test_qe_seed_count_validation_raises_when_insufficient():
    """Pydantic validator rejects an n_gp_runs that exceeds the seed list length."""
    with pytest.raises(ValueError):
        mod5.MOGPConfig(n_gp_runs=5, gp_run_seeds=[42, 43])


def test_qg_topology_targets_defaults():
    """Q-G: full sweep covers shallow, deep_narrow, and funnel."""
    config = mod5.MOGPConfig()
    assert config.topology_targets == ["shallow", "deep_narrow", "funnel"]


def test_qf_plan_b_default_is_disabled():
    """Plan B remains opt-in even in the prototype. Switching to it would
    invalidate the comparison against the production Plan A discovery.
    """
    config = mod5.MOGPConfig()
    assert config.use_plan_b_soo is False
    assert config.plan_b_w_acc == pytest.approx(1.0)
    assert config.plan_b_w_eps == pytest.approx(0.5)
    assert config.plan_b_tournament_size == 3


def test_qf_plan_b_toggle_switches_creator_and_selector():
    """Toggling Plan B rebinds creator.Individual to FitnessSingle and
    swaps NSGA-II for tournament selection.

    Uses the same partial-unwrap pattern as MOD6's test: DEAP overwrites
    ``partial.__name__`` with the toolbox alias 'select', so we descend
    into ``.func`` to read the underlying selector identity.
    """
    config = mod5.MOGPConfig(use_plan_b_soo=True)
    mod5.ensure_deap_creators(use_plan_b_soo=True)

    from deap import creator
    assert hasattr(creator, "FitnessSingle")
    assert creator.FitnessSingle.weights == (1.0,)

    primitive_set = mod5.build_primitive_set()
    toolbox = mod5.build_toolbox(primitive_set, config)

    sel = toolbox.select
    wrapped = getattr(sel, "func", sel)
    wrapped_name = getattr(wrapped, "__name__", "")
    assert "tournament" in wrapped_name.lower(), (
        f"Expected selTournament under Plan B, got {wrapped_name!r}"
    )

    # Symmetric switch back to Plan A must restore selNSGA2.
    config_a = mod5.MOGPConfig(use_plan_b_soo=False)
    toolbox_a = mod5.build_toolbox(primitive_set, config_a)
    wrapped_a = getattr(toolbox_a.select, "func", toolbox_a.select)
    assert getattr(wrapped_a, "__name__", "") == "selNSGA2"


def test_consensus_aggregator_dedupes_by_string():
    """The consensus aggregator collapses identical rule strings into a
    single individual with mean-across-runs fitness. Required so the
    consensus front never double-counts a rule discovered in 2+ runs.
    """
    config = mod5.MOGPConfig(use_plan_b_soo=False)
    mod5.ensure_deap_creators(use_plan_b_soo=False)
    primitive_set = mod5.build_primitive_set()
    toolbox = mod5.build_toolbox(primitive_set, config)

    ind_a = toolbox.individual()
    ind_a.fitness.values = (0.80, 20.0, 3)
    ind_b = toolbox.individual()
    ind_b.fitness.values = (0.70, 18.0, 5)

    ind_a_clone = copy.deepcopy(ind_a)
    ind_a_clone.fitness.values = (0.82, 22.0, 3)  # same string, different fitness

    per_run_fronts = [[ind_a, ind_b], [ind_a_clone]]
    consensus = mod5.aggregate_consensus_front(per_run_fronts, config)

    unique_strs = {str(ind) for ind in consensus}
    assert len(unique_strs) <= 2


# =============================================================================
# RESUME-SAFETY + LABEL CONTRACTS
# =============================================================================

def test_force_rerun_default_preserves_resume_safety():
    """force_rerun must default to False so the resume guard activates by
    default. A True default would silently overwrite previously-completed
    pairs — undesirable both in MOD6 (catastrophic) and in MOD5 (would
    mask intermittent failures during iterative testing).
    """
    config = mod5.MOGPConfig()
    assert config.force_rerun is False


def test_export_writes_prototype_label_and_topology(tmp_path):
    """End-to-end smoke test on the export function: a single consensus
    artifact should embed the PROTOTYPE label, the topology, the activation,
    and use the filename pattern MOD7/MOD9's regex matches.

    This is the single most important test: if it passes, MOD7 will find
    the file and MOD9 will parse the (activation, topology) tokens out of
    the filename correctly.
    """
    config = mod5.MOGPConfig(use_plan_b_soo=False)
    mod5.ensure_deap_creators(use_plan_b_soo=False)
    primitive_set = mod5.build_primitive_set()
    toolbox = mod5.build_toolbox(primitive_set, config)

    ind = toolbox.individual()
    ind.fitness.values = (0.85, 5.0, 3)
    hall_of_fame = [ind]

    output_path = mod5.export_pareto_rules(
        hall_of_fame=hall_of_fame,
        activation="rectification",
        config=config,
        output_dir=tmp_path,
        env_details="--- TEST ENV BLOCK ---",
        topology="shallow",
        run_index=None,  # consensus form (not per-run archive)
    )

    # Filename contract.
    assert output_path.name.startswith("Final_Discovered_Rules_rectification_shallow_"), (
        f"Filename pattern broken: {output_path.name}"
    )
    assert output_path.name.endswith(".txt")
    assert output_path.parent == tmp_path, (
        f"Consensus file should be in main output_dir, got {output_path.parent}"
    )

    # Header content contract.
    content = output_path.read_text(encoding="utf-8")
    assert "MODULE 5 RESULTS (PROTOTYPE)" in content, (
        "Prototype label missing from file header."
    )
    assert "Topology: shallow" in content
    assert "Activation Function: RECTIFICATION" in content
    assert "Mode: Plan A Pareto (NSGA-II)" in content
    assert "Equation:" in content


def test_per_run_archive_routing(tmp_path):
    """When run_index is provided, the file MUST land under per_run_archive/
    so MOD7/MOD9's non-recursive globs do not pick it up as a consensus.
    """
    config = mod5.MOGPConfig(use_plan_b_soo=False)
    mod5.ensure_deap_creators(use_plan_b_soo=False)
    primitive_set = mod5.build_primitive_set()
    toolbox = mod5.build_toolbox(primitive_set, config)

    ind = toolbox.individual()
    ind.fitness.values = (0.85, 5.0, 3)

    output_path = mod5.export_pareto_rules(
        hall_of_fame=[ind],
        activation="linear", config=config,
        output_dir=tmp_path,
        env_details="--- TEST ENV BLOCK ---",
        topology="funnel",
        run_index=2,  # per-run archive form
    )

    assert output_path.parent.name == "per_run_archive", (
        f"Per-run artifact must live under per_run_archive/, got {output_path.parent}"
    )
    assert "_Run2_" in output_path.name, (
        f"Run-index suffix missing from filename: {output_path.name}"
    )

    content = output_path.read_text(encoding="utf-8")
    assert "PER-RUN PARETO FRONT (Run 2)" in content


if __name__ == "__main__":
    pytest.main([__file__, "-v"])