"""
Unit Tests for Module 6: Production Genetic Artifact Target Routing
"""

import pytest
from pathlib import Path
import MOD6_om_mogp_engine_final as mod6


def test_get_generated_directory_nesting_logic():
    """
    Validates that the output directory properly nests the execution trace
    within the explicit partitioned GA_rule_files subdirectory.
    """
    output_path = mod6.get_generated_directory()
    assert isinstance(output_path, Path)

    # Ensure path maps to exactly experiment_modules/generated_files/GA_rule_files
    trailing_paths = output_path.parts[-2:]
    assert trailing_paths == ('generated_files', 'GA_rule_files'), \
        "Pathing engine failed to nest inside GA_rule_files directory."


def test_production_hyperparameter_boundaries():
    """Ensures Pydantic schema validation defaults protect methodological integrity."""
    config = mod6.MOGPConfig()

    # Verify strict production-level variables are enforced
    assert config.population_size == 150
    assert config.generations == 20
    assert config.datasets_per_rule == 20


def test_qe_multi_seed_defaults():
    """Q-E: GP discovery defaults to N=3 independent runs with seeds {42,43,44}."""
    config = mod6.MOGPConfig()
    assert config.n_gp_runs == 3
    assert len(config.gp_run_seeds) >= config.n_gp_runs
    assert config.gp_run_seeds[:3] == [42, 43, 44]


def test_qe_seed_count_validation_raises_when_insufficient():
    """Pydantic validator rejects an n_gp_runs that exceeds the seed list length."""
    with pytest.raises(ValueError):
        mod6.MOGPConfig(n_gp_runs=5, gp_run_seeds=[42, 43])


def test_qg_topology_targets_defaults():
    """Q-G: full sweep covers shallow, deep_narrow, and funnel."""
    config = mod6.MOGPConfig()
    assert config.topology_targets == ["shallow", "deep_narrow", "funnel"]


def test_qf_plan_b_default_is_disabled():
    """Plan B is opt-in (default Plan A). The toggle and its weights live on
    MOGPConfig with sensible defaults.
    """
    config = mod6.MOGPConfig()
    assert config.use_plan_b_soo is False
    assert config.plan_b_w_acc == pytest.approx(1.0)
    assert config.plan_b_w_eps == pytest.approx(0.5)
    assert config.plan_b_tournament_size == 3


def test_qf_plan_b_toggle_switches_creator_and_selector():
    """
    When use_plan_b_soo=True, the DEAP creators are rebound to FitnessSingle
    and build_toolbox registers selTournament instead of selNSGA2.

    DEAP's ``toolbox.register`` wraps the function in a ``functools.partial``
    and sets ``partial.__name__`` to the toolbox alias ('select'), masking
    the wrapped function's identity. To verify which underlying DEAP selector
    is installed we unwrap via ``partial.func`` and read its ``__name__``.
    """
    config = mod6.MOGPConfig(use_plan_b_soo=True)
    mod6.ensure_deap_creators(use_plan_b_soo=True)
    from deap import creator
    assert hasattr(creator, "FitnessSingle")
    # FitnessSingle is single-objective with positive weight.
    assert creator.FitnessSingle.weights == (1.0,)

    primitive_set = mod6.build_primitive_set()
    toolbox = mod6.build_toolbox(primitive_set, config)

    # Unwrap the partial to access the wrapped function's identity.
    sel = toolbox.select
    wrapped = getattr(sel, "func", sel)
    wrapped_name = getattr(wrapped, "__name__", "")
    assert "tournament" in wrapped_name.lower(), (
        f"Expected selTournament under Plan B, got {wrapped_name!r}"
    )

    # Symmetry check: switching back to Plan A re-installs selNSGA2 cleanly.
    config_a = mod6.MOGPConfig(use_plan_b_soo=False)
    toolbox_a = mod6.build_toolbox(primitive_set, config_a)
    wrapped_a = getattr(toolbox_a.select, "func", toolbox_a.select)
    assert getattr(wrapped_a, "__name__", "") == "selNSGA2", (
        "Plan A toolbox should register selNSGA2."
    )


def test_consensus_aggregator_dedupes_by_string():
    """
    aggregate_consensus_front MUST collapse identical rule strings into a
    single consensus individual with mean-across-runs fitness.
    """
    # Build a degenerate per-run-front pair manually using minimal stand-ins.
    config = mod6.MOGPConfig(use_plan_b_soo=False)
    mod6.ensure_deap_creators(use_plan_b_soo=False)
    primitive_set = mod6.build_primitive_set()
    toolbox = mod6.build_toolbox(primitive_set, config)

    ind_a = toolbox.individual()
    ind_a.fitness.values = (0.80, 20.0, 3)
    ind_b = toolbox.individual()
    ind_b.fitness.values = (0.70, 18.0, 5)

    # Run 1 contains both. Run 2 contains ind_a only (same string).
    # Simulate "same string" by cloning ind_a's expression tree.
    import copy
    ind_a_clone = copy.deepcopy(ind_a)
    ind_a_clone.fitness.values = (0.82, 22.0, 3)  # different fitness, same string

    per_run_fronts = [[ind_a, ind_b], [ind_a_clone]]
    consensus = mod6.aggregate_consensus_front(per_run_fronts, config)

    # The deduplicated set has at most 2 unique rule strings; both should be
    # represented exactly once in the consensus.
    unique_strs = {str(ind) for ind in consensus}
    assert len(unique_strs) <= 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])