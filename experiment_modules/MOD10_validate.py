"""Local validation for MOD10 — runs without pydantic/pytest.

Verifies:
    * Numeric helpers (cliffs_delta, bootstrap, holm) on synthetic data.
    * FailureModeTaxonomist on the real 75 MOD9 equations extracted from
      the user's repository export. Prints the distribution of categories.
    * HeDistanceAnalyzer pipeline on synthetic data with the known
      consensus Rank-1 rules.
    * SymPy parseability of every real MOD9 equation.

This is a sanity harness, not a test of MOD10's I/O. Tests of I/O live in
MOD10_unit_test.py and require pytest + pydantic in the runtime env.
"""

from __future__ import annotations
import sys, math, types
import numpy as np
import pandas as pd
import sympy as sp

# ---- stub pydantic so we can import MOD10 without it ----
if "pydantic" not in sys.modules:
    pydantic = types.ModuleType("pydantic")
    class _BaseModel:
        def __init__(self, **kw):
            for k, v in kw.items():
                setattr(self, k, v)
            for k, v in getattr(self.__class__, "__annotations__", {}).items():
                if not hasattr(self, k):
                    setattr(self, k, getattr(self.__class__, k, None))
    def _field(default=None, **_kw): return default
    def _field_validator(*_a, **_kw):
        def deco(fn): return classmethod(fn)
        return deco
    pydantic.BaseModel = _BaseModel
    pydantic.Field = _field
    pydantic.field_validator = _field_validator
    sys.modules["pydantic"] = pydantic

from MOD10_post_hoc_analyzer import (
    META_FEATURES, HE_CONSTANT, HE_FAN_IN_BY_TOPOLOGY,
    cliffs_delta, bootstrap_paired_median_ci, holm_bonferroni, safe_wilcoxon,
    he_target_variance, FailureModeTaxonomist, HeDistanceAnalyzer,
    ParetoRankComparator,
)

# ============================================================================
# 1. NUMERIC HELPERS
# ============================================================================
print("=" * 70)
print("1. NUMERIC HELPERS")
print("=" * 70)

x_dom = np.array([10, 11, 12]); y_dom = np.array([0, 1, 2])
assert cliffs_delta(x_dom, y_dom) == 1.0, "expected +1.0"
assert cliffs_delta(y_dom, x_dom) == -1.0, "expected -1.0"
assert cliffs_delta(x_dom, x_dom) == 0.0, "ties -> 0"
print("  cliffs_delta: OK  (dominance=+1, reverse=-1, ties=0)")

a = np.full(100, 5.0); b = np.full(100, 3.0)
med, lo, hi = bootstrap_paired_median_ci(a, b, n_boot=400, seed=0)
assert med == lo == hi == 2.0, f"constant diff should be tight; got ({med},{lo},{hi})"
print("  bootstrap_paired_median_ci: OK  (constant diff CI collapses)")

adj = holm_bonferroni([0.001, 0.01, 0.04, 0.06, 0.20])
assert np.all(adj <= 1.0), "p-values must be clipped"
assert adj[0] < adj[-1], "monotonicity broken"
print("  holm_bonferroni: OK  (clipped, non-decreasing)")

stat, p = safe_wilcoxon(np.array([1, 2, 3]), np.array([1, 2, 3]))
assert stat == 0.0 and p == 1.0, "zero-diff should short-circuit to (0.0, 1.0) per MOD8 convention"
print("  safe_wilcoxon: OK  (zero-diff short-circuit)")

# ============================================================================
# 2. TAXONOMY — against the REAL 75 MOD9 equations
# ============================================================================
print()
print("=" * 70)
print("2. FAILURE-MODE TAXONOMY against the real 75 MOD9 equations")
print("=" * 70)

# All 75 simplified equations as (activation, topology, rank, eq_str)
EQS = [
("aggregation","deep_narrow",1,"-0.0383016545544630"),
("aggregation","deep_narrow",2,"-0.0510272806050976"),
("aggregation","funnel",1,"0.00111964582576813"),
("aggregation","funnel",2,"0.0297326363843313*sin(feat_kurtosis) + 0.0151192344354004"),
("aggregation","funnel",3,"-0.000973762128391928"),
("aggregation","funnel",4,"0.000949007532534608"),
("aggregation","funnel",5,"0.00631934144237656*cos(feat_kurtosis*iqr_dev*n_d_ratio + 0.303047381024217)"),
("aggregation","shallow",1,"0.0297326363843313*sin(0.398492264818324/davies_bouldin)"),
("aggregation","shallow",2,"0.0222699067809933"),
("aggregation","shallow",3,"0.0297326363843313*log(Abs(hopkins) + 1/100000)"),
("aggregation","shallow",4,"0.0297326363843313*cos(hopkins)"),
("aggregation","shallow",5,"0.0297326363843313*silhouette"),
("linear","deep_narrow",1,"n_d_ratio**2"),
("linear","deep_narrow",2,"0.149943969093358*sqrt(Abs(n_d_ratio))"),
("linear","deep_narrow",3,"feat_kurtosis*n_d_ratio*target_entropy"),
("linear","deep_narrow",4,"0.172431541152805"),
("linear","deep_narrow",5,"0.0970277172177405*target_entropy"),
("linear","funnel",1,"0.785572724006331"),
("linear","funnel",2,"0.999736974700946"),
("linear","funnel",3,"exp(target_entropy/feat_kurtosis)"),
("linear","funnel",4,"hopkins/iqr_dev"),
("linear","funnel",5,"silhouette"),
("linear","shallow",1,"n_d_ratio - exp(pc_eigen)"),
("linear","shallow",2,"-silhouette - sqrt(Abs(feat_kurtosis + pc_eigen))"),
("linear","shallow",3,"-silhouette - sqrt(Abs(pc_eigen + sqrt(Abs(feat_kurtosis + pc_eigen))))"),
("linear","shallow",4,"sqrt(Abs(davies_bouldin - target_entropy**2*sqrt(Abs(davies_bouldin))))"),
("linear","shallow",5,"-silhouette - 0.826122334741168"),
("rectification","deep_narrow",1,"0.0195246700939174"),
("rectification","deep_narrow",2,"0.0129683302207380"),
("rectification","deep_narrow",3,"0.091837715145637*feat_kurtosis*sin(iqr_dev)"),
("rectification","deep_narrow",4,"0.0297326363843313*pc_eigen - 0.00557273290319536"),
("rectification","deep_narrow",5,"-0.0297326363843313"),
("rectification","funnel",1,"-0.0633807484236808*pc_eigen*sqrt(Abs(n_d_ratio))"),
("rectification","funnel",2,"iqr_dev*cos(log(Abs(silhouette) + 1/100000))"),
("rectification","funnel",3,"0.0294219026838247"),
("rectification","funnel",4,"sin(0.0679651755178023*hopkins)"),
("rectification","funnel",5,"-0.0297326363843313"),
("rectification","shallow",1,"-sin(0.0243970223673645*iqr_dev*n_d_ratio)"),
("rectification","shallow",2,"-0.0229362957012469*n_d_ratio"),
("rectification","shallow",3,"-sin(0.0183273290034117*n_d_ratio)"),
("rectification","shallow",4,"0.0229362957012469"),
("rectification","shallow",5,"0.0297326363843313*feat_kurtosis"),
("smooth","deep_narrow",1,"-0.119742648004175*sin(sin(iqr_dev))"),
("smooth","deep_narrow",2,"-0.0308059658594664*silhouette"),
("smooth","deep_narrow",3,"0.0308059658594664"),
("smooth","deep_narrow",4,"-0.0367699337044863"),
("smooth","deep_narrow",5,"log(Abs(cos(iqr_dev)) + 1/100000)"),
("smooth","funnel",1,"0.0297326363843313*davies_bouldin - 0.0297326363843313*target_entropy"),
("smooth","funnel",2,"-0.0587513001999703"),
("smooth","funnel",3,"0.0619261093149091"),
("smooth","funnel",4,"hopkins*iqr_dev*n_d_ratio*pc_eigen"),
("smooth","funnel",5,"-0.105534243576035"),
("smooth","shallow",1,"log((Abs(n_d_ratio) + 1/100000)**(-0.0308059658594664))"),
("smooth","shallow",2,"-0.012968330220738*sqrt(Abs(target_entropy))"),
("smooth","shallow",3,"-0.012968330220738*pc_eigen"),
("smooth","shallow",4,"-0.012968330220738*feat_kurtosis"),
("smooth","shallow",5,"0.0187796920530200"),
("squashing","deep_narrow",1,"0.0297326363843313*sqrt(Abs(feat_kurtosis)) + 0.00465900679264226"),
("squashing","deep_narrow",2,"0.0594652727686626*sqrt(Abs(feat_kurtosis))"),
("squashing","deep_narrow",3,"0.0236475935460351"),
("squashing","deep_narrow",4,"0.0297326363843313*cos(sin(davies_bouldin + silhouette + exp(pc_eigen)))"),
("squashing","deep_narrow",5,"0.0245538641063852"),
("squashing","funnel",1,"-0.0229362957012469*silhouette"),
("squashing","funnel",2,"-0.00476004227707586*feat_kurtosis - 0.0113355926872378"),
("squashing","funnel",3,"-0.0125921576985643"),
("squashing","funnel",4,"-0.00697079950872524"),
("squashing","funnel",5,"0.0129683302207380"),
("squashing","shallow",1,"0.0432484823334625*n_d_ratio"),
("squashing","shallow",2,"-0.0308059658594664*sqrt(Abs(target_entropy))"),
("squashing","shallow",3,"-0.00729057326381466"),
("squashing","shallow",4,"0.0308059658594664"),
("squashing","shallow",5,"-0.0229362957012469*target_entropy"),
]
print(f"  N equations to test: {len(EQS)}")

class _MiniCfg:  pass
tax = FailureModeTaxonomist(_MiniCfg())
syms = {m: sp.Symbol(m, real=True) for m in META_FEATURES}
local = {**syms, "Abs": sp.Abs, "log": sp.log, "exp": sp.exp,
         "sqrt": sp.sqrt, "sin": sp.sin, "cos": sp.cos}
parse_fail = 0
cats = []
for (a, t, r, s) in EQS:
    try:
        expr = sp.parse_expr(s, local_dict=local, evaluate=True)
    except Exception as e:
        print(f"    PARSE FAIL: {a}/{t}/Rank{r}: {e}")
        parse_fail += 1
        cats.append((a,t,r,s,"PARSE_FAIL"))
        continue
    c = tax.classify(expr)
    cats.append((a,t,r,s,c))
print(f"  parse failures: {parse_fail}")
from collections import Counter
counts = Counter(c[4] for c in cats)
print("  category distribution:")
for k, v in counts.most_common():
    print(f"    {k:24s} {v}")

# print linear-family categorization (should mostly be 'unbounded' or 'protected_artifact')
lin = [c for c in cats if c[0] == "linear"]
print(f"  linear-family categories: {Counter(c[4] for c in lin).most_common()}")

# print the 4 consensus Rank-1 rules that are constants — sanity
const_r1 = [c for c in cats if c[2]==1 and c[4]=="pure_constant"]
print(f"  Rank-1 pure_constant cells: {[(c[0],c[1]) for c in const_r1]}")

# ============================================================================
# 3. HE-DISTANCE — synthetic Phase B meta-features
# ============================================================================
print()
print("=" * 70)
print("3. HE-DISTANCE on a synthetic Phase B with He-scale constant rule")
print("=" * 70)
cfg = _MiniCfg(); cfg.sigma_floor = 1e-5; cfg.save_dpi = 100
meta = pd.DataFrame({"did": np.arange(25)})
rng = np.random.default_rng(0)
for m in META_FEATURES:
    meta[m] = rng.normal(0, 1, 25)
rules = pd.DataFrame([
    {"activation":"squashing","topology":"deep_narrow","rank":1,
     "equation_str":"0.0297326363843313*sqrt(Abs(feat_kurtosis)) + 0.00465900679264226",
     "expr": sp.parse_expr("0.0297326363843313*sqrt(Abs(feat_kurtosis)) + 0.00465900679264226",
                           local_dict=local)},
    {"activation":"squashing","topology":"shallow","rank":1,
     "equation_str":"0.0432484823334625*n_d_ratio",
     "expr": sp.parse_expr("0.0432484823334625*n_d_ratio", local_dict=local)},
])
tbl = HeDistanceAnalyzer(cfg).run(rules, meta)
print(tbl[["activation","topology","sigma2_median","He_target_sigma2",
           "ratio_median_to_He","n_pinned_to_floor"]].to_string(index=False))

# ============================================================================
# 4. PARETO RANK comparator on the deep_narrow_linear pathology
# ============================================================================
print()
print("=" * 70)
print("4. PARETO-RANK CROSS-COMPARISON (linear-cell pathology check)")
print("=" * 70)
summary = pd.DataFrame([
    {"topology":"deep_narrow","activation":"linear","method":"GP_Rule_1","Accuracy_Mean":70.57,"Loss_Mean":1.98e7,"Epochs_Mean":29.0},
    {"topology":"deep_narrow","activation":"linear","method":"GP_Rule_2","Accuracy_Mean":71.0, "Loss_Mean":3.9,    "Epochs_Mean":28.0},
    {"topology":"deep_narrow","activation":"linear","method":"GP_Rule_3","Accuracy_Mean":70.8, "Loss_Mean":4.5,    "Epochs_Mean":27.5},
    {"topology":"deep_narrow","activation":"squashing","method":"GP_Rule_1","Accuracy_Mean":77.32,"Loss_Mean":0.4766,"Epochs_Mean":18.5},
    {"topology":"deep_narrow","activation":"squashing","method":"GP_Rule_2","Accuracy_Mean":77.07,"Loss_Mean":0.5408,"Epochs_Mean":19.0},
])
prc = ParetoRankComparator(_MiniCfg()); tbl = prc.run(summary)
print(tbl.to_string(index=False))
print()
print("validation complete.")
