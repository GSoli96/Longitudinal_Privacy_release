"""Statistics of the analysis: uncertainty, trend tests and inter-rater agreement (pure functions, no I/O).

* ``cluster_bootstrap_ci``: percentile CI obtained by resampling whole units (persons) with replacement;
  resamples whose statistic is nan are ignored and counted out of ``n_valid_resamples``.
* ``permutation_trend_test``: statistic = mean over units (at least two non-missing values) of the OLS slope
  of the values on the step index 0, 1, 2, ...; null = the values (with their missing pattern) are
  permuted among the steps of each unit independently; two-sided p = (1 + #{|perm| >= |obs|}) / (1 + B).
* ``mixed_model_trend``: ``value ~ step`` with a random intercept per unit (REML). It never raises: with too
  little data, a failed fit or a convergence warning it returns nan values and ``converged=False``.
* Agreement: ``percent_agreement``, ``cohen_kappa`` (2 raters), ``fleiss_kappa`` (m raters, same m per unit),
  ``krippendorff_alpha_nominal`` (missing values allowed), ``gwet_ac1`` (m raters), all nominal labels.
  Pairs or units with a missing label (None) are dropped as documented per function; an undefined
  coefficient (zero variance, expected agreement 1) is nan.
* All functions are deterministic given ``seed`` and keep no global state.
"""
from __future__ import annotations

import math
import warnings
from collections import Counter

import numpy as np

NAN = float("nan")
TOL = 1e-12  # tolerance on |perm stat| >= |obs stat| against floating-point noise


def _isnan(x) -> bool:
    return x is None or (isinstance(x, float) and math.isnan(x))


# ---- cluster bootstrap ---------------------------------------------------------------------------
def cluster_bootstrap_ci(units, statistic, n_resamples: int = 2000, alpha: float = 0.05, seed: int = 42) -> dict:
    """Percentile bootstrap CI of ``statistic(units)`` resampling units with replacement."""
    units = list(units)
    n = len(units)
    if n == 0:
        return {"estimate": NAN, "lo": NAN, "hi": NAN, "n_units": 0, "n_valid_resamples": 0}
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_resamples, n))
    vals = np.array([float(statistic([units[i] for i in row])) for row in idx], dtype=float)
    vals = vals[~np.isnan(vals)]
    if vals.size:
        lo, hi = (float(q) for q in np.quantile(vals, [alpha / 2, 1 - alpha / 2]))
    else:
        lo = hi = NAN
    return {"estimate": float(statistic(units)), "lo": lo, "hi": hi, "n_units": n,
            "n_valid_resamples": int(vals.size)}


# ---- permutation trend test ----------------------------------------------------------------------
def _slopes(y: np.ndarray, x: np.ndarray) -> np.ndarray:
    """OLS slope of each row of ``y`` (nan = missing) on ``x``; rows need at least two valid values."""
    ok = ~np.isnan(y)
    k = ok.sum(axis=1)
    xs = np.where(ok, x, 0.0)
    xm = xs.sum(axis=1) / k
    ym = np.where(ok, y, 0.0).sum(axis=1) / k
    dx = np.where(ok, x - xm[:, None], 0.0)
    dy = np.where(ok, y - ym[:, None], 0.0)
    return (dx * dy).sum(axis=1) / (dx * dx).sum(axis=1)


def permutation_trend_test(values_by_unit: dict, n_permutations: int = 10000, seed: int = 42) -> dict:
    """Mean within-unit OLS slope against the step index, with a within-unit permutation p-value."""
    rows = []
    for vals in values_by_unit.values():
        y = np.array([NAN if _isnan(v) else float(v) for v in vals], dtype=float)
        if np.count_nonzero(~np.isnan(y)) >= 2:
            rows.append(y)
    if not rows:
        return {"slope": NAN, "p_value": NAN, "n_units": 0}
    rng = np.random.default_rng(seed)
    obs = 0.0
    perm = np.zeros(n_permutations)
    for y in rows:
        x = np.arange(len(y), dtype=float)
        obs += _slopes(y[None, :], x)[0]
        order = rng.random((n_permutations, len(y))).argsort(axis=1)
        perm += _slopes(y[order], x)
    obs /= len(rows)
    perm /= len(rows)
    extreme = int(np.count_nonzero(np.abs(perm) >= abs(obs) - TOL))
    return {"slope": float(obs), "p_value": (1 + extreme) / (1 + n_permutations), "n_units": len(rows)}


# ---- mixed model ---------------------------------------------------------------------------------
def mixed_model_trend(rows) -> dict:
    """``value ~ step`` with a random intercept per unit (REML); ``rows`` = (unit, step_index, value)."""
    import pandas as pd
    import statsmodels.formula.api as smf
    from statsmodels.tools.sm_exceptions import ConvergenceWarning

    data = [(u, float(s), float(v)) for u, s, v in rows if not _isnan(v)]
    n_obs, n_units = len(data), len({u for u, _, _ in data})
    out = {"slope": NAN, "se": NAN, "p_value": NAN, "n_obs": n_obs, "n_units": n_units, "converged": False}
    if n_units < 2 or n_obs < 3 or len({s for _, s, _ in data}) < 2:
        return out
    df = pd.DataFrame(data, columns=["unit", "step", "value"])
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            warnings.simplefilter("error", ConvergenceWarning)
            fit = smf.mixedlm("value ~ step", df, groups=df["unit"]).fit(reml=True)
        slope, se, p = float(fit.params["step"]), float(fit.bse["step"]), float(fit.pvalues["step"])
    except Exception:  # singular or non-convergent fit: report it, never raise
        return out
    if not all(math.isfinite(v) for v in (slope, se, p)):
        return out
    out.update(slope=slope, se=se, p_value=p, converged=bool(fit.converged))
    return out


# ---- agreement coefficients ----------------------------------------------------------------------
def _pairs(a, b) -> list:
    """Aligned (a, b) pairs without None; unequal lengths raise ValueError."""
    a, b = list(a), list(b)
    if len(a) != len(b):
        raise ValueError(f"label lists differ in length: {len(a)} != {len(b)}")
    return [(x, y) for x, y in zip(a, b) if x is not None and y is not None]


def percent_agreement(a, b) -> float:
    pairs = _pairs(a, b)
    return sum(x == y for x, y in pairs) / len(pairs) if pairs else NAN


def cohen_kappa(a, b) -> float:
    pairs = _pairs(a, b)
    n = len(pairs)
    if n == 0:
        return NAN
    po = sum(x == y for x, y in pairs) / n
    ca, cb = Counter(x for x, _ in pairs), Counter(y for _, y in pairs)
    pe = sum(ca[k] * cb[k] for k in ca) / (n * n)
    return NAN if pe >= 1 else (po - pe) / (1 - pe)


def _complete_units(ratings) -> list:
    """Units without None that have the most frequent number of ratings (ties: the larger)."""
    units = [list(u) for u in ratings if all(r is not None for r in u)]
    if not units:
        return []
    sizes = Counter(len(u) for u in units)
    m = max(sizes, key=lambda s: (sizes[s], s))
    return [u for u in units if len(u) == m]


def fleiss_kappa(ratings) -> float:
    units = _complete_units(ratings)
    if not units or len(units[0]) < 2:
        return NAN
    n, m = len(units), len(units[0])
    counts = [Counter(u) for u in units]
    pa = sum((sum(c * c for c in cnt.values()) - m) / (m * (m - 1)) for cnt in counts) / n
    totals = Counter()
    for cnt in counts:
        totals.update(cnt)
    pe = sum((t / (n * m)) ** 2 for t in totals.values())
    return NAN if pe >= 1 else (pa - pe) / (1 - pe)


def krippendorff_alpha_nominal(ratings) -> float:
    """Nominal alpha from the coincidence matrix; units with fewer than two valid values are ignored."""
    o = Counter()  # coincidences o[c, k]
    for u in ratings:
        vals = [r for r in u if r is not None]
        mu = len(vals)
        if mu < 2:
            continue
        cnt = Counter(vals)
        for c in cnt:
            for k in cnt:
                pairs = cnt[c] * (cnt[k] - (c == k))
                o[c, k] += pairs / (mu - 1)
    nc = Counter()
    for (c, _), v in o.items():
        nc[c] += v
    n = sum(nc.values())
    if n < 2:
        return NAN
    d_obs = sum(v for (c, k), v in o.items() if c != k)
    d_exp = sum(nc[c] * nc[k] for c in nc for k in nc if c != k)
    return NAN if d_exp == 0 else 1 - (n - 1) * d_obs / d_exp


def gwet_ac1(ratings, n_categories: int | None = None) -> float:
    """AC1 with pe = sum_k pi_k (1 - pi_k) / (q - 1). q is ``n_categories`` if given (the number of possible
    labels, 2 for yes/no), else the number of observed labels; q = 1 gives pe = 0."""
    units = _complete_units(ratings)
    if not units or len(units[0]) < 2:
        return NAN
    n, m = len(units), len(units[0])
    counts = [Counter(u) for u in units]
    pa = sum(sum(c * (c - 1) for c in cnt.values()) / (m * (m - 1)) for cnt in counts) / n
    totals = Counter()
    for cnt in counts:
        totals.update(cnt)
    q = n_categories or len(totals)
    pe = sum((t / (n * m)) * (1 - t / (n * m)) for t in totals.values()) / (q - 1) if q > 1 else 0.0
    return (pa - pe) / (1 - pe)


def agreement_summary(a, b) -> dict:
    pairs = _pairs(a, b)
    return {"percent_agreement": percent_agreement(a, b), "cohen_kappa": cohen_kappa(a, b),
            "gwet_ac1": gwet_ac1([[x, y] for x, y in pairs])}


# ---- precision / recall --------------------------------------------------------------------------
def precision_recall(pred, truth, positive) -> dict:
    pairs = _pairs(pred, truth)
    tp = sum(p == positive and t == positive for p, t in pairs)
    fp = sum(p == positive and t != positive for p, t in pairs)
    fn = sum(p != positive and t == positive for p, t in pairs)
    return {"precision": tp / (tp + fp) if tp + fp else NAN,
            "recall": tp / (tp + fn) if tp + fn else NAN,
            "f1": 2 * tp / (2 * tp + fp + fn) if tp + fp + fn else NAN,
            "tp": tp, "fp": fp, "fn": fn}
