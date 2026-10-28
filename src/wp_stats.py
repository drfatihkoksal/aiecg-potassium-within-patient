"""Shared estimators for WP3-WP6.

Every interval is bootstrapped with resampling at the PATIENT level (Section 8):
patients contribute multiple pairs, so a naive pair-level bootstrap would
understate the intervals.
"""
import numpy as np
from scipy import stats


def cluster_bootstrap(df, fn, n_boot=2000, seed=11, cluster="subject_id"):
    """Bootstrap `fn(df) -> float or dict` resampling whole patients."""
    rng = np.random.default_rng(seed)
    groups = {k: np.asarray(v) for k, v in df.groupby(cluster).indices.items()}
    keys = np.array(list(groups.keys()))
    point = fn(df)
    keys_out = list(point) if isinstance(point, dict) else None
    draws = []
    for _ in range(n_boot):
        pick = rng.choice(keys, size=len(keys), replace=True)
        idx = np.concatenate([groups[k] for k in pick])
        try:
            draws.append(fn(df.iloc[idx]))
        except Exception:                                    # noqa: BLE001
            continue
    if keys_out is None:
        d = np.asarray(draws, dtype=float)
        return point, float(np.nanpercentile(d, 2.5)), float(np.nanpercentile(d, 97.5))
    out = {}
    for k in keys_out:
        d = np.array([x[k] for x in draws], dtype=float)
        out[k] = (point[k], float(np.nanpercentile(d, 2.5)), float(np.nanpercentile(d, 97.5)))
    return out


def deming(x, y, lam=1.0):
    """Deming regression slope/intercept. lam = var(err_x)/var(err_y) (Section 8:
    both axes carry measurement error, so OLS is not appropriate)."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    mx, my = x.mean(), y.mean()
    sxx = ((x - mx) ** 2).sum(); syy = ((y - my) ** 2).sum(); sxy = ((x - mx) * (y - my)).sum()
    d = syy - lam * sxx
    slope = (d + np.sqrt(d ** 2 + 4 * lam * sxy ** 2)) / (2 * sxy)
    return float(slope), float(my - slope * mx)


def icc_oneway(values, groups):
    """One-way random-effects ICC(1) with unequal group sizes, plus the variance
    components. Matches the WP3 mixed model y_ij = b0 + u_i + e_ij."""
    values = np.asarray(values, float)
    groups = np.asarray(groups)
    _, inv, cnt = np.unique(groups, return_inverse=True, return_counts=True)
    k = len(cnt)
    n = len(values)
    gm = values.mean()
    gmeans = np.bincount(inv, weights=values) / cnt
    ssb = (cnt * (gmeans - gm) ** 2).sum()
    ssw = ((values - gmeans[inv]) ** 2).sum()
    msb = ssb / (k - 1)
    msw = ssw / (n - k)
    n0 = (n - (cnt ** 2).sum() / n) / (k - 1)          # average group size
    var_b = max((msb - msw) / n0, 0.0)
    var_w = msw
    return {"icc": var_b / (var_b + var_w) if (var_b + var_w) > 0 else np.nan,
            "var_between": var_b, "var_within": var_w,
            "n_obs": int(n), "n_groups": int(k)}


def attenuation_correct(r, cv_analytical, sd_delta, mean_level):
    """Correct r for measurement error in the lab delta.

    Differencing two independent draws doubles the analytical error variance:
    var(err_delta) = 2 * (CV * mean_level)^2. Reliability of the observed delta
    is then 1 - var(err)/var(observed delta), and r_true = r / sqrt(reliability).
    """
    err_var = 2.0 * (cv_analytical * mean_level) ** 2
    rel = max(1.0 - err_var / (sd_delta ** 2), 1e-6)
    return float(r / np.sqrt(rel)), float(rel)


def bland_altman(a, b):
    """Bias and 95% limits of agreement, computed on the delta scale (WP4)."""
    d = np.asarray(a, float) - np.asarray(b, float)
    return {"bias": float(d.mean()), "sd_diff": float(d.std(ddof=1)),
            "loa_lower": float(d.mean() - 1.96 * d.std(ddof=1)),
            "loa_upper": float(d.mean() + 1.96 * d.std(ddof=1))}


def mdc95(sd_within):
    """Minimal detectable change at 95% (WP5): 1.96 * sqrt(2) * SD_w."""
    return float(1.96 * np.sqrt(2) * sd_within)
