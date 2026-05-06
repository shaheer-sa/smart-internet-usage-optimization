# ╔══════════════════════════════════════════════════════════════════╗
# ║  prob_stats_tools.py — NetSmart Probabilistic & Statistical     ║
# ║  Analysis Toolkit                                               ║
# ║                                                                  ║
# ║  Pure-Python / NumPy / SciPy functions that implement the three  ║
# ║  probability distributions used across the application:          ║
# ║    • Normal (Gaussian)  — speed uncertainty / bell curve         ║
# ║    • Poisson            — user-load / congestion modelling        ║
# ║    • Binomial           — weekly slow-session reliability         ║
# ╚══════════════════════════════════════════════════════════════════╝

import numpy       as np
import scipy.stats as stats

from config import SLOW_THRESHOLD


# ════════════════════════════════════════════════════════════════════
#   GAUSSIAN (NORMAL) DISTRIBUTION TOOLS
# ════════════════════════════════════════════════════════════════════

def gaussianSlowProbability(mu: float, sigma: float) -> float:
    """
    Calculate the probability that a session falls below the slow
    threshold using the Gaussian CDF.

    P(speed < SLOW_THRESHOLD | μ=mu, σ=sigma) = Φ((threshold − μ) / σ)

    Parameters
    ----------
    mu    : float — Mean predicted speed (Mbps)
    sigma : float — Standard deviation / RMSE of model (Mbps)

    Returns
    -------
    float — Probability in [0, 1]
    """
    if sigma <= 0:
        sigma = 1e-6    # Guard against zero division
    return float(stats.norm.cdf(SLOW_THRESHOLD, loc=mu, scale=sigma))


def gaussianPdfCurve(mu: float, sigma: float, n_points: int = 400):
    """
    Generate x-y arrays for plotting the Normal PDF bell curve.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        (x_values, y_pdf_values) — covering ±4σ around the mean.
    """
    x = np.linspace(mu - 4 * sigma, mu + 4 * sigma, n_points)
    y = stats.norm.pdf(x, mu, sigma)
    return x, y


def gaussianIntervalMask(x: np.ndarray, mu: float, sigma: float):
    """
    Return a boolean mask for values within ±1σ of the mean (the
    68% shaded region on a bell curve plot).
    """
    return (x >= mu - sigma) & (x <= mu + sigma)


def gaussianDescriptiveStats(series) -> dict:
    """
    Return a dictionary of basic descriptive statistics for a
    numeric pandas Series (or array-like).

    Returns
    -------
    dict with keys: mean, std, variance, median, range, slow_pct
    """
    arr  = np.asarray(series, dtype=float)
    mu   = float(arr.mean())
    sigma = float(arr.std())
    return {
        "mean":     round(mu, 4),
        "std":      round(sigma, 4),
        "variance": round(float(arr.var()), 4),
        "median":   round(float(np.median(arr)), 4),
        "range":    round(float(arr.max() - arr.min()), 4),
        "slow_pct": round(gaussianSlowProbability(mu, sigma) * 100, 2),
    }


# ════════════════════════════════════════════════════════════════════
#   POISSON DISTRIBUTION TOOLS
# ════════════════════════════════════════════════════════════════════

def poissonPmfCurve(lam: float, k_max: int = 85):
    """
    Compute the Poisson PMF for k = 0 … k_max given rate λ.

    This models the probability of exactly k students being online
    simultaneously.

    Parameters
    ----------
    lam   : float — Average user count (λ = mean of NumberOfUsers)
    k_max : int   — Upper bound for the k-axis

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        (k_values, pmf_values)
    """
    k  = np.arange(0, k_max + 1)
    pmf = stats.poisson.pmf(k, lam)
    return k, pmf


def poissonTailProbabilities(lam: float, k_max: int = 85) -> dict:
    """
    Compute practically useful Poisson tail probabilities for the
    user-count interpretation:

    • P(k > 50)  — probability of severe congestion
    • P(k < 20)  — probability of a quiet / fast session

    Returns
    -------
    dict with keys: lambda, peak_k, p_over_50, p_under_20
    """
    k, pmf = poissonPmfCurve(lam, k_max)
    peak_k    = int(k[np.argmax(pmf)])
    p_over_50 = float(sum(stats.poisson.pmf(i, lam) for i in range(50, k_max + 1)))
    p_under_20 = float(sum(stats.poisson.pmf(i, lam) for i in range(0, 20)))
    return {
        "lambda":     round(lam, 2),
        "peak_k":     peak_k,
        "p_over_50":  round(p_over_50, 4),
        "p_under_20": round(p_under_20, 4),
    }


def poissonInstantProbability(k: int, lam: float) -> float:
    """
    P(K = k | λ) — probability that exactly k users are online.

    Useful for the Speed-Check page mathematical breakdown section.
    """
    return float(stats.poisson.pmf(int(k), mu=lam))


# ════════════════════════════════════════════════════════════════════
#   BINOMIAL DISTRIBUTION TOOLS
# ════════════════════════════════════════════════════════════════════

def binomialPmfCurve(n_trials: int, p_slow: float):
    """
    Compute the Binomial PMF for k = 0 … n_trials slow sessions.

    Models: "In n_trials sessions, how many will be slow?"

    Parameters
    ----------
    n_trials : int   — Total sessions (e.g. 50)
    p_slow   : float — Probability any single session is slow

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        (k_values, pmf_values)
    """
    k   = np.arange(0, n_trials + 1)
    pmf = stats.binom.pmf(k, n_trials, p_slow)
    return k, pmf


def binomialExpected(n_trials: int, p_slow: float) -> float:
    """Expected number of slow sessions = n * p."""
    return n_trials * p_slow


def binomialWeeklyReliability(p_slow: float,
                               n_days: int = 7,
                               min_slow_days: int = 3) -> float:
    """
    Probability of experiencing at least `min_slow_days` slow
    sessions in `n_days` days (Binomial CDF complement).

    Returns
    -------
    float — Probability in [0, 1]
    """
    # P(X >= min_slow_days) = 1 - P(X <= min_slow_days - 1)
    return float(1.0 - stats.binom.cdf(min_slow_days - 1, n=n_days, p=p_slow))


def binomialTailProbability(n_trials: int, p_slow: float,
                             k_threshold: int) -> float:
    """
    P(X > k_threshold) for a Binomial(n, p) variable.

    Useful for: "probability of more than k_threshold slow sessions."
    """
    k, pmf = binomialPmfCurve(n_trials, p_slow)
    return float(sum(pmf[i] for i in k if i > k_threshold))
