import numpy as np
import pandas as pd
from scipy.stats import gamma


def GBSO(S, GBS, Y):
    '''
    Graph-Based Spatial Outlier Detection (GBSO).

    Computes the standardised absolute deviation of each observation from
    the mean of its Voronoi neighbours.

    Based on: Shekhar, S., Lu, C. T., & Zhang, P. (2001). Detecting
    graph-based spatial outliers: algorithms and applications.

    Parameters
    ----------
    S : array of float, shape (N, 2)
        Spatial attribute 2D: Longitude-Latitude coordinates.
    GBS : list of list of int
        Voronoi neighbourhood: GBS[i] contains the indices of the
        neighbours of observation i.
    Y : array of float, shape (N,)
        Non-spatial attribute values (the variable to analyse).

    Returns
    -------
    detectOutlier : pd.Series, shape (N,)
        Standardised GBSO outlier scores.
    '''
    AvgDist_Set = []
    for i in range(len(GBS)):
        AvgDist_Set.append(Y[i] - np.mean(Y[GBS[i]]))

    mu = np.mean(AvgDist_Set)
    sigma = np.std(AvgDist_Set)

    detectOutlier = abs((AvgDist_Set - mu) / sigma)
    return pd.Series(detectOutlier)


def SWOD(X, Y, k):
    '''
    Spatial Weighted Outlier Detection (SWOD).

    Like GBSO but weights each neighbour by its inverse distance,
    giving more influence to closer neighbours.

    Based on: Kou, Y., Lu, C. T., & Chen, D. (2006). Spatial weighted
    outlier detection.

    Parameters
    ----------
    X : array of float, shape (N, N)
        Pairwise Euclidean distances between geographic points.
    Y : array of float, shape (N,)
        Non-spatial attribute values.
    k : list of list of int
        Voronoi neighbourhood: k[i] contains the indices of the
        neighbours of observation i.

    Returns
    -------
    OF : pd.Series, shape (N,)
        Standardised SWOD outlier scores.
    '''
    size = len(X)
    weight = [0] * size
    for i in range(size):
        weight[i] = list(X[i, k[i]] / (sum(X[i, k[i]])))

    NbrAvg = [0] * size
    for i in range(size):
        NbrAvg[i] = sum(Y[k[i]] * weight[i])

    diff = [0] * size
    for i in range(size):
        diff[i] = Y[i] - NbrAvg[i]

    mu = np.mean(diff)
    sigma = np.std(diff)

    OF = [0] * size
    for i in range(size):
        OF[i] = abs((diff[i] - mu) / sigma)

    return pd.Series(OF)


def spatial_LOF(dist, Vnn, k):
    '''
    Local Outlier Factor for Spatial Data (Spatial LOF).

    Adaptation of the classic LOF (Breunig et al., 2000) that uses the
    Voronoi neighbourhood instead of simple Euclidean k-NN.

    Parameters
    ----------
    dist : array of float, shape (N, N)
        Distances between values (non-spatial attribute distances).
    Vnn : list of list of int
        Voronoi neighbourhood.
    k : array of int, shape (N,)
        Number of neighbours per observation.

    Returns
    -------
    LOF : pd.Series, shape (N,)
        LOF outlier scores.
    '''
    size = len(Vnn)
    order = np.argsort(dist)

    idx_Vnn = []
    for i in range(size):
        idx_Vnn.append(list(order[i, :len(Vnn[i])]))

    radius = [0] * size
    for i in range(size):
        radius[i] = dist[i, idx_Vnn[i][-1]]
    radius = np.array(radius)

    LRD = []
    for i in range(size):
        LRD.append(np.mean(np.maximum(dist[i, idx_Vnn[i]], radius[idx_Vnn[i]])))

    rho = 1. / np.array(LRD)

    LOF = [0] * size
    for i in range(size):
        LOF[i] = np.sum(rho[idx_Vnn[i]]) / np.array(rho[i])

    LOF *= 1. / k
    return pd.Series(LOF)


def gamma_standardization(Score):
    '''
    Gamma Cumulative Density Function Standardization.

    Maps raw outlier scores to a [0, 1] probability scale using the
    fitted Gamma CDF, enabling comparison across algorithms and datasets.

    Parameters
    ----------
    Score : array-like, shape (N,)
        Raw outlier scores (must be positive).

    Returns
    -------
    standard_score_gamma : pd.Series, shape (N,)
        Standardised scores in [0, 1].
    '''
    mu = np.mean(Score)
    sigma = np.std(Score)

    shape, scale = (mu ** 2) / (sigma ** 2), sigma / (mu ** 2)

    cdf_gamma = gamma.cdf(Score, a=shape, scale=1 / scale)
    gamma_mu_s = gamma.cdf(mu, a=shape, scale=1 / scale)

    standard_score_gamma = pd.Series(
        np.maximum(0, (cdf_gamma - gamma_mu_s) / (1 - gamma_mu_s))
    )
    return standard_score_gamma


# ─── SPATIO-TEMPORAL EXTENSION ────────────────────────────────────────────────

def GBSO_ST(Y_matrix, Vnn, k=24, theta=2.0, verbose=True):
    '''
    Spatio-Temporal Graph-Based Spatial Outlier (GBSO_ST).

    Extension of GBSO to the temporal dimension. For each observation i
    at time step t, the score measures the deviation of Y_i(t) from the
    spatio-temporal distribution of its Voronoi neighbours within a
    centred temporal window [t-k, t+k]:

        GBSO_ST(i, t) = |Y_i(t) - mean{ Y_j(s) | j in Vnn(i), s in [t-k,t+k] }|
                        ─────────────────────────────────────────────────────────
                                         sigma_t

    This detects anomalies that are simultaneously:
      1. Unusual with respect to the immediate geospatial environment
         (Voronoi neighbours).
      2. Unusual with respect to the recent behaviour of that environment
         (temporal window of half-width k).

    Design notes
    ------------
    - The Voronoi neighbourhood is fixed in space (computed once from
      coordinates) and reused at every time step.
    - The temporal window is centred at t (non-causal). For online or
      streaming applications, use a causal window [t-2k, t] instead.
    - Missing values (NaN) are excluded from the window mean.
    - At least 3 valid values in the window are required to compute a score.

    Parameters
    ----------
    Y_matrix : np.ndarray, shape (T, N)
        Matrix of observed values indexed by (time, observation).
        NaN indicates a missing observation.
    Vnn : list of list of int
        Voronoi neighbourhood: Vnn[i] contains the indices of the
        Voronoi neighbours of observation i.
    k : int
        Half-width of the temporal window in time steps.
        Typical values:
          k=12  → window of 25 steps  (e.g. ±12 hours)
          k=24  → window of 49 steps  (e.g. ±1 day)
          k=168 → window of 337 steps (e.g. ±1 week)
    theta : float
        Outlier threshold in standard deviations (default 2.0).
    verbose : bool
        If True, prints progress information.

    Returns
    -------
    scores : np.ndarray, shape (T, N)
        Standardised GBSO_ST scores. NaN where no observation or the
        window contains fewer than 3 valid values.
    mu_st : np.ndarray, shape (T, N)
        Spatio-temporal reference mean for each (observation, time) pair.
    is_outlier : np.ndarray of bool, shape (T, N)
        True where score >= theta and observation is not NaN.

    Example
    -------
    >>> import numpy as np
    >>> from src.voronoi import voronoi_neighbors
    >>> from src.models import GBSO_ST
    >>>
    >>> # coords: (N, 2) array of coordinates
    >>> # Y_matrix: (T, N) array of hourly observations
    >>> dist_points, dist_values, Vnn, k_nbrs = voronoi_neighbors(
    ...     coords, Y_matrix[0])
    >>> scores, mu_st, is_outlier = GBSO_ST(Y_matrix, Vnn, k=24)
    '''
    T, N = Y_matrix.shape
    scores = np.full((T, N), np.nan)
    mu_st = np.full((T, N), np.nan)

    if verbose:
        print(f"  GBSO_ST | T={T} steps, N={N} observations, "
              f"window k={k} (±{k} steps = {2*k+1} total)")

    for t in range(T):
        t_ini = max(0, t - k)
        t_fin = min(T - 1, t + k)
        raw = np.full(N, np.nan)

        for i in range(N):
            y_i = Y_matrix[t, i]
            if np.isnan(y_i):
                continue
            nbrs = Vnn[i]
            if not nbrs:
                continue
            # All neighbour values within the temporal window
            window_vals = Y_matrix[t_ini:t_fin + 1, :][:, nbrs]
            valid_vals = window_vals[~np.isnan(window_vals)]
            if len(valid_vals) < 3:
                continue
            mu_st[t, i] = np.mean(valid_vals)
            raw[i] = y_i - mu_st[t, i]

        # Standardise across valid scores at this time step
        valid_mask = ~np.isnan(raw)
        if valid_mask.sum() < 3:
            continue
        mu_r = np.mean(raw[valid_mask])
        sigma_r = np.std(raw[valid_mask])
        if sigma_r == 0:
            scores[t, valid_mask] = 0.0
        else:
            scores[t, valid_mask] = np.abs(
                (raw[valid_mask] - mu_r) / sigma_r
            )

    is_outlier = (~np.isnan(scores)) & (scores >= theta)
    return scores, mu_st, is_outlier
