"""Utility to create detection-error-tradeoff curves."""

import numpy as np


def get_fpr_fnr(scores, truth, n_steps: int = 1000, hist_bins=None):
    """Plot a single detection/error tradeoff curve."""
    hist_bins = hist_bins or get_hist_bins(scores, n_steps)
    pos_pmf = _get_pmf(scores, truth, True, hist_bins)
    neg_pmf = _get_pmf(scores, truth, False, hist_bins)

    fpr = 1.0 - np.cumsum(neg_pmf)
    fnr = np.cumsum(pos_pmf)
    return fpr, fnr


def get_hist_bins(scores, n_steps):
    return np.linspace(min(scores), max(scores), num=n_steps + 1, endpoint=True)


def _get_pmf(scores, truth, _class, hist_bins):
    idxs = truth == _class
    hist, _ = np.histogram(scores[idxs], hist_bins)
    pmf = hist / np.sum(idxs)
    return pmf
