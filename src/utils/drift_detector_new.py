import numpy as np
from scipy.spatial.distance import cdist, pdist
from scipy.special import ndtr
from scipy.stats import norm

import matplotlib.pyplot as plt


def compute_histogram(X, n_bins):
    return np.array([np.histogram(X[:, i], bins=n_bins, density=False)[0] for i in range(X.shape[1])])


class DriftDetectorNew(object):
    def __init__(self, C, reference_data=None, winsize=None, alpha=0.05, saver=None):
        # TODO: adapt for not fixed reference data

        self.centroids = C

        self.X_reference = reference_data
        self.winsize = winsize
        self.alpha = alpha
        self.saver = saver

        self.n_bins = int(np.floor(np.sqrt(len(reference_data))))
        self.bin_edges = np.histogram_bin_edges(reference_data, bins=self.n_bins)
        self.hist = np.histogram(reference_data, bins=self.bin_edges, density=False)[0]
        self.N = len(reference_data)

        #self.d_th = np.max(pdist(C))
        #self.d_th2 = self.bin_edges[-1]

        self.p_values = []
        self.detect_change = False

    def add_element(self, X):
        bin_edges = self.bin_edges
        # Values which are close to a bin edge are susceptible to numeric
        # instability. Add eps to X so these values are binned correctly
        # with respect to their decimal truncation. See documentation of
        # numpy.isclose for an explanation of ``rtol`` and ``atol``.
        rtol = 1.e-5
        atol = 1.e-8
        eps = atol + rtol * np.abs(X)
        Xt = np.digitize(X + eps, bin_edges)
        Xt = np.clip(Xt, 0, self.n_bins - 1)

        p_val = np.sum(self.hist[Xt:]) / self.N

        # gamma = 1 - np.maximum(0, (X - self.d_th2)/X)
        # gamma = (self.d_th2 - X)**2/self.d_th2/X
        # p_val *= gamma

        self.p_values.append(p_val)

        # if p_val > self.alpha:
        #    self.N += 1
        #    self.hist[Xt] += 1
        return self

    def detected_change(self):
        if self.p_values[-1] < self.alpha:
            return True
        else:
            return False

    def plot_drift_detection(self, drift_points=()):
        n = len(self.p_values)
        plt.figure()
        plt.plot([x for x in range(n)], self.p_values)
        plt.axhline(self.alpha, linestyle='--', c='red')
        for t in drift_points:
            plt.axvline(t, linestyle='-.', c='blue')
        plt.title(f'p-values ({self.winsize=})')
        plt.tight_layout()
        self.saver.save_fig(plt.gcf(), 'drift_detection')


class DriftDetectorZ(DriftDetectorNew):
    def __init__(self, C, X_reference):
        super(DriftDetectorZ, self).__init__(C, X_reference)

        self.Xmin = np.min(X_reference)
        self.Xmax = np.max(X_reference)
        self.Xmean = np.mean(X_reference)
        self.Xstd = np.std(X_reference)
        self.N = len(X_reference)

        self.z_scores = []

        self.in_concept_change = False

    def add_element(self, X):
        self.in_concept_change = False

        z_scores = ((X - self.Xmean) / self.Xstd)
        self.z_scores.append(z_scores)

        pVal = norm.sf(abs(z_scores))
        self.p_values.append(pVal)

        if pVal < self.alpha:
            self.in_concept_change = True

        return self

    def detected_change(self):
        return self.in_concept_change
