from src.utils.drift_detector_meta import BaseDetector, BaseWindowDetector
import numpy as np
from scipy.stats import norm
from scipy.spatial.distance import cdist, pdist

from src.utils.kernel_two_sample_test import kernel_two_sample_test, MMD2u
from sklearn.metrics import pairwise_distances, pairwise_kernels

from alibi_detect.cd import MMDDrift, ChiSquareDrift, KSDrift, LSDDDrift

import matplotlib.pyplot as plt

# TODO: now only single drift works. Do also for multiple. Modify detect function.


########################################################################################################################
class DetectorZScores(BaseDetector):
    def __init__(self, reference_data, alpha=0.01, counter_th=5, winsize=100, dynamic_update=True, gamma=1.):
        super().__init__(reference_data, winsize, counter_th, dynamic_update)

        self.X_std = np.array([])
        self.X_mean = np.array([])

        self.alpha = alpha
        self.gamma = gamma

        self.z_scores = []
        self.p_values = []

        self.detector_name = 'Z-Score'

        self.compute_statistic()

    def compute_statistic(self):
        self.X_mean = np.mean(self.X_reference)
        self.X_std = np.std(self.X_reference)

    def add_element(self, X):
        pVal = self.get_pval(X)

        drift = pVal < self.alpha

        self.drift_hysteresis(drift)

        if not (self.in_concept_change or self.in_warning_zone) and self.dyn_update:
            # TODO: make X_reference size bounded
            self.X_reference = np.append(self.X_reference, X)
            self.compute_statistic()
        return self

    def get_pval(self, X):
        # Normalize data - Z-score
        z = (X - self.X_mean) / (self.X_std ** self.gamma)
        self.z_scores.append(z)

        # Get p-values from normal distribution
        pVal = norm.sf(abs(z))
        self.p_values.append(pVal)
        return pVal

    def detect(self, data, saver=None, drift_points=()):
        for t, X in enumerate(data):
            self.add_element(X)

            if self.detected_warning_zone():
                self.idx_warning.append(t)
            if self.detected_change():
                self.idx_drift.append(t)


        if saver is not None:
            self.plot_detection(self.p_values, self.alpha, drift_points)
            saver.save_fig(plt.gcf(), f'{self.detector_name}')


########################################################################################################################
class DetectorIRQ(BaseDetector):
    def __init__(self, reference_data, counter_th=5, winsize=100, dynamic_update=True, percentile=(75, 25), gamma=1.5):
        super().__init__(reference_data, winsize, counter_th, dynamic_update)
        self.percentile = percentile
        self.gamma = gamma

        self.q_high = 0
        self.q_low = 0
        self.irq = 0
        self.treshold = 0

        self.compute_statistic()

        self.detector_name = 'IRQ'

    def compute_statistic(self):
        """
        Initialize some statistic for the drift detector
        """
        q_high, q_low = np.percentile(self.X_reference, self.percentile)
        self.q_high = q_high
        self.q_low = q_low
        self.irq = self.q_high - self.q_low
        self.treshold = self.irq * self.gamma
        return self

    def add_element(self, X):
        if X > self.q_high + self.treshold:  # or X < self.q_low - self.treshold:
            drift = True
        else:
            drift = False

        self.drift_hysteresis(drift)

        if not (self.in_concept_change or self.in_warning_zone) and self.dyn_update:
            # TODO: make X_reference size bounded
            self.X_reference = np.append(self.X_reference, X)
            self.compute_statistic()
        return self

    def detect(self, data, saver=None, drift_points=()):
        for t, X in enumerate(data):
            self.add_element(X)

            if self.detected_warning_zone():
                self.idx_warning.append(t)
            if self.detected_change():
                self.idx_drift.append(t)

        if saver is not None:
            self.plot_detection(data, None, drift_points)
            saver.save_fig(plt.gcf(), f'{self.detector_name}')


########################################################################################################################
class DetectorPmodified(BaseDetector):
    def __init__(self, reference_data, counter_th=5, winsize=100, dynamic_update=True, percentile=99, gamma=3,
                 alpha=0.05):
        super().__init__(reference_data, winsize, counter_th, dynamic_update)
        self.percentile = percentile
        self.gamma = gamma
        self.alpha = alpha

        self.critical_distance = 0
        self.treshold = 0
        self.s_score = []

        self.compute_statistic()

        self.detector_name = 'P-modified'

    def compute_statistic(self):
        """
        Initialize some statistic for the drift detector
        """
        self.critical_distance = np.percentile(self.X_reference, self.percentile)
        return self

    def add_element(self, X):
        sVal = self.get_sval(X)

        drift = sVal > self.alpha

        self.drift_hysteresis(drift)

        if not (self.in_concept_change or self.in_warning_zone) and self.dyn_update:
            # TODO: make X_reference size bounded
            self.X_reference = np.append(self.X_reference, X)
            self.compute_statistic()
        return self

    def get_sval(self, X):
        # get S-score. Anomaly score
        s = ((X - self.critical_distance) / (self.critical_distance * self.alpha)) ** self.gamma
        self.s_score.append(s)
        return s

    def detect(self, data, saver=None, drift_points=()):
        for t, X in enumerate(data):
            self.add_element(X)

            if self.detected_warning_zone():
                self.idx_warning.append(t)
            if self.detected_change():
                self.idx_drift.append(t)

        if saver is not None:
            self.plot_detection(self.s_score, self.alpha, drift_points)
            saver.save_fig(plt.gcf(), f'{self.detector_name}')


########################################################################################################################
class DetectorEMA(BaseDetector):
    def __init__(self, reference_data, counter_th=5, winsize=100, dynamic_update=True, gamma=0.5,
                 alpha=1 - 0.001, win_transient=100):
        super().__init__(reference_data, winsize, counter_th, dynamic_update)
        self.gamma = gamma
        self.alpha = alpha

        self.k = 0
        self.M = 0
        self.S = 0
        self.max = 0
        self.min = 999
        self.beta = reference_data.mean() + gamma * reference_data.std()
        self.window_transient = win_transient
        self.ema_M = []
        self.ema_S = []

        self.detector_name = 'EMA'

    def update(self):
        # if self.M > self.max:
        #     self.max = self.M
        # if self.M < self.min:
        #     self.min = self.M
        # avg = (self.max - self.min) / 2
        # self.beta = self.max + avg * self.gamma

        newBeta = self.M + self.gamma * self.S
        self.beta = newBeta

    def add_element(self, X):
        self.k += 1
        newM = self.M * self.alpha + (1 - self.alpha) * X
        newS = self.S * self.alpha + (1 - self.alpha) * (X - newM) ** 2

        self.ema_M.append(newM)
        self.ema_S.append(newS)

        self.M = newM
        self.S = newS

        if self.k > self.window_transient:
            drift = newM > self.beta
            self.drift_hysteresis(drift)

        if not (self.in_concept_change or self.in_warning_zone) and self.dyn_update:
            self.update()



        return self

    def detect(self, data, saver=None, drift_points=()):
        for t, X in enumerate(data):
            self.add_element(X)

            if self.detected_warning_zone():
                self.idx_warning.append(t)
            if self.detected_change():
                self.idx_drift.append(t)

        if saver is not None:
            self.plot_detection(self.ema_M, self.beta, drift_points)
            saver.save_fig(plt.gcf(), f'{self.detector_name}')


########################################################################################################################
class DetectorSimple(BaseDetector):
    def __init__(self, reference_data, counter_th=5, winsize=100, dynamic_update=True, alpha=0.01):
        super().__init__(reference_data, winsize, counter_th, dynamic_update)

        self.alpha = alpha

        self.p_values = []

        self.detector_name = 'Simple'

    def add_element(self, X):
        self.p_values.append(X)

        drift = X < self.alpha
        self.drift_hysteresis(drift)

        return self

    def detect(self, data, saver=None, drift_points=()):
        for t, X in enumerate(data):
            self.add_element(X)

            if self.detected_warning_zone():
                self.idx_warning.append(t)
            if self.detected_change():
                self.idx_drift.append(t)

        if saver is not None:
            self.plot_detection(self.p_values, self.alpha, drift_points)
            saver.save_fig(plt.gcf(), f'{self.detector_name}')


########################################################################################################################
def compute_histogram(X, n_bins):
    return np.array([np.histogram(X[:, i], bins=n_bins, density=False)[0] for i in range(X.shape[1])])


def compute_hellinger_dist(P, Q):
    return np.mean(
        [np.sqrt(np.sum(np.square(np.sqrt(P[i, :] / np.sum(P[i, :])) - np.sqrt(Q[i, :] / np.sum(Q[i, :]))))) for i in
         range(P.shape[0])])


class DetectorEmbedding(BaseDetector):
    def __init__(self, reference_data, winsize, counter_th, dynamic_update, gamma=3, min_winsize=100,
                 metric='Hellinger', name=None):
        super().__init__(reference_data, winsize, counter_th, dynamic_update)

        self.min_winsize = min_winsize
        self.gamma = gamma
        self.metric = metric

        if metric == 'Hellinger':
            self.n_bins = 0
            self.bins = 0
            self.hist_baseline = np.array([])

            self.calculate_reference_hist()
        else:
            raise NotImplementedError

        self.eps = []
        self.beta = []
        self.t_denom = 0
        self.distance = []
        self.z = []
        self.dist_old = 0

        if name is None:
            self.detector_name = metric
        else:
            self.detector_name = name

    def calculate_reference_hist(self):
        self.n_bins = int(np.floor(np.sqrt(len(self.X_reference))))
        self.bins = np.histogram_bin_edges(self.X_reference, bins=self.n_bins)
        self.hist_baseline = compute_histogram(self.X_reference, self.bins)

    def add_element(self, X):
        n = len(X)
        if n < self.min_winsize:
            self.distance.append(0)
            self.z.append(0)
        else:
            if self.metric == 'Hellinger':
                hist = compute_histogram(X, self.bins)
                distance = compute_hellinger_dist(self.hist_baseline, hist)
            else:
                raise NotImplementedError

            self.distance.append(distance)

            eps = distance - self.dist_old
            #self.dist_old = distance
            # self.eps.append(eps)

            d = len(self.eps)
            if d > self.min_winsize:
                epsilon_hat = (1. / d) * np.sum(np.array(np.abs(self.eps)))
                sigma_hat = np.sqrt(np.sum(np.square(np.array(np.abs(self.eps)) - epsilon_hat)) / d)

                beta = epsilon_hat + self.gamma * sigma_hat

                # Test for drift
                drift = np.abs(eps) > beta

                self.drift_hysteresis(drift)
                #if not (self.in_concept_change or self.in_warning_zone):
                #self.hist_baseline += hist
                self.eps.append(eps)
            else:
                self.eps.append(eps)
                #self.X_reference = np.vstack((self.X_reference, X))
                #self.calculate_reference_hist()

    def detect(self, data, saver=None, drift_points=()):
        n = len(data)
        for t in range(n):
            if t < self.min_winsize:
                X = data[:t]
            else:
                X = data[t - self.min_winsize: t]
            self.add_element(X)

            if self.detected_warning_zone():
                self.idx_warning.append(t)
            if self.detected_change():
                self.idx_drift.append(t)

        if saver is not None:
            self.plot_detection(self.distance, None, drift_points)
            saver.save_fig(plt.gcf(), f'{self.detector_name}')


#################################################################################################
class MMDDetector(BaseDetector):
    def __init__(self, reference_data, winsize, counter_th, dynamic_update, min_winsize=100,
                 p_val=0.05, n_permutations=10, name=None):
        super().__init__(reference_data, winsize, counter_th, dynamic_update)

        self.alpha = p_val
        self.cd = MMDDrift(self.X_reference, backend='pytorch', p_val=p_val, n_permutations=n_permutations)
        self.min_winsize = min_winsize

        self.p_vals = []
        if name is not None:
            self.detector_name = name
        else:
            self.detector_name = 'MMD'
    def add_element(self, X):
        n = len(X)
        if n < self.min_winsize:
            self.p_vals.append(0)
        else:
            pred = self.cd.predict(X)

            pVal = pred['data']['p_val']
            drift = pred['data']['is_drift'] == 1

            self.drift_hysteresis(drift)
            self.p_vals.append(pVal)

            return self

    def detect(self, data, saver=None, drift_points=()):
        n = len(data)
        for t in range(n):
            if t < self.min_winsize:
                X = data[:t]
            else:
                X = data[t - self.min_winsize: t]
            self.add_element(X)

            if self.detected_warning_zone():
                self.idx_warning.append(t)
            if self.detected_change():
                self.idx_drift.append(t)
                if t > drift_points[0]:
                    break

        if saver is not None:
            self.plot_detection(self.p_vals, self.alpha, drift_points)
            saver.save_fig(plt.gcf(), f'{self.detector_name}')


#################################################################################################
class LSDDDetector(BaseDetector):
    def __init__(self, reference_data, winsize, counter_th, dynamic_update, min_winsize=100,
                 p_val=0.05, n_permutations=10, name=None):
        super().__init__(reference_data, winsize, counter_th, dynamic_update)

        self.alpha = p_val
        self.cd = LSDDDrift(self.X_reference, backend='pytorch', p_val=p_val, n_permutations=n_permutations)
        self.min_winsize = min_winsize

        self.p_vals = []
        if name is not None:
            self.detector_name = name
        else:
            self.detector_name = 'LSDD'

    def add_element(self, X):
        n = len(X)
        if n < self.min_winsize:
            self.p_vals.append(0)
        else:
            pred = self.cd.predict(X)

            pVal = pred['data']['p_val']
            drift = pred['data']['is_drift'] == 1

            self.drift_hysteresis(drift)
            self.p_vals.append(pVal)

            return self

    def detect(self, data, saver=None, drift_points=()):
        n = len(data)
        for t in range(n):
            if t < self.min_winsize:
                X = data[:t]
            else:
                X = data[t - self.min_winsize: t]
            self.add_element(X)

            if self.detected_warning_zone():
                self.idx_warning.append(t)
            if self.detected_change():
                self.idx_drift.append(t)
                if t > drift_points[0]:
                    break

        if saver is not None:
            self.plot_detection(self.p_vals, self.alpha, drift_points)
            saver.save_fig(plt.gcf(), f'{self.detector_name}')


#################################################################################################
class ChiSquareDetector(BaseDetector):
    def __init__(self, reference_data, winsize, counter_th, dynamic_update, min_winsize=100,
                 p_val=0.05, type='batch', name=None):
        super().__init__(reference_data, winsize, counter_th, dynamic_update)

        self.alpha = p_val
        self.cd = ChiSquareDrift(self.X_reference, p_val=p_val)
        self.min_winsize = min_winsize
        self.drift_type = type

        self.p_vals = []
        if name is not None:
            self.detector_name = name
        else:
            self.detector_name = 'ChiSquare'

    def add_element(self, X):
        n = len(X)
        if n < self.min_winsize:
            self.p_vals.append(0)
        else:
            pred = self.cd.predict(X, drift_type=self.drift_type)

            pVal = pred['data']['p_val'].min()
            drift = pred['data']['is_drift'] >= 1

            self.drift_hysteresis(drift)
            self.p_vals.append(pVal)

            return self

    def detect(self, data, saver=None, drift_points=()):
        n = len(data)
        for t in range(n):
            if t < self.min_winsize:
                X = data[:t]
            else:
                X = data[t - self.min_winsize: t]
            self.add_element(X)

            if self.detected_warning_zone():
                self.idx_warning.append(t)
            if self.detected_change():
                self.idx_drift.append(t)
                if t > drift_points[0]:
                    break

        if saver is not None:
            self.plot_detection(self.p_vals, self.alpha, drift_points)
            saver.save_fig(plt.gcf(), f'{self.detector_name}')


#################################################################################################
class KSDetector(BaseDetector):
    def __init__(self, reference_data, winsize, counter_th, dynamic_update, min_winsize=100,
                 p_val=0.05, type='batch', name=None):
        super().__init__(reference_data, winsize, counter_th, dynamic_update)

        self.alpha = p_val
        self.cd = KSDrift(self.X_reference, p_val=p_val)
        self.min_winsize = min_winsize
        self.drift_type = type


        self.p_vals = []
        if name is not None:
            self.detector_name = name
        else:
            self.detector_name = 'KS'

    def add_element(self, X):
        n = len(X)
        if n < self.min_winsize:
            self.p_vals.append(0)
        else:
            pred = self.cd.predict(X, drift_type=self.drift_type)

            pVal = pred['data']['p_val'].min()
            drift = pred['data']['is_drift'] >= 1

            self.drift_hysteresis(drift)
            self.p_vals.append(pVal)

            return self

    def detect(self, data, saver=None, drift_points=()):
        n = len(data)
        for t in range(n):
            if t < self.min_winsize:
                X = data[:t]
            else:
                X = data[t - self.min_winsize: t]
            self.add_element(X)

            if self.detected_warning_zone():
                self.idx_warning.append(t)
            if self.detected_change():
                self.idx_drift.append(t)
                if t > drift_points[0]:
                    break

        if saver is not None:
            self.plot_detection(self.p_vals, self.alpha, drift_points)
            saver.save_fig(plt.gcf(), f'{self.detector_name}')