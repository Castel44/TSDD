import numpy as np
from scipy.stats import kstest, mannwhitneyu
from scipy.spatial.distance import cdist
from scipy.spatial.distance import cosine
from scipy.stats import entropy
from itertools import accumulate
from scipy.stats import t

import matplotlib.pyplot as plt


class DriftDetector(object):
    def __init__(self, data_test, centroids, data_train=None, stattest: str = 'KS', winsize=100, saver=None,
                 statistic='default', mode='sliding', alpha=0.05):
        """
        : param : correction - sidak or bonferroni. Bonferroni is more conservative
        """
        avalable_stattest = {'KS': kstest}

        if statistic == 'default':
            self.data_statistic = {'mean': np.mean,
                                   'std': np.std,
                                   'min': np.min,
                                   'max': np.max}
        else:
            self.data_statistic = statistic

        self.f_test = avalable_stattest[stattest]
        self.test = data_test
        self.train = data_train
        self.centroids = centroids
        self.winsize = winsize
        self.p_values_dict = dict.fromkeys(self.data_statistic)
        self.mode = mode
        self.saver = saver
        self.alpha = alpha

        self.distances_train = None
        self.distances_test = None
        self.statistics_train = None
        self.statistics_test = None

    @staticmethod
    def sliding_reference(data, f_test, winsize=100, alpha=0.05, tolerance=5):
        drift_ct = 0
        normal_ct = 0
        p_values = np.ones(len(data))
        for t in range(len(data)):
            if t < 2 * winsize:
                continue
            data_test = data[t - winsize:t]
            data_reference = data[t - 2 * winsize: t - winsize]
            p = f_test(data_reference, data_test)[1]
            p_values[t] = p

            # if p <= alpha:
            #    normal_ct = 0
            #    drift_ct += 1
            #    if drift_ct < tolerance:
            #        print(f'Warning in {t=}')
            #    else:
            #        print(f'Drift detect at {t=}')
            # else:
            #    normal_ct += 1
            #    if normal_ct > tolerance:
            #        drift_ct = 0

        return p_values

    @staticmethod
    def fixed_reference(data, f_test, data_reference=None, winsize=100, alpha=0.05, tolerance=5):
        if data_reference is None:
            data_reference = data[:winsize]

        drift_ct = 0
        normal_ct = 0
        p_values = np.ones(len(data))
        for t in range(len(data)):
            if t < winsize:
                continue
            data_test = data[t - winsize:t]
            p = f_test(data_reference, data_test)[1]
            p_values[t] = p

            # if p <= alpha:
            #    normal_ct = 0
            #    drift_ct += 1
            #    if drift_ct < tolerance:
            #        print(f'Warning in {t=}')
            #    else:
            #        print(f'Drift detect at {t=}')
            # else:
            #    normal_ct += 1
            #    if normal_ct > tolerance:
            #        drift_ct = 0

        return p_values

    def _cdist(self, data):
        pairwise_distance = cdist(data, self.centroids)
        return pairwise_distance

    def _compute_statistic(self, data):
        stat_dict = {key: func(data, axis=1) for key, func in self.data_statistic.items()}
        return stat_dict

    def detect(self):
        if self.distances_test is None:
            self.distances_test = self._cdist(self.test)

        self.statistics_test = self._compute_statistic(self.distances_test)

        for key, value in self.statistics_test.items():
            print(f'*** Statistic: {key} ***')
            if self.mode == 'fixed':
                p_val = self.fixed_reference(value, self.f_test, winsize=self.winsize)
            elif self.mode == 'sliding':
                p_val = self.sliding_reference(value, self.f_test, winsize=self.winsize)
            else:
                raise NotImplementedError
            self.p_values_dict[key] = p_val
        return self

    def correct_alpha(self, correction='sidak'):
        if self.p_values_dict.get(list(self.p_values_dict.keys())[0]) is None:
            self.detect()

        m = len(self.p_values_dict.keys())
        if correction == 'bonferroni':
            alpha_ = self.alpha / m
        elif correction == 'sidak':
            alpha_ = 1 - (1 - self.alpha) ** (1 / m)
        alpha_corrected = 1 - (1 - alpha_) ** m
        return alpha_corrected

    def plot_train_test_statistic(self, drift_points=()):
        if self.distances_test is None:
            self.distances_test = self._cdist(self.test)
        if self.distances_train is None:
            self.distances_train = self._cdist(self.train)
        if self.statistics_train is None:
            self.statistics_train = self._compute_statistic(self.distances_train)
        if self.statistics_test is None:
            self.statistics_test = self._compute_statistic(self.distances_test)

        plt.figure()
        for key, value in self.statistics_train.items():
            train_samples = value.shape[0]
            plt.plot([x for x in range(train_samples)], value, label='Train {} distance'.format(key))
        for key, value in self.statistics_test.items():
            plt.plot([x for x in range(train_samples, train_samples + value.shape[0])], value,
                     label='Test {} distance'.format(key))
            for t in drift_points:
                plt.axvline(train_samples+t, linestyle='-.', c='blue')
        plt.legend()
        self.saver.save_fig(plt.gcf(), 'Drift_statistics')

    def plot_drift_detection(self, drift_points=()):
        fig, axes = plt.subplots(len(self.p_values_dict.keys()), 1, sharex='all')
        for ax, key, value in zip(axes, self.p_values_dict.keys(), self.p_values_dict.values()):
            n = len(self.p_values_dict[key])
            ax.plot([x for x in range(n)], self.p_values_dict[key])
            ax.axhline(self.alpha, linestyle='--', c='red')
            ax.set_title(f'Statistic: {key} ({self.winsize=})')
            for t in drift_points:
                ax.axvline(t, linestyle='-.', c='blue')

            # Visualize drift points
            drift = np.array(value < 0.05)
            ax.plot(np.argwhere(drift), value[drift], linestyle='None', marker='o', alpha=0.33)

            acc = np.array(list(accumulate(drift.astype(int), lambda x, y: x + y if y else 0)))
            ax.plot(np.argwhere(acc > 3), value[acc > 3], linestyle='None', marker='x', color='red', alpha=0.5)

        fig.tight_layout()
        self.saver.save_fig(plt.gcf(), 'drift_detection')

        p_tot = np.array(list(self.p_values_dict.values())).mean(axis=0)
        alpha_corrected = self.correct_alpha(correction='sidak')
        n = len(p_tot)
        plt.figure()
        plt.plot([x for x in range(n)], p_tot)
        plt.axhline(alpha_corrected, linestyle='--', c='red')
        for t in drift_points:
            plt.axvline(t, linestyle='-.', c='blue')

        # Visualize drift points
        drift = np.array(p_tot < alpha_corrected)
        plt.plot(np.argwhere(drift), p_tot[drift], linestyle='None', marker='o', alpha=0.33)

        acc = np.array(list(accumulate(drift.astype(int), lambda x, y: x + y if y else 0)))
        plt.plot(np.argwhere(acc > 3), p_tot[acc > 3], linestyle='None', marker='x', color='red', alpha=0.5)

        plt.title(f'p-values corrected ({self.winsize=})')
        plt.tight_layout()
        self.saver.save_fig(plt.gcf(), 'drift_detection_corrected')

    def get_corrected(self):
        pVal = np.array(list(self.p_values_dict.values())).mean(axis=0)
        alpha_corrected = self.correct_alpha(correction='sidak')
        return pVal, alpha_corrected


class DriftDetectionUnconstrained():
    def __init__(self, reference_data, test_data, winsize=100, alpha=0.05, saver=None, f_dist='cosine', mode='sliding'):
        f_d = {'cosine': cosine,
               'kld': entropy}

        self.test = test_data
        self.reference = reference_data
        self.winsize = winsize
        self.alpha = alpha
        self.saver = saver
        self.mode = mode
        self.f_d = f_d[f_dist]

        self.p_values = None
        self.D = None

    @staticmethod
    def _get_average(x):
        x_avg = np.mean(x, axis=0)
        return x_avg

    def detect(self):
        # TODO: implement KLD
        f_a = self._get_average(self.reference)
        n = len(self.test)
        w = self.winsize

        D = np.empty(n)
        p_values = np.ones(n)
        for t in range(n):
            # Calculate divergence from reference
            if t < w:
                continue
            x = self.test[t - w:t]
            f_b = self._get_average(x)

            if self.f_d == entropy:
                f_a = f_a / f_a.sum()
                f_b = f_b / f_b.sum()

            d = self.f_d(f_a, f_b)
            D[t] = d

            # Detect Drift from Divergence stream
            if self.mode == 'sliding':
                if t < 2 * w:
                    continue
                data_test = D[t - w:t]
                data_reference = D[t - 2 * w: t - w]
            elif self.mode == 'fixed':
                data_test = D[t - w:t]
                data_reference = D[:w]

            p = kstest(data_reference, data_test)[1]
            p_values[t] = p

        self.D = D
        self.p_values = p_values
        return self

    def plot_drift_detection(self, drift_points=()):
        n = len(self.p_values)
        plt.figure()
        plt.plot([x for x in range(n)], self.p_values)
        plt.axhline(self.alpha, linestyle='--', c='red')
        for t in drift_points:
            plt.axvline(t, linestyle='-.', c='blue')
        plt.title(f'p-values ({self.winsize=})')
        plt.tight_layout()
        self.saver.save_fig(plt.gcf(), 'drift_detection_unconstrained')

        plt.figure()
        plt.plot(self.D)
        plt.title('Divergences')


def compute_histogram(X, n_bins):
    return np.array([np.histogram(X[:, i], bins=n_bins, density=False)[0] for i in range(X.shape[1])])


def compute_hellinger_dist(P, Q):
    return np.mean(
        [np.sqrt(np.sum(np.square(np.sqrt(P[i, :] / np.sum(P[i, :])) - np.sqrt(Q[i, :] / np.sum(Q[i, :]))))) for i in
         range(P.shape[0])])


class HDDDM():
    def __init__(self, X, gamma=1., alpha=0.1, winsize=100, min_statistic=100):

        self.gamma = gamma
        self.alpha = alpha
        self.X_baseline = X
        self.n_bins = int(np.floor(np.sqrt(winsize)))

        self.bins = np.histogram_bin_edges(X, bins=self.n_bins)
        self.hist_baseline = compute_histogram(X, self.bins)

        self.min_statistic = min_statistic
        self.n_samples = len(X)
        self.dist_old = 0
        self.eps = []
        self.beta = []
        self.t_denom = 0
        self.drift_detected = False
        self.winsize = winsize
        self.dist = []
        self.z = []

    def add_batch(self, X):
        currLenght = len(X)
        if currLenght < self.winsize:
            self.dist.append(0)
            self.z.append(0)
        else:
            # Get metric to monitor
            hist = compute_histogram(X, self.bins)
            dist = compute_hellinger_dist(self.hist_baseline, hist)
            self.dist.append(dist)

            # Compute test statistic
            eps = dist - self.dist_old
            #self.dist_old = dist
            #self.eps.append(eps)

            d = len(self.eps)

            self.drift_detected = False
            if d >= 100:
                epsilon_hat = (1. / d) * np.sum(np.array(np.abs(self.eps)))
                sigma_hat = np.sqrt(np.sum(np.square(np.array(np.abs(self.eps)) - epsilon_hat)) /d)

                beta = epsilon_hat + self.gamma * sigma_hat
                #self.beta.append(beta)

                # Test for drift
                drift = np.abs(eps) > beta

                z = np.abs(eps - epsilon_hat) / sigma_hat
                self.z.append(z)

                if drift == True:
                    self.drift_detected = True

                    #self.eps.pop(-1)

                    #self.t_denom = 0
                    #self.eps = []
                    # self.n_bins = int(np.floor(np.sqrt(n_samples)))
                    # self.bins = np.histogram_bin_edges(X, bins=self.n_bins)
                    # self.hist_baseline = compute_histogram(X, self.bins)
                    # self.hist_baseline = hist
                    # self.n_samples = n_samples
                    # self.X_baseline = X
                else:
                    self.hist_baseline += hist
                    self.n_samples += currLenght
                    self.X_baseline = np.vstack((self.X_baseline, X))
                    self.eps.append(eps)
            else:
                self.eps.append(eps)
                self.z.append(0)
        return self

    def detected_change(self):
        return self.drift_detected

    def detect(self, data):
        for t in range(0, len(data)):
            if t < self.winsize:
                X = data[:t]
            else:
                X = data[t - self.winsize: t]

            self.add_batch(X)
            if self.drift_detected:
                print(f'Drift Detected {t=}')

        return self
