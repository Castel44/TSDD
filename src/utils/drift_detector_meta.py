from collections import deque
import numpy as np
from scipy.stats import kstest, mannwhitneyu

import matplotlib.pyplot as plt


class BaseDetector:
    def __init__(self, reference_data, winsize, counter_th, dynamic_update):
        super().__init__()
        self.in_concept_change = False
        self.in_warning_zone = False

        self.X_reference = reference_data
        self.counter_th = counter_th
        self.winsize = winsize
        self.dyn_update = dynamic_update

        self.lfsr = deque([], self.winsize)
        self.idx_drift = []
        self.idx_warning = []

        self.detector_name = "name"

    def get_idxs(self):
        return self.idx_drift, self.idx_warning

    def drift_hysteresis(self, drift):
        lfsr = self.lfsr
        if drift:
            lfsr.append(1)
        else:
            lfsr.append(0)

        if np.sum(lfsr) > 0:
            self.in_warning_zone = True
            if np.sum(lfsr) >= self.counter_th:
                self.in_concept_change = True
                self.in_warning_zone = False
        else:
            self.in_warning_zone = False
            self.in_concept_change = False

    def reset(self):
        """ reset
        Resets the change detector parameters.
        """
        self.in_concept_change = False
        self.in_warning_zone = False

    def detected_change(self):
        return self.in_concept_change

    def detected_warning_zone(self):
        return self.in_warning_zone

    def add_element(self, input_data):
        raise NotImplementedError

    def plot_detection(self, data, threshold=None, drift_points=()):
        # TODO: return figure handler
        X = np.array(data)
        plt.figure()
        plt.plot(X, label='Test Data')
        if threshold is not None:
            plt.axhline(threshold, linestyle='--', c='red', label='Alpha')
        for t in drift_points:
            plt.axvline(t, linestyle='-.', c='blue', label='Induced drift points')

        plt.plot(self.idx_warning, X[self.idx_warning], linestyle='None', marker='o', color='orange', alpha=0.33,
                 label='Warning points')
        plt.plot(self.idx_drift, X[self.idx_drift], linestyle='None', marker='x', color='red', alpha=0.5,
                 label='Drift points')
        plt.legend()
        plt.title(f'p-values (Detector Name: {self.detector_name})')
        plt.tight_layout()
