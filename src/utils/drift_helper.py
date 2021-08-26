import os
import shutil

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from skmultiflow.drift_detection import PageHinkley, KSWIN, EDDM, DDM
from skmultiflow.drift_detection.adwin import ADWIN

from src.utils.drift_detectors import DetectorZScores, DetectorIRQ, DetectorPmodified, DetectorEmbedding, DetectorEMA, \
    DetectorSimple, ChiSquareDetector, KSDetector, MMDDetector, LSDDDetector, DriftDetector
from src.utils.evaluate_drift import evaluate_single_drift

columns = shutil.get_terminal_size().columns


class ResultsHandler():
    def __init__(self, true_drift_idx):
        self.true_drift = true_drift_idx

        self.drift_idxs = {}
        self.warning_idxs = {}
        self.result_df = pd.DataFrame()

        self.name = None

    def update(self, name, seed, counter, detection_w, stat_w, idx_drift, idx_warning):
        self.name = name
        result_dict = self.create_result_dict(name, seed, counter, detection_w, stat_w)

        # Take only the first of consecutive drift detections.
        # idx_drift = [y[0] for y in list(group(idx_drift))]
        # idx_warning = [y[0] for y in list(group(idx_warning))]

        self.update_idxs(idx_drift, idx_warning)

        drift_scores = evaluate_single_drift(self.true_drift, idx_drift, prefix='drift')
        warning_scores = evaluate_single_drift(self.true_drift, idx_warning, prefix='warning')

        result_dict.update(drift_scores)
        result_dict.update(warning_scores)
        self.result_df = self.result_df.append(result_dict, ignore_index=True)

    def update_idxs(self, idx_drift, idx_warning):
        self.drift_idxs[self.name] = idx_drift
        self.warning_idxs[self.name] = idx_warning

    @staticmethod
    def create_result_dict(name, seed, counter, detection_w, stat_w):
        return {'name': name,
                'seed': seed,
                'counter': counter,
                'detection_winsize': detection_w,
                'stat_winsize': stat_w}

    def get_result_df(self):
        return self.result_df

    def get_idxs(self):
        return self.drift_idxs, self.warning_idxs


def skflow_detector_wrapper(detector, data, **kwargs):
    dd = detector(**kwargs)
    n = len(data)
    idx_drift = []
    idx_warning = []
    for i in range(n):
        dd.add_element(data[i])
        if dd.detected_change():
            idx_drift.append(i)
        if dd.detected_warning_zone():
            idx_warning.append(i)
    return idx_drift, idx_warning


def plot_detectors(results, data, drifts, winsize, counter, saver):
    drift_idxs, warning_idxs = results.get_idxs()
    df_results = results.get_result_df()
    n = len(drift_idxs.keys())
    fig, axes = plt.subplots(nrows=np.ceil(n / 5).astype(int), ncols=5, sharex='all', sharey='all',
                             figsize=(19.2, 10.8))
    axes = axes.ravel()
    for ax, (k, v), (kw, vw) in zip(axes, drift_idxs.items(), warning_idxs.items()):
        delay = int(df_results.loc[df_results.name == k]['drift_delay'].item())
        FP = int(df_results.loc[df_results.name == k]['drift_false_positive'].item())
        ax.plot(data)
        if len(vw) != 0:
            ax.plot(vw, data[vw], 'o', c='orange', alpha=.75, label='Warning')
        if len(v) != 0:
            ax.plot(v, data[v], 'x', c='red', alpha=.95, label='Drift')
        for drift_ in drifts:
            ax.axvline(drift_, linestyle='-.', c='blue', label='Drift')
        ax.set_title(f'{k} - Delay: {delay} FP: {FP}')
        ax.set_ylabel('Rolling Accuracy')
        ax.set_xlabel('Sample')
        ax.legend()

    fig.suptitle(f"winsize: {winsize} - counter: {counter}")
    fig.tight_layout()
    saver.save_fig(fig, 'Sota_drift_results')
    # plt.show(block=True)


def detection_helper(x_valid, x_test, valid_embedding, test_embedding, y_binary, running_accuracy, cluster_centers,
                     saver, detection_winsize, counter_percent, t_start, t_end, args, detectors_list=None,
                     stat_multiplier=5):
    dyn_update = args.dyn_update
    stat_winsize = detection_winsize * stat_multiplier
    counter_th = max(int(detection_winsize * 0.01 * counter_percent), 1)

    print('*' * shutil.get_terminal_size().columns)
    print(
        'Starting drift detection with: winsize={}, counter:{}, multiplier:{}'.format(
            detection_winsize, counter_th, stat_multiplier).center(columns))
    print('*' * shutil.get_terminal_size().columns)

    results_obj = ResultsHandler(t_start)
    available_detectors = (
        'ZScore', 'IRQ', 'P_modified', 'EMA', 'HDDDM_Emb', 'HDDDM_Input', 'IKS_Input', 'IKS_emb_raw', 'IKS_emb',
        'KSWIN_Emb', 'PH_Emb', 'PH_error', 'DDM', 'EDDM', 'ADWIN', 'MMD_Input', 'MMD_Emb', 'Chi_Input',
        'Chi_Emb', 'LSD_Input', 'LSD_Emb', 'KS_Input', 'KS_Emb')
    if detectors_list is None:
        detectors_list = available_detectors
    alpha = 0.05

    # Valid data used as reference
    d_ref = cdist(valid_embedding, cluster_centers)
    d_ref_min = np.min(d_ref, axis=1)

    d_test = cdist(test_embedding, cluster_centers)
    d_test_min = np.min(d_test, axis=1)

    drift_detectors = {
        'ZScore': DetectorZScores(d_ref_min, alpha=alpha, counter_th=counter_th, winsize=detection_winsize,
                                  dynamic_update=dyn_update, gamma=0.5),
        'IRQ': DetectorIRQ(d_ref_min, counter_th=counter_th, winsize=detection_winsize,
                           dynamic_update=dyn_update,
                           percentile=(95, 5), gamma=1.5),
        'P_modified': DetectorPmodified(d_ref_min, counter_th=counter_th, winsize=detection_winsize,
                                        dynamic_update=dyn_update, percentile=99, gamma=3., alpha=1.),
        'EMA': DetectorEMA(d_ref_min, counter_th=counter_th, winsize=detection_winsize,
                           dynamic_update=dyn_update, gamma=1., alpha=1 - 0.05, win_transient=stat_winsize)
    }

    for name, dd in drift_detectors.items():
        if name in detectors_list:
            print("Detector: ", name)
            dd.detect(d_test_min, saver=saver, drift_points=(t_start, t_end))
            idx_drift, idx_warning = dd.get_idxs()
            results_obj.update(name, args.init_seed, counter_percent, detection_winsize, stat_winsize, idx_drift,
                               idx_warning)

    name = 'HDDDM_Emb'
    if name in detectors_list:
        print("Detector: ", name)
        dd = DetectorEmbedding(valid_embedding, winsize=detection_winsize, counter_th=counter_th,
                               dynamic_update=dyn_update, gamma=2., min_winsize=stat_winsize, metric='Hellinger',
                               name='Hellinger distance embedding')
        dd.detect(test_embedding, saver=saver, drift_points=(t_start, t_end))
        idx_drift, idx_warning = dd.get_idxs()
        results_obj.update(name, args.init_seed, counter_percent, detection_winsize, stat_winsize, idx_drift,
                           idx_warning)

    name = 'HDDDM_Input'
    if name in detectors_list:
        print("Detector: ", name)
        dd = DetectorEmbedding(x_valid, winsize=detection_winsize, counter_th=counter_th,
                               dynamic_update=dyn_update, gamma=2., min_winsize=stat_winsize, metric='Hellinger',
                               name='Hellinger distance input')
        dd.detect(x_test, saver=saver, drift_points=(t_start, t_end))
        idx_drift, idx_warning = dd.get_idxs()
        results_obj.update(name, args.init_seed, counter_percent, detection_winsize, stat_winsize, idx_drift,
                           idx_warning)

    ## IKS on embedding extracted features
    name = 'IKS_emb'
    if name in detectors_list:
        print("Detector: ", name)
        detector = DriftDetector(data_test=test_embedding, centroids=cluster_centers, data_train=valid_embedding,
                                 winsize=stat_winsize, saver=saver, mode='fixed', alpha=alpha)
        detector.detect()
        pVal, _ = detector.get_corrected()
        detector.plot_drift_detection(drift_points=(t_start, t_end))
        detector.plot_train_test_statistic(drift_points=(t_start, t_end))

        dd = DetectorSimple(None, counter_th=counter_th, winsize=detection_winsize, dynamic_update=dyn_update,
                            alpha=alpha)
        dd.detect(pVal, saver=saver, drift_points=(t_start, t_end))
        idx_drift, idx_warning = dd.get_idxs()
        results_obj.update(name, args.init_seed, counter_percent, detection_winsize, stat_winsize, idx_drift,
                           idx_warning)

    ##skmultiflow
    detector = 'KSWIN on emebdding min distances'
    name = 'KSWIN_Emb'
    if name in detectors_list:
        print(f'Detector: {detector}')
        idx_drift, idx_warning = skflow_detector_wrapper(KSWIN, d_test_min, alpha=0.0001, window_size=stat_winsize,
                                                         stat_size=stat_winsize // 2)
        results_obj.update(name, args.init_seed, counter_percent, detection_winsize, stat_winsize, idx_drift,
                           idx_warning)

    detector = 'IKS_Input'
    name = detector
    if name in detectors_list:
        print(f'Detector: {detector}')
        features = x_test.shape[1]
        idxs_d = []
        idxs_w = []

        for i in range(features):
            id_d, id_w = skflow_detector_wrapper(KSWIN, x_test[:, i], alpha=0.01 / features, window_size=stat_winsize,
                                                 stat_size=stat_winsize // 2)
            idxs_d += id_d
            idxs_w += id_w
        idx_drift, counts = np.unique(idxs_d, return_counts=True)
        idx_warning, counts = np.unique(idxs_w, return_counts=True)
        results_obj.update(name, args.init_seed, counter_percent, detection_winsize, stat_winsize, idx_drift,
                           idx_warning)

    detector = 'IKS_emb_raw'
    name = detector
    if name in detectors_list:
        print(f'Detector: {detector}')
        features = d_test.shape[1]
        idxs_d = []
        idxs_w = []
        for i in range(features):
            id_d, id_w = skflow_detector_wrapper(KSWIN, d_test[:, i], alpha=alpha / features, window_size=stat_winsize,
                                                 stat_size=stat_winsize // 2)
            idxs_d += id_d
            idxs_w += id_w
        idx_drift, counts = np.unique(idxs_d, return_counts=True)
        idx_warning, counts = np.unique(idxs_w, return_counts=True)
        results_obj.update(name, args.init_seed, counter_percent, detection_winsize, stat_winsize, idx_drift,
                           idx_warning)

    # detector = 'KSWIN on error signal'
    # name = 'KS_error'
    # print(f'Detector: {detector}')
    # idx_drift, idx_warning = skflow_detector_wrapper(KSWIN, y_binary, alpha=0.00005, window_size=stat_winsize,
    #                                                 stat_size=stat_winsize // 2)
    # results_obj.update(name, args.init_seed, counter_th, detection_winsize, stat_winsize, idx_drift, idx_warning)

    detector = 'PageHinkley on emebdding min distances'
    name = 'PH_Emb'
    if name in detectors_list:
        print(f'Detector: {detector}')
        idx_drift, idx_warning = skflow_detector_wrapper(PageHinkley, d_test_min, min_instances=stat_winsize,
                                                         delta=0.005, threshold=2,
                                                         alpha=1 - 0.0001)
        results_obj.update(name, args.init_seed, counter_percent, detection_winsize, stat_winsize, idx_drift,
                           idx_warning)

    detector = 'PageHinkley on error signal'
    name = 'PH_error'
    if name in detectors_list:
        print(f'Detector: {detector}')
        idx_drift, idx_warning = skflow_detector_wrapper(PageHinkley, y_binary, min_instances=stat_winsize,
                                                         delta=0.005,
                                                         threshold=counter_th, alpha=1 - 0.0001)
        results_obj.update(name, args.init_seed, counter_percent, detection_winsize, stat_winsize, idx_drift,
                           idx_warning)

    detector = 'DDM'
    name = detector
    if name in detectors_list:
        print(f'Detector: {detector}')
        idx_drift, idx_warning = skflow_detector_wrapper(DDM, y_binary)
        results_obj.update(name, args.init_seed, counter_percent, detection_winsize, stat_winsize, idx_drift,
                           idx_warning)

    detector = 'EDDM'
    name = detector
    if name in detectors_list:
        print(f'Detector: {detector}')
        idx_drift, idx_warning = skflow_detector_wrapper(EDDM, y_binary)
        results_obj.update(name, args.init_seed, counter_percent, detection_winsize, stat_winsize, idx_drift,
                           idx_warning)

    detector = 'ADWIN'
    name = detector
    if name in detectors_list:
        print(f'Detector: {detector}')
        idx_drift, idx_warning = skflow_detector_wrapper(ADWIN, y_binary, delta=2.)
        results_obj.update(name, args.init_seed, counter_percent, detection_winsize, stat_winsize, idx_drift,
                           idx_warning)

    # Alibi stuff
    detector = 'MMD on input'
    name = 'MMD_Input'
    if name in detectors_list:
        print(f'Detector: {detector}')
        dd = MMDDetector(x_valid, counter_th=counter_th, winsize=detection_winsize, dynamic_update=dyn_update,
                         p_val=alpha, min_winsize=stat_winsize, n_permutations=20, name=name)
        dd.detect(x_test, saver=saver, drift_points=(t_start, t_end))
        idx_drift, idx_warning = dd.get_idxs()
        results_obj.update(name, args.init_seed, counter_percent, detection_winsize, stat_winsize, idx_drift,
                           idx_warning)

    detector = 'MMD on embedding'
    name = 'MMD_Emb'
    if name in detectors_list:
        print(f'Detector: {detector}')
        dd = MMDDetector(d_ref, counter_th=counter_th, winsize=detection_winsize, dynamic_update=dyn_update,
                         p_val=alpha, min_winsize=stat_winsize, n_permutations=20, name=name)
        dd.detect(d_test, saver=saver, drift_points=(t_start, t_end))
        idx_drift, idx_warning = dd.get_idxs()
        results_obj.update(name, args.init_seed, counter_percent, detection_winsize, stat_winsize, idx_drift,
                           idx_warning)

    detector = 'LSD on input'
    name = 'LSD_Input'
    if name in detectors_list:
        print(f'Detector: {detector}')
        dd = LSDDDetector(x_valid, counter_th=counter_th, winsize=detection_winsize, dynamic_update=dyn_update,
                          p_val=alpha, min_winsize=stat_winsize, n_permutations=20, name=name)
        dd.detect(x_test, saver=saver, drift_points=(t_start, t_end))
        idx_drift, idx_warning = dd.get_idxs()
        results_obj.update(name, args.init_seed, counter_percent, detection_winsize, stat_winsize, idx_drift,
                           idx_warning)

    detector = 'LSD on embedding'
    name = 'LSD_Emb'
    if name in detectors_list:
        print(f'Detector: {detector}')
        dd = LSDDDetector(d_ref, counter_th=counter_th, winsize=detection_winsize, dynamic_update=dyn_update,
                          p_val=alpha, min_winsize=stat_winsize, n_permutations=20, name=name)
        dd.detect(d_test, saver=saver, drift_points=(t_start, t_end))
        idx_drift, idx_warning = dd.get_idxs()
        results_obj.update(name, args.init_seed, counter_percent, detection_winsize, stat_winsize, idx_drift,
                           idx_warning)

    detector = 'ChiSquare on Input'
    name = 'Chi_Input'
    if name in detectors_list:
        print(f'Detector: {detector}')
        dd = ChiSquareDetector(x_valid, counter_th=counter_th, winsize=detection_winsize, dynamic_update=dyn_update,
                               p_val=alpha, min_winsize=stat_winsize, name=name)
        dd.detect(x_test, saver=saver, drift_points=(t_start, t_end))
        idx_drift, idx_warning = dd.get_idxs()
        results_obj.update(name, args.init_seed, counter_percent, detection_winsize, stat_winsize, idx_drift,
                           idx_warning)

    detector = 'ChiSquare on Embedding'
    name = 'Chi_Emb'
    if name in detectors_list:
        print(f'Detector: {detector}')
        dd = ChiSquareDetector(d_ref, counter_th=counter_th, winsize=detection_winsize, dynamic_update=dyn_update,
                               p_val=alpha, min_winsize=stat_winsize, name=name)
        dd.detect(d_test, saver=saver, drift_points=(t_start, t_end))
        idx_drift, idx_warning = dd.get_idxs()
        results_obj.update(name, args.init_seed, counter_percent, detection_winsize, stat_winsize, idx_drift,
                           idx_warning)

    detector = 'KS on Input'
    name = 'KS_Input'
    if name in detectors_list:
        print(f'Detector: {detector}')
        dd = KSDetector(x_valid, counter_th=counter_th, winsize=detection_winsize, dynamic_update=dyn_update,
                        p_val=alpha, min_winsize=stat_winsize, name=name)
        dd.detect(x_test, saver=saver, drift_points=(t_start, t_end))
        idx_drift, idx_warning = dd.get_idxs()
        results_obj.update(name, args.init_seed, counter_percent, detection_winsize, stat_winsize, idx_drift,
                           idx_warning)

    detector = 'KS on Embedding'
    name = 'KS_Emb'
    if name in detectors_list:
        print(f'Detector: {detector}')
        dd = KSDetector(d_ref, counter_th=counter_th, winsize=detection_winsize, dynamic_update=dyn_update,
                        p_val=alpha, min_winsize=stat_winsize, name=name)
        dd.detect(d_test, saver=saver, drift_points=(t_start, t_end))
        idx_drift, idx_warning = dd.get_idxs()
        results_obj.update(name, args.init_seed, counter_percent, detection_winsize, stat_winsize, idx_drift,
                           idx_warning)

    ######
    drift_idxs, warning_idxs = results_obj.get_idxs()
    df_results = results_obj.get_result_df()

    plot_detectors(results_obj, running_accuracy, (t_start, t_end), detection_winsize, counter_th, saver)

    return df_results, drift_idxs, warning_idxs, results_obj


def postprocess_df(path_list, features, alpha=2, beta=1, gamma=10):
    def load_csv(path):
        features = path.split(sep='/')[-4]
        try:
            dataset = path.split(sep='/')[-5]
            return pd.read_csv(os.path.join(path, '{}_{}_results.csv'.format(dataset, features)))
        except FileNotFoundError:
            try:
                dataset = path.split(sep='/')[-6]
                return pd.read_csv(os.path.join(path, '{}_{}_results.csv'.format(dataset, features)))
            except:
                dataset = path.split(sep='/')[-5]
                return pd.read_csv(os.path.join(path, '{}_results.csv'.format(dataset)))

    df = pd.DataFrame()
    # Load data
    for path in path_list:
        df = df.append(load_csv(path))

    # Drop useless columns
    df.drop(['stat_winsize', 'warning_FPR', 'warning_delay', 'warning_false_positive'], axis=1, inplace=True)
    datasets = df.dataset.unique()
    df.replace(datasets, [x.split(sep='_')[-1] for x in datasets], inplace=True)
    df['features'] = features

    # Detection when delay is > 0
    df['detection'] = (df.drift_delay >= 0) & (df.drift_delay < df.detection_winsize.values * gamma)

    # Drift
    GE = np.abs(df.train_acc - df.valid_acc)
    df['real_drift'] = (df.drift_acc < (df.valid_acc - GE)) | (df.drift_acc > (df.valid_acc + GE))

    df['delay'] = df['drift_delay'].clip(0)

    df['drift_detection_acc'] = (df['detection'] == df['real_drift']).astype(int)

    df['drift_acc_mod'] = (
            df['drift_detection_acc'] - (df['delay'] / (df.detection_winsize.values * gamma)) ** alpha).clip(0)
    df.loc[df['real_drift'] == False, 'drift_acc_mod'] = df.loc[df['real_drift'] == False]['drift_detection_acc']

    df['drift_TNR'] = 1 - df['drift_FPR']
    df['drift_H'] = ((1 + beta ** 2) * df['drift_acc_mod'] * df['drift_TNR']) / (
            beta ** 2 * df['drift_acc_mod'] + df['drift_TNR'] + 1e-10)
    df['features'] = features

    return df
