import os
import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectKBest

from src.utils.dataload_utils import DATAPATH


def feature_importance(x, y):
    selection = SelectKBest(k='all')
    model = selection.fit(x, y)
    scores = model.scores_
    # First is most imporant feature
    ordered_features = np.argsort(scores)[::-1]
    return ordered_features


def get_permutation(x):
    permuted = np.random.permutation(x)
    try:
        while (permuted == x).any():
            return get_permutation(x)
    except RecursionError:
        pass
    return permuted


def induce_drift(x, y, selected_class = 0, t_start=0, t_end=None, p=0.25, features='top', copy=True):
    """
    t_start : int
        startpoint of drift
    t_end : int
        endpoint of drift
    p : float or int (default: 0.25)
        percentage of features to flip if float
        number of features to flip if int
    features : list (default: None)
        If passed, the feature to flip
    """

    assert t_end == None or t_end > t_start

    assert t_end == None or t_end >= t_start

    if t_end == None:
        t_end = len(x)

    col_idxs = [x for x in range(x.shape[1])]

    # Given features
    if isinstance(features, list):
        permuted = get_permutation(features)
        permute_dict = dict(zip(permuted, features))
    elif isinstance(features, str):
        ranked_f = feature_importance(x, y)
        n = int(x.shape[1] * p)
        if features == 'top':
            f = ranked_f[:n]
        elif features == 'bottom':
            f = ranked_f[-n:]
        permuted = get_permutation(f)
        permute_dict = dict(zip(permuted, f))
    else:
        raise ValueError

    # Permuted column indices
    col_idx = np.array([permute_dict.get(e, e) for e in col_idxs])

    if copy:
        x2 = x.copy()
        for i in range(t_start,t_end):
            if y[i] == selected_class:
                x2[i,:] = x2[i, col_idx]
        return x2, permute_dict
    else:
        for i in range(t_start,t_end):
            if y[i] == selected_class:
                x[i,:] = x[i, col_idx]
        return x, permute_dict


def corrupt_drift(x, y=None, t_start=0, t_end=None, p=0.25, features='top', loc=0.0, std=1.0, copy=True):
    """
    t_end is the end of increase drift strenght.
    After t_end data is still corrupted!
    """

    assert t_end == None or t_end >= t_start

    if t_end == None:
        t_end = len(x)
    transient = t_end - t_start

    col_idxs = [x for x in range(x.shape[1])]

    # Given features
    if isinstance(features, list):
        drift_features = features
    elif isinstance(features, str):
        ranked_f = feature_importance(x, y)
        n = int(x.shape[1] * p)
        if features == 'top':
            f = ranked_f[:n]
        elif features == 'bottom':
            f = ranked_f[-n:]
        drift_features = f
    else:
        raise ValueError

    scale = np.linspace(0 ,std, transient)
    #mean = np.linspace(0, loc, transient)

    if copy:
        x2 = x.copy()
        for i in drift_features:
            if loc is None:
                loc = np.random.random(1) * np.random.randint(0, 10)
            x2[t_start:t_end, i] += np.random.normal(loc, scale)
            x2[t_end:, i] += np.random.normal(loc, std, len(x)-t_end)
        return x2, drift_features
    else:
        for i in drift_features:
            if loc is None:
                loc = np.random.random(1) * np.random.randint(0, 10)
            x[t_start:t_end, i] += np.random.normal(loc, scale)
            x[t_end:, i] += np.random.normal(loc, std, len(x)-t_end)
        return x, drift_features





if __name__ == '__main__':
    # df = pd.read_csv(os.path.join(DATAPATH, 'concept_drift', 'induced', 'fin_adult.csv'))
    # x = df.drop('target', axis=1).values
    # y = df.target.values

    # x2, permute_idxs = induce_drift(x, y, features='top', p=0.1, copy=False)

    # xx = np.arange(20)[:, np.newaxis]
    # xx = xx @ np.array([1,10,100, 1000])[np.newaxis, :]
    # xx2 = induce_drift(xx, features=[0, 1, 2, 3], t_start=5, t_end=10, copy=True)
    pass
