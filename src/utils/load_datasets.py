import os

import pandas as pd

from src.utils.global_var import DATAPATH


# Available datasets
def check_datasets(path: str):
    datasets = {}
    for path, subdirs, files in os.walk(path):
        for name in files:
            if name.endswith('.csv'):
                datasets[name[:-4]] = os.path.join(path, name)
    return datasets


def check_subgroup(grp: str = 'real-world'):
    """
    INPUT
    grp : str : real-world or artificial or induced

    RETURN
    datasets : dict
    """
    return check_datasets(os.path.join(DATAPATH, grp))


def load_data(dataset: str):
    available_datasets = check_datasets(DATAPATH)
    assert dataset in available_datasets.keys(), f'{dataset=} not available. ' \
                                                 f'Existing datasets are {list(available_datasets.keys())}'

    print(f'Loading {dataset}...')
    df = pd.read_csv(available_datasets[dataset], header=0)
    try:
        X = df.drop('target', axis=1).values
        y = df.target.values

    except KeyError:
        X = df.drop('class', axis=1).values
        y = df['class'].values

    return X, y


if __name__ == '__main__':
    data = 'movingRBF'

    X, y = load_data(data)
