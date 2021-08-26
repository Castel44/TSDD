import argparse
import copy
import json
import logging
import os
import shutil
import sys
import time
import warnings
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.multiprocessing as mp
import torch.nn as nn
from sklearn.datasets import make_classification, make_blobs
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

from src.models.MLP import MetaMLP
from src.utils.constraint_module import CentroidLoss
from src.utils.drift_helper import detection_helper
from src.utils.global_var import OUTPATH, columns
from src.utils.induce_concept_drift import induce_drift, corrupt_drift
from src.utils.load_datasets import load_data
from src.utils.log_utils import StreamToLogger
from src.utils.plotting_utils import plot_embedding
from src.utils.saver import Saver, SaverSlave
from src.utils.utils import readable, predict, check_ziplen, remove_duplicates, linear_interp, train_model, \
    evaluate_model

warnings.filterwarnings("ignore")
torch.backends.cudnn.benchmark = True
sns.set_style("whitegrid")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


######################################################################################################
def run(x, args, path, result_path):
    print('Process PID:', os.getpid())
    args.init_seed = x

    df_run = single_experiment(args, path)
    if os.path.exists(result_path):
        df_run.to_csv(result_path, mode='a', sep=',', header=False, index=False)
    else:
        df_run.to_csv(result_path, mode='a', sep=',', header=True, index=False)


def parse_args():
    parser = argparse.ArgumentParser(description='Concept drift detection experiments.'
                                                 'By default the constrained module is used.')

    parser.add_argument('--dataset', type=str, default='RBF', choices=(
        'fin_adult', 'fin_wine', 'fin_bank', 'fin_digits08', 'fin_digits17', 'fin_musk', 'fin_phis', 'phishing', 'spam',
        'RBF', 'MovRBF'))

    # Synthetic data parameters
    parser.add_argument('--features', type=int, default=20)
    parser.add_argument('--classes', type=int, default=4)
    parser.add_argument('--informative', type=int, default=10)
    parser.add_argument('--redundant', type=int, default=0)

    parser.add_argument('--drift_features', type=str, default='top',
                        help='Which features to corrupt. Choices: top - bottom - list with features')
    parser.add_argument('--drift_type', type=str, default='gradual', choices=('step', 'gradual'))
    parser.add_argument('--drift_p', type=float, default=0.25, help='Percentage of features to corrupt')

    parser.add_argument('--detectors', type=str, nargs='+',
                        default=['ZScore', 'IRQ', 'P_modified', 'EMA', 'HDDDM_Emb', 'HDDDM_Input', 'IKS_emb',
                                 'DDM', 'EDDM', 'ADWIN'],
                        choices=(
                            'ZScore', 'IRQ', 'P_modified', 'EMA', 'HDDDM_Emb', 'HDDDM_Input', 'IKS_Input',
                            'IKS_emb_raw', 'IKS_emb', 'KSWIN_Emb', 'PH_Emb', 'PH_error', 'DDM', 'EDDM', 'ADWIN',
                            'MMD_Input', 'MMD_Emb', 'Chi_Input', 'Chi_Emb', 'LSD_Input', 'LSD_Emb', 'KS_Input',
                            'KS_Emb'), help='Available drift detectors')

    parser.add_argument('--dyn_update', action='store_true', default=False, help='Drift statistic dynamic update')
    parser.add_argument('--drift_history', type=int, nargs='+', default=[25, 50], help='Drift detection history size')
    parser.add_argument('--drift_th', type=int, nargs='+', default=[10, 50], help='Drift detection threshold')

    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    parser.add_argument('--hidden_activation', type=str, default='nn.ReLU()')
    parser.add_argument('--unconstrained', action='store_true', default=False, help='Not use constrained module')

    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--normalization', type=str, default='none')
    parser.add_argument('--dropout', type=float, default=0.25)
    parser.add_argument('--l2penalty', type=float, default=0.001)

    parser.add_argument('--num_workers', type=int, default=0, help='PyTorch dataloader worker. Set to 0 if debug.')
    parser.add_argument('--init_seed', type=int, default=420, help='RNG seed. Typ. 42, 420, 1337, 0, 69.')
    parser.add_argument('--n_runs', type=int, default=1, help='Number or runs')
    parser.add_argument('--process', type=int, default=1, help='Number of parallel process. Single GPU.')

    parser.add_argument('--neurons', nargs='+', type=int, default=[100, 50, 20])
    parser.add_argument('--classifier_dim', type=int, default=20)
    parser.add_argument('--embedding_size', type=int, default=3)

    # Suppress terminal out
    parser.add_argument('--disable_print', action='store_true', default=False)
    parser.add_argument('--plt_loss', action='store_true', default=True)
    parser.add_argument('--plt_embedding', action='store_true', default=False)
    parser.add_argument('--plt_loss_hist', action='store_true', default=True)
    parser.add_argument('--plt_cm', action='store_true', default=True)
    parser.add_argument('--plt_recons', action='store_true', default=True)
    parser.add_argument('--headless', action='store_true', default=False, help='Matplotlib backend')

    # Add parameters for each particular network
    args = parser.parse_args()
    return args


######################################################################################################
def load_corrupt_data(args):
    # Loading data
    if args.dataset == 'MovRBF':
        # Blobs dataset with drift induced by moving centroids
        X, y, centers = make_blobs(n_samples=2000, n_features=args.features, centers=args.classes, cluster_std=1.0,
                                   return_centers=True)
        x_test_, Y_test_, centers_end = make_blobs(n_samples=1000, n_features=args.features, centers=args.classes,
                                                   cluster_std=1.0,
                                                   return_centers=True)
        Y_test_ = Y_test_[:, np.newaxis]
        x_test, Y_test = make_blobs(n_samples=500, n_features=args.features, centers=centers, cluster_std=1.0)
        Y_test = Y_test[:, np.newaxis]
        if args.drift_type == 'gradual':
            for i in [x * 0.1 for x in range(2, 12, 2)]:
                x_, y_ = make_blobs(n_samples=100, n_features=args.features,
                                    centers=linear_interp(centers_end, centers, i),
                                    cluster_std=1.0)
                y_ = y_[:, np.newaxis]
                x_test = np.vstack([x_test, x_])
                Y_test = np.vstack([Y_test, y_])
        x_test = np.vstack([x_test, x_test_])
        Y_test = np.vstack([Y_test, Y_test_])
        Y_test = Y_test.squeeze(1)
        t_start = 500
        t_end = t_start

        x_train, x_valid, Y_train, Y_valid = train_test_split(X, y, shuffle=True, test_size=0.2)

        # Data Scaling
        normalizer = StandardScaler()
        x_train = normalizer.fit_transform(x_train)
        x_valid = normalizer.transform(x_valid)
        x_test = normalizer.transform(x_test)

        # Encode labels
        unique_train = np.unique(Y_train)
        label_encoder = dict(zip(unique_train, range(len(unique_train))))
        unique_test = np.setdiff1d(np.unique(y), unique_train)
        label_encoder.update(dict(zip(unique_test, range(len(unique_train), len(np.unique(y))))))

        Y_train = np.array([label_encoder.get(e, e) for e in Y_train])
        Y_valid = np.array([label_encoder.get(e, e) for e in Y_valid])
        Y_test = np.array([label_encoder.get(e, e) for e in Y_test])

    else:
        # RBF or UCI datasets with drift induced by corrupting the features
        if args.dataset == 'RBF':
            # Loading data
            X, y = make_classification(n_samples=2000, n_features=args.features, n_informative=args.informative,
                                       n_redundant=args.redundant, n_repeated=0, n_classes=args.classes,
                                       n_clusters_per_class=1, weights=None, flip_y=0.0, class_sep=2., hypercube=True,
                                       shift=0.0, scale=1.0, shuffle=True, random_state=args.init_seed)
        else:
            # UCI datasets
            X, y = load_data(args.dataset)

        x_train, x_test, Y_train, Y_test = train_test_split(X, y, shuffle=False, train_size=0.6)
        x_train, x_valid, Y_train, Y_valid = train_test_split(x_train, Y_train, shuffle=False, test_size=0.2)

        # scale data before corruption!!!
        # Data Scaling
        normalizer = StandardScaler()
        x_train = normalizer.fit_transform(x_train)
        x_valid = normalizer.transform(x_valid)
        x_test = normalizer.transform(x_test)

        # Encode labels
        unique_train = np.unique(Y_train)
        label_encoder = dict(zip(unique_train, range(len(unique_train))))
        unique_test = np.setdiff1d(np.unique(y), unique_train)
        label_encoder.update(dict(zip(unique_test, range(len(unique_train), len(np.unique(y))))))

        Y_train = np.array([label_encoder.get(e, e) for e in Y_train])
        Y_valid = np.array([label_encoder.get(e, e) for e in Y_valid])
        Y_test = np.array([label_encoder.get(e, e) for e in Y_test])

        # Induce Drift
        test_samples = len(x_test)

        if args.drift_type == 'step':
            t_start = int(0.60 * test_samples)
            t_end = t_start
            x_test, permute_dict = induce_drift(x_test, y=Y_test, t_start=t_start, t_end=None, p=args.drift_p,
                                                features=args.drift_features, copy=False)
        elif args.drift_type == 'gradual':
            t_start = int(0.60 * test_samples)
            t_end = t_start + int(0.20 * test_samples)
            # t_end = t_start
            x_test, permute_dict = corrupt_drift(x_test, y=Y_test, t_start=t_start, t_end=t_end, p=args.drift_p,
                                                 features=args.drift_features, loc=1.0, std=1.0, copy=False)

    classes = len(np.unique(Y_train))
    print('Num Classes: ', classes)
    print('Train:', x_train.shape, Y_train.shape, [(Y_train == i).sum() for i in np.unique(Y_train)])
    print('Validation:', x_valid.shape, Y_valid.shape,
          [(Y_valid == i).sum() for i in np.unique(Y_valid)])
    print('Test:', x_test.shape, Y_test.shape,
          [(Y_test == i).sum() for i in np.unique(Y_test)])

    return x_train, Y_train, x_valid, Y_valid, x_test, Y_test, t_start, t_end


######################################################################################################
def single_experiment(args, path):
    path = os.path.join(path, 'seed_{}'.format(args.init_seed))
    saver = SaverSlave(path)

    # Logging setting
    print('run logfile at: ', os.path.join(saver.path, 'logfile.log'))
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(name)s: %(message)s',
        datefmt='%m/%d/%Y %H:%M:%S',
        filename=os.path.join(saver.path, 'logfile.log'),
        filemode='a'
    )

    # Redirect stdout
    stdout_logger = logging.getLogger('STDOUT')
    slout = StreamToLogger(stdout_logger, logging.INFO)
    sys.stdout = slout

    # Redirect stderr
    stderr_logger = logging.getLogger('STDERR')
    slerr = StreamToLogger(stderr_logger, logging.ERROR)
    sys.stderr = slerr

    # Suppress output
    if args.disable_print:
        slout.terminal = open(os.devnull, 'w')
        slerr.terminal = open(os.devnull, 'w')

    ######################################################################################################
    # Loading data
    x_train, Y_train, x_valid, Y_valid, x_test, Y_test, t_start, t_end = load_corrupt_data(args)
    test_samples = len(x_test)

    ######################################################################################################
    # Model definition
    classes = len(np.unique(Y_train))
    args.nbins = classes

    model = MetaMLP(input_shape=x_train.shape[1], embedding_dim=args.embedding_size, n_class=classes,
                    hidden_neurons=args.neurons, hidd_act=eval(args.hidden_activation), dropout=args.dropout,
                    normalization=args.normalization, name='MLP').to(device)
    nParams = sum([p.nelement() for p in model.parameters()])
    s = 'MODEL: %s: Number of parameters: %s' % ('MLP', readable(nParams))
    print(s)

    ######################################################################################################
    # Main loop
    train_dataset = TensorDataset(torch.from_numpy(x_train).float(), torch.from_numpy(Y_train).long())
    valid_dateset = TensorDataset(torch.from_numpy(x_valid).float(), torch.from_numpy(Y_valid).long())
    test_dateset = TensorDataset(torch.from_numpy(x_test).float(), torch.from_numpy(Y_test).long())

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=False,
                              num_workers=args.num_workers, pin_memory=True)
    valid_loader = DataLoader(valid_dateset, batch_size=args.batch_size, shuffle=False, drop_last=False,
                              num_workers=args.num_workers, pin_memory=True)
    test_loader = DataLoader(test_dateset, batch_size=args.batch_size, shuffle=False, drop_last=False,
                             num_workers=args.num_workers, pin_memory=True)

    ######################################################################################################
    if args.unconstrained:
        loss_centroids = None
        param_list = list(filter(lambda p: p.requires_grad, model.parameters()))
    else:
        loss_centroids = CentroidLoss(feat_dim=args.embedding_size, num_classes=classes, reduction='mean').to(device)
        param_list = list(filter(lambda p: p.requires_grad, model.parameters())) + list(loss_centroids.parameters())

    optimizer = torch.optim.SGD(param_list, lr=args.learning_rate, weight_decay=args.l2penalty, momentum=args.momentum)
    criterion = nn.CrossEntropyLoss(reduction='none', weight=None)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.1)

    # Train model
    model, loss_centroids = train_model(model, train_loader, valid_loader, latent_constraint=loss_centroids,
                                        epochs=args.epochs, optimizer=optimizer,
                                        scheduler=scheduler, criterion=criterion,
                                        saver=saver, plot_loss_flag=args.plt_loss)

    print('Train ended')

    ######################################################################################################
    # Eval
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False, drop_last=False,
                              num_workers=args.num_workers, pin_memory=True)
    train_results = evaluate_model(model, train_loader, Y_train, saver=saver, network='MLP', datatype='Train',
                                   plt_cm=args.plt_cm, plt_pred=False)
    valid_results = evaluate_model(model, valid_loader, Y_valid, saver=saver, network='MLP', datatype='Valid',
                                   plt_cm=args.plt_cm, plt_pred=False)
    test_results = evaluate_model(model, test_loader, Y_test, saver=saver, network='MLP', datatype='Test',
                                  plt_cm=args.plt_cm, plt_pred=True)

    # Embeddings
    train_embedding = predict(model.encoder, train_loader).squeeze()
    valid_embedding = predict(model.encoder, valid_loader).squeeze()
    test_embedding = predict(model.encoder, test_loader).squeeze()

    # Get centroids position
    if args.unconstrained:
        cc = []
        for i in range(len(np.unique(Y_train))):
            cc.append(train_embedding[Y_train == i].mean(axis=0))
        cluster_centers = np.array(cc)
    else:
        cluster_centers = loss_centroids.centers.detach().cpu().numpy()

    if train_embedding.shape[1] <= 3:
        plot_embedding(train_embedding, valid_embedding, Y_train, Y_valid, cluster_centers, classes=classes,
                       saver=saver,
                       figname='Train data')
        plot_embedding(valid_embedding, test_embedding, Y_valid, Y_test, cluster_centers, classes=classes, saver=saver,
                       figname='Test (drift) data')
    else:
        print('Skipping embedding plot')

    ######################################################################################################
    # Begin Drift detection part
    ######################################################################################################
    # Test predictions
    yhat_ = predict(model, test_loader)
    yhat = yhat_.argmax(axis=1)
    y_binary = (yhat == Y_test).astype(int)
    y_binary = 1 - y_binary

    # Running accuracy
    w = 50
    running_accuracy = np.empty(test_samples)
    for t in range(test_samples):
        if t < w:
            score = accuracy_score(yhat[:t], Y_test[:t])
        else:
            score = accuracy_score(yhat[t - w:t], Y_test[t - w:t])
        running_accuracy[t] = score
    # First sample will be NaN
    running_accuracy[0] = running_accuracy[1]
    # Check other nans and fill with zero
    if np.isnan(running_accuracy).any():
        running_accuracy[np.where(np.isnan(running_accuracy))[0]] = 0
    n = len(running_accuracy)

    plt.figure()
    plt.plot([x for x in range(n)], running_accuracy)
    for t in (t_start, t_end):
        plt.axvline(t, linestyle='-.', c='blue')
    plt.title(f'running accuracy (Winsize={w})')
    plt.tight_layout()
    saver.save_fig(plt.gcf(), 'running_acc')
    # plt.show(block=True)

    ######################################################################################################
    df = pd.DataFrame()
    detectors = ['ZScore', 'IRQ', 'P_modified', 'EMA', 'HDDDM_Emb', 'HDDDM_Input', 'IKS_emb',
                 'DDM', 'EDDM', 'ADWIN']

    for d in args.drift_history:
        skip_flag = False
        for c in args.drift_th:
            if skip_flag and d * c * 0.01 <= 1:
                continue

            df_results, drift_idxs, warning_idxs, obj = detection_helper(x_valid, x_test, valid_embedding,
                                                                         test_embedding, y_binary, running_accuracy,
                                                                         cluster_centers,
                                                                         saver, d, c, t_start, t_end, args,
                                                                         detectors_list=args.detectors)

            if d * c * 0.01 <= 1:
                skip_flag = True
            df = df.append(df_results)

    # Complete dataframe
    df['train_acc'] = train_results['acc']
    df['valid_acc'] = valid_results['acc']
    df['drift_acc'] = test_results['acc']
    df['dataset'] = args.dataset
    df['constrained'] = str(not args.unconstrained)
    df['features'] = args.drift_features
    df.to_csv(os.path.join(saver.path, 'results_df.csv'), sep=',', index=False)

    plt.close('all')
    # g = sns.relplot(data=df, y='drift_delay', x='drift_false_positive', hue='name', style='detection_winsize',
    #                col='counter', palette='Paired', kind='scatter', markers=True, dashes=False, sizes=10)
    # plt.savefig(os.path.join(saver.path, 'final.png'), bbox_inches='tight')
    # g = sns.relplot(data=df, y='drift_delay', x='drift_false_positive', hue='name', style='counter',
    #                col='detection_winsize', palette='Paired', kind='scatter', markers=True, dashes=False, sizes=10)
    # plt.savefig(os.path.join(saver.path, 'final2.png'), bbox_inches='tight')
    return df


if __name__ == '__main__':
    args = parse_args()
    print(args)
    print()

    ######################################################################################################
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    SEED = args.init_seed
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if device == 'cuda':
        torch.cuda.manual_seed(SEED)

    if args.headless:
        print('Setting Headless support')
        plt.switch_backend('Agg')
    else:
        backend = 'Qt5Agg'
        print('Swtiching matplotlib backend to', backend)
        plt.switch_backend(backend)
    print()

    ######################################################################################################
    # LOG STUFF
    # Declare saver object
    if args.dataset == 'MovRBF':
        saver = Saver(OUTPATH, os.path.basename(__file__).split(sep='.py')[0],
                      hierarchy=os.path.join(args.dataset,
                                             args.drift_type,
                                             str(not args.unconstrained)))
    else:
        saver = Saver(OUTPATH, os.path.basename(__file__).split(sep='.py')[0],
                      hierarchy=os.path.join(args.dataset,
                                             args.drift_type,
                                             args.drift_features,
                                             str(not args.unconstrained)))

    # Save json of args/parameters
    with open(os.path.join(saver.path, 'args.json'), 'w') as fp:
        json.dump(vars(args), fp, indent=4)

    ######################################################################################################
    # Main loop
    csv_path = os.path.join(saver.path, '{}_{}_results.csv'.format(args.dataset, args.drift_features))

    seeds_list = list(np.random.choice(1000, args.n_runs, replace=False))
    seeds_list = check_ziplen(seeds_list, args.process)
    total_run = args.n_runs

    iterator = zip(*[seeds_list[j::args.process] for j in range(args.process)])
    total_iter = len(list(iterator))

    torch.multiprocessing.set_start_method('spawn', force=True)
    start_datetime = datetime.now()
    print('{} - Start Experiments. {} parallel process. Single GPU. \r\n'.format(
        start_datetime.strftime("%d/%m/%Y %H:%M:%S"), args.process))
    for i, (x) in enumerate(zip(*[seeds_list[j::args.process] for j in range(args.process)])):
        start_time = time.time()
    x = remove_duplicates(x)
    n_process = len(x)
    idxs = [i * args.process + j for j in range(1, n_process + 1)]
    print('/' * shutil.get_terminal_size().columns)
    print(
        'ITERATION: {}/{}'.format(idxs, total_run).center(columns))
    print('/' * shutil.get_terminal_size().columns)

    process = []
    for j in range(n_process):
        process.append(mp.Process(target=run, args=(x[j], copy.deepcopy(args), saver.path, csv_path)))

    for p in process:
        p.start()

    for p in process:
        p.join()

    end_time = time.time()
    iter_seconds = end_time - start_time
    total_seconds = end_time - start_datetime.timestamp()
    print('Iteration time: {} - ETA: {}'.format(time.strftime("%Mm:%Ss", time.gmtime(iter_seconds)),
                                                time.strftime('%Hh:%Mm:%Ss',
                                                              time.gmtime(
                                                                  total_seconds * (total_iter / (i + 1) - 1)))))
    print()

    print('*' * shutil.get_terminal_size().columns)
    print('DONE!')
    end_datetime = datetime.now()
    total_seconds = (end_datetime - start_datetime).total_seconds()
    print('{} - Experiment took: {}'.format(end_datetime.strftime("%d/%m/%Y %H:%M:%S"),
                                            time.strftime("%Hh:%Mm:%Ss", time.gmtime(total_seconds))))
    print(f'results dataframe saved in: {csv_path}')
