import os
import argparse
import json
import logging
import sys
import warnings
from datetime import datetime
import copy

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn.preprocessing
from sklearn.metrics import accuracy_score
from sklearn.datasets import make_classification

import torch
import torch.multiprocessing as mp
from torch.utils.data import DataLoader, TensorDataset

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import *

from src.utils.dataload_utils import load_data
from src.utils.utils import readable
from src.utils.log_utils import StreamToLogger
from src.utils.new_dataload import OUTPATH, DATAPATH
from src.utils.saver import Saver
from src.models.AEs import MLPAE
from src.models.MultiTaskClassification import AEandClass, LinClassifier, NonLinClassifier
from src.utils.induce_concept_drift import induce_drift, corrupt_drift
from src.utils.concept_drift_datasets import load_data
from src.concept_drift.drift_helper import detection_helper

from src.utils.training_helper_v3 import *

warnings.filterwarnings("ignore")
torch.backends.cudnn.benchmark = True
columns = shutil.get_terminal_size().columns
sns.set_style("whitegrid")


######################################################################################################
class SaverSlave(Saver):
    def __init__(self, path):
        super(Saver)

        self.path = path
        self.makedir_()
        # self.make_log()


def categorizer(y_cont, y_discrete):
    Yd = np.diff(y_cont, axis=0)
    Yd = (Yd > 0).astype(int).squeeze()
    C = pd.Series([x + y for x, y in
                   zip(list(y_discrete[1:].astype(int).astype(str)), list((Yd).astype(str)))]).astype(
        'category')
    return C.cat.codes


def map_abg_main(x):
    if x is None:
        return 'Variable'
    else:
        return '_'.join([str(int(j)) for j in x])


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def remove_duplicates(sequence):
    unique = []
    [unique.append(item) for item in sequence if item not in unique]
    return unique


def check_ziplen(l, n):
    if len(l) % n != 0:
        l += [l[-1]]
        return check_ziplen(l, n)
    else:
        return l


def check_2d3d(train, test, centroids):
    if train.shape[-1] > 3:
        from umap import UMAP
        trs = UMAP(n_components=2, n_neighbors=50, min_dist=0.01, metric='euclidean')
        train = trs.fit_transform(train)
        # valid = trs.transform(valid)
        test = trs.transform(test)
        centroids = trs.transform(centroids)
    return train, test, centroids


def plot_embedding(train, test, Y_train, Y_test, centroids, classes, saver, figname='train_embedding'):
    print('Plot Embedding...')
    cmap = 'jet'
    COL = MplColorHelper(cmap, 0, classes)

    train, test, centroids = check_2d3d(train, test, centroids)

    plt.figure(figsize=(8, 6))
    if train.shape[1] == 3:
        ax = plt.axes(projection='3d')
    else:
        ax = plt.axes()
    l0 = ax.scatter(*train.T, s=50, alpha=0.5, marker='.', label='Train',
                    c=COL.get_rgb(Y_train))
    l1 = ax.scatter(*test.T, s=50, alpha=0.5, marker='^', label='Test',
                    c=COL.get_rgb(Y_test))
    l2 = ax.scatter(*centroids.T, s=250, marker='P', label='Centroids',
                    c=COL.get_rgb([i for i in range(classes)]), edgecolors='black')
    lines = [l0, l1, l2] + [Line2D([0], [0], marker='o', linestyle='', color=c, markersize=10) for c in
                            [COL.get_rgb(i) for i in np.unique(Y_train.astype(int))]]
    labels = [l0.get_label(), l1.get_label(), l2.get_label()] + [i for i in range(len(lines))]
    ax.legend(lines, labels)
    ax.set_title(figname)
    plt.tight_layout()
    saver.save_fig(plt.gcf(), figname)


# FINAL PLOT


def run(x, args, path, result_path):
    print('Process PID:', os.getpid())
    args.init_seed = x

    df_run = single_experiment(args, path)
    if os.path.exists(result_path):
        df_run.to_csv(result_path, mode='a', sep=',', header=False, index=False)
    else:
        df_run.to_csv(result_path, mode='a', sep=',', header=True, index=False)


def parse_args():
    # TODO: make external configuration file -json or similar.
    """
    Parse arguments
    """
    # List handling: https://stackoverflow.com/questions/15753701/how-can-i-pass-a-list-as-a-command-line-argument-with-argparse

    parser = argparse.ArgumentParser(description='Induced Concept Drift with benchmark data')

    parser.add_argument('--n_features', type=int, default=20)
    parser.add_argument('--classes', type=int, default=4)
    parser.add_argument('--informative', type=int, default=10)
    parser.add_argument('--redundant', type=int, default=0)

    parser.add_argument('--dataset', type=str, default='RBF')
    parser.add_argument('--drift_features', type=str, default='top',
                        help='Which features to corrupt. Choices: top - bottom - list with features')
    parser.add_argument('--drift_type', type=str, default='corrupt', help='flip or corrupt')
    parser.add_argument('--drift_p', type=float, default=0.25, help='Percentage of features to corrupt')

    parser.add_argument('--dyn_update', action='store_true', default=False, help='Drift statistic dynamic update')
    parser.add_argument('--drift_winsize', type=int, nargs='+', default=[10, 25, 50, 100], help='Drift window size')
    parser.add_argument('--drift_counter', type=int, nargs='+', default=[1, 10, 25, 50], help='Drift counter')

    parser.add_argument('--weighted', action='store_true', default=False)

    parser.add_argument('--mu', type=float, default=0., help='Mu additive noise')
    parser.add_argument('--sigma', type=float, default=0., help='sigma additive noise')

    parser.add_argument('--preprocessing', type=str, default='StandardScaler',
                        help='Any available preprocessing method from sklearn.preprocessing')

    parser.add_argument('--network', type=str, default='MLP',
                        help='Available networks: MLP')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--learning_rate', type=float, default=1e-2)
    parser.add_argument('--abg', type=float, nargs='+', default=[0, 1, 1])  # AE - Classification - Cluster
    parser.add_argument('--hidden_activation', type=str, default='nn.ReLU()')
    parser.add_argument('--gradient_clip', type=float, default=-1)

    parser.add_argument('--optimizer', type=str, default='torch.optim.SGD')
    parser.add_argument('--momentum', type=float, default=0.9)

    parser.add_argument('--normalization', type=str, default='none')
    parser.add_argument('--dropout', type=float, default=0.25)
    parser.add_argument('--l2penalty', type=float, default=0.0001)

    parser.add_argument('--num_workers', type=int, default=0, help='PyTorch dataloader worker. Set to 0 if debug.')
    parser.add_argument('--init_seed', type=int, default=0, help='RNG seed. Typ. 42, 420, 1337, 0, 69.')
    parser.add_argument('--n_runs', type=int, default=1, help='Number or runs')
    parser.add_argument('--process', type=int, default=1, help='Number of parallel process. Single GPU.')

    parser.add_argument('--nonlin_classifier', action='store_true', default=True, help='Final Classifier')
    parser.add_argument('--classifier_dim', type=int, default=20)
    parser.add_argument('--embedding_size', type=int, default=3)

    # MLP
    parser.add_argument('--neurons', nargs='+', type=int, default=[50, 50])


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
def main():
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
    saver = Saver(OUTPATH, os.path.basename(__file__).split(sep='.py')[0], network=os.path.join(args.dataset,
        args.drift_type,
        args.drift_features,
        map_abg_main(args.abg)))

    # Save json of args/parameters. This is handy for TL
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
    X, y = make_classification(n_samples=5000, n_features=args.n_features, n_informative=args.informative,
                               n_redundant=args.redundant, n_repeated=0, n_classes=args.classes,
                               n_clusters_per_class=1, weights=None, flip_y=0.0, class_sep=2., hypercube=True,
                               shift=0.0, scale=1.0, shuffle=True, random_state=args.init_seed)

    x_train, x_test, Y_train, Y_test = train_test_split(X, y, shuffle=False, train_size=0.6)
    x_train, x_valid, Y_train, Y_valid = train_test_split(x_train, Y_train, shuffle=False, test_size=0.2)

    ## Data Scaling
    normalizer = eval(args.preprocessing)()
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

    # Weighted
    weights = None
    if args.weighted:
        print('Weighted')
        nSamples = np.unique(Y_train, return_counts=True)[1]
        tot_samples = len(Y_train)
        weights = (nSamples / tot_samples).max() / (nSamples / tot_samples)

    classes = len(np.unique(Y_train))

    ###########################
    # Induce Concept Drift
    train_samples = len(x_train)
    valid_samples = len(x_valid)
    test_samples = len(x_test)

    if args.drift_type == 'flip':
        t_start = int(0.60 * test_samples)
        t_end = t_start
        x_test, permute_dict = induce_drift(x_test, y=Y_test, t_start=t_start, t_end=None, p=args.drift_p,
                                            features=args.drift_features, copy=False)
    elif args.drift_type == 'corrupt':
        t_start = int(0.60 * test_samples)
        t_end = t_start + int(0.20 * test_samples)
        # t_end = t_start
        x_test, permute_dict = corrupt_drift(x_test, y=Y_test, t_start=t_start, t_end=t_end, p=args.drift_p,
                                             features=args.drift_features, loc=1., std=1.0, copy=False)

    print('Num Classes: ', classes)
    print('Train:', x_train.shape, Y_train.shape, [(Y_train == i).sum() for i in np.unique(Y_train)])
    print('Validation:', x_valid.shape, Y_valid.shape,
          [(Y_valid == i).sum() for i in np.unique(Y_valid)])
    print('Test:', x_test.shape, Y_test.shape,
          [(Y_test == i).sum() for i in np.unique(Y_test)])

    ######################################################################################################
    classes = len(np.unique(Y_train))
    args.nbins = classes
    history = x_train.shape[1]

    # Network definition
    if args.nonlin_classifier:
        classifier = NonLinClassifier(args.embedding_size, classes, d_hidd=args.classifier_dim, dropout=args.dropout,
                                      norm=args.normalization)
    else:
        classifier = LinClassifier(args.embedding_size, classes)

    model_ae = MLPAE(input_shape=x_train.shape[1], embedding_dim=args.embedding_size, hidden_neurons=args.neurons,
                     hidd_act=eval(args.hidden_activation), dropout=args.dropout,
                     normalization=args.normalization).to(device)

    ######################################################################################################
    # model is multi task - AE Branch and Classification branch
    model = AEandClass(ae=model_ae, classifier=classifier, n_out=1, name='MLP').to(device)

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
    loss_centroids = CentroidLoss(feat_dim=args.embedding_size, num_classes=classes, reduction='mean').to(device)
    criterion = nn.CrossEntropyLoss(reduction='none', weight=None)

    print('Optimizer: ', args.optimizer)
    if 'SGD' in args.optimizer:
        optimizer = eval(args.optimizer)(
            list(filter(lambda p: p.requires_grad, model.parameters())) + list(loss_centroids.parameters()),
            lr=args.learning_rate, weight_decay=args.l2penalty, momentum=args.momentum)
    else:
        optimizer = eval(args.optimizer)(
            list(filter(lambda p: p.requires_grad, model.parameters())) + list(loss_centroids.parameters()),
            lr=args.learning_rate, weight_decay=args.l2penalty, eps=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.1)

    # Train model
    model, loss_centroids = train_model(model, train_loader, valid_loader, latent_constraint=loss_centroids,
                                        epochs=args.epochs, optimizer=optimizer,
                                        scheduler=scheduler, criterion=criterion,
                                        saver=saver, plot_loss_flag=args.plt_loss,
                                        clip=args.gradient_clip, abg=args.abg, sigma=args.sigma)
    cluster_centers = loss_centroids.centers.detach().cpu().numpy()
    print('Train ended')

    ######################################################################################################
    # Eval
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False, drop_last=False,
                              num_workers=args.num_workers, pin_memory=True)
    train_results = evaluate_class_recons(model, x_train, Y_train, train_loader, saver,
                                          network=args.network, datatype='Train', plt_cm=args.plt_cm,
                                          plt_lables=False, plt_recons=True if args.abg[0] != 0 else False)
    valid_results = evaluate_class_recons(model, x_valid, Y_valid, valid_loader, saver,
                                          network=args.network, datatype='Valid', plt_cm=args.plt_cm,
                                          plt_lables=False, plt_recons=True if args.abg[0] != 0 else False)
    test_results = evaluate_class_recons(model, x_test, Y_test, test_loader, saver,
                                         network=args.network, datatype='Test', plt_cm=args.plt_cm,
                                         plt_lables=True, plt_recons=True if args.abg[0] != 0 else False)

    # Embeddings
    train_embedding = predict(model.encoder, train_loader).squeeze()
    valid_embedding = predict(model.encoder, valid_loader).squeeze()
    test_embedding = predict(model.encoder, test_loader).squeeze()

    # Get centroids when not constrained representation
    if args.abg[2] == 0:
        cc = []
        for i in range(len(np.unique(Y_train))):
            cc.append(train_embedding[Y_train == i].mean(axis=0))
        cluster_centers = np.array(cc)

    if train_embedding.shape[1] <= 3:
        plot_embedding(train_embedding, valid_embedding, Y_train, Y_valid, cluster_centers, classes=classes,
                       saver=saver,
                       figname='Train data')
        plot_embedding(valid_embedding, test_embedding[t_start:], Y_valid, Y_test[t_start:], cluster_centers, classes=classes, saver=saver,
                       figname='Test (drift) data')
    else:
        print('Skipping embedding plot')

    x = valid_embedding
    y = Y_valid
    stds = np.empty((len(np.unique(y)), 3, 3))
    for i in range(len(np.unique(y))):
        stds[i,:] = np.cov(x[y == i].T)
    print(stds)
    s = '_yes' if args.abg == [0, 1, 1] else '_no'
    np.save(os.path.join('/home/castel/PycharmProjects/torchembedding/results/CD_results/covs/', args.dataset+s), stds)

    ######################################################################################################
    # Begin Drift detection part
    ######################################################################################################
    # Test predictions
    _, yhat_ = predict_multi(model, test_loader)
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
    detectors = ['ZScore', 'IRQ', 'P_modified', 'EMA', 'HDDDM_Emb', 'HDDDM_Input', 'KS_Win_Emb',
                 'DDM', 'EDDM', 'ADWIN']

    for d in args.drift_winsize:
        skip_flag = False
        for c in args.drift_counter:
            if skip_flag and d * c * 0.01 <= 1:
                continue

            df_results, drift_idxs, warning_idxs, obj = detection_helper(x_valid, x_test, valid_embedding,
                                                                         test_embedding, y_binary, running_accuracy,
                                                                         cluster_centers,
                                                                         saver, d, c, t_start, t_end, args,
                                                                         detectors_list=detectors)

            if d * c * 0.01 <= 1:
                skip_flag = True
            df = df.append(df_results)
            plt.close('all')

    # Complete dataframe
    df['train_acc'] = train_results['acc']
    df['valid_acc'] = valid_results['acc']
    df['drift_acc'] = test_results['acc']
    df['dataset'] = args.dataset
    df['constrained'] = args.abg == [0, 1, 1]
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
    main()
