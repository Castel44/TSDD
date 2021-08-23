import os
import os
import shutil
import time

import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from scipy.special import softmax
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, \
    classification_report
from torch.autograd import Variable

from src.utils.saver import Saver
from src.utils.torch_utils import predict, plot_loss, predict_multi
from src.utils.utils_postprocess import plot_prediction
from src.utils.metrics import evaluate_multi

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.lines import Line2D

columns = shutil.get_terminal_size().columns
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


######################################################################################################
class MplColorHelper:

    def __init__(self, cmap_name, start_val, stop_val):
        self.cmap_name = cmap_name
        self.cmap = plt.get_cmap(cmap_name)
        self.norm = mpl.colors.Normalize(vmin=start_val, vmax=stop_val)
        self.scalarMap = cm.ScalarMappable(norm=self.norm, cmap=self.cmap)

    def get_rgb(self, val):
        return self.scalarMap.to_rgba(val)


def remove_empty_dir(path):
    try:
        os.rmdir(path)
    except OSError:
        pass


def remove_empty_dirs(path):
    for root, dirnames, filenames in os.walk(path, topdown=False):
        for dirname in dirnames:
            remove_empty_dir(os.path.realpath(os.path.join(root, dirname)))


class SaverSlave(Saver):
    def __init__(self, path):
        super(Saver)

        self.path = path
        self.makedir_()
        # self.make_log()


def reset_seed_(seed):
    # Resetting SEED to fair comparison of results
    print('Settint seed: {}'.format(seed))
    torch.manual_seed(seed)
    if device == 'cuda':
        torch.cuda.manual_seed(seed)
    np.random.seed(seed)


def reset_model(model):
    print('Resetting model parameters...')
    for layer in model.modules():
        if hasattr(layer, 'reset_parameters'):
            layer.reset_parameters()
    return model


def RF_check(kernel_size, blocks, history):
    RF = (kernel_size - 1) * blocks * 2 ** (blocks - 1)
    print('Receptive field: {}, History window: {}'.format(RF, history))
    if RF > history:
        print('OK')
    else:
        while RF <= history:
            blocks += 1
            RF = (kernel_size - 1) * blocks * 2 ** (blocks - 1)
            print('Adding layers.. L: {}, RF:{}'.format(blocks, RF))

    print('Receptive field: {}, History window: {}, LAYERS:{}'.format(RF, history, blocks))
    return blocks


def append_results_dict(main, sub):
    for k, v in zip(sub.keys(), sub.values()):
        main[k].append(v)
    return main


def linear_comb(w, x1, x2):
    return (1 - w) * x1 + w * x2


def reduce_loss(loss, reduction='mean'):
    return loss.mean() if reduction == 'mean' else loss.sum() if reduction == 'sum' else loss


class CentroidLoss(nn.Module):
    def __init__(self, feat_dim, num_classes, reduction='mean'):
        super(CentroidLoss, self).__init__()
        self.classes = num_classes
        self.centers = nn.Parameter(torch.randn(num_classes, feat_dim), requires_grad=True)
        self.reduction = reduction
        self.rho = 1.0

    def forward(self, h, y):
        C = self.centers
        norm_squared = torch.sum((h.unsqueeze(1) - C) ** 2, 2)
        # Attractive
        distance = norm_squared.gather(1, y.unsqueeze(1)).squeeze()
        #distance = torch.clamp(distance-1, 0) # distance with margin
        # Repulsive
        logsum = torch.logsumexp(-torch.sqrt(norm_squared), dim=1)
        #logsum = -(norm_squared.sum(dim=1) - distance)**0.5 # remove euclidean
        loss = reduce_loss(distance + logsum, reduction=self.reduction)
        # Regularization
        if self.classes != 1:
            reg = self.regularization(reduction='sum')
            return loss + self.rho * reg
        else:
            return loss

    def regularization(self, reduction='sum'):
        C = self.centers
        pairwise_dist = torch.cdist(C, C, p=2) ** 2
        pairwise_dist = pairwise_dist.masked_fill(
            torch.zeros((C.size(0), C.size(0))).fill_diagonal_(1).bool().to(device), float('inf'))
        distance_reg = reduce_loss(-(torch.min(torch.log(pairwise_dist), dim=-1)[0]), reduction=reduction)
        return distance_reg


def cluster_accuracy(y_true, y_predicted, cluster_number=None):
    """
    Calculate clustering accuracy after using the linear_sum_assignment function in SciPy to
    determine reassignments.

    :param y_true: list of true cluster numbers, an integer array 0-indexed
    :param y_predicted: list  of predicted cluster numbers, an integer array 0-indexed
    :param cluster_number: number of clusters, if None then calculated from input
    :return: reassignment dictionary, clustering accuracy
    """
    if cluster_number is None:
        cluster_number = (
                max(y_predicted.max(), y_true.max()) + 1
        )  # assume labels are 0-indexed
    count_matrix = np.zeros((cluster_number, cluster_number), dtype=np.int64)
    for i in range(y_predicted.size):
        count_matrix[y_predicted[i], y_true[i]] += 1

    row_ind, col_ind = linear_sum_assignment(count_matrix.max() - count_matrix)
    reassignment = dict(zip(row_ind, col_ind))
    accuracy = count_matrix[row_ind, col_ind].sum() / y_predicted.size
    return reassignment, accuracy


def add_noise(x, sigma=0.2, mu=0.):
    noise = mu + torch.randn(x.size()) * sigma
    noisy_x = x + noise
    return noisy_x


def train_model(model, train_data, valid_data, epochs, optimizer, criterion, latent_constraint=None, scheduler=None,
                abg=(1, 1, 1), sigma=0, clip=-1, plot_loss_flag=True, saver=None):
    # TODO: add early stopping
    # TODO: make general also for regression
    alpha, beta, gamma = abg
    network = model.get_name()

    print('-' * shutil.get_terminal_size().columns)
    s = 'TRAINING MODEL {} WITH {} LOSS'.format(network, criterion._get_name())
    print(s)
    print('-' * shutil.get_terminal_size().columns)

    loss_ae = nn.MSELoss(reduction='mean')

    avg_train_loss = []
    avg_valid_loss = []
    avg_train_acc = []
    avg_valid_acc = []

    # Training loop
    try:
        for idx_epoch in range(1, epochs + 1):
            epochstart = time.time()
            train_loss = []
            train_acc = []
            train_acc_corrected = []
            epoch_losses = torch.Tensor()
            epoch_indices = torch.Tensor()

            # Train
            model.train()
            if latent_constraint is not None:
                latent_constraint.train()
            for data, target in train_data:
                target = target.to(device)
                clean_data = data.to(device)
                data = add_noise(data, sigma=sigma, mu=0.).to(device)
                batch_size = data.size(0)

                # Forward
                optimizer.zero_grad()
                out_AE, out_class, embedding = model(data)
                embedding = embedding.squeeze()

                loss_cntrs_ = latent_constraint(embedding, target)
                loss_class_ = criterion(out_class, target)
                loss_recons_ = loss_ae(out_AE, clean_data)

                loss = alpha * loss_recons_ + beta * loss_class_.mean() + gamma * loss_cntrs_.mean()
                loss.backward()

                # Gradient clip
                if clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
                optimizer.step()

                # Update loss  monitor
                train_loss.append(loss.data.item())

                # Accuracy
                prob = F.softmax(out_class, dim=1)
                train_acc.append((torch.argmax(prob, dim=1) == target).sum().item() / batch_size)

            scheduler.step()
            # Validate
            valid_loss, valid_acc = eval_model(model, valid_data, criterion, latent_constraint, abg)

            # calculate average loss over an epoch
            train_loss_epoch = np.average(train_loss)
            avg_train_loss.append(train_loss_epoch)
            avg_valid_loss.append(valid_loss)

            train_acc_epoch = 100 * np.average(train_acc)

            avg_train_acc.append(train_acc_epoch)
            avg_valid_acc.append(valid_acc)

            print(
                'Epoch [{}/{}], Time:{:.3f} - TrAcc:{:.3f} - ValAcc:{:.3f} - TrLoss:{:.5f} - ValLoss:{:.5f} - lr:{:.5f}'
                    .format(idx_epoch, epochs, time.time() - epochstart, train_acc_epoch,
                            valid_acc, train_loss_epoch, valid_loss, optimizer.param_groups[0]['lr']))

    except KeyboardInterrupt:
        print('*' * shutil.get_terminal_size().columns)
        print('Exiting from training early')

    if plot_loss_flag:
        plot_loss(avg_train_loss, avg_valid_loss, criterion._get_name(), network, kind='loss', saver=saver,
                  early_stop=0)
        plot_loss(avg_train_acc, avg_valid_acc, criterion._get_name(), network, kind='accuracy', saver=saver,
                  early_stop=0)

    return model, latent_constraint


def eval_model(model, loader, criterion, latent_constraint, abg=(1,1,1)):
    alpha, beta, gamma = abg
    loss_ae = nn.MSELoss(reduction='mean')
    losses = []
    accs = []

    with torch.no_grad():
        model.eval()
        latent_constraint.eval()
        for data in loader:
            # get the inputs
            inputs, target = data  # input shape must be (BS, C, L)
            inputs = Variable(inputs.float()).to(device)
            target = Variable(target.long()).to(device)
            batch_size = inputs.size(0)

            out_ae, out, embedding = model(inputs)
            ypred = torch.max(F.softmax(out, dim=1), dim=1)[1]

            loss_recons_ = loss_ae(out_ae, inputs)
            loss_class_ = criterion(out, target)
            loss_cntrs_ = latent_constraint(embedding.squeeze(), target)
            loss = beta*loss_class_.mean() + gamma*loss_cntrs_.mean() + alpha*loss_recons_

            losses.append(loss.data.item())

            accs.append((ypred == target).sum().item() / batch_size)

    return np.array(losses).mean(), 100 * np.average(accs)


def evaluate_model_multi(model, dataloder, y_true, x_true,
                         metrics=('mae', 'mse', 'rmse', 'std_ae', 'smape', 'rae', 'mbrae', 'corr', 'r2')):
    xhat, yhat = predict_multi(model, dataloder)

    # Classification
    y_hat_proba = softmax(yhat, axis=1)
    y_hat_labels = np.argmax(y_hat_proba, axis=1)

    accuracy = accuracy_score(y_true, y_hat_labels)
    f1_weighted = f1_score(y_true, y_hat_labels, average='weighted')

    cm = confusion_matrix(y_true, y_hat_labels)
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    report = classification_report(y_true, y_hat_labels)
    print(report)

    # AE
    residual = xhat - x_true
    results = evaluate_multi(actual=x_true, predicted=xhat, metrics=metrics)

    return report, y_hat_proba, y_hat_labels, accuracy, f1_weighted, xhat, residual, results



def plot_cm(cm, network='Net', title_str='', saver=None):
    classes = cm.shape[0]
    acc = np.diag(cm).sum() / cm.sum()
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(4 + classes / 2.5, 4 + classes / 2.5))
    sns.heatmap(cm_norm, annot=None, cmap=plt.cm.YlGnBu, cbar=False, ax=ax, linecolor='black', linewidths=0)
    ax.set(title=f'Model:{network} - Accuracy:{100 * acc:.1f}% - {title_str}',
           ylabel='Confusion Matrix (Predicted / True)',
           xlabel=None)
    # ax.set_ylim([1.5, -0.5])
    thresh = cm_norm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j + 0.5, i + 0.5, '%d (%.2f)' % (cm[i, j], cm_norm[i, j]),
                    ha="center", va="center",
                    color="white" if cm_norm[i, j] > thresh else "black")

    fig.tight_layout()

    if saver:
        saver.save_fig(fig, f'CM_{title_str}')


def plot_pred_labels(y_true, y_hat, accuracy, residuals=None, dataset='Train', saver=None):
    # TODO: add more metrics
    # Plot data as timeseries
    gridspec_kw = {'width_ratios': [1], 'height_ratios': [3, 1]}

    if residuals is not None:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10), sharex='all', gridspec_kw=gridspec_kw)

        ax2.plot(residuals ** 2, marker='o', color='red', label='Squared Residual Error', alpha=0.5, markersize='2')
        # ax2.set_ylim([0, 1])
        ax2.grid(True)
        ax2.legend(loc=1)
    else:
        fig, ax1 = plt.subplots(1, 1, figsize=(15, 10))

    ax1.plot(y_true.ravel(), linestyle='-', marker='o', color='black', label='True', markersize='2')
    ax1.plot(y_hat.ravel(), linestyle='--', marker='o', color='red', label='Prediction', alpha=0.5,
             markersize='2')
    ax1.set_title('%s data: top1 acc: %.4f' % (dataset, accuracy))
    ax1.legend(loc=1)

    fig.tight_layout()
    saver.save_fig(fig, name='%s series' % dataset)


def plot_results(data, keys, saver, x='losses', hue='correct', col='noise', kind='box', style='whitegrid', title=None):
    sns.set_style(style)
    n = len(keys)

    for k in keys:
        g = sns.catplot(x=x, y=k, hue=hue, col=col, data=data, kind=kind)
        g.set(ylim=(0, 1))
        if title is not None:
            g.fig.subplots_adjust(top=0.9)
            g.fig.suptitle('{} - {}'.format(k, title))
        saver.save_fig(g.fig, '{}_{}'.format(kind, k))


def evaluate_model(model, dataloder, y_true):
    # TODO: use dict for metrics
    yhat = predict(model, dataloder)

    # Classification
    y_hat_proba = softmax(yhat, axis=1)
    y_hat_labels = np.argmax(y_hat_proba, axis=1)

    accuracy = accuracy_score(y_true, y_hat_labels)
    f1_weighted = f1_score(y_true, y_hat_labels, average='weighted')

    cm = confusion_matrix(y_true, y_hat_labels)
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    report = classification_report(y_true, y_hat_labels)
    print(report)

    return report, y_hat_proba, y_hat_labels, accuracy, f1_weighted


def evaluate_class_recons(model, x, Y, dataloader, saver, network='Model', datatype='Train',
                          plt_cm=True, plt_lables=True, plt_recons=True):
    print(f'{datatype} score')
    results_dict = dict()
    title_str = f'{datatype}'

    results, yhat_proba, yhat, acc, f1, recons, _, ae_results = evaluate_model_multi(model, dataloader, Y, x)

    if plt_cm:
        plot_cm(confusion_matrix(Y, yhat), network=network,     title_str=title_str, saver=saver)
    if plt_lables:
        plot_pred_labels(Y, yhat, acc, residuals=None, dataset=f'{datatype}', saver=saver)
    if plt_recons:
        plot_prediction(x, recons, nrows=5, ncols=5, figsize=(19.2, 10.80), saver=saver,
                        title=f'{datatype} data: mse:%.4f rmse:%.4f corr:%.4f R2:%.4f' % (
                            ae_results['mse'], ae_results['rmse'],
                            ae_results['corr'], ae_results['r2']), figname=f'AE_{datatype}')

    results_dict['acc'] = acc
    results_dict['f1_weighted'] = f1
    return results_dict



