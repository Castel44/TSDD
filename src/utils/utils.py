import argparse
import argparse
import os
import shutil
import time

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy.special import softmax
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, \
    classification_report
from torch.autograd import Variable

import src.utils.plotting_utils as plt

######################################################################################################################
columns = shutil.get_terminal_size().columns
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


#####################################################################################################################
def remove_empty_dir(path):
    try:
        os.rmdir(path)
    except OSError:
        pass


def remove_empty_dirs(path):
    for root, dirnames, filenames in os.walk(path, topdown=False):
        for dirname in dirnames:
            remove_empty_dir(os.path.realpath(os.path.join(root, dirname)))


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


def readable(num):
    for unit in ['', 'k', 'M']:
        if abs(num) < 1e3:
            return "%3.3f%s" % (num, unit)
        num /= 1e3
    return "%.1f%s" % (num, 'B')


def append_results_dict(main, sub):
    for k, v in zip(sub.keys(), sub.values()):
        main[k].append(v)
    return main


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


def linear_interp(a, b, alpha):
    return a * alpha + (1 - alpha) * b


def train_model(model, train_data, valid_data, epochs, optimizer, criterion, latent_constraint=None, scheduler=None,
                plot_loss_flag=True, saver=None):
    network = model.get_name()

    print('-' * shutil.get_terminal_size().columns)
    s = 'TRAINING MODEL {} WITH {} LOSS'.format(network, criterion._get_name())
    print(s)
    print('-' * shutil.get_terminal_size().columns)

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

            # Train
            model.train()
            if latent_constraint is not None:
                latent_constraint.train()

            for data, target in train_data:
                target = target.to(device)
                data = data.to(device)
                batch_size = data.size(0)

                # Forward
                optimizer.zero_grad()
                out_class = model(data)

                loss_class_ = criterion(out_class, target)

                if latent_constraint is not None:
                    embedding = model.get_embedding(data).squeeze()
                    loss_cntrs_ = latent_constraint(embedding, target)
                    loss = loss_class_.mean() + loss_cntrs_.mean()
                else:
                    loss = loss_class_.mean()

                loss.backward()
                optimizer.step()

                # Update loss  monitor
                train_loss.append(loss.data.item())

                # Accuracy
                prob = F.softmax(out_class, dim=1)
                train_acc.append((torch.argmax(prob, dim=1) == target).sum().item() / batch_size)

            scheduler.step()
            # Validate
            valid_loss, valid_acc = eval_model(model, valid_data, criterion, latent_constraint)

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
        plt.plot_loss(avg_train_loss, avg_valid_loss, criterion._get_name(), network, kind='loss', saver=saver,
                      early_stop=0)
        plt.plot_loss(avg_train_acc, avg_valid_acc, criterion._get_name(), network, kind='accuracy', saver=saver,
                      early_stop=0)

    return model, latent_constraint


def eval_model(model, loader, criterion, latent_constraint):
    losses = []
    accs = []

    with torch.no_grad():
        model.eval()
        if latent_constraint is not None:
            latent_constraint.eval()
        for data in loader:
            # get the inputs
            inputs, target = data  # input shape must be (BS, C, L)
            inputs = Variable(inputs.float()).to(device)
            target = Variable(target.long()).to(device)
            batch_size = inputs.size(0)

            out = model(inputs)

            loss_class_ = criterion(out, target)

            if latent_constraint is not None:
                embedding = model.get_embedding(inputs).squeeze()
                loss_cntrs_ = latent_constraint(embedding.squeeze(), target)
                loss = loss_class_.mean() + loss_cntrs_.mean()
            else:
                loss = loss_class_.mean()

            losses.append(loss.data.item())

            ypred = torch.max(F.softmax(out, dim=1), dim=1)[1]
            accs.append((ypred == target).sum().item() / batch_size)

    return np.array(losses).mean(), 100 * np.average(accs)


def evaluate_model(model, dataloder, y_true, datatype='Train', network='model', plt_cm=True, plt_pred=True, saver=None):
    print(f'{datatype} score:')
    results_dict = dict()

    yhat = predict(model, dataloder)

    y_hat_proba = softmax(yhat, axis=1)
    y_hat_labels = np.argmax(y_hat_proba, axis=1)

    accuracy = accuracy_score(y_true, y_hat_labels)
    f1_weighted = f1_score(y_true, y_hat_labels, average='weighted')

    report = classification_report(y_true, y_hat_labels)
    print(report)

    if plt_cm:
        plt.plot_cm(confusion_matrix(y_true, y_hat_labels), network=network, title_str=f'{datatype}', saver=saver)
    if plt_pred:
        plt.plot_pred_labels(y_true, y_hat_labels, accuracy, residuals=None, dataset=f'{datatype}', saver=saver)

    results_dict['acc'] = accuracy
    results_dict['f1_weighted'] = f1_weighted
    return results_dict


def predict(model, test_data):
    prediction = []
    with torch.no_grad():
        model.eval()
        for data in test_data:
            data = data[0]
            data = data.float().to(device)
            output = model(data)
            prediction.append(output.cpu().numpy())

    prediction = np.concatenate(prediction, axis=0)
    return prediction
