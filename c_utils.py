"""Utilities for Neural Datathon competition."""

import os
import json
import shutil
import datetime
import numpy as np
from matplotlib import pyplot as plt
from scipy.spatial import distance


def zscore(x, chunks=1, mu=None, sd=None):
    """Zscore x splitting it into chunks-sized events."""
    def apply_norm(x, mu, sd):
        """Apply a z-score operation."""
        if mu is None:
            mu = x.mean(0)
        if sd is None:
            sd = x.mean(0)
        return (x - mu) / sd
    nc = x.shape[0] // chunks
    x_idx = np.arange(chunks).repeat(nc)
    x = x[:len(x_idx), :]
    output = []
    for idx in range(chunks):
        it_x = x[x_idx == idx, :]
        output += [apply_norm(it_x, mu, sd)]
    return np.concatenate(output)


def split_kernels(x, splits, num_weights):
    """Split x into splits number of num_weights-sized matrices."""
    out = []
    for idx in range(splits):
        ni = idx + 1
        out += [x[:, idx * num_weights: ni * num_weights]]
    return out


def plot_weights(xs, ls, save_plot=None):
    """Plot multiple weight distance matrices."""
    f, axs = plt.subplots(len(ls))
    plt.suptitle('Forward LSTM weight matrix similarity.')
    for ax, x, l in zip(axs, xs, ls):
        ax.imshow(x)
        ax.set_title(l)
    if save_plot is not None:
        plt.savefig(save_plot)
    plt.show()
    plt.close(f)


def sq_dist(x, metric='correlation'):
    """Wrapper for squareform vector distances."""
    return distance.squareform(distance.pdist(x, metric=metric))


def delay_mat(x, delay):
    """Create a delayed activity matrix for RNN training."""
    future_x = np.zeros((x.shape[0], delay, x.shape[-1]))
    for idx in range(x.shape[0]):
        if idx > (x.shape[0] - delay):
            it_joints = np.zeros((delay, x.shape[1]))
        else:
            it_joints = x[idx:idx + delay, :]
        future_x[idx] = it_joints
    return future_x


def model_performance(preds, gt, labels):
    """Evaluate model performance with correlation."""
    performance = [np.corrcoef(
        preds[:, idx], gt[:, idx])[0, 1]
        for idx, l in enumerate(labels)]
    print json.dumps(
        ['Corr %s: %s' % (l, x) for l, x in zip(labels, performance)],
        indent=4)
    return performance


def package_test_predictions(team_name, data, create=False):
    """Output a numpy with predictions."""
    fs = datetime.datetime.today().strftime('%m_%d')
    if create:
        fs = 'CREATE_%s' % fs
    else:
        fs = 'OPTIMIZE_%s' % fs
    filename = '%s_%s' % (fs, team_name)
    np.save(filename, data)
    return filename + '.npy'


def savefig(team_name):
    """Output a pdf with a create plot."""
    fs = datetime.datetime.today().strftime('%m_%d')
    filename = 'CREATE_%s_%s.pdf' % (fs, team_name)
    plt.savefig(filename)
    return filename


def movefile_for_eval(
        filename,
        eval_dir=os.path.join(
            '..%ssubmissions' % os.path.sep,
            datetime.datetime.today().strftime('%m_%d'))):
    """Moves files to the evaluation folder."""
    shutil.move(
        filename,
        os.path.join(eval_dir, filename.split(os.path.sep)[-1]))

