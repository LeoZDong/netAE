"""
Module for helper functions.

@author: Leo Zhengyang Dong
@contact: leozdong@stanford.edu
@date: 10/16/2019
"""

import random
random.seed(0)

import torch
import numpy as np

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def gen_idx_byclass(labels):
    """
    Neatly organize indices of labeled samples by their classes.

    Parameters
    ----------
    labels : list
        Note that labels should be a simple Python list instead of a tensor.

    Returns
    -------
    idx_byclass : dictionary {[class_label (int) : indices (list)]}
    """
    # print("in gen_idx_byclass...")
    from collections import Counter
    classes = Counter(labels).keys()  # obtain a list of classes
    idx_byclass = {}

    for class_label in classes:
        # Find samples of this class:
        class_idx = []  # indices for samples that belong to this class
        for idx in range(len(labels)):
            if labels[idx] == class_label:
                class_idx.append(idx)
        idx_byclass[class_label] = class_idx

    return idx_byclass

def split_train_val(data, labels=None, train_ratio=0.8):
    """
    For splitting the dataset into training and validation sets.

    Parameters
    ----------
    data : tensor (nsamples, nfeatures)

    Returns
    -------
    tuple (training set, validation set)
    """
    if labels is not None:
        # for labeled samples
        return split_train_val_byclass(data, labels, train_ratio)

    # print("in split_train_val...")
    nsamples = data.size()[0]
    idx = list(range(nsamples))
    random.shuffle(idx)

    train_idx = idx[0 : int(train_ratio * nsamples)]
    val_idx = idx[int(train_ratio * nsamples) : ]

    train_data = data[train_idx, : ]
    val_data = data[val_idx, : ]

    return (train_data, val_data)


def split_train_val_byclass(data, labels, train_ratio):
    """
    For splitting the dataset into training and validation sets.

    This is special for the labeled set because samples of different classes
    are first split separately and then combined to ensure that the resulting
    training and validation sets contain samples of all classes.

    Parameters
    ----------
    data : tensor (nsamples, nfeatures)
    labels : list
        Note that labels should be a simple Python list instead of a tensor.

    Returns
    -------
    tuple (training set, training set labels, validation set, validation set labels)
    """

    idx_byclass = gen_idx_byclass(labels)
    train_idx = []
    train_labels = []
    val_idx = []
    val_labels = []

    for class_label in idx_byclass:
        # Process for this class:
        idx_thisclass = idx_byclass[class_label]
        random.shuffle(idx_thisclass)
        train_idx_thisclass = idx_thisclass[0 : int(train_ratio * len(idx_thisclass))]
        val_idx_thisclass = idx_thisclass[int(train_ratio * len(idx_thisclass)) : ]

        # Append:
        train_idx += train_idx_thisclass
        val_idx += val_idx_thisclass
        train_labels += [class_label] * len(train_idx_thisclass)
        val_labels += [class_label] * len(val_idx_thisclass)

    train_data = data[train_idx, :]
    val_data = data[val_idx, :]

    # Note that the labels are passed in as a Python list.
    # Here we turn it into a tensor and reshape as a column vector.
    train_labels = torch.tensor(train_labels, dtype=torch.long).to(device)
    val_labels = torch.tensor(val_labels, dtype=torch.long).to(device)

    return (train_data, train_labels, val_data, val_labels)


def gen_batch_data(batch_i, nbatches, train_idx, train_data):
    """ Generate the batch data of a given batch index.
    The labeled set does not go through batching, so we do not need to
    worry about separating into classes here.

    Parameters
    ----------
    batch_i : int
        Batch index.
    nbatches : int
        Number of batches.
    train_idx : list
        Shuffled train_idx at a certain epoch.
    train_data : ndarray(nsamples, nfeatures)
        Training data.

    Returns
    -------
    batch_data : tensor (nsamples, nfeatures)
        Data for this batch.
    """
    # print("in gen_batch_data...")
    # obtain batch indices for the given batch index:
    batch_size = int(float(len(train_idx)) / nbatches)

    if batch_i < nbatches - 1:
        batch_idx = torch.LongTensor(train_idx[(batch_i*batch_size) : ((batch_i+1)*batch_size)])
    else:  # if processing the last batch
        batch_idx = torch.LongTensor(train_idx[(batch_i*batch_size):])

    # obtain data and labels of this mini-batch
    batch_data = train_data[batch_idx, :]
    return batch_data


def partition_labeled_data(data, labels):
    """ Partition the labeled dataset with more than 2 labels into groups of binary sets.
    e.g. Dataset with labels (0, 1, 2, 3) is partitioned into (0, (1, 2, 3) as 1), (1 as 0, (2, 3) as 1), (2 as 0, 3 as 1)
    This procedure allows for easy calculation of modularity for binary classes. See the Newman paper for details.

    Parameters
    ----------
    data : tensor (nsamples, nfeatures)
    labels : list

    Returns
    -------
    partitions : list of lists [partition1, partition2, partition3, ...]
        partition1 = [data of partition1, labels of partition1]
    """
    # print("in partition_labeled_data...")
    # convert labels from tensor to list for easy processing
    # labels = torch.reshape(labels, (1, -1)).cpu().numpy().tolist()[0]

    partitions = []  # initialize

    idx_byclass = gen_idx_byclass(labels)

    class_labels = list(idx_byclass.keys())
    for i in range(0, len(class_labels)-1):  # i from the first class to the second to last class
        # obtain indices of this class
        this_class_idx = idx_byclass[class_labels[i]]

        # obtain indices of other classes after this class
        other_classes_idx = []
        for j in range(i+1, len(class_labels)):
            other_classes_idx += idx_byclass[class_labels[j]]

        # obtain desired samples
        data_this_class = data[this_class_idx, :]
        labels_this_class = [0] * len(this_class_idx)

        data_other_classes = data[other_classes_idx, :]
        labels_other_classes = [1] * len(other_classes_idx)

        # data and labels of this partition
        data_partition = torch.cat((data_this_class, data_other_classes), 0)
        labels_partition = labels_this_class + labels_other_classes

        # wrap labels_partition into tensor and reshape
        labels_partition = torch.reshape(torch.tensor(labels_partition, dtype=torch.float).to(device), (-1, 1))

        # Add to partitions
        partitions.append((data_partition, labels_partition))

    return partitions


def log_metrics(metrics_dict, epoch, prnt=False):
    """
    Log all metrics in metrics_dict to file.
    """
    # clear log files in the beginning
    if epoch == 0:
        for name in list(metrics_dict.keys()):
            open("logs/{}.txt".format(name), "w+").close()

    for name in list(metrics_dict.keys()):
        with open("logs/{}.txt".format(name), "a") as log:
            line = "{},{}\n".format(epoch, metrics_dict[name])
            log.write(line)
        if prnt:
            print('{{"metric": "{}", "value": {}, "epoch": {}}}'.format(name, metrics_dict[name], epoch))
