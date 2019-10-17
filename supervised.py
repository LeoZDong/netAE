"""
Trains a supervised neural network using the encoder structure of netAE with
70% data. Represents the empirical upper bound accuracy.

@author: Leo Zhengyang Dong
@contact: leozdong@stanford.edu
@date: 10/16/2019
"""

#### Setup ####
import numpy as np
import torch
import torch.optim as optim
from sklearn.metrics import accuracy_score
import argparse

from nn_struct import NN_sup

# define device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("device:", device)

#### Argument parser ####
parser = argparse.ArgumentParser(description="Trains a supervised neural network using the encoder structure of netAE.")
parser.add_argument("--use-floyd", action="store_true", default=False, help="use floydhub to train")
parser.add_argument("-ds", "--dataset", default="cortex", help="name of the dataset (default: cortex)")
parser.add_argument("-spath", "--save-path", default="output", help="path to output directory")
parser.add_argument("-mpath", "--model-path", default="/Users/Leo/Research/2. Gene Expression Feature Learning/4. Modeling/Main/trained_models", help="path to trained models")
parser.add_argument("-ld", "--load-trained", action="store_true", default=False, help="load a trained model instead of training a new model")
parser.add_argument("-s", "--seed", type=int, default=0, help="random seed for loading dataset (default: 0)")
parser.add_argument("-lratio", "--lab-ratio", type=float, default=0.7, help="labeled set ratio for each cell type (default: 0.7)")

args = parser.parse_args()

# make sure saving path exists
import os
if not os.path.exists(args.save_path):
    os.makedirs(args.save_path)

def train_sup(data, labels):
    """
    Trains a supervised neural network.

    Parameters
    ----------
    data : tensor (n_samples, n_features)
    labels : list

    Returns
    -------
    trained_nn : model state_dict
    """
    # roughly define hyperparameters
    lr = 1e-5
    z_dim = 50
    epochs = 150
    early_stop = 50
    batch_size = 100

    print("training supervised nn...")
    print("params: {{lr: {}, z_dim: {}, epochs: {}, early_stop: {}, batch_size: {}}}".format(lr, z_dim, epochs, early_stop, batch_size))

    #### Define network structure ####
    input_dim = data.size()[1]
    n_classes = len(set(labels))

    nn = NN_sup(z_dim, input_dim, n_classes)
    print(nn)

    # put data and model into device
    nn = nn.to(device)
    data = data.to(device)
    # Note: labels will be wrapped into tensors when split into train and val, and the new tensors will be put into device when created.

    # create optimizer
    optimizer = optim.Adam(nn.parameters(), lr=lr)
    optimizer.zero_grad()

    # save best performing model
    best_model_state_dict = None
    best_epoch = 0
    best_loss = float("inf")

    #### Process data for training ####
    # split into train and val
    from helper import split_train_val
    train_data, train_labels, val_data, val_labels = split_train_val(data, labels=labels, train_ratio=0.8)

    # wrap training data into pytorch Dataset object for batching
    from data import Dataset
    from torch.utils.data import DataLoader
    train_dataloader = DataLoader(Dataset(train_data, labels=train_labels), batch_size=batch_size, shuffle=True)

    for epoch in range(epochs):
        print("epoch", epoch)

        # initialize training metrics
        train_loss, train_acc = 0, 0
        # useful for averaging and logging training metrics later
        nbatches = int(train_data.size()[0] / batch_size) + 1
        nbatches_left = nbatches

        #### Training phase ####
        nn.train()
        for x, y in train_dataloader:
            nbatches_left -= 1

            # calculate loss
            loss, acc = nn.calc_losses(x, y)

            # update training metric
            train_loss += loss.cpu().detach().numpy()
            train_acc += acc.cpu().detach().numpy()

            if nbatches_left == 0:
                # average over all nbatches
                train_loss /= nbatches
                train_acc /= nbatches

                # print training logs
                print('{{"metric": "Train loss", "value": {}, "epoch": {}}}'.format(train_loss, epoch))
                print('{{"metric": "Train acc", "value": {}, "epoch": {}}}'.format(train_acc, epoch))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        #### Validation phase ####
        with torch.no_grad():
            nn.eval()
            # calculate validation metrics
            val_loss, val_acc = nn.calc_losses(val_data, val_labels)

        # print validation logs
        print('{{"metric": "Val loss", "value": {}, "epoch": {}}}'.format(val_loss, epoch))
        print('{{"metric": "Val acc", "value": {}, "epoch": {}}}'.format(val_acc, epoch))

        # update best performance
        if val_loss < best_loss:
            best_loss = val_loss
            best_epoch = epoch
            best_model_state_dict = nn.state_dict()
            print("Saving new best model at epoch {}".format(epoch))
            torch.save(best_model_state_dict, args.save_path + "/supervised_trained_{}_{}".format(args.dataset, args.seed))

        if epoch - best_epoch > early_stop:
            print("early stopping reached at: {}".format(epoch))
            break

    print("training finished")
    print("best epoch: {}".format(best_epoch))

    return best_model_state_dict

def sup_infer(train_data, train_labels, test_data, test_labels):
    # initialize nn_trained
    nn_trained = NN_sup(50, train_data.size()[1], len(set(train_labels)))
    nn_trained.eval()
    # load state_dict
    if args.load_trained:
        print("load trained model...")
        state_dict = torch.load(args.model_path + "/supervised_trained_{}_{}".format(args.dataset, args.seed), map_location=device)
    else:
        print("train new model...")
        state_dict = train_sup(train_data, train_labels)

    nn_trained.load_state_dict(state_dict)

    # predict training labels
    train_labels = torch.tensor(train_labels, dtype=torch.long)
    train_pred = torch.max(nn_trained(train_data), dim=1)[1]
    train_acc = (train_pred == train_labels).sum().double() / train_labels.size()[0]
    print("training acc: {}".format(train_acc))

    # predict test labels
    test_labels = torch.tensor(test_labels, dtype=torch.long)
    test_pred = torch.max(nn_trained(test_data), dim=1)[1]
    test_acc = (test_pred == test_labels).sum().double() / test_labels.size()[0]
    print("test acc: {}".format(test_acc))

def main():
    # load data
    if args.use_floyd:
        data_path = "/data"
    else:
        data_path = "/Users/Leo/Research/2. Gene Expression Feature Learning/4. Modeling/Main/{}_data".format(args.dataset)

    print("Using dataset: {}".format(args.dataset))

    from data import Data
    prep_method = "log"

    dataset = Data(data_path, labeled_ratio=args.lab_ratio, seed=args.seed, prep_method=prep_method)

    data, lab_full, train_idx, test_idx, info = dataset.load_all()
    train_data = torch.tensor(data[train_idx, :], dtype=torch.float)
    train_labels = lab_full[train_idx]
    test_data = torch.tensor(data[test_idx, :], dtype=torch.float)
    test_labels = lab_full[test_idx]

    sup_infer(train_data, train_labels, test_data, test_labels)

if __name__ == "__main__":
    main()
