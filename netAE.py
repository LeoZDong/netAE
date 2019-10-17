"""
Trains a network-enhanced autoencoder (netAE) for semi-supervised dimensionality
reduction of single-cell RNA-sequencing. One can either train with a cloud service
or locally. We used the cloud service floydhub in our experiments, so the training
metrics are printed according to floydhub's specifications.

Hyperparameters are passed in through argument parser, and the default values
are the hyperparameters we used in the paper.

One can specify the autoencoder architecture (Dae, Vae, Dca, Vdca) for which to
incorporate the semi-supervised idea of netAE, but netAE uses the Vdca
architecture in our paper. The models are named `net{name_of_autoencoder}` if
trained in a semi-supervised fashion as described by the netAE paper, and named
`{name_of_autoencoder}` if trained in an unsupervised fashion but using the
same network architecture (i.e. the `AE` baseline in the paper).

@author: Leo Zhengyang Dong
@contact: leozdong@stanford.edu
@date: 10/16/2019
"""

# =============================================================================
# Setup
# =============================================================================
# import modules and packages
import numpy as np
# import pytorch packages
import torch
import torch.nn as nn
import torch.optim as optim
import argparse


# define device
# make sure to convert model and all input batches to device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("device:", device)

# =============================================================================
# Argument parser
# =============================================================================
parser = argparse.ArgumentParser(description="Training netAE")
# Training settings:
parser.add_argument("-dpath", "--data-path", default="/", help="path to the dataset folder")
parser.add_argument("-spath", "--save-path", default="output", help="path to output directory")
parser.add_argument("-mc", "--model-code", default="full", help="code name of the current model (default: full)")
parser.add_argument("-ds", "--dataset", default="cortex", help="name of the dataset (default: cortex)")
parser.add_argument("-sl", "--silence", action="store_true", default=False, help="do not print any stats during training")
parser.add_argument("-se", "--save_embd", action="store_true", default=False, help="saves the embedded space along with the model")

# Model property:
parser.add_argument("--ae", default="vdca", help="specify autoencoder structure (dae, vae, dca, or vdca)")
parser.add_argument("-s", "--seed", type=int, default=0, help="random seed for loading dataset (default: 0)")

# Model parameters:
parser.add_argument("-opt", "--optim", default="adam", help="name of the optimizer to use (default: adam)")
parser.add_argument("-bs", "--batch-size", type=int, default=100, help="batch size (default: 100)")
parser.add_argument("--lr", type=float, default=2e-4, help="learning rate (default: 2e-4)")
parser.add_argument("-lrd", "--lr-decay", type=float, default=1, help="learning rate decay (default: 1, i.e. no decay)")
parser.add_argument("--dim", type=int, default=50, help="encoding dimension (default: 50)")
parser.add_argument("-e", "--epochs", type=int, default=150, help="number of epochs (default: 100)")
parser.add_argument("-es", "--early-stop", type=int, default=10, help="number of epochs to wait before early stop (default: 10)")
parser.add_argument("-gm", "--gamma", type=float, default=2, help="RBF coefficient in calculating similarity in modularity (default: 2)")
parser.add_argument("-kp", "--kappa", type=float, default=5, help="weight for KL divergence term; only active when using variational autoencoder (default: 5)")
parser.add_argument("-ld", "--lambd", type=float, default=150, help="weight for modularity loss (default: 150)")
parser.add_argument("--phi", type=float, default=10, help="weight for logistic regression loss (default: 10)")

parser.add_argument("-lsize", "--lab-size", type=int, default=10, help="labeled set size for each cell type (default: 10)")
parser.add_argument("-lratio", "--lab-ratio", type=float, default=-1, help="labeled set size for each cell type (default: 10)")

args = parser.parse_args()

# make sure saving path exists
import os
if not os.path.exists(args.save_path):
    os.makedirs(args.save_path)

# =============================================================================
# Helper functions and network structures
# =============================================================================
from helper import *
from nn_struct import *

# =============================================================================
# Training
# =============================================================================
def train_netAE(all_data, labeled_data, labeled_lab, params=None, save_path=None, embd_save_path=None):
    """ Trains netAE for semi-supervised learning. AE structure can be either DAE or VAE.
    The original dataset (mostly unlabeled) is given by all_data.
    A small portion of the dataset that is labeled is given by labeled_data.

    Parameters
    ----------
    all_data : tensor (nsamples, nfeatures)
        Dataset of all samples.
    labeled_data : tensor (nsamples, nfeatures)
        Subset of the dataset that are labeled. Usually a very small subset.
    labeled_lab : list (nsamples)
        Labels of labeled_data.
    params : optional, dictionary {"parameter name" : value}
        Optinal dictionary of parameters to pass into netAE by parameter name.
        Default uses parameters read from argument parser.
    save_path : optional, string
        Optional path for saving the entire model dictionary.
    embd_save_path : optional, string
        Optional path for saving the embedded space of the trained model instead
        of the entire model. This significantly saves more space than the entire
        model dictionary. This is sufficient if one does not need to use the trained
        model later but only cares about the latent features.

    Returns
    -------
    best_model_state_dict : object
        Best-performing model at a certain epoch in terms of validation loss.
        Saved as the pytorch state dictionary.
    """

    #### Extract hyperparameters ####
    if params is not None:  # read hyperparameters from a dictionary passed in as a parameter
        lambd = params["lambd"] if "lambd" in params.keys() else args.lambd  # modularity loss weight
        phi = params["phi"] if "phi" in params.keys() else args.phi  # logistic regression loss weight
        kappa = params["kappa"] if "kappa" in params.keys() else args.kappa  # KL divergence weight
        gamma = params["gamma"] if "gamma" in params.keys() else args.gamma  # RBF coefficient

        optimizer = params["optim"] if "optimizer" in params.keys() else args.optim
        nbatches = params["batch_size"] if "batch_size" in params.keys() else args.batch_size
        lr = params["lr"] if "lr" in params.keys() else args.lr
        lr_decay = params["lr_decay"] if "lr_decay" in params.keys() else args.lr_decay
        encoding_dim = params["encoding_dim"] if "encoding_dim" in params.keys() else args.dim
        epochs = params["epochs"] if "epochs" in params.keys() else args.epochs
        early_stop = params["early_stop"] if "early_stop" in params.keys() else args.early_stop
    else:
        lambd = args.lambd
        phi = args.phi
        kappa = args.kappa
        gamma = args.gamma

        optimizer = args.optim
        batch_size = args.batch_size
        lr = args.lr
        lr_decay = args.lr_decay
        encoding_dim = args.dim
        epochs = args.epochs
        early_stop = args.early_stop

    # print hyperparameters
    print("params: {{lambda: {}, phi: {}, kappa: {}, gamma: {}, opt: {}, batch_size: {}, lr: {}, lr_decay: {}, encoding_dim: {}, epochs: {}, early_stop: {}}}".format(lambd, phi, kappa, gamma, optimizer, batch_size, lr, lr_decay, encoding_dim, epochs, early_stop))

    #### Define autoencoder structure ####
    input_dim = all_data.size()[1]
    n_classes = len(set(labeled_lab))

    if args.ae == "dae":
        netAE = Dae(encoding_dim, input_dim, n_classes, lambd=lambd, phi=phi, gamma=gamma)
        print("Training DAE...")
    elif args.ae == "vae":
        netAE = Vae(encoding_dim, input_dim, n_classes, lambd=lambd, phi=phi, kappa=kappa, gamma=gamma)
        print("Training VAE...")
    elif args.ae == "dca":
        netAE = Dca(encoding_dim, input_dim, n_classes, lambd=lambd, phi=phi, gamma=gamma)
        print("Training DCA...")
    elif args.ae == "vdca":
        netAE = Vdca(encoding_dim, input_dim, n_classes, lambd=lambd, phi=phi, kappa=kappa, gamma=gamma)
        print("Training VDCA...")
    else:
        raise Exception("Incorrect autoencoder structure name passed to argument parser! Enter dae, vae, dca, or vcda.")

    # put model and input to device
    netAE.to(device)
    all_data, labeled_data = all_data.to(device), labeled_data.to(device)

    print(netAE)

    # create optimizer and learning rate scheduler
    if optimizer == "adam":
        optimizer = optim.Adam(netAE.parameters(), lr=lr)
    elif optimizer == "sgd":
        optimizer = optim.SGD(netAE.parameters(), lr=lr, momentum=0.9, nesterov=True)
    elif optimizer == "adadelta":
        optimizer = optim.Adadelta(netAE.parameters(), lr=lr)
    else:
        raise Exception("Enter correct optimizer name!")
    optimizer.zero_grad()

    if lr_decay != 1:
        opt_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=lr_decay,\
                        patience=5, verbose=True, min_lr=1e-5, threshold=0.5, threshold_mode="abs")

    # save the best performing netAE
    best_model_state_dict = None
    latest_smoothed_model = None  # model at the latest smoothed epoch
    best_epoch = 0  # for early stopping
    best_unsup_loss = float("inf")


    # keep track of val loss and smoothed val loss for early stopping
    raw_val_unsup_rec = []
    smoothed_val_unsup_rec = []

    # a dictionary {epoch : model_state_dict} because
    # used sometimes when the best model needs to be saved, it is not at the current epoch
    model_state_rec = {}

    #### Split into training and validation sets ####
    # for the unsupervised part, split into train and validation sets
    train_data, val_data = split_train_val(all_data)
    # for the supervised part, use all of labeled data as train and no validation
    train_labeled_data, train_labels = labeled_data, labeled_lab

    # Partition the labeled data for calculating modularity
    # (from one set of multi-class to multiple sets of binary-class)
    train_labeled_data_sets = partition_labeled_data(train_labeled_data, train_labels)

    # wrap training data into pytorch Dataset object for batching
    from data import Dataset
    from torch.utils.data import DataLoader
    train_dataloader = DataLoader(Dataset(train_data), batch_size=batch_size, shuffle=True)

    for epoch in range(epochs):
        if not args.silence:
            print("epoch", epoch)

        # initialize training metrics
        train_total_loss, train_unsup_loss, train_softmax_acc, train_mod = 0, 0, 0, 0
        if args.ae == "vae" or args.ae == "vdca":
            # reconstruction loss and kl divergence
            # for DAE and DCA, train_unsup_loss = train_rl, so no need to specify train_rl
            train_rl, train_kld = 0, 0

        # useful for averaging and logging training metrics
        nbatches = int(train_data.size()[0] / batch_size) + 1
        nbatches_left = nbatches

        #### Training phase ####
        netAE.train()
        for inputs in train_dataloader:
            nbatches_left -= 1

            # calculate losses
            losses = netAE.calc_losses(inputs, train_labeled_data_sets, train_labeled_data, torch.LongTensor(train_labels).to(device))

            # update training metrics
            train_total_loss += losses[0].cpu().detach().numpy()
            train_unsup_loss += losses[1].cpu().detach().numpy()
            train_mod += losses[2].cpu().detach().numpy()
            train_softmax_acc += losses[-1].cpu().detach().numpy()

            if args.ae == "vae" or args.ae == "vdca":
                # rl = reconstruction loss
                # for variational models, unsup_loss = rl + kld, so they need to be stored separately
                train_rl += losses[3].cpu().detach().numpy()
                train_kld += losses[4].cpu().detach().numpy()

            # log training metrics at the last batch
            if nbatches_left == 0:
                train_metrics = {}

                # average training metrics over all batches
                # save metrics to log
                train_total_loss /= nbatches
                train_metrics["train_total_loss"] = train_total_loss
                train_unsup_loss /= nbatches
                train_metrics["train_unsup_loss"] = train_unsup_loss
                train_mod /= nbatches
                train_metrics["train_mod"] = train_mod
                train_softmax_acc /= nbatches
                train_metrics["train_softmax_acc"] = train_softmax_acc
                if args.ae == "vae" or args.ae == "vdca":
                    train_rl /= nbatches
                    train_metrics["train_rl"] = train_rl
                    train_kld /= nbatches
                    train_metrics["train_kld"] = train_kld

                # Write training logs to file:
                log_metrics(train_metrics, epoch, prnt=(False if args.silence else True))

            #### Backward step ####
            optimizer.zero_grad()
            losses[0].backward()

            #### Update weights ####
            optimizer.step()


        #### Validation phase ####
        # Calculate validation loss:
        with torch.no_grad():
            netAE.eval()
            # calculate losse
            val_losses = netAE.calc_losses(val_data, None, None, None, unsup=True)

        # log validation metrics
        val_metrics = {}
        val_unsup_loss = val_losses[1].cpu().detach().numpy()
        val_metrics["val_unsup_loss"] = val_unsup_loss
        if args.ae == "vae" or args.ae == "vdca":
            val_rl = val_losses[3].cpu().detach().numpy()
            val_metrics["val_rl"] = val_rl
            val_kld = val_losses[4].cpu().detach().numpy()
            val_metrics["val_kld"] = val_kld

        # update learning rate based on validation loss
        if lr_decay != 1:
            if args.phi != 0 and args.lambd != 0:
                opt_scheduler.step(train_total_loss-train_unsup_loss)
            else:
                opt_scheduler.step(val_unsup_loss)

        # smooth val_unsup_loss
        raw_val_unsup_rec.append(val_unsup_loss)
        back = round(0.9 * len(smoothed_val_unsup_rec))
        forward = round(1.1 * len(smoothed_val_unsup_rec))
        if forward <= len(raw_val_unsup_rec)-1:
            window = raw_val_unsup_rec[back : (forward + 1)]
            smoothed = sum(window) / len(window)
            smoothed_val_unsup_rec.append(smoothed)
            latest_smoothed_model = netAE.state_dict()
            log_metrics({"smoothed_val_unsup_loss":smoothed_val_unsup_rec[-1]}, len(smoothed_val_unsup_rec)-1, prnt=(False if args.silence else True))

        # Write validation logs to file:
        log_metrics(val_metrics, epoch, prnt=(False if args.silence else True))

        if best_unsup_loss - smoothed_val_unsup_rec[-1] > 1:
            best_unsup_loss = smoothed_val_unsup_rec[-1]
            best_epoch = len(smoothed_val_unsup_rec) - 1
            if save_path is not None:
                best_model_state_dict = latest_smoothed_model
                print("Saving new best model at epoch {}".format(best_epoch))
                print("(evaluated with smoothed val unsupervised loss)")
                torch.save(best_model_state_dict, save_path)
                if args.save_embd:
                    print("Also saving embedded space")
                    np.save(embd_save_path, netAE.get_z(all_data).cpu().detach().numpy())


        # Early stopping:
        lookahead = len(smoothed_val_unsup_rec) - 1
        lookback = min(round(0.8 * lookahead), max(0, lookahead-2))
        threshold = 2
        if lookahead != lookback and smoothed_val_unsup_rec[lookback] - smoothed_val_unsup_rec[lookahead] < threshold:
            print("early stopping (unsup based) reached at: {}".format(lookahead))
            break

        if args.lambd != 0 and args.phi != 0:  # early stopping for semi-supervised
            if train_softmax_acc > 0.97:
                print("early stopping (acc based) reached at: {}".format(epoch))
                best_model_state_dict = netAE.state_dict()
                best_epoch = epoch
                # save model at the stopping epoch
                # only for semi-supervised early stopping, because it is accuracy-based
                if save_path is not None:
                    print("Saving model at stopping epoch {} (accuracy-based)".format(epoch))
                    torch.save(best_model_state_dict, save_path)
                    if args.save_embd:
                        print("Also saving embedded space")
                        np.save(embd_save_path, netAE.get_z(all_data).cpu().detach().numpy())
                break


    # Print best netAE results:
    print("training finished")
    print("best epoch:", best_epoch)

    return best_model_state_dict


def main():
    if args.lambd == 0:
        print("Training without modularity loss!")
    if args.phi == 0:
        print("Training without logistic regression loss!")

    # Data loading
    from data import Data
    prep_method = "log"

    if args.lab_ratio != -1:  # use labeled ratio
        dataset = Data(args.data_path, labeled_ratio=args.lab_ratio, seed=args.seed, prep_method=prep_method)
    else:  # use labeled size
        dataset = Data(args.data_path, labeled_size=args.lab_size, seed=args.seed, prep_method=prep_method)
    # info contains information about the dataset (gene names, cell names, etc.)
    # it is not used in the program but one can take advantage of this data if needed
    data, lab_full, labeled_idx, unlabeled_idx, info = dataset.load_all()

    data = torch.tensor(data, dtype=torch.float)
    labeled_data = data[labeled_idx, :]
    labeled_lab = lab_full[labeled_idx].tolist()

    embd_save_path = None
    if args.lambd == 0 and args.phi == 0:
        model_save_path = args.save_path + "/{}_trained_{}.pt".format(args.ae.upper(), args.dataset)
        if args.save_embd:
            embd_save_path = args.save_path + "/{}_embd_space_{}.npy".format(args.ae.upper(), args.dataset)
    else:
        model_save_path = args.save_path + "/net{}_trained_{}_{}_{}.pt".format(args.ae.upper(), args.dataset, args.model_code, args.seed)
        if args.save_embd:
            embd_save_path = args.save_path + "/net{}_embd_space_{}_{}_{}.npy".format(args.ae.upper(), args.dataset, args.model_code, args.seed)

    trained_model = train_netAE(data, labeled_data, labeled_lab, save_path=model_save_path, embd_save_path=embd_save_path)


if __name__ == "__main__":
    main()
