"""
Trains a network-enhanced autoencoder (netAE) for semi-supervised dimensionality
reduction of single-cell RNA-sequencing.

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

parser = argparse.ArgumentParser(description="Training netAE")
# Training settings:
parser.add_argument("-dpath", "--data-path", default="/", help="path to the dataset folder")
parser.add_argument("-spath", "--save-path", default="output", help="path to output directory")
parser.add_argument("-mc", "--model-code", default="full", help="code name of the current model (default: full)")
parser.add_argument("-ds", "--dataset", default="cortex", help="name of the dataset (default: cortex)")
parser.add_argument("-sl", "--silence", action="store_true", default=False, help="do not print any stats during training")
parser.add_argument("-se", "--save-embd", type=bool_flag, default=True, help="saves the embedded space along with the model")

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
parser.add_argument("-lratio", "--lab-ratio", type=float, default=-1, help="labeled set ratio for each cell type (default: -1)")

args = parser.parse_args()

# make sure saving path exists
import os
if not os.path.exists(args.save_path):
    os.makedirs(args.save_path)

from netAE import train_netAE

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

    trained_model = train_netAE(data, labeled_data, labeled_lab, model_save_path, embd_save_path, args=args)


if __name__ == "__main__":
    main()
