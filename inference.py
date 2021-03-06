"""
Inference module after model has been trained.
Module for predicting cell labels using naive classifiers (KNN and logistic
regression) using the latent features learned by different dimensionality reduction
methods.

Specifically, we compare the features learned by netAE to the baselines of AE
(netAE with only unsupervised losses), scVI, ZIFA, and PCA. One should have already
trained the respective baselines and store their learned features as a numpy array.

@author: Leo Zhengyang Dong
@contact: leozdong@stanford.edu
@date: 10/16/2019
"""

#### Setup ####
# import modules and packages
from helper import gen_idx_byclass
import numpy as np
import torch
from sklearn.metrics import accuracy_score
import argparse

# define device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("device:", device)


#### Argument parser ####
parser = argparse.ArgumentParser(description="Predict cell types on latent features learned by different dimensionality reduction models.")
parser.add_argument("-dpath", "--data-path", default="/", help="path to the dataset folder")
parser.add_argument("-mpath", "--model-path", default="/Users/Leo/Research/2. Gene Expression Feature Learning/4. Modeling/Main/trained_models", help="path to trained models")
parser.add_argument("-ds", "--dataset", default="cortex", help="name of the dataset (default: cortex)")
parser.add_argument("-mc", "--model-code", default="full", help="code name of the model (e.g. xmod, xlog, xnb, xkl for ablation study)")
parser.add_argument("-s", "--seed", type=int, default=0, help="random seed for loading dataset (default: 0)")
parser.add_argument("-lsize", "--lab-size", type=int, default=10, help="labeled set size for each cell type (default: 10)")
parser.add_argument("-lratio", "--lab-ratio", type=float, default=-1, help="labeled set ratio for each cell type (default: -1, i.e. does not use lratio by default)")

args = parser.parse_args()


def knn_infer(embd_space, labeled_idx, labeled_lab, unlabeled_idx):
	"""
	Predicts the labels of unlabeled data in the embedded space with KNN.

	Parameters
	----------
	embd_space : ndarray (n_samples, embedding_dim)
		Each sample is described by the features in the embedded space.
		Contains all samples, both labeled and unlabeled.
	labeled_idx : list
		Indices of the labeled samples (used for training the classifier).
	labeled_lab : ndarray (n_labeled_samples)
		Labels of the labeled samples.
	unlabeled_idx : list
		Indices of the unlabeled samples.

	Returns
	-------
	pred_lab : ndarray (n_unlabeled_samples)
		Inferred labels of the unlabeled samples.
	"""

	# obtain labeled data and unlabled data from indices
	labeled_samp = embd_space[labeled_idx, :]
	unlabeled_samp = embd_space[unlabeled_idx, :]

	from sklearn.neighbors import KNeighborsClassifier

	knn = KNeighborsClassifier(n_neighbors=10)
	knn.fit(labeled_samp, labeled_lab)

	pred_lab = knn.predict(unlabeled_samp)
	return pred_lab

def log_infer(embd_space, labeled_idx, labeled_lab, unlabeled_idx):
	"""
	Infers the labels of unlabeled data in the embedded space with logistic regression.
	Same parameters as knn_infer.
	"""

	# obtain labeled data and unlabled data from indices
	labeled_samp = embd_space[labeled_idx, :]
	unlabeled_samp = embd_space[unlabeled_idx, :]

	from sklearn.linear_model import LogisticRegression
	log = LogisticRegression(penalty="l2", C=1, multi_class="multinomial", solver="lbfgs", max_iter=1e4)

	log.fit(labeled_samp, labeled_lab)

	pred_lab = log.predict(unlabeled_samp)
	return pred_lab


def main():
	#### Load data and trained models ####
	print("Using dataset: {}".format(args.dataset))

	from data import Data
	prep_method = "log"
	if args.lab_ratio != -1:
		dataset = Data(args.data_path, labeled_ratio=args.lab_ratio, seed=args.seed, prep_method=prep_method)
	else:
		dataset = Data(args.data_path, labeled_size=args.lab_size, seed=args.seed, prep_method=prep_method)
	data, lab_full, labeled_idx, unlabeled_idx, info = dataset.load_all()

	labeled_data = data[labeled_idx, :]
	labeled_lab = lab_full[labeled_idx]
	unlabeled_data = data[unlabeled_idx, :]
	unlabeled_lab = lab_full[unlabeled_idx]

	netAE_embd_space = np.load("{}/netAE_embd_space_{}_{}_{}.npy".format(args.model_path, args.dataset, args.model_code, args.seed))
	print("netAE space:", netAE_embd_space.shape)
	AE_embd_space = np.load("{}/AE_embd_space_{}.npy".format(args.model_path, args.dataset))
	print("AE space:", AE_embd_space.shape)
	scvi_embd_space = np.load("{}/scvi_embd_space_{}.npy".format(args.model_path, args.dataset))
	print("scvi space:", scvi_embd_space.shape)
	pca_embd_space = np.load("{}/pca_embd_space_{}.npy".format(args.model_path, args.dataset))
	print("pca space:", pca_embd_space.shape)
	zifa_embd_space = np.load("{}/zifa_embd_space_{}.npy".format(args.model_path, args.dataset))
	print("zifa space:", zifa_embd_space.shape)

	#### Infer with KNN ####
	netAE_knn_infd_lab = knn_infer(netAE_embd_space, labeled_idx, labeled_lab, unlabeled_idx)
	AE_knn_infd_lab = knn_infer(AE_embd_space, labeled_idx, labeled_lab, unlabeled_idx)
	scvi_knn_infd_lab = knn_infer(scvi_embd_space, labeled_idx, labeled_lab, unlabeled_idx)
	pca_knn_infd_lab = knn_infer(pca_embd_space, labeled_idx, labeled_lab, unlabeled_idx)
	zifa_knn_infd_lab = knn_infer(zifa_embd_space, labeled_idx, labeled_lab, unlabeled_idx)
	original_knn_infd_lab = knn_infer(data, labeled_idx, labeled_lab, unlabeled_idx)

	netAE_knn_acc = accuracy_score(unlabeled_lab, netAE_knn_infd_lab)
	AE_knn_acc = accuracy_score(unlabeled_lab, AE_knn_infd_lab)
	scvi_knn_acc = accuracy_score(unlabeled_lab, scvi_knn_infd_lab)
	pca_knn_acc = accuracy_score(unlabeled_lab, pca_knn_infd_lab)
	zifa_knn_acc = accuracy_score(unlabeled_lab, zifa_knn_infd_lab)
	origin_knn_acc = accuracy_score(unlabeled_lab, origin_knn_infd_lab)

	print("netAE knn accuracy:", netAE_knn_acc)
	print("AE knn accuracy:", AE_knn_acc)
	print("scvi knn accuracy:", scvi_knn_acc)
	print("pca knn accuracy:", pca_knn_acc)
	print("zifa knn accuracy:", zifa_knn_acc)
	print("original space knn accuracy:", origin_knn_acc)

	#### Infer with logistic regression ####
	netAE_log_infd_lab = log_infer(netAE_embd_space, labeled_idx, labeled_lab, unlabeled_idx)
	AE_log_infd_lab = log_infer(AE_embd_space, labeled_idx, labeled_lab, unlabeled_idx)
	scvi_log_infd_lab = log_infer(scvi_embd_space, labeled_idx, labeled_lab, unlabeled_idx)
	pca_log_infd_lab = log_infer(pca_embd_space, labeled_idx, labeled_lab, unlabeled_idx)
	zifa_log_infd_lab = log_infer(zifa_embd_space, labeled_idx, labeled_lab, unlabeled_idx)
	origin_log_infd_lab = log_infer(data, labeled_idx, labeled_lab, unlabeled_idx)

	netAE_log_acc = accuracy_score(unlabeled_lab, netAE_log_infd_lab)
	AE_log_acc = accuracy_score(unlabeled_lab, AE_log_infd_lab)
	scvi_log_acc = accuracy_score(unlabeled_lab, scvi_log_infd_lab)
	pca_log_acc = accuracy_score(unlabeled_lab, pca_log_infd_lab)
	zifa_log_acc = accuracy_score(unlabeled_lab, zifa_log_infd_lab)
	origin_log_acc = accuracy_score(unlabeled_lab, origin_log_infd_lab)

	print("netAE log accuracy:", netAE_log_acc)
	print("AE log accuracy:", AE_log_acc)
	print("scvi log accuracy:", scvi_log_acc)
	print("pca log accuracy:", pca_log_acc)
	print("zifa log accuracy:", zifa_log_acc)
	print("original space log accuracy:", origin_log_acc)


if __name__ == "__main__":
	main()
