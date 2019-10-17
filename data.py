"""
Handles all datasets loading and preprocessing.

netAE requires all input data to be preprocessed by the log method first.
The other preprocessing methods are never used in the experiments and are for
reference only, if one wishes to experiment.

@author: Leo Zhengyang Dong
@contact: leozdong@stanford.edu
@date: 10/16/2019
"""

import numpy as np
import random

class Data():

	def __init__(self, load_path, labeled_size=None, labeled_ratio=None, seed=0, prep_method=None):
		"""
		Parameters
		----------
		load_path : string
			Data directory.
		labeled_ratio : float
			Ratio of samples to retain labels. The other samples' labels are all hidden.
		labeled_size : int
			Number of samples per class to retain labels.
			Choose to specify either labeled_ratio or labeled_size, but not both.
		seed : int
			Random seed for choosing labeled and unlabeled sets.
		prep_method : string (bucket, clip, log, or None)
			Preprocessing method for the raw expression count.
		"""
		self.seed = seed
		self.dataset = np.load(load_path + "/dataset_matched.npy")
		self.gene_names = np.load(load_path + "/gene_names.npy")
		self.labeled_size = labeled_size
		self.labeled_ratio = labeled_ratio
		self.prep_method = prep_method

	def load_all(self):
		print("loading data...".format(self.seed))

		expr = self.dataset[:, 3:].astype(float)
		lab_full = self.dataset[:, 1].astype(int)
		cell_id = self.dataset[:, 0]

		# create dictionary mapping labels to cell_type
		cell_type = dict(zip(lab_full, self.dataset[:, 2]))

		# preprocess expression data
		if self.prep_method is not None:
			expr = self.preprocess(expr, self.prep_method)

		# manually hide labels
		labeled_idx, unlabeled_idx = self.hide_labs(lab_full)

		# combine information not necessary to training into a separate tuple
		# It is never used in experiments in our paper, but can be useful in
		# some cases.
		info = (cell_type, cell_id, self.gene_names)

		print("expression set dimensions:", expr.shape)
		return expr, lab_full, labeled_idx, unlabeled_idx, info

	def hide_labs(self, lab):
		"""
		Hide a portion of the labels to simulate semi-supervised learning.

		Parameters
		----------
		lab : ndarray (1, nsamples)
			Complete labels of all samples.

		Returns
		-------
		labeled_idx : list
			Indices of the labeled samples.
		unlabeled_idx : list
			Indices of the unlabeled samples.
		"""

		from helper import gen_idx_byclass
		idx_byclass = gen_idx_byclass(lab.tolist())

		# keep track of a list of sample indices whose labels are retained
		labeled_idx = []
		unlabeled_idx = []

		for class_label in idx_byclass:
			print("class: {}, size: {}".format(class_label, len(idx_byclass[class_label])))
			# Process for this class:
			idx_thisclass = idx_byclass[class_label]
			# shuffle with seed
			# this ensures that the labeled set is always the same whenever you obtain data from this module
			random.Random(self.seed).shuffle(idx_thisclass)

			# append indices
			if self.labeled_ratio is not None:
				self.labeled_size = int(self.labeled_ratio * len(idx_thisclass))

			if self.labeled_size >= len(idx_thisclass):
				print("Specified labeled_size is greater than number of samples for class", class_label)
				print("Use all samples of this class instead.")
				labeled_idx += idx_thisclass
			else:
				labeled_idx += idx_thisclass[0:self.labeled_size]
				unlabeled_idx += idx_thisclass[self.labeled_size:]

		# print labeled samples indices
		print("seed:", self.seed)
		print("labeled sample idx:", labeled_idx)
		print("labeled set size:", len(labeled_idx))

		return labeled_idx, unlabeled_idx

	def preprocess(self, X, method):
		"""
		Preprocess the data by scaling into the range of 0-1 with bins.
		"""
		if method == "bucket":  # scales into 0-1 range with bins
			print("using the bucket prep method")
			from sklearn.preprocessing import KBinsDiscretizer
			est = KBinsDiscretizer(n_bins=10, encode="ordinal", strategy="quantile")
			est.fit(X)
			X_processed = est.transform(X)
			X_processed /= 10  # transform from nominal values to 0-1
			return X_processed
		elif method == "clip":  # clips the raw counts into a certain range
			print("using the clip prep method")
			cutoff = 1000
			X_processed = np.minimum(X, cutoff) + np.sqrt(np.maximum(X-cutoff, 0))
			return X_processed
		elif method == "log":  # takes the log of the count
			print("using the log prep method")
			import numpy.ma as ma
			mask = ma.log(X)
			# mask logged data to replace NaN (log0) with 0
			X_processed = ma.fix_invalid(mask, fill_value=0).data
			return X_processed
		else:
			raise Exception("Incorrect preprocess method name passed!")


from torch.utils import data
class Dataset(data.Dataset):
	""" For using with Pytorch Dataloader when batching. """
	def __init__(self, data, labels=None):
		self.data = data
		self.labels = labels  # optional

	def __len__(self):
		""" The total number of samples. """
		return self.data.size()[0]

	def __getitem__(self, index):
		if self.labels is None:
			return self.data[index, :]
		else:
			return self.data[index, :], self.labels[index]
