
# netAE: network-enhanced autoencoder

Network-enhanced autoencoder (netAE) is a semi-supervised dimensionality reduction method. It was specifically designed to facilitate cell labeling of single cell RNA-sequencing data.

## Important Note
This code base is a bit old at this time, so some implementation flaws do not necessarily reflect methodology flaws. Initially it was created to run experiments on medium-sized datasets mentioned in the paper's experiments section. I have seen complaints that the code wouldn't work on super large datasets because it is requesting an unreasonably large amount of memory. This is most likely because my implementation of the modularity calculation is matrix-based for parallelization, but it is much less data-efficient. In cases where you have a large set of labeled data that needs to go through the modularity calculation, consider changing to an iteration-based implementation. Alternatively, you can also try changing the "weighted modularity" to "sparsely weighted modularity" by discarding connections that are too weak.

As with many neural networks, always tune hyperparameters for your particular datasets before settling on model performance. On the three datasets that I ran my experiment on, I was able to get good results without much tuning at all. But that does not mean the model could work on any dataset without tuning. 

If you have questions on any of the above-mentioned issues, or questions in general, feel free to reach out!


## Getting Started

Clone the repository to your local directory.

### Prerequisites

- pytorch 1.1
- scikit-learn 0.22
- numpy 1.18
- pandas 1.0

## Data loading

The `data.py` script handles data loading and preprocessing. For example, `dataset = Data(data_path, labeled_size=10, prep_method="log")` loads the the dataset from `data_path`, randomly assigns 10 samples per class to be labeled, and preprocesses by taking the log of the raw values. Within the data folder specified by `data_path`, the numpy array `dataset_matched.npy` contains the data matrix and `gene_names.npy` contains the names of the genes (not used in training, just for reference if needed). Note that the data matrix has shape (nsamples, ngenes), so the samples are rows and the genes are columns.

To use the data for training, `dataset.load_all()` returns the following:
- `expr`: preprocessed expression matrix as a numpy array
- `lab_full`: labels of all samples
- `labeled_idx`: indices of the randomly selected labeled set
- `unlabeled_idx`: indices of the rest of the samples
- `info`: additional dictionary containing information of the dataset. `info["cell_type"]` is a dictionary that maps each label to the name of the cell type. `info["cell_id"]` contains the cell ID in the original dataset. `info["gene_names"]` contains the gene names of the dataset.

To load a small subset of the samples for testing, call `dataset.load_subset(p)` instead, where `p` specifies the percentage of all samples to load.

## Basic Usage
### Training a netAE model
The `run.py` script handles training. Train the model using the following command and some important argument parsers:
```
python run.py
## environment setup
--data-path             # path to dataset folder
--save-path             # path to save trained models
--dataset               # name of dataset (used to name model when saved)
--save_embd             # whether to save the embedded space directly along with the trained model

## model hyperparameters
--optim                 # name of optimizer
--lr                    # learning rate
--dim                   # dimension of the embedded space
--gamma                 # RBF coefficient for calculating similarity matrix in modularity
--kappa                 # KL divergence loss weight
--lambd                 # modularity loss weight
--phi                   # logistic regression loss weight

## data
--lab-size              # labeled set size for each cell type
--lab-ratio             # labeled set ratio for each cell type (specify either lab-size or lab-ratio but not both)
--seed                  # seed for randomly selecting labeled set
```

The script handles data loading as specified above. The default is to load all data and preprocess with the `log` method. One can change the data loader parameters to load a subset or use a different preprocessing method. The default values of the hyperparameters of the model are set to be the combination used in the paper. Therefore, one can simply run `python run.py --data-path DATA_PATH --save-path SAVE_PATH --dataset DATASET` to train the model with the dataset specified in `DATA_PATH` with name `DATASET` and save the trained model in `SAVE_PATH`.

Alternatively, one may call `train_netAE(all_data, labeled_data, labeled_lab, save_path, embd_save_path, args=None, params=params)` method from `netAE.py`, which directly returns the Pytorch model state dictionary of the trained model. This is easier if the trained model should be used in subsequent tasks. In this case, it is easier to pass in hyperparameters as a `params` dictionary, instead of the argument parser.

### Performing classification with trained netAE model
After training, one may want to use a classifier on the embedded space to test its classification accuracy. The `inference.py` script deals with comparing classification accuracy of netAE with other baseline models when using KNN and logistic regression, two simple classifiers. To start, make sure netAE, AE (the unsuperivsed counterpart), scVI, PCA, and ZIFA are trained and have their embedded spaces located in `MODEL_PATH`. Then simply pass in `--data-path`, `--model-path`, `--lab-size`, and `--dataset`. Additionally, to ensure that the labeled set used in training netAE is the same as here, make sure that you pass in the same seed `--seed` here as when training netAE.

## Data description
We describe the three datasets we used in our paper. Specifically, we have
- Cortex dataset:
    - Zeisel, A. et al. (2015). Cell types in the mouse cortex and hippocampus revealed by single-cell rna-seq. Science, 347(6226), 1138–1142.
    - 3005 samples, 19972 genes, and 7 cell classes
- Embryo dataset:
    - Petropoulos, S. et al. (2016). Single-cell rna-seq reveals lineage and x chromosome dynamics in human preimplantation embryos. Cell, 165(4), 1012–1026.
    - 1529 samples, 24444 genes, and 5 cell classes
- HSC dataset:
    - Nestorowa, S. et al. (2016). A single-cell resolution map of mouse hematopoietic stem and progenitor cell differentiation. Blood, 128(8), e20–e31.
    - 1920 samples, 24557 genes, and 3 cell classes

All datasets could be downloaded from the link provided in the original paper.

## Reference
[Dong, Z and Alterovitz, G (2020). netAE: Semi-supervised dimensionality reduction of single-cell RNA sequencing to facilitate cell labeling. *Bioinformatics*.](https://doi.org/10.1093/bioinformatics/btaa669)

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details
