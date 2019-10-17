"""
Stores different neural network structures.

Vdca is the implementation of netAE in our paper. Dae, Vae, and Dca are all for
reference only if one wishes to experiment with other autoencoder architecture
based on the semi-supervised ideas behind netAE. Vdca is generally best performing
for single-cell RNA-sequencing data, but other autoencoder architecture may be
stronger for some other tasks.

Detailed function and parameter descriptions for Dae, Vae, and Dca can be found
in the comments for Vdca, because the purpose of the functions are very similar.

NN_sup is an additional neural network that is used as a supervised baseline in
out paper.

Credit: the implementation of VAE and the function for calculating ZINB likelihood
is not our work. The original source has been cited in the comments.

@author: Leo Zhengyang Dong
@contact: leozdong@stanford.edu
@date: 10/16/2019
"""

import torch
import torch.nn as nn
from torch.autograd import Variable
import modularity
from torch.nn import functional as F

class Dae(nn.Module):

    def __init__(self, z_dim, input_dim, n_classes, lambd=0, gamma=0, phi=0):
        super(Dae, self).__init__()

        #### Network parameters ####
        self.z_dim = z_dim  # dimension of the encoded layer
        self.input_dim = input_dim  # dimension of the input layer
        self.n_classes = n_classes  # number of classes, for using in softmax regression
        h_dim = 1000  # dimension of the hidden layer before the encoded layer

        #### Loss parameters ####
        self.lambd = lambd
        self.gamma = gamma
        self.phi = phi

        #### Encoder ####
        self.fc1_dropout = nn.Dropout(p=0.2)  # adding noise
        self.fc1 = nn.Linear(input_dim, h_dim)
        self.fc2 = nn.Linear(h_dim, z_dim)

        #### Decoder ####
        self.fc3 = nn.Linear(z_dim, h_dim)
        self.fc4 = nn.Linear(h_dim, input_dim)

        #### Softmax regression ####
        self.fc3_dropout = nn.Dropout(p=0.5)
        self.fc3_predict = nn.Linear(z_dim, self.n_classes)
        self.NLLLoss = nn.NLLLoss()

        # reconstruction loss
        self.bceLoss = nn.BCEWithLogitsLoss(reduction="none")

    def encode(self, x):
        if self.training:
            x_dropout = self.fc1_dropout(x)
        else:  # no need to add noise when in evaulation mode
            x_dropout = x
        h1 = F.elu(self.fc1(x_dropout))
        return F.elu(self.fc2(h1))

    def get_z(self, x):
        return self.encode(x)

    def decode(self, z):
        # pass through the hidden layer after the encoded layer first
        h3 = F.elu(self.fc3(z))
        # return unactivated logits
        return self.fc4(h3)

    def forward(self, x, make_pred=False):
        z = self.encode(x)

        if not make_pred:
            return self.decode(z)
        else:
            if self.training:
                z_dropout = self.fc3_dropout(z)
            else:
                z_dropout = z
            pred = F.log_softmax(self.fc3_predict(z_dropout), dim=-1)

            # calculate L2 norm of the weight vectors
            l2_norm = torch.norm(self.fc3_predict.weight, p=2, dim=1).mean()

            return l2_norm, pred, z

    def calc_losses(self, x, labeled_data_sets, labeled_data, labels, no_mod=False):
        #### Preparing for loss calculation ####
        # calculate mu, theta, pi for NB (unsupervised)
        recons_logits = self.forward(x, make_pred=False)

        if not unsup:
            # calculate L2 norm, prediction, and z for labeled data
            l2_norm, pred, z = self.forward(labeled_data, make_pred=True)
            # obtain sets of learned features for the labeled samples
            # (for calculating modularity)
            fea_labels_sets = []
            for sub_data, sub_lab in labeled_data_sets:
                fea = self.get_z(sub_data)
                fea_labels_sets.append((fea, sub_lab))

        #### Calculating losses ####
        # Unsupervised loss:
        reconsLoss = self.bceLoss(recons_logits, torch.sigmoid(x)).sum(dim=-1).mean()  # can also use L2 loss
        if unsup:
            return reconsLoss, reconsLoss, 0, 0

        # Supervised loss:
        # modularity
        q = modularity.calc_modularity_mult(fea_labels_sets, self.gamma)
        netLoss = -q

        # softmax regression
        nll = self.NLLLoss(pred, labels)
        pred_lab = torch.max(pred, dim=1)[1]
        acc = (pred_lab == labels).sum().double() / pred_lab.size()[0]

        # optional L2 regularization
        l2_penalty = l2_norm * 0

        totalLoss = reconsLoss + self.lambd * netLoss + self.phi * nll + l2_penalty

        return totalLoss, reconsLoss, q, acc



class Vae(nn.Module):
    """ Credit: implementation of VAE is adapted from blog:
    https://vxlabs.com/2017/12/08/variational-autoencoder-in-pytorch-commented-and-annotated/.
    """

    def __init__(self, z_dim, input_dim, n_classes, lambd=0, gamma=0, phi=0, kappa=0):
        super(Vae, self).__init__()

        #### Network parameters
        self.z_dim = z_dim  # dimension of the encoded layer
        self.input_dim = input_dim  # dimension of the input layer
        self.n_classes = n_classes  # number of classes, for using in softmax regression
        h_dim = 1000  # dimension of the hidden layer before the encoded layer

        #### Loss parameters ####
        self.lambd = lambd
        self.gamma = gamma
        self.phi = phi
        self.kappa = kappa

        #### Encoder ####
        self.fc1 = nn.Linear(input_dim, h_dim)
        self.fc2_z_mu = nn.Linear(h_dim, z_dim)  # mu layer
        self.fc2_z_var = nn.Linear(h_dim, z_dim)  # logvariance layer

        #### Decoder ####
        self.fc3 = nn.Linear(z_dim, h_dim)
        self.fc4 = nn.Linear(h_dim, input_dim)

        #### Softmax regression ####
        self.fc3_dropout = nn.Dropout(p=0.5)
        self.fc3_predict = nn.Linear(z_dim, self.n_classes)
        self.NLLLoss = nn.NLLLoss()

        # reconstruction loss
        self.bceLoss = nn.BCEWithLogitsLoss(reduction="none")

    def encode(self, x):
        h1 = F.elu(self.fc1(x))
        return self.fc2_z_mu(h1), self.fc2_z_var(h1)

    def reparameterize(self, mu, logvar):
        if self.training:
            # obtain standard deviation from log variance
            std = torch.exp(0.5*logvar)
            # values are sampled from unit normal distribution
            eps = torch.randn_like(std)
            return mu + eps*std
        else:
            # During inference, simply return the mean of the learned distribution
            # of the current input. This is close to randomly sample from the distribution.
            return mu

    def get_z(self, x):
        return self.reparameterize(*self.encode(x))

    def decode(self, z):
        # pass through the hidden layer after the encoded layer first
        h3 = F.elu(self.fc3(z))
        # return unactivated logits
        return self.fc4(h3)

    def forward(self, x, make_pred=False):
        # obtain mu and logvar of the latent space
        mu, logvar = self.encode(x.view(-1, self.input_dim))
        # sample from latent space
        z = self.reparameterize(mu, logvar)

        if not make_pred:
            return self.decode(z), mu, logvar
        else:
            if self.training:
                z_dropout = self.fc3_dropout(z)
            else:
                z_dropout = z
            pred = F.log_softmax(self.fc3_predict(z_dropout), dim=-1)

        # calculate L2 norm of the weight vectors
        l2_norm = torch.norm(self.fc3_predict.weight, p=2, dim=1).mean()

        return l2_norm, pred, z

    def calc_losses(self, x, labeled_x_sets, labeled_x, labels, unsup=False):
        """ See Vdca for parameter descriptions. """
        #### Preparing for loss calculation ####
        # calculate z_mu, z_logvar for KL (unsupervised)
        recons_logits, z_mu, z_logvar = self.forward(x, make_pred=False)

        if not unsup:
            # calculate L2 norm, prediction, and z for labeled x (supervised)
            l2_norm, pred, z = self.forward(labeled_x, make_pred=True)
            # obtain sets of learned features for the labeled samples
            # (for calculating modularity)
            fea_labels_sets = []
            for sub_x, sub_lab in labeled_x_sets:
                fea = self.get_z(sub_x)
                fea_labels_sets.append((fea, sub_lab))


        #### Calculating losses ####
        # Unsupervised loss:
        reconsLoss = self.bceLoss(recons_logits, torch.sigmoid(x)).sum(dim=-1).mean()  # can also use L2 loss
        kld = (-(0.5 * z_logvar) + (torch.exp(z_logvar) + z_mu**2 ) / 2. - 0.5).sum(dim=-1).mean()
        vaeLoss = reconsLoss + self.kappa * kld
        if unsup:
            return vaeLoss, vaeLoss, 0, reconsLoss, kld, 0
        # Supervised loss:
        q = modularity.calc_modularity_mult(fea_labels_sets, self.gamma)
        netLoss = -q

        # calculate NLL loss for softmax regression
        nll = self.NLLLoss(pred, labels)

        pred_lab = torch.max(pred, dim=1)[1]
        acc = (pred_lab == labels).sum().double() / pred_lab.size()[0]

        # optional L2 regularization
        l2_penalty = l2_norm * 0

        totalLoss = vaeLoss + self.lambd * netLoss + self.phi * nll + l2_penalty

        return totalLoss, vaeLoss, q, reconsLoss, kld, acc



class Dca(nn.Module):
    """
    Defines a semi-supervised autoencoder based on the deep count autoencoder,
    which is essentially an autoencoder with the output space defined to be the
    zero-inflated negative binomial distribution.
    """

    def __init__(self, z_dim, input_dim, n_classes, lambd=0, phi=0, gamma=0):
        super(Dca, self).__init__()

        #### Network parameters ####
        self.z_dim = z_dim  # dimension of the encoded layer
        self.input_dim = input_dim  # dimension of the input layer
        self.n_classes = n_classes  # number of classes, for using in softmax regression
        h_dim = 1000  # dimension of the hidden layer before the encoded layer

        #### Loss parameters ####
        self.lambd = lambd
        self.phi = phi
        self.gamma = gamma

        #### Encoder ####
        self.fc1 = nn.Linear(input_dim, h_dim)
        self.fc2 = nn.Linear(h_dim, z_dim)

        #### Decoder ####
        self.fc3 = nn.Linear(z_dim, h_dim)
        self.fc4_mu = nn.Linear(h_dim, input_dim)  # mu layer
        self.fc4_theta = nn.Linear(h_dim, input_dim)  # theta layer
        self.fc4_pi = nn.Linear(h_dim, input_dim)  # pi layer (logits)

        #### Softmax Regression ####
        self.fc3_dropout = nn.Dropout(p=0.5)
        self.fc3_predict = nn.Linear(z_dim, self.n_classes)
        self.NLLLoss = nn.NLLLoss()

    def encode(self, x):
        return F.elu(self.fc2(F.elu(self.fc1(x))))

    def get_z(self, x):
        return self.encode(x)

    def decode(self, z):
        h3 = F.elu(self.fc3(z))  # the hidden layer after the encoded layer

        h4_mu = torch.exp(self.fc4_mu(h3))
        h4_theta = torch.exp(self.fc4_theta(h3))
        h4_pi = self.fc4_pi(h3)  # logits
        return h4_mu, h4_theta, h4_pi

    def forward(self, x, make_pred=False):
        z = self.encode(x)

        if not make_pred:
            return self.decode(z)
        else:
            if self.training:
                z_dropout = self.fc3_dropout(z)
            else:
                z_dropout = z
            pred = F.log_softmax(self.fc3_predict(z_dropout), dim=-1)

            # calculate L2 norm of the weight vectors
            l2_norm = torch.norm(self.fc3_predict.weight, p=2, dim=1).mean()

            return l2_norm, pred, z

    def calc_losses(self, x, labeled_data_sets, labeled_data, labels, unsup=False):
        #### Preparing for loss calculation ####
        # calculate mu, theta, pi for NB (unsupervised)
        mu, theta, pi = self.forward(x, make_pred=False)

        if not unsup:
            # calculate L2 norm, prediction, and z for labeled data
            l2_norm, pred, z = self.forward(labeled_data, make_pred=True)
            # obtain sets of learned features for the labeled samples
            # (for calculating modularity)
            fea_labels_sets = []
            for sub_data, sub_lab in labeled_data_sets:
                fea = self.get_z(sub_data)
                fea_labels_sets.append((fea, sub_lab))

        #### Calculating losses ####
        # Unsupervised loss:
        reconsLoss = -log_zinb_positive(x, mu, theta, pi).mean()
        if unsup:
            return reconsLoss, reconsLoss, 0, 0

        # Supervised loss:
        q = modularity.calc_modularity_mult(fea_labels_sets, self.gamma)
        netLoss = -q

        # calculate NLL loss for softmax regression
        nll = self.NLLLoss(pred, labels)

        pred_lab = torch.max(pred, dim=1)[1]
        acc = (pred_lab == labels).sum().double() / pred_lab.size()[0]

        # optional L2 regularization
        l2_penalty = l2_norm * 0

        totalLoss = reconsLoss + self.lambd * netLoss + self.phi * nll + l2_penalty

        return totalLoss, reconsLoss, q, acc


class Vdca(nn.Module):
    """
    Defines a semi-supervised autoencoder based on a variational deep count
    autoencoder structure, i.e. a deep count autoencoder with a variational
    latent space.

    This is the implementation of netAE described in the paper. All previous
    autoencoder structures are for reference only.

    Parameters
    ----------
    z_dim : numeric
        Dimension of the latent space.
    input_dim : numeric
        Dimension of the input space.
    n_classes : numeric
        Number of classes (types) of cells.
    lambd : numeric
        Coefficient for the supervised modularity loss.
    gamma : numeric
        Coefficient for the RBF kernel for calculating modularity loss.
    phi : numeric
        Coefficient for the supervised regression loss.
    kappa : numeric
        Coefficient for the KL divergence term of the loss.
    """

    def __init__(self, z_dim, input_dim, n_classes, lambd=0, gamma=0, phi=0, kappa=0):
        super(Vdca, self).__init__()

        #### Network parameters ####
        self.z_dim = z_dim  # dimension of the encoded layer
        self.input_dim = input_dim  # dimension of the input layer
        self.n_classes = n_classes  # number of classes, for using in softmax regression
        h_dim = 1000  # dimension of the hidden layer before the encoded layer

        #### Loss parameters ####
        self.lambd = lambd
        self.phi = phi
        self.kappa = kappa
        self.gamma = gamma

        #### Encoder ####
        self.fc1 = nn.Linear(input_dim, h_dim)
        self.fc2_z_mu = nn.Linear(h_dim, z_dim)  # mu layer
        self.fc2_z_var = nn.Linear(h_dim, z_dim)  # logvariance layer

        #### Decoder ####
        self.fc3 = nn.Linear(z_dim, h_dim)
        self.fc4_mu = nn.Linear(h_dim, input_dim)
        self.fc4_theta = nn.Linear(h_dim, input_dim)
        self.fc4_pi = nn.Linear(h_dim, input_dim)

        #### Softmax regression ####
        self.fc3_dropout = nn.Dropout(p=0.5)
        self.fc3_predict = nn.Linear(z_dim, self.n_classes)
        self.NLLLoss = nn.NLLLoss()

    def encode(self, x):
        """ Encodes the input into latent features. """
        h1 = F.elu(self.fc1(x))
        return self.fc2_z_mu(h1), self.fc2_z_var(h1)

    def reparameterize(self, mu, logvar):
        """ Resample from the latent space distribution, specified by mu and logvar.
        The resampled samples will be later decoded to try to recover the input.

        Parameters
        ----------
        mu : tensor (n_samples, z_dim)
        logvar : tensor (n_samples, z_dim)

        Returns
        -------
        Training mode: returns a random sample from the learned latent space distribution.
        Evaluation/inference mode: returns the mean matrix.
        """
        if self.training:
            # obtain standard deviation from log variance
            std = torch.exp(0.5*logvar)
            # values are sampled from unit normal distribution
            eps = torch.randn_like(std)
            return mu + eps*std
        else:
            # During inference, simply return the mean of the learned distribution
            # of the current input. This is close to randomly sample from the distribution.
            return mu

    def get_z(self, x):
        """ Obtain the latent features from the input.
        Since different autoencoders have different ways of getting z (variational
        requires reparameterizing, while regular does not), this function is a
        uniform way of obtaining the latent features without the knowledge of
        the autoencoder type.
        """
        return self.reparameterize(*self.encode(x))

    def decode(self, z):
        """ Output of the autoencoder. """
        # pass through the hidden layer after the encoded layer first
        h3 = F.elu(self.fc3(z))

        h4_mu = torch.exp(self.fc4_mu(h3))
        h4_theta = torch.exp(self.fc4_theta(h3))
        h4_pi = self.fc4_pi(h3)  # logits
        return h4_mu, h4_theta, h4_pi

    def forward(self, x, make_pred=False):
        """
        One pass through the neural network. If make_pred is True, also return
        the prediction after the regression layer and the L2 norm for optional
        regularization.
        """
        # obtain mu and logvar of the latent space
        mu, logvar = self.encode(x.view(-1, self.input_dim))
        # sample from latent space
        z = self.reparameterize(mu, logvar)

        if not make_pred:
            return (*self.decode(z)), mu, logvar
        else:
            if self.training:
                z_dropout = self.fc3_dropout(z)
            else:
                z_dropout = z
            pred = F.log_softmax(self.fc3_predict(z_dropout), dim=-1)

            # calculate L2 norm of the weight vectors
            l2_norm = torch.norm(self.fc3_predict.weight, p=2, dim=1).mean()

            return l2_norm, pred, z

    def calc_losses(self, x, labeled_x_sets, labeled_x, labels, unsup=False):
        """ Defines a loss function for netVDCA.

        Parameters
        ----------
        x : tensor, shape (n_samples, n_features)
            Tensor of real inputs.
            For all samples, both labeled and unlabeled.
        labeled_x_sets : list of lists [set1, set2, set3, ...]
            set1 = [data of set1, labels of set1]
        labeled_x : tensor, shape (n_labeled_samples, n_features)
        labels : tensor, shape (n_labeled_samples)
        no_mod : boolean
            Indicates whether or not modularity should be calculated.
            To save time, we do not calculate modularity for validation.
        """
        #### Preparing for loss calculation ####
        # calculate mu, theta, pi for NB and z_mu, z_logvar for KL (unsupervised)
        mu, theta, pi, z_mu, z_logvar = self.forward(x, make_pred=False)

        if not unsup:
            # calculate L2 norm, prediction, and z for labeled x (supervised)
            l2_norm, pred, z = self.forward(labeled_x, make_pred=True)
            # obtain sets of learned features for the labeled samples
            # (for calculating modularity)
            fea_labels_sets = []
            for sub_x, sub_lab in labeled_x_sets:
                fea = self.get_z(sub_x)
                fea_labels_sets.append((fea, sub_lab))

        #### Calculating losses ####
        # Unsupervised loss:
        reconsLoss = -log_zinb_positive(x, mu, theta, pi).mean()
        kld = (-(0.5 * z_logvar) + (torch.exp(z_logvar) + z_mu**2 ) / 2. - 0.5).sum(dim=-1).mean()
        vaeLoss = reconsLoss + self.kappa * kld
        if unsup:
            return vaeLoss, vaeLoss, 0, reconsLoss, kld, 0

        # Supervised loss:
        q = modularity.calc_modularity_mult(fea_labels_sets, self.gamma)
        netLoss = -q

        # calculate NLL loss for softmax regression
        nll = self.NLLLoss(pred, labels)

        pred_lab = torch.max(pred, dim=1)[1]
        acc = (pred_lab == labels).sum().double() / pred_lab.size()[0]

        # optional L2 regularization
        l2_penalty = l2_norm * 0

        totalLoss = vaeLoss + self.lambd * netLoss + self.phi * nll + l2_penalty

        return totalLoss, vaeLoss, q, reconsLoss, kld, acc


def log_zinb_positive(x, mu, theta, pi, eps=1e-8):
    """
    Adopted from: https://github.com/YosefLab/scVI/blob/master/scvi/models/log_likelihood.py#L11
    Equations follow the paper: https://www.nature.com/articles/s41467-017-02554-5

    Parameters
    ----------
    mu: tensor (nsamples, nfeatures)
        Mean of the negative binomial (has to be positive support).
    theta: tensor (nsamples, nfeatures)
        Inverse dispersion parameter (has to be positive support).
    pi: tensor (n_samples, nfeatures)
        Logit of the dropout parameter (real support).
    eps: numeric
        Numerical stability constant.
    """

    # theta is the dispersion rate. If .ndimension() == 1, it is shared for all cells (regardless of batch or labels)
    if theta.ndimension() == 1:
        theta = theta.view(1, theta.size(0))  # In this case, we reshape theta for broadcasting

    softplus_pi = F.softplus(-pi)
    log_theta_eps = torch.log(theta + eps)
    log_theta_mu_eps = torch.log(theta + mu + eps)
    pi_theta_log = - pi + theta * (log_theta_eps - log_theta_mu_eps)

    case_zero = F.softplus(pi_theta_log) - softplus_pi
    mul_case_zero = torch.mul((x < eps).type(torch.float32), case_zero)

    case_non_zero = - softplus_pi + \
        pi_theta_log + \
        x * (torch.log(mu + eps) - log_theta_mu_eps) + \
        torch.lgamma(x + theta) - \
        torch.lgamma(theta) - \
        torch.lgamma(x + 1)
    mul_case_non_zero = torch.mul((x > eps).type(torch.float32), case_non_zero)

    res = mul_case_zero + mul_case_non_zero

    return torch.sum(res, dim=-1)



class NN_sup(nn.Module):
    """
    A baseline neural network that takes the encoder structure of netAE (class
    Vdca) and trains in a supervised fashion.
    """

    def __init__(self, z_dim, input_dim, n_classes):
        super(NN_sup, self).__init__()

        #### Network parameters ####
        self.z_dim = z_dim  # the final layer before softmax layer for prediction
        self.input_dim = input_dim
        self.n_classes = n_classes
        h_dim = 1000

        #### Network structure ####
        self.fc1 = nn.Linear(input_dim, h_dim)
        self.fc2 = nn.Linear(h_dim, z_dim)

        # softmax Regression
        self.fc3_dropout = nn.Dropout(p=0.5)
        self.fc3_predict = nn.Linear(z_dim, self.n_classes)
        self.NLLLoss = nn.NLLLoss()

    def forward(self, x):
        z = F.elu(self.fc2(F.elu(self.fc1(x))))

        if self.training:
            z_dropout = self.fc3_dropout(z)
        else:
            z_dropout = z
        pred = F.log_softmax(self.fc3_predict(z_dropout), dim=-1)

        return pred

    def calc_losses(self, x, labels):
        pred = self.forward(x)

        # calculate NLL loss for softmax regression
        nll = self.NLLLoss(pred, labels)

        # also calculate accuracy for easy viewing
        pred_lab = torch.max(pred, dim=1)[1]
        acc = (pred_lab == labels).sum().double() / pred_lab.size()[0]

        return nll, acc
