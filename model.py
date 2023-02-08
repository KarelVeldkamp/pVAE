from data import *
import torch
from torch import nn
import torch.nn.functional as F
import torch.nn.utils.prune
import pytorch_lightning as pl
from torch.utils.data import DataLoader
import numpy as np

class PartialEncoder(pl.LightningModule):
    def __init__(self, n_items, emb_dim, h_hidden_dim, latent_dim, hidden_layer_dim, mirt_dim):
        """

        :param n_items: total number of items
        :param emb_dim: dimension of the embedding layer
        :param latent_dim: dimension of the latent layer before pooling
        :param hidden_layer_dim: dimension of the hidden layer after pooling
        :param mirt_dim: latent dimension of the distribution that is sampled from
        """
        super(PartialEncoder, self).__init__()

        self.embedding = nn.Embedding(
                n_items+1,
                emb_dim,
        )

        self.emb_dim = emb_dim
        self.h_dense1 = nn.Linear(emb_dim, h_hidden_dim)
        self.h_dense2 = nn.Linear(h_hidden_dim, latent_dim)


        self.dense1 = nn.Linear(latent_dim, hidden_layer_dim)
        self.dense2m = nn.Linear(hidden_layer_dim, mirt_dim)
        self.dense2s = nn.Linear(hidden_layer_dim, mirt_dim)

        self.N = torch.distributions.Normal(0, 1)
        self.kl = 0

    def forward(self, item_ids: np.array, item_ratings: torch.Tensor) -> torch.Tensor:
        """
        A forward pass though the encoder network
        :param item_ids: a tensor with item ids
        :param item_ratings: a tensor with the corresponding item ratings
        :returns: (sample from the latent distribution, mean of the distribution, sd of the distribution)
        """

        E = self.embedding(item_ids)

        R = item_ratings.unsqueeze(2).repeat((1,1, self.emb_dim))

        S = E * R

        out = F.relu(self.h_dense1(S))
        out = F.relu(self.h_dense2(out))
        mean = torch.max(out, 1).values


        #dist = torch.cat([mean, sd, quantiles[0], quantiles[1], quantiles[2]], dim=1)
        hidden = F.relu(self.dense1(mean))
        mu = self.dense2m(hidden)
        log_sigma = self.dense2s(hidden)
        sigma = torch.exp(log_sigma)

        # sample from the latent dimensions
        z = mu + sigma * self.N.sample(mu.shape)

        # calculate kl divergence
        # self.kl = torch.mean(-0.5 * torch.sum(1 + torch.log(sigma) - mu ** 2 - torch.log(sigma).exp(), dim = 1), dim = 0)
        kl = 1 + 2 * log_sigma - torch.square(mu) - torch.exp(2 * log_sigma)

        kl = torch.sum(kl, dim=-1)
        self.kl = -.5 * torch.mean(kl)

        return z, mu, sigma


class Decoder(pl.LightningModule):
    """
    Neural network used as decoder
    """

    def __init__(self, nitems: int, latent_dims: int, qm: torch.Tensor=None):
        """
        Initialisation
        :param nitems: total number of items
        :param latent_dims: the number of latent factors
        :param qm: optional binary matrix specifying which weights should be removed

        """
        super().__init__()

        self.linear = nn.Linear(latent_dims, nitems)
        self.activation = nn.Sigmoid()

        # remove edges between latent dimensions and items that have a zero in the Q-matrix
        if qm is not None:
            msk_wts = torch.ones((nitems, latent_dims), dtype=torch.float32)
            for row in range(qm.shape[0]):
                for col in range(qm.shape[1]):
                    if qm[row, col] == 0:
                        msk_wts[row][col] = 0
            torch.nn.utils.prune.custom_from_mask(self.linear, name='weight', mask=msk_wts)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass though the network
        :param x: tensor representing a sample from the latent dimensions
        :return: tensor representing reconstructed item responses
        """
        out = self.linear(x)
        out = self.activation(out)

        return out


class PartialVariationalAutoencoder(pl.LightningModule):
    """
    Neural network for the entire partial variational autoencoder
    """

    def __init__(self,
                 emb_dim: int,
                 h_hidden_dim:int,
                 latent_dim: int,
                 hidden_layer_dim: int,
                 mirt_dim: int,
                 learning_rate: float,
                 batch_size: int,
                 dataset: str,
                 Q: np.array,
                 beta:int=1):
        """
        
        :param emb_dim: dimension of the item embeddings
        :param latent_dim: dimension of the layer before pooling
        :param hidden_layer_dim: dimension of the layer after pooling
        :param mirt_dim: dimension of the latent distribution to be sampled from
        :param learning_rate: learning rate
        :param batch_size: batch size
        :param dataset: which dataset to use
        """"""
        """
        super(PartialVariationalAutoencoder, self).__init__()
        # self.automatic_optimization = False
        nitems_dict = {'movielens': 3706,
                  'dylan': 28,
                  'missing': 400}
        self.nitems = nitems_dict[dataset]

        self.encoder = PartialEncoder(self.nitems, emb_dim, h_hidden_dim, latent_dim, hidden_layer_dim, mirt_dim)

        self.decoder = Decoder(self.nitems, mirt_dim, qm=Q)

        self.lr = learning_rate
        self.batch_size = batch_size
        self.dataset = dataset
        self.beta = beta

    def forward(self, user_ids: torch.Tensor, ratings: torch.Tensor):
        """
        forward pass though the entire network
        :param user_ids: tensor representing user ids
        :param ratings: tensor represeting ratings
        :return: tensor representing a reconstruction of the input response data
        """
        z, _, _ = self.encoder(user_ids, ratings)
        return self.decoder(z)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr, amsgrad=True)

    def training_step(self, batch, batch_idx):
        user_ids, ratings, output, mask = batch

        X_hat = self(user_ids, ratings)

        # calculate the likelihood, and take the mean of all non missing elements
        bce = torch.nn.functional.binary_cross_entropy(X_hat, batch[2], reduction='none') * mask
        bce = torch.mean(bce) * self.nitems
        bce = bce / torch.mean(mask.float())


        # sum the likelihood and the kl divergence
        loss = bce + self.beta * self.encoder.kl
        self.log('binary_cross_entropy', bce)
        self.log('kl_divergence', self.encoder.kl)
        self.log('train_loss', loss)

        return {'loss': loss}

    def train_dataloader(self):
        if self.dataset.lower() == 'movielens':
            dataset = MovielensDataset('./data/movielens/ratings.dat')
        elif self.dataset.lower() == 'missing':
            dataset = CSVDataset('./data/missing/data.csv')
        elif self.dataset.lower() == 'dylan':
            dataset = CSVDataset('./data/dylan/data.csv')
        else:
            raise ValueError('Invalid dataset name')
        train_loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)
        return train_loader
