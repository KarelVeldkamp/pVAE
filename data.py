import torch
import pandas as pd
from collections import defaultdict
import numpy as np

class MovielensDataset():
    def __init__(self, filename: str):
        self.df = pd.read_csv(filename, sep='::', header=None,
                           names=['UserID', 'MovieID', 'Rating', 'Time'])

        ids = self.df['MovieID']
        translate = defaultdict(lambda: len(translate))
        self.df['MovieID'] = [translate[id] for id in ids]


        # maximum number of ratings for a single user
        self.n_max = self.df['UserID'].value_counts().iloc[0]
        # total number of items
        self.nitems = len(np.unique(ids))

    def __len__(self):
        l = len(np.unique(self.df['UserID']))
        return l

    def __getitem__(self, idx):
        subset = self.df[self.df['UserID']==idx]
        movie_ids = subset['MovieID'].values
        ratings = subset['Rating'].values

        # output for the model to reconstruct
        output = np.ones(self.nitems)*0
        output[movie_ids] = ratings
        output = torch.Tensor(output)

        movie_ids = np.pad(movie_ids, (self.n_max-ratings.shape[0],0), 'constant', constant_values=0)
        ratings = torch.Tensor(np.pad(ratings, (self.n_max - ratings.shape[0], 0), 'constant', constant_values=0))

        return movie_ids, ratings, output


class CSVDataset():
    def __init__(self, filename: str):
        self.df = self.df = pd.read_csv(filename, index_col=0)
        self.n_max = np.max(np.sum(pd.notna(self.df), axis=1))

    def __len__(self):
        return len(self.df.index)


    def __getitem__(self, idx):
        output = self.df.iloc[idx,:]
        movie_ids = np.where(pd.notna(output))
        ratings = output.iloc[movie_ids]

        # convert to tensor and pad
        output = torch.Tensor(output)
        movie_ids = np.squeeze(np.pad(movie_ids, (self.n_max-ratings.shape[0],0), 'constant', constant_values=2))
        ratings = torch.Tensor(np.pad(ratings, (self.n_max - ratings.shape[0], 0), 'constant', constant_values=2))

        return movie_ids, ratings, output

