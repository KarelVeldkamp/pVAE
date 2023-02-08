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
        mask = torch.Tensor(output!=0)

        movie_ids = np.pad(movie_ids, (self.n_max-ratings.shape[0],0), 'constant', constant_values=0)
        ratings = torch.Tensor(np.pad(ratings, (self.n_max - ratings.shape[0], 0), 'constant', constant_values=0))

        return movie_ids, ratings, output, mask


class CSVDataset():
    def __init__(self, filename: str):
        self.df = self.df = pd.read_csv(filename, index_col=0)
        self.n_max = np.max(np.sum(pd.notna(self.df), axis=1))

    def __len__(self):
        return len(self.df.index)


    def __getitem__(self, idx):
        # output to reconstruct
        output = self.df.iloc[idx,:]

        # indicating which ratings are not missing
        mask = torch.Tensor((pd.notna(output)).astype(int))

        # save list of movie ids and ratings and replace NA's in output with 0s (will be disregarded in loss using mask)
        movie_ids = np.where(pd.notna(output))[0]
        ratings = output.iloc[movie_ids]
        output = torch.Tensor(output.fillna(0))

        # convert to tensor and pad with constant values
        movie_ids = np.squeeze(np.pad(movie_ids, (self.n_max-ratings.shape[0],0), 'constant', constant_values=2))
        ratings = torch.Tensor(np.pad(ratings, (self.n_max - ratings.shape[0], 0), 'constant', constant_values=2))


        return movie_ids, ratings, output, mask

