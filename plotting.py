import torch
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from data import *


def plot(vae, dataset):
    # read data and true parameter values
    DS = CSVDataset(f'./data/{dataset}/data.csv')
    dataloader = torch.utils.data.DataLoader(dataset=DS, batch_size=len(DS))
    items, ratings, _ , _= next(iter(dataloader))

    # read in true parameter estimates
    a_true = pd.read_csv(f'./data/{dataset}/a.csv').iloc[:, 1:].values
    theta_true = pd.read_csv(f'./data/{dataset}/theta.csv').iloc[:, 1:].values
    d_true = pd.read_csv(f'./data/{dataset}/d.csv').iloc[:, 1:].values

    a_est = vae.decoder.linear.weight.detach().numpy()[:, 0:3]
    d_est = vae.decoder.linear.bias.detach().numpy()

    _ , theta_est, _ = vae.encoder(items, ratings)
    theta_est = theta_est.detach().numpy()


    # invert factors for increased interpretability
    a_est, theta_est = inv_factors(a_est, theta_est)


    # parameter estimation plot for a
    for dim in range(3):
        plt.figure()
        ai_est = a_est[:,dim]
        ai_true = a_true[:,dim]
        mse = MSE(ai_est, ai_true)
        plt.scatter(y=ai_est, x=ai_true)
        plt.plot(ai_true, ai_true)
        #for i, x in enumerate(ai_true):
        #    plt.text(ai_true[i], ai_est[i], i)
        plt.title(f'Parameter estimation plot: a{dim+1}, MSE={round(mse,4)}')
        plt.xlabel('True values')
        plt.ylabel('Estimates')
        plt.savefig(f'./figures/{dataset}/param_est_plot_a{dim+1}.png')

        # parameter estimation plot for theta
        plt.figure()
        thetai_est = theta_est[:, dim]
        thetai_true = theta_true[:, dim]
        mse = MSE(thetai_est, thetai_true)
        plt.scatter(y=thetai_est, x=thetai_true)
        plt.plot(thetai_true, thetai_true)
        plt.title(f'Parameter estimation plot: theta{dim+1}, MSE={round(mse,4)}')
        plt.xlabel('True values')
        plt.ylabel('Estimates')
        plt.savefig(f'./figures/{dataset}/param_est_plot_theta{dim+1}.png')

    # parameter estimation plot for d
    plt.figure()
    plt.scatter(y=d_est, x=d_true)
    plt.plot(d_true, d_true)
    mse = MSE(d_est, d_true)
    plt.title(f'Parameter estimation plot: d, MSE={round(mse,4)}')
    plt.xlabel('True values')
    plt.ylabel('Estimates')
    plt.savefig(f'./figures/{dataset}/param_est_plot_d.png')


def inv_factors(a, theta=None):
    """
    Helper function that inverts factors when discrimination values are mostly negative this improves the
    interpretability of the solution
        theta: NxP matrix of theta estimates
        a: IxP matrix of a estimates

        returns: tuple of inverted theta and a paramters
    """
    totals = np.sum(a, axis=0)
    a *= totals / np.abs(totals)
    if theta is not None:
        theta *= totals / np.abs(totals)

    return a, theta

def MSE(est, true):
    """
    Mean square error
    Parameters
    ----------
    est: estimated parameter values
    true: true paremters values

    Returns
    -------
    the MSE
    """
    return np.mean(np.power(est-true,2))