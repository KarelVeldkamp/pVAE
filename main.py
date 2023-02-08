from model import *
from plotting import *
import pandas as pd
import matplotlib.pyplot as plt
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import CSVLogger
import yaml

# read configurations from file
with open("./config.yml", "r") as f:
    cfg = yaml.safe_load(f)
    cfg = cfg['config']

# initialise model and optimizer
logger = CSVLogger("logs", name=cfg['dataset'], version=0)
trainer = Trainer(fast_dev_run=False,
                  max_epochs=cfg['max_epochs'],
                  logger=logger,
                  callbacks=[EarlyStopping(monitor='train_loss', min_delta=cfg['min_delta'], patience=cfg['patience'], mode='min')])

if cfg['dataset'] == 'missing' or cfg['dataset'] == 'dylan':
    a_true = pd.read_csv(f'./data/{cfg["dataset"]}/a.csv', index_col=0).values
    Q = a_true != 0
    Q = Q.astype(int)
else:
    Q = None

pvae = PartialVariationalAutoencoder(
                 emb_dim=cfg['emb_dim'],
                 h_hidden_dim=cfg['h_hidden_dim'],
                 latent_dim=cfg['latent_dim'],
                 hidden_layer_dim=cfg['hidden_layer_dim'],
                 mirt_dim=cfg['mirt_dim'],
                 learning_rate=cfg['lr'],
                 batch_size=cfg['batch_size'],
                 dataset=cfg['dataset'],
                 Q=Q)
trainer.fit(pvae)


# plot training loss
logs = pd.read_csv(f'logs/{cfg["dataset"]}/version_0/metrics.csv')
plt.plot(logs['epoch'], logs['train_loss'])
plt.title('Training loss')
plt.savefig(f'./figures/{cfg["dataset"]}/training_loss.png')
# plot binary cross entropy
plt.clf()
plt.plot(logs['epoch'], logs['binary_cross_entropy'])
plt.title('Binary Cross Entropy')
plt.savefig(f'./figures/{cfg["dataset"]}/binary_cross_entropy.png')
# plot KL divergence
plt.clf()
plt.plot(logs['epoch'], logs['kl_divergence'])
plt.title('KL Divergence')
plt.savefig(f'./figures/{cfg["dataset"]}/kl_divergence.png')



if cfg['dataset'] == 'dylan' or cfg['dataset'] == 'missing':
    plot(pvae, cfg['dataset'])







