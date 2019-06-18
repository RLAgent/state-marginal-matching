"""Simple density model that discretizes states."""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from rlkit.torch import pytorch_util as ptu
from rlkit.torch.core import PyTorchModule


class VAEDensity(PyTorchModule):
    def __init__(self,
                 input_size,
                 num_skills,
                 code_dim,
                 beta=0.5,
                 lr=1e-3,
                 ):
        """Initialize the density model.

        Args:
          num_skills: number of densities to simultaneously track
        """
        self.save_init_params(locals())
        super().__init__()
        self._num_skills = num_skills

        input_dim = np.prod(input_size)
        self.enc = nn.Sequential(
            nn.Linear(input_dim, 150),
            nn.ReLU(),
            nn.Linear(150, 150),
            nn.ReLU(),
        )
        self.enc_mu = nn.Linear(150, code_dim)
        self.enc_logvar = nn.Linear(150, code_dim)
        self.dec = nn.Sequential(
            nn.Linear(code_dim, 150),
            nn.ReLU(),
            nn.Linear(150, 150),
            nn.ReLU(),
            nn.Linear(150, input_dim),
        )

        self.lr = lr
        self.beta = beta
        params = (list(self.enc.parameters()) +
                  list(self.enc_mu.parameters()) +
                  list(self.enc_logvar.parameters()) +
                  list(self.dec.parameters()))
        self.optimizer = optim.Adam(params, lr=self.lr)

    def get_output_for(self, aug_obs, sample=True):
        """
        Returns the log probability of the given observation.
        """
        obs = aug_obs
        with torch.no_grad():
            enc_features = self.enc(obs)
            mu = self.enc_mu(enc_features)
            logvar = self.enc_logvar(enc_features)

            stds = (0.5 * logvar).exp()
            if sample:
                epsilon = ptu.randn(*mu.size())
            else:
                epsilon = torch.ones_like(mu)
            code = epsilon * stds + mu

            obs_distribution_params = self.dec(code)
            log_prob = -1. * F.mse_loss(obs, obs_distribution_params,
                                        reduction='none')
            log_prob = torch.sum(log_prob, -1, keepdim=True)
        return log_prob.detach()

    def update(self, aug_obs):
        obs = aug_obs

        enc_features = self.enc(obs)
        mu = self.enc_mu(enc_features)
        logvar = self.enc_logvar(enc_features)

        stds = (0.5 * logvar).exp()
        epsilon = ptu.randn(*mu.size())
        code = epsilon * stds + mu

        kle = -0.5 * torch.sum(
            1 + logvar - mu.pow(2) - logvar.exp(), dim=1
        ).mean()

        obs_distribution_params = self.dec(code)
        log_prob = -1. * F.mse_loss(obs, obs_distribution_params,
                                    reduction='elementwise_mean')

        loss = self.beta * kle - log_prob

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.cpu().item()
