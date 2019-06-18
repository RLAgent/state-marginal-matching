import numpy as np
import torch
import torch.optim as optim
from torch import nn as nn
import torch.nn.functional as F

import rlkit.torch.pytorch_util as ptu


class ICMHook:
    """
    Intrinsic Curiosity Module (ICM)

    :param base_algorithm: (TorchRLAlgorithm) Base RL algorithm
    :param embedding_model: Embedding model
    :param forward_model: Forward model
    :param inverse_model: Inverse model
    :param rl_coeff: (TorchRLAlgorithm) Weight on the base RL algorithm reward relative to the ICM reward.
    :param lr: (float) Learning rate
    :param optimizer_class: (float) Optimizer class
    """
    def __init__(
            self,
            base_algorithm,
            embedding_model,
            forward_model,
            inverse_model,
            rl_coeff=1.,
            lr=1e-3,
            optimizer_class=optim.Adam,
    ):
        self.base_algorithm = base_algorithm
        self.embedding_model = embedding_model
        self.forward_model   = forward_model
        self.inverse_model   = inverse_model

        self.rl_coeff = rl_coeff
        
        self.optimizer = optimizer_class(
            list(self.embedding_model.parameters()) +
            list(self.forward_model.parameters()) +
            list(self.inverse_model.parameters()),
            lr=lr,
        )

        # Do this hack to make wrapping algorithms as easy as possible for rlkit's API
        # This makes SMM function as a reward shaper, being completely transparent to the
        # blackbox RL algorithm
        def wrapped_get_batch():
            return self.get_batch()
        def wrapped_networks():
            return self.base_algorithm.__orig_networks() + [self.embedding_model,
                                                            self.forward_model,
                                                            self.inverse_model]
        def wrapped_get_epoch_snapshot(epoch):
            snapshot = self.base_algorithm.__orig_get_epoch_snapshot(epoch)
            snapshot.update(
                embedding_model=self.embedding_model,
                forward_model=self.forward_model,
                inverse_model=self.inverse_model,
            )
            return snapshot

        self.base_algorithm.__orig_get_batch          = self.base_algorithm.get_batch
        self.base_algorithm.__orig_networks           = self.base_algorithm.networks
        self.base_algorithm.__orig_get_epoch_snapshot = self.base_algorithm.get_epoch_snapshot

        self.base_algorithm.get_batch          = wrapped_get_batch
        self.base_algorithm.networks           = wrapped_networks
        self.base_algorithm.get_epoch_snapshot = wrapped_get_epoch_snapshot

    def get_batch(self):
        """Get the next batch of data and log relevant information.

        We log the entropies H[z], H[z|s] and H[s|z]. If we are using a binary skill
        encoding, then we also log the per-bit conditional entropy H[z_i|s].
        """
        batch = self.base_algorithm.__orig_get_batch()
        rewards = batch['rewards']
        act = batch['actions']
        obs = batch['observations']
        next_obs = batch['next_observations']

        phi      = self.embedding_model(obs)
        next_phi = self.embedding_model(next_obs)
        pred_next_phi = self.forward_model(torch.cat([phi, act], -1))

        # Forward Loss
        forward_loss = F.mse_loss(pred_next_phi, next_phi)

        # Inverse Loss
        pred_act = self.inverse_model(torch.cat([phi, next_phi], -1))
        inverse_loss = nn.MSELoss()(pred_act, act)
        

        # Calculate intrinsic reward
        shaped_rewards = (self.rl_coeff * rewards
                          + torch.norm(next_phi - pred_next_phi, p=2, dim=-1, keepdim=True))

        # Update networks
        self.optimizer.zero_grad()
        (forward_loss + inverse_loss).mean().backward()
        self.optimizer.step()

        # Save some statistics for eval using just one batch.
        if self.base_algorithm.need_to_update_eval_statistics:
            self.base_algorithm.eval_statistics['forward'] = np.mean(ptu.get_numpy(forward_loss))
            self.base_algorithm.eval_statistics['inverse'] = np.mean(ptu.get_numpy(inverse_loss))

        batch['rewards'] = shaped_rewards.detach()
        return batch
