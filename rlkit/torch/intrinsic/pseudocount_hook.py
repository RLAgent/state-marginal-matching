import copy
import math
import numpy as np
import torch
import torch.nn.functional as F

import rlkit.torch.pytorch_util as ptu
from rlkit.density_models.discretized_density import DiscretizedDensity


class PseudocountHook:
    """
    Pseudocount

    :param base_algorithm: (TorchRLAlgorithm) Base RL algorithm
    :param density_model: Density model
    :param count_coeff: (float) Weight on the state counting intrinsic reward
    :param online: (bool) Whether to update the counts online
    """
    def __init__(
            self,
            base_algorithm,
            density_model,
            count_coeff=1.0,
            online=True,
    ):
        self.base_algorithm = base_algorithm
        self.density_model = density_model
        self.count_coeff = count_coeff

        self._online = online
        self._n = 0

        # Do this hack to make wrapping algorithms as easy as possible for rlkit's API
        # This makes SMM function as a reward shaper, being completely transparent to the
        # blackbox RL algorithm
        def wrapped_get_batch():
            return self.get_batch()
        def wrapped_networks():
            # The DiscretizedDensity model is not PyTorch friendly, so don't
            # treat it as a PyTorch network.
            if isinstance(self.density_model, DiscretizedDensity):
                return self.base_algorithm.__orig_networks()
            else:
                return self.base_algorithm.__orig_networks() + [self.density_model]
        def wrapped_get_epoch_snapshot(epoch):
            snapshot = self.base_algorithm.__orig_get_epoch_snapshot(epoch)
            snapshot.update(
                density_model=self.density_model,
            )
            return snapshot
        def wrapped__handle_step(observation, action, reward,
                                 next_observation, terminal, agent_info, env_info):
            return self._handle_step(observation, action, reward,
                                     next_observation, terminal, agent_info, env_info)

        self.base_algorithm.__orig_get_batch          = self.base_algorithm.get_batch
        self.base_algorithm.__orig_networks           = self.base_algorithm.networks
        self.base_algorithm.__orig_get_epoch_snapshot = self.base_algorithm.get_epoch_snapshot
        self.base_algorithm.__orig__handle_step       = self.base_algorithm._handle_step

        self.base_algorithm.get_batch          = wrapped_get_batch
        self.base_algorithm.networks           = wrapped_networks
        self.base_algorithm.get_epoch_snapshot = wrapped_get_epoch_snapshot
        self.base_algorithm._handle_step       = wrapped__handle_step

    def _handle_step(self, observation, action, reward,
                     next_observation, terminal, agent_info, env_info):

        # Update the count online
        aug_obs = ptu.from_numpy(observation).float()[None,:]
        log_prob = self.density_model.get_output_for(aug_obs, sample=False)
        self.density_model.update(aug_obs)
        self._n += 1
        stepped_log_prob = self.density_model.get_output_for(aug_obs, sample=False)
        pseudocount = self._calc_pseudocount(log_prob, stepped_log_prob)
        shaping_reward = 1. / torch.sqrt(pseudocount)

        if self._online:
            reward += shaping_reward.item()

        self.base_algorithm.__orig__handle_step(observation, action, reward,
                                                next_observation, terminal, agent_info, env_info)

    def _calc_pseudocount(self, log_prob, stepped_log_prob):
        prediction_gain = F.relu(stepped_log_prob - log_prob)
        return 1. / (torch.exp(self.count_coeff / math.sqrt(self._n) * prediction_gain) - 1.0)

    def get_batch(self):
        """Get the next batch of data and log relevant information.

        We log the entropies H[z], H[z|s] and H[s|z]. If we are using a binary skill
        encoding, then we also log the per-bit conditional entropy H[z_i|s].
        """
        batch = self.base_algorithm.__orig_get_batch()
        rewards = batch['rewards']
        obs = batch['observations']

        # Get the predictive gain
        obs_ = obs.clone()
        log_prob = self.density_model.get_output_for(obs, sample=False)

        stepped_density_model = copy.deepcopy(self.density_model)
        if next(self.density_model.parameters()).is_cuda:
            stepped_density_model.cuda()
        stepped_density_model.update(obs)
        stepped_log_prob = stepped_density_model.get_output_for(obs, sample=False)

        pseudocount = self._calc_pseudocount(log_prob, stepped_log_prob)
        shaping_reward = 1. / torch.sqrt(pseudocount)

        shaped_rewards = rewards
        if not self._online:
            # Calculate intrinsic reward
            shaped_rewards = rewards + shaping_reward

        # Save some statistics for eval using just one batch.
        if self.base_algorithm.need_to_update_eval_statistics:
            self.base_algorithm.eval_statistics['average pseudocount'] = np.mean(ptu.get_numpy(
                pseudocount[pseudocount != float('Inf')]))
            self.base_algorithm.eval_statistics['average pseudocount shaping reward'] = np.mean(
                ptu.get_numpy(shaping_reward))

        batch['rewards'] = shaped_rewards.detach()
        return batch
