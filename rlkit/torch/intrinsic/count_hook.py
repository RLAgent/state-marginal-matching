import numpy as np
import torch

import rlkit.torch.pytorch_util as ptu
from rlkit.density_models.discretized_density import DiscretizedDensity


class CountHook:
    """
    Count-based Exploration

    :param base_algorithm: (TorchRLAlgorithm) Base RL algorithm
    :param count_coeff: (float) Weight on the state counting intrinsic reward
    :param histogram_axes: (list of int) State dimensions to use for the histogram
    :param histogram_bin_width: (float) Length of bin in each dimension for the histogram
    """
    def __init__(
            self,
            base_algorithm,
            count_coeff=1.0,
            histogram_axes=None,
            histogram_bin_width=1.0,
    ):
        self.base_algorithm = base_algorithm
        self.count_coeff = count_coeff

        self.histogram = DiscretizedDensity(
            num_skills=1,
            bin_width=histogram_bin_width,
            axes=histogram_axes,
        )

        # This makes SMM function as a reward shaper, being completely transparent to the
        # blackbox RL algorithm.
        def wrapped_get_batch():
            return self.get_batch()
        def wrapped_networks():
            return self.base_algorithm.__orig_networks()
        def wrapped_get_epoch_snapshot(epoch):
            snapshot = self.base_algorithm.__orig_get_epoch_snapshot(epoch)
            snapshot.update(
                histogram=self.histogram,
            )
            return snapshot
        def wrapped__take_step_in_env(observation):
            return self._take_step_in_env(observation)

        self.base_algorithm.__orig_get_batch          = self.base_algorithm.get_batch
        self.base_algorithm.__orig_networks           = self.base_algorithm.networks
        self.base_algorithm.__orig_get_epoch_snapshot = self.base_algorithm.get_epoch_snapshot
        self.base_algorithm.__orig__take_step_in_env  = self.base_algorithm._take_step_in_env

        self.base_algorithm.get_batch          = wrapped_get_batch
        self.base_algorithm.networks           = wrapped_networks
        self.base_algorithm.get_epoch_snapshot = wrapped_get_epoch_snapshot
        self.base_algorithm._take_step_in_env  = wrapped__take_step_in_env

    def _take_step_in_env(self, observation):
        new_observation = self.base_algorithm.__orig__take_step_in_env(
            observation,
        )

        # Update the count online
        aug_obs = np.concatenate([observation, np.ones((1,))], -1)
        self.histogram._update_ob(aug_obs)
        return new_observation

    def get_batch(self):
        """Get the next batch of data and log relevant information.

        We log the entropies H[z], H[z|s] and H[s|z]. If we are using a binary skill
        encoding, then we also log the per-bit conditional entropy H[z_i|s].
        """
        batch = self.base_algorithm.__orig_get_batch()
        rewards = batch['rewards']
        obs = batch['observations']

        aug_obs = torch.cat([obs, torch.ones_like(obs[:,:-1])], -1)
        count = self.histogram.get_count_for(aug_obs)
        count_reward = 1 / torch.sqrt(count)

        # Calculate intrinsic reward
        shaped_rewards = (rewards
                          + self.count_coeff * count_reward
        ) # be careful here, make sure all tensors are 2D, e.g. (B, 1), or it will force broadcast

        # Save some statistics for eval using just one batch.
        if self.base_algorithm.need_to_update_eval_statistics:
            self.base_algorithm.eval_statistics['average count'] = np.mean(ptu.get_numpy(count))
            self.base_algorithm.eval_statistics['average count shaping reward'] = self.count_coeff * np.mean(ptu.get_numpy(count_reward))

        batch['rewards'] = shaped_rewards.detach()
        return batch
