import numpy as np
import torch
import torch.optim as optim
from torch import nn as nn

import rlkit.torch.pytorch_util as ptu
import rlkit.torch.smm.utils as utils
from rlkit.density_models.discretized_density import DiscretizedDensity


class SMMHook:
    """
    State Marginal Matching Hook

    :param base_algorithm: (TorchRLAlgorithm) Base RL algorithm
    :param discriminator: Discriminator model
    :param density_model: Density model
    :param num_skills: (int) Number of latent skills
    :param update_p_z_prior_coeff (float): If set, initializes the latent prior
            p(z) using this value, and updates p(z) after each rollout. By
            default, it is set to None, so p(z) is not updated.
    :param rl_coeff: (float) Weight on the base RL algorithm reward relative to the SMM reward.
    :param state_entropy_coeff: (float) Weight on the state entropy loss.
    :param latent_entropy_coeff: (float) Weight on the latent entropy loss.
    :param latent_conditional_entropy_coeff: (float) Weight on the latent conditional entropy loss.
    :param discriminator_lr: (float) Discriminator learning rate.
    :param optimizer_class: (float) Optimizer class
    """
    def __init__(
            self,
            base_algorithm,
            discriminator,
            density_model,
            num_skills=1,
            update_p_z_prior_coeff=None,
            rl_coeff=1.,
            state_entropy_coeff=1.,
            latent_entropy_coeff=1.,
            latent_conditional_entropy_coeff=1.,
            discriminator_lr=1e-3,
            optimizer_class=optim.Adam,
    ):
        self.base_algorithm = base_algorithm

        self.discriminator = discriminator
        self.density_model = density_model

        self.num_skills = num_skills
        self.p_z = np.full(self.num_skills, 1.0 / self.num_skills)

        self.update_p_z = (update_p_z_prior_coeff is not None)
        if self.update_p_z:
            self._p_z_num_rollouts = update_p_z_prior_coeff * np.ones(num_skills)
            self._p_z_num_success = np.ones(num_skills)

        self.rl_coeff = rl_coeff
        self.state_entropy_coeff = state_entropy_coeff
        self.latent_entropy_coeff = latent_entropy_coeff
        self.latent_conditional_entropy_coeff = latent_conditional_entropy_coeff

        self.discriminator_optimizer = optimizer_class(
            self.discriminator.parameters(),
            lr=discriminator_lr,
        )

        # Do this hack to make wrapping algorithms as easy as possible for rlkit's API
        # This makes SMM function as a reward shaper, being completely transparent to the
        # blackbox RL algorithm
        def wrapped_get_batch():
            return self.get_batch()
        def wrapped__proc_observation(ob, z=None):
            return self._proc_observation(ob, z=z)
        def wrapped__start_new_rollout():
            return self._start_new_rollout()
        def wrapped_networks():
            # The DiscretizedDensity model is not PyTorch friendly, so don't
            # treat it as a PyTorch network.
            if isinstance(self.density_model, DiscretizedDensity):
                return self.base_algorithm.__orig_networks() + [self.discriminator]
            else:
                return self.base_algorithm.__orig_networks() + [self.discriminator, self.density_model]
        def wrapped_get_epoch_snapshot(epoch):
            snapshot = self.base_algorithm.__orig_get_epoch_snapshot(epoch)
            snapshot.update(
                discriminator=self.discriminator,
                density_model=self.density_model,
            )
            return snapshot
        def wrapped__handle_rollout_ending():
            return self._handle_rollout_ending()
        def wrapped_eval_sampler_start_new_rollout():
            return self.eval_sampler_start_new_rollout()
        def wrapped__get_action_and_info(observation):
            return self._get_action_and_info(observation)

        self.base_algorithm.__orig_get_batch          = self.base_algorithm.get_batch
        self.base_algorithm.__orig__proc_observation  = self.base_algorithm._proc_observation
        self.base_algorithm.__orig__start_new_rollout = self.base_algorithm._start_new_rollout
        self.base_algorithm.__orig_networks           = self.base_algorithm.networks
        self.base_algorithm.__orig_get_epoch_snapshot = self.base_algorithm.get_epoch_snapshot
        self.base_algorithm.__orig__handle_rollout_ending = self.base_algorithm._handle_rollout_ending
        self.base_algorithm.__orig_eval_policy = self.base_algorithm.eval_policy
        self.base_algorithm.__orig__eval_sampler_start_new_rollout = self.base_algorithm.eval_sampler.start_new_rollout
        self.base_algorithm.__orig__get_action_and_info = self.base_algorithm._get_action_and_info

        self.base_algorithm.get_batch          = wrapped_get_batch
        self.base_algorithm._proc_observation  = wrapped__proc_observation
        self.base_algorithm._start_new_rollout = wrapped__start_new_rollout
        self.base_algorithm.networks           = wrapped_networks
        self.base_algorithm.get_epoch_snapshot = wrapped_get_epoch_snapshot
        self.base_algorithm._handle_rollout_ending = wrapped__handle_rollout_ending
        self.base_algorithm.eval_sampler.start_new_rollout = wrapped_eval_sampler_start_new_rollout
        self.base_algorithm._get_action_and_info = wrapped__get_action_and_info

    def discriminator_criterion(self, logits, z_tensor):
        z_labels = torch.argmax(z_tensor, 1)
        return nn.CrossEntropyLoss(reduce=False)(logits, z_labels)

    def get_batch(self):
        """Get the next batch of data and log relevant information.

        We log the entropies H[z], H[z|s] and H[s|z]. If we are using a binary skill
        encoding, then we also log the per-bit conditional entropy H[z_i|s].
        """
        batch = self.base_algorithm.__orig_get_batch()
        rewards = batch['rewards']
        obs = batch['observations']

        # Update the density model offline
        density_loss = self.density_model.update(obs)

        # Compute discriminator loss.
        obs_env, z_tensor = self._split_observations(obs)
        discriminator_logits = self.discriminator(obs_env)
        discriminator_loss = self.discriminator_criterion(discriminator_logits, z_tensor)
        discriminator_loss = discriminator_loss.unsqueeze(-1)

        # Compute SMM-shaped reward.
        h_z = np.log(self.num_skills)  # One-hot skill encoding
        h_z *= torch.ones_like(rewards) 
        h_s_z = -self.density_model.get_output_for(obs)
        h_z_s = discriminator_loss  # The discriminator loss should be exactly equal to the marginal entropy.

        pred_log_ratios = self.rl_coeff * rewards + self.state_entropy_coeff * h_s_z
        for tensor in [pred_log_ratios, h_z, h_z_s]:
            expected_shape = (self.base_algorithm.batch_size, 1)
            error_msg = 'Wrong shape. Expected %s, received %s' % (expected_shape, tensor.shape)
            assert tensor.shape == expected_shape, error_msg
        shaped_rewards = (pred_log_ratios
                          + self.latent_entropy_coeff * h_z
                          + self.latent_conditional_entropy_coeff * h_z_s
        ) # be careful here, make sure all tensors are 2D, e.g. (B, 1), or it will force broadcast

        # Update discriminator.
        self.discriminator_optimizer.zero_grad()
        discriminator_loss.mean().backward()
        self.discriminator_optimizer.step()

        # Log statistics for eval using just one batch.
        if self.base_algorithm.need_to_update_eval_statistics:
            self.base_algorithm.eval_statistics['H(Z)'] = np.mean(ptu.get_numpy(h_z))
            self.base_algorithm.eval_statistics['H(S|Z)'] = np.mean(ptu.get_numpy(h_s_z))
            self.base_algorithm.eval_statistics['H(Z|S)'] = np.mean(ptu.get_numpy(h_z_s))
            self.base_algorithm.eval_statistics['log probability'] = density_loss

            # One-hot skill encoding
            discriminator_logits_np = ptu.get_numpy(discriminator_logits)
            discriminator_logits_mean = np.mean(discriminator_logits_np, axis=0)
            discriminator_logits_std = np.std(discriminator_logits_np, axis=0)
            for z in range(self.num_skills):
                self.base_algorithm.eval_statistics["H(Z={}|S) mean".format(z)] = discriminator_logits_mean[z]
            for z in range(self.num_skills):
                self.base_algorithm.eval_statistics["H(Z={}|S) std".format(z)] = discriminator_logits_std[z]

        batch['rewards'] = shaped_rewards.detach()
        return batch

    def _update_latent_prior(self, paths):
        if self.update_p_z:
            is_success = np.float(np.any([info['is_goal'] for info in paths['env_infos']]))
            self._p_z_num_success[self._current_rollout_z] += is_success
            self._p_z_num_rollouts[self._current_rollout_z] += 1

            self.p_z = self._p_z_num_success / self._p_z_num_rollouts
            self.p_z /= np.sum(self.p_z)

            if self.base_algorithm.need_to_update_eval_statistics:
                self.base_algorithm.need_to_update_eval_statistics = False

                # One-hot skill encoding
                for z, p_z in enumerate(self.p_z):
                    self.base_algorithm.eval_statistics["p(z={})".format(z)] = p_z

    def _handle_rollout_ending(self):
        paths = self.base_algorithm._current_path_builder.get_all_stacked()
        self._update_latent_prior(paths=paths)
        return self.base_algorithm.__orig__handle_rollout_ending()

    def _sample_z(self, batch_size=1):
        """ Samples z from p(z)."""
        # One-hot skill encoding: Sample using probabilities in self.p_z.
        return np.random.choice(self.num_skills, p=self.p_z, size=batch_size)

    def _proc_observation(self, ob, z=None):
        if z is None:
            z = self._current_rollout_z
        return utils.concat_ob_z(ob, z, self.num_skills)

    def _split_observations(self, obs):
        return torch.split(obs, [obs.size(-1)-self.num_skills, self.num_skills], 1)

    def _start_new_rollout(self):
        # Sample z for this current rollout
        self._current_rollout_z = self._sample_z()[0]
        return self.base_algorithm.__orig__start_new_rollout()

    def eval_sampler_start_new_rollout(self):
        # Sample new z.
        self._current_rollout_z = self._sample_z()[0]

        class PartialPolicy:
            def __init__(polself, policy, z, num_skills):
                polself._policy = policy
                polself._z = z
                polself._num_skills = num_skills

            def get_action(polself, ob):
                aug_ob = self.base_algorithm._proc_observation(ob, z=polself._z)
                action, agent_info = polself._policy.get_action(aug_ob)
                agent_info['z'] = polself._z
                return action, agent_info

            def reset(polself):
                return polself._policy.reset()

            def parameters(self):
                return self._policy.parameters()

        self.base_algorithm.eval_sampler.policy = PartialPolicy(self.base_algorithm.eval_policy, self._current_rollout_z, self.num_skills)

        return self.base_algorithm.__orig__eval_sampler_start_new_rollout()

    def _get_action_and_info(self, observation):
        action, agent_info = self.base_algorithm.__orig__get_action_and_info(observation)
        agent_info['z'] = self._current_rollout_z
        return action, agent_info
