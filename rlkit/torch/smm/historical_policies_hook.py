import glob
import joblib
import numpy as np
import os
import re
import rlkit.torch.pytorch_util as ptu


SAMPLE_STRATEGIES = [
    'uniform',      # Sample all checkpoints uniformly.
    'exponential',  # Sample recent checkpoints more often.
    'last',         # Take the last num_historical_policies.
]


class HistoricalPoliciesHook:
    def __init__(
            self,
            base_algorithm,
            log_dir,
            num_historical_policies=10,
            sample_strategy='uniform',
            on_policy_prob=1.0,
    ):
        """
        Samples historical policies at the start of each rollout during
        training. (For eval, it uses the current trained policy.)

        :param base_algorithm: (TorchRLAlgorithm) Base RL algorithm
        :param log_dir: (str) Log directory to load policy checkpoints from
        :param num_historical_policies: (int) # historical policies to sample
        :param sample_strategy (str): How to sample historical policies
        :param on_policy_prob: (float) Probability of sampling on-policy rollout
        """
        assert num_historical_policies > 0
        assert sample_strategy in SAMPLE_STRATEGIES
        assert on_policy_prob >= 0 and on_policy_prob <= 1, on_policy_prob

        self.base_algorithm = base_algorithm
        self.on_policy_prob = on_policy_prob

        self.historical_policies = self.load_historical_policies(
            log_dir, num_historical_policies, sample_strategy)

        # This makes SMM function as a reward shaper, being completely
        # transparent to the blackbox RL algorithm.
        def wrapped_networks():
            return self.base_algorithm.__orig_networks() + self.historical_policies
        def wrapped_get_epoch_snapshot(epoch):
            snapshot = self.base_algorithm.__orig_get_epoch_snapshot(epoch)
            snapshot.update(
                historical_policies=self.historical_policies,
            )
            return snapshot
        def wrapped__start_new_rollout():
            return self._start_new_rollout()

        self.base_algorithm.__orig_networks           = self.base_algorithm.networks
        self.base_algorithm.__orig_get_epoch_snapshot = self.base_algorithm.get_epoch_snapshot
        self.base_algorithm.__orig__start_new_rollout = self.base_algorithm._start_new_rollout
        self.base_algorithm.__orig_exploration_policy = self.base_algorithm.exploration_policy

        self.base_algorithm._start_new_rollout = wrapped__start_new_rollout
        self.base_algorithm.networks           = wrapped_networks
        self.base_algorithm.get_epoch_snapshot = wrapped_get_epoch_snapshot

    def load_historical_policies(self, log_dir, num_historical_policies, sample_strategy):
        historical_policies = []
        ckpt_paths = glob.glob(os.path.join(log_dir, 'itr_*.pkl'))

        # Sort checkpoints by iteration number.
        ckpt_paths_dict = {}
        for ckpt_path in ckpt_paths:
            itr_pkl_str = ckpt_path.split('/')[-1]
            itr_num = int(re.findall(r'\d+', itr_pkl_str)[0])
            ckpt_paths_dict[itr_num] = ckpt_path
        ckpt_paths = [ckpt_paths_dict[key] for key in sorted(ckpt_paths_dict)]

        if sample_strategy == 'uniform':
            num_ckpt_paths = len(ckpt_paths)
            ckpt_prob = np.full(num_ckpt_paths, 1. / num_ckpt_paths)
        elif sample_strategy == 'exponential':
            # Sample later checkpoints exponentially more than earlier ones.
            ckpt_prob = np.array([1.2 ** i for i in range(len(ckpt_paths))], dtype=np.float)
            ckpt_prob /= np.sum(ckpt_prob)
        elif sample_strategy == 'last':
            num_ckpt_paths = len(ckpt_paths)
            ckpt_prob = np.zeros(num_ckpt_paths)
            ckpt_prob[-num_historical_policies:] = 1. / num_historical_policies
        else:
            raise NotImplementedError()
        assert np.isclose(np.sum(ckpt_prob), 1), "ckpt_prob does not sum to 1: sum{} = {}".format(ckpt_prob, np.sum(ckpt_prob))
        assert len(ckpt_prob) == len(ckpt_paths)
        ckpt_paths = list(np.random.choice(ckpt_paths, size=(num_historical_policies-1), replace=False, p=ckpt_prob))
        ckpt_paths.append(os.path.join(log_dir, 'params.pkl'))

        for ckpt_path in ckpt_paths:
            print('Loading ckpt:', ckpt_path)
            data = joblib.load(ckpt_path)
            policy = data['policy']
            policy.to(ptu.device)
            historical_policies.append(policy)
        print("Loaded {} historical policies.".format(len(historical_policies)))

        return historical_policies

    def _sample_policy(self, current_policy):
        """
        With probability (1 - self.on_policy_prob), sample a historical policy.
        Else return the current_policy.
        """
        use_on_policy = bool(np.random.binomial(1, self.on_policy_prob))
        if use_on_policy:
            return current_policy
        else:
            # Sample a model checkpoint.
            return np.random.choice(self.historical_policies)

    def _start_new_rollout(self):
        self.base_algorithm.exploration_policy = self._sample_policy(self.base_algorithm.__orig_exploration_policy)
        return self.base_algorithm.__orig__start_new_rollout()
