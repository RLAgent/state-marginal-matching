import argparse
import numpy as np

from examples.experiment_utils import load_experiment
from rlkit.core import logger
from rlkit.envs.manipulation_env import ManipulationEnv
from rlkit.envs.wrappers import NormalizedBoxEnv
from rlkit.policies.simple import RandomPolicy
from rlkit.samplers.util import rollout
from rlkit.torch.pytorch_util import set_gpu_mode


def visualize_policy(args):
    variant_overwrite = dict(
        params_pkl=args.params_pkl,
        num_historical_policies=args.num_historical_policies,
        env_kwargs=dict(
            reward_type='indicator',
            sample_goal=False,
            shape_rewards=False,
            distance_threshold=0.1,
            terminate_upon_success=False,
            terminate_upon_failure=False,
        )
    )
    if args.logdir == '':
        variant = variant_overwrite
        env = NormalizedBoxEnv(ManipulationEnv(**variant_overwrite['env_kwargs']))
        eval_policy = RandomPolicy(env.action_space)
    else:
        env, _, data, variant = load_experiment(args.logdir, variant_overwrite)
        eval_policy = data['eval_policy'] if args.use_deterministic_policy else data['policy']
        if not args.cpu:
            set_gpu_mode(True)
            eval_policy.cuda()
        print("Loaded policy:", eval_policy)

        if 'smm_kwargs' in variant:
            # Iterate through each latent-conditioned policy.
            num_skills = variant['smm_kwargs']['num_skills']
            print('Running SMM policy with {} skills.'.format(num_skills))
            import rlkit.torch.smm.utils as utils
            class PartialPolicy:
                def __init__(polself, policy):
                    polself._policy = policy
                    polself._num_skills = num_skills
                    polself._z = -1
                    polself.reset()

                def get_action(polself, ob):
                    aug_ob = utils.concat_ob_z(ob, polself._z, polself._num_skills)
                    return polself._policy.get_action(aug_ob)

                def sample_skill(polself):
                    z = np.random.choice(polself._num_skills)
                    return z

                def reset(polself):
                    polself._z = (polself._z + 1) % polself._num_skills
                    print("Using skill z:", polself._z)
                    return polself._policy.reset()

            eval_policy = PartialPolicy(eval_policy)

    paths = []
    for _ in range(args.num_episodes):
        eval_policy.reset()
        path = rollout(
            env,
            eval_policy,
            max_path_length=args.max_path_length,
            animated=(not args.norender),
        )
        paths.append(path)
        if hasattr(env, "log_diagnostics"):
            env.log_diagnostics(paths)
        if hasattr(env, "get_diagnostics"):
            diagnostics = env.get_diagnostics(paths)
            for key, val in diagnostics.items():
                logger.record_tabular(key, val)
            logger.dump_tabular(with_prefix=False, with_timestamp=False)
    if hasattr(env, "draw"):
        env.draw(paths, save_dir="")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('logdir', type=str,
                        help='path to the log dir')
    parser.add_argument('--params-pkl', type=str, default='params.pkl',
                        help='Pickle file from which to load the policy')
    parser.add_argument('--norender', action='store_true')
    parser.add_argument('--cpu', action='store_true')
    parser.add_argument('--max-path-length', type=int, default=50,
                        help='Max length of rollout')
    parser.add_argument('--num-episodes', type=int, default=200,
                        help='Number of episodes to simulate.')
    parser.add_argument('--use-deterministic-policy', action='store_true')
    parser.add_argument('--num-historical-policies', type=int, default=0,
                        help='Number of historical policies.')
    args = parser.parse_args()

    visualize_policy(args)
