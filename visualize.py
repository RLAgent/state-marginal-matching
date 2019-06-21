import click
import numpy as np

from configs.default import default_visualize_config
from experiment_utils import load_experiment
from rlkit.core import logger
from rlkit.envs.manipulation_env import ManipulationEnv
from rlkit.envs.wrappers import NormalizedBoxEnv
from rlkit.policies.simple import RandomPolicy
from rlkit.samplers.util import rollout
from rlkit.torch.pytorch_util import set_gpu_mode


def visualize(log_dir, variant_overwrite, num_episodes, max_path_length, deterministic=False, cpu=False, render=True):
    if log_dir == '':
        variant = variant_overwrite
        env = NormalizedBoxEnv(ManipulationEnv(**variant_overwrite['env_kwargs']))
        eval_policy = RandomPolicy(env.action_space)
    else:
        env, _, data, variant = load_experiment(log_dir, variant_overwrite)
        eval_policy = data['eval_policy'] if deterministic else data['policy']
        if not cpu:
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
    for _ in range(num_episodes):
        eval_policy.reset()
        path = rollout(
            env,
            eval_policy,
            max_path_length=max_path_length,
            animated=render,
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


@click.command()
@click.argument('log-dir', default=None)
@click.option('--num-episodes', default=50, help="Number of episodes")
@click.option('--max-path-length', default=50, help="Maximum path length in the environment")
@click.option('--deterministic', default=False, is_flag=True,
    help="Whether to run deterministic or stochastic policy")
@click.option('--cpu', default=False, is_flag=True, help="Run on CPU")
@click.option('--norender', default=False, is_flag=True)


def main(log_dir, num_episodes, max_path_length, deterministic, cpu, norender):
    variant = default_visualize_config
    visualize(log_dir, variant, num_episodes, max_path_length,
        deterministic=deterministic, cpu=cpu, render=(not norender))


if __name__ == "__main__":
    main()
