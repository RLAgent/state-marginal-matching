import click
import json
import os

from configs.default import default_train_config
from experiment_utils import get_exp_id, create_env, overwrite_dict
import rlkit.torch.pytorch_util as ptu
from rlkit.launchers.launcher_util import setup_logger, create_exp_name
from rlkit.torch.sac.policies import TanhGaussianPolicy
from rlkit.torch.sac.sac import SoftActorCritic
from rlkit.torch.networks import FlattenMlp
from rlkit.torch.smm.smm_hook import SMMHook
from rlkit.torch.intrinsic.icm_hook import ICMHook
from rlkit.torch.intrinsic.count_hook import CountHook
from rlkit.torch.intrinsic.pseudocount_hook import PseudocountHook
from rlkit.density_models.vae_density import VAEDensity


def experiment(variant):
    intrinsic_reward = variant['intrinsic_reward']

    # Create environment.
    num_skills = variant['smm_kwargs']['num_skills'] if variant['intrinsic_reward'] == 'smm' else 0
    env, training_env = create_env(variant['env_id'], variant['env_kwargs'], num_skills)
    obs_dim = env.observation_space.low.size
    action_dim = env.action_space.low.size

    # Initialize networks.
    net_size = variant['net_size']
    qf = FlattenMlp(
        input_size=obs_dim + action_dim,
        hidden_sizes=[net_size, net_size],
        output_size=1,
    )
    vf = FlattenMlp(
        input_size=obs_dim,
        hidden_sizes=[net_size, net_size],
        output_size=1,
    )
    policy = TanhGaussianPolicy(
        obs_dim=obs_dim,
        hidden_sizes=[net_size, net_size],
        action_dim=action_dim,
    )
    algorithm = SoftActorCritic(
        env=env,
        training_env=training_env,  # can't clone box2d env cause of swig
        save_environment=False,  # can't save box2d env cause of swig
        policy=policy,
        qf=qf,
        vf=vf,
        **variant['algo_kwargs']
    )

    # Hook classes (SMMHook, ICMHook, CountHook, PseudocountHook) override
    # appropriate methods of `algorithm`.
    if intrinsic_reward == 'smm':
        discriminator = FlattenMlp(
            input_size=obs_dim - num_skills,
            hidden_sizes=[net_size, net_size],
            output_size=num_skills,
        )
        density_model = VAEDensity(
            input_size=obs_dim,
            num_skills=num_skills,
            code_dim=128,
            **variant['vae_density_kwargs'])
        SMMHook(
            base_algorithm=algorithm,
            discriminator=discriminator,
            density_model=density_model,
            **variant['smm_kwargs'])
    elif intrinsic_reward == 'icm':
        embedding_model = FlattenMlp(
            input_size=obs_dim,
            hidden_sizes=[net_size, net_size],
            output_size=net_size,
        )
        forward_model = FlattenMlp(
            input_size=net_size + action_dim,
            hidden_sizes=[net_size, net_size],
            output_size=net_size,
        )
        inverse_model = FlattenMlp(
            input_size=net_size + net_size,
            hidden_sizes=[],
            output_size=action_dim,
        )
        ICMHook(
            base_algorithm=algorithm,
            embedding_model=embedding_model,
            forward_model=forward_model,
            inverse_model=inverse_model,
            **variant['icm_kwargs'])
    elif intrinsic_reward == 'count':
        CountHook(
            base_algorithm=algorithm,
            **variant['count_kwargs'])
    elif intrinsic_reward == 'pseudocount':
        density_model = VAEDensity(
            input_size=obs_dim,
            num_skills=0,
            code_dim=128,
            **variant['vae_density_kwargs'],
        )
        PseudocountHook(
            base_algorithm=algorithm,
            density_model=density_model,
            **variant['pseudocount_kwargs'])

    algorithm.to(ptu.device)
    algorithm.train()


@click.command()
@click.argument('config', default=None)
@click.option('--cpu', default=False, is_flag=True, help="Run on CPU")
@click.option('--log-dir', default='out', help="Output directory")
@click.option('--snapshot-gap', default=50,
    help='How often to save model checkpoints (by # epochs).')


def main(config, cpu, log_dir, snapshot_gap):
    variant = default_train_config
    if config:
        with open(os.path.join(config)) as f:
            exp_params = json.load(f)
        overwrite_dict(variant, exp_params)

    # Set log directory.
    exp_id = get_exp_id(variant)
    variant.update(exp_id=exp_id)
    log_dir = create_exp_name(os.path.join(log_dir, exp_id))
    print('Logging to:', log_dir)
    setup_logger(log_dir=log_dir,
                 variant=variant,
                 snapshot_mode='gap_and_last',
                 snapshot_gap=snapshot_gap,
    )

    # Set GPU.
    if not cpu:
        ptu.set_gpu_mode(True)

    experiment(variant)


if __name__ == "__main__":
    main()
