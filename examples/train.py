from examples.args import parse_args
from examples.experiment_utils import get_exp_id, create_env
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

        # Overwrite appropriate functions of algorithm.
        smm_algorithm_hook = SMMHook(
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

        # Overwrite appropriate functions of algorithm.
        ICMHook(
            base_algorithm=algorithm,
            embedding_model=embedding_model,
            forward_model=forward_model,
            inverse_model=inverse_model,
            **variant['icm_kwargs'])
    elif intrinsic_reward == 'count':
        count_algorithm_hook = CountHook(
            base_algorithm=algorithm,
            **variant['count_kwargs'])
    elif intrinsic_reward == 'pseudocount':
        density_model = VAEDensity(
            input_size=obs_dim,
            num_skills=0,
            code_dim=128,
            **variant['vae_density_kwargs'],
        )

        # Overwrite appropriate functions of algorithm.
        PseudocountHook(
            base_algorithm=algorithm,
            density_model=density_model,
            **variant['pseudocount_kwargs'])

    algorithm.to(ptu.device)
    algorithm.train()


if __name__ == "__main__":
    args = parse_args()

    variant = dict(
        log_prefix=args.log_dir,
        algo='sac',
        intrinsic_reward=args.intrinsic_reward,
        env_id=args.env_id,
        net_size=args.net_size,
        algo_kwargs=dict(
            num_epochs=args.num_epochs,
            num_steps_per_epoch=args.num_steps_per_epoch,
            num_steps_per_eval=args.num_steps_per_eval,
            max_path_length=args.max_path_length,
            batch_size=args.batch_size,
            discount=args.discount,

            # SAC parameters
            eval_deterministic=False,
            reward_scale=args.reward_scale,
            soft_target_tau=args.soft_target_tau,
            policy_lr=args.policy_lr,
            qf_lr=args.qf_lr,
            vf_lr=args.vf_lr,
            target_entropy=args.target_entropy,
        ),
    )
    if args.env_id == 'ManipulationEnv':
        variant.update(
            env_kwargs=dict(
                goal_prior=args.goal_prior,
                shaped_rewards=args.shaped_rewards,
                init_object_pos_prior=args.init_object_pos_prior,
            ),
        )
    elif args.env_id == 'StarEnv':
        variant.update(
            env_kwargs=dict(
                num_halls=args.num_halls,
                halls_with_goals=list(range(args.num_halls)),
                hall_length=args.hall_length,
            ))
    else:
        raise NotImplementedError('Unrecognized environment: {}'.format(args.env_id))

    variant.update(
        vae_density_kwargs=dict(
            beta=args.vae_beta,
            lr=args.vae_lr,
        )
    )

    if args.intrinsic_reward == 'smm':
        variant.update(
            smm_kwargs=dict(
                num_skills=args.num_skills,
                rl_coeff=args.rl_coeff,
                state_entropy_coeff=args.state_entropy_coeff,
                latent_entropy_coeff=args.latent_entropy_coeff,
                latent_conditional_entropy_coeff=args.latent_conditional_entropy_coeff,
                discriminator_lr=args.discriminator_lr,
                ),
        )
    elif args.intrinsic_reward == 'icm':
        variant.update(
            icm_kwargs=dict(
                rl_coeff=args.rl_coeff,
                lr=args.icm_lr,
            ),
        )
    elif args.intrinsic_reward == 'count':
        variant.update(
            count_kwargs=dict(
                count_coeff=args.count_coeff,
                histogram_axes=[3, 4, 5] if (args.block_density_only and args.env_id == 'ManipulationEnv') else None,
                histogram_bin_width=args.count_histogram_bin_width,
            ),
        )
    elif args.intrinsic_reward == 'pseudocount':
        variant.update(
            pseudocount_kwargs=dict(
                count_coeff=args.count_coeff,
            ),
        )
    elif args.intrinsic_reward == 'none':
        pass
    else:
        raise NotImplementedError('Unrecognized intrinsic_reward: {}'.format(args.algo))

    # Set log directory.
    exp_id = get_exp_id(variant)
    variant.update(exp_id=exp_id)
    log_dir = create_exp_name(variant['exp_id'])
    print('Logging to:', log_dir)
    setup_logger(log_dir=log_dir,
                 variant=variant,
                 snapshot_mode='gap_and_last',
                 snapshot_gap=args.snapshot_gap,
    )

    # Set GPU.
    if not args.cpu:
        ptu.set_gpu_mode(True)

    experiment(variant)
