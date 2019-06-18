import argparse
import os

from examples.experiment_utils import load_experiment
from rlkit.launchers.launcher_util import setup_logger, create_exp_name
import rlkit.torch.pytorch_util as ptu
from rlkit.torch.sac.sac import SoftActorCritic
from rlkit.torch.smm.smm_hook import SMMHook
from rlkit.torch.smm.historical_policies_hook import HistoricalPoliciesHook


def experiment(args):
    if not args.cpu:
      ptu.set_gpu_mode(True)  # optionally set the GPU (default=False)

    variant_overwrite = dict(
        # Evaluate model on num_episodes.
        algo_kwargs=dict(
            reward_scale=args.reward_scale,
            collection_mode='episodic',
            num_episodes=args.num_episodes,
            max_path_length=args.max_path_length,
            render=args.render,

            # Evaluate without additional training
            num_updates_per_episode=0,
            min_num_steps_before_training=(args.max_path_length * args.num_episodes + 1),
        ),

        # Environment settings
        env_kwargs=dict(
            sample_goal=False,
            goal_prior=args.test_goal,
            shaped_rewards=['object_off_table', 'object_goal_indicator', 'object_gripper_indicator', 'action_penalty'],
            terminate_upon_success=False,
            terminate_upon_failure=False,
        ),

        # SMM settings
        smm_kwargs=dict(
            # Posterior adaptation of latent skills p(z)
            update_p_z_prior_coeff=args.update_p_z_prior_coeff,

            # Turn off SMM reward.
            state_entropy_coeff=0,
            latent_entropy_coeff=0,
            latent_conditional_entropy_coeff=0,
            discriminator_lr=0,
        ),
    )

    # Load experiment from file.
    env, _, data, variant = load_experiment(args.logdir, variant_overwrite)
    assert all([a == b for a, b in zip(env.sampled_goal, args.test_goal)])
    variant.update(
        test_goal=list(env.sampled_goal)
    )
    if args.num_historical_policies > 0:
        variant.update(
            historical_policies_kwargs=dict(
                log_dir=args.logdir,
                num_historical_policies=args.num_historical_policies,
                sample_strategy=args.sample_strategy,
                on_policy_prob=args.on_policy_prob,
            )
        )

    # Set log directory.
    exp_id = 'eval/ne{}-mpl{}-{}-rs{}/nhp{}-{}-opp{}'.format(
        args.num_episodes,
        args.max_path_length,
        ','.join(variant_overwrite['env_kwargs']['shaped_rewards']),
        args.reward_scale,
        args.num_historical_policies,
        args.sample_strategy,
        args.on_policy_prob,
    )
    exp_id = create_exp_name(exp_id)
    log_dir = os.path.join(args.logdir, exp_id)
    print('Logging to:', log_dir)
    setup_logger(log_dir=log_dir,
                 variant=variant,
                 snapshot_mode='none',
                 snapshot_gap=50,
    )

    # Load trained model from file.
    policy = data['policy']
    vf = data['vf']
    qf = data['qf']
    algorithm = SoftActorCritic(
        env=env,
        training_env=env,  # can't clone box2d env cause of swig
        save_environment=False,  # can't save box2d env cause of swig
        policy=policy,
        qf=qf,
        vf=vf,
        **variant['algo_kwargs'],
    )

    # Overwrite algorithm for p(z) adaptation (if model is SMM).
    if 'smm_kwargs' in variant:
        discriminator = data['discriminator']
        density_model = data['density_model']
        SMMHook(
            base_algorithm=algorithm,
            discriminator=discriminator,
            density_model=density_model,
            **variant['smm_kwargs'])

    # Overwrite algorithm for historical averaging.
    if args.num_historical_policies > 0:
        HistoricalPoliciesHook(
            base_algorithm=algorithm,
            **variant['historical_policies_kwargs'],
        )

    algorithm.to(ptu.device)
    algorithm.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('logdir', type=str,
                        help='path to the log dir')
    parser.add_argument('--render', action='store_true')
    parser.add_argument('--cpu', action='store_true')

    # Experiment args
    parser.add_argument('--test-goal', type=str, default='1.12871704,0.46767739,0.42',
                        help='Test-time goal (comma-separated xyz-coordinates).')
    parser.add_argument('--num-episodes', type=int, default=100,
                        help='Number of episodes to evaluate.')
    parser.add_argument('--max-path-length', type=int, default=50,
                        help='Max length of rollout')
    parser.add_argument('--reward-scale', type=float, default=100,
                        help='Reward scale for SAC')

    # SMM args
    parser.add_argument('--update-p-z-prior-coeff', type=int, default=1,
                        help='SMM latent prior initialization coefficient for posterior adaptation.')
    parser.add_argument('--num-historical-policies', type=int, default=0,
                        help='Number of historical policies.')
    parser.add_argument('--sample-strategy', type=str, default='exponential',
                        choices=['uniform', 'exponential', 'last'],
                        help='How to sample historical policies.')
    parser.add_argument('--on-policy-prob', type=float, default=0,
                        help='Probability of sampling on-policy rollout (for HistoricalPoliciesHook only).')

    args = parser.parse_args()
    args.test_goal = [float(x) for x in args.test_goal.split(',')]

    experiment(args)
