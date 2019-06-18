import argparse


def parse_args():
    parser = argparse.ArgumentParser(description='State Marginal Matching')

    # Experiment parameters
    parser.add_argument(
        '--log-dir',
        type=str,
        default='out',
        help='Used to define custom logging directories.')
    parser.add_argument(
        '--cpu',
        action='store_true',
        help='Run on CPU only')
    parser.add_argument(
        '--snapshot-gap',
        type=int,
        default=50,
        help='How often to save model checkpoints (by # epochs).')

    # Environment flags
    parser.add_argument(
        '--env-id',
        type=str,
        default='ManipulationEnv',
        help='Environment to run the policy on.',
        choices=['ManipulationEnv', 'StarEnv'])

    # ManipulationEnv parameters
    parser.add_argument(
        '--goal-prior',
        type=str,
        default='uniform',
        help='ManipulationEnv goal prior',
        choices=['uniform', 'half'])
    parser.add_argument(
        '--shaped-rewards',
        type=str,
        default='object_off_table,object_goal_indicator,object_gripper_indicator,action_penalty',
        help='ManipulationEnv shaped reward terms (comma-separated list).')
    parser.add_argument(
        '--distance-threshold',
        type=float,
        default=0.1,
        help='Goal distance threhsold')
    parser.add_argument(
        '--init-object-pos-prior',
        type=str,
        default='center',
        help='Prior for initial object position.')

    # StarEnv parameters
    parser.add_argument(
        '--num-halls',
        type=int,
        default=3,
        help='Number of halls in the StarEnv.')
    parser.add_argument(
        '--hall-length',
        type=float,
        default=10.0,
        help='Length of each hall in the StarEnv.')

    # Training algorithm flags
    parser.add_argument(
        '--intrinsic-reward',
        type=str,
        default='smm',
        help='Which intrinsic reward to use',
        choices=['none', 'smm', 'icm', 'count', 'pseudocount'])
    parser.add_argument(
        '--num-epochs',
        type=int,
        default=1000,
        help='Number of training epochs.')
    parser.add_argument(
        '--num-steps-per-epoch',
        type=int,
        default=1000,
        help='Number of environment steps per epoch.')
    parser.add_argument(
        '--num-steps-per-eval',
        type=int,
        default=1000,
        help='Number of environment steps per evaluation.')
    parser.add_argument(
        '--max-path-length',
        type=int,
        default=50,
        help='Maximum path length in the environment.')
    parser.add_argument(
        '--batch-size',
        type=int,
        default=128,
        help='Samples per batch update.')
    parser.add_argument(
        '--discount',
        type=float,
        default=0.99,
        help='Discount factor.')
    parser.add_argument(
        '--net-size',
        type=int,
        default=300,
        help='Number of hidden units.')

    # SAC flags
    parser.add_argument(
        '--reward-scale',
        type=float,
        default=1,
        help='How much to weigh the extrinsic reward relative to the SAC reward.')
    parser.add_argument(
        '--soft-target-tau',
        type=float,
        default=0.001)
    parser.add_argument(
        '--policy-lr',
        type=float,
        default=3E-4)
    parser.add_argument(
        '--qf-lr',
        type=float,
        default=3E-4)
    parser.add_argument(
        '--vf-lr',
        type=float,
        default=3E-4)
    parser.add_argument(
        '--target-entropy',
        type=float,
        default=None)

    # SMM flags
    parser.add_argument(
        '--num-skills',
        type=int,
        default=4,
        help='Latent dimension of policy.')
    parser.add_argument(
        '--vae-lr',
        type=float,
        default=1e-2,
        help='VAE learning rate')
    parser.add_argument(
        '--vae-beta',
        type=float,
        default=0.5,
        help='Density beta coeff (VAE density model only).')
    parser.add_argument(
        '--rl-coeff',
        type=float,
        default=1.0,
        help='Weight on the extrinsic & SAC reward relative to the intrinsic exploration bonus.')
    parser.add_argument(
        '--state-entropy-coeff',
        type=float,
        default=1.0,
        help='Weight on the state entropy loss.')
    parser.add_argument(
        '--latent-entropy-coeff',
        type=float,
        default=1.0,
        help='Weight on the latent entropy loss.')
    parser.add_argument(
        '--latent-conditional-entropy-coeff',
        type=float,
        default=1.0,
        help='Weight on the latent conditional entropy loss.')
    parser.add_argument(
        '--discriminator-lr',
        type=float,
        default=1e-3,
        help='Discriminator learning rate.')

    # ICM flags
    parser.add_argument(
        '--icm-lr',
        type=float,
        default=1e-3,
        help='ICM learning rate.')

    # Count-based Exploration flags
    parser.add_argument(
        '--count-coeff',
        type=float,
        default=1.0,
        help='Weight on the state counting intrinsic reward (Count and Pseudocount only)')
    parser.add_argument(
        '--count-histogram-bin-width',
        type=float,
        default=1,
        help='Length of bin in each dimension for the discretized density model (Count only)')
    parser.add_argument(
        '--block-density-only',
        action='store_true',
        help='Only use block position for the discretized density model (ManipulationEnv and Count only)')

    args = parser.parse_args()
    args.shaped_rewards = args.shaped_rewards.split(',')

    return args
