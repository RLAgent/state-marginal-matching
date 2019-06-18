import os
import joblib
import json
from rlkit.envs.manipulation_env import ManipulationEnv
from rlkit.envs.star_env import StarEnv
from rlkit.envs.wrappers import NormalizedBoxEnv, AugmentedBoxObservationShapeEnv


def create_env(env_id, env_kwargs, num_skills=0):
    if env_id == 'ManipulationEnv':
        env = NormalizedBoxEnv(ManipulationEnv(**env_kwargs))
        training_env = NormalizedBoxEnv(ManipulationEnv(**env_kwargs))
    elif env_id == 'StarEnv':
        env = NormalizedBoxEnv(StarEnv(**env_kwargs))
        training_env = NormalizedBoxEnv(StarEnv(**env_kwargs))
    else:
        raise NotImplementedError('Unrecognized environment:', env_id)

    # Append skill to observation vector.
    if num_skills > 0:
        env = AugmentedBoxObservationShapeEnv(env, num_skills)
        training_env = AugmentedBoxObservationShapeEnv(env, num_skills)

    return env, training_env


def load_experiment(log_dir, variant_overwrite=dict()):
    """
    Loads environment and trained policy from file.
    """
    # Load variant.json.
    with open(os.path.join(log_dir, 'variant.json')) as json_file:
        variant = json.load(json_file)
    variant['log_dir'] = log_dir
    print("Read variants:")
    print(json.dumps(variant, indent=4, sort_keys=True))

    # Overwrite variants.
    def walk_dict(d, variants_subtree):
        for key, val in d.items():
            if isinstance(val, dict):
                if key not in variants_subtree:
                    print("While overwriting variants, skipping:", key, val)
                else:
                    walk_dict(val, variants_subtree[key])
            else:
                variants_subtree[key] = val
    walk_dict(variant_overwrite, variant)
    print('Overwrote variants:')
    print(json.dumps(variant, indent=4, sort_keys=True))

    # Load trained policy from file.
    if 'params_pkl' in variant:
        pkl_file = variant['params_pkl']
    else:
        pkl_file = 'params.pkl'
    ckpt_path = os.path.join(log_dir, pkl_file)
    print('Loading checkpoint:', ckpt_path)
    data = joblib.load(ckpt_path)
    print('Data:')
    print(data)

    # Create environment.
    num_skills = variant['smm_kwargs']['num_skills'] if variant['intrinsic_reward'] == 'smm' else 0
    env, training_env = create_env(variant['env_id'], variant['env_kwargs'], num_skills)
    print('env.action_space.low.shape:', env.action_space.low.shape)

    return env, training_env, data, variant


def get_exp_id(variant):
    algo_suffix = '-{}'.format(variant['intrinsic_reward'])
    if variant['intrinsic_reward'] == 'none':
        algo_suffix += '-rs{}'.format(variant['algo_kwargs']['reward_scale'])
    elif variant['intrinsic_reward'] == 'smm':
        algo_suffix += '-{}-rl{}-sec{}-lec{}-lcec{}'.format(
            variant['smm_kwargs']['num_skills'],
            variant['smm_kwargs']['rl_coeff'],
            variant['smm_kwargs']['state_entropy_coeff'],
            variant['smm_kwargs']['latent_entropy_coeff'],
            variant['smm_kwargs']['latent_conditional_entropy_coeff'],
        )
    elif variant['intrinsic_reward'] == 'pseudocount':
        algo_suffix += '-cc{}-lr{}-beta{}'.format(
            variant['pseudocount_kwargs']['count_coeff'],
            variant['vae_density_kwargs']['lr'],
            variant['vae_density_kwargs']['beta'],
            )
    elif variant['intrinsic_reward'] == 'count':
        algo_suffix += '-cc{}'.format(
            variant['count_kwargs']['count_coeff'])
    elif variant['intrinsic_reward'] == 'icm':
        algo_suffix += '-rl{}-lr{}'.format(
            variant['icm_kwargs']['rl_coeff'],
            variant['icm_kwargs']['lr'])
    else:
        raise NotImplementedError('Unrecognized intrinsic_reward: {}'.format(variant['intrinsic_reward']))

    if variant['env_id'] == 'StarEnv':
        exp_id = '{}/{}-{}-{}/{}'.format(
            variant['log_prefix'],
            variant['env_id'],
            variant['env_kwargs']['num_halls'],
            variant['env_kwargs']['hall_length'],
            variant['algo'] + algo_suffix,
        )
    elif variant['env_id'] == 'ManipulationEnv':
        exp_id = '{}/{}-{}-{}/{}'.format(
            variant['log_prefix'],
            variant['env_id'],
            variant['env_kwargs']['goal_prior'],
            ','.join(variant['env_kwargs']['shaped_rewards']),
            variant['algo'] + algo_suffix
        )
    else:
        raise NotImplementedError('Unrecognized environment: ', variant['env_id'])

    return exp_id
