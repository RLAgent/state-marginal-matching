# Default train settings
default_train_config = dict(
    intrinsic_reward='none',  # Choices: 'none', 'smm', 'icm', 'count', 'pseudocount'
    algo='sac',  # RL algorithm
    algo_kwargs=dict(
        batch_size=128,  # Number of samples per batch update
        discount=0.99,  # Discount factor
        eval_deterministic=False,
        max_path_length=50,  # Maximum path length in the environment
        num_epochs=1000,  # Number of training epochs
        num_steps_per_epoch=1000,  # Number of environment steps per epoch
        num_steps_per_eval=1000,  # Number of environment steps per evaluation
        policy_lr=0.0003,  # Policy learning rate
        qf_lr=0.0003,  # Q-function learning rate
        vf_lr=0.0003, # Value-function learning rate
        reward_scale=1,  # Weight of the extrinsic reward relative to the SAC reward
        soft_target_tau=0.001,
        target_entropy=None,
  ),
  net_size=300,  # Number of hidden units
  env_id='ManipulationEnv',
  env_kwargs=dict(
    goal_prior='uniform',  # Target distribution
    shaped_rewards=[  # ManipulationEnv reward shaping terms
      'object_off_table',
      'object_goal_indicator',
      'object_gripper_indicator',
      'action_penalty'
    ],
    distance_threshold=0.1,  # Goal distance threshold
    init_object_pos_prior='center',  # Prior for initial object position
  ),
)

# Default test settings
default_test_config = dict(
  env_kwargs=dict(
    goal_prior=[1.12871704, 0.46767739, 0.42], # Test-time object goal position
    sample_goal=False,
    shaped_rewards=['object_off_table', 'object_goal_indicator', 'object_gripper_indicator', 'action_penalty'],
    terminate_upon_success=False,
    terminate_upon_failure=False,
  ),
  test_goal=[1.12871704, 0.46767739, 0.42],  
  algo_kwargs=dict(
    max_path_length=50,  # Maximum path length in the environment
    num_episodes=100,  # Number of test episodes
    reward_scale=100,  # Weight of the extrinsic reward relative to the SAC reward
    collection_mode='episodic',  # Each epoch is one episode
    num_updates_per_episode=0,  # Evaluate without additional training
  ),
  smm_kwargs=dict(
    update_p_z_prior_coeff=1,  # p(z) coeff for SMM posterior adaptation (higher value corresponds to more uniform p(z))

    # Turn off SMM reward.
    state_entropy_coeff=0,
    latent_entropy_coeff=0,
    latent_conditional_entropy_coeff=0,
    discriminator_lr=0,
  ),
)

# Default visualization settings
default_visualize_config = dict(
  params_pkl='params.pkl',  # Pickle file to load policy from
  env_kwargs=dict(
      reward_type='indicator',
      sample_goal=False,
      shape_rewards=False,
      distance_threshold=0.1,
      terminate_upon_success=False,
      terminate_upon_failure=False,
  )
)
