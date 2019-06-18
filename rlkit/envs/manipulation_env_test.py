import numpy as np

from rlkit.envs.manipulation_env import ManipulationEnv


def rollout_episode(env, policy=None, num_episodes=5, render=False):
    if policy is None:
        policy = lambda o: env.action_space.sample()

    returns_list = []
    for episode in range(num_episodes):
        obs = env.reset()
        done = False
        ret = 0
        while not done:
            action = policy(obs)
            obs, rew, done, info = env.step(action)
            ret += rew
            if render:
                env.render()
        returns_list.append(ret)
    avg_return = np.mean(returns_list)
    print('Average return: {} over {} episodes.'.format(avg_return, num_episodes))
    return avg_return


def run_optimal_policy_object_gripper_indicator(num_episodes=5, render=False):
    env = ManipulationEnv(shaped_rewards=[
        'object_goal_indicator',
        'object_gripper_indicator',
        'object_off_table', 'action_penalty',
    ],
        init_object_pos_prior='center',
        sample_goal=False,
        max_episode_length=10,
        terminate_upon_failure=False)

    def optimal_policy(obs):
        action = np.zeros(4)
        gripper_pos = obs[0:3]
        object_pos = obs[3:6]
        action[:2] = object_pos[:2] - gripper_pos[:2]

        d = np.sum(np.abs(action))
        if d < 0.05 and gripper_pos[2] > object_pos[2]:
            action[2] = -1

        if np.all(np.isclose(object_pos, gripper_pos, 0.03)):
            action = np.zeros(4)
        return action

    rollout_episode(env, policy=optimal_policy, num_episodes=num_episodes, render=render)


def run_optimal_policy_object_gripper_distance_l2(num_episodes=5, render=False):
    env = ManipulationEnv(shaped_rewards=[
        'object_goal_indicator',
        'object_gripper_distance_l2',
        'object_off_table', 'action_penalty',
    ],
        init_object_pos_prior='center',
        sample_goal=False,
        max_episode_length=10,
        terminate_upon_failure=False)

    def optimal_policy(obs):
        action = np.zeros(4)

        gripper_pos = obs[0:3]
        object_pos = obs[3:6]
        action[:3] = object_pos - gripper_pos
        action *= 10
        return action

    rollout_episode(env, policy=optimal_policy, num_episodes=num_episodes, render=render)


def test_reward_object_goal_indicator(num_resets=1000, render=False):
    # Sample objects uniformly over table surface with 'uniform' object goal prior (+0 over table surface).
    # Total reward should always be 0.
    env = ManipulationEnv(shaped_rewards=['object_goal_indicator'],
                          goal_prior='uniform',
                          init_object_pos_prior='uniform',
                          sample_goal=False,
                          terminate_upon_success=False,
                          max_episode_length=1,
                          terminate_upon_failure=False)
    avg_return = rollout_episode(env, num_episodes=num_resets, render=render)
    assert avg_return == 0.0

    # Sample objects uniformly over table surface with 'half' object goal prior (+1 on left side of table, +0 on right side).
    # Total reward should be close to 0.5.
    env = ManipulationEnv(shaped_rewards=['object_goal_indicator'],
                          goal_prior='half',
                          init_object_pos_prior='uniform',
                          sample_goal=False,
                          terminate_upon_success=False,
                          max_episode_length=1,
                          terminate_upon_failure=False)
    avg_return = rollout_episode(env, num_episodes=num_resets, render=render)
    assert np.isclose(avg_return, 0.5, 0.1), avg_return

    # Center of table is considered "right", so return should be 0.
    env = ManipulationEnv(shaped_rewards=['object_goal_indicator'],
                          goal_prior='half',
                          init_object_pos_prior='center',
                          sample_goal=False,
                          terminate_upon_success=False,
                          max_episode_length=1,
                          terminate_upon_failure=False)
    avg_return = rollout_episode(env, num_episodes=num_resets, render=render)
    assert avg_return == 0.0, avg_return


def test_reward_object_off_table(num_episodes=1, terminate_upon_failure=True, render=False):
    env = ManipulationEnv(shaped_rewards=['object_off_table'],
                          init_object_pos_prior='off_table',
                          sample_goal=False,
                          max_episode_length=10,
                          terminate_upon_failure=terminate_upon_failure)
    rollout_episode(env, num_episodes=num_episodes, render=render)


def main():
    render = False
    test_reward_object_goal_indicator(render=render)
    test_reward_object_off_table(render=render)

    # Uncomment to visualize hardcoded optimal policies for different environment reward functions.
    # run_optimal_policy_object_gripper_indicator()
    # run_optimal_policy_object_gripper_distance_l2()


if __name__ == '__main__':
    main()
