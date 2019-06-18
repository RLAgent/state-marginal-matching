import numpy as np
from collections import OrderedDict, defaultdict
import os

from gym import spaces
from gym.envs.robotics import rotations, robot_env, utils

from rlkit.density_models.discretized_density import DiscretizedDensity
import rlkit.torch.pytorch_util as ptu
from rlkit.torch.smm.utils import concat_ob_z


MODEL_XML_PATH = os.path.join('fetch', 'pick_and_place.xml')


GOAL_PRIORS = [
    'uniform',   # p(g) is uniform over the table surface.
    'half',      # p(g) is GOAL_PRIOR_HALF_P on left half of the table, and (1 - GOAL_PRIOR_HALF_P) on the right half.
]

INIT_OBJECT_POS_PRIOR = [
    'center',       # Object is always spawned at the center of the table surface.
    'uniform',      # Object is spawned uniformly at random on the table surface.
    'off_table',    # Object is spawned at the edge such that it falls off the table in a few environment steps.
]

SHAPED_REWARDS = {
    'object_off_table': 20.0,           # -1 if object falls off table, +0 otherwise.
    'object_goal_indicator': 1.0,       # If sample_goal=True: +1 if close to sampled goal, +0 otherwise.
                                        # If sample_goal is False:
                                        #   If goal_prior is 'uniform': +0 always.
                                        #   If goal_prior is 'half': +0 on left half, +1 on right half.
    'object_gripper_indicator': 0.1,    # +1 if gripper is close enough to object, 0 otherwise.
    'action_penalty': 0.1,              # Action penalty -||a||
}

TABLE_HEIGHT = 0.42
TABLE_XY_RANGE = np.array([
    [1.0, 1.6],
    [0.34, 1.12]
])
TABLE_XY_MARGIN = 0.1

# If goal_prior is 'half', then goal is on left half of table with probability GOAL_PRIOR_HALF_P.
GOAL_PRIOR_HALF_P = 0.75
TABLE_X_MID = 0.5 * (TABLE_XY_RANGE[0][0] + TABLE_XY_RANGE[0][1])
TABLE_Y_MID = 0.5 * (TABLE_XY_RANGE[1][0] + TABLE_XY_RANGE[1][1])
TABLE_RIGHT_Y_RANGE = np.array([
    TABLE_XY_RANGE[1][0],
    TABLE_Y_MID
])
TABLE_LEFT_Y_RANGE = np.array([
    TABLE_Y_MID,
    TABLE_XY_RANGE[1][1]
])

# Distance threshold for indicator object-gripper reward.
OBJECT_GRIPPER_THRESHOLD = 0.1
OBJECT_GRIPPER_OFFSET = 0.00572


class ManipulationEnv(robot_env.RobotEnv):
    def __init__(self,
        goal_prior='uniform',
        shaped_rewards=['object_goal_indicator', 'object_off_table', 'object_gripper_indicator', 'action_penalty'],
        sample_goal=False,
        distance_threshold=0.1,
        max_episode_length=50,
        terminate_upon_failure=True,
        terminate_upon_success=False,
        init_object_pos_prior='center',
        **kwargs):
        """
        :param goal_prior: Must be in GOAL_PRIORS or a list of three floats (corresponding to goal location).
        :param shaped_rewards: (list) Must be in SHAPED_REWARDS.
        :param sample_goal: Whether to sample a goal from p(g).
        :param distance_threshold: Distance threshold for goal indicator reward.
        :param max_episode_length: Max number of timesteps in an episode.
        :param terminate_upon_success: Whether to terminate episode if object falls off table.
        :param terminate_upon_success: Whether to terminate episode upon reaching a goal state.
        :param init_object_pos_prior: Must be in INIT_OBJECT_POS_PRIOR.
        """
        if isinstance(goal_prior, list):
            assert len(goal_prior) == 3
            assert not sample_goal
            self._goal_prior = None
            self.sampled_goal = np.array(goal_prior, dtype=np.float)
        else:
            assert goal_prior in GOAL_PRIORS
            self._goal_prior = goal_prior
            self.sampled_goal = None

        for shaped_reward in shaped_rewards:
            assert shaped_reward in SHAPED_REWARDS, shaped_reward
        assert init_object_pos_prior in INIT_OBJECT_POS_PRIOR
        assert max_episode_length > 0
        self._shaped_rewards = shaped_rewards
        self._distance_threshold = distance_threshold
        self._max_episode_length = max_episode_length
        self._terminate_upon_failure = terminate_upon_failure
        self._terminate_upon_success = terminate_upon_success
        self._init_object_pos_prior = init_object_pos_prior

        self.gripper_extra_height = 0.2  # additional height above the table when positioning the gripper
        self.block_gripper = False       # whether or not the gripper is blocked (i.e. not movable) or not
        self.obj_range = 0.15            # range of a uniform distribution for sampling initial object positions
        initial_qpos = {
            'robot0:slide0': 0.405,
            'robot0:slide1': 0.48,
            'robot0:slide2': 0.0,
            'object0:joint': [1.25, 0.53, 0.4, 1., 0., 0., 0.],
        }
        super(ManipulationEnv, self).__init__(
            model_path=MODEL_XML_PATH, n_substeps=20, n_actions=4,
            initial_qpos=initial_qpos)

        obs = self._get_obs()['observation']
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=obs.shape, dtype='float32')

        # Initialize object position.
        self.reset()

        if sample_goal:
            self.sampled_goal = self._sample_goal().copy()
            is_goal = self._is_success()[0]
            while is_goal:
                self.sampled_goal = self._sample_goal().copy()
                is_goal = self._is_success()[0]
            print("Sampled goal", self.sampled_goal)
        else:
            # Otherwise, every env step is successful so the episode would terminate after 1 step.
            if self._goal_prior in ['uniform', 'half']:
                assert not terminate_upon_success
                self.sampled_goal = None

    @staticmethod
    def is_off_table(object_pos):
        # Detect if object fell off the table by comparing its z-coord to TABLE_HEIGHT.
        return (object_pos[2] < TABLE_HEIGHT or
                object_pos[0] < TABLE_XY_RANGE[0][0] or
                object_pos[0] > TABLE_XY_RANGE[0][1] or
                object_pos[1] < TABLE_XY_RANGE[1][0] or
                object_pos[1] > TABLE_XY_RANGE[1][1])

    @staticmethod
    def is_on_table_left(object_pos):
        return (object_pos[0] > TABLE_XY_RANGE[0][0] and
                object_pos[0] < TABLE_XY_RANGE[0][1] and
                object_pos[1] > TABLE_LEFT_Y_RANGE[0] and
                object_pos[1] < TABLE_LEFT_Y_RANGE[1])

    @staticmethod
    def compute_distance(pos1, pos2, dist_threshold, ord=2):
        dist = np.linalg.norm(pos1 - pos2, ord=ord)
        is_close = (dist < dist_threshold)
        return dist, is_close

    def reset(self):
        self._step_count = 0

        # Attempt to reset the simulator. Since we randomize initial conditions, it
        # is possible to get into a state with numerical issues (e.g. due to penetration or
        # Gimbel lock) or we may not achieve an initial condition (e.g. an object is within the hand).
        # In this case, we just keep randomizing until we eventually achieve a valid initial
        # configuration.
        did_reset_sim = False
        while not did_reset_sim:
            did_reset_sim = self._reset_sim()

        # Open gripper
        self._set_action(np.array([0, 0, 0, 1], dtype=np.float))
        self.sim.step()
        self._step_callback()

        obs = self._get_obs()
        return obs['observation']

    def step(self, action):
        self._step_count += 1
        info = {}

        action = np.clip(action, self.action_space.low, self.action_space.high)
        self._set_action(action)
        self.sim.step()
        self._step_callback()
        obs = self._get_obs()

        is_goal, off_table, is_on_table_left = self._is_success(object_pos=obs['achieved_goal'])
        info['is_goal'] = is_goal
        info['off_table'] = off_table
        info['is_on_table_left'] = is_on_table_left
        done = (self._step_count >= self._max_episode_length or
                self._terminate_upon_success and is_goal or
                self._terminate_upon_failure and off_table)

        shaped_rewards = self.compute_reward(obs['achieved_goal'], obs['grip_pos'], action)
        reward = np.sum(list(shaped_rewards.values()))
        info['shaped_rewards'] = shaped_rewards

        return obs['observation'], reward, done, info

    def _is_success(self, object_pos=None):
        # Get object position from simulator.
        if object_pos is None:
            object_pos = self.sim.data.get_site_xpos('object0')

        off_table = ManipulationEnv.is_off_table(object_pos)
        is_on_table_left = ManipulationEnv.is_on_table_left(object_pos)

        if self.sampled_goal is None:
            is_goal = not off_table
        else:
            # Successful if object is close enough to the sampled goal.
            goal_dist, is_goal = ManipulationEnv.compute_distance(object_pos, self.sampled_goal, self._distance_threshold, ord=2)
        return is_goal, off_table, is_on_table_left

    def _compute_reward_object_goal_indicator(self, object_pos, sampled_goal=None):
        # If sampled_goal is set: +1 if close to sampled goal, +0 otherwise.
        # If sampled_goal is None:
        #   If goal_prior is 'uniform': +0 always.
        #   If goal_prior is 'half': +0 on right half, +1 on left half of table.
        if sampled_goal is not None:
            is_goal = self._is_success(object_pos)[0]
            r = int(is_goal)
        else:
            if self._goal_prior == 'uniform':
                r = 0
            elif self._goal_prior == 'half':
                is_on_table_left = self._is_success(object_pos)[2]
                r = int(is_on_table_left)
        return r

    def compute_reward(self, object_pos, grip_pos, action):
        """
        Returns a dictionary (str -> float) of reward shaping terms.
        """
        shaped_rewards = {}
        for shaped_reward in self._shaped_rewards:
            # -----------------------------------------------------------------
            # Distance between object & goal
            # -----------------------------------------------------------------
            # -1 if object falls off table, 0 otherwise.
            if shaped_reward == 'object_off_table':
                off_table = self._is_success(object_pos)[1]
                r = - 1.0 * int(off_table)
            # Depends on object position & goal prior (or sampled goal).
            elif shaped_reward == 'object_goal_indicator':
                r = self._compute_reward_object_goal_indicator(object_pos, self.sampled_goal)
            # -----------------------------------------------------------------
            # Distance between object & gripper
            # -----------------------------------------------------------------
            # +1 if gripper is close enough to object, 0 otherwise.
            elif shaped_reward == 'object_gripper_indicator':
                dist, is_close = ManipulationEnv.compute_distance(object_pos, grip_pos, OBJECT_GRIPPER_THRESHOLD, ord=2)
                r = int(is_close)
            # -----------------------------------------------------------------
            # Other reward shaping terms
            # -----------------------------------------------------------------
            # Action penalty -||a||
            elif shaped_reward == 'action_penalty':
                r = - np.square(action).sum()
            else:
                raise NotImplementedError('Unrecognized reward type: {}'.format(shaped_reward))

            r *= SHAPED_REWARDS[shaped_reward]
            shaped_rewards[shaped_reward] = r
        return shaped_rewards

    def _step_callback(self):
        if self.block_gripper:
            self.sim.data.set_joint_qpos('robot0:l_gripper_finger_joint', 0.)
            self.sim.data.set_joint_qpos('robot0:r_gripper_finger_joint', 0.)
            self.sim.forward()

    def _set_action(self, action):
        assert action.shape == (4,)
        action = action.copy()  # ensure that we don't change the action outside of this scope
        pos_ctrl, gripper_ctrl = action[:3], action[3]

        pos_ctrl *= 0.05  # limit maximum change in position
        rot_ctrl = [1., 0., 1., 0.]  # fixed rotation of the end effector, expressed as a quaternion
        gripper_ctrl = np.array([gripper_ctrl, gripper_ctrl])
        assert gripper_ctrl.shape == (2,)
        if self.block_gripper:
            gripper_ctrl = np.zeros_like(gripper_ctrl)
        action = np.concatenate([pos_ctrl, rot_ctrl, gripper_ctrl])

        # Apply action to simulation.
        utils.ctrl_set_action(self.sim, action)
        utils.mocap_set_action(self.sim, action)

    def _get_obs(self):
        # positions
        grip_pos = self.sim.data.get_site_xpos('robot0:grip')
        dt = self.sim.nsubsteps * self.sim.model.opt.timestep
        grip_velp = self.sim.data.get_site_xvelp('robot0:grip') * dt
        robot_qpos, robot_qvel = utils.robot_get_obs(self.sim)

        object_pos = self.sim.data.get_site_xpos('object0')
        # rotations
        object_rot = rotations.mat2euler(self.sim.data.get_site_xmat('object0'))
        # velocities
        object_velp = self.sim.data.get_site_xvelp('object0') * dt
        object_velr = self.sim.data.get_site_xvelr('object0') * dt
        # gripper state
        object_rel_pos = object_pos - grip_pos

        object_velp -= grip_velp
        gripper_state = robot_qpos[-2:]
        gripper_vel = robot_qvel[-2:] * dt  # change to a scalar if the gripper is made symmetric

        obs = np.concatenate([
            grip_pos, object_pos.ravel(), object_rel_pos.ravel(), gripper_state, object_rot.ravel(),
            object_velp.ravel(), object_velr.ravel(), grip_velp, gripper_vel,
        ])

        achieved_goal = np.squeeze(object_pos.copy())
        return {
            'observation': obs.copy(),
            'achieved_goal': achieved_goal.copy(),
            'grip_pos': grip_pos,
        }

    def _viewer_setup(self):
        body_id = self.sim.model.body_name2id('robot0:gripper_link')
        lookat = self.sim.data.body_xpos[body_id]
        for idx, value in enumerate(lookat):
            self.viewer.cam.lookat[idx] = value
        self.viewer.cam.distance = 2.5
        self.viewer.cam.azimuth = 132.
        self.viewer.cam.elevation = -14.

    def _render_callback(self):
        # Visualize goals.
        if self.sampled_goal is None:
            # Hide goal marker inside table.
            goal = (np.mean(TABLE_XY_RANGE[0]), np.mean(TABLE_XY_RANGE[1]), TABLE_HEIGHT - 0.2)
        else:
            goal = self.sampled_goal
        sites_offset = (self.sim.data.site_xpos - self.sim.model.site_pos).copy()
        site_id = self.sim.model.site_name2id('target0')
        self.sim.model.site_pos[site_id] = goal - sites_offset[0]
        self.sim.forward()

    def _reset_sim(self):
        self.sim.set_state(self.initial_state)

        # Randomize start position of object.
        object_qpos = self._sample_object_pos()
        self.sim.data.set_joint_qpos('object0:joint', object_qpos)
        self.sim.forward()

        return True

    def _sample_pos(self, prior):
        """
        Sample a position on the table.
        """
        if prior is None:
            assert self.sampled_goal is not None
            return self.sampled_goal
        elif prior == 'uniform':
            # Sample uniformly on the table.
            pos = np.zeros(3)
            pos[0] = np.random.uniform(TABLE_XY_RANGE[0][0] + TABLE_XY_MARGIN, TABLE_XY_RANGE[0][1] - TABLE_XY_MARGIN)
            pos[1] = np.random.uniform(TABLE_XY_RANGE[1][0] + TABLE_XY_MARGIN, TABLE_XY_RANGE[1][1] - TABLE_XY_MARGIN)
            pos[2] = TABLE_HEIGHT
        elif prior == 'half':
            #  Sample with probability GOAL_PRIOR_HALF_P on left half of the table, and 1-GOAL_PRIOR_HALF_P on the right half.
            pos = np.zeros(3)
            pos[0] = np.random.uniform(TABLE_XY_RANGE[0][0] + TABLE_XY_MARGIN, TABLE_XY_RANGE[0][1] - TABLE_XY_MARGIN)
            pos[2] = TABLE_HEIGHT
            if bool(np.random.binomial(1, GOAL_PRIOR_HALF_P)):
                pos[1] = np.random.uniform(TABLE_LEFT_Y_RANGE[0] + TABLE_XY_MARGIN, TABLE_LEFT_Y_RANGE[1] - TABLE_XY_MARGIN)
            else:
                pos[1] = np.random.uniform(TABLE_RIGHT_Y_RANGE[0] + TABLE_XY_MARGIN, TABLE_RIGHT_Y_RANGE[1] - TABLE_XY_MARGIN)
        else:
            raise NotImplementedError()
        return pos

    def _sample_object_pos(self):
        object_xpos = self.initial_gripper_xpos[:2]
        if self._init_object_pos_prior == 'uniform':
            object_xpos = self._sample_pos('uniform')[:2]
        elif self._init_object_pos_prior == 'center':
            object_xpos = [TABLE_X_MID, TABLE_Y_MID]
        elif self._init_object_pos_prior == 'off_table':
            object_xpos = [ (TABLE_XY_RANGE[0][0]) + 0.05,
                            (TABLE_XY_RANGE[1][0] ) + 0.05]
        else:
            raise NotImplementedError('Unrecognized init_object_pos_prior: {}'.format(self.__init_object_pos_prior))
        object_qpos = self.sim.data.get_joint_qpos('object0:joint')
        assert object_qpos.shape == (7,)
        object_qpos[:2] = object_xpos
        return object_qpos

    def _sample_goal(self):
        return self._sample_pos(self._goal_prior)

    def _env_setup(self, initial_qpos):
        for name, value in initial_qpos.items():
            self.sim.data.set_joint_qpos(name, value)
        utils.reset_mocap_welds(self.sim)
        self.sim.forward()

        # Move end effector into position.
        gripper_target = np.array([-0.498, 0.005, -0.431 + self.gripper_extra_height]) + self.sim.data.get_site_xpos('robot0:grip')
        gripper_rotation = np.array([1., 0., 1., 0.])
        self.sim.data.set_mocap_pos('robot0:mocap', gripper_target)
        self.sim.data.set_mocap_quat('robot0:mocap', gripper_rotation)
        for _ in range(10):
            self.sim.step()

        # Extract information for sampling goals.
        self.initial_gripper_xpos = self.sim.data.get_site_xpos('robot0:grip').copy()
        self.height_offset = self.sim.data.get_site_xpos('object0')[2]

    def render(self, mode='human'):
        super(ManipulationEnv, self).render(mode=mode)

    def _process_pos_statistics(self, z_to_pos_list, pos_axis=None, histogram_bin_width=0.05, statistics=None, name='pos', 
                                compute_kl=False):
        """
        Args:
          statistics (OrderedDict)
          z_to_pos_list (defaultdict(list)) is keyed by skill z.
        """
        # Fit a histogram density model.
        num_skills = max(z_to_pos_list.keys()) + 1
        histogram = DiscretizedDensity(num_skills=1, bin_width=histogram_bin_width)
        histogram_z = DiscretizedDensity(num_skills=num_skills, bin_width=histogram_bin_width)
        obs_list = []
        obs_z_list = []
        pos_list = []
        for z, pos_list_z in z_to_pos_list.items():
            for pos in pos_list_z:
                if pos_axis is not None:
                    pos = [pos[i] for i in pos_axis]
                obs_list.append(concat_ob_z(pos, z=0, num_skills=1))
                obs_z_list.append(concat_ob_z(pos, z=z, num_skills=num_skills))
                pos_list.append(pos)
        obs_list = ptu.from_numpy(np.array(obs_list))
        obs_z_list = ptu.from_numpy(np.array(obs_z_list))

        histogram.update(obs_list)
        histogram_z.update(obs_z_list)

        if statistics:
            pos_entropy = -1 * ptu.get_numpy(histogram.get_output_for(obs_list))
            pos_entropy_mean = np.mean(pos_entropy, axis=0)[0]
            pos_entropy_std = np.std(pos_entropy, axis=0)[0]
            statistics['ManipulationEnv.{}_H[s]_mean'.format(name)] = pos_entropy_mean
            statistics['ManipulationEnv.{}_H[s]_std'.format(name)] = pos_entropy_std

            # Compute KL[pi(s) | p*(s)].
            if compute_kl and self._goal_prior is not None:
                kl_terms = []
                for i, pos in enumerate(pos_list):
                    # Ignore off-table object positions.
                    if not ManipulationEnv.is_off_table(pos):
                        log_pi = pos_entropy[i]
                        log_p = self._compute_reward_object_goal_indicator(pos)
                        kl_terms.append(log_p - log_pi)
                statistics['ManipulationEnv.KL[pi(object_pos)||p*(object_pos)]'.format(name)] = np.mean(kl_terms)

            pos_list = np.array(pos_list)
            pos_min = np.min(pos_list, axis=0)
            pos_max = np.max(pos_list, axis=0)
            pos_mean = np.mean(pos_list, axis=0)
            pos_std = np.std(pos_list, axis=0)
            for i in range(3):
                statistics['ManipulationEnv.{}_min_axis{}'.format(name, i)] = pos_min[i]
                statistics['ManipulationEnv.{}_max_axis{}'.format(name, i)] = pos_max[i]
            for i in range(3):
                statistics['ManipulationEnv.{}_mean_axis{}'.format(name, i)] = pos_mean[i]
            for i in range(3):
                statistics['ManipulationEnv.{}_std_axis{}'.format(name, i)] = pos_std[i]
        return histogram, histogram_z

    def get_diagnostics(self, paths, **kwargs):
        statistics = OrderedDict()
        statistics['ManipulationEnv.num_episodes'] = len(paths)

        success_episodes_list = []  # 1 if at least one state is success, 0 otherwise.
        num_goal_states_list = []
        num_off_table_states_list = []
        num_left_states_list = []
        num_left_goals_list = []
        num_right_goals_list = []
        shaped_returns_list = defaultdict(list)
        object_pos_list = defaultdict(list)  # key: z
        gripper_pos_list = defaultdict(list)  # key: z
        for episode, path in enumerate(paths):
            env_infos = path['env_infos']
            observations = path['observations']
            agent_infos = path['agent_infos']

            shaped_rewards_list = defaultdict(list)
            for env_info in env_infos:
                shaped_rewards = env_info['shaped_rewards']
                for k, v in shaped_rewards.items():
                    shaped_rewards_list[k].append(v)
            for k, v in shaped_rewards_list.items():
                shaped_return = np.sum(v)
                shaped_returns_list[k].append(shaped_return)

            is_goal_list = [env_info['is_goal'] for env_info in env_infos]
            num_goal_states = np.sum(is_goal_list)
            num_goal_states_list.append(num_goal_states)
            success_episodes_list.append(int(num_goal_states > 0))

            off_table_list = [env_info['off_table'] for env_info in env_infos]
            num_off_table_states = np.sum(off_table_list)
            num_off_table_states_list.append(num_off_table_states)

            is_on_table_left_list = [env_info['is_on_table_left'] for env_info in env_infos]
            num_left_states = np.sum(is_on_table_left_list)
            num_left_states_list.append(num_left_states)

            num_right_goals = 0
            num_left_goals = 0
            for is_goal, is_on_table_left in zip(is_goal_list, is_on_table_left_list):
                if is_goal:
                    if is_on_table_left:
                        num_left_goals += 1
                    else:
                        num_right_goals += 1
            num_left_goals_list.append(num_left_goals)
            num_right_goals_list.append(num_right_goals)

            for ob, agent_info in zip(observations, agent_infos):
                if 'z' in agent_info:
                    z = agent_info['z']
                else:
                    z = 0
                gripper_pos = ob[0:3]
                object_pos = ob[3:6]
                gripper_pos_list[z].append(gripper_pos)
                object_pos_list[z].append(object_pos)

        self._process_pos_statistics(object_pos_list, name='object_pos', statistics=statistics, compute_kl=True)
        self._process_pos_statistics(gripper_pos_list, name='gripper_pos', statistics=statistics)

        num_success_episodes = np.sum(success_episodes_list)
        statistics['ManipulationEnv.num_success_episodes'] = num_success_episodes
        success_episode_indices = np.where(np.array(success_episodes_list) == 1)[0]
        if len(success_episode_indices) == 0:
            statistics['ManipulationEnv.num_episodes_till_first_goal'] = len(paths)
        else:
            statistics['ManipulationEnv.num_episodes_till_first_goal'] = success_episode_indices[0]

        statistics['ManipulationEnv.num_goal_states_mean'] = np.mean(num_goal_states_list)
        statistics['ManipulationEnv.num_off_table_states_mean'] = np.mean(num_off_table_states_list)
        statistics['ManipulationEnv.num_left_states_mean'] = np.mean(num_left_states_list)
        statistics['ManipulationEnv.num_left_goals_mean'] = np.mean(num_left_goals_list)
        statistics['ManipulationEnv.num_right_goals_mean'] = np.mean(num_right_goals_list)

        for k, v in shaped_returns_list.items():
            statistics['ManipulationEnv.return_mean:{}'.format(k)] = np.mean(v)
            statistics['ManipulationEnv.return_std:{}'.format(k)] = np.std(v)

        return statistics

    def log_diagnostics(self, paths, **kwargs):
        statistics = self.get_diagnostics(paths, **kwargs)

        max_key_len = 1
        max_val_len = 1
        for k, v in statistics.items():
            num_digits = round(np.log(v) / np.log(10))
            max_val_len = max(max_val_len, num_digits)
            max_key_len = max(max_key_len, len(k))
        num_sig_fig = 2
        max_val_len += (num_sig_fig + 2)  # Add 2 for negative & decimal symbols

        max_key_len = str(int(max_key_len))
        max_val_len = str(int(max_val_len))
        num_sig_fig = str(int(num_sig_fig))
        str_fmt = '{: <' + max_key_len + '}: {:' + max_val_len + '.' + num_sig_fig + 'f}'
        print('----------------------------------------------------------------')
        for k, v in statistics.items():
            print(str_fmt.format(k, v))
        print('----------------------------------------------------------------')

    def draw(self, paths, save_dir, **kwargs):
        object_pos_list = []
        z_to_object_pos = defaultdict(list)
        for path in paths:
            for ob, agent_info in zip(path['observations'], path['agent_infos']):
                if 'z' in agent_info:
                    z = agent_info['z']
                else:
                    z = 0
                object_pos = ob[3:6]
                object_pos_list.append(object_pos)
                z_to_object_pos[z].append(object_pos)
        object_pos_list = np.array(object_pos_list)
        np.save(os.path.join(save_dir, 'object_pos.npy'), object_pos_list)

        histogram_x, _ = self._process_pos_statistics(z_to_object_pos, pos_axis=[0])
        histogram_y, _ = self._process_pos_statistics(z_to_object_pos, pos_axis=[1])
        histogram_x.draw(fig_path=os.path.join(save_dir, 'object_pos_axis_x.png'), data_range=np.array(TABLE_XY_RANGE[0]))
        histogram_y.draw(fig_path=os.path.join(save_dir, 'object_pos_axis_y.png'), data_range=np.array(TABLE_XY_RANGE[1]))

        # Draw (x, y) object_pos.
        # Note: histogram_z refers to skills, not z-axis.
        histogram, histogram_z = self._process_pos_statistics(z_to_object_pos, pos_axis=[0, 1])
        histogram.draw(fig_path=os.path.join(save_dir, 'object_pos_all.png'), data_range=np.array(TABLE_XY_RANGE))
        histogram_z.draw(fig_path=os.path.join(save_dir, 'object_pos_z.png'), data_range=np.array(TABLE_XY_RANGE))
