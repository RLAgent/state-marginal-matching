"""Simple navigation environment."""

import time
from collections import OrderedDict

import Box2D
import gym
import numpy as np
from Box2D.b2 import circleShape, fixtureDef, polygonShape
from gym import spaces

POINT_RADIUS = 0.1
INNER_RADIUS = 2.0
HALL_WIDTH = 1.0
assert POINT_RADIUS < 0.5 * HALL_WIDTH, 'Point cannot fit through wallway.'
CAP_WIDTH = 1.0

GOAL_RADIUS = HALL_WIDTH / 2.0

VIEWPORT_W = 600
VIEWPORT_H = 400


class StarEnv(gym.Env):
    def __init__(self,
                 num_halls=3,
                 halls_with_goals=range(3),
                 hall_length=10.0,
                 max_episode_length=100):
        min_radius = HALL_WIDTH * num_halls / (2 * np.pi)
        error_msg = ('Inner circle must be big enough for all '
                     'the hallways. For a %d-point star, set'
                     'INNER_RADIUS > %.2f') % (num_halls, min_radius)
        assert HALL_WIDTH * num_halls < 2 * np.pi * INNER_RADIUS, error_msg
        assert num_halls >= 3, 'Must use at least three hallways.'

        self.action_space = spaces.Box(
            low=np.full(2, -1.0),
            high=np.full(2, 1.0),
        )
        self.observation_space = spaces.Box(
            low=np.full(2, np.inf),
            high=np.full(2, -np.inf),
        )
        self._world = Box2D.b2World(gravity=(0, 0))

        self._num_halls = num_halls
        self._hall_length = hall_length
        self._max_episode_length = max_episode_length
        self._walls = self._make_walls(num_halls)
        self._point = self._make_point()
        self._goals = self._get_goals(halls_with_goals)
        self.viewer = None

    def _get_goals(self, halls_with_goals):
        goals = []
        dist = INNER_RADIUS + self._hall_length - 0.5 * CAP_WIDTH

        for hall in halls_with_goals:
            angle = hall / float(self._num_halls) * 2 * np.pi
            x = dist * np.cos(angle)
            y = dist * np.sin(angle)
            goals.append(np.array([x, y]))

        return goals

    def _make_point(self):
        point = self._world.CreateDynamicBody(
            position=(0, 0),
            angle=0.0,
            fixedRotation=True,
            linearDamping=10.0,
            fixtures=fixtureDef(
                shape=circleShape(radius=POINT_RADIUS, pos=(0, 0)),
                density=0.01,
                friction=1.0,
                restitution=0.3,
            ))
        point.bullet = True
        return point

    def _make_walls(self, num_points):
        walls = []
        theta = 2.0 * np.pi / num_points
        rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)],
                                    [np.sin(theta), np.cos(theta)]])

        for point_index in range(num_points):
            corners = np.array([
                [INNER_RADIUS, 0.5 * HALL_WIDTH],
                [INNER_RADIUS + self._hall_length, 0.5 * HALL_WIDTH],
                [INNER_RADIUS + self._hall_length, -0.5 * HALL_WIDTH],
                [INNER_RADIUS, -0.5 * HALL_WIDTH],
            ])
            multi_rotation_matrix = np.linalg.matrix_power(rotation_matrix,
                                                           point_index)
            rotated_corners = np.dot(multi_rotation_matrix, corners.T).T
            walls.append(rotated_corners)

        bodies = []
        for point_index in range(num_points):
            next_point_index = (point_index + 1) % num_points
            vertices = np.array([
                walls[point_index][0],
                walls[point_index][1],
                walls[next_point_index][2],
                walls[next_point_index][3],
            ])

            body = self._world.CreateStaticBody(
                shapes=polygonShape(vertices=vertices.tolist()),
            )
            bodies.append(body)

        walls = []
        for point_index in range(num_points):
            corners = np.array([
                [INNER_RADIUS + self._hall_length - 0.5 * CAP_WIDTH, 0.5 * HALL_WIDTH],
                [INNER_RADIUS + self._hall_length, 0.5 * HALL_WIDTH],
                [INNER_RADIUS + self._hall_length, -0.5 * HALL_WIDTH],
                [INNER_RADIUS + self._hall_length - 0.5 * CAP_WIDTH, -0.5 * HALL_WIDTH],
            ])
            multi_rotation_matrix = np.linalg.matrix_power(rotation_matrix,
                                                           point_index)
            rotated_corners = np.dot(multi_rotation_matrix, corners.T).T
            walls.append(rotated_corners)

        for point_index in range(num_points):
            vertices = np.array([
                walls[point_index][3],
                walls[point_index][2],
                walls[point_index][1],
                walls[point_index][0],
            ])

            body = self._world.CreateStaticBody(
                shapes=polygonShape(vertices=vertices.tolist()),
            )
            bodies.append(body)

        return bodies

    def _get_point_position(self):
        return np.array(self._point.position)  # / self._hall_length

    def reset(self):
        self._point.position = (0, 0)
        self._step_count = 0
        return self._get_point_position()

    def step(self, action):
        self._step_count += 1
        self._point.ApplyForceToCenter(action.tolist(), wake=True)
        self._world.Step(0.01, 1000, 1000)
        self._world.ClearForces()
        obs = self._get_point_position()
        rew, info = self._process_obs(obs)
        done = (self._step_count >= self._max_episode_length)
        return obs, rew, done, info

    def _process_obs(self, obs):
        rew = 0.0
        info = {
            'is_goal': False,
        }
        for idx, goal in enumerate(self._goals):
            if np.linalg.norm(goal - obs) < GOAL_RADIUS:
                rew = 1.0 / len(self._goals)
                info['goal'] = idx
                info['is_goal'] = True
                break
        return rew, info

    def _render(self, mode=None, close=None):
        from gym.envs.classic_control import rendering
        if self.viewer is None:
            self.viewer = rendering.Viewer(VIEWPORT_W, VIEWPORT_H)

        scale = 0.5 * min(VIEWPORT_W, VIEWPORT_H) / (INNER_RADIUS + self._hall_length)
        # Draw the walls.
        for wall in self._walls:
            vertices = np.array(wall.fixtures[0].shape.vertices)
            # Scale the vertices
            vertices *= scale
            vertices += 0.5 * np.array([VIEWPORT_W, VIEWPORT_H])
            wall = rendering.FilledPolygon(vertices)
            self.viewer.add_geom(wall)

        # Draw the point.
        center = self._get_point_position()
        theta = np.linspace(0, 2 * np.pi, 30)
        vertices = POINT_RADIUS * np.vstack([np.cos(theta), np.sin(theta)]).T
        vertices += center
        vertices *= scale
        vertices += 0.5 * np.array([VIEWPORT_W, VIEWPORT_H])
        circle = rendering.FilledPolygon(vertices)
        circle.set_color(0, 0, 1)
        self.viewer.add_onetime(circle)
        self.viewer.render()

    def get_diagnostics(self, paths, **kwargs):
        statistics = OrderedDict()
        num_goal_states_list = []
        goal_states_list = [[] for _ in self._goals]
        for path in paths:
            env_infos = path['env_infos']
            observations = path['observations']

            for g in range(len(self._goals)):
                goal_states_list[g].append(0)
            for t, env_info in enumerate(env_infos):
                if 'goal' in env_info:
                    goal_states_list[env_info['goal']][-1] += 1

            is_goal_list = [env_info['is_goal'] for env_info in env_infos]
            num_goal_states = np.sum(is_goal_list)
            num_goal_states_list.append(num_goal_states)

        statistics['StarEnv.num_goal_states_mean'] = np.mean(num_goal_states_list)
        statistics['StarEnv.num_goal_states_std'] = np.std(num_goal_states_list)
        for g in range(len(self._goals)):
            statistics['StarEnv.num_goal{}_states_mean'.format(g)] = np.mean(goal_states_list[g])
            statistics['StarEnv.num_goal{}_states_std'.format(g)] = np.std(goal_states_list[g])

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

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None


if __name__ == '__main__':
    render = True

    env = GymEnv('StarEnv-n3-l10.0-v1')

    env.reset()
    obs_list = []

    while True:
        obs, rew, done, info = env.step(env.action_space.sample())
        obs_list.append(obs)
        if rew > 0:
            print(obs, rew, done, info)
        if render:
            env.render()
            time.sleep(0.01)
