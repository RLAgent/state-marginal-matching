"""Simple density model that discretizes states."""
import collections

import matplotlib
import numpy as np

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch

import rlkit.torch.smm.utils as utils


class DiscretizedDensity(object):

    def __init__(self, num_skills, bin_width=1.0, axes=None):
        """Initialize the density model.

        Args:
          num_skills: number of densities to simultaneously track
          bin_width: (float) Width of bins used for hashing.
          axes: (list of ints) State dimensions to use for the density model.
            For the StarAnt environment, the first two dimensions are the
            XY coordinates.
        """
        self._num_skills = num_skills
        self._bin_width = bin_width
        self._counter = [collections.Counter() for _ in range(num_skills)]
        self._axes = axes

    def _discretize_ob(self, ob):
        if self._axes:
            ob = [ob[i] for i in self._axes]
        ob = np.array(ob)
        ob = ob / self._bin_width
        ob = np.floor(ob).astype(np.int)
        ob = str(ob.tolist())
        return ob

    def _get_output_for_ob(self, aug_ob):
        """
        Returns the log probability of the given observation.
        """
        ob, z = utils.split_aug_ob(aug_ob, self._num_skills)
        ob = self._discretize_ob(ob)
        count = self._counter[z].get(ob, 1)
        total = max(sum(self._counter[z].values()), 1)
        prob = count / total
        return np.log(prob)

    def get_output_for(self, aug_obs):
        """
        Returns the log probability of the given observation.
        """
        logprob = np.array([self._get_output_for_ob(aug_ob.cpu().numpy()) for aug_ob in aug_obs],
                           dtype=np.float32)
        logprob = torch.from_numpy(logprob)[:, None]
        if aug_obs.is_cuda:
            logprob = logprob.cuda()
        return logprob

    def _get_count_for_ob(self, aug_ob):
        """
        Returns the count of the given observation.
        """
        ob, z = utils.split_aug_ob(aug_ob, self._num_skills)
        ob = self._discretize_ob(ob)
        count = self._counter[z].get(ob, 1)
        return count

    def get_count_for(self, aug_obs):
        """
        Returns the count of the given observation.
        """
        count = np.array([self._get_count_for_ob(aug_ob.cpu().numpy()) for aug_ob in aug_obs],
                         dtype=np.float32)
        count = torch.from_numpy(count)[:, None]
        if aug_obs.is_cuda:
            count = count.cuda()
        return count

    def _update_ob_npy(self, ob, z):
        ob = self._discretize_ob(ob)
        self._counter[z].update([ob])

    def _update_ob(self, aug_ob):
        ob, z = utils.split_aug_ob(aug_ob, self._num_skills)
        ob = self._discretize_ob(ob)
        self._counter[z].update([ob])

    def update(self, aug_obs):
        pre_update_logprob = torch.mean(self.get_output_for(aug_obs)).item()
        for aug_ob in aug_obs:
            self._update_ob(aug_ob.cpu().numpy())
        return pre_update_logprob

    def draw(self, fig_path, data_range=None, z=0, figsize_scale=1):
        if len(data_range.shape) == 1:
            # 1D drawing
            assert data_range is not None, "Currently, automatic x-range for 1D plots is not supported."
            grids, x_grids = np.exp(self.get_grids_1d(data_range, z))
            num_skills, w = grids.shape
            l = 0.5
            fig, ax_list = plt.subplots(1, self._num_skills, sharex=True, sharey=True,
                                        figsize=(figsize_scale * w * num_skills, figsize_scale * l))
            if self._num_skills == 1:
                ax_list = [ax_list]
            for (index, grid) in enumerate(grids):
                x_grid = x_grids[index]
                ax = ax_list[index]
                ax.plot(x_grid, grid)
                ax.set_xticklabels([])
                ax.set_yticklabels([])
                ax.axis('tight')
        else:
            assert len(data_range.shape) == 2, "Only 1D and 2D data supported: {}".format(len(data_range.shape))
            grids = self.get_grids(data_range, z)
            num_skills, w, l = grids.shape
            fig, ax_list = plt.subplots(1, self._num_skills, sharex=True, sharey=True,
                                        figsize=(figsize_scale * w * num_skills, figsize_scale * l))
            if self._num_skills == 1:
                ax_list = [ax_list]
            for (index, grid) in enumerate(grids):
                ax = ax_list[index]
                ax.imshow(grid, interpolation='nearest', cmap='binary')
                ax.set_xticklabels([])
                ax.set_yticklabels([])
                ax.axis('tight')
        plt.subplots_adjust(hspace=0, wspace=0)
        plt.tight_layout()
        plt.savefig(fig_path, transparent=True)
        plt.close()
        return grids

    def get_grids_1d(self, x_range, z=0):
        x_range = x_range / self._bin_width
        grids = []
        x_grids = []
        grid_size_x = int(x_range[1] - x_range[0])
        for z in range(self._num_skills):
            grid = np.zeros(grid_size_x)
            x_grid = np.zeros(grid_size_x)
            for i, x in enumerate(np.linspace(x_range[0], x_range[1], grid_size_x)):
                ob = np.array([x]) * self._bin_width
                ob = utils.concat_ob_z(ob, z, self._num_skills)
                grid[i] = self._get_output_for_ob(ob)
                x_grid[i] = x * self._bin_width
            grids.append(grid)
            x_grids.append(x_grid)
        grids = np.array(grids)
        x_grids = np.array(x_grids)
        return grids, x_grids

    def get_grids(self, xy_range=None, z=0):
        if xy_range is not None:
            min_xy = xy_range[:, 0] / self._bin_width
            max_xy = xy_range[:, 1] / self._bin_width
        else:
            min_xy = np.full(2, np.inf)
            max_xy = np.full(2, -np.inf)
            for z in range(self._num_skills):
                cells = np.array([eval(key) for key in self._counter[z].keys()])
                if len(cells) > 0:
                    max_xy = np.max([max_xy, np.max(cells, axis=0)], axis=0)
                    min_xy = np.min([min_xy, np.min(cells, axis=0)], axis=0)

        grids = []
        grid_size_x = int(max_xy[0] - min_xy[0])
        grid_size_y = int(max_xy[1] - min_xy[1])
        for z in range(self._num_skills):
            grid = np.zeros((grid_size_x, grid_size_y))
            for i, x in enumerate(np.linspace(min_xy[0], max_xy[0], grid_size_x)):
                for j, y in enumerate(np.linspace(min_xy[1], max_xy[1], grid_size_y)):
                    ob = np.array([x, y]) * self._bin_width
                    ob = utils.concat_ob_z(ob, z, self._num_skills)
                    grid[i, j] = self._get_output_for_ob(ob)
            grids.append(grid)
        grids = np.array(grids)

        return grids
