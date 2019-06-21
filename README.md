# Efficient Exploration via State Marginal Matching

This is the reference implementation for the following paper:

[Efficient Exploration via State Marginal Matching](https://sites.google.com/view/state-marginal-matching).  
Lisa Lee\*, Benjamin Eysenbach\*, Emilio Parisotto\*, Eric Xing, Ruslan Salakhutdinov, Sergey Levine. [arXiv preprint](https://arxiv.org/abs/1906.05274), 2019.

# Getting Started

## Installation

This repository is based on [rlkit](https://github.com/vitchyr/rlkit).

1. You can clone this repository by running:
```
git clone https://github.com/RLAgent/state-marginal-matching.git
cd state-marginal-matching
```

All subsequent commands in this README should be run from the top-level directory of this repository (i.e., `/path/to/state-marginal-matching/`).

2. Install [Mujoco 1.5](https://www.roboti.us/index.html) and [mujoco-py](https://github.com/openai/mujoco-py). Note that it requires a Mujoco license.

3. Create and activate conda enviroment:
```
conda env create -f conda_env.yml
source activate smm_env
```
*Note*: If running on Mac OS X, comment out `patchelf`, `box2d`, and `box2d-kengz` in `conda_env.yml`.

To deactivate the conda environment, run `conda deactivate`. To remove it, run `conda env remove -n smm_env`.

## Running the code

### 1. Training a policy on ManipulationEnv
```
python -m train configs/smm_manipulation.json          # State Marginal Matching (SMM) with 4 latent skills
python -m train configs/sac_manipulation.json          # Soft Actor-Critic (SAC)
python -m train configs/icm_manipulation.json          # Intrinsic Curiosity Module (ICM)
python -m train configs/count_manipulation.json        # Count-based Exploration
python -m train configs/pseudocount_manipulation.json  # Pseudocount
```

The log directory can be set with `--log-dir /path/to/log/dir`. By default, the log directory is set to `out/`.

### 2. Visualizing a trained policy
```
python -m visualize /path/to/log/dir                               # Without historical averaging
python -m visualize /path/to/log/dir --num-historical-policies 10  # With historical averaging
```

### 3. Evaluating a trained policy
```
python -m test /path/to/log/dir                                # Without historical averaging
python -m test /path/to/log/dir --config configs/test_ha.json  # With historical averaging
```

To view more flag options, run the scripts with the `--help` flag. For example:
```
$ python -m train --help
Usage: train.py [OPTIONS] CONFIG

Options:
  --cpu
  --log-dir TEXT
  --snapshot-gap INTEGER  How often to save model checkpoints (by # epochs).
  --help                  Show this message and exit.
```


# References

The algorithms are based on the following papers:

[Efficient Exploration via State Marginal Matching](https://sites.google.com/view/state-marginal-matching).  
Lisa Lee\*, Benjamin Eysenbach\*, Emilio Parisotto\*, Eric Xing, Ruslan Salakhutdinov, Sergey Levine. [arXiv preprint](https://arxiv.org/abs/1906.05274), 2019.  

[Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor](https://arxiv.org/abs/1801.01290).  
Tuomas Haarnoja, Aurick Zhou, Pieter Abbeel, Sergey Levine. ICML 2018.

[Curiosity-driven Exploration by Self-supervised Prediction](https://arxiv.org/abs/1705.05363).  
Deepak Pathak, Pulkit Agrawal, Alexei A. Efros, Trevor Darrell. ICML 2017.

[Unifying Count-Based Exploration and Intrinsic Motivation](https://arxiv.org/abs/1606.01868).  
Marc G. Bellemare, Sriram Srinivasan, Georg Ostrovski, Tom Schaul, David Saxton, Remi Munos. NIPS 2016.

# Citation
```
@article{smm2019,
  title={Efficient Exploration via State Marginal Matching},
  author={Lisa Lee and Benjamin Eysenbach and Emilio Parisotto and Eric Xing and Sergey Levine and Ruslan Salakhutdinov},
  year={2019}
}
```
