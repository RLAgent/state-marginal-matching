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

### 1. Training a policy on FetchEnv
```
python -m examples.train --intrinsic-reward smm --num-skills 4           # State Marginal Matching (SMM) with 4 latent skills
python -m examples.train --intrinsic-reward none                         # Soft Actor-Critic (SAC)
python -m examples.train --intrinsic-reward icm                          # Intrinsic Curiosity Module (ICM)
python -m examples.train --intrinsic-reward count --count-coeff 10       # Count
python -m examples.train --intrinsic-reward pseudocount --count-coeff 1  # Pseudocount
```
The log directory can be set with `--log-dir /path/to/log/dir`. By default, the log directory is set to `out/`.

### 2. Visualizing a trained policy
```
python -m examples.visualize /path/to/log/dir                               # Without historical averaging
python -m examples.visualize /path/to/log/dir --num-historical-policies 10  # With historical averaging
```

### 3. Evaluating a trained policy
```
python -m examples.eval /path/to/log/dir                               # Without historical averaging
python -m examples.eval /path/to/log/dir --num-historical-policies 10  # With historical averaging
```

To view more flag options, run the scripts with the `--help` flag. For example:
```
python -m examples.train --help
usage: train.py [-h] [--log-dir LOG_DIR] [--cpu] [--snapshot-gap SNAPSHOT_GAP]
                [--env-id {FetchEnv,StarEnv}]
                [--goal-prior {uniform,half}]
                [--shaped-rewards SHAPED_REWARDS]
                [--distance-threshold DISTANCE_THRESHOLD]
                [--init-object-pos-prior INIT_OBJECT_POS_PRIOR]
                [--num-halls NUM_HALLS] [--hall-length HALL_LENGTH]
                [--intrinsic-reward {none,smm,icm,count,pseudocount}]
                [--num-epochs NUM_EPOCHS]
                [--num-steps-per-epoch NUM_STEPS_PER_EPOCH]
                [--num-steps-per-eval NUM_STEPS_PER_EVAL]
                [--max-path-length MAX_PATH_LENGTH] [--batch-size BATCH_SIZE]
                [--discount DISCOUNT] [--net-size NET_SIZE]
                [--reward-scale REWARD_SCALE]
                [--soft-target-tau SOFT_TARGET_TAU] [--policy-lr POLICY_LR]
                [--qf-lr QF_LR] [--vf-lr VF_LR]
                [--target-entropy TARGET_ENTROPY] [--num-skills NUM_SKILLS]
                [--vae-lr VAE_LR] [--vae-beta VAE_BETA] [--rl-coeff RL_COEFF]
                [--state-entropy-coeff STATE_ENTROPY_COEFF]
                [--latent-entropy-coeff LATENT_ENTROPY_COEFF]
                [--latent-conditional-entropy-coeff LATENT_CONDITIONAL_ENTROPY_COEFF]
                [--discriminator-lr DISCRIMINATOR_LR] [--icm-lr ICM_LR]
                [--count-coeff COUNT_COEFF]
                [--count-histogram-bin-width COUNT_HISTOGRAM_BIN_WIDTH]
                [--block-density-only]
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
