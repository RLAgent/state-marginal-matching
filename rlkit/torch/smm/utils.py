import numpy as np


def concat_ob_z(ob, z, num_skills):
    """Concatenates the observation to a one-hot encoding of Z."""
    assert np.isscalar(z)
    z_vec = np.zeros(num_skills)
    z_vec[z] = 1
    return np.hstack([ob, z_vec])


def split_aug_ob(aug_ob, num_skills):
    """Splits an augmented observation into the observation and Z."""
    (ob, z_one_hot) = (aug_ob[:-num_skills], aug_ob[-num_skills:])
    z = np.where(z_one_hot == 1)[0][0]
    return (ob, z)


def get_policy_weight_norm(policy):
    return next(policy.parameters()).data.norm()
