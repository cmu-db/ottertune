import numpy as np


def get_beta_t(t, **kwargs):
    assert t > 0.
    return 2. * np.log(t / np.sqrt(np.log(2. * t)))


def get_beta_td(t, ndim, bound=1.0, **kwargs):
    assert t > 0.
    assert ndim > 0.
    assert bound > 0.
    bt = 2. * np.log(float(ndim) * t**2 * np.pi**2 / (6. * bound))
    return np.sqrt(bt) if bt > 0. else 0.


_UCB_MAP = {
    'get_beta_t': get_beta_t,
    'get_beta_td': get_beta_td,
}


def get_ucb_beta(ucb_beta, scale=1., **kwargs):
    check_valid(ucb_beta)
    if not isinstance(ucb_beta, float):
        ucb_beta = _UCB_MAP[ucb_beta](**kwargs)
    assert isinstance(ucb_beta, float), type(ucb_beta)
    ucb_beta *= scale
    assert ucb_beta >= 0.0
    return ucb_beta


def check_valid(ucb_beta):
    if isinstance(ucb_beta, float):
        if ucb_beta < 0.0:
            raise ValueError(("Invalid value for 'ucb_beta': {} "
                              "(expected >= 0.0)").format(ucb_beta))
    else:
        if ucb_beta not in _UCB_MAP:
            raise ValueError(("Invalid value for 'ucb_beta': {} "
                              "(expected 'get_beta_t' or 'get_beta_td')").format(ucb_beta))
