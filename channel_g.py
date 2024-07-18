import numpy as np


def channel_g(x_ue, y_ue, x_bs, y_bs):
    H, fc, c = 25, 2e9, 3e8

    d = np.sqrt((x_ue - x_bs) ** 2 + (y_ue - y_bs) ** 2)
    dd = np.sqrt(d ** 2 + H ** 2)

    if d <= 18:
        PLos = 1
    else:
        PLos = 18 / d + ((d - 18) / d) * np.exp(1) ** (-d / 36)

    PL = PLos * (28 + 22 * np.log10(fc * dd / c)) + (1 - PLos) * (32.4 + 20 * np.log10(fc * dd / c))
    G = np.power(10, -PL / 10)
    return G
