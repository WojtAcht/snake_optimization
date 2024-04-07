import numpy as np
import enum


class Direction(enum.Enum):
    LEFT = (-1, 0)
    UP = (0, 1)
    RIGHT = (1, 0)
    DOWN = (0, -1)
    LU = (-1, 1)
    RU = (1, 1)
    LD = (-1, -1)
    RD = (1, -1)


ALL_DIRECTIONS = [
    Direction.UP,
    Direction.RU,
    Direction.RIGHT,
    Direction.RD,
    Direction.DOWN,
    Direction.LD,
    Direction.LEFT,
    Direction.LU,
]


def add_vectors(v1, v2):
    return (v1[0] + v2[0], v1[1] + v2[1])


def mul_vector(v, n):
    return (v[0] * n, v[1] * n)


def sqrt(x):
    return np.sqrt(np.abs(x)) * np.sign(x)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def softmax(x):
    shift_x = x - np.max(x, axis=-1, keepdims=True)
    exp_x = np.exp(shift_x)
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

def shifted_sigmoid(x):
    return sigmoid(x) - sigmoid(0)


def dist_to_matrix(dist):
    """
    Note that weights in NN should have some symmetry, because:
        - side directions should have the same meaning
        - output directions neurons should be linear combination of the same values, but turned 90deg
    That's why we can have only 12 different values instead of 4x18, what caused slow convergence.
    (https://en.wikipedia.org/wiki/Curse_of_dimensionality)
    """
    w_u, w_ru, w_r, w_rd, w_d, w_ld, w_l, w_lu, s_u, s_ru, s_r, s_rd, s_d, s_ld, s_l, s_lu, a_x, a_y = dist

    dist_matrix = np.array(
        [
            [
                w_u,
                (w_ru + w_lu) / 2,
                (w_r + w_l) / 2,
                (w_rd + w_ld) / 2,
                w_d,
                s_u,
                (s_ru + s_lu) / 2,
                (s_r + s_l) / 2,
                (s_rd + s_ld) / 2,
                s_d,
                a_x,
                a_y,
            ],
            [
                w_r,
                (w_ru + w_rd) / 2,
                (w_u + w_d) / 2,
                (w_lu + w_ld) / 2,
                w_l,
                s_r,
                (s_ru + s_rd) / 2,
                (s_u + s_d) / 2,
                (s_lu + s_ld) / 2,
                s_l,
                -a_y,
                a_x,
            ],
            [
                w_d,
                (w_rd + w_ld) / 2,
                (w_r + w_l) / 2,
                (w_ru + w_lu) / 2,
                w_u,
                s_d,
                (s_rd + s_ld) / 2,
                (s_r + s_l) / 2,
                (s_ru + s_lu) / 2,
                s_u,
                -a_x,
                -a_y,
            ],
            [
                w_l,
                (w_lu + w_ld) / 2,
                (w_u + w_d) / 2,
                (w_ru + w_rd) / 2,
                w_r,
                s_l,
                (s_lu + s_ld) / 2,
                (s_u + s_d) / 2,
                (s_ru + s_rd) / 2,
                s_r,
                a_y,
                -a_x,
            ],
        ]
    )

    return dist_matrix
