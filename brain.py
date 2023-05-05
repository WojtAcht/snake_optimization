from map import *
from utils import *
import random
import numpy as np
from typing import Optional


SHAPES = [(12, 4), (4, 1)]

ACTIVATION_FUNCTIONS = [sqrt, sigmoid, sigmoid]


class Brain:
    def __init__(self):
        """
        Genotype should be 12 values vector approx. between -1 and 1.
        """

        # PSO parameters:
        self.x: np.ndarray = get_initial_random_weights(SHAPES)
        self.v: np.ndarray = get_initial_random_weights(SHAPES)
        self.best_x: np.ndarray = self.x
        self.best_fitness_function_value: float = 0.0

        # Neural network hyperparameter:
        self.functions = ACTIVATION_FUNCTIONS

    def predict_move(self, map) -> Direction:
        """
        Calculates distances from walls, apple and snake itself.
        Runs neural network on calculated distances and predicts snake's direction.
        """
        dist = map.get_distances()
        dist_matrix: np.ndarray = dist_to_matrix(dist)

        res = dist_matrix

        for m, f in zip(self.x, self.functions):
            res = res @ f(m)

        res = res.T[0]
        directions = [Direction.UP, Direction.RIGHT, Direction.DOWN, Direction.LEFT]
        return directions[np.argmax(res)]


def pso(
    current_brain: Brain,
    current_brain_fitness_function_value: float,
    best_brain: Brain,
    w: float = 0.729,
    c1: float = 2.05,
    c2: float = 2.05,
    lr: float = 0.5,
) -> Brain:
    """
    PSO implementation for snake's Brain objects.
    It should return updated current_brain object (hint: we need to update x and v).
    Arguments:
    - w, c1, c2, lr are PSO hyperparameters.
    """
    if current_brain is best_brain:
        return current_brain

    if current_brain.best_fitness_function_value < current_brain_fitness_function_value:
        current_brain.best_fitness_function_value = current_brain_fitness_function_value
        current_brain.best_x = current_brain.x

    r1 = random.random()
    r2 = random.random()
    current_brain.v = (
        current_brain.v * w
        + (current_brain.best_x - current_brain.x) * r1 * c1
        + (best_brain.x - current_brain.x) * r2 * c2
    )
    current_brain.x = current_brain.x + lr * current_brain.v

    return current_brain
