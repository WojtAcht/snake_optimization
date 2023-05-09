from map import *
from utils import *
import random
import numpy as np
from typing import Optional


class Brain:
    def __init__(self):
        """
        Genotype should be 12 values vector approx. between -1 and 1.
        """

        # Neural network hyperparameter:
        self.layer_shapes = [(12, 4), (4, 1)]
        self.functions = [sqrt, sigmoid, sigmoid]

        # PSO parameters:
        # Note: x is also used as a list of weights for consecutive neural network layers.
        self.x: np.ndarray = get_initial_random_weights(self.layer_shapes)
        self.v: np.ndarray = get_initial_random_weights(self.layer_shapes)
        self.best_x: np.ndarray = self.x
        self.best_fitness_function_value: float = 0.0

    def predict_move(self, map) -> Direction:
        """
        Calculates distances from walls, apple and snake itself.
        Runs neural network on calculated distances and predicts snake's direction.
        """
        dist: list[int] = map.get_distances()
        dist_matrix: np.ndarray = dist_to_matrix(dist)

        # Run neural network:
        result: np.ndarray = dist_matrix

        for m, f in zip(self.x, self.functions):
            result = result @ f(m)

        direction_probabilities = result.T[0]
        directions = [Direction.UP, Direction.RIGHT, Direction.DOWN, Direction.LEFT]
        # Return direction that is the most likely one:
        return directions[np.argmax(direction_probabilities)]

class PSO:
    """
    PSO implementation for snake's Brain objects.
    It should return updated current_brain object (hint: we need to update x and v).
    Attributes:
    w: Particle inertia weight factor (Choose between 0.4 and 0.9)
    c1: Scaling factor to search away from the particle's best known position
    c2: Scaling factor to search away from the swarm's best known position
    """

    def __init__(self, w: float, c1: float, c2: float) -> None:
        self.w = w
        self.c1 = c1
        self.c2 = c2

    def apply(self, current_brain: Brain, current_brain_fitness_function_value: float, best_brain: Brain) -> Brain:
        if current_brain is best_brain:
            return current_brain

        if current_brain.best_fitness_function_value < current_brain_fitness_function_value:
            # TODO: update current_brain
            ...

        # TODO: implement PSO:
        # r1 = 
        # r2 = 
        # current_brain.v =
        # current_brain.x =
        ######################

        return current_brain
