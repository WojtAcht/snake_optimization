from map import Map
from brain import Brain, PSO
from visualization import *
import numpy as np
from typing import Optional
import pygame

POPULATION_SIZE = 100
EPOCHS_COUNT = 30

if __name__ == "__main__":
    brains: list[Brain] = [Brain() for _ in range(POPULATION_SIZE)]
    best_brain: Optional[Brain] = None
    history: list[np.ndarray] = []

    # TODO: implement PSO
    pso = PSO(w=0.5, c1=0.5, c2=1.0)

    print(f"no | best | generation")

    for i in range(EPOCHS_COUNT):
        # Calculate fitness values:
        fitness_values: np.ndarray = np.array([Map.calculate_expected_result(brain, probe=3) for brain in brains])
        max_fitness_value: float = np.max(fitness_values)
        best_brain: Brain = brains[np.argmax(fitness_values)]
        history.append(fitness_values)

        print(f"{i} | {max_fitness_value} | {np.mean(fitness_values)}")
        # Update brains using PSO:
        brains: list[Brain] = [pso.apply(brain, result, best_brain) for brain, result in zip(brains, fitness_values)]

    # Run visualization for best brain:
    map = Map()
    V = Visualization(map, best_brain, save_frames=False)
    V.run()
    pygame.quit()

    # Plot results:
    P = Plots(history)
    P.plot_best_per_epoch()
    P.plot_pop_quality_per_epoch()
