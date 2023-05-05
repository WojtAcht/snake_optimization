from map import Map
from brain import Brain
from visualization import *
import numpy as np
from typing import Optional

POPULATION_SIZE = 50
EPOCHS_COUNT = 50

brains: list[Brain] = [Brain() for _ in range(POPULATION_SIZE)]
best_brain: Optional[Brain] = None

print(f"no | best | generation")

for i in range(EPOCHS_COUNT):
    # Calculate fitness values:
    fitness_values: np.ndarray = np.array([Map.calculate_expected_result(brain, probe=3) for brain in brains])
    max_fitness_value: float = np.max(fitness_values)
    best_brain: Brain = brains[np.argmax(fitness_values)]

    print(f"{i} | {max_fitness_value} | {np.mean(fitness_values)}")
    # Update brains using PSO:
    brains: list[Brain] = [pso(brain, result, best_brain) for brain, result in zip(brains, fitness_values)]

# Run visualization for best brain:
map = Map()
V = Visualization(map, best_brain, save_frames=False)
V.run()
