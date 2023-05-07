from brain import *
from map import *
from utils import *

import pygame
import time
import itertools
import matplotlib.pyplot as plt

import sys

pygame.font.init()
myfont = pygame.font.SysFont("Comic Sans MS", 30)


class HumanInput(Brain):
    def __init__(self):
        pass

    def predict_move(self, map: Map, *args, **kwargs):
        dir = None
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                sys.exit(1)
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_LEFT or event.key == ord("a"):
                    dir = Direction.LEFT
                if event.key == pygame.K_DOWN or event.key == ord("s"):
                    dir = Direction.UP
                if event.key == pygame.K_RIGHT or event.key == ord("d"):
                    dir = Direction.RIGHT
                if event.key == pygame.K_UP or event.key == ord("w"):
                    dir = Direction.DOWN

        return dir


class Visualization:
    def __init__(self, map: Map, brain: Brain, save_frames=False):
        self.map = map
        self.brain = brain
        self.save_frames = save_frames

    def run(self):
        screen = pygame.display.set_mode([self.map.w * 20, self.map.h * 20])

        pygame.init()
        direction = Direction.RIGHT
        frames = itertools.count(1000)

        while True:
            direction = self.brain.predict_move(self.map) or direction

            points, end = self.map.move(direction.value)

            # drawing part

            screen.fill((255, 255, 255))

            for cell in self.map.walls:
                pygame.draw.rect(screen, (0, 0, 0), (cell[0] * 20, cell[1] * 20, 20, 20))

            cell = self.map.apple
            pygame.draw.rect(screen, (200, 0, 0), (cell[0] * 20, cell[1] * 20, 20, 20))

            for cell in self.map.snake:
                pygame.draw.rect(screen, (0, 200, 0), (cell[0] * 20, cell[1] * 20, 20, 20))

            cell = self.map.snake[0]
            pygame.draw.rect(screen, (0, 150, 0), (cell[0] * 20, cell[1] * 20, 20, 20))

            textsurface = myfont.render(str((points)), False, (255, 255, 255))
            screen.blit(textsurface, (5, 0))

            if self.save_frames:
                pygame.image.save(screen, f"tmp/frame{next(frames)}.png")

            pygame.display.flip()

            if end:
                time.sleep(1)
                if isinstance(self.brain, HumanInput):
                    self.map.restart()
                else:
                    print(points)
                    break
            time.sleep(1 / 5)



class Plots:
    def __init__(self, history: list[np.ndarray]):
        self.history = history

    def plot_best_per_epoch(self):
        plt.plot(np.max(self.history, axis=1))
        plt.title("Best Fitness Value per Epoch")
        plt.xlabel("Epoch")
        plt.ylabel("Fitness Value (snake's length)")
        plt.show()

    def plot_pop_quality_per_epoch(self):
        plt.boxplot(self.history)
        plt.title("Population Fitness Value per Epoch")
        plt.xlabel("Epoch")
        plt.ylabel("Fitness Value (snake's length)")
        plt.show()


if __name__ == "__main__":
    map = Map()
    brain = HumanInput()

    V = Visualization(map, brain, save_frames=False)
    V.run()
