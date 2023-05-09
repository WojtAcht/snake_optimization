from utils import *
import random
from itertools import count


class Map:
    _MAX_MOVES_SINCE_EATING_APPLE = 100

    def __init__(self, w=20, h=20):
        self.w = w
        self.h = h
        self.walls = set(
            [(0, i - 1) for i in range(h)]
            + [(i - 1, 0) for i in range(w)]
            + [(w - 1, i) for i in range(h)]
            + [(i, h - 1) for i in range(w)]
        )
        self.last_move = Direction.RIGHT
        self.snake = self.new_snake()
        self.apple = self.new_apple()
        self.live = 0

    def get_fitness(self) -> float:
        eaten_apples = len(self.snake) - 3
        moves_since_eating_apple = min(self._MAX_MOVES_SINCE_EATING_APPLE, self.live)
        return eaten_apples + moves_since_eating_apple / self._MAX_MOVES_SINCE_EATING_APPLE

    def game_ended(self, next_cell: tuple[int, int]) -> bool:
        maximum_live = max(self._MAX_MOVES_SINCE_EATING_APPLE, 5 * len(self.snake))
        return self.live == maximum_live or next_cell in self.walls or next_cell in self.snake

    def move(self, direction: Direction) -> tuple[float, bool]:
        """return actual fitness and game_over flag"""
        next_cell = (self.snake[0][0] + direction[0], self.snake[0][1] + direction[1])

        if self.game_ended(next_cell):
            return self.get_fitness(), True

        if self.apple in self.snake:
            self.live = 0
            # Update apple:
            while self.apple in self.snake:
                self.apple = self.new_apple()
        else:
            self.snake.pop()

        self.snake = [next_cell] + self.snake
        self.live += 1
        return self.get_fitness(), False

    def restart(self):
        self.snake = self.new_snake()
        self.apple = self.new_apple()
        self.live = 0

    def new_apple(self):
        apple = None
        while not apple or apple in self.snake:
            apple = (random.randint(1, self.w - 2), random.randint(1, self.h - 2))
        return apple

    def new_snake(self):
        snake = [
            (random.randint(2, self.w - 3), random.randint(2, self.h - 3))
        ] * 3  # makes snake len. 3 at the beggining
        return snake

    def get_distances(self):
        """return list of distances to walls [<,\,^,/,>,\\,v,/], snake [<,\,^,/,>,\\,v,/] and apple [x,y]"""

        wall_distances = []
        for dir in ALL_DIRECTIONS:
            dir = dir.value
            for i in count(0):
                v = mul_vector(dir, i)
                if add_vectors(self.snake[0], v) in self.walls:
                    wall_distances.append(i)
                    break

        snake_distances = []
        for dir in ALL_DIRECTIONS:
            dir = dir.value
            for i in count(0):
                v = mul_vector(dir, i)
                if add_vectors(self.snake[0], v) in self.walls or add_vectors(self.snake[0], v) in self.snake[1:]:
                    snake_distances.append(i)
                    break

        apple_distance = add_vectors(self.apple, mul_vector(self.snake[0], -1))

        return wall_distances + snake_distances + list(apple_distance)

    def calculate_expected_result(brain, w=20, h=20, probe=100):
        """return expected points result with usage of this brain"""
        result = 0
        for i in range(probe):
            map = Map(w, h)
            end = False
            while not end:
                direction = brain.predict_move(map)
                points, end = map.move(direction.value)
            result += points
        return result / probe
