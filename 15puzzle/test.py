import numpy as np

import gymnasium as gym
from gymnasium.spaces import Discrete, Box, Dict
from gymnasium.utils import seeding


def calculate_manhattan_distance(current_pos, goal_pos):
    return abs(current_pos[0] - goal_pos[0]) + abs(current_pos[1] - goal_pos[1])


def find_tile_position(grid, tile):
    row, col = np.where(grid == tile)
    return row[0], col[0]


def total_manhattan_distance(current_state):
    goal_state = np.arange(1, 17).reshape(4, 4)
    total_manhattan_distance = 0
    for num in range(1, 16):
        current_pos = find_tile_position(current_state, num)
        goal_pos = find_tile_position(goal_state, num)
        total_manhattan_distance += calculate_manhattan_distance(current_pos, goal_pos)
    return total_manhattan_distance


def is_solvable(state):
    permutation = state.flatten()
    inversions = 0
    for i in range(len(permutation)):
        for j in range(i + 1, len(permutation)):
            if permutation[i] == 0 or permutation[j] == 0:
                continue
            if permutation[i] > permutation[j]:
                inversions += 1
    num_rows = state.shape[0]
    if num_rows % 2 == 0:
        empty_row = np.where(state == 0)[0][0]
        return (inversions + empty_row) % 2 == 0
    else:
        return inversions % 2 == 0


class FifteenPuzzleEnv(gym.Env):
    def __init__(self, config=None):
        self.current_steps = 0
        self.max_episode_steps = 100
        self.action_space = Discrete(4)  # 0: left, 1: up, 2: right, 3: down
        self.grid_size = 4
        self.observation_space = Box(low=0, high=self.grid_size ** 2 - 1, shape=(self.grid_size, self.grid_size),
                                     dtype=np.int32)
        self.seed()
        self.reset()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self, seed=None, options=None):
        self.state = np.arange(self.grid_size ** 2)
        self.np_random.shuffle(self.state)
        while not is_solvable(self.state):
            self.np_random.shuffle(self.state)
        self.current_steps = 0
        self.state = self.state.reshape((self.grid_size, self.grid_size))
        self.zero_pos = np.argwhere(self.state == 0)[0]
        return self.state, {}

    def step(self, action):
        self.current_steps += 1
        distance_1 = total_manhattan_distance(self.state)
        valid_move = self.move(action)
        d = self.is_solved() or self.current_steps >= self.max_episode_steps
        distance_2 = total_manhattan_distance(self.state)
        r = distance_1 - distance_2
        if not valid_move:
            r -= 5
        return self.state, r, d, False, {}

    def move(self, action):
        new_zero_pos = np.array(self.zero_pos)
        if action == 0:  # left
            new_zero_pos[1] -= 1
        elif action == 1:  # up
            new_zero_pos[0] -= 1
        elif action == 2:  # right
            new_zero_pos[1] += 1
        elif action == 3:  # down
            new_zero_pos[0] += 1

        if (0 <= new_zero_pos[0] < self.grid_size) and (0 <= new_zero_pos[1] < self.grid_size):
            self.state[self.zero_pos[0], self.zero_pos[1]], self.state[new_zero_pos[0], new_zero_pos[1]] = (
                self.state[new_zero_pos[0], new_zero_pos[1]],
                self.state[self.zero_pos[0], self.zero_pos[1]],
            )
            self.zero_pos = new_zero_pos
            return True
        else:
            return False

    def is_solved(self):
        goal_state = np.arange(1, 17).reshape((self.grid_size, self.grid_size))
        goal_state[3][3] = 0
        return np.array_equal(self.state, goal_state)
