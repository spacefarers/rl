import gymnasium as gym
from gymnasium import spaces
import numpy as np
import random
import pygame


class SnakeEnv(gym.Env):
    def __init__(self, config=None):
        super(SnakeEnv, self).__init__()

        self.grid_size = 42
        self.observation_space = spaces.Box(low=0, high=255, shape=(self.grid_size, self.grid_size, 3), dtype=np.uint8)
        self.action_space = spaces.Discrete(4)  # 0: up, 1: down, 2: left, 3: right

        self.snake = [(4, 4)]
        self.food = self._spawn_food()
        self.direction = 3  # Initial direction (0: up, 1: down, 2: left, 3: right)
        self.current_steps = 0

        # Pygame initialization
        self.window_size = 300
        self.scale_factor = self.window_size // self.grid_size

    def seed(self, seed=None):
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        return [seed]

    def reset(self, *, seed=None, options=None):
        self.current_steps = 0
        self.snake = [(4, 4)]
        self.food = self._spawn_food()
        self.direction = 3
        return self._get_observation(),{}

    def _spawn_food(self):
        while True:
            food = (random.randint(0, self.grid_size - 1), random.randint(0, self.grid_size - 1))
            if food not in self.snake:
                return food

    def _get_observation(self):
        observation = np.zeros((self.grid_size, self.grid_size, 3), dtype=np.uint8)
        for segment in self.snake:
            observation[segment[0], segment[1]] = [0, 255, 0]  # Snake segment
        observation[self.food[0], self.food[1]] = [255, 0, 0]  # Food
        return observation

    def step(self, action):
        self.current_steps += 1
        self.direction = action

        # Update snake's head based on the direction
        if self.direction == 0:  # Up
            new_head = (self.snake[0][0] - 1, self.snake[0][1])
        elif self.direction == 1:  # Down
            new_head = (self.snake[0][0] + 1, self.snake[0][1])
        elif self.direction == 2:  # Left
            new_head = (self.snake[0][0], self.snake[0][1] - 1)
        else:  # Right
            new_head = (self.snake[0][0], self.snake[0][1] + 1)

        self.snake.insert(0, new_head)

        if self.snake[0] == self.food:
            self.food = self._spawn_food()
        else:
            self.snake.pop()

        # Check for collisions
        game_over = self.current_steps > 5000
        if (new_head in self.snake[1:] or
                new_head[0] < 0 or new_head[0] >= self.grid_size or
                new_head[1] < 0 or new_head[1] >= self.grid_size):
            reward = -100
            game_over = True
        else:
            reward = len(self.snake)

        return self._get_observation(), reward, game_over, False, {}

    def render(self, mode='human'):
        self.window = pygame.display.set_mode((self.window_size, self.window_size))
        pygame.display.set_caption("Snake Game")
        self.window.fill((0, 0, 0))

        for segment in self.snake:
            pygame.draw.rect(self.window, (0, 255, 0),
                             (segment[1] * self.scale_factor, segment[0] * self.scale_factor,
                              self.scale_factor, self.scale_factor))

        pygame.draw.rect(self.window, (255, 0, 0),
                         (self.food[1] * self.scale_factor, self.food[0] * self.scale_factor,
                          self.scale_factor, self.scale_factor))

        pygame.display.flip()



if __name__ == "__main__":
    env = SnakeEnv()
    obs = env.reset()
    done = False

    while not done:
        a = env.action_space.sample()  # Random action for testing
        obs, r, d, _, _ = env.step(a)
        env.render()

    env.close()
