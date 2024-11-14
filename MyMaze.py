import numpy as np
import matplotlib.pyplot as plt
import random

class MyMaze:
    def __init__(self, size=(10, 10), goal_state=(9, 9), fences=None):
        self.size = size
        self.goal_state = goal_state
        self.fences = fences if fences else []
        self.maze = np.zeros(self.size)
        self._setup_maze()

    def _setup_maze(self):
        # Set goal state to 2 for visualization
        self.maze[self.goal_state] = 2
        # Set fences to -1
        for fence in self.fences:
            self.maze[fence] = -1

    def display_maze(self, title="Maze Layout"):
        plt.figure(figsize=(6, 6))
        plt.imshow(self.maze, cmap='coolwarm', origin='upper')
        plt.colorbar()
        plt.title(title)
        plt.show()

    def get_maze(self):
        return self.maze

    def get_goal_state(self):
        return self.goal_state

    def get_fences(self):
        return self.fences