import numpy as np
import matplotlib.pyplot as plt

class MazeRL:
    def __init__(self, maze_setup):
        self.maze = maze_setup.get_maze()
        self.goal_state = maze_setup.get_goal_state()
        self.fences = maze_setup.get_fences()
        self.size = self.maze.shape
        self.gamma = 1.0  # Discount factor for evaluation
        self.reward = -1  # Reward at each non-goal state
        self.goal_reward = 0  # Reward at the goal state

    def policy_evaluation(self, policy, iterations=100):
        """
        Evaluate a given policy by updating the value function over iterations.
        """
        V = np.zeros(self.size)  # Initialize value function with zeros

        for _ in range(iterations):
            new_V = np.copy(V)
            for i in range(self.size[0]):
                for j in range(self.size[1]):
                    s = (i, j)
                    if s == self.goal_state:
                        new_V[s] = self.goal_reward
                    elif s in self.fences:
                        new_V[s] = 0  # No value assigned to fence locations
                    else:
                        # Use policy to get the action and next state
                        a = policy[s]
                        s_prime = (max(0, min(i + a[0], self.size[0] - 1)),
                                      max(0, min(j + a[1], self.size[1] - 1)))

                        # Bellman update for the value function
                        new_V[s] = self.reward + self.gamma * V[s_prime]

            V = new_V  # Update the value function after each iteration
        return V

    def policy_evaluation_model_free(self, policy, gamma=0.9, theta=0.0001):
        # Initialize value function V(s) for all states
        V = np.zeros(self.size)
        
        while True:
            delta = 0
            for x in range(self.size[0]):
                for y in range(self.size[1]):
                    s = (x, y)
                    if s == self.goal_state or s in self.fences:
                        continue

                    # Current action `a` from policy π(s)
                    a = policy.get(s, None)
                    if a is None:
                        continue
                    
                    # Calculate next state s' and reward
                    s_prime, reward = self.step(s, a)

                    # Update V(s) based on Bellman equation for policy evaluation
                    v = V[x, y]
                    V[x, y] = reward + gamma * V[s_prime]
                    delta = max(delta, abs(v - V[x, y]))
            
            if delta < theta:
                break
        
        return V
    

    def step(self, s, a):
        # Given current state `s` and action `a`, return next state `s'` and reward
        x, y = s
        if a == "up":
            s_prime = (max(0, x - 1), y)
        elif a == "down":
            s_prime = (min(self.size[0] - 1, x + 1), y)
        elif a == "left":
            s_prime = (x, max(0, y - 1))
        elif a == "right":
            s_prime = (x, min(self.size[1] - 1, y + 1))
        else:
            s_prime = s  # No move if invalid action
        
        # Check if the next state is a fence or out of bounds
        if s_prime in self.fences:
            s_prime = s  # Reset to the same state if it's a fence

        reward = -1 if s_prime != self.goal_state else 0
        return s_prime, reward
    def display_value_function(self, V, title="Value Function"):
        """
        Visualizes the value function on the maze layout.
        """
        plt.figure(figsize=(6, 6))
        plt.imshow(V, cmap='coolwarm', origin='upper')
        plt.colorbar(label="Value")
        plt.title(title)
        
        # Annotate the goal state and fences
        for (i, j), val in np.ndenumerate(self.maze):
            if (i, j) == self.goal_state:
                plt.text(j, i, "Goal", ha='center', va='center', color="red", fontsize=8, weight="bold")
            elif (i, j) in self.fences:
                plt.text(j, i, "Fence", ha='center', va='center', color="black", fontsize=8)
        
        plt.show()

    def get_random_policy(self):
        """
        Generates a random deterministic policy for each state.
        """
        policy = {}
        for i in range(self.size[0]):
            for j in range(self.size[1]):
                # Random action (move up, down, left, or right)
                action = (np.random.choice([-1, 0, 1]), np.random.choice([-1, 0, 1]))
                policy[(i, j)] = action
        return policy

    def get_manual_policy(self, radius=2):
        """
        Creates a policy where actions near the goal state are optimal.
        """
        policy = self.get_random_policy()  # Start with a random policy
        
        # Set optimal actions in the radius of 2 states from the goal
        gx, gy = self.goal_state
        for i in range(max(0, gx - radius), min(self.size[0], gx + radius + 1)):
            for j in range(max(0, gy - radius), min(self.size[1], gy + radius + 1)):
                if (i, j) != self.goal_state and (i, j) not in self.fences:
                    # Move directly towards the goal
                    action = (np.sign(gx - i), np.sign(gy - j))
                    policy[(i, j)] = action
        return policy

    def generate_trajectory(self,start_state, policy, max_steps=100):
        trajectory = []
        state = start_state
        total_reward = 0

        for _ in range(max_steps):
            action = policy.get(state, (0, 0))  # Get action from policy, default to no movement
            next_state, reward = self.step(state, action)
            trajectory.append((state, action, reward))
            total_reward += reward
            state = next_state

            if state == self.goal_state:  # Stop if the goal state is reached
                break

        return trajectory, total_reward





    def task_5_generate_trajectories(self, policy, num_trajectories=10):
        trajectories = []
        rewards = []

        for _ in range(num_trajectories):
            start_state = (np.random.randint(0, self.size[0]), np.random.randint(0, self.size[1]))  # Random initial state
            if start_state == self.goal_state or start_state in self.fences:
                continue  # Skip if initial state is goal or a fence
            trajectory, total_reward = self.generate_trajectory(start_state, policy)
            trajectories.append(trajectory)
            rewards.append(total_reward)

        return trajectories, rewards

