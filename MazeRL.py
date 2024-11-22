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


    def get_random_policy(self):
            policy = {}
            actions = {
                0: (-1, 0),  # Up
                1: (1, 0),   # Down
                2: (0, -1),  # Left
                3: (0, 1)    # Right
            }

            # Iterate over all cells in the maze
            for i in range(self.size[0]):
                for j in range(self.size[1]):
                    s = (i, j)
                    if s == self.goal_state or s in self.fences:
                        policy[s] = None  # Explicitly set None for goal and fence states
                        continue

                    # Assign a random action (0, 1, 2, 3) to non-goal and non-fence states
                    random_action = np.random.choice(list(actions.keys()))
                    policy[s] = random_action

            return policy

    def policy_evaluation(self, policy, gamma=0.9, iterations=100):

        V = {s: 0 for s in policy.keys()}  # Initialize values to 0
        
        for _ in range(iterations):
            new_V = V.copy()
            for s in policy.keys():
                if s == self.goal_state:
                    new_V[s] = 0  # Goal state always has value 0
                    continue
                    
                if s in self.fences or policy[s] is None:
                    new_V[s] = None  # Skip fence states
                    continue
                    
                action = policy[s]
                s_prime, _ = self.step(s, action)
                
                # Reward is -1 for all transitions except at goal state
                reward = 0 if s_prime == self.goal_state else -1
                
                if s_prime in V and V[s_prime] is not None:
                    new_V[s] = reward + gamma * V[s_prime]
                
            V = new_V
            
        return V

    def policy_improvement(self, V, gamma=0.9):
        """
        Improve the policy based on the current value function.
        Returns a dictionary mapping states to actions.
        """
        policy = {}  # Use dictionary instead of numpy array
        
        for i in range(self.size[0]):
            for j in range(self.size[1]):
                if (i, j) == self.goal_state:
                    policy[(i, j)] = None
                    continue
                if (i, j) in self.fences:
                    policy[(i, j)] = None
                    continue
                    
                best_action = None
                max_value = float('-inf')
                
                # Try all actions and pick the best one
                for a in range(4):
                    next_state, reward = self.step((i, j), a)
                    value = reward + gamma * V[next_state[0], next_state[1]]
                    if value > max_value:
                        max_value = value
                        best_action = a
                        
                policy[(i, j)] = best_action
                
        return policy

    def display_value_function(self, V_dict, title="Optimal Value Function"):
        """
        Visualizes the value function that is in dictionary format.
        V_dict: Dictionary with (x,y) tuples as keys and values as values
        """
        # Convert dictionary to numpy array
        V = np.zeros(self.size)
        for (i, j), value in V_dict.items():
            # Replace None with a very low value or 0
            V[i, j] = float('-inf') if value is None else value
        
        plt.figure(figsize=(8, 8))
        # Create a masked array to handle inf values
        V_masked = np.ma.masked_where(np.isinf(V), V)
        
        plt.imshow(V_masked, cmap='coolwarm', origin='upper')
        plt.colorbar(label="Value")
        plt.title(title)
        
        # Annotate the goal state and fences
        for i in range(self.size[0]):
            for j in range(self.size[1]):
                if (i, j) == self.goal_state:
                    plt.text(j, i, "Goal", ha='center', va='center', color="white", 
                            fontsize=8, weight="bold", bbox=dict(facecolor='red', alpha=0.5))
                elif (i, j) in self.fences:
                    plt.text(j, i, "■", ha='center', va='center', color="black", 
                            fontsize=12, weight="bold")
                else:
                    # Add value annotations for non-None values
                    value = V_dict.get((i, j))
                    if value is not None:
                        plt.text(j, i, f'{value:.1f}', ha='center', va='center',
                                color='black' if value > np.mean([v for v in V_dict.values() if v is not None]) else 'white', 
                                fontsize=7)
        
        plt.grid(True)
        plt.show()


    def run_policy_iteration(self,random_policy):
        # Evaluate the random policy
        V_random = self.policy_evaluation(random_policy)
        
        # Improve the policy
        optimal_policy = self.policy_improvement(V_random)
        
        # Evaluate the improved policy
        V_optimal = self.policy_evaluation(optimal_policy)
        
        return optimal_policy, V_optimal

    def step(self, s, a):
        """Step function for the maze environment."""
        x, y = s
        if a == 0:  # Up
            s_prime = (max(0, x - 1), y)
        elif a == 1:  # Down
            s_prime = (min(self.size[0] - 1, x + 1), y)
        elif a == 2:  # Left
            s_prime = (x, max(0, y - 1))
        elif a == 3:  # Right
            s_prime = (x, min(self.size[1] - 1, y + 1))
        else:
            s_prime = s

        # Check if the next state is a fence or out of bounds
        if s_prime in self.fences:
            s_prime = s  # Reset to the same state if it's a fence

        r = -1 if s_prime != self.goal_state else 0
        return s_prime, r
    







    def get_manual_policy(self, radius=2):
        """
        Creates a policy where actions near the goal state are optimal.
        """
        policy = self.get_random_policy()  # Start with a random policy
        # Set optimal actions in the radius of 2 states from the goal
        gx, gy = self.goal_state
        
        # Convert directional movement to action numbers
        direction_to_action = {
            (-1, 0): 0,  # Up
            (1, 0): 1,   # Down
            (0, -1): 2,  # Left
            (0, 1): 3    # Right
        }
        
        for i in range(max(0, gx - radius), min(self.size[0], gx + radius + 1)):
            for j in range(max(0, gy - radius), min(self.size[1], gy + radius + 1)):
                if (i, j) != self.goal_state and (i, j) not in self.fences:
                    # Determine direction to move towards goal
                    dx = np.sign(gx - i)
                    dy = np.sign(gy - j)
                    
                    # Prioritize the dimension with larger distance
                    if abs(gx - i) > abs(gy - j):
                        action = direction_to_action.get((dx, 0))
                    else:
                        action = direction_to_action.get((0, dy))
                        
                    if action is not None:
                        policy[(i, j)] = action
        
        return policy

    def generate_trajectory(self, start_state, policy, max_steps=10):
        """
        Generates a single trajectory from a given start state based on the provided policy.
        """
        trajectory = []
        s = start_state
        total_reward = 0

        for _ in range(max_steps):
            a = policy.get(s, (0, 0))  # Get action from policy, default to no movement
            s_prime, r = self.step(s, a)
            trajectory.append((s, a, r))
            total_reward += r
            s = s_prime

            if s == self.goal_state:  # Stop if the goal state is reached
                break

        return trajectory, total_reward

    def task_5_generate_trajectories(self, policy, num_trajectories=10):
        """
        Generates multiple trajectories and total rewards using the provided policy.
        """
        trajectories = []
        rewards = []
        max_retries = 20  # Limit retries to avoid infinite loops

        for _ in range(num_trajectories):
            retries = 0
            while retries < max_retries:
                start_state = (np.random.randint(0, self.size[0]), np.random.randint(0, self.size[1]))
                
                # Check if start_state is valid (not goal or fence)
                if start_state != self.goal_state and start_state not in self.fences:
                    trajectory, total_reward = self.generate_trajectory(start_state, policy)
                    trajectories.append(trajectory)
                    rewards.append(total_reward)
                    break  # Successfully generated trajectory, exit the retry loop
                retries += 1

        return trajectories, rewards

    # Q-learning function
    def q_learning(self, num_episodes=500, alpha=0.1, gamma=0.9, epsilon=0.1):
        # """Q-learning
        Q = np.zeros((*self.size, 4))  # Q-table start with zero; 4 possible actions
        rewards_per_episode = []

        for _ in range(num_episodes): 
            s = (np.random.randint(0, self.size[0]), np.random.randint(0, self.size[1]))
            while s in self.fences or s == self.goal_state:
                s = (np.random.randint(0, self.size[0]), np.random.randint(0, self.size[1]))

            total_reward = 0
            done = False
            while not done:
                # Epsilon-greedy action selection
                if np.random.random() < epsilon:
                    a = np.random.choice(4)
                else:
                    a = np.argmax(Q[s[0], s[1]])

                s_prime, reward = self.step(s, a)
                total_reward += reward

                # Q-learning update rule
                best_next_action = np.argmax(Q[s_prime[0], s_prime[1]])
                Q[s[0], s[1], a] += alpha * (
                    reward + gamma * Q[s_prime[0], s_prime[1], best_next_action] - Q[s[0], s[1], a]
                )

                s = s_prime
                done = s == self.goal_state

            rewards_per_episode.append(total_reward)

        return rewards_per_episode, Q

    def sarsa(self, num_episodes=500, alpha=0.1, gamma=0.9, epsilon=0.1):
        """SARSA implementation."""
        Q = np.zeros((*self.size, 4))  # Q-table initialized to zero; 4 possible actions
        rewards_per_episode = []

        for _ in range(num_episodes):
            s = (np.random.randint(0, self.size[0]), np.random.randint(0, self.size[1]))
            while s in self.fences or s == self.goal_state:
                s = (np.random.randint(0, self.size[0]), np.random.randint(0, self.size[1]))

            # Epsilon-greedy action selection
            a = np.random.choice(4) if np.random.random() < epsilon else np.argmax(Q[s[0], s[1]])

            total_reward = 0
            done = False
            while not done:
                s_prime, reward = self.step(s, a)
                total_reward += reward

                # Epsilon-greedy action selection for the next state
                a_prime = np.random.choice(4) if np.random.random() < epsilon else np.argmax(Q[s_prime[0], s_prime[1]])

                # SARSA update rule
                Q[s[0], s[1], a] += alpha * (
                    reward + gamma * Q[s_prime[0], s_prime[1], a_prime] - Q[s[0], s[1], a]
                )

                s, a = s_prime, a_prime
                done = s == self.goal_state

            rewards_per_episode.append(total_reward)

        return rewards_per_episode, Q



    def  plot_accumulated_reward(self,num_episodes=500):
        # Rerunning the Q-learning and SARSA simulations
        q_learning_rewards = []
        num_runs = 5
        for _ in range(num_runs):
            q_rewards, _ = self.q_learning(num_episodes=num_episodes)
            q_learning_rewards.append(q_rewards)

        # Compute averages and variances
        q_learning_mean = np.mean(q_learning_rewards, axis=0)
        q_learning_std = np.std(q_learning_rewards, axis=0)

        # Plot Q-learning results
        plt.figure(figsize=(10, 6))
        for run in q_learning_rewards:
            plt.plot(run, alpha=0.5, label="Individual Run" if run == q_learning_rewards[0] else None)
        plt.plot(q_learning_mean, label="Mean Reward", linewidth=2)
        plt.fill_between(range(num_episodes), q_learning_mean - q_learning_std, q_learning_mean + q_learning_std, alpha=0.2, label="Variance")
        plt.title("Q-Learning: Accumulated Reward")
        plt.xlabel("Episodes")
        plt.ylabel("Accumulated Reward")
        plt.legend()
        plt.show()

    def plot_sarsa_acc_R(self,num_episodes=500):
        # Rerunning the Q-learning and SARSA simulations
        sarsa_rewards = []
        num_runs = 5
        for _ in range(num_runs):
            s_rewards, _ = self.sarsa(num_episodes=num_episodes)

            sarsa_rewards.append(s_rewards)

        # Compute averages and variances
        sarsa_mean = np.mean(sarsa_rewards, axis=0)
        sarsa_std = np.std(sarsa_rewards, axis=0)

        # Plot SARSA results
        plt.figure(figsize=(10, 6))
        for run in sarsa_rewards:
            plt.plot(run, alpha=0.5, label="Individual Run" if run == sarsa_rewards[0] else None)
        plt.plot(sarsa_mean, label="Mean Reward", linewidth=2)
        plt.fill_between(range(num_episodes), sarsa_mean - sarsa_std, sarsa_mean + sarsa_std, alpha=0.2, label="Variance")
        plt.title("SARSA: Accumulated Reward")
        plt.xlabel("Episodes")
        plt.ylabel("Accumulated Reward")
        plt.legend()
        plt.show()


    def sarsa_trajectory(self, num_episodes=500, alpha=0.1, gamma=0.9, epsilon=0.1):
        """SARSA implementation with trajectory generation."""
        Q = np.zeros((*self.size, 4))  # Q-table initialized to zero; 4 possible actions
        rewards_per_episode = []
        trajectories = []

        for _ in range(num_episodes):
            s = (np.random.randint(0, self.size[0]), np.random.randint(0, self.size[1]))
            while s in self.fences or s == self.goal_state:
                s = (np.random.randint(0, self.size[0]), np.random.randint(0, self.size[1]))

            # Epsilon-greedy action selection
            a = np.random.choice(4) if np.random.random() < epsilon else np.argmax(Q[s[0], s[1]])

            total_reward = 0
            trajectory = []  # Store (state, action, reward) for this episode
            done = False

            while not done:
                s_prime, reward = self.step(s, a)
                total_reward += reward

                # Record (state, action, reward) in the trajectory
                trajectory.append((s, a, reward))

                # Epsilon-greedy action selection for the next state
                a_prime = np.random.choice(4) if np.random.random() < epsilon else np.argmax(Q[s_prime[0], s_prime[1]])

                # SARSA update rule
                Q[s[0], s[1], a] += alpha * (
                    reward + gamma * Q[s_prime[0], s_prime[1], a_prime] - Q[s[0], s[1], a]
                )

                s, a = s_prime, a_prime
                done = s == self.goal_state

            rewards_per_episode.append(total_reward)
            trajectories.append(trajectory)  # Store trajectory for this episode

        return rewards_per_episode, Q, trajectories