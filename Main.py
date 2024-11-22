from MyMaze import MyMaze
from MazeRL import MazeRL
import random
import numpy as np
def main():
    model_based = True
    is_task_2 = False
    is_task_3 = False
    is_task_4 = True
    task5_Rpolicy = False
    task5_OPolicy= False
    Q_learning = False
    is_task5_sarsa = False

    # Step 1: Set up the maze environment
    maze_setup = MyMaze(size=(10, 10), goal_state=(9, 9), fences=[(3, 3), (5, 5), (7, 2)])
    
    # Visualize the initial maze layout with goal and fences
    maze_setup.display_maze("Maze Layout with Goal and Fences")
    
    # Step 2: Initialize the MazeRL instance with the maze setup
    maze_rl = MazeRL(maze_setup)
    
    # Task 2: Evaluate a random deterministic policy
  
    random_policy = maze_rl.get_random_policy()
    manual_policy = maze_rl.get_manual_policy(radius=2)
    if model_based == True:
    # Task
        if is_task_2 == True:
            print("Evaluating random policy...")
            V_random_policy = maze_rl.policy_evaluation(random_policy)
            maze_rl.display_value_function(V_random_policy, "Value Function for Random Policy on Maze")


        elif is_task_3 == True:
        # Task 3: Evaluate a policy with manually set optimal actions near the goal
            print("Evaluating manual policy with optimal actions near the goal...")
            
            V_manual_policy = maze_rl.policy_evaluation(manual_policy)
            maze_rl.display_value_function(V_manual_policy, "Value Function for Manual Policy (Radius=2) on Maze")

        elif is_task_4 == True:
            print("Policy Improvement with optimal policy with optimal actions near the goal...")

            # Start with the initial random policy
            current_policy = random_policy

            # Run 3 iterations of policy improvement
            for i in range(2):
                print(f"Iteration {i+1} of Policy Improvement")

                # Policy Evaluation on the current policy
                V_optimal = maze_rl.policy_evaluation(current_policy)
                
                # Policy Improvement to get the new policy
                current_policy = maze_rl.policy_improvement(V_optimal)

                # Display the value function for the improved policy
                maze_rl.display_value_function(V_optimal, f"Optimal Value Function after Iteration {i+1}")

            # Final Value Function after 3 iterations
            V_final = maze_rl.policy_evaluation(current_policy)
            maze_rl.display_value_function(V_final, "Final Optimal Value Function after 3 Iterations")

    else:
        # Generate 10 trajectories with random policy
        if task5_OPolicy== True:
            V_optimal = maze_rl.policy_evaluation(random_policy)
            new_policy = maze_rl.policy_improvement(V_optimal)
            optimal_trajectories, optimal_rewards = maze_rl.task_5_generate_trajectories(new_policy)
            print("\nOptimal Policy Trajectories:")
            for i, traj in enumerate(optimal_trajectories):
                print(f"Trajectory {i+1}: States and Actions - {traj}, Total Reward: {optimal_rewards[i]}")
                print("\nObservations:")
                print("Average reward with optimal policy:", np.mean(optimal_rewards))
    
        elif task5_Rpolicy == True:
        # Analysis and observations
            random_trajectories, random_rewards = maze_rl.task_5_generate_trajectories(random_policy)
            print("Random Policy Trajectories:") 
            for i, traj in enumerate(random_trajectories):
                print(f"Trajectory {i+1}: States and Actions - {traj}, Total Reward: {random_rewards[i]}")
                print("\nObservations:")
                print("Average reward with random policy:", np.mean(random_rewards))

        elif Q_learning == True:
            maze_rl.plot_accumulated_reward()
        elif is_task5_sarsa == True:
            rewards_per_episode, Q, trajectories = maze_rl.sarsa_trajectory(num_episodes=10)
            for i, trajectory in enumerate(trajectories):
                print(f"Trajectory {i + 1}:")
                for state, action, reward in trajectory:
                    print(f"State: {state}, Action: {action}, Reward: {reward}")
                print(f"Total Reward for Episode {i + 1}: {rewards_per_episode[i]}\n")


if __name__ == "__main__":
    main()