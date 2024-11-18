from MyMaze import MyMaze
from MazeRL import MazeRL
import random
import numpy as np
def main():
    model_based = True
    is_task_2 = False
    is_task_3 = True
    is_task_4 = False
    task5_Rpolicy = True
    task5_OPolicy= False
    Q_learning = False
    is_Sarsa = False

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
             print("Policy Improvement  with optimal policy with optimal actions near the goal...")
            
             V_optimal = maze_rl.policy_evaluation(random_policy)
             new_policy = maze_rl.policy_improvement(V_optimal)
             V_final = maze_rl.policy_evaluation(new_policy)  # Get the value function for the improved policy
             maze_rl.display_value_function(V_final, "Optimal Value Function")
    else:
        # Generate 10 trajectories with random policy
        if task5_OPolicy== True:
                # Assume 'optimal_policy' is the policy derived from Task 4
            optimal_trajectories, optimal_rewards = maze_rl.task_5_generate_trajectories(manual_policy)
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
        elif is_Sarsa == True:
            maze_rl.plot_sarsa_acc_R()

if __name__ == "__main__":
    main()