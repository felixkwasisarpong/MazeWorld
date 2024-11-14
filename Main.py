from MyMaze import MyMaze
from MazeRL import MazeRL
import random

def main():
    model_free = True
    manually_set_policy = True
    # Step 1: Set up the maze environment
    maze_setup = MyMaze(size=(10, 10), goal_state=(9, 9), fences=[(3, 3), (5, 5), (7, 2)])
    
    # Visualize the initial maze layout with goal and fences
    maze_setup.display_maze("Maze Layout with Goal and Fences")
    
    # Step 2: Initialize the MazeRL instance with the maze setup
    maze_rl = MazeRL(maze_setup)
    
    # Task 2: Evaluate a random deterministic policy
    print("Evaluating random policy...")
    random_policy = maze_rl.get_random_policy()
    if model_free == True:
       V_random_policy = maze_rl.policy_evaluation_model_free(random_policy)
    else:
        V_random_policy = maze_rl.policy_evaluation(random_policy)

    maze_rl.display_value_function(V_random_policy, "Value Function for Random Policy on Maze")


    if manually_set_policy:
    # Task 3: Evaluate a policy with manually set optimal actions near the goal
       print("Evaluating manual policy with optimal actions near the goal...")
       manual_policy = maze_rl.get_manual_policy(radius=2)
       V_manual_policy = maze_rl.policy_evaluation(manual_policy)
       maze_rl.display_value_function(V_manual_policy, "Value Function for Manual Policy (Radius=2) on Maze")

if __name__ == "__main__":
    main()