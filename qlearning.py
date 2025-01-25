import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# Define constants
LEARNING_RATE = 0.1 # How quickly the agent learns
DISCOUNT = 0.95     # How important are future rewards
EPISODES = 30001    # Number of episodes to run
SHOW_EVERY = 200    # How often to render the environment/plot the graph
PERIODIC_RENDERING = False

EPSILON = 0.5 # Exploration rate
START_EPSILON_DECAYING = 1
END_EPSILON_DECAYING = EPISODES // 2
epsilon_decay_value = EPSILON / (END_EPSILON_DECAYING - START_EPSILON_DECAYING)

Total_rewards = []
aggregated_rewards = {'ep': [], 'avg': [], 'min': [], 'max': []}

# Helper function to get discrete state
def get_discrete_state(env, state, discrete_os_win_size):
    discrete_state = (state - env.observation_space.low) / discrete_os_win_size
    return tuple(discrete_state.astype(int))

def run():
    # Set up environment parameters
    env = gym.make('MountainCar-v0')
    discrete_os_size = [40] * len(env.observation_space.high)
    discrete_os_win_size = (env.observation_space.high - env.observation_space.low) / discrete_os_size
    q_table = np.random.uniform(low=-2, high=0, size = (discrete_os_size + [env.action_space.n])) 
    q_original = q_table.copy()
    # Main training loop
    EPSILON = 0.5
    for episode in tqdm(range(EPISODES), ascii=True, unit='episodes'):
        
        if PERIODIC_RENDERING:
            if episode % SHOW_EVERY == 0:
                # Create the environment with rendering
                env = gym.make('MountainCar-v0', render_mode="human")
            else:
                # Create the environment without rendering
                env = gym.make('MountainCar-v0')
        else:
            env = gym.make('MountainCar-v0')

        # Reset environment
        state, _ = env.reset()
        discrete_state = get_discrete_state(env, state, discrete_os_win_size)
        done = False
        rewards = 0
        while not done:
                # Exploration vs exploitation
                if np.random.random() > EPSILON:
                    action = np.argmax(q_table[discrete_state]) # 1-epsilon % of the time we take the best action
                else:    
                    action = np.random.randint(0, env.action_space.n) # Random action
                
                next_state, reward, terminated, truncated, _ = env.step(action)
                # print(reward, next_state)
                # -1.0 [-0.41134322  0.0005396 ]
                rewards += reward
                new_discrete_state = get_discrete_state(env, next_state, discrete_os_win_size)
                if not terminated:
                    max_future_q = np.max(q_table[new_discrete_state])
                    current_q = q_table[discrete_state + (action, )]
                    new_q = (1-LEARNING_RATE) * current_q + LEARNING_RATE * (reward + DISCOUNT * max_future_q)
                    # new_q = current_q + LEARNING_RATE * (reward + DISCOUNT * max_future_q - current_q)
                    q_table[discrete_state + (action, )] = new_q
                
                elif next_state[0] >= 0.5:
                    q_table[discrete_state + (action, )] = 0
                    #print(f"Goal reached on episode {episode}")
                done = terminated or truncated
                discrete_state = new_discrete_state
        
        
        if END_EPSILON_DECAYING >= episode >= START_EPSILON_DECAYING:
            EPSILON -= epsilon_decay_value
        
        Total_rewards.append(rewards)

        if episode % SHOW_EVERY == 0:
            aggregated_rewards['ep'].append(episode)
            aggregated_rewards['avg'].append(np.mean(Total_rewards[-SHOW_EVERY:])) # Average rewards over the last 2000 episodes
            aggregated_rewards['min'].append(np.min(Total_rewards[-SHOW_EVERY:]))
            aggregated_rewards['max'].append(np.max(Total_rewards[-SHOW_EVERY:]))

        env.close()

    plt.plot(aggregated_rewards['ep'], aggregated_rewards['avg'], label="avg")
    plt.plot(aggregated_rewards['ep'], aggregated_rewards['min'], label="min")
    plt.plot(aggregated_rewards['ep'], aggregated_rewards['max'], label="max")
    plt.legend(loc=4)
    plt.xlabel('Episodes')
    plt.ylabel('Rewards')
    plt.show()

    # Plot Q-table
    plot_best_action(q_table, q_original, gym.make('MountainCar-v0'), discrete_os_size, discrete_os_win_size)
   
    return q_table
    

def plot_best_action(q_table, q_original, env, discrete_os_size, discrete_os_win_size):
    # Prepare data for plotting
    position = []
    velocity = []
    colors = []
    original_colors = []

    for pos in range(discrete_os_size[0]):
        for vel in range(discrete_os_size[1]):
            # Extract max Q-value action for each discrete state
            state_action_values = q_table[pos, vel, :]
            max_action = np.argmax(state_action_values)
            
            original_state_action_values = q_original[pos, vel, :]
            original_max_action = np.argmax(original_state_action_values)
            
            # Map actions to colors: 0 -> red, 1 -> blue, 2 -> green
            if max_action == 0:
                colors.append('red')
            elif max_action == 1:
                colors.append('blue')
            elif max_action == 2:
                colors.append('green')
            
            if original_max_action == 0:
                original_colors.append('red')
            elif original_max_action == 1:
                original_colors.append('blue')
            elif original_max_action == 2:
                original_colors.append('green')
            
            # Convert discrete state to continuous values
            position.append(pos * discrete_os_win_size[0] + env.observation_space.low[0])
            velocity.append(vel * discrete_os_win_size[1] + env.observation_space.low[1])

    # Plotting the graph
    fig, axs = plt.subplots(1, 2, figsize=(20, 8))
    
        
    axs[0].scatter(position, velocity, c=original_colors, s=100, marker='s')
    axs[0].set_title("Original Q-Table: Position vs Velocity")
    axs[0].set_xlabel("Position")
    axs[0].set_ylabel("Velocity")
    
    axs[1].scatter(position, velocity, c=colors, s=100, marker='s')
    axs[1].set_title("Trained Q-Table: Position vs Velocity")
    axs[1].set_xlabel("Position")
    axs[1].set_ylabel("Velocity")
    
    # legend
    red_patch = plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=10, label='Left')
    blue_patch = plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=10, label='Center')
    green_patch = plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='green', markersize=10, label='Right')
    axs[0].legend(handles=[red_patch, blue_patch, green_patch])
    axs[1].legend(handles=[red_patch, blue_patch, green_patch])
    plt.tight_layout()
    plt.show()



if __name__ == "__main__":
    run()