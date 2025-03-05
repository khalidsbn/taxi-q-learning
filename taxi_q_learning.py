import random
import gym
import numpy as np
import time  # For adding delays during rendering

# Disable the environment checker to avoid np.bool8 error
env = gym.make('Taxi-v3', disable_env_checker=True)

# Hyperparameters
alpha = 0.9  # Learning rate
gamma = 0.95  # Discount factor
epsilon = 1.0  # Exploration rate
epsilon_decay = 0.9995  # Decay rate for exploration
min_epsilon = 0.01  # Minimum exploration rate
num_episodes = 10000  # Number of training episodes
max_steps = 100  # Maximum steps per episode

# Initialize Q-table with zeros
q_table = np.zeros((env.observation_space.n, env.action_space.n))

# Function to choose an action using epsilon-greedy policy
def choose_action(state):
    if random.uniform(0, 1) < epsilon:
        return env.action_space.sample()  # Explore: choose a random action
    else:
        return np.argmax(q_table[state, :])  # Exploit: choose the best known action

# Training process
for episode in range(num_episodes):
    state, _ = env.reset()
    done = False

    for step in range(max_steps):
        action = choose_action(state)
        
        # Gym 0.26+ returns 5 values: next_state, reward, terminated, truncated, info
        next_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        # Update Q-value using the Bellman equation
        old_value = q_table[state, action]
        next_max = np.max(q_table[next_state, :])
        q_table[state, action] = (1 - alpha) * old_value + alpha * (reward + gamma * next_max)

        state = next_state

        if done:
            break

    # Decay epsilon to reduce exploration over time
    epsilon = max(min_epsilon, epsilon * epsilon_decay)

    # Print progress every 1000 episodes
    if (episode + 1) % 1000 == 0:
        print(f"Episode {episode + 1} completed")

print("Training completed.")

# Create environment for rendering (also disable checker here)
env = gym.make('Taxi-v3', render_mode='human', disable_env_checker=True)

# Evaluation
for episode in range(5):
    state, _ = env.reset()
    done = False

    print(f'Episode {episode + 1}')

    for step in range(max_steps):
        action = np.argmax(q_table[state, :])  # Always choose the best action
        next_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        state = next_state

        time.sleep(0.1)  # Add a small delay to see each step

        if done:
            print(f'Finished episode {episode + 1} with reward {reward}')
            time.sleep(1)  # Pause before the next episode starts
            break

env.close()