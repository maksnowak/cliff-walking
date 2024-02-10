# Note: code in this file has been extracted from the jupyter notebook with slight modifications

import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt

# Q-learning algorithm used to solve the cliff walking problem
def q_learning(env, episodes, _lambda, beta, epsilon, t_max=150):
    Q = np.zeros((env.observation_space.n, env.action_space.n))
    for _ in range(episodes):
        state = env.reset()[0]
        done = False
        t = 0
        while t < t_max and not done:
            # Epsilon-zachłanna strategia
            if np.random.uniform() < epsilon:
                action = env.action_space.sample()
            else:
                action = np.argmax(Q[state])
            next, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            Q[state][action] += beta * (reward + _lambda * np.max(Q[next]) - Q[state][action])
            state = next
            t += 1
    return Q

# Display the results of the learning process
def learning_results(enviro, episodes, _lambda, beta, epsilon, t_max=150):
    Q = q_learning(enviro, episodes, _lambda, beta, epsilon, t_max)

    # Helper function to display arrows in the heatmap
    def arrows(direction, value):
        if value.all() == 0:
            return ' '
        if direction == '0':
            return '↑'
        elif direction == '1':
            return '→'
        elif direction == '2':
            return '↓'
        elif direction == '3':
            return '←'

    # Q-values heatmap
    heatmap = np.max(Q, axis=1).reshape((4, 12))
    plt.figure(figsize=(12, 4))
    plt.imshow(heatmap, cmap='summer', interpolation='nearest')
    for y in range(4):
        for x in range(12):
            q_value = Q[y * 12 + x]
            plt.text(x, y, arrows(str(np.argmax(q_value)), q_value), color='black', ha='center', va='center')
    plt.annotate('FINISH', xy=(11.5, 3.5), xytext=(11, 3), color='black', ha='center', va='center')
    plt.xticks([])
    plt.yticks([])
    plt.colorbar()
    plt.title('Learned Q-values - arrows represent the best action')
    plt.show()

# ---------------------------------------------------
# Create the environment and run the learning process
env = gym.make('CliffWalking-v0')

learning_results(env, 300, 0.5, 0.5, 0.1)