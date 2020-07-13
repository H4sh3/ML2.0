import random
import gym
import numpy as np
from time import sleep
from IPython.display import clear_output

env = gym.make('gym_collect:collect-v0')
print('env init done!')

epochs = 0
penalties, reward = 0, 0

# Hyperparameters
alpha = 0.1
gamma = 0.6
epsilon = 0.1

q_table = {}
for s in env.states:
    q_table[str(s)] = np.zeros([4])

print('q-learning part')
for i in range(1, 2000):
    env.reset()
    env.s = env.get_initial_state()
    state = env.s
    epochs, penalties, reward, = 0, 0, 0
    done = False
    while not done:
        env.iter_count += 1
        if random.uniform(0, 1) < epsilon:
            action = env.action_space.sample() # Explore action space
        else:
            action = np.argmax(q_table[state]) # Exploit learned values
        old_value = q_table[state][action]

        next_state, reward, done, info = env.step(action)

        next_max = np.max(q_table[str(next_state)])

        new_value = (1 - alpha) * old_value + alpha * (reward + gamma * next_max)
        q_table[state][action] = new_value

        if reward == -10:
            penalties += 1
        state = next_state
        epochs += 1

    if i % 100 == 0:
        clear_output(wait=True)
        print(f"Episode: {i} {env.iter_count}")
    
    if i % 1000 == 0:
        sleep(0.5)

print("Training finished.\n")
"""Evaluate agent's performance after Q-learning"""

total_epochs, total_penalties = 0, 0


env.reset()
env.s = env.get_initial_state()
epochs, penalties, reward = 0, 0, 0
done = False

i = 0

history = []
while not done:
    action = np.argmax(q_table[env.s])
    state, reward, done, info = env.step(action)

    if reward == -10:
        penalties += 1

    epochs += 1

    history.append({
        'frame': env.render(state, 'ansi'),
        'state': env.s,
        'action': action,
        'reward': reward
    })


total_penalties += penalties
total_epochs += epochs

def print_frames(h):
    for i, frame in enumerate(h):
        clear_output(wait=True)
        print('-'*10)
        print(frame['frame'])
        print(f"Timestep: {i + 1}")
        print(f"State: {frame['state']}")
        print(f"Action: {frame['action']}")
        print(f"Reward: {frame['reward']}")
        sleep(.5)
    print('-- done --')

while True:
    sleep(2)
    print_frames(history)
