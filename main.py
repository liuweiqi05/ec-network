import enviro as envi
import numpy as np
import agency as ag

num_users = 10
num_stations = 5
num_tasks = 2
bandwidth_capacity = np.array([20, 20, 20, 20, 20])
computing_capacity = np.array([32, 32, 32, 32, 32])
user_location_range = (0, 1000)
bs_locations = np.array([[0, 0], [0, 500], [500, 500], [500, 0], [1000, 1000]])
task_size_range = (100, 200)

env = envi.TaskEnvironment(num_users, num_stations, num_tasks, bandwidth_capacity, computing_capacity,
                          user_location_range, bs_locations, task_size_range)
agent = ag.DDPGAgent(state_dim=env.state_dim, num_users=env.num_users, num_stations=env.num_stations,
                     num_tasks=env.num_tasks,
                     bandwidth_capacity=env.bandwidth_capacity, computing_capacity=env.computing_capacity)

for ep in range(20):
    state = env.reset()
    episode_reward = 0

    for step in range(10):
        actions = agent.policy(state)
        action = np.concatenate([action.flatten() for action in actions])

        next_state, reward, done, _ = env.step(action)

        agent.remember(state, action, reward, next_state, done)
        agent.replay()

        state = next_state
        episode_reward += reward

        if done:
            break

    print(f"Episode {ep + 1}, Reward: {episode_reward}")