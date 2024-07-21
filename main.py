import enviro_2 as envi
import numpy as np
import agency as ag
import matplotlib.pyplot as plt

num_users = 40
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

ep_reward_list = []
avg_reward_list = []
proc_time = []
best_episode = None
best_reward = float('-inf')

for ep in range(1000):
    state = env.reset()
    episode_reward = 0

    for step in range(40):
        actions = agent.policy(state)
        action = np.concatenate([action.flatten() for action in actions])

        t_pos, next_state, reward, done, _ = env.step(action)
        # print(next_state)
        agent.remember(state, action, reward, next_state, done)
        agent.replay()

        state = next_state
        episode_reward += reward

        if done:
            break

    ep_reward_list.append(episode_reward)
    avg_reward = np.mean(ep_reward_list[-15:])
    avg_reward_list.append(avg_reward)
    proc_time.append(t_pos)
    if episode_reward > best_reward:
        best_reward = episode_reward
        best_episode = ep + 1
    print(f"Episode {ep + 1}, Reward: {episode_reward}")
# print(ep_reward_list)
print(proc_time[best_episode])
plt.figure()
plt.plot(avg_reward_list, color='b')
# plt.scatter(best_episode-1, best_reward, color='red', label='Best Episode', zorder=5)
# plt.plot(seq_greed1, color='r', label='No power control')
# plt.plot(seq_greed2, color='g', label='DQN')
plt.legend(loc='best')
plt.xlabel('Episode')
plt.ylabel('Average Reward')
#
# plt.plot(t_pos, color='b')
# plt.legend(loc='best')
# plt.xlabel('Time epochs')
# plt.ylabel('Average Reward')
plt.show()
