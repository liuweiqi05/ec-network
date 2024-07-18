import numpy as np
import gym
from gym import spaces
import channel_g as cg


class TaskEnvironment(gym.Env):
    def __init__(self, num_ue, num_bs, num_tsk, bw_cap, cp_cap,
                 user_location_range, bs_locations, task_size_range):
        super(TaskEnvironment, self).__init__()

        self.num_users = num_ue
        self.num_stations = num_bs
        self.num_tasks = num_tsk
        self.bandwidth_capacity = bw_cap
        self.computing_capacity = cp_cap

        self.user_location_range = user_location_range
        self.predefined_bs_locations = bs_locations
        self.task_size_range = task_size_range

        self.state_dim = (num_ue * 2) + (num_bs * 2) + (num_ue * num_tsk) + num_tsk + (num_ue * num_tsk) + (num_bs * 2)
        self.action_dim = num_ue + (num_ue * num_bs) + (num_ue * num_bs) + (num_bs * num_tsk)

        self.action_space = spaces.Box(low=0, high=1, shape=(self.action_dim,), dtype=np.float32)
        self.observation_space = spaces.Box(low=0, high=1, shape=(self.state_dim,), dtype=np.float32)

    def reset(self):
        users_locations = np.random.uniform(low=self.user_location_range[0], high=self.user_location_range[1],
                                            size=(self.num_users * 2))

        task_sizes = np.random.uniform(low=self.task_size_range[0], high=self.task_size_range[1],
                                       size=self.num_tasks)

        bit_sizes = np.random.uniform(low=0.4, high=0.8, size=(self.num_users * self.num_tasks))

        users_init_time = 9 * np.ones((self.num_users * self.num_tasks))

        self.state = np.concatenate((users_locations, self.predefined_bs_locations.flatten(), bit_sizes,
                                     task_sizes, users_init_time, self.bandwidth_capacity, self.computing_capacity))
        return self.state

    def step(self, action):

        action = np.array(action, dtype=np.float32)

        connect_probs = action[:self.num_users]
        base_station_probs = action[self.num_users:self.num_users + self.num_users * self.num_stations].reshape((self.num_users, self.num_stations))
        bw_allocation = action[self.num_users + self.num_users * self.num_stations:self.num_users + 2 * self.num_users * self.num_stations].reshape((self.num_users, self.num_stations))
        cp_allocation = action[-self.num_stations * self.num_tasks:].reshape((self.num_stations, self.num_tasks))

        connection = np.zeros(self.num_users, dtype=int) - 1
        bandwidths = np.zeros((self.num_users, self.num_stations))

        for i in range(self.num_users):
            if connect_probs[i] > 0.5:
                selected_station = np.argmax(base_station_probs[i])
                connection[i] = selected_station
                bandwidths[i][selected_station] = bw_allocation[i][selected_station]

        x_lo = self.state[:2 * self.num_users].reshape((self.num_users, 2))
        bs_lo = self.state[self.num_users * 2:self.num_users * 2 + self.num_stations * 2].reshape((self.num_stations, 2))

        task_sizes = self.state[self.num_users * 2 + self.num_stations * 2:self.num_users * 2 + self.num_stations * 2 + self.num_users * self.num_tasks].reshape((self.num_users, self.num_tasks))
        bit_size = self.state[self.num_users * 2 + self.num_stations * 2 + self.num_users * self.num_tasks:self.num_users * 2 + self.num_stations * 2 + self.num_users * self.num_tasks + self.num_tasks]

        t_pos = 9 * np.ones(self.num_users)
        tsk_t = 9 * np.ones((self.num_users, self.num_tasks))

        tsk_S_bs = np.zeros((self.num_stations, self.num_tasks))
        for i in range(self.num_users):
            if connection[i] != -1:
                for j in range(self.num_tasks):
                    tsk_S_bs[connection[i]][j] += task_sizes[i][j]

        for i in range(self.num_users):
            if connection[i] == -1:
                tsk_t[i] = task_sizes[i] * bit_size * 1000 * 100 / 3e8
                t_of, t_lo = 0, np.sum(tsk_t[i])
                t_pos[i] = t_of + t_lo
            else:
                G = cg.channel_g(x_lo[i][0], x_lo[i][1], bs_lo[connection[i]][0], bs_lo[connection[i]][1])
                R = 180e3 * bandwidths[i][connection[i]] * np.log2(1 + 200 * G / (4e-15 * 180e3))
                t_w, t_c = task_sizes[i] / R, tsk_S_bs[connection[i]] * bit_size * 1000 * 100 / (cp_allocation[connection[i]] * 15e9)
                tsk_t[i] = t_w + t_c
                t_of, t_lo = np.sum(task_sizes[i] / R) + np.sum(tsk_S_bs[connection[i]] * bit_size / cp_allocation[connection[i]]), 0
                t_pos[i] = t_of + t_lo
        # print("processing", t_pos)
        # print("process t", tsk_t)
        users_locations = self.state[:2 * self.num_users]
        base_stations_locations = self.state[self.num_users * 2:self.num_users * 2 + self.num_stations * 2]
        task_sizes = self.state[self.num_users * 2 + self.num_stations * 2:self.num_users * 2 + self.num_stations * 2 + self.num_users * self.num_tasks]
        bit_size = self.state[self.num_users * 2 + self.num_stations * 2 + self.num_users * self.num_tasks:self.num_users * 2 + self.num_stations * 2 + self.num_users * self.num_tasks + self.num_tasks]
        tsk_t = tsk_t.flatten()
        self.state = np.concatenate((users_locations, base_stations_locations, bit_size, task_sizes, tsk_t, self.bandwidth_capacity, self.computing_capacity))
        reward = -np.sum(t_pos) / self.num_tasks

        done = False

        return self.state, reward, done, {}
