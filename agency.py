import tensorflow as tf
from tensorflow.keras import layers
from collections import deque
import random


class DDPGAgent:
    def __init__(self, state_dim, num_users, num_stations, num_tasks, bandwidth_capacity, computing_capacity):
        self.state_dim = state_dim
        self.num_users = num_users
        self.num_stations = num_stations
        self.num_tasks = num_tasks
        self.bandwidth_capacity = bandwidth_capacity
        self.computing_capacity = computing_capacity

        self.actor = self.build_actor()
        self.critic = self.build_critic()
        self.target_actor = self.build_actor()
        self.target_critic = self.build_critic()
        self.update_target(self.target_actor, self.actor, tau=1.0)
        self.update_target(self.target_critic, self.critic, tau=1.0)

        self.actor_optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
        self.critic_optimizer = tf.keras.optimizers.Adam(learning_rate=0.002)

        self.memory = deque(maxlen=10000)
        self.gamma = 0.99
        self.tau = 0.005

    def build_actor(self):
        state_input = layers.Input(shape=(self.state_dim,))
        last_init = tf.random_uniform_initializer(minval=-0.00003, maxval=0.00003)

        out = layers.Dense(256, activation="selu", kernel_initializer="lecun_normal")(state_input)
        out = layers.Dropout(rate=0.5)(out)
        out = layers.BatchNormalization()(out)
        out = layers.Dense(256, activation="selu", kernel_initializer="lecun_normal")(out)
        out = layers.Dropout(rate=0.5)(out)
        out = layers.BatchNormalization()(out)

        connect_output = layers.Dense(self.num_users, activation="sigmoid", kernel_initializer=last_init)(out)
        base_station_output = layers.Dense(self.num_users * self.num_stations, activation="softmax",
                                           kernel_initializer=last_init)(out)
        base_station_output = layers.Reshape((self.num_users, self.num_stations))(base_station_output)
        bandwidth_output = layers.Dense(self.num_users * self.num_stations, activation="relu",
                                        kernel_initializer=last_init)(out)
        bandwidth_output = layers.Reshape((self.num_users, self.num_stations))(bandwidth_output)
        bandwidth_output = layers.Lambda(lambda x: tf.nn.softmax(x, axis=1) * self.bandwidth_capacity)(bandwidth_output)
        computing_power_output = layers.Dense(self.num_stations * self.num_tasks, activation="relu",
                                              kernel_initializer=last_init)(out)
        computing_power_output = layers.Reshape((self.num_stations, self.num_tasks))(computing_power_output)
        computing_capacity_tensor = tf.constant(self.computing_capacity, dtype=tf.float32)
        computing_capacity_tensor = tf.reshape(computing_capacity_tensor, (self.num_stations, 1))
        computing_power_output = layers.Lambda(lambda x: tf.nn.softmax(x, axis=1) * computing_capacity_tensor)(
            computing_power_output)

        model = tf.keras.Model(inputs=state_input,
                               outputs=[connect_output, base_station_output, bandwidth_output, computing_power_output])
        return model

    def build_critic(self):
        state_input = layers.Input(shape=(self.state_dim,))
        conn_action_input = layers.Input(shape=(self.num_users,))
        bs_action_input = layers.Input(shape=(self.num_users, self.num_stations))
        bw_action_input = layers.Input(shape=(self.num_users, self.num_stations))
        cp_action_input = layers.Input(shape=(self.num_stations, self.num_tasks))

        state_out = layers.Dense(256, activation="selu", kernel_initializer="lecun_normal")(state_input)
        state_out = layers.BatchNormalization()(state_out)
        conn_out = layers.Dense(256, activation="selu", kernel_initializer="lecun_normal")(conn_action_input)
        conn_out = layers.BatchNormalization()(conn_out)
        bs_out = layers.Flatten()(bs_action_input)
        bs_out = layers.Dense(256, activation="selu", kernel_initializer="lecun_normal")(bs_out)
        bs_out = layers.BatchNormalization()(bs_out)
        bw_out = layers.Flatten()(bw_action_input)
        bw_out = layers.Dense(256, activation="selu", kernel_initializer="lecun_normal")(bw_out)
        bw_out = layers.BatchNormalization()(bw_out)
        cp_out = layers.Flatten()(cp_action_input)
        cp_out = layers.Dense(256, activation="selu", kernel_initializer="lecun_normal")(cp_out)
        cp_out = layers.BatchNormalization()(cp_out)

        concat = layers.Concatenate()([state_out, conn_out, bs_out, bw_out, cp_out])
        out = layers.Dense(256, activation="selu", kernel_initializer="lecun_normal")(concat)
        out = layers.BatchNormalization()(out)
        out = layers.Dense(256, activation="selu", kernel_initializer="lecun_normal")(out)
        out = layers.BatchNormalization()(out)
        q_value = layers.Dense(1, kernel_initializer="lecun_normal")(out)

        model = tf.keras.Model(
            inputs=[state_input, conn_action_input, bs_action_input, bw_action_input, cp_action_input], outputs=q_value)
        return model

    def update_target(self, target_weights, weights, tau):
        for (a, b) in zip(target_weights.variables, weights.variables):
            a.assign(b * tau + a * (1 - tau))

    def policy(self, state):
        state = tf.convert_to_tensor(state)
        state = tf.expand_dims(state, axis=0)
        actions = self.actor(state)
        return [tf.squeeze(action).numpy() for action in actions]

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def replay(self):
        if len(self.memory) < 64:
            return

        batch = random.sample(self.memory, 64)
        for state, action, reward, next_state, done in batch:
            state = tf.convert_to_tensor(state)
            next_state = tf.convert_to_tensor(next_state)

            state = tf.expand_dims(state, axis=0)
            next_state = tf.expand_dims(next_state, axis=0)

            # conn_action, bs_action, bw_action, cp_action = action
            conn_action = action[:self.num_users]
            bs_action = action[self.num_users:self.num_users + self.num_users * self.num_stations].reshape(
                (self.num_users, self.num_stations))
            bw_action = action[
                        self.num_users + self.num_users * self.num_stations:self.num_users + 2 * self.num_users * self.num_stations].reshape(
                (self.num_users, self.num_stations))
            cp_action = action[-self.num_stations * self.num_tasks:].reshape((self.num_stations, self.num_tasks))
            conn_action = tf.expand_dims(conn_action, axis=0)
            bs_action = tf.expand_dims(bs_action, axis=0)
            bw_action = tf.expand_dims(bw_action, axis=0)
            cp_action = tf.expand_dims(cp_action, axis=0)

            next_conn_action, next_bs_action, next_bw_action, next_cp_action = self.target_actor(next_state)

            target_q = reward + self.gamma * self.target_critic(
                [next_state, next_conn_action, next_bs_action, next_bw_action, next_cp_action]) * (1 - done)
            with tf.GradientTape() as tape:
                q_value = self.critic([state, conn_action, bs_action, bw_action, cp_action])
                critic_loss = tf.math.reduce_mean(tf.math.square(target_q - q_value))
            critic_grad = tape.gradient(critic_loss, self.critic.trainable_variables)
            self.critic_optimizer.apply_gradients(zip(critic_grad, self.critic.trainable_variables))

            with tf.GradientTape() as tape:
                actions = self.actor(state)
                actor_loss = -self.critic([state] + actions)
                actor_loss = tf.math.reduce_mean(actor_loss)
            actor_grad = tape.gradient(actor_loss, self.actor.trainable_variables)
            self.actor_optimizer.apply_gradients(zip(actor_grad, self.actor.trainable_variables))

        self.update_target(self.target_actor, self.actor, self.tau)
        self.update_target(self.target_critic, self.critic, self.tau)