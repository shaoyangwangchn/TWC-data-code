import numpy as np
import tensorflow as tf


class DDPG_PA_Network:
    def __init__(
            self,
            n_user,
            n_subcarrier,
            agent_index,
            learning_rate_critic,
            learning_rate_actor,
            reward_decay,
            e_greedy,
            replace_target_iter,
            memory_size,
            batch_size,
            e_greedy_increment,
            power_change_indicator,
            p_max
    ):
        self.n_user = n_user
        self.n_subcarrier = n_subcarrier
        self.agent_index = agent_index
        self.lr_critic = learning_rate_critic
        self.lr_actor = learning_rate_actor
        self.gamma = reward_decay
        self.epsilon_max = e_greedy
        self.replace_target_iter = replace_target_iter
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.epsilon_increment = e_greedy_increment
        self.power_change_indicator = power_change_indicator
        self.p_max = p_max
        self.epsilon = 0.0 if e_greedy_increment is not None else self.epsilon_max
        self.learn_step_counter = 0
        self.memory_counter = 0
        self.memory = np.zeros((self.memory_size, self.n_user * (8 * self.n_subcarrier + 4) + 1))
        self.sess = tf.Session()

        # collect network parameters
        oa_net_name = 'online_actor_net' + str(self.agent_index)
        oc_net_name = 'online_critic_net' + str(self.agent_index)
        ta_net_name = 'target_actor_net' + str(self.agent_index)
        tc_net_name = 'target_critic_net' + str(self.agent_index)
        self.oa_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=oa_net_name)
        self.oc_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=oc_net_name)
        self.ta_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=ta_net_name)
        self.tc_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=tc_net_name)

        # soft update
        soft_replace_name = 'soft_replacement' + str(self.agent_index)
        with tf.variable_scope(soft_replace_name):
            self.actor_replace_op = [tf.assign(ta, oa) for ta, oa in zip(self.ta_params, self.oa_params)]
            self.critic_replace_op = [tf.assign(tc, oc) for tc, oc in zip(self.tc_params, self.oc_params)]

        # --------------------------build network--------------------------
        # naming
        state_name = 's' + str(self.agent_index)
        state_next_name = 's_' + str(self.agent_index)
        reward_name = 'r' + str(self.agent_index)
        action_name = 'a' + str(self.agent_index)
        self.s = tf.placeholder(tf.float32, [None, self.n_user * (3 * self.n_subcarrier + 2)],
                                name=state_name)  # input State
        self.s_ = tf.placeholder(tf.float32, [None, self.n_user * (3 * self.n_subcarrier + 2)],
                                 name=state_next_name)  # input Next State
        self.r = tf.placeholder(tf.float32, [None, ], name=reward_name)  # input Reward
        # self.a = tf.placeholder(tf.float32, [None, self.n_subcarrier], name=action_name)  # input Action
        self.a_all = tf.placeholder(tf.float32, [None, self.n_user * self.n_subcarrier], name=action_name)
        self.a_all_ = tf.placeholder(tf.float32, [None, self.n_user * self.n_subcarrier], name=action_name)

        # built online actor network
        self.online_actor_net = self.built_actor_net(name="online", s=self.s)
        print("||-----------PA_agent_" + str(self.agent_index) + " online actor network is successfully built -----")
        # built target actor network, and also to calculate the next action
        self.target_actor_net = self.built_actor_net(name="target", s=self.s_)
        print("||-----------PA_agent_" + str(self.agent_index) + " target actor network is successfully built -----")
        # built online critic network
        self.online_critic_net = self.built_critic_net(name="online", s=self.s, a_all=self.a_all)
        print("||-----------PA_agent_" + str(self.agent_index) + " online critic network is successfully built -----")
        # built target critic network, and also to calculate the target Q
        self.target_critic_net = self.built_critic_net(name="target", s=self.s_, a_all=self.a_all_)
        print("||-----------PA_agent_" + str(self.agent_index) + " target critic network is successfully built -----\n")

        # --------------------------train network--------------------------
        # training critic network
        self.target_q = self.r + self.gamma * self.target_critic_net
        self.critic_loss = tf.reduce_mean(tf.squared_difference(self.target_q, self.online_critic_net))
        self.critic_train_op = tf.train.AdamOptimizer(self.lr_critic).minimize(self.critic_loss)

        # training actor network
        self.actor_loss = -tf.reduce_mean(self.online_critic_net)
        self.actor_train_op = tf.train.AdamOptimizer(self.lr_actor).minimize(self.actor_loss)

        # initialize
        self.sess.run(tf.global_variables_initializer())

        # output board graph
        output_graph = True
        if output_graph:
            path_name = "logs/PA_agent_" + str(self.agent_index) + "/"
            tf.summary.FileWriter(path_name, self.sess.graph)

    # learning
    def learn(self):
        # soft update
        if self.learn_step_counter % self.replace_target_iter == 0:
            self.sess.run(self.actor_replace_op)  # target actor network update
            self.sess.run(self.critic_replace_op)  # target critic network update

        # sampling training date
        if self.memory_counter > self.memory_size:
            sample_index = np.random.choice(self.memory_size, size=self.batch_size)
        else:
            sample_index = np.random.choice(self.memory_counter, size=self.batch_size)
        batch_memory = self.memory[sample_index, :]

        # optimize network
        _, _ = self.sess.run(
            [self.actor_train_op, self.critic_train_op],
            feed_dict={
                self.s: batch_memory[:, :self.n_user * (self.n_subcarrier * 3 + 2)],
                self.a_all: batch_memory[:, self.n_user * (self.n_subcarrier * 3 + 2): self.n_user * (
                        self.n_subcarrier * 4 + 2)],
                self.r: batch_memory[:, self.n_user * (self.n_subcarrier * 4 + 2)],
                self.s_: batch_memory[:,
                         self.n_user * (self.n_subcarrier * 4 + 2) + 1: self.n_user * (self.n_subcarrier * 7 + 4) + 1],
                self.a_all_: batch_memory[:, self.n_user * (
                        self.n_subcarrier * 7 + 4) + 1: self.n_user * (
                        self.n_subcarrier * 8 + 4) + 1],
            })

        # update utilize probability
        self.epsilon = self.epsilon + self.epsilon_increment if self.epsilon < self.epsilon_max else self.epsilon_max

        # update learning step
        self.learn_step_counter += 1

    # built actor network
    def built_actor_net(self, name, s):

        eval_net_name = name + '_actor_net' + str(self.agent_index)  # naming
        with tf.variable_scope(eval_net_name):
            # input layers (including the CNN compressing)
            # Split the input into two parts: self agent and other agents
            if self.agent_index == 0:
                s_self = s[:, :(self.agent_index + 1) * (3 * self.n_subcarrier + 2)]
                s_other = s[:, (self.agent_index + 1) * (3 * self.n_subcarrier + 2):]
            elif self.agent_index == self.n_user - 1:
                s_self = s[:, self.agent_index * (3 * self.n_subcarrier + 2):]
                s_other = s[:, :self.agent_index * (3 * self.n_subcarrier + 2)]
            else:
                s_self = s[:, self.agent_index * (3 * self.n_subcarrier + 2):(self.agent_index + 1) * (
                            3 * self.n_subcarrier + 2)]
                s_other_part1 = s[:, :self.agent_index * (3 * self.n_subcarrier + 2)]
                s_other_part2 = s[:, (self.agent_index + 1) * (3 * self.n_subcarrier + 2):]
                s_other = tf.concat([s_other_part1, s_other_part2], axis=-1)

            # compress states of other agents by utilizing the CNN
            # transform the state to a matrix
            s_other = tf.reshape(s_other, [-1, self.n_user - 1, 3 * self.n_subcarrier + 2, 1])

            # the first convolution layer
            conv1_w = self.cnn_weight_variable([2, 2, 1, 64])
            conv1_b = self.cnn_base_variable([64])
            conv1 = tf.nn.conv2d(s_other, conv1_w, strides=[1, 1, 1, 1], padding="SAME")
            h_conv1 = tf.nn.relu(conv1 + conv1_b)
            # the first pooling layer
            h_pool1 = tf.nn.max_pool(h_conv1, ksize=[1, 1, 2, 1], strides=[1, 1, 2, 1], padding="SAME")

            # the second convolution layer
            conv2_w = self.cnn_weight_variable([2, 2, 64, 256])
            conv2_b = self.cnn_base_variable([256])
            conv2 = tf.nn.conv2d(h_pool1, conv2_w, strides=[1, 1, 1, 1], padding="SAME")
            h_conv2 = tf.nn.relu(conv2 + conv2_b)
            # the second pooling layer
            h_pool2 = tf.nn.max_pool(h_conv2, ksize=[1, 1, 2, 1], strides=[1, 1, 2, 1], padding="SAME")

            # flatten
            h_pool2_shape = h_pool2.get_shape()
            flattened_shape = h_pool2_shape[1].value * h_pool2_shape[2].value * h_pool2_shape[3].value

            # the first full connection layer
            fc1_W = self.cnn_weight_variable([flattened_shape, 256])
            fc1_b = self.cnn_base_variable([256])
            h_pool2_flat = tf.reshape(h_pool2, [-1, flattened_shape])
            h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, fc1_W) + fc1_b)

            # the second full connection layer
            fc2_W = self.cnn_weight_variable([256, 128])
            fc2_b = self.cnn_base_variable([128])
            h_fc2 = tf.nn.relu(tf.matmul(h_fc1, fc2_W) + fc2_b)

            # the third full connection layer
            fc3_W = self.cnn_weight_variable([128, 3 * self.n_subcarrier + 2])
            fc3_b = self.cnn_base_variable([3 * self.n_subcarrier + 2])
            h_fc3 = tf.nn.sigmoid(tf.matmul(h_fc2, fc3_W) + fc3_b)

            # merge self and other states
            s_other = tf.reshape(h_fc3, [-1, 3 * self.n_subcarrier + 2])
            s_input = tf.concat([s_self, s_other], axis=-1)

            # hidden layers (including ResNet)
            layer_hidden_1_name = 'a_hidden_1'
            a_hidden_1 = tf.layers.dense(s_input, 128, tf.nn.relu, name=layer_hidden_1_name)
            layer_hidden_2_name = 'a_hidden_2'
            a_hidden_2 = tf.layers.dense(a_hidden_1, 128, tf.nn.relu, name=layer_hidden_2_name)
            layer_hidden_3_name = 'a_hidden_3'
            a_hidden_3 = tf.layers.dense(a_hidden_2, 128, tf.nn.relu, name=layer_hidden_3_name)

            res_base_1 = self.cnn_base_variable([128])
            ResNet_block_1 = tf.add(a_hidden_1, a_hidden_3)
            ResNet_block_1 = tf.nn.relu(ResNet_block_1 + res_base_1)

            layer_hidden_4_name = 'a_hidden_4'
            a_hidden_4 = tf.layers.dense(ResNet_block_1, 128, tf.nn.relu, name=layer_hidden_4_name)
            layer_hidden_4_name = 'a_hidden_5'
            a_hidden_5 = tf.layers.dense(a_hidden_4, 128, tf.nn.relu, name=layer_hidden_4_name)

            res_base_2 = self.cnn_base_variable([128])
            ResNet_block_2 = tf.add(ResNet_block_1, a_hidden_5)
            ResNet_block_2 = tf.nn.relu(ResNet_block_2 + res_base_2)

            # output layers
            layer_output_1_name = 'a_output_1'
            a_output_1 = tf.layers.dense(ResNet_block_2, 128, tf.nn.relu, name=layer_output_1_name)
            layer_output_2_name = 'a_output_2'
            a_output_2 = tf.layers.dense(a_output_1, 128, tf.nn.relu, name=layer_output_2_name)
            layer_output_3_name = 'a_output_3'
            a_output = tf.layers.dense(a_output_2, self.n_subcarrier, tf.nn.tanh, name=layer_output_3_name)

        return a_output

    # built critic network
    def built_critic_net(self, name, s, a_all):

        eval_net_name = name + '_critic_net' + str(self.agent_index)  # naming
        with tf.variable_scope(eval_net_name):
            # input layers (including the CNN compressing)
            # Split the input into two parts: self agent and other agents
            if self.agent_index == 0:
                s_self = s[:, :(self.agent_index + 1) * (3 * self.n_subcarrier + 2)]
                s_other = s[:, (self.agent_index + 1) * (3 * self.n_subcarrier + 2):]
            elif self.agent_index == self.n_user - 1:
                s_self = s[:, self.agent_index * (3 * self.n_subcarrier + 2):]
                s_other = s[:, :self.agent_index * (3 * self.n_subcarrier + 2)]
            else:
                s_self = s[:, self.agent_index * (3 * self.n_subcarrier + 2):(self.agent_index + 1) * (
                            3 * self.n_subcarrier + 2)]
                s_other_part1 = s[:, :self.agent_index * (3 * self.n_subcarrier + 2)]
                s_other_part2 = s[:, (self.agent_index + 1) * (3 * self.n_subcarrier + 2):]
                s_other = tf.concat([s_other_part1, s_other_part2], axis=-1)

            # compress states of other agents by utilizing the CNN
            # transform the state to a matrix
            s_other = tf.reshape(s_other, [-1, self.n_user - 1, 3 * self.n_subcarrier + 2, 1])

            # the first convolution layer
            conv1_w = self.cnn_weight_variable([2, 2, 1, 64])
            conv1_b = self.cnn_base_variable([64])
            conv1 = tf.nn.conv2d(s_other, conv1_w, strides=[1, 1, 1, 1], padding="SAME")
            h_conv1 = tf.nn.relu(conv1 + conv1_b)
            # the first pooling layer
            h_pool1 = tf.nn.max_pool(h_conv1, ksize=[1, 1, 2, 1], strides=[1, 1, 2, 1], padding="SAME")

            # the second convolution layer
            conv2_w = self.cnn_weight_variable([2, 2, 64, 256])
            conv2_b = self.cnn_base_variable([256])
            conv2 = tf.nn.conv2d(h_pool1, conv2_w, strides=[1, 1, 1, 1], padding="SAME")
            h_conv2 = tf.nn.relu(conv2 + conv2_b)
            # the second pooling layer
            h_pool2 = tf.nn.max_pool(h_conv2, ksize=[1, 1, 2, 1], strides=[1, 1, 2, 1], padding="SAME")

            # flatten
            h_pool2_shape = h_pool2.get_shape()
            flattened_shape = h_pool2_shape[1].value * h_pool2_shape[2].value * h_pool2_shape[3].value

            # the first full connection layer
            fc1_W = self.cnn_weight_variable([flattened_shape, 256])
            fc1_b = self.cnn_base_variable([256])
            h_pool2_flat = tf.reshape(h_pool2, [-1, flattened_shape])
            h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, fc1_W) + fc1_b)

            # the second full connection layer
            fc2_W = self.cnn_weight_variable([256, 128])
            fc2_b = self.cnn_base_variable([128])
            h_fc2 = tf.nn.relu(tf.matmul(h_fc1, fc2_W) + fc2_b)

            # the third full connection layer
            fc3_W = self.cnn_weight_variable([128, 3 * self.n_subcarrier + 2])
            fc3_b = self.cnn_base_variable([3 * self.n_subcarrier + 2])
            h_fc3 = tf.nn.sigmoid(tf.matmul(h_fc2, fc3_W) + fc3_b)

            # merge self and other states
            s_other = tf.reshape(h_fc3, [-1, 3 * self.n_subcarrier + 2])
            c_input = tf.concat([s_self, s_other, a_all], axis=-1)

            # hidden layers (including ResNet)
            layer_hidden_1_name = 'c_hidden_1'
            c_hidden_1 = tf.layers.dense(c_input, 128, tf.nn.relu, name=layer_hidden_1_name)
            layer_hidden_2_name = 'c_hidden_2'
            c_hidden_2 = tf.layers.dense(c_hidden_1, 128, tf.nn.relu, name=layer_hidden_2_name)
            layer_hidden_3_name = 'c_hidden_3'
            c_hidden_3 = tf.layers.dense(c_hidden_2, 128, tf.nn.relu, name=layer_hidden_3_name)

            res_base_1 = self.cnn_base_variable([128])
            ResNet_block_1 = tf.add(c_hidden_1, c_hidden_3)
            ResNet_block_1 = tf.nn.relu(ResNet_block_1 + res_base_1)

            layer_hidden_4_name = 'c_hidden_4'
            c_hidden_4 = tf.layers.dense(ResNet_block_1, 128, tf.nn.relu, name=layer_hidden_4_name)
            layer_hidden_4_name = 'c_hidden_5'
            c_hidden_5 = tf.layers.dense(c_hidden_4, 128, tf.nn.relu, name=layer_hidden_4_name)

            res_base_2 = self.cnn_base_variable([128])
            ResNet_block_2 = tf.add(ResNet_block_1, c_hidden_5)
            ResNet_block_2 = tf.nn.relu(ResNet_block_2 + res_base_2)

            # output layers
            layer_output_1 = 'c_output_1'
            c_output_1 = tf.layers.dense(ResNet_block_2, 64, tf.nn.relu, name=layer_output_1)
            layer_output_2 = 'c_output_2'
            c_output = tf.layers.dense(c_output_1, 1, name=layer_output_2)

        return c_output

    # initialize the weight of CNN
    def cnn_weight_variable(self, shape):
        init = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(init)

    # initialize the base of CNN
    def cnn_base_variable(self, shape):
        init = tf.constant(0.1, shape=shape)
        return tf.Variable(init)

    # date store, for the sake of calculation, the next action is also saved
    def store_transition(self, s, a, r, s_, a_):
        index = self.memory_counter % self.memory_size
        self.memory[index, :self.n_user * (self.n_subcarrier * 3 + 2)] = s.flatten()
        self.memory[index,
        self.n_user * (self.n_subcarrier * 3 + 2): self.n_user * (self.n_subcarrier * 4 + 2)] = a.flatten()
        self.memory[index, self.n_user * (self.n_subcarrier * 4 + 2)] = r
        self.memory[index, self.n_user * (self.n_subcarrier * 4 + 2) + 1: self.n_user * (
                self.n_subcarrier * 7 + 4) + 1] = s_.flatten()
        self.memory[index, self.n_user * (
                self.n_subcarrier * 7 + 4) + 1: self.n_user * (
                self.n_subcarrier * 8 + 4) + 1] = a_.flatten()
        self.memory_counter += 1

    # choose action based on the current observation state
    def choose_actions(self, user_index, action_SA_all, observation_PA_int):
        self.epsilon = 0

        if np.random.random() > self.epsilon:
            action_PA = action_SA_all[user_index, :] * np.random.randn(self.n_subcarrier)
        else:
            observation_PA_int = observation_PA_int.reshape(-1, self.n_user * (3 * self.n_subcarrier + 2))
            action_PA = self.sess.run(self.online_actor_net, feed_dict={self.s: observation_PA_int})
            action_PA = action_PA.flatten()
            action_PA = action_SA_all[user_index, :] * action_PA
            print(action_PA)
            print(action_SA_all)

        for i in range(len(action_PA)):
            if action_PA[i] > 0:
                action_PA[i] = 1
            elif action_PA[i] < 0:
                action_PA[i] = -1
            else:
                action_PA[i] = 0

        return action_PA

    # choose action based on the next observation state
    def choose_actions_next(self, user_index, action_SA_all, observation_PA_next_int):

        observation_PA_next_int = observation_PA_next_int.reshape(-1, self.n_user * (3 * self.n_subcarrier + 2))
        action_PA = self.sess.run(self.target_actor_net, feed_dict={self.s_: observation_PA_next_int})
        action_PA = action_PA.flatten()
        action_PA = action_SA_all[user_index, :] * action_PA

        for i in range(len(action_PA)):
            if action_PA[i] > 0:
                action_PA[i] = 1
            elif action_PA[i] < 0:
                action_PA[i] = -1
            else:
                action_PA[i] = 0

        return action_PA

