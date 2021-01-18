import numpy as np
import tensorflow as tf


class DDPG_SA_Network:
    def __init__(
            self,
            n_user,
            n_subcarrier,
            learning_rate_critic,
            learning_rate_actor,
            reward_decay,
            e_greedy,
            replace_target_iter,
            memory_size,
            batch_size,
            e_greedy_increment,
    ):
        self.n_user = n_user
        self.n_subcarrier = n_subcarrier
        self.lr_critic = learning_rate_critic
        self.lr_actor = learning_rate_actor
        self.gamma = reward_decay
        self.epsilon_max = e_greedy
        self.replace_target_iter = replace_target_iter
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.epsilon_increment = e_greedy_increment
        self.epsilon = 0.0 if e_greedy_increment is not None else self.epsilon_max
        self.learn_step_counter = 0
        self.memory_counter = 0  # Count the sample number of the experience replay pool
        self.memory = np.zeros((self.memory_size, (self.n_subcarrier + 1) * (4 * self.n_user + 1)))  # experience pool
        self.sess = tf.Session()

        # collect network parameters
        oa_net_name = 'online_actor_net'
        oc_net_name = 'online_critic_net'
        ta_net_name = 'target_actor_net'
        tc_net_name = 'target_critic_net'
        self.oa_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=oa_net_name)
        self.oc_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=oc_net_name)
        self.ta_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=ta_net_name)
        self.tc_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=tc_net_name)

        # soft update
        soft_replace_name = 'soft_replacement'
        with tf.variable_scope(soft_replace_name):
            self.actor_replace_op = [tf.assign(ta, oa) for ta, oa in zip(self.ta_params, self.oa_params)]
            self.critic_replace_op = [tf.assign(tc, oc) for tc, oc in zip(self.tc_params, self.oc_params)]

        # --------------------------build network--------------------------
        # naming
        state_name = 's'
        state_next_name = 's_'
        reward_name = 'r'
        action_name = 'a'
        self.s = tf.placeholder(tf.float32, [None, self.n_user * (2 * self.n_subcarrier + 2)],
                                name=state_name)  # input State
        self.s_ = tf.placeholder(tf.float32, [None, self.n_user * (2 * self.n_subcarrier + 2)],
                                 name=state_next_name)  # input Next State
        self.r = tf.placeholder(tf.float32, [None, ], name=reward_name)  # input Reward
        self.a = tf.placeholder(tf.float32, [None, self.n_subcarrier], name=action_name)  # input Action
        self.a_ = tf.placeholder(tf.float32, [None, self.n_subcarrier], name=action_name)  # input next Action

        # built online actor network
        self.online_actor_net_A, self.online_actor_net_B = self.built_actor_net(name="online", s=self.s)
        print("||-----------SA_agent online actor network is successfully built -------")
        # built target actor network, and also to calculate the next action
        self.target_actor_net_A, self.target_actor_net_B = self.built_actor_net(name="target", s=self.s_)
        print("||-----------SA_agent target actor network is successfully built -------")
        # built online critic network
        self.online_critic_net = self.built_critic_net(name="online", s=self.s, a=self.a)
        print("||-----------SA_agent online critic network is successfully built -------")
        # built target critic network, and also to calculate the target Q
        self.target_critic_net = self.built_critic_net(name="target", s=self.s_, a=self.a_)
        print("||-----------SA_agent target critic network is successfully built -------\n")

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
            path_name = "logs/SA_agent" + "/"
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

        # perform target actor network, obtain next action for training critic network
        a_output_A_next, a_output_B_next = self.sess.run([self.target_actor_net_A, self.target_actor_net_B],
                                                         feed_dict={self.s_: batch_memory[:, self.n_user * (
                                                                     self.n_subcarrier * 2 + 2) + self.n_subcarrier + 1: self.n_user * (
                                                                     self.n_subcarrier * 4 + 4) + self.n_subcarrier + 1], })

        # calculate the real next action output by the B network, based on the A network
        a_next = self.action_convert(a_output_A_next, a_output_B_next)

        # optimize network
        _, _ = self.sess.run(
            [self.actor_train_op, self.critic_train_op],
            feed_dict={
                self.s: batch_memory[:, :self.n_user * (self.n_subcarrier * 2 + 2)],
                self.a: batch_memory[:, self.n_user * (self.n_subcarrier * 2 + 2): self.n_user * (
                        self.n_subcarrier * 2 + 2) + self.n_subcarrier],
                self.r: batch_memory[:, self.n_user * (self.n_subcarrier * 2 + 2) + self.n_subcarrier],
                self.s_: batch_memory[:,
                         self.n_user * (self.n_subcarrier * 2 + 2) + self.n_subcarrier + 1: self.n_user * (
                                 self.n_subcarrier * 4 + 4) + self.n_subcarrier + 1],
                self.a_: a_next,
            })

        # update utilize probability
        self.epsilon = self.epsilon + self.epsilon_increment if self.epsilon < self.epsilon_max else self.epsilon_max

        # update learning step
        self.learn_step_counter += 1


    # built actor network
    def built_actor_net(self, name, s):

        eval_net_name = name + '_actor_net'  # naming
        with tf.variable_scope(eval_net_name):
            # input layers containing the "Input Design"
            a_input = []  # initialize
            for i in range(self.n_user):
                s_each = s[:, i * (2 * self.n_subcarrier + 2):(i + 1) * (2 * self.n_subcarrier + 2)]
                layer_input_1_name = 'a_input_1_' + str(i)
                a_input_1_each = tf.layers.dense(s_each, 64, tf.nn.relu, name=layer_input_1_name)
                layer_input_2_name = 'a_input_2_' + str(i)
                a_input_2_each = tf.layers.dense(a_input_1_each, 32, tf.nn.relu, name=layer_input_2_name)
                layer_input_3_name = 'a_input_3_' + str(i)
                a_input_3_each = tf.layers.dense(a_input_2_each, 32, tf.nn.relu, name=layer_input_3_name)

                if i == 0:
                    a_input = a_input_3_each
                else:
                    a_input = tf.concat([a_input, a_input_3_each], axis=-1)  # action merge

            # hidden layers (including ResNet)
            layer_hidden_1_name = 'a_hidden_1'
            a_hidden_1 = tf.layers.dense(a_input, 128, tf.nn.relu, name=layer_hidden_1_name)
            layer_hidden_2_name = 'a_hidden_2'
            a_hidden_2 = tf.layers.dense(a_hidden_1, 128, tf.nn.relu, name=layer_hidden_2_name)
            layer_hidden_3_name = 'a_hidden_3'
            a_hidden_3 = tf.layers.dense(a_hidden_2, 128, tf.nn.relu, name=layer_hidden_3_name)

            res_base_1 = self.resnet_base_variable([128])
            ResNet_block_1 = tf.add(a_hidden_1, a_hidden_3)
            ResNet_block_1 = tf.nn.relu(ResNet_block_1 + res_base_1)

            layer_hidden_4_name = 'a_hidden_4'
            a_hidden_4 = tf.layers.dense(ResNet_block_1, 128, tf.nn.relu, name=layer_hidden_4_name)
            layer_hidden_4_name = 'a_hidden_5'
            a_hidden_5 = tf.layers.dense(a_hidden_4, 128, tf.nn.relu, name=layer_hidden_4_name)

            res_base_2 = self.resnet_base_variable([128])
            ResNet_block_2 = tf.add(ResNet_block_1, a_hidden_5)
            ResNet_block_2 = tf.nn.relu(ResNet_block_2 + res_base_2)

            # output layers
            # A network
            layer_output_A_1_name = 'a_output_A_1'
            a_output_A_1 = tf.layers.dense(ResNet_block_2, 128, tf.nn.relu, name=layer_output_A_1_name)
            layer_output_A_2_name = 'a_output_A_2'
            a_output_A_2 = tf.layers.dense(a_output_A_1, 128, tf.nn.relu, name=layer_output_A_2_name)
            layer_output_A_3_name = 'a_output_A_3'
            a_output_A_3 = tf.layers.dense(a_output_A_2, self.n_subcarrier, tf.nn.softmax, name=layer_output_A_3_name)
            # B network
            layer_output_B_1_name = 'a_output_B_1'
            a_output_B_1 = tf.layers.dense(ResNet_block_2, 128, tf.nn.relu, name=layer_output_B_1_name)
            layer_output_B_2_name = 'a_output_B_2'
            a_output_B_2 = tf.layers.dense(a_output_B_1, 128, tf.nn.relu, name=layer_output_B_2_name)
            layer_output_B_3_name = 'a_output_B_3'
            a_output_B_3 = tf.layers.dense(a_output_B_2, self.n_subcarrier, tf.nn.softmax, name=layer_output_B_3_name)

        return a_output_A_3, a_output_B_3

    # built critic network
    def built_critic_net(self, name, s, a):
        eval_net_name = name + '_critic_net'  # naming
        with tf.variable_scope(eval_net_name):
            # input layers
            c_input = []
            for i in range(self.n_user):
                s_each = s[:, i * (2 * self.n_subcarrier + 2):(i + 1) * (2 * self.n_subcarrier + 2)]
                a_each = a[:, i * self.n_subcarrier:(i + 1) * self.n_subcarrier]
                c_input_each = tf.concat([s_each, a_each], axis=-1)
                layer_input_1_name = 'c_input_1_' + str(i)
                c_input_1_each = tf.layers.dense(c_input_each, 64, tf.nn.relu, name=layer_input_1_name)
                layer_input_2_name = 'c_input_2_' + str(i)
                c_input_2_each = tf.layers.dense(c_input_1_each, 32, tf.nn.relu, name=layer_input_2_name)
                layer_input_3_name = 'c_input_3_' + str(i)
                c_input_3_each = tf.layers.dense(c_input_2_each, 32, tf.nn.relu, name=layer_input_3_name)

                if i == 0:
                    c_input = c_input_3_each
                else:
                    c_input = tf.concat([c_input, c_input_3_each], axis=-1)

            # hidden layers (including ResNet)
            layer_hidden_1_name = 'c_hidden_1'
            c_hidden_1 = tf.layers.dense(c_input, 128, tf.nn.relu, name=layer_hidden_1_name)
            layer_hidden_2_name = 'c_hidden_2'
            c_hidden_2 = tf.layers.dense(c_hidden_1, 128, tf.nn.relu, name=layer_hidden_2_name)
            layer_hidden_3_name = 'c_hidden_3'
            c_hidden_3 = tf.layers.dense(c_hidden_2, 128, tf.nn.relu, name=layer_hidden_3_name)

            res_base_1 = self.resnet_base_variable([128])
            ResNet_block_1 = tf.add(c_hidden_1, c_hidden_3)
            ResNet_block_1 = tf.nn.relu(ResNet_block_1 + res_base_1)

            layer_hidden_4_name = 'c_hidden_4'
            c_hidden_4 = tf.layers.dense(ResNet_block_1, 128, tf.nn.relu, name=layer_hidden_4_name)
            layer_hidden_4_name = 'c_hidden_5'
            c_hidden_5 = tf.layers.dense(c_hidden_4, 128, tf.nn.relu, name=layer_hidden_4_name)

            res_base_2 = self.resnet_base_variable([128])
            ResNet_block_2 = tf.add(ResNet_block_1, c_hidden_5)
            ResNet_block_2 = tf.nn.relu(ResNet_block_2 + res_base_2)

            # output layers
            layer_output_1 = 'c_output_1'
            c_output_1 = tf.layers.dense(ResNet_block_2, 64, tf.nn.relu, name=layer_output_1)
            layer_output_2 = 'c_output_2'
            c_output = tf.layers.dense(c_output_1, 1, name=layer_output_2)

        return c_output

    # date store
    def store_transition(self, s, a, r, s_):
        index = self.memory_counter % self.memory_size
        self.memory[index, :self.n_user * (self.n_subcarrier * 2 + 2)] = s.flatten()
        self.memory[index,
        self.n_user * (self.n_subcarrier * 2 + 2): self.n_user * (self.n_subcarrier * 2 + 2) + self.n_subcarrier] = a
        self.memory[index, self.n_user * (self.n_subcarrier * 2 + 2) + self.n_subcarrier] = r
        self.memory[index, self.n_user * (self.n_subcarrier * 2 + 2) + self.n_subcarrier + 1: self.n_user * (
                self.n_subcarrier * 4 + 4) + self.n_subcarrier + 1] = s_.flatten()
        self.memory_counter += 1

    # concert the action of B network to the real action based on the action of network
    def action_convert(self, a_output_A_3, a_output_B_3):
        a_output = np.zeros(a_output_B_3.shape)
        a_output_A_3_max_index = np.argmax(a_output_A_3, 1)
        for i in range(len(a_output_A_3)):
            a_output_B_3_reduce_index = np.argsort(-a_output_B_3[i])
            a_output_B_3_top_index = a_output_B_3_reduce_index[
                                           0:a_output_A_3_max_index[i] + 1]
            a_output[i, a_output_B_3_top_index] = 1

        return a_output

    # choose action based on the observation state
    def choose_actions(self, observation_SA):
        self.epsilon = 0
        if np.random.random() > self.epsilon:
            action_SA = (np.random.random(self.n_subcarrier) >= 0.5) + 0
        else:
            a_output_A, a_output_B = self.sess.run([self.online_actor_net_A, self.online_actor_net_B],
                                                   feed_dict={self.s: observation_SA})
            action_SA = self.action_convert(a_output_A, a_output_B)
        return action_SA

    # initialize the base of ResNet
    def resnet_base_variable(self, shape):
        init = tf.constant(0.1, shape=shape)
        return tf.Variable(init)