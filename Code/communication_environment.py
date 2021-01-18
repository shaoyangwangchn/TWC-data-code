import numpy as np


class Environment():
    def __init__(self,
                 n_user,
                 user_radius,
                 user_radius_min,
                 n_subcarrier,
                 noise_density,
                 p_max,
                 QoS_mean,
                 QoS_var,
                 poisson_density,
                 pl_exponent,
                 n_user_subcarrier_max,
                 omega_sa_int,
                 power_delta,
                 omega_pa_int_part1,
                 omega_pa_int_part2,
                 sic_error,
                 bandwith,
                 omega_sa_jo_inner,
                 omega_sa_jo,
                 omega_pa_jo,
                 omega_sa_jo_each,
                 power_change_indicator,):
        super(Environment, self).__init__()
        self.n_user = n_user
        self.user_radius = user_radius
        self.user_radius_min = user_radius_min
        self.n_subcarrier = n_subcarrier
        self.noise_density = noise_density
        self.p_max = p_max
        self.QoS_mean = QoS_mean
        self.QoS_var = QoS_var
        self.poisson_density = poisson_density
        self.pl_exponent = pl_exponent
        self.n_user_subcarrier_max = n_user_subcarrier_max
        self.omega_sa_int = omega_sa_int
        self.power_delta = power_delta
        self.omega_pa_int_part1 = omega_pa_int_part1
        self.omega_pa_int_part2 = omega_pa_int_part2
        self.sic_error = sic_error
        self.bandwith = bandwith
        self.omega_sa_jo_inner = omega_sa_jo_inner
        self.omega_sa_jo = omega_sa_jo
        self.omega_pa_jo = omega_pa_jo
        self.omega_pa_jo_each = omega_sa_jo_each
        self.power_change_indicator = power_change_indicator

        self.distance = self.generate_distance()
        self.weight = self.cal_weight()
        self.QoS = np.random.normal(self.QoS_mean, self.QoS_var, self.n_user)
        self.h_gain = self.H_gain()
        self.noise = np.random.normal(0, self.bandwith * self.noise_density / self.n_subcarrier, self.n_user * self.n_subcarrier).reshape(self.n_user, self.n_subcarrier)

    def SA_reset(self):
        observation_SA = np.zeros(self.n_user * (2 * self.n_subcarrier + 2)).reshape(self.n_user,
                                                                                     2 * self.n_subcarrier + 2)
        observation_SA[:, 0] = self.weight
        observation_SA[:, 1] = self.QoS
        observation_SA[:, 2:self.n_subcarrier + 2] = self.h_gain
        return observation_SA

    def PA_reset(self, action_SA_all):
        observation_PA = np.zeros(self.n_user * (3 * self.n_subcarrier + 2)).reshape(self.n_user,
                                                                                     3 * self.n_subcarrier + 2)
        observation_PA[:, 0] = self.weight
        observation_PA[:, 1] = self.QoS
        observation_PA[:, 2:self.n_subcarrier + 2] = self.h_gain
        observation_PA[:, self.n_subcarrier + 2: 2 * self.n_subcarrier + 2] = action_SA_all
        observation_PA_power_indicator = action_SA_all * np.random.random(self.n_user * self.n_subcarrier).reshape(self.n_user, self.n_subcarrier)
        observation_PA_power_indicator = np.maximum(observation_PA_power_indicator, 0)
        observation_PA_power = self.p_max * (observation_PA_power_indicator/np.sum(np.sum(observation_PA_power_indicator)))
        observation_PA[:, 2 * self.n_subcarrier + 2: 3 * self.n_subcarrier + 2] = observation_PA_power

        return observation_PA

    def SA_step(self, steps_SA, observation_SA_int, action_SA_one_int):
        observation_SA_int[steps_SA, self.n_subcarrier + 2: 2 * self.n_subcarrier + 2] = action_SA_one_int
        observation_SA_next = observation_SA_int

        action_SA_all_current = observation_SA_next[:, self.n_subcarrier + 2: 2 * self.n_subcarrier + 2]
        action_SA_all_current_sum = np.sum(action_SA_all_current, 0) <= self.n_user_subcarrier_max
        C4_indictor = np.sum(action_SA_all_current_sum + 0)

        if C4_indictor == self.n_subcarrier:
            reward_SA_int = 0
        else:
            reward_SA_int = self.omega_sa_int

        return reward_SA_int, observation_SA_next

    def PA_step(self, action_SA_all, observation_PA_int, action_PA_all_int):
        observation_PA_int[:, 2 * self.n_subcarrier + 2: 3 * self.n_subcarrier + 2] = action_PA_all_int
        observation_PA_next = observation_PA_int

        observation_PA_h_gain = observation_PA_int[:, 2:self.n_subcarrier + 2]

        reward_PA_part1 = np.zeros(self.n_user * self.n_subcarrier).reshape(self.n_user, self.n_subcarrier)
        action_PA_all_int = action_SA_all * action_PA_all_int
        for i in range(self.n_subcarrier):
            action_PA_reduce = np.argsort(-action_PA_all_int[:, i])
            for j in range(len(action_PA_reduce)):
                if action_PA_all_int[action_PA_reduce[j], i] > 10e-100:
                    if observation_PA_h_gain[action_PA_reduce[j], i] * (action_PA_all_int[action_PA_reduce[j], i] - np.sum(
                            action_PA_all_int[action_PA_reduce[j:], i])) < self.power_delta:
                        reward_PA_part1[action_PA_reduce[j], i] = 1

        reward_PA_part1 = (np.sum(reward_PA_part1, 1) > 0) + 0
        reward_PA_part1 = reward_PA_part1 * self.omega_pa_int_part1

        reward_PA_part2 = np.zeros(self.n_user)
        for i in range(self.n_user):
            throughput_user_each = self.Cal_throught(i, action_SA_all, observation_PA_h_gain, action_PA_all_int)
            QoS_user_each = self.QoS[i]
            reward_PA_part2[i] = self.omega_pa_int_part2 * (throughput_user_each - QoS_user_each)

        reward_PA = reward_PA_part1 + reward_PA_part2

        return reward_PA, observation_PA_next

    def ALL_step(self, observation_SA, observation_PA, action_SA_all, action_PA_all):

        observation_SA_all = observation_SA

        observation_PA[:, 2 * self.n_subcarrier + 2: 3 * self.n_subcarrier + 2] = action_PA_all
        observation_PA_next_jo = observation_PA

        reward_PA_jo = np.zeros(self.n_user)
        throughput_sum_buff = np.zeros(self.n_user)
        observation_PA_h_gain = observation_PA[:, 2:self.n_subcarrier + 2]
        for i in range(self.n_user):
            throughput_sum_each = self.Cal_throught(i, action_SA_all, observation_PA_h_gain, action_PA_all)
            throughput_sum_buff[i] = self.weight[i] * throughput_sum_each

        throughput_sum = np.sum(throughput_sum_buff)
        reward_SA_jo = self.omega_sa_jo * np.exp(self.omega_sa_jo_inner * throughput_sum)

        for i in range(self.n_user):
            reward_PA_jo[i] = (throughput_sum_buff[i] / throughput_sum) * (
                        self.omega_pa_jo * np.exp(self.omega_pa_jo_each * throughput_sum))

        return reward_SA_jo, reward_PA_jo, observation_SA_all, observation_PA_next_jo

    def H_gain(self):
        path_loss = np.power(self.distance, self.pl_exponent)
        path_loss_buff = np.repeat(path_loss, self.n_subcarrier).reshape(self.n_user, self.n_subcarrier)
        h_gain_g = np.random.normal(0, 1, self.n_user * self.n_subcarrier).reshape(self.n_user, self.n_subcarrier)
        h_gain = h_gain_g / path_loss_buff
        return h_gain

    def Cal_throught(self, index_user, action_SA_all, observation_PA_h_gain, action_PA_all_int):
        throughput_each_user = 0
        action_PA_all_int = action_SA_all * action_PA_all_int
        # action_PA_all_int = self.p_max * (action_PA_all_int / np.sum(np.sum(action_PA_all_int)))
        for i in range(self.n_subcarrier):
            power_current = action_PA_all_int[index_user, i]
            if power_current > 10e-100:
                h_gian_current = observation_PA_h_gain[index_user, i]
                action_PA_reduce = np.argsort(-action_PA_all_int[:, i])
                index_user_current_subcarrier = int(np.argwhere(action_PA_reduce == index_user))
                SIC_error = self.sic_error * np.sum(
                    observation_PA_h_gain[action_PA_reduce[: index_user_current_subcarrier], i] * action_PA_all_int[
                        action_PA_reduce[: index_user_current_subcarrier], i])
                interference = np.sum(
                    observation_PA_h_gain[action_PA_reduce[index_user_current_subcarrier + 1:], i] * action_PA_all_int[
                        action_PA_reduce[index_user_current_subcarrier + 1:], i])
                noise = self.noise[index_user, i]
                throughput_subcarrier_each = self.bandwith / self.n_subcarrier * np.log2(
                    1 + (power_current * h_gian_current) / (SIC_error + interference + noise))
                throughput_each_user = throughput_each_user + throughput_subcarrier_each
        return throughput_each_user

    def cal_weight(self):
        weight = self.distance / np.max(self.distance)
        return weight

    def generate_distance(self):
        distance = np.random.poisson(self.poisson_density, self.n_user)
        distance_min = np.min(distance)
        distance_max = np.max(distance)
        k_convert = (self.user_radius - self.user_radius_min)/(distance_max - distance_min)
        distance = self.user_radius_min + k_convert * (distance - distance_min)
        return distance
