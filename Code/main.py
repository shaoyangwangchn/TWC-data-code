import numpy as np
from communication_environment import Environment
from SA_module import DDPG_SA_Network
from PA_module_each import DDPG_PA_Network
import warnings
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
warnings.filterwarnings('ignore')

# Hyper-parameter related to MC-NOMA communication systems
N_USER = 20
N_SUBCARRIER = 10
POSSION_DENSITY = 2
PL_EXPONENT = 3.6
USER_RADIUS = 300
USER_RADIUS_MIN = 30
N_USER_SUBCARRIER_MAX = 5
P_MAX = np.power(10, 30 / 10)
NOISE_DENSITY = np.power(10, -173 / 10)
POWER_DELTA = np.power(10, 0.5 / 10)
BANWIDTH = 5_000_000
POWER_CHANGE_INDICATOR = 10e-5
QOS_MEAN = 80_000
QOS_VAR = 10
SIC_ERROR = 10e-3

# Hyper-parameter related to reinforcement learning
MAX_EPISODES = 15000  # 最大的周期学习数
MAX_EPISODES_SA = 20000
MAX_EPISODES_PA = 35000
SA_LEARNING_START = 5000
PA_LEARNING_START = 4000
ALL_LEARNING_START = 1000
MAX_EPISODES_PA_EACH = 100
MAX_EPISODES_ALL_EACH = 200
OMEGA_SA_INT = -5
OMEGA_PA_INT_PART1 = -8
OMEGA_PA_INT_PART2 = 3
OMEGA_SA_JO_INNER = 1.5
OMEGA_SA_JO = 0.25
OMEGA_PA_JO = 0.45
OMEGA_PA_JO_EACH = 16
LEARNING_FREQUENCY = 5
BATCH_SIZE = 128
SA_MEMORY_SIZE = 5000
PA_MEMORY_SIZE = 4000
SA_LEARNING_RATE_CRITIC = 0.003
SA_LEARNING_RATE_ACTOR = 0.001
PA_LEARNING_RATE_CRITIC = 0.005
PA_LEARNING_RATE_ACTOR = 0.002
REWARD_DECAY = 0.99
E_GREEDY = 0.9
REPLACE_TARGET_ITER = 5
E_GREEDY_INCRE = 10e-4


def run_env(env, RL):
    step_ALL = 0
    step_SA = 0
    step_PA = 0
    for episode in range(MAX_EPISODES):
        print("EPISODE: ", episode)
        observation_SA = env.SA_reset()
        for steps_SA in range(N_USER):
            action_SA_one = RL[0].choose_actions(observation_SA)
            observation_SA[steps_SA, N_SUBCARRIER + 2: 2 * N_SUBCARRIER + 2] = action_SA_one

        action_SA_all = observation_SA[:, N_SUBCARRIER + 2:2 * N_SUBCARRIER + 2]
        observation_PA = env.PA_reset(action_SA_all)
        for episode_all_each in range(MAX_EPISODES_ALL_EACH):
            action_SA_all_current_sum = np.sum(action_SA_all, 0) <= N_USER_SUBCARRIER_MAX
            suitable_indicator_SA = np.sum(action_SA_all_current_sum + 0)
            if suitable_indicator_SA == N_SUBCARRIER:
                action_PA_all = np.zeros(N_USER * N_SUBCARRIER).reshape(N_USER, N_SUBCARRIER)
                for i in range(N_USER):
                    action_PA_all[i] = RL[i + 1].choose_actions(i, action_SA_all, observation_PA)

                action_PA_indicator = observation_PA[:,
                                      2 * N_SUBCARRIER + 2: 3 * N_SUBCARRIER + 2] + POWER_CHANGE_INDICATOR * action_PA_all
                action_PA_indicator = action_PA_indicator * action_SA_all
                action_PA_indicator = np.maximum(action_PA_indicator, 0)
                action_PA_all = P_MAX * (action_PA_indicator / np.sum(np.sum(action_PA_indicator)))

                observation_PA_h_gain = observation_PA[:, 2:N_SUBCARRIER + 2]
                suitable_indicator_PA = True
                action_PA_all = action_SA_all * action_PA_all
                for i in range(N_SUBCARRIER):
                    action_PA_reduce = np.argsort(-action_PA_all[:, i])
                    for j in range(len(action_PA_reduce)):
                        if action_PA_all[action_PA_reduce[j], i] > 10e-100:
                            if observation_PA_h_gain[action_PA_reduce[j], i] * (
                                    action_PA_all[action_PA_reduce[j], i] - np.sum(
                                action_PA_all[action_PA_reduce[j:], i])) < POWER_DELTA:
                                suitable_indicator_PA = False
                                break

                if suitable_indicator_PA == True:
                    reward_SA_jo, reward_PA_jo, observation_SA_all, observation_PA_next_jo = env.ALL_step(
                        observation_SA, observation_PA, action_SA_all, action_PA_all)

                    for i in range(N_USER + 1):
                        if i == 0:
                            observation_SA_current = np.zeros(N_USER * (2 * N_SUBCARRIER + 2)).reshape(N_USER,
                                                                                                       2 * N_SUBCARRIER + 2)
                            observation_SA_next = np.zeros(N_USER * (2 * N_SUBCARRIER + 2)).reshape(N_USER,
                                                                                                    2 * N_SUBCARRIER + 2)
                            for j in range(N_USER):
                                action_SA_one_jo = action_SA_all[i]
                                observation_SA_next[j, N_SUBCARRIER + 2: 2 * N_SUBCARRIER + 2] = action_SA_one_jo
                                RL[i].store_transition(observation_SA_current, action_SA_one_jo, reward_SA_jo,
                                                       observation_SA_next)
                                observation_SA_current = observation_SA_next
                        else:
                            action_PA_all_next = np.zeros(N_USER * N_SUBCARRIER).reshape(N_USER, N_SUBCARRIER)
                            for i in range(N_USER):
                                action_PA_all_next[i] = RL[i + 1].choose_actions_next(i, action_SA_all,
                                                                                      observation_PA_next_jo)
                            RL[i].store_transition(observation_PA, action_PA_all, reward_PA_jo[i - 1],
                                                   observation_PA_next_jo, action_PA_all_next)

                    if (step_ALL > ALL_LEARNING_START) and (episode % LEARNING_FREQUENCY == 0):
                        for i in range(N_USER + 1):
                            RL[i].learn()

                    observation_SA_all[:, N_SUBCARRIER + 2: 2 * N_SUBCARRIER + 2] = 0
                    observation_SA = observation_SA_all
                    observation_PA = observation_PA_next_jo
                    step_ALL = step_ALL + 1
                    print("all is learning, step_all is : ", step_ALL)

                else:
                    for episode_PA in range(MAX_EPISODES_PA):
                        observation_PA_int = env.PA_reset(action_SA_all)
                        for episode_PA_each in range(MAX_EPISODES_PA_EACH):
                            action_PA_all_int = np.zeros(N_USER * N_SUBCARRIER).reshape(N_USER, N_SUBCARRIER)
                            for i in range(N_USER):
                                action_PA_all_int[i] = RL[i + 1].choose_actions(i, action_SA_all, observation_PA_int)

                            action_PA_indicator = observation_PA_int[:,
                                                  2 * N_SUBCARRIER + 2: 3 * N_SUBCARRIER + 2] + POWER_CHANGE_INDICATOR * action_PA_all_int
                            action_PA_indicator = action_PA_indicator * action_SA_all
                            action_PA_indicator = np.maximum(action_PA_indicator, 0)
                            action_PA_all_int = P_MAX * (action_PA_indicator / np.sum(np.sum(action_PA_indicator)))

                            reward_PA_int, observation_PA_next_int = env.PA_step(action_SA_all, observation_PA_int,
                                                                                 action_PA_all_int)

                            action_PA_all_next = np.zeros(N_USER * N_SUBCARRIER).reshape(N_USER, N_SUBCARRIER)
                            for i in range(N_USER):
                                action_PA_all_next[i] = RL[i + 1].choose_actions_next(i, action_SA_all,
                                                                                      observation_PA_next_int)
                            for i in range(N_USER):
                                RL[i + 1].store_transition(observation_PA_int, action_PA_all_int, reward_PA_int[i],
                                                           observation_PA_next_int, action_PA_all_next)

                            if (step_PA > PA_LEARNING_START) and (step_PA % LEARNING_FREQUENCY == 0):
                                # action_PA_all_next = action_PA_all_next.flatten()
                                for i in range(N_USER):
                                    RL[i + 1].learn()

                            step_PA = step_PA + 1
                            observation_PA_int = observation_PA_next_int

                        print("pa is learning, step_pa is: ", step_PA)

            else:
                for episode_SA in range(MAX_EPISODES_SA):
                    observation_SA_int = env.SA_reset()
                    for steps_SA in range(N_USER):
                        action_SA_one_int = RL[0].choose_actions(observation_SA_int)
                        reward_SA_int, observation_SA_next_int = env.SA_step(steps_SA, observation_SA_int,
                                                                             action_SA_one_int)
                        RL[0].store_transition(observation_SA_int, action_SA_one_int, reward_SA_int,
                                               observation_SA_next_int)

                        if (step_SA > SA_LEARNING_START) and (step_SA % LEARNING_FREQUENCY == 0):
                            RL[0].learn()

                        step_SA = step_SA + 1
                        observation_SA_int = observation_SA_next_int

                    print("sa is learning, step_sa: ", step_SA)


if __name__ == "__main__":
    # -----------------Build the MC-NOMA communication environment---------
    print('------------Build the MC-NOMA communication environment---------')
    env = Environment(N_USER,
                      USER_RADIUS,
                      USER_RADIUS_MIN,
                      N_SUBCARRIER,
                      NOISE_DENSITY,
                      P_MAX,
                      QOS_MEAN,
                      QOS_VAR,
                      POSSION_DENSITY,
                      PL_EXPONENT,
                      N_USER_SUBCARRIER_MAX,
                      OMEGA_SA_INT,
                      POWER_DELTA,
                      OMEGA_PA_INT_PART1,
                      OMEGA_PA_INT_PART2,
                      SIC_ERROR,
                      BANWIDTH,
                      OMEGA_SA_JO_INNER,
                      OMEGA_SA_JO,
                      OMEGA_PA_JO,
                      OMEGA_PA_JO_EACH,
                      POWER_CHANGE_INDICATOR, )
    # ----------------------------Build SA- and PA-agents------------------
    print('-----------------------Build SA- and PA-agents------------------')

    # store all agents (A total of N_USER+1)
    RL = dict()

    for i in range(N_USER + 1):
        if i == 0:
            # build the SA agent
            RL[i] = DDPG_SA_Network(N_USER,
                                    N_SUBCARRIER,
                                    SA_LEARNING_RATE_CRITIC,
                                    SA_LEARNING_RATE_ACTOR,
                                    REWARD_DECAY,
                                    E_GREEDY,
                                    REPLACE_TARGET_ITER,
                                    SA_MEMORY_SIZE,
                                    BATCH_SIZE,
                                    E_GREEDY_INCRE, )
        else:
            # build the PA agents (A total of N_USER)
            RL[i] = DDPG_PA_Network(N_USER,
                                    N_SUBCARRIER,
                                    i - 1,
                                    PA_LEARNING_RATE_CRITIC,
                                    PA_LEARNING_RATE_ACTOR,
                                    REWARD_DECAY,
                                    E_GREEDY,
                                    REPLACE_TARGET_ITER,
                                    PA_MEMORY_SIZE,
                                    BATCH_SIZE,
                                    E_GREEDY_INCRE,
                                    POWER_CHANGE_INDICATOR,
                                    P_MAX, )

    # -------------------Staring the reinforcement learning ---------------
    print('# ------------Staring the reinforcement learning----------------')
    run_env(env, RL)
