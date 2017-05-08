# Copyright (c) 2016, NVIDIA CORPORATION. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#  * Neither the name of NVIDIA CORPORATION nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
# OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

from datetime import datetime
from multiprocessing import Process, Queue, Value

import numpy as np
import time, sys

from Config import Config
from Environment import Environment
from Experience import Experience


class ProcessAgent(Process):
    def __init__(self, id, prediction_q, training_q, episode_log_q):
        super(ProcessAgent, self).__init__()

        self.id = id
        self.prediction_q = prediction_q
        self.training_q = training_q
        self.episode_log_q = episode_log_q
        self.concurrent_episodes = Config.CONCURRENT_EPISODES

        self.envs = [Environment() for i in range(self.concurrent_episodes)]
        self.reward_sums = [0 for i in range(self.concurrent_episodes)]
        self.time_counts = [0 for i in range(self.concurrent_episodes)]
        self.experiences_list = [[] for i in range(self.concurrent_episodes)]
        self.total_rewards = [0 for i in range(self.concurrent_episodes)]
        self.total_lengths = [0 for i in range(self.concurrent_episodes)]
        self.num_actions = self.envs[0].get_num_actions()
        self.actions = np.arange(self.num_actions)

        self.discount_factor = Config.DISCOUNT
        self.wait_q = Queue(maxsize=self.concurrent_episodes)
        self.exit_flag = Value('i', 0)

    @staticmethod
    def _accumulate_rewards(experiences, discount_factor, terminal_reward):
        reward_sum = terminal_reward
        for t in reversed(range(0, len(experiences)-1)):
            r = np.clip(experiences[t].reward, Config.REWARD_MIN, Config.REWARD_MAX)
            reward_sum = discount_factor * reward_sum + r
            experiences[t].reward = reward_sum
        return experiences[:-1]

    def convert_data(self, experiences):
        x_ = np.array([exp.state for exp in experiences])
        a_ = np.eye(self.num_actions)[np.array([exp.action for exp in experiences])].astype(np.float32)
        r_ = np.array([exp.reward for exp in experiences])
        return x_, r_, a_

    def select_action(self, prediction):
        if Config.PLAY_MODE:
            action = np.argmax(prediction)
        else:
            action = np.random.choice(self.actions, p=prediction)
        return action

    def reset_episode(self, idx):
        env = self.envs[idx]
        self.reward_sums[idx] = 0
        self.experiences_list[idx] = []
        self.time_counts[idx] = 0
        self.total_lengths[idx] = 0
        self.total_rewards[idx] = 0

        env.reset()
        while env.current_state is None:
            env.step(0)
            continue

    def predict_episode(self, idx):
        env = self.envs[idx]
        self.prediction_q.put((self.id, idx, env.current_state))

    def step_episode(self, idx, prediction, value):
        env = self.envs[idx]
        experiences = self.experiences_list[idx]

        action = self.select_action(prediction)
        sys.stdout.flush()
        reward, done = env.step(action)
        self.reward_sums[idx] += reward
        exp = Experience(env.previous_state, action, prediction, reward, done)
        experiences.append(exp)

        if done or self.time_counts[idx] == Config.TIME_MAX:
            terminal_reward = 0 if done else value

            updated_exps = ProcessAgent._accumulate_rewards(experiences, self.discount_factor, terminal_reward)
            x_, r_, a_ = self.convert_data(updated_exps)
            self.total_rewards[idx] += self.reward_sums[idx]
            self.total_lengths[idx] += len(r_) + 1
            self.training_q.put((x_, r_, a_))

            # reset the tmax count
            self.time_counts[idx] = 0
            # keep the last experience for the next batch
            # experiences = [experiences[-1]]
            self.experiences_list[idx] = [experiences[-1]]
            self.reward_sums[idx] = 0.0

        self.time_counts[idx] += 1
        if done:
            self.episode_log_q.put((datetime.now(), self.total_rewards[idx], self.total_lengths[idx]))
            self.reset_episode(idx)
        

    def run(self):
        # randomly sleep up to 1 second. helps agents boot smoothly.
        time.sleep(np.random.rand())
        np.random.seed(np.int32(time.time() % 1 * 1000 + self.id * 10))

        for idx in range(self.concurrent_episodes):
            self.reset_episode(idx)
            self.predict_episode(idx)

        while self.exit_flag.value == 0:
            s = time.time()
            idx, p, v = self.wait_q.get()
            t1 = time.time() - s 
            self.step_episode(idx, p, v)
            self.predict_episode(idx)
            t2 = time.time() - s 
