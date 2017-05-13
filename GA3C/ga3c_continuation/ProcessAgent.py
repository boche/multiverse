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
from uuid import uuid4

import numpy as np
import time

from Config import Config
from Environment import Environment
from Experience import Experience


class ProcessAgent(Process):
    def __init__(self, id, prediction_q, combining_q, episode_log_q):
        super(ProcessAgent, self).__init__()

        self.id = id
        self.prediction_q = prediction_q
        self.combining_q = combining_q
        self.episode_log_q = episode_log_q

        self.env = Environment()
        self.num_actions = self.env.get_num_actions()
        self.actions = np.arange(self.num_actions)

        self.discount_factor = Config.DISCOUNT
        # one frame at a time
        self.wait_q = Queue(maxsize=1)
        self.exit_flag = Value('i', 0)

    @staticmethod
    def _accumulate_rewards(experiences, discount_factor, terminal_reward):
        reward_sum = terminal_reward
        for t in reversed(range(0, len(experiences)-1)):
            r = np.clip(experiences[t].reward, Config.REWARD_MIN, Config.REWARD_MAX)
            reward_sum = discount_factor * reward_sum + r
            experiences[t].reward = reward_sum
        return experiences[:-1]

    def predict(self, state):
        state_id = uuid4()
        # put the state in the prediction q
        self.prediction_q.put((self.id, state_id, state))
        # wait for the prediction to come back
        p, v = self.wait_q.get()
        return state_id, p, v

    def select_action(self, prediction):
        if Config.PLAY_MODE:
            action = np.argmax(prediction)
        else:
            action = np.random.choice(self.actions, p=prediction)
        return action

    def run_episode(self):
        self.env.reset()
        done = False
        experiences = []

        time_count = 0
        total_reward = 0.0
        total_length = 0

        while not done:
            # very first few frames
            if self.env.current_state is None:
                self.env.step(0)  # 0 == NOOP
                continue

            state_id, prediction, value = self.predict(self.env.current_state)
            action = self.select_action(prediction)
            reward, done = self.env.step(action)
            total_reward += reward
            exp = Experience(state_id, self.env.previous_state, action, prediction, reward, done)
            experiences.append(exp)

            if done or time_count == Config.TIME_MAX:
                terminal_reward = 0 if done else value

                if done:
                    # add a dummy experience
                    experiences.append(None)

                updated_exps = ProcessAgent._accumulate_rewards(experiences, self.discount_factor, terminal_reward)
                for u_exp in updated_exps:
                    self.combining_q.put((u_exp.sid, u_exp.reward, u_exp.action))

                # reset the tmax count
                time_count = 0
                # keep the last experience for the next batch
                experiences = [experiences[-1]]

            time_count += 1
            total_length += 1
    
        return total_reward, total_length

    def run(self):
        # randomly sleep up to 1 second. helps agents boot smoothly.
        time.sleep(np.random.rand())
        np.random.seed(np.int32(time.time() % 1 * 1000 + self.id * 10))

        while self.exit_flag.value == 0:
            total_reward, total_length = self.run_episode()
            self.episode_log_q.put((datetime.now(), total_reward, total_length))
