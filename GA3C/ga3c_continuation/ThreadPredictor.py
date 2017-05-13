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

from threading import Thread

import numpy as np

from Config import Config
from Batch import Batch


class ThreadPredictor(Thread):
    def __init__(self, server, id):
        super(ThreadPredictor, self).__init__()
        self.setDaemon(True)

        self.id = id
        self.server = server
        self.exit_flag = False

    def run(self):
        agent_ids = np.zeros(Config.PREDICTION_BATCH_SIZE, dtype=np.uint16)
        state_id = [None] * Config.PREDICTION_BATCH_SIZE
        states = np.zeros(
            (Config.PREDICTION_BATCH_SIZE, Config.IMAGE_HEIGHT, Config.IMAGE_WIDTH, Config.STACKED_FRAMES),
            dtype=np.float32)

        while not self.exit_flag:
            agent_ids[0], state_id[0], states[0] = self.server.prediction_q.get()

            size = 1
            while size < Config.PREDICTION_BATCH_SIZE and not self.server.prediction_q.empty():
                agent_ids[size], state_id[size], states[size] = self.server.prediction_q.get()
                size += 1

            x__ = states[:size]
            p, v, train_func = self.server.model.predict_then_train(x__)
            batch = Batch(size, train_func)

            for i in range(size):
                self.server.state2batch[state_id[i]] = (batch, i)
                if agent_ids[i] < len(self.server.agents):
                    self.server.agents[agent_ids[i]].wait_q.put((p[i], v[i]))
