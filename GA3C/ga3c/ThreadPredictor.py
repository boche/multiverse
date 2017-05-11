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
import time, sys

import numpy as np

from Config import Config


class ThreadPredictor(Thread):
    def __init__(self, server, id):
        super(ThreadPredictor, self).__init__()
        self.setDaemon(True)

        self.id = id
        self.server = server
        self.exit_flag = False

    def run(self):
        data_queue = self.server.local_prediction_q
        total_time = 0
        collect_time = 0
        predict_time = 0
        step = 0
        acc_batch_size = 0.0

        ids = np.zeros(Config.PREDICTION_BATCH_SIZE, dtype=np.uint16)
        episode_ids = np.zeros(Config.PREDICTION_BATCH_SIZE, dtype = np.uint16)
        states = np.zeros(
            (Config.PREDICTION_BATCH_SIZE, Config.IMAGE_HEIGHT, Config.IMAGE_WIDTH, Config.STACKED_FRAMES),
            dtype=np.float32)

        while not self.exit_flag:
            step += 1
            s0 = time.time()
            size = 0
            states[0], ids[0], episode_ids[0] = data_queue.get()

            size = 1
            # while size < Config.PREDICTION_BATCH_SIZE and not data_queue.empty():
            while size < Config.PREDICTION_BATCH_SIZE:
                states[size], ids[size], episode_ids[size] = data_queue.get()
                size += 1
            s1 = time.time()

            batch = states[:size]
            p, v = self.server.model.predict_p_and_v(batch)
            s2 = time.time()

            for i in range(size):
                if ids[i] < len(self.server.agents):
                    self.server.agents[ids[i]].wait_q.put((episode_ids[i], p[i], v[i]))
            s3 = time.time()
            total_time += s3 - s0
            collect_time += s1 - s0
            predict_time += s2 - s1
            acc_batch_size += size
            if self.id == 0 and step % 1000 == 0:
                print("[predictor %d] collect: %.1f %.1f%%, predict: %.1f %.1f%%, total: %.1f, batch: %d, local_q: %d, remote_q: %d" % 
                    (step, collect_time, collect_time / total_time * 100, 
                    predict_time, predict_time / total_time * 100,
                    total_time, acc_batch_size / step, self.server.local_prediction_q.qsize(), self.server.prediction_q.qsize()))
                sys.stdout.flush()
