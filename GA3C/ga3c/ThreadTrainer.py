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
import sys, time

from Config import Config


class ThreadTrainer(Thread):
    def __init__(self, server, id):
        super(ThreadTrainer, self).__init__()
        self.setDaemon(True)

        self.id = id
        self.server = server
        self.exit_flag = False

    def run(self):
        if Config.LOCAL_QUEUE:
            data_queue = self.server.local_training_q
        else:
            data_queue = self.server.training_q
        total_time = 0
        collect_time = 0
        train_time = 0
        step = 0
        acc_batch_size = 0.0

        while not self.exit_flag:
            step += 1
            s0 = time.time()

            batch_size = 0
            while batch_size <= Config.TRAINING_MIN_BATCH_SIZE:
                x_, r_, a_ = data_queue.get() 
                if batch_size == 0:
                    x__ = x_; r__ = r_; a__ = a_
                else:
                    x__ = np.concatenate((x__, x_))
                    r__ = np.concatenate((r__, r_))
                    a__ = np.concatenate((a__, a_))
                batch_size += x_.shape[0]
            s1 = time.time()
            
            if Config.TRAIN_MODELS:
                self.server.train_model(x__, r__, a__, self.id)
            s2 = time.time()

            total_time += s2 - s0
            collect_time += s1 - s0
            train_time += s2 - s1
            acc_batch_size += batch_size
            if self.id == 0 and step % 1000 == 0:
                print("[train %d] collect: %.1f %.1f%%, train: %.1f %.1f%%, total: %.1f, batch: %d, local_q: %d, remote_q: %d" % 
                    (step, collect_time, collect_time / total_time * 100, 
                    train_time, train_time / total_time * 100,
                    total_time, acc_batch_size / step, self.server.local_training_q.qsize(), self.server.training_q.qsize()))
                sys.stdout.flush()
