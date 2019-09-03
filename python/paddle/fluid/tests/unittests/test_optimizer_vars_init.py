#   Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import print_function

import unittest

import contextlib
import numpy
import paddle.fluid as fluid
import paddle.fluid.core as core
import paddle.fluid.optimizer as optimizer


class TestOptimizerVarsInit(unittest.TestCase):
    def SetUp(self):
        self.loss = None

    @contextlib.contextmanager
    def run_startup_before_minimize(self):
        with self.program_scope_guard():
            x = fluid.layers.data(name="X", shape=[10], dtype="float32")
            y = fluid.layers.data(name="Y", shape=[1], dtype="float32")
            y_ = fluid.layers.fc(input=x, size=1, act=None)
            cost = fluid.layers.square_error_cost(input=y_, label=y)
            self.loss = fluid.layers.mean(cost)

            place = core.CPUPlace()
            exe = fluid.Executor(place)

            exe.run(fluid.default_startup_program())

            yield

            input_x = numpy.random.random(size=(10, 10)).astype('float32')
            input_y = numpy.random.random(size=(10, 1)).astype('float32')
            loss_data, = exe.run(fluid.default_main_program(),
                                 feed={'X': input_x,
                                       'Y': input_y},
                                 fetch_list=[self.loss.name])

    @contextlib.contextmanager
    def run_startup_after_minimize(self):
        with self.program_scope_guard():
            x = fluid.layers.data(name="X", shape=[10], dtype="float32")
            y = fluid.layers.data(name="Y", shape=[1], dtype="float32")
            y_ = fluid.layers.fc(input=x, size=1, act=None)
            cost = fluid.layers.square_error_cost(input=y_, label=y)
            self.loss = fluid.layers.mean(cost)

            place = core.CPUPlace()
            exe = fluid.Executor(place)

            yield

            exe.run(fluid.default_startup_program())

            input_x = numpy.random.random(size=(10, 10)).astype('float32')
            input_y = numpy.random.random(size=(10, 1)).astype('float32')
            loss_data, = exe.run(fluid.default_main_program(),
                                 feed={'X': input_x,
                                       'Y': input_y},
                                 fetch_list=[self.loss.name])

            print("run success: ", loss_data[0])

    def test_sgd_optimizer(self):
        with self.run_startup_after_minimize():
            opt = optimizer.SGD(learning_rate=0.01)
            opt.minimize(self.loss)
        with self.run_startup_before_minimize():
            opt = optimizer.SGD(learning_rate=0.01)
            opt.minimize(self.loss)

    def test_momentum_optimizer(self):
        with self.run_startup_after_minimize():
            opt = optimizer.Momentum(learning_rate=0.01, momentum=0.2)
            opt.minimize(self.loss)
        with self.run_startup_before_minimize():
            opt = optimizer.Momentum(learning_rate=0.01, momentum=0.2)
            opt.minimize(self.loss)

    def test_dgc_momentum_optimizer(self):
        with self.run_startup_after_minimize():
            opt = optimizer.DGCMomentumOptimizer(
                learning_rate=0.01, momentum=0.2, rampup_begin_step=1252)
            opt.minimize(self.loss)
        with self.run_startup_before_minimize():
            opt = optimizer.DGCMomentumOptimizer(
                learning_rate=0.01, momentum=0.2, rampup_begin_step=1252)
            opt.minimize(self.loss)

    def test_lars_momentum_optimizer(self):
        with self.run_startup_after_minimize():
            opt = optimizer.LarsMomentumOptimizer(
                learning_rate=0.01, momentum=0.2)
            opt.minimize(self.loss)
        with self.run_startup_before_minimize():
            opt = optimizer.LarsMomentumOptimizer(
                learning_rate=0.01, momentum=0.2)
            opt.minimize(self.loss)

    def test_adagrad_optimizer(self):
        with self.run_startup_after_minimize():
            opt = optimizer.Adagrad(learning_rate=0.01, epsilon=1.0e-6)
            opt.minimize(self.loss)
        with self.run_startup_before_minimize():
            opt = optimizer.Adagrad(learning_rate=0.01, epsilon=1.0e-6)
            opt.minimize(self.loss)

    def test_adam_optimizer(self):
        with self.run_startup_after_minimize():
            opt = optimizer.Adam(learning_rate=0.01, beta1=0.9, beta2=0.999)
            opt.minimize(self.loss)
        with self.run_startup_before_minimize():
            opt = optimizer.Adam(learning_rate=0.01, beta1=0.9, beta2=0.999)
            opt.minimize(self.loss)

    def test_adamax_optimizer(self):
        with self.run_startup_after_minimize():
            opt = optimizer.Adamax(learning_rate=0.01, beta1=0.9, beta2=0.999)
            opt.minimize(self.loss)
        with self.run_startup_before_minimize():
            opt = optimizer.Adamax(learning_rate=0.01, beta1=0.9, beta2=0.999)
            opt.minimize(self.loss)

    def test_decaye_adagrad_optimizer(self):
        with self.run_startup_after_minimize():
            opt = optimizer.DecayedAdagrad(
                learning_rate=0.01, decay=0.95, epsilon=1.0e-6)
            opt.minimize(self.loss)
        with self.run_startup_before_minimize():
            opt = optimizer.DecayedAdagrad(
                learning_rate=0.01, decay=0.95, epsilon=1.0e-6)
            opt.minimize(self.loss)

    def test_ftrl_optimizer(self):
        with self.run_startup_after_minimize():
            opt = optimizer.Ftrl(
                learning_rate=0.01, l1=0.0, l2=0.0, lr_power=-0.5)
            opt.minimize(self.loss)
        with self.run_startup_before_minimize():
            opt = optimizer.Ftrl(
                learning_rate=0.01, l1=0.0, l2=0.0, lr_power=-0.5)
            opt.minimize(self.loss)

    def test_adadelta_optimizer(self):
        with self.run_startup_after_minimize():
            opt = optimizer.Adadelta(
                learning_rate=0.01, epsilon=1.0e-6, rho=0.95)
            opt.minimize(self.loss)
        with self.run_startup_before_minimize():
            opt = optimizer.Adadelta(
                learning_rate=0.01, epsilon=1.0e-6, rho=0.95)
            opt.minimize(self.loss)

    def test_rmsprop_optimizer(self):
        with self.run_startup_after_minimize():
            opt = optimizer.RMSProp(learning_rate=0.01)
            opt.minimize(self.loss)
        with self.run_startup_before_minimize():
            opt = optimizer.RMSProp(learning_rate=0.01)
            opt.minimize(self.loss)

    def test_lamb_optimizer(self):
        with self.run_startup_after_minimize():
            opt = optimizer.Lamb(learning_rate=0.01)
            opt.minimize(self.loss)
        with self.run_startup_before_minimize():
            opt = optimizer.Lamb(learning_rate=0.01)
            opt.minimize(self.loss)

    @contextlib.contextmanager
    def program_scope_guard(self):
        prog = fluid.Program()
        startup_prog = fluid.Program()
        scope = fluid.core.Scope()
        with fluid.scope_guard(scope):
            with fluid.program_guard(prog, startup_prog):
                yield


if __name__ == '__main__':
    unittest.main()
