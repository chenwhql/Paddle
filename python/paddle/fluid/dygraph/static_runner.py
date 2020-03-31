# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

import collections
import copy
import logging
import numpy as np
import os
import six

from . import base
from . import layers
from .. import core
from .. import executor
from .. import framework
from .. import backward

from ... import compat as cpt

# Set Log level
logging.getLogger().setLevel(logging.ERROR)

# DESIGN IDEA: Add an special operator, execute static program inside operator.
#
# Op's Inputs:
#   - the input variable of the user feed
#   - the necessary parameters of the network
# Op's Outputs:
#   - the output variable of fetch
# 
# This op receives a complete program desc, internally creates scope
# and executor, executes this program. Key points:
#
# 1. Data Sharing: 
#   The varBase of the dynamic graph is not in the scope, so before the op
#   executes the program internally, create persistent variables with the
#   same name as feed, parameters, and fetch in the scope, and share the
#   LoDTensor of the op input.
# 
# 2. Forward and Backward Separation:
#   Because the dynamic graph op performs the forward and backward separately,
#   the forward program is used as the execution object of the forward op,
#   and the reverse program is used as the execution object of the grad op.


class StaticModelRunner(layers.Layer):
    """
    A Dynamic graph Layer for loading inference program and related parameters,
    and then performing fine-tune training or inference.

    The loaded program and parameters are saved by `fluid.io.save_inference_model`.

    .. note::
        **1. Dynamic graph mode do not support LoDTensor. 
             All original static graph model's feed targets or parametars 
             that depend on LoD are temporarily unavailable.**
        **2. All saved inference model's feed targets need be given.**
        **3. The ``stop_gradient`` information is lost and can not be recovered.**
        **4. The parameter's ``trainable`` information is lost and can not be recovered.**
        **5. Double gradient model is not supported now.**
        **6. Now only supports loading models saved by `fluid.io.save_inference_model`.**

    Args:
        model_dir(str): The directory path where the model is saved.
        model_filename(str, optional): The file name of saved inference program. 
                                       If set to None, a default filename is
                                       :code:`__model__`.
                                       The default value is None.
        params_filename(str, optional): The file name of saved all related parameters.
                                        If set to None, parameters are saved
                                        in separate files. 
                                        The default value is None.

    Returns:
        Layer: A Layer object can run loaded program.

    Examples:
      .. code-block:: python

        import numpy as np
        import paddle.fluid as fluid

        BATCH_SIZE = 32
        BATCH_NUM = 20
        SAVE_DIRNAME = "fc.inference.model"

        def random_batch_reader():
            def _get_random_images_and_labels(image_shape, label_shape):
                image = np.random.random(size=image_shape).astype('float32')
                label = np.random.random(size=label_shape).astype('int64')
                return image, label

            def __reader__():
                for _ in range(BATCH_NUM):
                    batch_image, batch_label = _get_random_images_and_labels(
                        [BATCH_SIZE, 784], [BATCH_SIZE, 1])
                    yield batch_image, batch_label

            return __reader__

        def train_and_save_static_model(place):
            img = fluid.data(name='img', shape=[None, 784], dtype='float32')
            label = fluid.data(name='label', shape=[None, 1], dtype='int64')

            pred = fluid.layers.fc(input=img, size=10, act='softmax')

            loss = fluid.layers.cross_entropy(input=pred, label=label)
            avg_loss = fluid.layers.mean(loss)

            optimizer = fluid.optimizer.SGD(learning_rate=0.001)
            optimizer.minimize(avg_loss)

            exe = fluid.Executor(place)
            exe.run(fluid.default_startup_program())

            loader = fluid.io.DataLoader.from_generator(
                feed_list=[img, label], capacity=5, iterable=True)
            loader.set_batch_generator(random_batch_reader(), places=place)

            for data in loader():
                exe.run(
                    fluid.default_main_program(),
                    feed=data, 
                    fetch_list=[avg_loss])

            # save model by fluid.io.save_inference_model
            fluid.io.save_inference_model(
                SAVE_DIRNAME, ["img"], [pred], exe)


        # Step 1. train and save inference model in static graph mode
        place = fluid.CPUPlace()
        train_and_save_static_model(place)

        # Step 2. load inference model in dygraph and fine-tune
        with fluid.dygraph.guard(place):
            fc = fluid.dygraph.static_runner.StaticModelRunner(SAVE_DIRNAME)

            sgd = fluid.optimizer.SGD(learning_rate=0.001,
                                    parameter_list=fc.parameters())

            train_loader = fluid.io.DataLoader.from_generator(capacity=5)
            train_loader.set_batch_generator(
                random_batch_reader(), places=place)

            for data in train_loader():
                img = data[0]
                label = data[1]
                label.stop_gradient = True

                cost = fc(inputs=img)

                loss = fluid.layers.cross_entropy(cost, label)
                avg_loss = fluid.layers.mean(loss)

                avg_loss.backward()
                sgd.minimize(avg_loss)
    """

    def __init__(self, model_dir, model_filename=None, params_filename=None):
        super(StaticModelRunner, self).__init__()

        # Step 0. key variable definitions
        self._load_program_desc = None
        self._program_desc = None
        # the layer outputs var desc
        self._output_descs = []
        # input, output, params name list
        self._input_names = []
        self._output_names = []
        self._param_names = []

        # Step 1. load program desc from disk
        # the saved model hold feed, fetch & scale op, no need, can be remove
        self._load_program_desc = self._load_static_model(model_dir,
                                                          model_filename)

        # Step 2. set all `is_test` attributes to False
        self._change_is_test_status(False)

        # Step 3. load all parameters
        self._load_persisitable_dict(model_dir, params_filename)

        # Step 4. generate backwar program desc
        self._program_desc = self._append_backward_desc()

        # Step 5. recheck parameters stop gradients
        self._recheck_stop_gradients()

        # For Debug
        # DescParser.print_program_desc(self._program_desc)

    def train(self):
        framework._dygraph_tracer().train_mode()
        self._change_is_test_status(False)

    def eval(self):
        framework._dygraph_tracer().eval_mode()
        self._change_is_test_status(True)

    def forward(self, inputs):
        # Step 1. prepare inputs, outputs, attrs
        if not isinstance(inputs, (list, tuple)):
            inputs = [inputs]

        input_vars = []
        for i, value in enumerate(inputs):
            if not isinstance(value, (np.ndarray, core.VarBase)):
                raise TypeError(
                    "The type of inputs.value in StaticModelRunner.forward must be numpy array or Variable(VarBase), but received %s."
                    % type(value))
            # NOTE: In order to unify the API, firstly convert the input to VarBase
            if isinstance(value, np.ndarray):
                var = core.VarBase(
                    value=value,
                    name=self._input_names[i],
                    persistable=False,
                    place=framework._current_expected_place(),
                    zero_copy=True)
            else:
                var = value
                # NOTE: change desc var name, if the inputs user given are VarBase, 
                # they may have important name set by the user!
                if var.name != self._input_names[i]:
                    self._change_var_name_in_desc(self._input_names[i],
                                                  var.name)
                self._input_names[i] = var.name
            input_vars.append(var)

        params = []
        for param in self._parameters.values():
            params.append(param)

        output_vars = []
        for var_desc in self._output_descs:
            var = core.VarBase(var_desc.dtype(),
                               var_desc.shape(),
                               var_desc.name(), var_desc.type(), False)
            var.stop_gradient = False
            output_vars.append(var)

        # hold forward variables
        tmp_scope_vec = core.VarBase(core.VarDesc.VarType.FP32, [],
                                     "program_out_scope",
                                     core.VarDesc.VarType.STEP_SCOPES, True)
        inner_scope = core.Scope()
        tmp_scope_vec.value().set_scope(inner_scope)

        # Step 2. run prorgam by op
        framework._dygraph_tracer().trace_op(
            type='run_program',
            inputs={'X': input_vars,
                    'Params': params},
            outputs={'Out': output_vars,
                     'OutScope': tmp_scope_vec},
            attrs={
                'global_block': self._program_desc.block(0),
                'start_op_index': 0,
                'end_op_index': self._load_program_desc.block(0).op_size()
            })

        # Step 3. prepare output, keep same form with inputs
        outs = output_vars
        if len(output_vars) == 1:
            outs = output_vars[0]
        return outs

    def _load_static_model(self, model_dir, model_filename=None):
        # Step 1. dir and filename check
        load_dirname = os.path.normpath(model_dir)
        if not os.path.isdir(load_dirname):
            raise ValueError("There is no directory named '%s'" % load_dirname)

        if model_filename is not None:
            model_filename = os.path.basename(model_filename)
        else:
            model_filename = "__model__"
        model_filename = os.path.join(load_dirname, model_filename)

        # Step 2. parse program desc
        with open(model_filename, "rb") as f:
            program_desc_str = f.read()

        program_desc = core.ProgramDesc(program_desc_str)
        if not core._is_program_version_supported(program_desc._version()):
            raise ValueError("Unsupported program version: %d\n" %
                             program_desc._version())

        # Step 3. 
        # - remove feed, fetch and useless scale-1 op
        # - remove op_callstack attr
        ops_to_remove = []
        root_block = program_desc.block(0)
        for i in six.moves.range(root_block.op_size()):
            op = root_block.op(i)
            if op.type() == 'feed':
                ops_to_remove.append(i)
                feed_var_name = cpt.to_bytes(op.input('X')[0])
                root_block._remove_var(feed_var_name)
                self._input_names.append(cpt.to_bytes(op.output('Out')[0]))
            elif op.type() == 'scale' and op.output('Out')[0].startswith(
                    'save_infer_model/scale_'):
                ops_to_remove.append(i)
                out_var_name = cpt.to_bytes(op.output('Out')[0])
                root_block._remove_var(out_var_name)
                self._output_names.append(cpt.to_bytes(op.input('X')[0]))
                self._output_descs.append(
                    root_block.find_var(cpt.to_bytes(op.input('X')[0])))
            elif op.type() == 'fetch' and op.input('X')[0].startswith(
                    'save_infer_model/scale_'):
                ops_to_remove.append(i)
                fetch_var_name = cpt.to_bytes(op.output('Out')[0])
                root_block._remove_var(fetch_var_name)
            else:
                if op.has_attr("op_callstack"):
                    op.remove_attr("op_callstack")

        for op_idx in reversed(ops_to_remove):
            root_block._remove_op(op_idx, op_idx + 1)

        return program_desc

    def _append_backward_desc(self):
        with framework._static_graph_guard():
            assert self._load_program_desc is not None, "The StaticModelRunner not initialized properly."
            program_desc_copy = core.ProgramDesc(self._load_program_desc)

            # Step 1. prepare program and related var
            # NOTE: To reuse backward interfaces, build Program firstly.
            # Originally, there is no need to build a program, but need to almost
            # rewrite a series of methods for append_backward for program_desc. 
            # Therefore, in order to reuse the method of backward.py, build the program here.
            fwd_op_num = program_desc_copy.block(0).op_size()
            program = self._build_program_by_desc(program_desc_copy)

            # TODO: could the targets be in sub block?
            targets = []
            for out in self._output_descs:
                targets.append(program.global_block().var(out.name()))

            # Step 2. append backward
            backward.gradients(targets=targets, inputs=[])
            return program.desc

    def _load_persisitable_dict(self, model_dir, params_filename=None):
        load_dirname = os.path.normpath(model_dir)
        assert self._load_program_desc is not None, "The StaticModelRunner not initialized properly."

        persis_vars = self._get_persis_vars(self._load_program_desc)
        load_var_map = {}
        for each_var in persis_vars:
            orig_each_name = each_var.name()
            # append suffix
            self._append_loaded_suffix_to_param(each_var)
            # create output varbase
            new_var = framework.ParamBase(
                shape=each_var.shape(),
                dtype=each_var.dtype(),
                name=each_var.name(),
                type=each_var.type(),
                persistable=True)
            if params_filename is None:
                if not self._is_parameter(each_var):
                    continue
                # logging.info("persis var name %s" % each_var.name())
                framework._dygraph_tracer().trace_op(
                    type='load',
                    inputs={},
                    outputs={'Out': new_var},
                    attrs={
                        'file_path': os.path.join(load_dirname, orig_each_name)
                    })
                new_var.stop_gradient = False
                self.add_parameter(name=new_var.name, parameter=new_var)
                self._param_names.append(new_var.name)
            else:
                load_var_map[each_var.name()] = new_var

        if params_filename is not None:
            load_var_list = []
            for name in sorted(load_var_map.keys()):
                load_var_list.append(load_var_map[name])

            framework._dygraph_tracer().trace_op(
                type='load_combine',
                inputs={},
                outputs={'Out': load_var_list},
                attrs={
                    'file_path': os.path.join(load_dirname, params_filename)
                })

            for each_var in persis_vars:
                if not self._is_parameter(each_var):
                    continue
                param = load_var_map[each_var.name()]
                param.stop_gradient = False
                self.add_parameter(name=param.name, parameter=param)
                self._param_names.append(param.name)

    def _recheck_stop_gradients(self):
        assert self._program_desc is not None, "The StaticModelRunner not initialized properly."
        # NOTE: After loading the model, the stop_gradient information 
        # of the original variable is lost, but if a parameter does not
        # have a corresponding @GRAD variable in the backward program,
        # it can be said that it is also stop_gradient
        all_var_names = self._get_all_var_names(self._program_desc)
        for param_name in self._parameters:
            param_grad_name = param_name + core.grad_var_suffix()
            if param_grad_name not in all_var_names:
                logging.info("set %s stop gradient = True" % param_grad_name)
                self._parameters[param_name].stop_gradient = True

    def _get_all_var_names(self, program_desc):
        all_var_names = set()
        for i in six.moves.range(program_desc.num_blocks()):
            block = program_desc.block(i)
            for var in block.all_vars():
                logging.info(var.name())
                all_var_names.add(var.name())
        return all_var_names

    def _get_persis_vars(self, program_desc):
        persis_vars = []
        for i in six.moves.range(program_desc.num_blocks()):
            block = program_desc.block(i)
            persis_vars.extend(
                list(filter(self._is_persistable, block.all_vars())))
        return persis_vars

    def _build_program_by_desc(self, program_desc):
        with framework._static_graph_guard():
            prog = framework.Program()
            prog.desc = program_desc
            prog.blocks = [
                framework.Block(prog, i)
                for i in six.moves.range(prog.desc.num_blocks())
            ]
            prog._sync_with_cpp()
        return prog

    def _is_persistable(self, var_desc):
        if var_desc.type() == core.VarDesc.VarType.FEED_MINIBATCH or \
                var_desc.type() == core.VarDesc.VarType.FETCH_LIST or \
                var_desc.type() == core.VarDesc.VarType.READER or \
                var_desc.type() == core.VarDesc.VarType.RAW:
            return False
        return var_desc.persistable()

    def _is_parameter(self, persis_var_desc):
        assert self._load_program_desc is not None, "The StaticModelRunner not initialized properly."
        # 1. firstly, param should be input of op
        input_ops = []  # op can be repeated
        for block_idx in six.moves.range(self._load_program_desc.num_blocks()):
            block = self._load_program_desc.block(block_idx)
            for op_idx in six.moves.range(block.op_size()):
                op = block.op(op_idx)
                # NOTE: parameter is the input of a certain op
                if persis_var_desc.name() in op.input_arg_names():
                    input_ops.append(op)
        # 2. secondly, param should not be output of op or be same op's output
        for block_idx in six.moves.range(self._load_program_desc.num_blocks()):
            block = self._load_program_desc.block(block_idx)
            for op_idx in six.moves.range(block.op_size()):
                op = block.op(op_idx)
                if persis_var_desc.name() in op.output_arg_names():
                    # such as batch_norm_op
                    if op in input_ops:
                        continue
                    else:
                        return False
        return True

    def _change_is_test_status(self, is_test):
        # change all `is_test` attributes
        assert self._load_program_desc is not None, "The StaticModelRunner not initialized properly."
        for i in six.moves.range(self._load_program_desc.num_blocks()):
            block = self._load_program_desc.block(i)
            for j in six.moves.range(block.op_size()):
                op = block.op(j)
                if op.has_attr('is_test'):
                    op._set_attr('is_test', is_test)

    def _append_loaded_suffix(self, name):
        """
        Append grad suffix to the given variable name
        e.g. x ==> x@LOADED
        """
        suffix = core.loaded_var_suffix()
        name = cpt.to_text(name)
        if suffix not in name:
            name = name + suffix
        return name

    def _append_loaded_suffix_to_param(self, param_desc):
        old_name = param_desc.name()
        new_name = self._append_loaded_suffix(param_desc.name())
        param_desc.set_name(new_name)
        for block_idx in six.moves.range(self._load_program_desc.num_blocks()):
            block = self._load_program_desc.block(block_idx)
            for op_idx in six.moves.range(block.op_size()):
                op = block.op(op_idx)
                op._rename_input(old_name, new_name)
                op._rename_output(old_name, new_name)

    def _change_var_name_in_desc(self, orig_name, new_name):
        assert self._program_desc is not None, "The StaticModelRunner not initialized properly."
        # 1. change all var and grad var name
        # 2. change all related op input name
        # 3. change all related grad op output name
        orig_grad_name = cpt.to_text(orig_name) + core.grad_var_suffix()
        new_grad_name = new_name + core.grad_var_suffix()
        for i in six.moves.range(self._program_desc.num_blocks()):
            block = self._program_desc.block(i)
            # change var name
            for var in block.all_vars():
                if var.name() == orig_name:
                    var.set_name(new_name)
                elif var.name() == orig_grad_name:
                    var.set_name(new_grad_name)
            # change op input & output
            for j in six.moves.range(block.op_size()):
                op = block.op(j)
                op._rename_input(orig_name, new_name)
                op._rename_output(orig_grad_name, new_grad_name)


######### Debug Functions ##########


# NOTE: The StaticModelRunner is still unstable at this stage. 
# I hope to keep this debugging tool, and it can be deleted 
# after the function is stable.
class DescParser():
    @classmethod
    def print_program_desc(cls, prog, skip_op_callstack=True):
        block_idx = 0
        for i in six.moves.range(prog.num_blocks()):
            block = prog.block(i)
            cls.print_block_desc(block, block_idx, skip_op_callstack)
            block_idx += 1

    @classmethod
    def print_block_desc(cls, block, block_idx, skip_op_callstack=True):
        indent = 0

        print("{0}{1} // block {2}".format(
            cls._get_indent_space(indent), '{', block_idx))

        indent += 1
        # sort all vars
        all_vars = block.all_vars()
        for var in all_vars:
            print("{}{}".format(
                cls._get_indent_space(indent), cls.var_desc_to_code(var)))

        if len(all_vars) > 0:
            print("")

        for i in six.moves.range(block.op_size()):
            op = block.op(i)
            print("{}{}".format(
                cls._get_indent_space(indent),
                cls.op_desc_to_code(op, skip_op_callstack)))
        indent -= 1

        print("{0}{1}".format(cls._get_indent_space(indent), '}'))

    @classmethod
    def var_desc_to_code(cls, var):
        if var.type() == core.VarDesc.VarType.SELECTED_ROWS or var.type(
        ) == core.VarDesc.VarType.LOD_TENSOR:
            var_str = "{name} : fluid.{type}.shape{shape}.astype({dtype})".\
                format(i="{", e="}", name=var.name(), type=var.type(), shape=var.shape(), dtype=var.dtype())
        else:
            var_str = "{name} : fluid.{type})".\
                format(i="{", e="}", name=var.name(), type=var.type())

        var_str = "var " + var_str

        if var.persistable():
            var_str = "persist " + var_str

        return var_str

    @classmethod
    def op_desc_to_code(cls, op, skip_op_callstack=True):
        outputs_str = "{"
        for i in range(0, len(op.output_names())):
            outputs_str += "{name}=".format(name=op.output_names()[i])
            o = op.output(op.output_names()[i])
            outputs_str += "{value}".format(value=o)
            if i != len(op.output_names()) - 1:
                outputs_str += ", "
        outputs_str += "}"

        inputs_str = "{"
        for i in range(0, len(op.input_names())):
            inputs_str += "{name}=".format(name=op.input_names()[i])
            o = op.input(op.input_names()[i])
            inputs_str += "{value}".format(value=o)

            if i != len(op.input_names()) - 1:
                inputs_str += ", "
        inputs_str += "}"

        attr_names = sorted(op.attr_names())
        attrs_str = ""
        for i in range(0, len(attr_names)):
            name = attr_names[i]
            if skip_op_callstack and name == "op_callstack":
                continue

            attr_type = op.attr_type(name)
            if attr_type == core.AttrType.BLOCK:
                a = "{name} = block[{value}]".format(
                    name=name, type=attr_type, value=op._block_attr_id(name))
                attrs_str += a
                if i != len(attr_names) - 1:
                    attrs_str += ", "
                continue

            if attr_type == core.AttrType.BLOCKS:
                a = "{name} = blocks{value}".format(
                    name=name, type=attr_type, value=op._blocks_attr_ids(name))
                attrs_str += a
                if i != len(attr_names) - 1:
                    attrs_str += ", "
                continue

            a = "{name} = {value}".format(
                name=name, type=attr_type, value=op.attr(name))
            attrs_str += a
            if i != len(attr_names) - 1:
                attrs_str += ", "

        if outputs_str != "{}":
            op_str = "{outputs} = {op_type}(inputs={inputs}, {attrs})".\
                format(outputs = outputs_str, op_type=op.type(), inputs=inputs_str, attrs=attrs_str)
        else:
            op_str = "{op_type}(inputs={inputs}, {attrs})".\
                format(op_type=op.type(), inputs=inputs_str, attrs=attrs_str)
        return op_str

    @classmethod
    def _get_indent_space(cls, indent, space_num=4):
        ret = ""
        for i in range(0, indent * space_num):
            ret += " "

        return ret
