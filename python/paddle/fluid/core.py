# Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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

import site
import sys
import os

core_suffix = 'so'
if os.name == 'nt':
    core_suffix = 'pyd'

has_avx_core = False
has_noavx_core = False

current_path = os.path.abspath(os.path.dirname(__file__))
if os.path.exists(current_path + os.sep + 'core_avx.' + core_suffix):
    has_avx_core = True

if os.path.exists(current_path + os.sep + 'core_noavx.' + core_suffix):
    has_noavx_core = True

try:
    if os.name == 'nt':
        third_lib_path = current_path + os.sep + '..' + os.sep + 'libs'
        os.environ['path'] = third_lib_path + ';' + os.environ['path']
        sys.path.insert(0, third_lib_path)

except ImportError as e:
    from .. import compat as cpt
    if os.name == 'nt':
        executable_path = os.path.abspath(os.path.dirname(sys.executable))
        raise ImportError(
            """NOTE: You may need to run \"set PATH=%s;%%PATH%%\"
        if you encounters \"DLL load failed\" errors. If you have python
        installed in other directory, replace \"%s\" with your own
        directory. The original error is: \n %s""" %
            (executable_path, executable_path, cpt.get_exception_message(e)))
    else:
        raise ImportError(
            """NOTE: You may need to run \"export LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH\"
        if you encounters \"libmkldnn.so not found\" errors. If you have python
        installed in other directory, replace \"/usr/local/lib\" with your own
        directory. The original error is: \n""" + cpt.get_exception_message(e))
except Exception as e:
    raise e


def avx_supported():
    """
    Whether current system(Linux, MacOS, Windows) is supported with AVX.
    """
    import platform
    from .. import compat as cpt
    sysstr = platform.system().lower()
    has_avx = False
    if sysstr == 'linux':
        try:
            has_avx = os.popen('cat /proc/cpuinfo | grep -i avx').read() != ''
        except Exception as e:
            sys.stderr.write('Can not get the AVX flag from /proc/cpuinfo.\n'
                             'The original error is: %s\n' %
                             cpt.get_exception_message(e))
        return has_avx
    elif sysstr == 'darwin':
        try:
            has_avx = os.popen(
                'sysctl machdep.cpu.features | grep -i avx').read() != ''
        except Exception as e:
            sys.stderr.write(
                'Can not get the AVX flag from machdep.cpu.features.\n'
                'The original error is: %s\n' % cpt.get_exception_message(e))
        if not has_avx:
            try:
                has_avx = os.popen(
                    'sysctl machdep.cpu.leaf7_features | grep -i avx').read(
                    ) != ''
            except Exception as e:
                sys.stderr.write(
                    'Can not get the AVX flag from machdep.cpu.leaf7_features.\n'
                    'The original error is: %s\n' %
                    cpt.get_exception_message(e))
        return has_avx
    elif sysstr == 'windows':
        import ctypes
        ONE_PAGE = ctypes.c_size_t(0x1000)

        def asm_func(code_str, restype=ctypes.c_uint32, argtypes=()):
            # Call the code_str as a function
            # Alloc 1 page to ensure the protection
            pfnVirtualAlloc = ctypes.windll.kernel32.VirtualAlloc
            pfnVirtualAlloc.restype = ctypes.c_void_p
            MEM_COMMIT = ctypes.c_ulong(0x1000)
            PAGE_READWRITE = ctypes.c_ulong(0x4)
            address = pfnVirtualAlloc(None, ONE_PAGE, MEM_COMMIT,
                                      PAGE_READWRITE)
            if not address:
                raise Exception("Failed to VirtualAlloc")

            # Copy the code into the memory segment
            memmove = ctypes.CFUNCTYPE(ctypes.c_void_p, ctypes.c_void_p,
                                       ctypes.c_void_p,
                                       ctypes.c_size_t)(ctypes._memmove_addr)
            if memmove(address, code_str, len(code_str)) < 0:
                raise Exception("Failed to memmove")

            # Enable execute permissions
            PAGE_EXECUTE = ctypes.c_ulong(0x10)
            pfnVirtualProtect = ctypes.windll.kernel32.VirtualProtect
            res = pfnVirtualProtect(
                ctypes.c_void_p(address), ONE_PAGE, PAGE_EXECUTE,
                ctypes.byref(ctypes.c_ulong(0)))
            if not res:
                raise Exception("Failed VirtualProtect")

            # Flush instruction cache
            pfnGetCurrentProcess = ctypes.windll.kernel32.GetCurrentProcess
            pfnGetCurrentProcess.restype = ctypes.c_void_p
            prochandle = ctypes.c_void_p(pfnGetCurrentProcess())
            res = ctypes.windll.kernel32.FlushInstructionCache(
                prochandle, ctypes.c_void_p(address), ONE_PAGE)
            if not res:
                raise Exception("Failed FlushInstructionCache")

            # Cast the memory to function
            functype = ctypes.CFUNCTYPE(restype, *argtypes)
            func = functype(address)
            return func, address

        # http://en.wikipedia.org/wiki/CPUID#EAX.3D1:_Processor_Info_and_Feature_Bits
        # mov eax,0x1; cpuid; mov cx, ax; ret
        code_str = b"\xB8\x01\x00\x00\x00\x0f\xa2\x89\xC8\xC3"
        avx_bit = 28
        retval = 0
        try:
            # Convert the code_str into a function that returns uint
            func, address = asm_func(code_str)
            retval = func()
            ctypes.windll.kernel32.VirtualFree(
                ctypes.c_void_p(address), ctypes.c_size_t(0), ONE_PAGE)
        except Exception as e:
            sys.stderr.write('Failed getting the AVX flag on Windows.\n'
                             'The original error is: %s\n' %
                             cpt.get_exception_message(e))
        return (retval & (1 << avx_bit)) > 0
    else:
        sys.stderr.write('Do not get AVX flag on %s\n' % sysstr)
        return False


load_noavx = False

if avx_supported():
    try:
        from .core_avx import *
        from .core_avx import __doc__, __file__, __name__, __package__
        from .core_avx import __unittest_throw_exception__
        from .core_avx import _append_python_callable_object_and_return_id
        from .core_avx import _cleanup, _Scope
        from .core_avx import _get_use_default_grad_op_desc_maker_ops
        from .core_avx import _get_all_register_op_kernels
        from .core_avx import _is_program_version_supported
        from .core_avx import _set_eager_deletion_mode
        from .core_avx import _set_fuse_parameter_group_size
        from .core_avx import _set_fuse_parameter_memory_size
        from .core_avx import _is_dygraph_debug_enabled
        from .core_avx import _dygraph_debug_level
        from .core_avx import _switch_tracer
        from .core_avx import _set_paddle_lib_path
        from .core_avx import _save_static_dict
        from .core_avx import _load_static_dict
        from .core_avx import _save_dygraph_dict
        from .core_avx import _load_dygraph_dict
        from .core_avx import _create_loaded_parameter
        if sys.platform != 'win32':
            from .core_avx import _set_process_pid
            from .core_avx import _erase_process_pid
            from .core_avx import _set_process_signal_handler
            from .core_avx import _throw_error_if_process_failed
            from .core_avx import _convert_to_tensor_list
            from .core_avx import _mmap_fd_clear
    except Exception as e:
        if has_avx_core:
            raise e
        else:
            from .. import compat as cpt
            sys.stderr.write(
                'WARNING: Do not have avx core. You may not build with AVX, '
                'but AVX is supported on local machine.\n You could build paddle '
                'WITH_AVX=ON to get better performance.\n'
                'The original error is: %s\n' % cpt.get_exception_message(e))
            load_noavx = True
else:
    load_noavx = True

if load_noavx:
    try:
        from .core_noavx import *
        from .core_noavx import __doc__, __file__, __name__, __package__
        from .core_noavx import __unittest_throw_exception__
        from .core_noavx import _append_python_callable_object_and_return_id
        from .core_noavx import _cleanup, _Scope
        from .core_noavx import _get_use_default_grad_op_desc_maker_ops
        from .core_noavx import _get_all_register_op_kernels
        from .core_noavx import _is_program_version_supported
        from .core_noavx import _set_eager_deletion_mode
        from .core_noavx import _set_fuse_parameter_group_size
        from .core_noavx import _set_fuse_parameter_memory_size
        from .core_noavx import _is_dygraph_debug_enabled
        from .core_noavx import _dygraph_debug_level
        from .core_noavx import _switch_tracer
        from .core_noavx import _set_paddle_lib_path
        from .core_noavx import _save_static_dict
        from .core_noavx import _load_static_dict
        from .core_noavx import _save_dygraph_dict
        from .core_noavx import _load_dygraph_dict
        from .core_noavx import _create_loaded_parameter
        if sys.platform != 'win32':
            from .core_noavx import _set_process_pid
            from .core_noavx import _erase_process_pid
            from .core_noavx import _set_process_signal_handler
            from .core_noavx import _throw_error_if_process_failed
            from .core_noavx import _convert_to_tensor_list
            from .core_noavx import _mmap_fd_clear
    except Exception as e:
        if has_noavx_core:
            sys.stderr.write(
                'Error: Can not import noavx core while this file exists ' +
                current_path + os.sep + 'core_noavx.' + core_suffix + '\n')
        raise e


# set paddle lib path
def set_paddle_lib_path():
    site_dirs = site.getsitepackages() if hasattr(
        site,
        'getsitepackages') else [x for x in sys.path if 'site-packages' in x]
    for site_dir in site_dirs:
        lib_dir = os.path.sep.join([site_dir, 'paddle', 'libs'])
        if os.path.exists(lib_dir):
            _set_paddle_lib_path(lib_dir)
            return
    if hasattr(site, 'USER_SITE'):
        lib_dir = os.path.sep.join([site.USER_SITE, 'paddle', 'libs'])
        if os.path.exists(lib_dir):
            _set_paddle_lib_path(lib_dir)


set_paddle_lib_path()
