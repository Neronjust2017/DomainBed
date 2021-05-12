# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

"""
A command launcher launches a list of commands on a cluster; implement your own
launcher to add support for your cluster. We've provided an example launcher
which runs all commands serially on the local machine.
"""

import subprocess
import time
import torch

def local_launcher(commands):
    """Launch commands serially on the local machine."""
    for cmd in commands:
        subprocess.call(cmd, shell=True)

def dummy_launcher(commands):
    """
    Doesn't run anything; instead, prints each command.
    Useful for testing.
    """
    for cmd in commands:
        print(f'Dummy launcher: {cmd}')

# def multi_gpu_launcher(commands):
#     """
#     Launch commands on the local machine, using all GPUs in parallel.
#     """
#     print('WARNING: using experimental multi_gpu_launcher.')
#     n_gpus = torch.cuda.device_count()
#     procs_by_gpu = [[] for i in range(n_gpus)]
#     procs_per_gpu = 3
#
#     while len(commands) > 0:
#         for gpu_idx in range(n_gpus):
#             proc = procs_by_gpu[gpu_idx]
#             # if (proc is None) or (proc.poll() is not None):
#             if len(proc) == 0:
#                 # Nothing is running on this GPU; launch a command.
#                 for i in range(procs_per_gpu):
#                     cmd = commands.pop(0)
#                     new_proc = subprocess.Popen(
#                         f'CUDA_VISIBLE_DEVICES={gpu_idx} {cmd}', shell=True)
#                     procs_by_gpu[gpu_idx].append(new_proc)
#                 break
#         time.sleep(1)
#
#     # Wait for the last few tasks to finish before returning
#     for procs in procs_by_gpu:
#         for p in procs:
#             if p is not None:
#                 p.wait()

def multi_gpu_launcher(commands):
    """
    Launch commands on the local machine, using all GPUs in parallel.
    """
    print('WARNING: using experimental multi_gpu_launcher.')
    n_gpus = torch.cuda.device_count()
    procs_per_gpu = 3
    procs_by_gpu = [None]* (n_gpus*procs_per_gpu)

    while len(commands) > 0:
        for idx in range(n_gpus*procs_per_gpu):
            proc = procs_by_gpu[idx]
            gpu_idx = int(idx / procs_per_gpu)
            if (proc is None) or (proc.poll() is not None):
                # Nothing is running on this GPU; launch a command.
                cmd = commands.pop(0)
                new_proc = subprocess.Popen(
                    f'CUDA_VISIBLE_DEVICES={gpu_idx} {cmd}', shell=True)
                procs_by_gpu[idx] = new_proc
                print(f'CUDA_VISIBLE_DEVICES={gpu_idx} {cmd}')
                break
        time.sleep(1)

    # Wait for the last few tasks to finish before returning
    for p in procs_by_gpu:
        if p is not None:
            p.wait()

REGISTRY = {
    'local': local_launcher,
    'dummy': dummy_launcher,
    'multi_gpu': multi_gpu_launcher
}

try:
    from domainbed import facebook
    facebook.register_command_launchers(REGISTRY)
except ImportError:
    pass
