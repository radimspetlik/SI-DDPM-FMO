# Copyright (c) 2022 Huawei Technologies Co., Ltd.
# Licensed under CC BY-NC-SA 4.0 (Attribution-NonCommercial-ShareAlike 4.0 International) (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode
#
# The code is released for academic research use only. For commercial use, please contact Huawei Technologies Co., Ltd.
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# This repository was forked from https://github.com/openai/guided-diffusion, which is under the MIT license

"""
Helpers for distributed training.
"""

import io
import os
import socket
import random

import blobfile as bf
import numpy as np
import torch
import torch as th
import torch.distributed as dist
from torch.multiprocessing import set_start_method


def setup_dist():
    """
    Setup a distributed process group.
    """
    if dist.is_initialized():
        return

    set_start_method('forkserver', force=True)

    global_rank = 0
    if torch.cuda.is_available():
        dist.init_process_group(backend='nccl',
                                init_method='env://')

        global_rank = dist.get_rank()
        torch.cuda.set_device(dev())

    torch.manual_seed(0 + global_rank)
    random.seed(0 + global_rank)
    np.random.seed(0 + global_rank)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(0 + global_rank)
        torch.cuda.manual_seed_all(0 + global_rank)

    return global_rank



def _find_free_port():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.bind(("", 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return s.getsockname()[1]
    finally:
        s.close()


def dev(device=None):
    """
    Get the device to use for torch.distributed.
    """
    if device is None:
        if th.cuda.is_available():
            return th.device(int(os.environ['LOCAL_RANK']))
        return th.device("cpu")
    return th.device(device)


def load_state_dict(path, backend=None, **kwargs):
    with bf.BlobFile(path, "rb") as f:
        data = f.read()
    return th.load(io.BytesIO(data), **kwargs)


def world_size():
    if torch.cuda.is_available():
        return dist.get_world_size()
    return 1


def global_rank():
    if torch.cuda.is_available():
        return dist.get_rank()
    return 0


def sync_params(params):
    """
    Synchronize a sequence of Tensors across ranks from rank 0.
    """
    if not torch.cuda.is_available():
        return

    for p in params:
        with th.no_grad():
            dist.broadcast(p, 0)
