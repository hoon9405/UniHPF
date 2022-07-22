import pickle
import struct
from collections import OrderedDict
from typing import Any, Dict, List, Mapping, Optional

import torch
import torch.distributed as dist

def get_rank(group):
    return dist.get_rank(group=group)

def get_world_size(group):
    if torch.distributed.is_initialized():
        return dist.get_world_size(group=group)
    else:
        return 1

def get_global_group():
    if torch.distributed.is_initialized():
        if not hasattr(get_global_group, "_global_group"):
            # ideally we could use torch.distributed.group.WORLD, but it seems
            # to cause random NCCL hangs in some cases
            get_global_group._global_group = dist.new_group()
        return get_global_group._global_group
    else:
        return None

def get_global_rank():
    if torch.distributed.is_initialized():
        return torch.distributed.get_rank()
    else:
        return 0

def get_global_world_size():
    if torch.distributed.is_initialized():
        return torch.distributed.get_world_size()
    else:
        return 1

def get_data_parallel_group():
    """Get the data parallel group the caller rank belongs to."""
    return get_global_group()

def get_data_parallel_rank():
    """Return my rank for the data parallel group."""
    return get_rank(get_data_parallel_group())

def get_data_parallel_world_size():
    """Return world size for the data parallel group."""
    return get_world_size(get_data_parallel_group())

def all_reduce(tensor, group, op='sum'):
    if op == 'sum':
        op = dist.ReduceOp.SUM
    elif op == 'max':
        op = dist.ReduceOp.MAX
    else:
        raise NotImplementedError
    dist.all_reduce(tensor, op=op, group=group)
    return tensor

def broadcast(tensor, src, group):
    dist.broadcast(tensor, src = src, group = group)

def all_gather(tensor, group, return_tensor = False):
    """Perform an all-gather operation."""
    world_size = get_world_size(group = group)
    rank = get_rank(group = group)
    tensor_list = [
        tensor if i == rank else torch.empty_like(tensor) for i in range(world_size)
    ]
    dist.all_gather(tensor_list, tensor, group = group)
    if return_tensor:
        return torch.stack(tensor_list, dim = 0)
    else:
        return tensor_list

def all_gather_list(data, group = None, max_size = 16384):
    """Gathers arbitrary data from all nodes into a list.

    Similar to :func:`~torch.distributed.all_gather` but for arbitrary Python
    data. Note that *data* must be picklable and any CUDA tensors will be moved
    to CPU and returned on CPU as well.

    Args:
        data (Any): data from the local worker to be gathered on other workers
        group: group of the collective
        max_size (int, optional): maximum size of the data to be gathered
            across workers
    """
    from utils import utils

    if group is None:
        group = get_global_group()
    rank = get_rank(group = group)
    world_size = get_world_size(group = group)

    buffer_size = max_size * world_size
    if (
        not hasattr(all_gather_list, "_buffer")
        or all_gather_list._buffer.numel() < buffer_size
    ):
        all_gather_list._buffer = torch.cuda.ByteTensor(buffer_size)
        all_gather_list._cpu_buffer = torch.ByteTensor(max_size).pin_memory()
    buffer = all_gather_list._buffer
    buffer.zero_()
    cpu_buffer = all_gather_list._cpu_buffer

    data = utils.move_to_cpu(data)
    enc = pickle.dumps(data)
    enc_size = len(enc)
    header_size = 4 # size of header that contains the length of the encoded data
    size = header_size + enc_size
    if size > max_size:
        raise ValueError(
            "encoded data size ({}) exceeds max_size ({})".format(size, max_size)
        )
    
    header = struct.pack(">I", enc_size)
    cpu_buffer[:size] = torch.ByteTensor(list(header + enc))
    start = rank * max_size
    buffer[start : start + size].copy_(cpu_buffer[:size])

    all_reduce(buffer, group = group)

    buffer = buffer.cpu()
    try:
        result = []
        for i in range(world_size):
            out_buffer = buffer[i * max_size : (i + 1) * max_size]
            (enc_size,) = struct.unpack(">I", bytes(out_buffer[:header_size].tolist()))
            if enc_size > 0:
                result.append(
                    pickle.loads(
                        bytes(out_buffer[header_size : header_size + enc_size].tolist())
                    )
                )
        return result
    except pickle.UnpicklingError:
        raise Exception(
            "Unable to unpickle data from other workers. all_gather_list requires all "
            "workers to enter the function together, so this error usually indicates "
            "that the workers have fallen out of sync somehow. Workers can fall out of "
            "sync if one of them runs out of memory, or if there are other conditions "
            "in your training script that can cause one worker to finish an epoch "
            "while other workers are still iterating over their portions of the data. "
            # "Try rerunning with --ddp-backend=legacy_ddp and see if that helps."
        )

def all_reduce_dict(data: Mapping[str, Any], device, group) -> Dict[str, Any]:
    """
    AllReduce a dictionary of values across workers. We separately
    reduce items that are already on the device and items on CPU for
    better performance.

    Args:
        data (Mapping[str, Any]): dictionary of data to all-reduce, but
            cannot be a nested dictionary
        device (torch.device): device for the reduction
        group: group of the collective
    """
    data_keys = list(data.keys())

    # We want to separately reduce items that are already on the
    # device and items on CPU for performance reasons.
    cpu_data = OrderedDict()
    device_data = OrderedDict()
    for k in data_keys:
        t = data[k]
        if not torch.is_tensor(t):
            cpu_data[k] = torch.tensor(t, dtype = torch.double)
        elif t.device.type != device.type:
            cpu_data[k] = t.to(dtype = torch.double)
        else:
            device_data[k] = t.to(dtype = torch.double)
    
    def _all_reduce_dict(data: OrderedDict):
        if len(data) == 0:
            return data
        
        buf = torch.cat([t.view(-1) for t in data.values()]).to(device = device)
        all_reduce(buf, group = group)
        split_buf = torch.split(buf, [t.numel() for t in data.values()])
        reduced_data = [t.view_as(orig) for t, orig in zip(split_buf, data.values())]
        return OrderedDict(zip(data.keys(), reduced_data))
    
    cpu_data = _all_reduce_dict(cpu_data)
    device_data = _all_reduce_dict(device_data)

    def get_from_stack(key):
        if key in cpu_data:
            return cpu_data[key]
        elif key in device_data:
            return device_data[key]
        raise KeyError
    
    return OrderedDict([(key, get_from_stack(key)) for key in data_keys])
