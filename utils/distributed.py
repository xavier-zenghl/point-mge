import torch
import torch.distributed as dist


def is_dist_initialized() -> bool:
    return dist.is_available() and dist.is_initialized()

def get_rank() -> int:
    if not is_dist_initialized():
        return 0
    return dist.get_rank()

def get_world_size() -> int:
    if not is_dist_initialized():
        return 1
    return dist.get_world_size()

def is_main_process() -> bool:
    return get_rank() == 0

def reduce_tensor(tensor: torch.Tensor, op=dist.ReduceOp.SUM) -> torch.Tensor:
    if not is_dist_initialized():
        return tensor
    rt = tensor.clone()
    dist.all_reduce(rt, op=op)
    rt /= get_world_size()
    return rt

def setup_distributed():
    if not dist.is_available():
        return
    dist.init_process_group(backend="nccl")
    local_rank = get_rank()
    torch.cuda.set_device(local_rank)

def cleanup_distributed():
    if is_dist_initialized():
        dist.destroy_process_group()
