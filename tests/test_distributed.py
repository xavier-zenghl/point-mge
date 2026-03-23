import torch
from utils.distributed import is_main_process, get_world_size, get_rank, reduce_tensor


def test_single_process_defaults():
    assert is_main_process() is True
    assert get_world_size() == 1
    assert get_rank() == 0


def test_reduce_tensor_single_process():
    t = torch.tensor([1.0, 2.0, 3.0])
    result = reduce_tensor(t)
    assert torch.allclose(result, t)
