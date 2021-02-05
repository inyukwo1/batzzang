import torch
from typing import List, Tuple


def assert_dim(dim, tensor: torch.Tensor) -> None:
    tensor_dim = list(tensor.size())
    assert len(dim) == len(tensor_dim), "expected: {} real: {}".format(dim, tensor_dim)
    for expected, real in zip(dim, tensor_dim):
        if expected is not None:
            assert expected == real, "expected: {} real: {}".format(dim, tensor_dim)


def stack_sequential_tensor_with_mask(
    sequential_tensor_list: List[torch.Tensor],
) -> Tuple[torch.Tensor, torch.Tensor]:
    length_list = [len(tensor) for tensor in sequential_tensor_list]
    max_length = max(length_list)
    first_tensor = sequential_tensor_list[0]
    mask = torch.ones(
        (len(sequential_tensor_list), max_length),
        dtype=torch.bool,
        device=first_tensor.device,
    )
    if len(first_tensor.size()) == 1:
        stacked_tensor = torch.zeros(
            (len(sequential_tensor_list), max_length), dtype=first_tensor.dtype, device=first_tensor.device
        )
        for idx, tensor in enumerate(sequential_tensor_list):
            stacked_tensor[idx, :len(tensor)] = tensor
            mask[idx, : len(tensor)] = 0
    elif len(first_tensor.size()) == 2:
        _, embed_dim = list(first_tensor.size())

        stacked_tensor = torch.zeros(
            (len(sequential_tensor_list), max_length, embed_dim), dtype=first_tensor.dtype, device=first_tensor.device
        )
        for idx, tensor in enumerate(sequential_tensor_list):
            stacked_tensor[idx, :len(tensor)] = tensor
            mask[idx, : len(tensor)] = 0
    elif len(first_tensor.size()) == 3:
        _, _, embed_dim = list(first_tensor.size())
        max_length_dim0 = max_length
        max_length_dim1 = max([len(tensor[0]) for tensor in sequential_tensor_list])

        stacked_tensor = torch.zeros(
            (len(sequential_tensor_list), max_length_dim0, max_length_dim1, embed_dim), dtype=first_tensor.dtype, device=first_tensor.device
        )
        for idx, tensor_3d in enumerate(sequential_tensor_list):
            stacked_tensor[idx, :len(tensor_3d)] = tensor_3d
            mask[idx, : len(tensor_3d)] = 0
    else:
        raise NotImplementedError

    return stacked_tensor, mask
