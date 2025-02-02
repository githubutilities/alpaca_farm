# Copyright 2023 The Alpaca Team
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Sequence, Union

import torch
import torch.nn.functional as F

from . import utils
from .types import Tensor


def batch_select(input: Tensor, index: Tensor):
    """Select elements from a batched tensor with a batched index tensor.

    Example:
        input = torch.tensor([
            [0, 1, 2],
            [3, 0, 9],
            [6, 7, 8],
        ])
        index = torch.tensor([[0, 1], [1, 0], [0, 0]])
        batch_select(input, index) = tensor([
            [0, 1],
            [0, 3],
            [6, 6]
        ])
    """
    dummy_index = torch.arange(input.size(0), device=input.device).unsqueeze(-1)
    return input[dummy_index, index]


def pad_sequence_from_left(
    sequences: Sequence[Tensor],
    batch_first: bool = False,
    padding_value: float = 0.0,
):
    """Mirror of `torch.nn.utils.rnn.pad_sequence`, but pad from left."""
    sequences = tuple(sequence.flip(0) for sequence in sequences)
    padded_sequence = torch._C._nn.pad_sequence(sequences, batch_first, padding_value)  # noqa
    padded_sequence = padded_sequence.flip(int(batch_first))
    return padded_sequence


def compute_logprobs(logits: Tensor, labels: Tensor, ignore_index: int) -> Tensor:
    """Compute per-token logprobs, zeroing out places with ignore_index (padding)."""
    return -F.cross_entropy(logits.permute(0, 2, 1), labels, reduction="none", ignore_index=ignore_index)

def mean_std_with_mask(x, mask_id):
    x_mask = x.not_equal(mask_id) # [1, 1, 0, 0]
    x_mean = (x * x_mask).sum() / x_mask.sum()
    x_pow = torch.pow( (x - x_mean) , 2 )
    print('x_pow', x_pow)
    print('x', x * x_mask)
    x_pow_mean = (x_pow * x_mask).sum() / x_mask.sum()
    return x_mean, torch.sqrt(x_pow_mean)

def whiten(values: Tensor, shift_mean=True, epsilon=1e-8, max_len=0) -> Tensor:
    if values.size(0) < 8:
        print('values', values.shape, values.device)
        if max_len > 0:
            pad_value = 100.0
            values_pad = pad(values, [values.shape[0], max_len], value=pad_value, left=False)
            tensor_list = [torch.zeros_like(values_pad, device=values_pad.device) for _ in range(torch.distributed.get_world_size())]
            #deepspeed.comm.all_gather(tensor_list, values_pad, async_op=True)
            torch.distributed.all_gather(tensor_list, values_pad)
            values_all = torch.cat(tensor_list, 0)
            mean, std = mean_std_with_mask(values_all, pad_value)
            whitened = (values - mean) / (std + epsilon)
            print('values_all', values_all, values_all.shape)
            print('whitened', whitened)
        else:
            tensor_list = [torch.zeros_like(values, device=values.device) for _ in range(torch.distributed.get_world_size())]
            torch.distributed.all_gather(tensor_list, values)
            values_all = torch.cat(tensor_list, 0)
            mean, std = values_all.mean(), values_all.std(unbiased=False)  # noqa
            whitened = (values - mean) / (std + epsilon)
        if not shift_mean:
            whitened = whitened + mean
        return whitened
    assert values.size(0) >= 8, f"Internal error: Minibatch size {values.size(0)} is insufficient for whitening."
    mean, std = values.mean(), values.std(unbiased=False)  # noqa
    whitened = (values - mean) / (std + epsilon)
    if not shift_mean:
        whitened = whitened + mean
    return whitened


def pad(inputs: Tensor, target_size: Union[torch.Size, Sequence[int]], value=0.0, left=True):
    current_size = inputs.size()
    diffs = tuple(ti - ci for ti, ci in utils.zip_(target_size, current_size))
    pad_params = []
    for diff in diffs:
        pad_params = ([diff, 0] if left else [0, diff]) + pad_params
    res = F.pad(inputs, pad=pad_params, value=value)
    return res


def left_pad(inputs: Tensor, target_size: Union[torch.Size, Sequence[int]], value=0.0):
    return pad(inputs=inputs, target_size=target_size, value=value, left=True)


def right_pad(inputs: Tensor, target_size: Union[torch.Size, Sequence[int]], value=0.0):
    return pad(inputs=inputs, target_size=target_size, value=value, left=False)
