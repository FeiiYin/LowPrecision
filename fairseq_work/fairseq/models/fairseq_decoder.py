# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

import torch.nn as nn
import torch.nn.functional as F
import torch


def simulation_softmax(input):
    # type : (Tensor) -> Tensor
    output = torch.IntTensor(input.size())

    ratio = 1000

    dim = input.size()[0]
    len = input.size()[1]

    for now_dim in range(0, dim):
        for now_len in range(0, len):
            output[now_dim, now_len] = input[now_dim, now_len] * ratio

    for now_dim in range(0, dim):
        # now_dim_max = -10000
        # for now_len in range(0, len):
        #     if output[now_dim][now_len] > now_dim_max:
        #         now_dim_max = output[now_dim][now_len]
        now_dim_max = torch.max(output[now_dim])
        for now_len in range(0, len):
            # print(input[now_dim][now_len])
            output[now_dim, now_len] -= (now_dim_max + 2)  # this value could change
            # print(input[now_dim][now_len])

    output = output.float()
    output = output.mul(1.0 / ratio)

    # for now_dim in range(0, dim):
    #     for now_len in range(0, len):
    #         if output[now_dim, now_len] < -20:
    #             output[now_dim, now_len] = -20

    # return torch.from_numpy(output.cuda().data.cpu().numpy())

    if torch.cuda.is_available:
        # gpu return
        return output.cuda().data  # .numpy()
    else:
        # cpu return
        return output
    # failed try
    # return torch.from_numpy(output.cuda().data.cpu().numpy())
    # return torch.from_numpy(output).cuda().data.cpu().numpy()


def simulation_softmax_init(input):
    # type : (Tensor) -> Tensor
    output = torch.FloatTensor(input.size())

    ratio = 1000

    dim = input.size()[0]
    len = input.size()[1]

    for now_dim in range(0, dim):
        for now_len in range(0, len):
            output[now_dim, now_len] = input[now_dim, now_len] * ratio

    for now_dim in range(0, dim):
        # now_dim_max = -10000
        # for now_len in range(0, len):
        #     if output[now_dim][now_len] > now_dim_max:
        #         now_dim_max = output[now_dim][now_len]
        now_dim_max = torch.max(output[now_dim])
        now_dim_min = torch.min(output[now_dim])
        now_hash = 1.0 / (now_dim_max - now_dim_min)
        for now_len in range(0, len):
            output[now_dim, now_len] = (output[now_dim, now_len] - now_dim_min) * now_hash
        now_dim_max = torch.max(output[now_dim])
        for now_len in range(0, len):
            # print(input[now_dim][now_len])
            output[now_dim, now_len] -= (now_dim_max + 2)  # this value could change
            # print(input[now_dim][now_len])

    # output = output.float()
    output = output.mul(1.0 / ratio)

    # return torch.from_numpy(output.cuda().data.cpu().numpy())

    # if torch.cuda.is_available:
    #     # gpu return
    return output.cuda().data  # .numpy()
    # else:
        # cpu return
    # return output
    # failed try
    # return torch.from_numpy(output.cuda().data.cpu().numpy())
    # return torch.from_numpy(output).cuda().data.cpu().numpy()


def simulation_softmax_bad(input):
    # type : (Tensor) -> Tensor
    # output = torch.FloatTensor(input.size())

    # ratio = 1000

    dim = input.size()[0]
    len = input.size()[1]

    # for now_dim in range(0, dim):
    #     for now_len in range(0, len):
    #         output[now_dim, now_len] = input[now_dim, now_len]

    for now_dim in range(0, dim):
        # now_dim_max = -10000
        # for now_len in range(0, len):
        #     if output[now_dim][now_len] > now_dim_max:
        #         now_dim_max = output[now_dim][now_len]
        now_dim_max = torch.max(input[now_dim])
        for now_len in range(0, len):
            # print(input[now_dim][now_len])
            input[now_dim, now_len] -= (now_dim_max + 2)  # this value could change
            # print(input[now_dim][now_len])

    # output = output.float()
    # output = output.mul(1.0 / ratio)

    # for now_dim in range(0, dim):
    #     for now_len in range(0, len):
    #         if output[now_dim, now_len] < -20:
    #             output[now_dim, now_len] = -20

    # return torch.from_numpy(output.cuda().data.cpu().numpy())

    # if torch.cuda.is_available:
        # gpu return
        # return input.cuda().data  # .numpy()
    # else:
        # cpu return
    return input
    # failed try
    # return torch.from_numpy(output.cuda().data.cpu().numpy())
    # return torch.from_numpy(output).cuda().data.cpu().numpy()


class FairseqDecoder(nn.Module):
    """Base class for decoders."""

    def __init__(self, dictionary):
        super().__init__()
        self.dictionary = dictionary

    def forward(self, prev_output_tokens, encoder_out):
        """
        Args:
            prev_output_tokens (LongTensor): previous decoder outputs of shape
                `(batch, tgt_len)`, for input feeding/teacher forcing
            encoder_out (Tensor, optional): output from the encoder, used for
                encoder-side attention

        Returns:
            tuple:
                - the last decoder layer's output of shape
                  `(batch, tgt_len, vocab)`
                - the last decoder layer's attention weights of shape
                  `(batch, tgt_len, src_len)`
        """
        raise NotImplementedError


    def get_normalized_probs(self, net_output, log_probs, sample):
        """Get normalized probabilities (or log probs) from a net's output."""
        if hasattr(self, 'adaptive_softmax') and self.adaptive_softmax is not None:
            if sample is not None:
                assert 'target' in sample
                target = sample['target']
            else:
                target = None
            out = self.adaptive_softmax.get_log_prob(net_output[0], target=target)
            return out.exp_() if not log_probs else out

        logits = net_output[0].float()
        if log_probs:
            # print("i do not know 5\n\n")
            return simulation_softmax_init(logits)
            # return F.log_softmax(logits, dim=-1)
        else:
            return F.softmax(logits, dim=-1)

    def max_positions(self):
        """Maximum input length supported by the decoder."""
        return 1e6  # an arbitrary large number

    def upgrade_state_dict(self, state_dict):
        """Upgrade a (possibly old) state dict for new versions of fairseq."""
        return state_dict
