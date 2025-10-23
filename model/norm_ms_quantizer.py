
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as distributed
from einops import rearrange, repeat
from typing import List, Optional, Sequence, Tuple, Union
import numpy as np

from model.standard_1020_chorder import standard_1020, ms_ch_lst, ms_ch_dict, ch_nums


def l2norm(t):
    return F.normalize(t, p = 2, dim = -1)

def ema_inplace(moving_avg, new, decay):
    moving_avg.data.mul_(decay).add_(new, alpha = (1 - decay))

def sample_vectors(samples, num):
    num_samples, device = samples.shape[0], samples.device

    if num_samples >= num:
        indices = torch.randperm(num_samples, device = device)[:num]
    else:
        indices = torch.randint(0, num_samples, (num,), device = device)

    return samples[indices]



class Phi(nn.Conv1d):
    def __init__(self, embed_dim, quant_resi):
        ks = 3
        super().__init__(in_channels=embed_dim, out_channels=embed_dim, kernel_size=ks, stride=1, padding=ks//2)
        self.resi_ratio = abs(quant_resi)
    
    def forward(self, h_BChw):
        return h_BChw.mul(1-self.resi_ratio) + super().forward(h_BChw).mul_(self.resi_ratio)


class PhiShared(nn.Module):
    def __init__(self, qresi: Phi):
        super().__init__()
        self.qresi: Phi = qresi
    
    def __getitem__(self, _) -> Phi:
        return self.qresi


class PhiPartiallyShared(nn.Module):
    def __init__(self, qresi_ls: nn.ModuleList):
        super().__init__()
        self.qresi_ls = qresi_ls
        K = len(qresi_ls)
        self.ticks = np.linspace(1/3/K, 1-1/3/K, K) if K == 4 else np.linspace(1/2/K, 1-1/2/K, K)
    
    def __getitem__(self, at_from_0_to_1: float) -> Phi:
        return self.qresi_ls[np.argmin(np.abs(self.ticks - at_from_0_to_1)).item()]
    
    def extra_repr(self) -> str:
        return f'ticks={self.ticks}'


class PhiNonShared(nn.ModuleList):
    def __init__(self, qresi: List):
        super().__init__(qresi)
        # self.qresi = qresi
        K = len(qresi)
        self.ticks = np.linspace(1/3/K, 1-1/3/K, K) if K == 4 else np.linspace(1/2/K, 1-1/2/K, K)
    
    def __getitem__(self, at_from_0_to_1: float) -> Phi:
        return super().__getitem__(np.argmin(np.abs(self.ticks - at_from_0_to_1)).item())
    
    def extra_repr(self) -> str:
        return f'ticks={self.ticks}'



class NormMSVectorQuantizer(nn.Module):
    def __init__(self, n_embed, embedding_dim, beta, decay=0.99, eps=1e-5, 
                statistic_code_usage=True, kmeans_init=False, codebook_init_path=''):
        super().__init__()
        self.codebook_dim = embedding_dim
        self.num_tokens = n_embed
        self.beta = beta
        self.decay = decay
        self.standard_1020 = standard_1020
        self.ms_ch_lst: List = ms_ch_lst
        self.ms_ch_dict: dict = ms_ch_dict
        self.ch_nums: List = ch_nums

        # learnable = True if orthogonal_reg_weight > 0 else False
        self.embedding = nn.Embedding(self.num_tokens, self.codebook_dim)
        self.init_vocab(-1)

        share_quant_resi = 1
        quant_resi = -0.5
        if share_quant_resi == 0:   # non-shared: \phi_{1 to K} for K scales
            self.quant_resi = PhiNonShared([(Phi(embedding_dim, quant_resi) if abs(quant_resi) > 1e-6 else nn.Identity()) for _ in range(len(self.ch_nums))])
        elif share_quant_resi == 1: # fully shared: only a single \phi for K scales
            self.quant_resi = PhiShared(Phi(embedding_dim, quant_resi) if abs(quant_resi) > 1e-6 else nn.Identity())
        else:                       # partially shared: \phi_{1 to share_quant_resi} for K scales
            self.quant_resi = PhiPartiallyShared(nn.ModuleList([(Phi(embedding_dim, quant_resi) if abs(quant_resi) > 1e-6 else nn.Identity()) for _ in range(share_quant_resi)]))

        self.statistic_code_usage = statistic_code_usage
        if statistic_code_usage:
            self.register_buffer('cluster_size', torch.zeros(n_embed))
        if distributed.is_available() and distributed.is_initialized():
            print("ddp is enable, so use ddp_reduce to sync the statistic_code_usage for each gpu!")
            self.all_reduce_fn = distributed.all_reduce
        else:
            self.all_reduce_fn = nn.Identity()
    
    def init_vocab(self, eini: float):
        if eini > 0:
            nn.init.trunc_normal_(self.embedding.weight.data, std=eini)
        elif eini < 0:
            base = self.codebook_dim ** -0.5
            base /= 36
            self.embedding.weight.data.uniform_(-abs(eini) * base, abs(eini) * base)

    def eini(self, eini):
        if eini > 0: 
            nn.init.trunc_normal_(self.embedding.weight.data, std=eini)
        elif eini < 0: 
            self.embedding.weight.data.uniform_(-abs(eini) / self.num_tokens, abs(eini) / self.num_tokens)

    def extra_repr(self) -> str:
        return f'{self.ms_ch_lst}, beta={self.beta}  |  S={len(self.ms_ch_lst)}'

    def ch_trans(self, tensor, input_chans, input_time, num_standard_ch):
        """
        Rearranges the input tensor to align with the standard channel format, 
        ensuring time-step alignment while removing the 'pad' channel.

        Parameters:
        - tensor: Original tensor of shape (B, N, C)
        - input_chans: Channel index tensor of shape (B, N)
        - input_time: Time index tensor of shape (B, N)
        - num_standard_ch: Number of standard channels (excluding the 'pad' channel)

        Returns:
        - output_tensor: Transformed tensor of shape (num_times.sum(), num_standard_ch, C)
        - num_times: Number of time steps for each sample (B,)
        """
        B, N, C = tensor.shape

        num_times = torch.max(input_time, dim=1).values + 1  # Compute the number of time steps for each sample
        num_chs = torch.sum(input_time == 1, dim=1)  # Compute the number of channels used for each sample

        output_tensor = torch.zeros((num_times.sum(), num_standard_ch, C), dtype=tensor.dtype, device=tensor.device)  # Initialize the output tensor

        cur_time = 0  # Accumulated time index
        for i, (num_ch, num_time) in enumerate(zip(num_chs, num_times)):
            i_input_chans_id = input_chans[i, :num_ch]  # Select the first num_ch channels
            output_tensor[cur_time:cur_time + num_time, i_input_chans_id] = tensor[i, :num_ch * num_time].view(num_time, num_ch, C)  # Fill transformed data
            cur_time += num_time  # Update time index

        return output_tensor


    def ch_intrans(self, output_tensor, tensor, input_chans, input_time):
        """
        Restores the transformed tensor back to its original shape.

        Parameters:
        - output_tensor: Transformed tensor of shape (num_times.sum(), num_standard_ch, C)
        - tensor: Original tensor before transformation (B, N, C)
        - input_chans: Channel index tensor of shape (B, N)
        - input_time: Time index tensor of shape (B, N)

        Returns:
        - tensor_return: Restored tensor of shape (B, N, C)
        """
        B, N, C = tensor.shape
        
        num_times = torch.max(input_time, dim=1).values + 1  # Compute the number of time steps for each sample
        num_chs = torch.sum(input_time == 1, dim=1)  # Compute the number of channels used for each sample

        cur_time = 0  # Accumulated time index
        for i, (num_ch, num_time) in enumerate(zip(num_chs, num_times)):
            i_input_chans_id = input_chans[i, :num_ch]  # Select the first num_ch channels
            tensor[i, :num_ch * num_time] = output_tensor[cur_time:cur_time + num_time, i_input_chans_id].view(num_ch * num_time, C)  # Fill restored data
            cur_time += num_time  # Update time index

        return tensor


    def group_chans_by_mean(self, data, original_channels, final_groups):
        """
        Perform mean aggregation on data based on specified channel grouping.

        Parameters:
        data: torch.Tensor
            The raw data has the shape of (B, N, D), N: num_channels
        original_channels: list of str
            List of names for each channel corresponding to the data, with a length of num_channels
        final_groups: list of list of str
            Each element is a sub list containing several original channel names,
            Compressing the signals of these channels into one channel by averaging them.
            For example: [['Fp1', 'AF7', 'AF3', 'AFz'], ['F7', 'F5', 'FT9', ...], ...]

        return:
        final_data: torch.Tensor
            The new data obtained through mean aggregation has the shape of (B, G, D), Where G is the number of groups.
        """

        # Build a mapping dictionary from channel names to indexes
        chan_to_idx = {ch: i for i, ch in enumerate(original_channels)}
        L, N, D = data.shape

        final_data = torch.empty(L, len(final_groups), D, dtype=data.dtype, device=data.device)

        # Calculate the mean of each group and fill it in final_data
        for i, group in enumerate(final_groups):
            indices = [chan_to_idx[ch] for ch in group]
            indices = torch.tensor(indices, dtype=torch.long, device=data.device)
            group_data = data.index_select(dim=1, index=indices)
            group_mean = group_data.mean(dim=1)
            final_data[:, i, :] = group_mean
        return final_data

    def expand_chans_by_copy(self, data, cur_groups, expanded_groups):
        """
        Extend the data of (B, G, D) back to (B, N, D) based on the original grouping information.
        data: torch.Tensor,  The shape is (B, G, D)
            Data that has been grouped through channels and averaged
        cur_groups: list of list of str
            A channel grouping list used for mean aggregation, with each sub list containing several original channel names.
            G channels (i.e. G groups) corresponding to the data.
        expanded_groups: list of list of str
            The expanded single channel grouping list contains only one original channel name per sub list.
            The length is N, corresponding to the final number of channels to be expanded back.
            (N, 1)
        return:
            expanded_data: torch.Tensor,  The shape is (B, N, D)
        """

        B, G, D = data.shape
        N = len(expanded_groups)

        # Build the mapping from the original channel to the group index
        channel_to_group = {}
        for g_idx, group in enumerate(cur_groups):
            for ch in group:
                channel_to_group[ch] = g_idx
        # print(channel_to_group)
        # print(expanded_groups)

        expanded_data = torch.empty(B, N, D, dtype=data.dtype, device=data.device)

        # Traverse each single channel group, find the corresponding group index in the data, and then copy the data
        for n_idx, group in enumerate(expanded_groups):
            ch = group[0]  # only 1 channel in each group
            g_idx = channel_to_group[ch]
            expanded_data[:, n_idx, :] = data[:, g_idx, :]
        return expanded_data


    def forward(self, z, input_chans=None, input_time=None):
        # reshape z -> (batch, height, width, channel) and flatten
        z = l2norm(z) 

        z_LND = self.ch_trans(z, input_chans, input_time, len(self.standard_1020)).contiguous()  # L num_ch_standard_1020 codebook_dim

        L, N, D = z_LND.shape
        rest_LmsND = torch.zeros((L, sum(ch_nums), D), dtype=z_LND.dtype, device=z_LND.device)
        idx_LmsN = torch.zeros((L, sum(ch_nums)), dtype=torch.int64, device=z_LND.device)

        f_no_grad = z_LND.detach()
        f_rest = f_no_grad.clone()
        f_hat = torch.zeros_like(f_rest)

        with torch.amp.autocast('cuda', enabled=False):
            mean_vq_loss: torch.Tensor = 0.0
            SN = len(self.ms_ch_dict)
            for si in range(SN): # from small to large
                rest_hC = self.group_chans_by_mean(
                    f_rest, self.ms_ch_dict[f'ch_lst_scale0'][0], self.ms_ch_dict[f'ch_lst_scale{si}']
                ).reshape(-1, D) if (si != SN-1) else f_rest.reshape(-1, D)

                # find the nearest embedding
                d_no_grad = torch.sum(rest_hC.square(), dim=1, keepdim=True) + torch.sum(self.embedding.weight.data.square(), dim=1, keepdim=False)
                d_no_grad.addmm_(rest_hC, self.embedding.weight.data.T, alpha=-2, beta=1)  # (B*h*w, vocab_size)
                idx_h = torch.argmin(d_no_grad, dim=1)

                idx_LG = idx_h.view(L, -1)  # len(self.ms_ch_dict[f'ch_lst_scale{si}'])
                rest_LGC = rest_hC.view(L, -1, D)  # len(self.ms_ch_dict[f'ch_lst_scale{si}'])

                # record data and index for GPT
                if si == 0: 
                    rest_LmsND[:, :sum(ch_nums[:si+1])] = rest_LGC
                    idx_LmsN[:, :sum(ch_nums[:si+1])] = idx_LG
                else: 
                    rest_LmsND[:, sum(ch_nums[:si]):sum(ch_nums[:si+1])] = rest_LGC
                    idx_LmsN[:, sum(ch_nums[:si]):sum(ch_nums[:si+1])] = idx_LG

                h_LGD = self.embedding(idx_LG)
                h_LND = self.expand_chans_by_copy(
                    h_LGD, self.ms_ch_dict[f'ch_lst_scale{si}'], self.ms_ch_dict[f'ch_lst_scale{SN-1}']
                ).contiguous() if (si != SN-1) else self.embedding(idx_LG).contiguous()

                # Conv2D
                h_LND = self.quant_resi[si/(SN-1)](h_LND.permute(0,2,1)).permute(0,2,1)
                f_hat = f_hat + h_LND
                f_rest -= h_LND

                # compute loss for embedding
                mean_vq_loss += F.mse_loss(f_hat.data, z_LND).mul_(self.beta) + F.mse_loss(f_hat, f_no_grad)
            mean_vq_loss *= 1. / SN
            f_hat = (f_hat.data - f_no_grad).add_(z_LND)  # [L, c(10-20), 768]

        z_q = self.ch_intrans(f_hat, z, input_chans, input_time)  # [B, 1024, 768]        
        return z_q, mean_vq_loss, idx_LmsN, rest_LmsND
    
