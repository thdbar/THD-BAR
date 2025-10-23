
from einops import rearrange
import pickle
import os
from pathlib import Path
import random
import itertools

import torch
from torch.utils.data import Dataset, DistributedSampler, DataLoader, ConcatDataset
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.nn import functional as F

from model.standard_1020_chorder import gen_stair_scale_mask, ms_ch_dict
from model.standard_1020_chorder import remove_unused_ch

standard_1020 = [
    'AF9', 'AF7', 'F9', 'FP1', 'AF5', 'AF3', 'F7', 'F5', 'F3', \
    'FPZ', 'AF1', 'AFZ', 'AF2', 'F1', 'FZ', 'F2', \
    'FP2', 'AF4', 'AF6', 'AF8', 'AF10', 'F10', 'F4', 'F6', 'F8', \
    'FT9', 'FT7', 'A1', 'T9', 'T7', 'TP9', 'TP7', \
    'FC5', 'FC3', 'C5', 'C3', 'CP5', 'CP3', \
    'FC1', 'FCZ', 'FC2', 'C1', 'CZ', 'C2','CP1', 'CPZ', 'CP2', \
    'FC4', 'FC6', 'C4', 'C6','CP4', 'CP6', \
    'FT8', 'FT10', 'T8', 'T10', 'A2', 'TP8', 'TP10', \
    'P9', 'P7', 'P5', 'P3', 'PO9', 'PO7', 'PO5', 'PO3', \
    'P1', 'PZ', 'P2', 'PO1', 'POZ', 'PO2', \
    'P4', 'P6', 'P8', 'P10', 'PO4', 'PO6', 'PO8', 'PO10', \
    'O9', 'O1', 'I1', 'OZ', 'IZ', 'O2', 'O10', 'I2', 'pad'\
]  # add 'pad'


class PickleLoader(Dataset):
    def __init__(self, files, block_size=1024, sampling_rate=200, GPT_training=False):
        self.files = files
        self.default_rate = 200
        self.sampling_rate = sampling_rate
        self.block_size = block_size
        self.GPT_training = GPT_training

        self.standard_1020_dict = {name: idx for idx, name in enumerate(standard_1020)} 
        

    def __len__(self):
        return len(self.files)
    
    def std_norm(self, x):
        mean = torch.mean(x, dim=(0, 1), keepdim=True)
        std = torch.std(x, dim=(0, 1), keepdim=True)
        x = (x - mean) / std
        return x

    def get_chans(self, ch_names):
        return [self.standard_1020_dict[ch_name] for ch_name in ch_names]

    def __getitem__(self, index):
        sample = pickle.load(open(self.files[index], "rb"))
        data = sample["X"]
        ch_names = sample["ch_names"]
        data = torch.FloatTensor(data)
        time = data.size(1) // 200
        input_time = [i  for i in range(time) for _ in range(data.size(0))]  # [0, 0, 0, 1, 1, 1, 2, 2, 2, ...]

        data = rearrange(data, 'N (A T) -> (A N) T', T=200)
        
        X = torch.zeros((self.block_size, 200))
        X[:data.size(0)] = data

        if not self.GPT_training:
            Y_freq = torch.zeros((self.block_size, 100))
            Y_raw = torch.zeros((self.block_size, 200))
            
            x_fft = torch.fft.fft(data, dim=-1)
            amplitude = torch.abs(x_fft)
            amplitude = self.std_norm(amplitude)
            Y_freq[:data.size(0)] = amplitude[:, :100]
            Y_raw[:data.size(0)] = self.std_norm(data)
        
        # input_chans is the indices of the channels in the standard_1020 list
        # used for the spatial embedding
        input_chans = list(ch_names) * time
        input_chans.extend(['pad'] * (self.block_size - data.size(0)))
        input_chans = torch.IntTensor(self.get_chans(input_chans))
        # input_time is the mask for padding zeros
        # ensure that padding zeros are not used in the attention mechanism
        input_time.extend([0] * (self.block_size - data.size(0)))
        input_time = torch.IntTensor(input_time)

        # EEG data mask
        input_mask = torch.ones(self.block_size)
        input_mask[data.size(0):] = 0 

        # remove unused ch
        ms_bool, ms_ch = remove_unused_ch(ms_ch_dict, ch_names)
        ms_bool_bp = ms_bool.copy()
        ms_bool = list(itertools.chain(*ms_bool))
        num_ms_ch = ms_bool.count(True)


        if self.GPT_training:
            # gpt_mask is the attn mask for the GPT model
            mask_type = 'stair_scale' # stair, stair_scale
            if mask_type == 'stair':
                num_ms_ch = ms_bool_bp[-1].count(True)
                len_gpt_mask = num_ms_ch*time
                gpt_mask = torch.ones(len_gpt_mask, len_gpt_mask).view(1, len_gpt_mask, len_gpt_mask)
                for i in range(time):
                    if i == 0:
                        continue
                    gpt_mask[:, (i-1)*num_ms_ch:i*num_ms_ch, i*num_ms_ch:] = 0

            elif mask_type == 'stair_scale':
                gpt_mask = gen_stair_scale_mask(ms_ch, time)
                gpt_mask = torch.from_numpy(gpt_mask)
            num_chans = len(ch_names)
            ms_bool = torch.tensor(ms_bool, dtype=torch.bool)
            return X, input_chans, input_time, input_mask.bool(), gpt_mask.bool(), num_chans, data.size(0), ms_bool
        return X, Y_freq, Y_raw, input_chans, input_time, input_mask.bool()
    

def collate_fn(batch):
    X_eeg, input_chans, input_time, input_mask, gpt_mask, num_chans, num_tokens, ms_bool = zip(*batch)
    max_len = max(m.size(0) for m in gpt_mask)
    padded_masks = [
        F.pad(mask, (0, 0, 0, max_len - mask.size(0)), value=1)
        for mask in gpt_mask
    ]
    padded_masks = [
        F.pad(mask, (0, max_len - mask.size(1), 0, 0), value=0)
        for mask in padded_masks
    ]
    gpt_mask = torch.stack(padded_masks).unsqueeze(1)
    (X_eeg, input_chans, input_time, input_mask, ms_bool) = map(torch.stack, (X_eeg, input_chans, input_time, input_mask, ms_bool))
    (num_chans, num_tokens) = map(torch.tensor, (num_chans, num_tokens))

    return X_eeg, input_chans, input_time, input_mask, gpt_mask, num_chans, num_tokens, ms_bool