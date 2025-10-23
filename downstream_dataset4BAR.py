
import itertools
from torch.utils.data import Dataset
from pathlib import Path
import h5py
import bisect
import torch
from einops import rearrange
import tiktoken
import numpy as np
import pickle
import os

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
standard_1020_change = {
    'CB1': 'PO7',
    'CB2': 'PO8',
    'T1': 'T9',
    'T2': 'T10',
    'T3': 'T7',
    'T4': 'T8',
    'T5': 'T9',
    'T6': 'T10',
    'M1': 'TP9',
    'M2': 'TP10',
}

def get_chans(ch_names):
    chans = []
    for ch_name in ch_names:
        chans.append(standard_1020.index(ch_name))
    return chans

def get_gpt_eeg_mask(len_gpt_mask, ms_ch, num_ms_ch, time):
    # gpt_mask is the attn mask for the GPT model
    mask_type = 'stair_scale' # stair, stair_scale
    if mask_type == 'stair':
        gpt_mask = torch.ones(len_gpt_mask, len_gpt_mask).view(1, len_gpt_mask, len_gpt_mask)
        for i in range(time):
            if i == 0: continue
            gpt_mask[:, (i-1)*num_ms_ch:i*num_ms_ch, i*num_ms_ch:] = 0
    elif mask_type == 'stair_scale':
        gpt_eeg_mask = gen_stair_scale_mask(ms_ch, time)
        gpt_eeg_mask = torch.from_numpy(gpt_eeg_mask).unsqueeze(0)
    return gpt_eeg_mask


class SEEDLoader4BAR(Dataset):
    # happy: 'H'
    # sad: 'S'
    # neutral: 'N'
    def __init__(self, root, files, sampling_rate=200, eeg_max_len=-1, text_max_len=-1, is_instruct=False, is_val=False):
        self.root = root
        self.files = files
        self.default_rate = 200
        self.sampling_rate = sampling_rate
        self.is_instruct = is_instruct
        self.is_val = is_val
        self.eeg_max_len = eeg_max_len
        self.text_max_len = text_max_len
        self.__seed_label = {
            'H': 0,
            'N': 1,
            'S': 2
        }
        ch_names = ['FP1', 'AF3', 'F7', 'F5', 'F3', 'FPZ', 'F1', 'FZ', 'F2', 'FP2', 'AF4', 'F4', 'F6', 'F8', 'FT7', 'T7', 'TP7', 'FC5', 'FC3', 'C5', 'C3', 'CP5', 'CP3', 'FC1', 'FCZ', 'FC2', 'C1', 'CZ', 'C2', 'CP1', 'CPZ', 'CP2', 'FC4', 'FC6', 'C4', 'C6', 'CP4', 'CP6', 'FT8', 'T8', 'TP8', 'P7', 'P5', 'P3', 'PO7', 'PO5', 'PO3', 'P1', 'PZ', 'P2', 'POZ', 'P4', 'P6', 'P8', 'PO4', 'PO6', 'PO8', 'O1', 'OZ', 'O2']
        self.ch_names = [name.split(' ')[-1].split('-')[0] for name in ch_names] 
        self.ch_names = [standard_1020_change.get(name, name) for name in self.ch_names] 
        self.ch_names = [name for name in standard_1020 if name in self.ch_names] 

        if is_instruct:
            enc = tiktoken.get_encoding("gpt2")
            encode = lambda s: enc.encode(s, allowed_special={"<|endoftext|>"})
            # 50257 for [SEP]
            self.text = {
                0: torch.IntTensor([50257] + encode('Question: Which emotion type does this EEG segment belong to? Answer: Positive <|endoftext|>')),
                2: torch.IntTensor([50257] + encode('Question: Which emotion type does this EEG segment belong to? Answer: Negative <|endoftext|>')),
                1: torch.IntTensor([50257] + encode('Question: Which emotion type does this EEG segment belong to? Answer: Neutral <|endoftext|>'))
            }
            self.prompt = torch.IntTensor([50257] + encode('Question: Which emotion type does this EEG segment belong to? Answer:'))

    def __len__(self):
        return len(self.files)

    def std_norm(self, x):
            mean = torch.mean(x, dim=(0, 1), keepdim=True)
            std = torch.std(x, dim=(0, 1), keepdim=True)
            x = (x - mean) / std
            return x

    def __getitem__(self, index):
        sample = pickle.load(open(os.path.join(self.root, self.files[index]), "rb"))
        X = sample["X"]
        Y = sample["y"]
        Y = self.__seed_label[Y]
        current_ch_names = sample["ch_names"]
        target_ch_names = self.ch_names
        if current_ch_names != target_ch_names:
            aligned_X = np.zeros((len(target_ch_names), X.shape[1]), dtype=X.dtype)
            ch_name_to_idx = {name: idx for idx, name in enumerate(target_ch_names)}
            for i, ch_name in enumerate(current_ch_names):
                if ch_name in ch_name_to_idx:
                    target_idx = ch_name_to_idx[ch_name]
                    aligned_X[target_idx] = X[i]
            X = aligned_X  
        data = torch.FloatTensor(X / 100)
        time = data.size(1) // 200
        input_time = [i  for i in range(time) for _ in range(data.size(0))]
        data = rearrange(data, 'N (A T) -> (A N) T', T=200)        
        input_chans = list(self.ch_names) * time

        if not self.is_instruct:
            input_chans = torch.IntTensor(get_chans(input_chans))
            input_time = torch.IntTensor(input_time)

            gpt_mask = torch.tril(torch.ones(data.size(0), data.size(0))).view(1, data.size(0), data.size(0))
            num_chans = len(self.ch_names)
            for i in range(time):
                gpt_mask[:, i * num_chans:(i + 1) * num_chans,  i * num_chans:(i + 1) * num_chans] = 1
            return data, Y, input_chans, input_time, gpt_mask.bool()
        
        if self.is_val:
            text = self.prompt
        else:
            text = self.text[int(Y)]
            # pad text to text_max_len
            valid_text_len = text.size(0)
            if self.text_max_len > valid_text_len:
                text_pad = torch.full((self.text_max_len,), fill_value=50256)
                text_pad[:valid_text_len] = text
                text = text_pad

        # pad eeg to eeg_max_len
        valid_eeg_len = data.size(0)
        if self.eeg_max_len > data.size(0):
            X_eeg = torch.zeros((self.eeg_max_len, 200))
            X_eeg[:data.size(0)] = data
            eeg_mask = torch.ones(self.eeg_max_len)
            eeg_mask[valid_eeg_len:] = 0

            input_chans.extend(['pad'] * (self.eeg_max_len - data.size(0)))
            input_time.extend([0] * (self.eeg_max_len - data.size(0)))
        else:
            X_eeg = data
            eeg_mask = torch.ones(data.size(0))

        input_chans = torch.IntTensor(get_chans(input_chans))
        input_time = torch.IntTensor(input_time)

        # remove unused ch
        ms_bool, ms_ch = remove_unused_ch(ms_ch_dict, self.ch_names) 
        ms_bool = list(itertools.chain(*ms_bool)) 
        num_ms_ch = ms_bool.count(True) 

        len_eeg_gpt_mask = num_ms_ch*time
        num_tokens = len_eeg_gpt_mask + text.size(0) 
        gpt_mask = torch.tril(torch.ones(num_tokens, num_tokens)).view(1, num_tokens, num_tokens)
        # gpt eeg mask
        gpt_eeg_mask = get_gpt_eeg_mask(len_eeg_gpt_mask, ms_ch, num_ms_ch, time)
        gpt_mask[:, :len_eeg_gpt_mask, :len_eeg_gpt_mask] = gpt_eeg_mask
        ms_bool = torch.tensor(ms_bool, dtype=torch.bool)
        
        if self.is_val:
            return X_eeg, text, Y, input_chans, input_time, eeg_mask.bool(), gpt_mask.bool(), ms_bool, len_eeg_gpt_mask
        
        Y_text = torch.full_like(text, fill_value=-1)
        prompt_len = self.prompt.size(0)
        Y_text[prompt_len - 1:valid_text_len - 1] = text[prompt_len:valid_text_len]
        return X_eeg, text, Y_text, input_chans, input_time, eeg_mask.bool(), gpt_mask.bool(), ms_bool, len_eeg_gpt_mask


class TUABLoader4BAR(Dataset):
    # abnormal: 1
    # normal: 0
    def __init__(self, root, files, sampling_rate=200, eeg_max_len=-1, text_max_len=-1, is_instruct=False, is_val=False):
        self.root = root
        self.files = files
        self.default_rate = 200
        self.sampling_rate = sampling_rate
        self.is_instruct = is_instruct
        self.is_val = is_val
        self.eeg_max_len = eeg_max_len
        self.text_max_len = text_max_len

        ch_names = ['EEG FP1', 'EEG FP2-REF', 'EEG F3-REF', 'EEG F4-REF', 'EEG C3-REF', 'EEG C4-REF', 'EEG P3-REF', 'EEG P4-REF', 'EEG O1-REF', 'EEG O2-REF', 'EEG F7-REF', \
                    'EEG F8-REF', 'EEG T3-REF', 'EEG T4-REF', 'EEG T5-REF', 'EEG T6-REF', 'EEG A1-REF', 'EEG A2-REF', 'EEG FZ-REF', 'EEG CZ-REF', 'EEG PZ-REF', 'EEG T1-REF', 'EEG T2-REF']
        self.ch_names = [name.split(' ')[-1].split('-')[0] for name in ch_names] 
        self.ch_names = [standard_1020_change.get(name, name) for name in self.ch_names] 
        self.ch_names = [name for name in standard_1020 if name in self.ch_names] 

        if is_instruct:
            enc = tiktoken.get_encoding("gpt2")
            encode = lambda s: enc.encode(s, allowed_special={"<|endoftext|>"})
            # 50257 for [SEP]
            self.text = {
                1: torch.IntTensor([50257] + encode('Question: Is this EEG segment abnormal? Answer: Yes <|endoftext|>')),
                0: torch.IntTensor([50257] + encode('Question: Is this EEG segment abnormal? Answer: No <|endoftext|>'))
            }
            self.prompt = torch.IntTensor([50257] + encode('Question: Is this EEG segment abnormal? Answer:'))

    def __len__(self):
        return len(self.files)

    def std_norm(self, x):
            mean = torch.mean(x, dim=(0, 1), keepdim=True)
            std = torch.std(x, dim=(0, 1), keepdim=True)
            x = (x - mean) / std
            return x

    def __getitem__(self, index):
        sample = pickle.load(open(os.path.join(self.root, self.files[index]), "rb"))
        X = sample["X"]
        Y = sample["y"]

        current_ch_names = sample["ch_names"]
        target_ch_names = self.ch_names
        if current_ch_names != target_ch_names:
            aligned_X = np.zeros((len(target_ch_names), X.shape[1]), dtype=X.dtype)
            ch_name_to_idx = {name: idx for idx, name in enumerate(target_ch_names)}
            for i, ch_name in enumerate(current_ch_names):
                if ch_name in ch_name_to_idx:
                    target_idx = ch_name_to_idx[ch_name]
                    aligned_X[target_idx] = X[i]
            X = aligned_X  
        data = torch.FloatTensor(X / 100)
        time = data.size(1) // 200
        input_time = [i  for i in range(time) for _ in range(data.size(0))]
        data = rearrange(data, 'N (A T) -> (A N) T', T=200)        
        input_chans = list(self.ch_names) * time

        if not self.is_instruct:
            input_chans = torch.IntTensor(get_chans(input_chans))
            input_time = torch.IntTensor(input_time)

            gpt_mask = torch.tril(torch.ones(data.size(0), data.size(0))).view(1, data.size(0), data.size(0))
            num_chans = len(self.ch_names)
            for i in range(time):
                gpt_mask[:, i * num_chans:(i + 1) * num_chans,  i * num_chans:(i + 1) * num_chans] = 1
            return data, Y, input_chans, input_time, gpt_mask.bool()
        
        if self.is_val:
            text = self.prompt
        else:
            text = self.text[int(Y)]
            # pad text to text_max_len
            valid_text_len = text.size(0)
            if self.text_max_len > valid_text_len:
                text_pad = torch.full((self.text_max_len,), fill_value=50256)
                text_pad[:valid_text_len] = text
                text = text_pad

        # pad eeg to eeg_max_len
        valid_eeg_len = data.size(0)
        if self.eeg_max_len > data.size(0):
            X_eeg = torch.zeros((self.eeg_max_len, 200))
            X_eeg[:data.size(0)] = data
            eeg_mask = torch.ones(self.eeg_max_len)
            eeg_mask[valid_eeg_len:] = 0

            input_chans.extend(['pad'] * (self.eeg_max_len - data.size(0)))
            input_time.extend([0] * (self.eeg_max_len - data.size(0)))
        else:
            X_eeg = data
            eeg_mask = torch.ones(data.size(0))

        input_chans = torch.IntTensor(get_chans(input_chans))
        input_time = torch.IntTensor(input_time)

        # remove unused ch
        ms_bool, ms_ch = remove_unused_ch(ms_ch_dict, self.ch_names) 
        ms_bool = list(itertools.chain(*ms_bool)) 
        num_ms_ch = ms_bool.count(True) 

        len_eeg_gpt_mask = num_ms_ch*time
        num_tokens = len_eeg_gpt_mask + text.size(0)
        gpt_mask = torch.tril(torch.ones(num_tokens, num_tokens)).view(1, num_tokens, num_tokens)
        # gpt eeg mask
        gpt_eeg_mask = get_gpt_eeg_mask(len_eeg_gpt_mask, ms_ch, num_ms_ch, time)
        gpt_mask[:, :len_eeg_gpt_mask, :len_eeg_gpt_mask] = gpt_eeg_mask
        ms_bool = torch.tensor(ms_bool, dtype=torch.bool)
        if self.is_val:
            return X_eeg, text, Y, input_chans, input_time, eeg_mask.bool(), gpt_mask.bool(), ms_bool, len_eeg_gpt_mask
        
        Y_text = torch.full_like(text, fill_value=-1)
        prompt_len = self.prompt.size(0)
        Y_text[prompt_len - 1:valid_text_len - 1] = text[prompt_len:valid_text_len]
        return X_eeg, text, Y_text, input_chans, input_time, eeg_mask.bool(), gpt_mask.bool(), ms_bool, len_eeg_gpt_mask
    

class TUEVLoader4BAR(Dataset):
    # spsw: spike and slow wave
    # gped: generalized periodic epileptiform discharge
    # pled: periodic lateralized epileptiform dischage
    # eyem: eye movement
    # artf: artifact
    # bckg: background
    # 1: spsw
    # 2: gped
    # 3: pled
    # 4: eyem
    # 5: artf
    # 6: bckg
    def __init__(self, root, files, sampling_rate=200, eeg_max_len=-1, text_max_len=-1, is_instruct=False, is_val=False):
        self.root = root
        self.files = files
        self.default_rate = 200
        self.sampling_rate = sampling_rate
        self.is_instruct = is_instruct
        self.is_val = is_val
        self.eeg_max_len = eeg_max_len
        self.text_max_len = text_max_len

        ch_names = ['EEG FP1', 'EEG FP2-REF', 'EEG F3-REF', 'EEG F4-REF', 'EEG C3-REF', 'EEG C4-REF', 'EEG P3-REF', 'EEG P4-REF', 'EEG O1-REF', 'EEG O2-REF', 'EEG F7-REF', \
                    'EEG F8-REF', 'EEG T3-REF', 'EEG T4-REF', 'EEG T5-REF', 'EEG T6-REF', 'EEG A1-REF', 'EEG A2-REF', 'EEG FZ-REF', 'EEG CZ-REF', 'EEG PZ-REF', 'EEG T1-REF', 'EEG T2-REF']
        self.ch_names = [name.split(' ')[-1].split('-')[0] for name in ch_names]
        self.ch_names = [standard_1020_change.get(name, name) for name in self.ch_names] 
        self.ch_names = [name for name in standard_1020 if name in self.ch_names] 

        if is_instruct:
            enc = tiktoken.get_encoding("gpt2")
            encode = lambda s: enc.encode(s, allowed_special={"<|endoftext|>"})
            # 50257 for [SEP]
            self.text = {
                0: torch.IntTensor([50257] + encode('Question: Which event type does this EEG segment belong to? Options: (A) spike and slow wave. (B) generalized periodic epileptiform discharge. (C) periodic lateralized epileptiform discharge. (D) eye movement. (E) artifact. (F) background. Answer: (A) <|endoftext|>')),
                1: torch.IntTensor([50257] + encode('Question: Which event type does this EEG segment belong to? Options: (A) spike and slow wave. (B) generalized periodic epileptiform discharge. (C) periodic lateralized epileptiform discharge. (D) eye movement. (E) artifact. (F) background. Answer: (B) <|endoftext|>')),
                2: torch.IntTensor([50257] + encode('Question: Which event type does this EEG segment belong to? Options: (A) spike and slow wave. (B) generalized periodic epileptiform discharge. (C) periodic lateralized epileptiform discharge. (D) eye movement. (E) artifact. (F) background. Answer: (C) <|endoftext|>')),
                3: torch.IntTensor([50257] + encode('Question: Which event type does this EEG segment belong to? Options: (A) spike and slow wave. (B) generalized periodic epileptiform discharge. (C) periodic lateralized epileptiform discharge. (D) eye movement. (E) artifact. (F) background. Answer: (D) <|endoftext|>')),
                4: torch.IntTensor([50257] + encode('Question: Which event type does this EEG segment belong to? Options: (A) spike and slow wave. (B) generalized periodic epileptiform discharge. (C) periodic lateralized epileptiform discharge. (D) eye movement. (E) artifact. (F) background. Answer: (E) <|endoftext|>')),
                5: torch.IntTensor([50257] + encode('Question: Which event type does this EEG segment belong to? Options: (A) spike and slow wave. (B) generalized periodic epileptiform discharge. (C) periodic lateralized epileptiform discharge. (D) eye movement. (E) artifact. (F) background. Answer: (F) <|endoftext|>')),
            }
            self.prompt = torch.IntTensor([50257] + encode('Question: Which event type does this EEG segment belong to? Options: (A) spike and slow wave. (B) generalized periodic epileptiform discharge. (C) periodic lateralized epileptiform discharge. (D) eye movement. (E) artifact. (F) background. Answer: ('))

    def std_norm(self, x):
            mean = torch.mean(x, dim=(0, 1), keepdim=True)
            std = torch.std(x, dim=(0, 1), keepdim=True)
            x = (x - mean) / std
            return x

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        sample = pickle.load(open(os.path.join(self.root, self.files[index]), "rb"))
        X = sample["signal"]
        Y = int(sample["label"][0] - 1)

        current_ch_names = sample["ch_names"]
        target_ch_names = self.ch_names
        if current_ch_names != target_ch_names:
            aligned_X = np.zeros((len(target_ch_names), X.shape[1]), dtype=X.dtype)
            ch_name_to_idx = {name: idx for idx, name in enumerate(target_ch_names)}
            for i, ch_name in enumerate(current_ch_names):
                if ch_name in ch_name_to_idx:
                    target_idx = ch_name_to_idx[ch_name]
                    aligned_X[target_idx] = X[i]
            X = aligned_X  
        data = torch.FloatTensor(X / 100)
        time = data.size(1) // 200
        input_time = [i  for i in range(time) for _ in range(data.size(0))]
        data = rearrange(data, 'N (A T) -> (A N) T', T=200)
        input_chans = list(self.ch_names) * time

        if not self.is_instruct:
            input_chans = torch.IntTensor(get_chans(input_chans))
            input_time = torch.IntTensor(input_time)

            gpt_mask = torch.tril(torch.ones(data.size(0), data.size(0))).view(1, data.size(0), data.size(0))
            num_chans = len(self.ch_names)
            for i in range(time):
                gpt_mask[:, i * num_chans:(i + 1) * num_chans,  i * num_chans:(i + 1) * num_chans] = 1
            return data, Y, input_chans, input_time, gpt_mask.bool()
        
        if self.is_val:
            text = self.prompt
        else:
            text = self.text[int(Y)]
            # pad text to text_max_len
            valid_text_len = text.size(0)
            if self.text_max_len > valid_text_len:
                text_pad = torch.full((self.text_max_len,), fill_value=50256)
                text_pad[:valid_text_len] = text
                text = text_pad

        # pad eeg to eeg_max_len
        valid_eeg_len = data.size(0)
        if self.eeg_max_len > data.size(0):
            X_eeg = torch.zeros((self.eeg_max_len, 200))
            X_eeg[:data.size(0)] = data
            eeg_mask = torch.ones(self.eeg_max_len)
            eeg_mask[valid_eeg_len:] = 0

            input_chans.extend(['pad'] * (self.eeg_max_len - data.size(0)))
            input_time.extend([0] * (self.eeg_max_len - data.size(0)))
        else:
            X_eeg = data
            eeg_mask = torch.ones(data.size(0))

        input_chans = torch.IntTensor(get_chans(input_chans))
        input_time = torch.IntTensor(input_time)

        # remove unused ch
        ms_bool, ms_ch = remove_unused_ch(ms_ch_dict, self.ch_names) 
        ms_bool = list(itertools.chain(*ms_bool)) 
        num_ms_ch = ms_bool.count(True) 

        len_eeg_gpt_mask = num_ms_ch*time
        num_tokens = len_eeg_gpt_mask + text.size(0) 
        gpt_mask = torch.tril(torch.ones(num_tokens, num_tokens)).view(1, num_tokens, num_tokens)
        # gpt eeg mask
        gpt_eeg_mask = get_gpt_eeg_mask(len_eeg_gpt_mask, ms_ch, num_ms_ch, time)
        gpt_mask[:, :len_eeg_gpt_mask, :len_eeg_gpt_mask] = gpt_eeg_mask
        ms_bool = torch.tensor(ms_bool, dtype=torch.bool)

        if self.is_val:
            return X_eeg, text, Y, input_chans, input_time, eeg_mask.bool(), gpt_mask.bool(), ms_bool, len_eeg_gpt_mask
        
        Y_text = torch.full_like(text, fill_value=-1)
        prompt_len = self.prompt.size(0) - 1
        Y_text[prompt_len - 1:valid_text_len - 1] = text[prompt_len:valid_text_len]
        return X_eeg, text, Y_text, input_chans, input_time, eeg_mask.bool(), gpt_mask.bool(), ms_bool, len_eeg_gpt_mask
    

class TUSLLoader4BAR(Dataset):
    def __init__(self, root, files, sampling_rate=200, eeg_max_len=-1, text_max_len=-1, is_instruct=False, is_val=False):
        self.root = root
        self.files = files
        self.default_rate = 200
        self.sampling_rate = sampling_rate
        self.is_instruct = is_instruct
        self.is_val = is_val
        self.eeg_max_len = eeg_max_len
        self.text_max_len = text_max_len

        ch_names = ['EEG FP1', 'EEG FP2-REF', 'EEG F3-REF', 'EEG F4-REF', 'EEG C3-REF', 'EEG C4-REF', 'EEG P3-REF', 'EEG P4-REF', 'EEG O1-REF', 'EEG O2-REF', 'EEG F7-REF', \
                    'EEG F8-REF', 'EEG T3-REF', 'EEG T4-REF', 'EEG T5-REF', 'EEG T6-REF', 'EEG A1-REF', 'EEG A2-REF', 'EEG FZ-REF', 'EEG CZ-REF', 'EEG PZ-REF', 'EEG T1-REF', 'EEG T2-REF']
        self.ch_names = [name.split(' ')[-1].split('-')[0] for name in ch_names]
        self.ch_names = [standard_1020_change.get(name, name) for name in self.ch_names] 
        self.ch_names = [name for name in standard_1020 if name in self.ch_names] 

        if is_instruct:
            enc = tiktoken.get_encoding("gpt2")
            encode = lambda s: enc.encode(s, allowed_special={"<|endoftext|>"})
            # 50257 for [SEP]
            self.text = {
                0: torch.IntTensor([50257] + encode('Question: Which type does this EEG segment belong to? Options: (G) background. (H) seizure. (I) slowing. Answer: (G) <|endoftext|>')),
                1: torch.IntTensor([50257] + encode('Question: Which type does this EEG segment belong to? Options: (G) background. (H) seizure. (I) slowing. Answer: (H) <|endoftext|>')),
                2: torch.IntTensor([50257] + encode('Question: Which type does this EEG segment belong to? Options: (G) background. (H) seizure. (I) slowing. Answer: (I) <|endoftext|>'))
            }
            self.prompt = torch.IntTensor([50257] + encode('Question: Which type does this EEG segment belong to? Options: (G) background. (H) seizure. (I) slowing. Answer: ('))

    def std_norm(self, x):
            mean = torch.mean(x, dim=(0, 1), keepdim=True)
            std = torch.std(x, dim=(0, 1), keepdim=True)
            x = (x - mean) / std
            return x

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        sample = pickle.load(open(os.path.join(self.root, self.files[index]), "rb"))
        X = sample["X"]
        Y = int(sample["y"])

        current_ch_names = sample["ch_names"]
        target_ch_names = self.ch_names
        if current_ch_names != target_ch_names:
            aligned_X = np.zeros((len(target_ch_names), X.shape[1]), dtype=X.dtype)
            ch_name_to_idx = {name: idx for idx, name in enumerate(target_ch_names)}
            for i, ch_name in enumerate(current_ch_names):
                if ch_name in ch_name_to_idx:
                    target_idx = ch_name_to_idx[ch_name]
                    aligned_X[target_idx] = X[i]
            X = aligned_X  
        data = torch.FloatTensor(X / 100)
        time = data.size(1) // 200
        input_time = [i  for i in range(time) for _ in range(data.size(0))]
        data = rearrange(data, 'N (A T) -> (A N) T', T=200)
        input_chans = list(self.ch_names) * time

        if not self.is_instruct:
            input_chans = torch.IntTensor(get_chans(input_chans))
            input_time = torch.IntTensor(input_time)

            gpt_mask = torch.tril(torch.ones(data.size(0), data.size(0))).view(1, data.size(0), data.size(0))
            num_chans = len(self.ch_names)
            for i in range(time):
                gpt_mask[:, i * num_chans:(i + 1) * num_chans,  i * num_chans:(i + 1) * num_chans] = 1
            return data, Y, input_chans, input_time, gpt_mask.bool()
        
        if self.is_val:
            text = self.prompt
        else:
            text = self.text[int(Y)]
            # pad text to text_max_len
            valid_text_len = text.size(0)
            if self.text_max_len > valid_text_len:
                text_pad = torch.full((self.text_max_len,), fill_value=50256)
                text_pad[:valid_text_len] = text
                text = text_pad

        # pad eeg to eeg_max_len
        valid_eeg_len = data.size(0)
        if self.eeg_max_len > data.size(0):
            X_eeg = torch.zeros((self.eeg_max_len, 200))
            X_eeg[:data.size(0)] = data
            eeg_mask = torch.ones(self.eeg_max_len)
            eeg_mask[valid_eeg_len:] = 0

            input_chans.extend(['pad'] * (self.eeg_max_len - data.size(0)))
            input_time.extend([0] * (self.eeg_max_len - data.size(0)))
        else:
            X_eeg = data
            eeg_mask = torch.ones(data.size(0))

        input_chans = torch.IntTensor(get_chans(input_chans))
        input_time = torch.IntTensor(input_time)
        # remove unused ch
        ms_bool, ms_ch = remove_unused_ch(ms_ch_dict, self.ch_names)
        ms_bool = list(itertools.chain(*ms_bool)) 
        num_ms_ch = ms_bool.count(True) 

        len_eeg_gpt_mask = num_ms_ch*time
        num_tokens = len_eeg_gpt_mask + text.size(0) 
        gpt_mask = torch.tril(torch.ones(num_tokens, num_tokens)).view(1, num_tokens, num_tokens)
        # gpt eeg mask
        gpt_eeg_mask = get_gpt_eeg_mask(len_eeg_gpt_mask, ms_ch, num_ms_ch, time)
        gpt_mask[:, :len_eeg_gpt_mask, :len_eeg_gpt_mask] = gpt_eeg_mask
        ms_bool = torch.tensor(ms_bool, dtype=torch.bool)        

        if self.is_val:
            return X_eeg, text, Y, input_chans, input_time, eeg_mask.bool(), gpt_mask.bool(), ms_bool, len_eeg_gpt_mask
        
        Y_text = torch.full_like(text, fill_value=-1)
        prompt_len = self.prompt.size(0)
        Y_text[prompt_len - 1:valid_text_len - 1] = text[prompt_len:valid_text_len]
        return X_eeg, text, Y_text, input_chans, input_time, eeg_mask.bool(), gpt_mask.bool(), ms_bool, len_eeg_gpt_mask


class HMCLoader4BAR(Dataset):
    def __init__(self, root, files, sampling_rate=200, eeg_max_len=-1, text_max_len=-1, is_instruct=False, is_val=False):
        self.root = root
        self.files = files
        self.default_rate = 200
        self.sampling_rate = sampling_rate
        self.is_instruct = is_instruct
        self.is_val = is_val
        self.eeg_max_len = eeg_max_len
        self.text_max_len = text_max_len

        if is_instruct:
            enc = tiktoken.get_encoding("gpt2")
            encode = lambda s: enc.encode(s, allowed_special={"<|endoftext|>"})
            # 50257 for [SEP]
            self.text = {
                0: torch.IntTensor([50257] + encode('Question: Which sleep type does this EEG segment belong to? Options: (J) Wake. (K) NREM-1. (L) NREM-2. (M) NREM-3. (N) REM. Answer: (J) <|endoftext|>')),
                1: torch.IntTensor([50257] + encode('Question: Which sleep type does this EEG segment belong to? Options: (J) Wake. (K) NREM-1. (L) NREM-2. (M) NREM-3. (N) REM. Answer: (K) <|endoftext|>')),
                2: torch.IntTensor([50257] + encode('Question: Which sleep type does this EEG segment belong to? Options: (J) Wake. (K) NREM-1. (L) NREM-2. (M) NREM-3. (N) REM. Answer: (L) <|endoftext|>')),
                3: torch.IntTensor([50257] + encode('Question: Which sleep type does this EEG segment belong to? Options: (J) Wake. (K) NREM-1. (L) NREM-2. (M) NREM-3. (N) REM. Answer: (M) <|endoftext|>')),
                4: torch.IntTensor([50257] + encode('Question: Which sleep type does this EEG segment belong to? Options: (J) Wake. (K) NREM-1. (L) NREM-2. (M) NREM-3. (N) REM. Answer: (N) <|endoftext|>')),
            }
            self.prompt = torch.IntTensor([50257] + encode('Question: Which sleep type does this EEG segment belong to? Options: (J) Wake. (K) NREM-1. (L) NREM-2. (M) NREM-3. (N) REM. Answer: ('))

    def std_norm(self, x):
            mean = torch.mean(x, dim=(0, 1), keepdim=True)
            std = torch.std(x, dim=(0, 1), keepdim=True)
            x = (x - mean) / std
            return x

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        sample = pickle.load(open(os.path.join(self.root, self.files[index]), "rb"))
        X = sample["X"]
        Y = int(sample["y"])
        data = torch.FloatTensor(X / 100)
        time = data.size(1) // 200
        input_time = [i  for i in range(time) for _ in range(data.size(0))]
        data = rearrange(data, 'N (A T) -> (A N) T', T=200)

        ch_names = sample["ch_names"]
        input_chans = list(ch_names) * time

        if not self.is_instruct:
            input_chans = torch.IntTensor(get_chans(input_chans))
            input_time = torch.IntTensor(input_time)

            gpt_mask = torch.tril(torch.ones(data.size(0), data.size(0))).view(1, data.size(0), data.size(0))
            num_chans = len(ch_names)
            for i in range(time):
                gpt_mask[:, i * num_chans:(i + 1) * num_chans,  i * num_chans:(i + 1) * num_chans] = 1
            return data, Y, input_chans, input_time, gpt_mask.bool()
        
        if self.is_val:
            text = self.prompt
        else:
            text = self.text[int(Y)]
            # pad text to text_max_len
            valid_text_len = text.size(0)
            if self.text_max_len > valid_text_len:
                text_pad = torch.full((self.text_max_len,), fill_value=50256)
                text_pad[:valid_text_len] = text
                text = text_pad

        # pad eeg to eeg_max_len
        valid_eeg_len = data.size(0)
        if self.eeg_max_len > data.size(0):
            X_eeg = torch.zeros((self.eeg_max_len, 200))
            X_eeg[:data.size(0)] = data
            eeg_mask = torch.ones(self.eeg_max_len)
            eeg_mask[valid_eeg_len:] = 0

            input_chans.extend(['pad'] * (self.eeg_max_len - data.size(0)))
            input_time.extend([0] * (self.eeg_max_len - data.size(0)))
        else:
            X_eeg = data
            eeg_mask = torch.ones(data.size(0))

        input_chans = torch.IntTensor(get_chans(input_chans))
        input_time = torch.IntTensor(input_time)

        # remove unused ch
        ms_bool, ms_ch = remove_unused_ch(ms_ch_dict, ch_names)
        ms_bool = list(itertools.chain(*ms_bool)) 
        num_ms_ch = ms_bool.count(True) 

        len_eeg_gpt_mask = num_ms_ch*time
        num_tokens = len_eeg_gpt_mask + text.size(0) 
        gpt_mask = torch.tril(torch.ones(num_tokens, num_tokens)).view(1, num_tokens, num_tokens)
        num_chans = len(ch_names)
        # gpt eeg mask
        gpt_eeg_mask = get_gpt_eeg_mask(len_eeg_gpt_mask, ms_ch, num_ms_ch, time)
        gpt_mask[:, :len_eeg_gpt_mask, :len_eeg_gpt_mask] = gpt_eeg_mask
        ms_bool = torch.tensor(ms_bool, dtype=torch.bool)
        
        if self.is_val:
            return X_eeg, text, Y, input_chans, input_time, eeg_mask.bool(), gpt_mask.bool(), ms_bool, len_eeg_gpt_mask
        
        Y_text = torch.full_like(text, fill_value=-1)
        prompt_len = self.prompt.size(0) - 1
        Y_text[prompt_len - 1:valid_text_len - 1] = text[prompt_len:valid_text_len]
        return X_eeg, text, Y_text, input_chans, input_time, eeg_mask.bool(), gpt_mask.bool(), ms_bool, len_eeg_gpt_mask


class WorkloadLoader4BAR(Dataset):
    def __init__(self, root, files, sampling_rate=200, eeg_max_len=-1, text_max_len=-1, is_instruct=False, is_val=False):
        self.root = root
        self.files = files
        self.default_rate = 200
        self.sampling_rate = sampling_rate
        self.is_instruct = is_instruct
        self.is_val = is_val
        self.eeg_max_len = eeg_max_len
        self.text_max_len = text_max_len

        if is_instruct:
            enc = tiktoken.get_encoding("gpt2")
            encode = lambda s: enc.encode(s, allowed_special={"<|endoftext|>"})
            # 50257 for [SEP]
            self.text = {
                1: torch.IntTensor([50257] + encode('Question: Is this EEG segment of high workload? Answer: Yes <|endoftext|>')),
                0: torch.IntTensor([50257] + encode('Question: Is this EEG segment of high workload? Answer: No <|endoftext|>')),
            }
            self.prompt = torch.IntTensor([50257] + encode('Question: Is this EEG segment of high workload? Answer:'))

    def std_norm(self, x):
            mean = torch.mean(x, dim=(0, 1), keepdim=True)
            std = torch.std(x, dim=(0, 1), keepdim=True)
            x = (x - mean) / std
            return x

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        sample = pickle.load(open(os.path.join(self.root, self.files[index]), "rb"))
        X = sample["X"]
        Y = int(sample["y"])
        data = torch.FloatTensor(X / 100)
        # data = self.std_norm(data)

        time = data.size(1) // 200
        input_time = [i  for i in range(time) for _ in range(data.size(0))]
        data = rearrange(data, 'N (A T) -> (A N) T', T=200)

        ch_names = sample["ch_names"]
        input_chans = list(ch_names) * time

        if not self.is_instruct:
            input_chans = torch.IntTensor(get_chans(input_chans))
            input_time = torch.IntTensor(input_time)

            gpt_mask = torch.tril(torch.ones(data.size(0), data.size(0))).view(1, data.size(0), data.size(0))
            num_chans = len(ch_names)
            for i in range(time):
                gpt_mask[:, i * num_chans:(i + 1) * num_chans,  i * num_chans:(i + 1) * num_chans] = 1
            return data, Y, input_chans, input_time, gpt_mask.bool()
        
        if self.is_val:
            text = self.prompt
        else:
            text = self.text[int(Y)]
            # pad text to text_max_len
            valid_text_len = text.size(0)
            if self.text_max_len > valid_text_len:
                text_pad = torch.full((self.text_max_len,), fill_value=50256)
                text_pad[:valid_text_len] = text
                text = text_pad

        # pad eeg to eeg_max_len
        valid_eeg_len = data.size(0)
        if self.eeg_max_len > data.size(0):
            X_eeg = torch.zeros((self.eeg_max_len, 200))
            X_eeg[:data.size(0)] = data
            eeg_mask = torch.ones(self.eeg_max_len)
            eeg_mask[valid_eeg_len:] = 0

            input_chans.extend(['pad'] * (self.eeg_max_len - data.size(0)))
            input_time.extend([0] * (self.eeg_max_len - data.size(0)))
        else:
            X_eeg = data
            eeg_mask = torch.ones(data.size(0))

        input_chans = torch.IntTensor(get_chans(input_chans))
        input_time = torch.IntTensor(input_time)

        # remove unused ch
        ms_bool, ms_ch = remove_unused_ch(ms_ch_dict, ch_names) 
        ms_bool = list(itertools.chain(*ms_bool)) 
        num_ms_ch = ms_bool.count(True) 

        len_eeg_gpt_mask = num_ms_ch*time
        num_tokens = len_eeg_gpt_mask + text.size(0) 
        gpt_mask = torch.tril(torch.ones(num_tokens, num_tokens)).view(1, num_tokens, num_tokens)
        num_chans = len(ch_names)
        # gpt eeg mask
        gpt_eeg_mask = get_gpt_eeg_mask(len_eeg_gpt_mask, ms_ch, num_ms_ch, time)
        gpt_mask[:, :len_eeg_gpt_mask, :len_eeg_gpt_mask] = gpt_eeg_mask
        ms_bool = torch.tensor(ms_bool, dtype=torch.bool)

        if self.is_val:
            return X_eeg, text, Y, input_chans, input_time, eeg_mask.bool(), gpt_mask.bool(), ms_bool, len_eeg_gpt_mask
        
        Y_text = torch.full_like(text, fill_value=-1)
        prompt_len = self.prompt.size(0)
        Y_text[prompt_len - 1:valid_text_len - 1] = text[prompt_len:valid_text_len]
        return X_eeg, text, Y_text, input_chans, input_time, eeg_mask.bool(), gpt_mask.bool(), ms_bool, len_eeg_gpt_mask
