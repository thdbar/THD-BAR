from itertools import chain

import numpy as np

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
    'O9', 'O1', 'I1', 'OZ', 'IZ', 'O2', 'O10', 'I2'\
]


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

ms_ch_lst = [
    [
        [[['AF9'], ['AF7'], ['F9']], [['FP1'], ['AF5'], ['AF3']], [['F7'], ['F5'], ['F3']],],
        [[['FPZ'], ['AF1'], ['AFZ'], ['AF2']], [['F1'], ['FZ'], ['F2']],],
        [[['FP2'], ['AF4'], ['AF6']], [['AF8'], ['AF10'], ['F10']], [['F4'], ['F6'], ['F8']],],
    ],
    [
        [[['FT9'], ['FT7']], [['A1'], ['T9'], ['T7']], [['TP9'], ['TP7']],],
        [[['FC5'], ['FC3']], [['C5'], ['C3']], [['CP5'], ['CP3']],],
        [[['FC1'], ['FCZ'], ['FC2']], [['C1'], ['CZ'], ['C2']], [['CP1'], ['CPZ'], ['CP2']],],
        [[['FC4'], ['FC6']], [['C4'], ['C6']], [['CP4'], ['CP6']],],
        [[['FT8'], ['FT10']], [['T8'], ['T10'], ['A2']], [['TP8'], ['TP10']],],
    ],
    [
        [[['P9'], ['P7']], [['P5'], ['P3']], [['PO9'], ['PO7']], [['PO5'], ['PO3']],],
        [[['P1'], ['PZ'], ['P2']], [['PO1'], ['POZ'], ['PO2']],],
        [[['P4'], ['P6']], [['P8'], ['P10']], [['PO4'], ['PO6']], [['PO8'], ['PO10']],],
        [[['O9'], ['O1'], ['I1']], [['OZ'], ['IZ']], [['O2'], ['O10'], ['I2']],],
    ],
]

ch_nums = [1, 3, 12, 36, 90]

def flatten_from_dimension(data, current_dim, target_dim):
    """
    Flatten the list starting from `target_dim`.

    Parameters:
    - current_dim: The current depth level, starting from 0.
    - target_dim: The target depth level to retain. Dimensions below this level remain unchanged,
                  while dimensions at and below this level are flattened.
    """

    if not isinstance(data, list):
        return data

    if current_dim >= target_dim:
        flat_list = []
        for item in data:
            if isinstance(item, list):
                flat_list.extend(flatten_from_dimension(item, current_dim + 1, target_dim))
            else:
                flat_list.append(item)
        return flat_list
    else:
        new_list = []
        for item in data:
            new_list.append(flatten_from_dimension(item, current_dim + 1, target_dim))
        return new_list

def cat_lst_to_2dim(lst):
    num_dim = list_dimension(lst)
    if num_dim == 1:
        return [lst]
    elif num_dim == 2:
        return lst
    else:
        for i in range(num_dim-2):
            lst = list(chain(*lst))
        return lst

def list_dimension(lst):
    if not isinstance(lst, list):
        return 0
    if len(lst) == 0:
        return 1
    return 1 + list_dimension(lst[0])

ms_ch_dict = {
    f"ch_lst_scale{i}": cat_lst_to_2dim(flatten_from_dimension(ms_ch_lst, 0, i))
    for i in range(5)
}


def generate_ms_spatial_mask(block_sizes):
    total_size = sum(block_sizes)
    matrix = np.zeros((total_size, total_size), dtype=int)
    current_start = 0
    for size in block_sizes:
        current_end = current_start + size
        matrix[current_start:current_end, current_start:current_end] = 0
        if current_end < total_size:
            matrix[current_end:, :current_end] = 1
        current_start = current_end
    return matrix


def gen_stair_scale_mask(ms_ch, time: int) -> np.ndarray:
    num_ms_lst = [len(ch_s_i) for ch_s_i in ms_ch]
    ms_spatial_mask = generate_ms_spatial_mask(num_ms_lst)
    block_size = ms_spatial_mask.shape[0]
    total_size = time * block_size
    matrix = np.zeros((total_size, total_size), dtype=int)
    row_indices = np.arange(total_size)[:, None]
    block_starts = (row_indices // block_size) * block_size
    mask = (np.arange(total_size) <= block_starts).astype(int)
    matrix += mask
    view_shape = (time, block_size, time, block_size)
    strides = (block_size*matrix.strides[0], matrix.strides[0], 
              block_size*matrix.strides[1], matrix.strides[1])
    block_view = np.lib.stride_tricks.as_strided(matrix, view_shape, strides)
    diag_idx = np.diag_indices(time)
    block_view[diag_idx[0], :, diag_idx[1], :] = np.ones_like(ms_spatial_mask)
    block_view[diag_idx[0][:-1], :, diag_idx[1][:-1]+1, :] = ms_spatial_mask
    return matrix


def find_elements_in_sublists(list1, list2):
    """
    Check if elements from list2 exist in any sublist of list1 and return a corresponding boolean list.

    Parameters:
    - list1: A list containing multiple sublists.
    - list2: A list of elements to check.

    Returns:
    - A boolean list indicating whether any sublist in list1 contains any element from list2.
    """
    result = []
    for item in list2:
        for idx, sublist in enumerate(list1):
            if idx in result:
                continue
            if item in sublist:
                result.append(idx)
                break  # Stop checking once a matching sublist is found

    bool_result = [idx in result for idx in range(len(list1))]
    return bool_result


def remove_unused_ch(ms_ch_dict, target_ch_list):
    ms_bool = []
    ms_ch = []
    for i in range(len(ms_ch_dict)):
        ch_lst_scale_i = ms_ch_dict[f"ch_lst_scale{i}"]
        bool_scale_i = find_elements_in_sublists(ch_lst_scale_i, target_ch_list)
        ch_lst_scale_i_odd = [ch for ch, include in zip(ch_lst_scale_i, bool_scale_i) if include]
        ms_ch.append(ch_lst_scale_i_odd)
        ms_bool.append(bool_scale_i)

    # Convert ms_bool to a tensor on the specified device (GPU)
    return ms_bool, ms_ch

