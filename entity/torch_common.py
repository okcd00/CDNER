# coding: utf-8
# ==========================================================================
#   Copyright (C) 2020 All rights reserved.
#
#   filename : torch_common.py
#   origin   : cyx
#   author   : chendian / okcd00@qq.com
#   date     : 2020-10-21
#   desc     : pick from utie with minor modifies.
# ==========================================================================


from six import iteritems
import numpy as np
import torch
from torch.autograd import Variable


use_cuda = torch.cuda.is_available()
print('use_cuda: {}'.format(use_cuda))


def dict_to_variable(dic, device, dtype=None):
    return {k: to_variable(v, dtype=dtype, device=device) for k, v in iteritems(dic)}


def dict_to_numpy(dic):
    return {k: to_numpy(v) for k, v in iteritems(dic)}


def to_numpy(tensor, dtype=None):
    """
    把 torch.Tensor 或者 torch.autograd.Variable 变成 np.ndarray
    :param tensor:
    :param dtype:
    :return:
    """
    if isinstance(tensor, Variable):
        tensor = tensor.data
    return tensor.cpu().numpy()


def to_tensor(np_array, device, dtype=None):
    """
    根据 np_array 来构建 tensor。 Variable的 dtype 应该是和 np_array 一样的，所以这里的 dtype 应该用不到
    :param np_array:
    :param device: gpu 编号
    :param dtype:
    :return:
    """
    if isinstance(np_array, np.ndarray):
        np_array = np_array.astype(dtype)
    elif isinstance(np_array, list):
        np_array = np.asarray(np_array, dtype=dtype)

    if not torch.is_tensor(np_array):
        tensor = torch.from_numpy(np_array)
    else:
        tensor = np_array

    if use_cuda:
        tensor = tensor.to(device)
    return tensor


def to_variable(np_array, device, dtype=None):
    """
    根据 np_array 来构建 Variable。
    Variable的 dtype 应该是和 np_array 一样的，所以这里的 dtype 应该用不到
    :param np_array:
    :param device: gpu 编号
    :param dtype:
    :return:
    """
    # from utie.torch_common import to_variable
    if isinstance(np_array, Variable):
        if use_cuda and not np_array.is_cuda:
            np_array = np_array.to(device)
        return np_array
    return Variable(to_tensor(np_array, dtype=dtype, device=device))


def gen_mask(lengths, max_len=None, dtype=np.float32):
    """
    lengths = [2, 1, 3, 1]
    return: shape=[batch-size, max-len]
    [
    [1, 1, 0],
    [1, 0, 0],
    [1, 1, 1],
    [1, 0, 0],
    ]
    """
    lengths = np.asarray(lengths)
    if lengths.ndim == 2:
        assert lengths.shape[0] == 1 or lengths.shape[1] == 1
        lengths = lengths.flatten()
    n = len(lengths)
    if max_len is None:
        max_len = lengths.max()
    mask = np.zeros([n, max_len], dtype=dtype)
    for i, length in enumerate(lengths):
        mask[i, : length] = 1
    return mask


def new_variable(torch_variable, size):
    """ 从一个 torch Variable 在同用的地方(cpu/gpu) 生成一个zeros 的 Variable"""
    return Variable(torch_variable.data.new(*size).zero_())


def load_param(path, cuda_x=None):
    """ model_path can be a path string or a file object"""
    if use_cuda:
        if cuda_x is None:
            param = torch.load(path, map_location=lambda storage, loc: storage.cuda())
        else:
            param = torch.load(path, map_location=lambda storage, loc: storage.cuda(cuda_x))
    else:
        param = torch.load(path, map_location=lambda storage, loc: storage.cpu())
    return param


def load_model(model, model_path, strict=True, cuda_x=None):
    """ model_path can be a path string or a file object"""
    state_dict = load_param(model_path, cuda_x)

    def move_embedding(model_param_dict):
        # 为了适应模型代码结构的改动
        for k in model_param_dict.keys():
            if k == 'language_rep.word_embedding.weight':
                model_param_dict['embedding.word_embedding.weight'] = model_param_dict[k]
                model_param_dict.pop(k)
            elif k == 'language_rep.position_encoding.pe':
                model_param_dict['embedding.position_encoding.pe'] = model_param_dict[k]
                model_param_dict.pop(k)
    move_embedding(state_dict)

    model.load_state_dict(state_dict, strict=strict)
    if isinstance(model_path, str):
        print('model load from {}'.format(model_path))
    else:
        print('model loaded from file object')
    return model
