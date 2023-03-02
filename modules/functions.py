import torch
import torch.nn as nn


def segment_mean(index, src, n_max_segment=50):
    """
    每个句子有 src.shape[1] 个 token
    最多允许有 n_max_segment 个 segment 
    每个 token 的 index 是属于第几个 segment 
    返回每个 segment 的表征均值
    index: [batch, sequence_length]    
    src: [batch, sequence_length, hidden_size]
    """
    index_one_hot = nn.functional.one_hot(index, n_max_segment).float()
    output = torch.einsum('ijk,ijb->ikb', index_one_hot, src)
    count = index_one_hot.sum(axis=1).unsqueeze(-1)
    return output / (1e-8 + count)


if __name__ == "__main__":
    print(
        segment_mean(
            torch.ones(2,100).long(), 
            torch.arange(1000).view(2, 100, 5).float()
        )
    )
