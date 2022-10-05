# coding: utf-8
# ==========================================================================
#   Copyright (C) 2016-2021 All rights reserved.
#
#   filename : feature_fusion.py
#   author   : chendian / okcd00@qq.com
#   date     : 2021-07-02
#   desc     :
# ==========================================================================
import torch
from torch import nn
import torch.nn.functional as F


class FeatureFusion(nn.Module):
    def __init__(self, method='none',
                 input_dim=768, head_hidden_dim=150):
        super(FeatureFusion, self).__init__()
        self.aligned_dimensions = False
        self.method = self.method_mapping(method)

        if self.method in ['none', 'concat']:
            self.fusion = self.concat_fusion
        elif self.method in ['weighted-sum', 'gated']:  # weighted-sum / gated
            self.fusion = self.gated_fusion
            self.aligned_dimensions = True
        elif self.method in ['biaffine', 'bi-affine']:
            # from modules.bi_affine import Biaffine
            self.fusion = self.biaffine_fusion
            self.aligned_dimensions = True
        else:
            raise ValueError(
                "Invalid feature fusion method type: {}".format(self.method))

        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=-1)
    
    def scaling(self, logit_vector):
        # make tensors in the last dimention have the same sum value (scale value).
        self.eps = 1e-8
        self.scale = 10  # [4.5398e-05, 9.9995e-01]
        value_m = logit_vector.sum(-1).unsqueeze(-1)
        logit_vector = logit_vector / (value_m + self.eps) * self.scale
        return logit_vector

    def method_mapping(self, method):
        method = method.lower()
        method = {
            'gated': 'weighted-sum',
            'bi_affine': 'bi-affine',
            'weighted_sum': 'weighted-sum',
            'weighted_concat': 'weighted-concat'
        }.get(method, method)
        return method

    def concat_fusion(self, feature_case, module_node=None):
        return torch.cat(feature_case, dim=-1), None

    def biaffine_fusion(self, feature_case, module_node=None):
        # need a biaffine module node
        bi_affine = module_node
        
        if len(feature_case) > 4:
            feature_case = (  
                feature_case[0] + feature_case[1],  # name = name_left + name_right
                sum(*feature_case[2:])   # ctx = ctx_left + ctx_right + ctx_attn
            )
        elif len(feature_case) == 4:
            feature_case = (  
                feature_case[0] + feature_case[1],  # name = name_left + name_right
                feature_case[2] + feature_case[3],  # ctx = ctx_left + ctx_right
            )
        elif len(feature_case) == 3:
            feature_case = (  
                (feature_case[0] + feature_case[1]) * 0.5,  # name = name_left + name_right
                feature_case[2]   # ctx = ctx_attn
            )
        assert len(feature_case) == 2
        feat_1, feat_2 = feature_case
        bat_span_size = feat_1.shape[:2]

        # join batch_size & span_size dimensions.
        # => [batch_size * n_spans, 1, input_dim]
        feat_1 = feat_1.view(-1, 1, *feat_1.shape[2:])
        feat_2 = feat_2.view(-1, 1, *feat_2.shape[2:])
        
        # [batch_size * n_spans, 1, input_dim] 
        # => [batch_size * n_spans, 1, 1, 1] => [batch_size * n_spans, 1]
        coeff = self.sigmoid(bi_affine(feat_1, feat_2).view(-1, 1))
        # => [batch_size * n_spans, 1, n_feature=2]
        weights = torch.stack([coeff, 1. - coeff], dim=-1)
        # => [batch_size * n_spans, n_feature=2, input_dim]
        features = torch.stack(feature_case, dim=-2).view(-1, 2, feat_1.shape[-1])
        
        # print('coeff', coeff.shape)
        # print('weights', weights.shape)
        # print('features', features.shape)
        
        # [batch_size * n_spans, 1, input_dim]
        # => [batch_size * n_spans, input_dim]
        # print(weights.shape)
        # print(features.shape)
        fusion = torch.bmm(weights, features).squeeze(1)

        # ret_tensor: [batch_size, n_spans, input_dim]
        # coeff: [batch_size * n_spans, 1]
        ret_tensor = fusion.view(*bat_span_size, *fusion.shape[1:])
        return ret_tensor, weights

    def gated_fusion(self, feature_case, module_node=None):
        # need a mlp module node
        mlp_module = module_node

        # [batch_size * n_spans, n_feature, input_dim]
        features = torch.stack(feature_case, dim=-2)
        bat_span_size = features.shape[:2]
        features = features.view(-1, *features.shape[2:])

        # [batch_size * n_spans, 1, n_feature]
        try:
            logits = self.scaling(mlp_module(features).transpose(-2, -1))
            weights = self.softmax(logits)
        except Exception as e:
            for feat in feature_case:
                print(feat.device)
            raise ValueError(str(e))

        if self.method in ['weighted-sum']:
            # [batch_size * n_spans, input_dim]
            fusion = torch.bmm(weights, features).squeeze(1)
        else:  # ['weighted-concat]
            # [batch_size * n_spans, n_feature, input_dim]
            fusion_mat = weights.transpose(-2, -1) * features
            fusion = fusion_mat.view(*fusion_mat.shape[:-2], -1)
        
        return fusion.view(*bat_span_size, *fusion.shape[1:]), weights

    def forward(self, feature_case, module_node, return_alpha=False):
        # ret_tensor a list of [batch_size, input_dim]
        # alpha: [batch_size * n_spans, 1, n_feature]
        ret_tensor, alpha = self.fusion(
            feature_case, module_node=module_node)
        if return_alpha:
            return ret_tensor, alpha
        return ret_tensor


if __name__ == "__main__":
    pass
