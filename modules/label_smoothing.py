# coding: utf-8
# ==========================================================================
#   Copyright (C) 2016-2021 All rights reserved.
#
#   filename : LabelSmoothingLoss.py
#   author   : chendian / okcd00@qq.com
#   date     : 2020-12-05
#   desc     :
# ==========================================================================
import torch
from torch import nn


class LabelSmoothingLoss(nn.Module):
    """
    NLL loss with label smoothing.
    """
    def __init__(self, smoothing=0.1, reduction='mean', ignore_index=-1):
        """
        Constructor for the LabelSmoothing module.
        :param smoothing: label smoothing factor
        """
        super(LabelSmoothingLoss, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.ignore_index = ignore_index
        self.nll_loss = nn.NLLLoss(
            ignore_index=ignore_index, reduction='none')
        self.reduction = reduction

    def forward(self, input, target, skip_log_softmax=False):
        if skip_log_softmax:
            log_prob = input
        else:
            log_prob = torch.nn.functional.log_softmax(input, dim=-1)

        nll_loss = self.nll_loss(log_prob, target)
        smooth_loss = -log_prob.mean(dim=-1)
        loss = self.confidence * nll_loss + self.smoothing * smooth_loss

        if self.reduction == 'sum':
            return loss.sum()
        # as default
        return loss.mean()


class LabelSmoothingLossV2(nn.Module):
    def __init__(self, classes, smoothing=0.0, dim=-1):
        super(LabelSmoothingLossV2, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.cls = classes
        self.dim = dim

    def forward(self, pred, target):
        pred = pred.log_softmax(dim=self.dim)
        with torch.no_grad():
            # true_dist = pred.data.clone()
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (self.cls - 1))
            true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        return torch.mean(torch.sum(-true_dist * pred, dim=self.dim))


class LabelSmoothingLossV3(nn.Module):
    def __init__(self, smoothing=0.0):
        super(LabelSmoothingLossV3, self).__init__()
        self.smoothing = smoothing

    def forward(self, input, target):
        log_prob = torch.nn.functional.log_softmax(input, dim=-1)
        # print(torch.exp(log_prob))
        weight = input.new_ones(input.size()) * \
                 self.smoothing / (input.size(-1) - 1.)
        weight.scatter_(-1, target.unsqueeze(-1), (1. - self.smoothing))
        loss = (-weight * log_prob).sum(dim=-1).mean()
        return loss


class LabelSmoothSoftmaxCEV1(nn.Module):
    '''
    This is the autograd version, you can also try the LabelSmoothSoftmaxCEV2 that uses derived gradients
    '''

    def __init__(self, lb_smooth=0.1, reduction='mean', ignore_index=-100):
        super(LabelSmoothSoftmaxCEV1, self).__init__()
        self.lb_smooth = lb_smooth
        self.reduction = reduction
        self.lb_ignore = ignore_index
        self.log_softmax = nn.LogSoftmax(dim=1)

    def forward(self, logits, label):
        '''
        args: logits: tensor of shape (N, C, H, W)
        args: label: tensor of shape(N, H, W)
        '''
        # overcome ignored label
        with torch.no_grad():
            num_classes = logits.size(1)
            label = label.clone().detach()
            ignore = label == self.lb_ignore
            n_valid = (ignore == 0).sum()
            label[ignore] = 0
            lb_pos, lb_neg = 1. - self.lb_smooth, self.lb_smooth / num_classes
            label = torch.empty_like(logits).fill_(
                lb_neg).scatter_(1, label.unsqueeze(1), lb_pos).detach()

        logs = self.log_softmax(logits)
        loss = -torch.sum(logs * label, dim=1)
        loss[ignore] = 0
        if self.reduction == 'mean':
            loss = loss.sum() / n_valid
        if self.reduction == 'sum':
            loss = loss.sum()

        return loss


class LSRCrossEntropyFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, logits, label, lb_smooth, reduction, lb_ignore):
        # prepare label
        num_classes = logits.size(1)
        label = label.clone().detach()
        ignore = label == lb_ignore
        n_valid = (ignore == 0).sum()
        label[ignore] = 0
        lb_pos, lb_neg = 1. - lb_smooth, lb_smooth / num_classes
        label = torch.empty_like(logits).fill_(
            lb_neg).scatter_(1, label.unsqueeze(1), lb_pos).detach()

        ignore = ignore.nonzero()
        _, M = ignore.size()
        a, *b = ignore.chunk(M, dim=1)
        mask = [a, torch.arange(label.size(1)), *b]
        label[mask] = 0

        coeff = (num_classes - 1) * lb_neg + lb_pos
        ctx.coeff = coeff
        ctx.mask = mask
        ctx.logits = logits
        ctx.label = label
        ctx.reduction = reduction
        ctx.n_valid = n_valid

        loss = torch.log_softmax(logits, dim=1).neg_().mul_(label).sum(dim=1)
        if reduction == 'mean':
            loss = loss.sum().div_(n_valid)
        if reduction == 'sum':
            loss = loss.sum()
        return loss

    @staticmethod
    def backward(ctx, grad_output):
        coeff = ctx.coeff
        mask = ctx.mask
        logits = ctx.logits
        label = ctx.label
        reduction = ctx.reduction
        n_valid = ctx.n_valid

        scores = torch.softmax(logits, dim=1).mul_(coeff)
        scores[mask] = 0
        grad = None
        if reduction == 'none':
            grad = scores.sub_(label).mul_(grad_output.unsqueeze(1))
        elif reduction == 'sum':
            grad = scores.sub_(label).mul_(grad_output)
        elif reduction == 'mean':
            grad = scores.sub_(label).mul_(grad_output.div_(n_valid))
        return grad, None, None, None, None, None


class LabelSmoothSoftmaxCEV2(nn.Module):
    def __init__(self, lb_smooth=0.1, reduction='mean', ignore_index=-100):
        super(LabelSmoothSoftmaxCEV2, self).__init__()
        self.lb_smooth = lb_smooth
        self.reduction = reduction
        self.lb_ignore = ignore_index

    def forward(self, logits, label):
        return LSRCrossEntropyFunction.apply(
            logits, label,
            self.lb_smooth,
            self.reduction,
            self.lb_ignore)


def test_for_no_smoothing(class_):
    ls_loss = class_(smoothing=0.0)
    ce_loss = nn.CrossEntropyLoss(
        ignore_index=-1, reduction='mean')

    # exp(log_prob): tensor([[1.0000e+00, 2.7895e-10]])
    outputs = torch.tensor([[11., -11.]])
    labels = torch.tensor([0]).long()

    loss = ls_loss(outputs, labels)
    print(loss, ce_loss(outputs, labels))
    # tensor(0.)

def test_for_just_smoothing(class_):
    ls_loss = class_(smoothing=0.1)
    ce_loss = nn.CrossEntropyLoss(
        ignore_index=-1, reduction='mean')

    # exp(log_prob): tensor([[0.9000, 0.1000]])
    outputs = torch.tensor([[1.0987, -1.0987]])
    # exp(log_prob): tensor([[0.9500, 0.1500]])
    # outputs = torch.tensor([[1.472, -1.472]])

    labels = torch.tensor([0]).long()
    loss = ls_loss(outputs, labels)
    print(loss, ce_loss(outputs, labels))
    # LabelSmoothingLoss: (tensor(0.2152), tensor(0.1053))
    # LabelSmoothingLossV2: (tensor(0.3251), tensor(0.1053))
    # LabelSmoothingLossV3: (tensor(0.3251), tensor(0.1053))
    # LabelSmoothSoftmaxCEV1: (tensor(0.2099), tensor(0.1053))
    # LabelSmoothSoftmaxCEV2: (tensor(0.2099), tensor(0.1053))


if __name__ == "__main__":
    pass
