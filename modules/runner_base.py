# coding: utf-8
# ==========================================================================
#   Copyright (C) 2016-2021 All rights reserved.
#
#   filename : RunnerBase.py
#   author   : chendian / okcd00@qq.com
#   date     : 2020-07-15
#   desc     : api class as a runner
# ==========================================================================
import re
import sys
import time
import torch
import logging
from torch import nn
from torch.utils.data import DataLoader

from modules.vocab import CharVocabBert
from modules.database import Database
from modules.label_smoothing import LabelSmoothingLoss

from pprint import pprint
from anywhere import get_path, get_cur_time, TARGET_CLASSES


class RunnerBase(object):
    """docstring for Runner."""

    def __init__(self,
                 model_path=None,
                 model_mode=None,
                 device=None,
                 batch_size=16,
                 target_types=TARGET_CLASSES,
                 predict_only=False,
                 dump_model_dir='./data/',
                 runner_id='cc',
                 do_init=True):
        # Basic info initialization
        super(RunnerBase, self).__init__()
        self.classes = target_types
        self.num_class = self.classes.__len__()
        self.best_f = 0.  # current best F1-Score
        self.n_epoch = 0  # current epoch counts
        self.n_update = 0  # current step counts
        self.batch_size = batch_size  # how many sentences in one cc_sample.
        self.predict_only = predict_only  #
        self.skip_save = False
        self.distinguishing_span_sign = False
        self.device = device or torch.device('cuda:0')
        self.runner_id = runner_id  # runner id for naming
        self.model_version = '{}_model_{}'.format(
            runner_id, time.strftime("%y%m%d"))

        self.model = None
        self.ce_loss = None
        self.optimizer = None
        self.scheduler = None
        self.vocab_path = get_path('bert_vocab')
        self.pretrained_path = get_path('pret_dir')
        self.vocab = CharVocabBert(True, self.vocab_path)

        # hyper-params
        self.eps = 1e-8
        self.epoch = 100  # max epoch
        self.lr = 3e-5  # learning rate
        self.dim = 768  # bert's hidden_size is 768
        self.hidden = 512  # default hidden_size for current model
        self.model_path = model_path  # for loading the model
        self.model_mode = model_mode  # custom switches
        self.dump_model_dir = dump_model_dir or './data/'  # for saving the model

        # to be re-assigned in child-classes
        self.train_path = ''
        self.valid_path = ''
        self.test_path = ''
        self.infer_path = ''
        self.model_mode_description = [
            ['mean', 'un1', 'un2', 'un1Aun2'],
            ['self', 'context', 'both']
        ]
        self.default_update_parameters = [
            'bert.encoder.layer.10',
            'bert.encoder.layer.11',
            'bert.pooler',
        ]
        self.default_freeze_parameters = []

        self.sigmoid = torch.nn.Sigmoid()
        self.softmax = torch.nn.Softmax(dim=-1)
        self.mse_loss = torch.nn.MSELoss()
        self.nll_loss = torch.nn.NLLLoss(
            ignore_index=-1, reduction='mean')
        self.ls_loss = LabelSmoothingLoss(
            ignore_index=-1, reduction='mean', smoothing=0.1)
        self.ce_loss = nn.CrossEntropyLoss(
            ignore_index=-1, reduction='mean')
        self.ce_loss_separate = nn.CrossEntropyLoss(
            ignore_index=-1, reduction='none')

        if do_init:
            self.train_loader = self.make_loader(self.train_path)
            self.valid_loader = self.make_loader(self.valid_path)
            self.init_model()
            if self.predict_only:
                self.model.eval()
            else:
                self.model.train()
                self.init_optimizer()

    def init_model(self):
        # load model_class
        # load from self.model_path
        raise NotImplementedError()

    def init_optimizer(self, show_grads=False, method='BertAdam', need_scheduler=False):

        def has_trigger(text, trigger_words):
            return any(tw in text for tw in trigger_words)

        grad_parameters = self.set_requires_grad()
        if show_grads:
            print("grad_parameters:")
            pprint(grad_parameters)
        param_optimizer = list(self.model.named_parameters())

        if method.lower() == 'bertadam':
            from transformers.optimization import BertAdam
            no_decay = ['bias', 'gamma', 'beta']
            optimizer_grouped_parameters = [
                {'params': [p for n, p in param_optimizer
                            if p.requires_grad and not has_trigger(n, no_decay)],
                 'weight_decay_rate': 0.01},
                {'params': [p for n, p in param_optimizer
                            if p.requires_grad and has_trigger(n, no_decay)],
                 'weight_decay_rate': 0.0},
            ]
            self.optimizer = BertAdam(optimizer_grouped_parameters, lr=self.lr)
        elif method.lower() == 'adamw':
            from transformers import AdamW, get_linear_schedule_with_warmup
            optimizer_grouped_parameters = [
                {'params': [p for n, p in param_optimizer
                            if 'bert' in n], 'lr': 1e-5},
                {'params': [p for n, p in param_optimizer
                            if 'bert' not in n], 'lr': 5e-4}]
            self.optimizer = AdamW(optimizer_grouped_parameters, lr=self.lr,
                                   correct_bias=True)
            if need_scheduler:
                warmup_rate = 0.1
                total = self.epoch * len(self.train_loader) // self.batch_size
                # [lr] 0 -> lr when [train-steps] 0 -> warm-up
                # [lr] lr -> 0 when [train-steps] warm-up -> end
                self.scheduler = get_linear_schedule_with_warmup(
                    self.optimizer, int(total * warmup_rate), total)
        else:
            raise ValueError("Invalid optimizer method: {}".format(method))

    def docstring_for_model_mode(self):
        # override this for different tasks.
        if not self.model_mode:
            return []
        return [str(self.model_mode_description[i][m])
                for i, m in enumerate(self.model_mode)]

    def generate_save_path(self, postfix):
        mode = '{}_{}_lr{}'.format(
            'mode-{}'.format('-'.join(self.docstring_for_model_mode())),
            'bat{}'.format(self.batch_size), self.lr)
        save_path = '{}/{}.{}{}.pt'.format(
            self.dump_model_dir,
            self.model_version, mode, postfix)
        return save_path

    def save_model(self, save_path=None, need_optim=False, postfix=''):
        if save_path is None:  # save with original path
            save_path = self.generate_save_path(postfix)
        check_point = {'model': self.model.state_dict(),
                       'epoch': self.n_epoch,
                       'n_steps': self.n_update}
        if need_optim:  # save as cpu tensors.
            """
            optimizer_dict = collections.OrderedDict()
            for state in self.optimizer.state.values():
                for k, v in state.items():
                    if torch.is_tensor(v):
                        optimizer_dict.update({k: v.cpu()})
            """
            check_point['optim'] = self.optimizer.state_dict()
        # if lightweight:
        #     check_point = self.make_model_lightweight(check_point)
        if not self.skip_save:
            torch.save(check_point, save_path)
        print("Now saving model into path: {}".format(save_path))

    def load_state_dict(self, path):
        self.model_path = path
        model_states = torch.load(path, map_location=self.device)
        if 'model' in model_states:
            model_states = model_states.get('model')
        if 'epoch' in model_states:
            self.n_epoch = int(model_states.get('epoch', 0))
        if 'n_steps' in model_states:
            self.n_update = int(model_states.get('n_steps', 0))
        self.model.load_state_dict(model_states)

    def set_requires_grad(self, targets=None, ignores=None, default_no_grad=False):
        grad_parameters = []
        if targets is None:
            targets = self.default_update_parameters
        if ignores is None:
            ignores = self.default_freeze_parameters

        for name, p in self.model.named_parameters():
            if targets and (True in [target in name for target in targets]):
                p.requires_grad = True
                grad_parameters.append(name)
                continue
            if ignores and (True in [ignore in name for ignore in ignores]):
                p.requires_grad = False
                continue

            if default_no_grad:
                p.requires_grad = False
            else:
                p.requires_grad = True
                grad_parameters.append(name)

        return grad_parameters

    def get_optimizer_parameters(self, targets=None, ignores=None, method='parameters'):
        """
        (deprecated)
        show optimizer names or parameters in current model.
        you can also change requires_grad in this function call.
        :param targets:
        :param ignores:
        :param method:
        :return:
        """
        target_keys = []
        target_parameters = []
        para_keys = list(self.model.state_dict().keys())
        target_para_patterns = targets or self.default_update_parameters
        ignore_para_patterns = ignores or self.default_freeze_parameters
        pat_fcs = '|'.join(['({})'.format(p.replace('.', '\.')) for p in target_para_patterns])
        pat_ign = '|'.join(['({})'.format(p.replace('.', '\.')) for p in ignore_para_patterns])
        for idx, _p in enumerate(self.model.parameters()):
            _key = para_keys[idx]
            if re.search(pat_ign, _key):
                _p.requires_grad = False
                continue  # ignore previous transformer
            if re.search(pat_fcs, _key):
                _p.requires_grad = True  # force setting
            target_parameters.append(_p)
            target_keys.append(_key)
        if method.startswith('name'):
            return target_keys
        return target_parameters

    def make_loader(self, path, samples=None):
        """

        :param path:
        :param samples:
        :return:
        """
        if not samples:
            samples = Database(path)

        data_loader = DataLoader(
            dataset=samples,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=0,
            collate_fn=self.collate_fn,
        )
        return data_loader

    @staticmethod
    def get_cur_time(delta=8):
        # C8 has no delta, C14 has 8 hours delta.
        return get_cur_time(delta)

    def print_a_update(self, cost, n_step, n_sample):
        print(' | '.join([
            'Ep {:02}'.format(self.n_epoch),
            'Bat {:>4}'.format(n_step if n_step else 0),
            'Upd {:>4}'.format(self.n_update),
            '{}'.format(self.get_cur_time()),
            # 'PRF.T [{:.3f}|{:.3f}] {:.6f} | Acc.T {:.6f}'.format(*_prf),
            'Sample {}'.format(n_sample),
            'Loss {}'.format(cost),
        ]))
        sys.stdout.flush()

    def left_span_sign(self, temporary_label=None):
        if self.distinguishing_span_sign:
            if temporary_label not in self.classes:
                return '[unused1]'
            sign_index = self.classes.index(temporary_label)
            return '[unused{}1]'.format(str(sign_index) if sign_index else "")
        return '[unused1]'

    def right_span_sign(self, temporary_label=None):
        if self.distinguishing_span_sign:
            if temporary_label not in self.classes:
                return '[unused2]'
            sign_index = self.classes.index(temporary_label)
            return '[unused{}2]'.format(str(sign_index) if sign_index else "")
        return '[unused2]'

    def collate_fn(self, samples):
        raise NotImplementedError()

    def predict(self, input_data):
        # self.model.eval()
        raise NotImplementedError()

    def train(self):
        raise NotImplementedError()

    def run(self):
        self.train()
        return


logging.basicConfig(
    level=logging.INFO,
    format='[%(process)d %(filename)s %(lineno)d]:  %(message)s')

if __name__ == '__main__':
    _mode = [3, 2]
    if len(sys.argv) > 1:
        _mode[0] = int(sys.argv[1])
        _mode[1] = int(sys.argv[2])
    a = RunnerBase(model_mode=_mode)
    a.run()
