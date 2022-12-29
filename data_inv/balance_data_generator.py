# coding: utf-8
# ==========================================================================
#   Copyright (C) 2020 All rights reserved.
#
#   filename : balance_data_generator.py
#   author   : chendian / okcd00@qq.com
#   date     : 2021-01-16
#   desc     : generate training data group by generating methods.
#              better design for balanced positive/negative samples.
# ==========================================================================
from data_generator import DataGenerator


class BalanceDataGenerator(object):
    def __init__(self,
                 database_path_positive,
                 database_path_negative,
                 read_only=True,
                 samples=None, n_samples=None, decay=1.0,
                 random_method='take', unsupervised_alpha=False,
                 name='train', load_now=False):

        # foundry-like FolderDB for single json samples.
        self.positive_loader = DataGenerator(
            database_path_positive, read_only=read_only,
            samples=samples, n_samples=n_samples, decay=decay,
            random_method=random_method, 
            unsupervised_alpha=unsupervised_alpha,
            name=name, load_now=load_now)

        self.negative_loader = DataGenerator(
            database_path_negative, read_only=read_only,
            samples=samples, n_samples=n_samples, decay=decay,
            random_method=random_method, 
            unsupervised_alpha=unsupervised_alpha,
            name=name, load_now=load_now)

        self.dispatch = self.positive_loader.dispatch
        self.function_dict = self.positive_loader.function_dict

        self.iter_n = 0
        self.pos_offset = 0
        self.neg_offset = 0
        self.aug_decay = decay

        self.balance_length = min(
            self.positive_loader.__len__(),
            self.negative_loader.__len__())

        print("get balance_length as {} from positive: {} & negative: {}".format(
            self.balance_length,
            self.positive_loader.__len__(),
            self.negative_loader.__len__()
        ))

    def reset_augment_probability(self):
        self.positive_loader.aug_line = 1.0
        self.negative_loader.aug_line = 1.0

    def augment_probability_decay(self, decay=None):
        if decay is None:
            decay = self.aug_decay
        self.positive_loader.aug_line *= decay
        self.negative_loader.aug_line *= decay

    def __len__(self):
        return self.balance_length * 2
        # return self.positive_loader.__len__() + self.negative_loader.__len__()

    def __getitem__(self, item):
        # hook the generate function in __getitem__
        def _slice(key):
            start, stop, step = key.indices(self.__len__())
            for i in range(start, stop, step):
                yield self.__getitem__(i)

        if isinstance(item, slice):
            return _slice(item)

        if item % 2 == 0:
            return self.positive_loader[item // 2 + self.pos_offset]
        else:
            return self.negative_loader[item // 2 + self.neg_offset]

    def get_by_sid(self, sid):
        positive_sample = None
        negative_sample = None
        try:
            positive_sample = self.positive_loader.get_by_sid(sid)
            negative_sample = self.negative_loader.get_by_sid(sid)
        except Exception as e:
            pass
        return positive_sample or negative_sample

    def offset_update(self):
        self.pos_offset = (self.pos_offset + self.balance_length) % self.positive_loader.__len__()
        self.neg_offset = (self.neg_offset + self.balance_length) % self.negative_loader.__len__()
        print("data-loaders' offset changed:", self.pos_offset, self.neg_offset)

    def __iter__(self):
        self.iter_n = 0
        return self

    def __next__(self):
        if self.iter_n == self.__len__():
            # self.offset_update()
            raise StopIteration
        n = self.iter_n
        self.iter_n += 1
        return self[n]

    @property
    def all_samples(self):
        """return all samples in this dataset"""
        cur_sample_list = [self[i] for i in range(len(self))]
        self.offset_update()
        return cur_sample_list


if __name__ == "__main__":
    pass
