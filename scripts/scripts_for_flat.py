# coding: utf-8
# ==========================================================================
#   Copyright (C) 2016-2021 All rights reserved.
#
#   filename : flat_utils.py
#   author   : chendian / okcd00@qq.com
#   date     : 2021-01-31
#   desc     : need fastNLP module
# ==========================================================================
from data_utils import *
import os, sys
os.environ['CUDA_VISIBLE_DEVICES'] = '5'
sys.path.append('/home/chendian/Flat-Lattice-Transformer/')

import torch
from paths import *
from load_data import *
from fastNLP import Vocabulary
from fastNLP import cache_results
from fastNLP_module import StaticEmbedding
from anywhere import (
    yangjie_rich_pretrain_word_path, 
    yangjie_rich_pretrain_unigram_path,
    yangjie_rich_pretrain_bigram_path)


class FakeArgs(object):
    def __init__(self):
        self.dic = dict()


@cache_results(_cache_fp='cache/training', _refresh=False)
def load_train_ner(path, char_embedding_path=None, bigram_embedding_path=None, index_token=True, train_clip=False,
                    char_min_freq=1, bigram_min_freq=1, only_train_min_freq=0):
    from fastNLP.io.loader import ConllLoader
    from basic_utils import get_bigrams
    if train_clip:
        train_path = os.path.join(path, 'train.char.bmes_clip1')
        dev_path = os.path.join(path, 'dev.char.bmes_clip1')
        test_path = os.path.join(path, 'test.char.bmes_clip1')
    else:
        train_path = os.path.join(path, 'train.char.bmes')
        dev_path = os.path.join(path, 'dev.char.bmes')
        test_path = os.path.join(path, 'test.char.bmes')

    loader = ConllLoader(['chars', 'target'])
    train_bundle = loader.load(train_path)
    dev_bundle = loader.load(dev_path)
    test_bundle = loader.load(test_path)

    datasets = dict()
    datasets['train'] = train_bundle.datasets['train']
    datasets['dev'] = dev_bundle.datasets['train']
    datasets['test'] = test_bundle.datasets['train']

    datasets['train'].apply_field(get_bigrams, field_name='chars', new_field_name='bigrams')
    datasets['dev'].apply_field(get_bigrams, field_name='chars', new_field_name='bigrams')
    datasets['test'].apply_field(get_bigrams, field_name='chars', new_field_name='bigrams')

    datasets['train'].add_seq_len('chars')
    datasets['dev'].add_seq_len('chars')
    datasets['test'].add_seq_len('chars')

    char_vocab = Vocabulary()
    bigram_vocab = Vocabulary()
    label_vocab = Vocabulary()
    print(datasets.keys())
    print(len(datasets['dev']))
    print(len(datasets['test']))
    print(len(datasets['train']))

    char_vocab.from_dataset(datasets['train'], field_name='chars',
                            no_create_entry_dataset=[datasets['dev'], datasets['test']])
    bigram_vocab.from_dataset(datasets['train'], field_name='bigrams',
                              no_create_entry_dataset=[datasets['dev'], datasets['test']])
    label_vocab.from_dataset(datasets['train'], field_name='target')
    if index_token:
        char_vocab.index_dataset(datasets['train'], datasets['dev'], datasets['test'],
                                 field_name='chars', new_field_name='chars')
        bigram_vocab.index_dataset(datasets['train'], datasets['dev'], datasets['test'],
                                   field_name='bigrams', new_field_name='bigrams')
        label_vocab.index_dataset(datasets['train'], datasets['dev'], datasets['test'],
                                  field_name='target', new_field_name='target')

    vocabs = {}
    vocabs['char'] = char_vocab
    vocabs['label'] = label_vocab
    vocabs['bigram'] = bigram_vocab
    vocabs['label'] = label_vocab

    embeddings = {}
    if char_embedding_path is not None:
        char_embedding = StaticEmbedding(char_vocab, char_embedding_path, word_dropout=0.01,
                                         min_freq=char_min_freq, only_train_min_freq=only_train_min_freq)
        embeddings['char'] = char_embedding

    if bigram_embedding_path is not None:
        bigram_embedding = StaticEmbedding(bigram_vocab, bigram_embedding_path, word_dropout=0.01,
                                           min_freq=bigram_min_freq, only_train_min_freq=only_train_min_freq)
        embeddings['bigram'] = bigram_embedding

    return datasets, vocabs, embeddings


@cache_results(_cache_fp='cache/testing', _refresh=False)
def load_test_ner(path, char_embedding_path=None, bigram_embedding_path=None, index_token=True, train_clip=False,
                    char_min_freq=1, bigram_min_freq=1, only_train_min_freq=0):
    from fastNLP.io.loader import ConllLoader
    from basic_utils import get_bigrams
    if train_clip:
        test_path = os.path.join(path, 'test.char.bmes_clip1')
    else:
        test_path = os.path.join(path, 'test.char.bmes')

    loader = ConllLoader(['chars', 'target'])
    test_bundle = loader.load(test_path)

    datasets = dict()
    datasets['test'] = test_bundle.datasets['train']
    datasets['test'].apply_field(get_bigrams, field_name='chars', new_field_name='bigrams')
    datasets['test'].add_seq_len('chars')

    char_vocab = Vocabulary()
    bigram_vocab = Vocabulary()
    label_vocab = Vocabulary()

    datasets['dev'] = []
    datasets['train'] = []

    print(datasets.keys())
    print(len(datasets['test']))

    char_vocab.from_dataset(datasets['train'], field_name='chars',
                            no_create_entry_dataset=[datasets['dev'], datasets['test']])
    bigram_vocab.from_dataset(datasets['train'], field_name='bigrams',
                              no_create_entry_dataset=[datasets['dev'], datasets['test']])
    label_vocab.from_dataset(datasets['train'], field_name='target')
    if index_token:
        char_vocab.index_dataset(datasets['train'], datasets['dev'], datasets['test'],
                                 field_name='chars', new_field_name='chars')
        bigram_vocab.index_dataset(datasets['train'], datasets['dev'], datasets['test'],
                                   field_name='bigrams', new_field_name='bigrams')
        label_vocab.index_dataset(datasets['train'], datasets['dev'], datasets['test'],
                                  field_name='target', new_field_name='target')

    vocabs = {}
    vocabs['char'] = char_vocab
    vocabs['label'] = label_vocab
    vocabs['bigram'] = bigram_vocab
    vocabs['label'] = label_vocab

    embeddings = {}
    if char_embedding_path is not None:
        char_embedding = StaticEmbedding(char_vocab, char_embedding_path, word_dropout=0.01,
                                         min_freq=char_min_freq, only_train_min_freq=only_train_min_freq)
        embeddings['char'] = char_embedding

    if bigram_embedding_path is not None:
        bigram_embedding = StaticEmbedding(bigram_vocab, bigram_embedding_path, word_dropout=0.01,
                                           min_freq=bigram_min_freq, only_train_min_freq=only_train_min_freq)
        embeddings['bigram'] = bigram_embedding

    return datasets, vocabs, embeddings


def load_flat_datasets(dataset_name='resume', ner_path=None, refresh_data=False):
    args = FakeArgs()
    args.dataset = dataset_name
    load_dataset_seed = 100
    args.number_normalized = 3

    args.lattice = 1
    args.use_bert = 1
    args.train_clip = True
    args.char_min_freq = 1
    args.word_min_freq = 1
    args.bigram_min_freq = 1
    args.lattice_min_freq = 1
    args.only_train_min_freq = 1
    args.only_train_min_freq = True
    args.only_lexicon_in_train = False
    args.lexicon_name = 'yj'

    raw_dataset_cache_name = os.path.join('cache',
                                          args.dataset + '_trainClip:{}'.format(args.train_clip)
                                          + 'bgminfreq_{}'.format(args.bigram_min_freq)
                                          + 'char_min_freq_{}'.format(args.char_min_freq)
                                          + 'word_min_freq_{}'.format(args.word_min_freq)
                                          + 'only_train_min_freq{}'.format(args.only_train_min_freq)
                                          + 'number_norm{}'.format(args.number_normalized)
                                          + 'load_dataset_seed{}'.format(load_dataset_seed))

    if args.dataset == 'resume':
        datasets, vocabs, embeddings = load_resume_ner(resume_ner_path,
                                                       yangjie_rich_pretrain_unigram_path,
                                                       yangjie_rich_pretrain_bigram_path,
                                                       _refresh=refresh_data,
                                                       index_token=False,
                                                       _cache_fp=raw_dataset_cache_name,
                                                       char_min_freq=args.char_min_freq,
                                                       bigram_min_freq=args.bigram_min_freq,
                                                       only_train_min_freq=args.only_train_min_freq
                                                       )
    elif args.dataset == 'msra':
        datasets, vocabs, embeddings = load_msra_ner_1(msra_ner_cn_path,
                                                       yangjie_rich_pretrain_unigram_path,
                                                       yangjie_rich_pretrain_bigram_path,
                                                       _refresh=refresh_data,
                                                       index_token=False,
                                                       train_clip=args.train_clip,
                                                       _cache_fp=raw_dataset_cache_name,
                                                       char_min_freq=args.char_min_freq,
                                                       bigram_min_freq=args.bigram_min_freq,
                                                       only_train_min_freq=args.only_train_min_freq
                                                       )

    elif args.dataset == 'nc':
        datasets, vocabs, embeddings = load_nc_ner(nc_ner_path, yangjie_rich_pretrain_unigram_path,
                                                   yangjie_rich_pretrain_bigram_path,
                                                   _refresh=refresh_data, index_token=False, train_clip=args.train_clip,
                                                   _cache_fp=raw_dataset_cache_name,
                                                   char_min_freq=args.char_min_freq,
                                                   bigram_min_freq=args.bigram_min_freq,
                                                   only_train_min_freq=args.only_train_min_freq
                                                   )
    elif args.dataset.startswith('train'):
        # use msra form
        # test_path = '/home/chendian/flat_files/testing'
        datasets, vocabs, embeddings = load_train_ner(ner_path,
                                                     yangjie_rich_pretrain_unigram_path,
                                                     yangjie_rich_pretrain_bigram_path,
                                                     _refresh=refresh_data,
                                                     index_token=False,
                                                     train_clip=args.train_clip,
                                                     _cache_fp=raw_dataset_cache_name,
                                                     char_min_freq=args.char_min_freq,
                                                     bigram_min_freq=args.bigram_min_freq,
                                                     only_train_min_freq=args.only_train_min_freq
                                                     )
    elif args.dataset.startswith('test'):
        # use msra form
        # test_path = '/home/chendian/flat_files/testing'
        datasets, vocabs, embeddings = load_test_ner(ner_path,
                                                     yangjie_rich_pretrain_unigram_path,
                                                     yangjie_rich_pretrain_bigram_path,
                                                     _refresh=refresh_data,
                                                     index_token=False,
                                                     train_clip=args.train_clip,
                                                     _cache_fp=raw_dataset_cache_name,
                                                     char_min_freq=args.char_min_freq,
                                                     bigram_min_freq=args.bigram_min_freq,
                                                     only_train_min_freq=args.only_train_min_freq
                                                     )

    from V1.add_lattice import equip_chinese_ner_with_lexicon
    print('用的词表的路径:{}'.format(yangjie_rich_pretrain_word_path))

    w_list = load_yangjie_rich_pretrain_word_list(yangjie_rich_pretrain_word_path,
                                                  _refresh=refresh_data,
                                                  _cache_fp='cache/{}'.format(args.lexicon_name))

    cache_name = os.path.join('cache', (args.dataset + '_lattice' + '_only_train:{}' +
                                        '_trainClip:{}' + '_norm_num:{}'
                                        + 'char_min_freq{}' + 'bigram_min_freq{}' + 'word_min_freq{}' + 'only_train_min_freq{}'
                                        + 'number_norm{}' + 'lexicon_{}' + 'load_dataset_seed_{}')
                              .format(args.only_lexicon_in_train,
                                      args.train_clip, args.number_normalized, args.char_min_freq,
                                      args.bigram_min_freq, args.word_min_freq, args.only_train_min_freq,
                                      args.number_normalized, args.lexicon_name, load_dataset_seed))

    datasets, vocabs, embeddings = equip_chinese_ner_with_lexicon(datasets, vocabs, embeddings,
                                                                  w_list, yangjie_rich_pretrain_word_path,
                                                                  _refresh=refresh_data, _cache_fp=cache_name,
                                                                  only_lexicon_in_train=args.only_lexicon_in_train,
                                                                  word_char_mix_embedding_path=yangjie_rich_pretrain_char_and_word_path,
                                                                  number_normalized=args.number_normalized,
                                                                  lattice_min_freq=args.lattice_min_freq,
                                                                  only_train_min_freq=args.only_train_min_freq)
    return datasets, vocabs, embeddings


def load_flat_model_from_path(model_path):
    model = torch.load(model_path).to(torch.device('cuda:0'))
    return model


def predict_with_flat_model(model, sample, vocabs=None):
    # sample = datasets['test'][11]
    # print(datasets['test'][11].items())
    device = torch.device('cuda:0')
    params = dict((k, torch.tensor(v, device=device).unsqueeze(0))
                  for (k, v) in sample.items() if k in [
                      'lattice', 'bigrams', 'seq_len', 'lex_num', 'pos_s', 'pos_e', 'target'])
    model.training = False
    result = model(**params)
    tags = result['pred'][0].cpu().numpy().tolist()
    if vocabs:
        tag_vocab = vocabs['label']
        tags = [tag_vocab.to_word(idx) for idx in tags]
    return tags


if __name__ == '__main__':
    pass
