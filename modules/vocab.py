# coding=utf8
"""
字典的基础类
"""
import os
import pickle
import logging
from io import open
from collections import Counter

# 项目根目录
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))  


class VocabBase(object):
    """
    Vocab基础类
    Vocab 中必须包含以下属性：
    unknown_index, 如果词汇超过词表大小，用哪个序号表示
    """
    unknown_index = None
    vocab_size = 1000000

    def __init__(self, vocab_size, voc_path, load=True):
        self.idx2word, self.word2idx = None, None
        pass

    def to_index(self, tokens, **kwargs):
        """
        tokens 是一个 list
        返回一个list，其值是每个 token 在词表中的 index 序号
        """
        assert isinstance(tokens, list)
        raise NotImplementedError()

    def save_as_txt(self, save_path, voc_with_tags, voc_count):
        with open(save_path, 'wb') as fwrite:
            for word in voc_with_tags:
                count = 0
                if word in voc_count:
                    count = voc_count[word]
                fwrite.write(u'{}: \t{}\n'.format(word, count).encode('utf8'))
        print('vocabulary has {} words'.format(len(voc_with_tags)))
        print('vocabulary save to txt: ' + save_path)

    def stem_lemmatize(self, sample):
        """ 把单词进行 词干化、屈折变化形式（或异体形式）进行归类
        :param sample: utie sample
        :return result: [unicode, ]
        """
        raise NotImplementedError()
        # return [w['word'] for w in sample['words']]

    def voc_count_organize(self, voc_count):
        """ 可以在这里对 voc_count 进行一些修改。
        比如使某些词的词频变大，
        删除停用词等
        :param voc_count: collections.Counter
        """
        return voc_count

    def build_vocabulary(self, dataset, special_tags, save_path, smallest_count=5, max_size=None):
        """
        get the vocabulary of the dataset
        vocabulary 应该是一个全局性的东西，train/valid/test使用同一个vocabulary
        vocabulary 按照词频排序
        vocabulary 的index 从 0 开始

        :param dataset:
        :param special_tags: list, 特殊字符列表，第一个代表 UNKNOWN，会加到最终Vocab的最开头
        :param save_path:
        :param smallest_count:
        :param max_size: 词典最大包含多少个词
        :return:
        """
        if smallest_count and max_size:
            logging.warn(u'You provide both smallest_count and max_size, '
                         u'we will make a voc that satisfies both '
                         u'(which means the smaller vocabulary)')
        assert len(special_tags), 'at least UNKNOWN tag should be provided'
        voc_count = Counter()
        for sample in dataset:
            words = self.stem_lemmatize(sample)
            voc_count.update(words)

        print('{} sentence in build_vocabulary'.format(len(dataset)))
        for tag in special_tags:
            print(u'{}: {}'.format(tag, voc_count.pop(tag, None)))
        voc_count = self.voc_count_organize(voc_count)
        voc_count_sorted = voc_count.most_common(max_size)
        voc_without_tags = [w[0] for w in voc_count_sorted if w[1] > smallest_count]
        voc_with_tags = special_tags + voc_without_tags

        self.idx2word = {i: voc_with_tags[i] for i in range(len(voc_with_tags))}
        self.word2idx = {voc_with_tags[i]: i for i in range(len(voc_with_tags))}
        variables = {'idx2word': self.idx2word, 'word2idx': self.word2idx, 'word_count': voc_count}
        self.save_as_txt(save_path + '.txt', voc_with_tags, voc_count)
        pickle.dump(variables, open(save_path, 'wb'), protocol=2)
        logging.info('{} words with frequency > {}'.format(len(self.idx2word), smallest_count))
        logging.info('save in: {}'.format(save_path))
        logging.info('you can view word count in file: {}'.format(save_path + '.txt'))
        return self.idx2word, self.word2idx


class BasicVocab(VocabBase):

    def __init__(self, vocab_size, voc_path, load=True):
        if not load:
            return
        print(u'voc_path: {}'.format(voc_path))
        self.idx2word, self.word2idx = None, None
        self.load(voc_path)
        self.unknown_index = self.word2idx['$UNK$']
        self.bos_index = self.word2idx['$BOS$']
        self.eos_index = self.word2idx['$EOS$']
        self.vocab_size = vocab_size
        self.trunc_voc()

    def stem_lemmatize(self, sample):
        return [w['word'] for w in sample['words']]

    def trunc_voc(self):
        for word in list(self.word2idx.keys()):
            if self.word2idx[word] >= self.vocab_size:
                self.word2idx.pop(word)

    def load(self, voc_path):
        if '.' not in voc_path and '/' not in voc_path:
            # we think it is a key, instead of a path
            pass
        else:
            if not os.path.isabs(voc_path):
                voc_path = os.path.join(project_root, voc_path)
        with open(voc_path, 'rb') as fr:
            voc = pickle.load(fr)
        self.idx2word, self.word2idx = voc['idx2word'], voc['word2idx']
        return self.idx2word, self.word2idx

    def to_index(self, tokens, **kwargs):
        return [self.word2idx.get(t, self.unknown_index) for t in tokens]


class CharVocabBert(VocabBase):

    def __init__(self, vocab_size, vocab_path=None, load=True):
        super(CharVocabBert, self).__init__(vocab_size=vocab_size, voc_path=vocab_path, load=load)
        if not load:
            raise ValueError('Bert vocab is fixed')
        if vocab_size:
            logging.warn('Setting vocab_size')
        self.vocab = self.load(vocab_path)
        self.eos_index = self.vocab.vocab['[unused1]']
        self.bos_index = self.vocab.vocab['[unused2]']
        self.unknown_index = self.vocab.vocab['[UNK]']

    def tokenize(self, text):
        return self.vocab.tokenize(text)

    def to_index(self, tokens, **kwargs):
        return self.vocab.convert_tokens_to_ids(tokens)

    def to_tokens(self, ids):
        return self.vocab.convert_ids_to_tokens(ids)

    @staticmethod
    def load(vocab_path):
        from transformers import BertTokenizer as FullTokenizer
        return FullTokenizer(vocab_file=vocab_path, do_lower_case=True)

    def stem_lemmatize(self, sample):
        raise ValueError('Bert vocab is fixed')


def main():
    import argparse
    from modules.database import Database
    parser = argparse.ArgumentParser()
    parser.add_argument('data_path')
    parser.add_argument('save_path')
    parser.add_argument('--min_count', dest='min_count', default=5, type=int)
    parser = parser.parse_args()
    vocab = BasicVocab(0, 0, load=False)
    dataset = Database(parser.data_path)
    special_tags = ['$UNK$', '$BOS$', '$EOS$', '$SPEC1$', '$SPEC2$', '$SPEC3$']
    vocab.build_vocabulary(dataset, special_tags, parser.save_path, smallest_count=parser.min_count)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='[Func:%(funcName)s %(lineno)d]:  %(message)s')
    main()
