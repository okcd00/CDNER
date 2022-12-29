# coding: utf-8
# ==========================================================================
#   Copyright (C) 2020 All rights reserved.
#
#   filename : data_generation.py
#   author   : chendian / okcd00@qq.com
#   date     : 2020-11-23
#   desc     : generate three kinds of training data for different sub-tasks
#              from one single data-set. (deprecated since 2020-12-18)
# ==========================================================================
from data_utils import *
from utie.vocab import CharVocabBert


class DataGenerator(Database):
    def __init__(self, database_path, read_only=True,
                 samples=None, n_samples=None, load_now=False, do_shift=True,
                 randomly_generating=True, generating_methods="123"):
        # foundry-like FolderDB for single json samples.
        self.database_path = database_path
        super().__init__(
            path=database_path, samples=samples, n_samples=n_samples,
            read_only=read_only, load_now=load_now)
        self.do_shift = do_shift
        self.entity_case = None
        self.randomly_generating = randomly_generating
        self.generating_methods = generating_methods
        self.vocab = CharVocabBert(True, get_path('bert_vocab'))
        self.sample_counts = int(self.__len__())
        self.init_generator_library()

    @staticmethod
    def determine_mode(label_path=None):
        # fixed DB type for super().__init__()
        mode = 'one_sample_per_file'  # class Folder as default
        # directory path without postfix
        return mode

    def init_generator_library(self):
        entity_library_path = self.database_path.rstrip('/') + '.elib'
        if os.path.exists(entity_library_path):
            self.entity_case = pickle.load(open(entity_library_path, 'rb'))
            return  # load instead of re-generating
        print("Now generating entity_library for database {}. {}".format(
            self.database_path, get_cur_time(8)))
        self.entity_case = {_type: set() for _type in TARGET_CLASSES}
        for s_idx, sample in tqdm(enumerate(self)):
            words = sample['words']
            for ent in sample['entities']:
                ent_str = get_string_from_entity(entity=ent, words=words)
                ent_type = ent['type']
                self.entity_case[ent_type].add(ent_str)
        self.entity_case = {k: list(v) for k, v in self.entity_case.items()}
        pickle.dump(self.entity_case, open(entity_library_path, 'wb'))

    def is_known_entity(self, text):
        for key in self.entity_case.keys():
            if text in self.entity_case[key]:
                return True
        return False

    @staticmethod
    def reid(sample):
        # re-construct id for words/entities/relations in this sample.
        sample, old2new = re_id_func(sample)
        return sample

    @staticmethod
    def dummy_random_sample():
        # show how a sample looks like
        sample = {
            'info': {'sid': 'test_{}'.format(random.randint(1, 100)),
                     'doc_id': '-1'},
            'alpha': 0.,  # -1 or a float number from 0. to 1.
            'words': [{'id': str(i), 'word': c}
                      for i, c in enumerate('庖丁科技是一家科技公司')],
            'entities': [{'id': 'relation-0', 'type': '公司',
                          'tokens': ['0', '1', '2', '3']}],
            'relations': [],
        }
        return sample

    def random_n_gram(self, min_length=2, max_length=6, method='vocab', foundry_form=False):
        if method.startswith('vocab'):
            # return a list of random characters
            char_list = random.choices(
                SINGLE_CHAR_VOCAB_LIST,
                k=random.randint(min_length, max_length))
        elif method.startswith('take'):
            sent_index = random.randint(0, self.sample_counts - 1)
            gram_length = random.randint(min_length, max_length)
            sent_words = self[sent_index]['words']
            end_position = len(sent_words) - gram_length
            if end_position < 0:  # re-select a sentence
                return self.random_n_gram(
                    min_length=min_length, max_length=max_length,
                    method=method, foundry_form=foundry_form)
            word_index = random.randint(0, end_position)
            char_list = [sent_words[_idx]['word']
                         for _idx in range(word_index, word_index+gram_length)]
        else:  # return any key in vocab
            char_list = random.choices(
                VOCAB_LIST,
                k=random.randint(min_length, max_length))

        if foundry_form:
            return [{'id': 'rd-{}'.format(i), 'word': c}
                    for i, c in enumerate(char_list)]

        return char_list

    def random_ent_str(self, entity_type):
        return random.choice(self.entity_case[entity_type])

    def post_process(self, sample):
        if sample is None:
            # we better drop all invalid samples in pre-processing
            return None
        # re-construct the ids
        sample = self.reid(sample)
        # re-generate the text
        sample = add_text_key_in_sample(sample)
        # re-calculate the span info
        sample = add_span_in_sample(sample)
        return sample

    def get_entity(self, sample, generate=False, entity_type=None):
        """
        get the entity from sample
        :param sample: target sample
        :param generate: if the entity does not exist,
                         generate an empty_entity at random position.
        :return: the entity in target sample, or generated one (might be non-entity label).
        """
        if sample['entities']:
            return sample['entities'][0]
        # there's no entities in this sample.
        if generate:
            words_length = sample['words'].__len__()
            random_index = random.randint(0, words_length - 1)
            # default dummy_entity is an empty string.
            sample['entities'] = [{
                'id': 'dummy-0',
                'tokens': [],
                'type': entity_type,
                'span': (random_index, random_index),
            }]
            return sample['entities'][0]
        return None

    def replace_entity_in_sample(self, sample, entity_str, entity_type=None):
        """
        replace the entity in current sample with another entity_string.
        :param sample: a copy of original sample
        :param entity_str: the entity used to be added in this sample
        :param entity_type:
        :return: a sample with different entity_str but the same type_label
        """
        # remain original entity type.
        entity_words = [{'word': c, 'id': 'rd-{}'.format(i)}
                        for i, c in enumerate(entity_str)]

        origin_entity = self.get_entity(sample)
        # replace original tokens with new tokens
        # if no existing entity, it acts the same as insertion.
        left, right = origin_entity['span']
        new_words = sample['words'][:left] + entity_words + \
                    sample['words'][right:]

        sample['words'] = new_words
        origin_entity['tokens'] = [w['id'] for w in entity_words]
        if entity_type:
            origin_entity['type'] = entity_type
        sample["entities"] = [origin_entity]
        return add_span_in_sample(sample)

    def insert_entity_in_sample(self, sample, entity_str, entity_type):
        """
        randomly insert the entity_str into the sample
        :param sample: a copy of original sample
        :param entity_str: the entity_str used to be added in this sample, list or str
        :param entity_type: the entity type, not optional
        :return:
        """
        entity_words = [{'id': 'ins-{}'.format(i), 'word': c}
                        for i, c in enumerate(entity_str)]

        # randomly select a position for insertion
        insert_position = random.randint(0, sample['words'].__len__())
        new_words = sample['words'][:insert_position] + entity_words + \
                    sample['words'][insert_position:]

        sample['words'] = new_words
        sample['entities'] = [{  # overwrite the original entity
            'id': 'ins-ent-01',
            'type': entity_type,
            'tokens': [w['id'] for w in entity_words],
        }]
        return add_span_in_sample(sample)

    def move_entity_in_sample(self, sample, index=None):
        """
        randomly move the entity to another position (Deprecated)
        (if we need to change entity, change at first, and then move it)
        :param sample: a copy of original sample
        :param index: the index position for moving
        (indexing is for after-removed, e.g. 0, 2, -3)
        :return:
        """
        origin_entity = self.get_entity(sample, generate=False)
        if not origin_entity:
            # if there's no entities and we don't generate one.
            raise ValueError(
                "The sample for moving operation must be a one-entity sample.",
                sample['info']['sid'])
        lef, rig = origin_entity['span'][0], origin_entity['span'][1]
        words = sample['words']

        target_words = words[lef:rig]
        surrounding_words = words[:lef] + words[rig:]
        if index is None:
            index = random.randint(0, surrounding_words.__len__() - 1)
        words = surrounding_words[:index] + target_words + \
                surrounding_words[index:]
        sample['words'] = words
        return add_span_in_sample(sample)

    def shift_entity_in_sample(self, sample, origin_entity=None, max_shift_offset=2):
        if origin_entity is None:
            # sample for shifting must be a one-entity sample
            origin_entity = self.get_entity(sample)
        lef, rig = origin_entity['span']
        span_length = rig - lef
        sent_length = len(sample['words'])
        dist_to_lef = lef
        dist_to_rig = sent_length - 1 - rig
        offset_lef = random.randint(
            -max(0, min(max_shift_offset, dist_to_lef)),
            max(0, min(max_shift_offset, span_length)))
        lef += offset_lef
        offset_rig = random.randint(
            -max(0, min(max_shift_offset, span_length-offset_lef)),
            max(0, min(max_shift_offset, dist_to_rig)))
        rig += offset_rig
        sample['entities'][0]['span'] = (lef, rig)
        if offset_lef != 0 and offset_rig != 0:
            sample['entities'][0]['type'] = TARGET_CLASSES[0]
        return sample

    def generate_random_context_sample(self, method='take'):
        random_context = self.random_n_gram(
            min_length=2, max_length=10,
            method=method, foundry_form=True)

        _sample = {
            'info': {'sid': ''},
            'words': random_context,
            'entities': [],
            'relations': [],
        }
        return _sample

    def sample_for_name_model(self, sample):
        """
        正例：
        1.将随机实体字符串 Insert 到当前句子中的任意位置
        2.将随机实体字符串 Replace 当前 span 的字符串
        3.将当前实体字符串 Insert 到随机生成的句子中去
        负例：
        1.将随机 n-gram 字符串 Insert 到当前句子中的任意位置
        2.将随机 n-gram 字符串 Replace 当前 span 的字符串
        3.将当前实体字符串的前后修剪一部分，变为负例
        :param sample:
        :return:
        """
        _sample = copy.deepcopy(sample)

        # calculate loss for alpha, entity-type classifier
        _sample['alpha'] = 1.
        _sample['info'].update({'augment_type': 'name_aware'})

        def pos_ent_insert(_sample):
            origin_entity = self.get_entity(_sample)
            if origin_entity:
                ent_type = origin_entity['type']
            else:  # M samples without entities
                ent_type = random.choice(TARGET_CLASSES[1:])
            random_word = self.random_ent_str(
                entity_type=ent_type)
            return self.insert_entity_in_sample(
                sample=_sample, entity_str=random_word,
                entity_type=ent_type)

        def pos_ent_replace(_sample):
            ent_type = self.get_entity(_sample)['type']
            random_word = self.random_ent_str(
                entity_type=ent_type)
            return self.replace_entity_in_sample(
                sample=_sample, entity_str=random_word,
                entity_type=ent_type)

        def pos_rand_context(_sample):
            origin_ent = self.get_entity(_sample)
            ent_type = origin_ent['type']
            ent_str = get_string_from_entity(
                entity=origin_ent, words=_sample['words'])
            dummy_sample = self.generate_random_context_sample(method='take')
            return self.insert_entity_in_sample(
                sample=dummy_sample, entity_str=ent_str,
                entity_type=ent_type)

        def neg_gram_insert(_sample):
            ent_type = TARGET_CLASSES[0]
            random_word = self.random_n_gram(
                min_length=2, max_length=6, method='take')
            _sample = self.insert_entity_in_sample(
                sample=_sample, entity_str=random_word,
                entity_type=ent_type)
            return _sample

        def neg_gram_replace(_sample):
            ent_type = TARGET_CLASSES[0]
            random_word = self.random_n_gram(
                min_length=2, max_length=6, method='take')
            _sample = self.replace_entity_in_sample(
                sample=_sample, entity_str=random_word,
                entity_type=ent_type)
            return _sample

        def neg_trim(_sample):
            def generate_trim(ent_str, ent_type):
                ent_length = len(ent_str)
                lef_offset = random.randint(0, min(2, ent_length - 1))
                rig_offset = random.randint(0, min(2, ent_length - 1 - lef_offset))
                if lef_offset + rig_offset > 0:
                    ent_type = TARGET_CLASSES[0]
                # print(ent_str)
                trim_str = ent_str[lef_offset: ent_length - rig_offset]
                if trim_str != ent_str and trim_str in self.entity_case[ent_type]:
                    return generate_trim(ent_str, ent_type)
                return trim_str, ent_type

            origin_ent = self.get_entity(_sample)
            ent_str = get_string_from_entity(
                entity=origin_ent, words=_sample['words'])
            ent_type = origin_ent['type']
            ent_str, ent_type = generate_trim(ent_str, ent_type)

            return self.replace_entity_in_sample(
                sample=_sample, entity_str=ent_str,
                entity_type=ent_type)

        pos_fn_case = [
            pos_ent_insert,
            pos_ent_replace,
            pos_rand_context,
        ]

        neg_fn_case = [
            neg_gram_insert,
            neg_gram_replace,
            neg_trim,
        ]

        if self.get_entity(_sample) is None:
            pos_fn_case = pos_fn_case[:1]
            neg_fn_case = pos_fn_case[:1]

        if random.random() > 0.5:  # positive sample
            fn = random.choice(pos_fn_case)
            _sample = fn(_sample)
            _sample['info']['augment_fn'] = fn.__name__
        else:  # negative sample
            fn = random.choice(neg_fn_case)
            _sample = fn(_sample)
            _sample['info']['augment_fn'] = fn.__name__

        return _sample

    def sample_for_context_model(self, sample):
        """
        正例：
        1.将实体替换随机实体或 n-gram 字符串
        负例：
        (into joint) 将实体的 span 做前后偏移的调整
        1.在没有实体的句子中随机位置插入随机实体或 n-gram 字符串
        :param sample:
        :return:
        """
        _sample = copy.deepcopy(sample)
        # calculate loss for alpha, entity-type classifier
        _sample['alpha'] = 0.
        _sample['info'].update({'augment_type': 'context_aware'})

        def _get_random_word(ent_type, method='take'):
            if ent_type in [TARGET_CLASSES[0], None]:
                random_word = self.random_n_gram(
                    min_length=2, max_length=6, method=method)
            else:
                random_word = self.random_ent_str(
                    entity_type=ent_type)
            return random_word

        origin_entity = self.get_entity(_sample)
        if origin_entity is None:  # MBN- samples
            # generate a random entity for samples without entities
            if random.random() > 0.5:
                ent_type = TARGET_CLASSES[0]
            else:
                ent_type = random.choice(TARGET_CLASSES[1:])
            random_word = _get_random_word(ent_type)
            _sample = self.insert_entity_in_sample(
                sample=_sample, entity_str=random_word,
                entity_type=ent_type)  # set the insertion as non-entity
        else:  # MA+/MA-/MB- samples
            ent_type = origin_entity['type']
            if random.random() > 0.5:  # ent_str with same type
                random_word = _get_random_word(ent_type)
            else:  # random n-gram
                random_word = _get_random_word(TARGET_CLASSES[0])
                ent_type = TARGET_CLASSES[0]
            # remain origin type_label
            _sample = self.replace_entity_in_sample(
                # random_word is a str or list here
                sample=_sample, entity_str=random_word,
                entity_type=ent_type)

        # move to a random position
        # _sample = self.move_entity_in_sample(sample=_sample)

        # the sample has the origin type_label
        return _sample

    def sample_for_joint_model(self, sample, do_shift=None):
        # calculate loss for entity-type classifier
        if do_shift is None:
            do_shift = self.do_shift
        _sample = copy.deepcopy(sample)

        # means do not learn alpha from this sample
        _sample['alpha'] = -1.

        """
        # Evidence:
        logits = torch.from_numpy(np.array([[1.,2.], [1.,2.], [1.,2.]]))
        labels = torch.from_numpy(np.array([-1, 0, 1]))
        
        nn.CrossEntropyLoss(ignore_index=-1, reduction='none')(logits, labels)
        # => tensor([0.0000, 1.3133, 0.3133], dtype=torch.float64)
        """

        if do_shift:
            origin_entity = self.get_entity(_sample, generate=False)
            if origin_entity and random.random() > 0.5:
                _sample = self.shift_entity_in_sample(_sample, origin_entity)

        # positive or negative depends on original type label.
        _sample['info'].update({'augment_type': None})
        return _sample

    def sample_for_joint_model_valid(self, sample):
        return self.sample_for_joint_model(sample=sample, do_shift=False)

    def generate_sample(self, sample, generating_methods=None):
        """
        :param sample: a foundry-like sample, zero or one entity in the sample
        :param generating_methods: a string with numbers for selection, e.g. '23'
        :return:
        """
        _sample = copy.deepcopy(sample)
        function_dict = {
            '0': self.sample_for_joint_model_valid,
            '1': self.sample_for_joint_model,
            '2': self.sample_for_name_model,
            '3': self.sample_for_context_model,
        }
        if generating_methods is None:
            generating_methods = str(self.generating_methods)
        _sample = add_span_in_sample(_sample)
        if _sample['entities'].__len__() == 0:
            # samples without spans can not be transformed into a joint sample.
            generating_methods = generating_methods.replace('1', '')
        fn = function_dict[random.choice(generating_methods)]
        return self.post_process(fn(_sample))

    def __getitem__(self, item):
        # hook the generate function in __getitem__
        def _slice(key):
            start, stop, step = key.indices(len(self))
            for i in range(start, stop, step):
                yield self.generate_sample(self.db[i])

        if isinstance(item, slice):
            return _slice(item)
        return self.generate_sample(self.db[item])

    def get_by_sid(self, sid):
        return self.generate_sample(
            self.db.get_by_sid(sid))

    def __call__(self):
        pass


if __name__ == "__main__":
    pass
