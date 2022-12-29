# coding: utf-8
# ==========================================================================
#   Copyright (C) 2020 All rights reserved.
#
#   filename : data_generator.py
#   author   : chendian / okcd00@qq.com
#   date     : 2020-12-18
#   desc     : generate training data group by generating methods.
#              for the new data-flow design on Dec.18 2020
# ==========================================================================
from data_utils import *
from utie.vocab import CharVocabBert


class DataGenerator(Database):
    def __init__(self, database_path, read_only=True,
                 samples=None, n_samples=None,
                 decay=1.0, min_aug_rate=0.1,
                 load_external=False,
                 unsupervised_alpha=False,
                 target_types=TARGET_CLASSES,
                 name='train', load_now=False):
        # foundry-like FolderDB for single json samples.
        self.database_path = database_path
        super().__init__(
            path=database_path, samples=samples, n_samples=n_samples,
            read_only=read_only, load_now=load_now)
        self.name = name
        self.aug_init = 0.7
        self.aug_line = copy.copy(self.aug_init)
        self.aug_decay = decay
        self.min_aug_rate = min_aug_rate
        self.entity_case = None
        self.target_types = []
        # self.random_method = random_method
        # self.generating_methods = generating_methods or "0123456"
        self.vocab = CharVocabBert(True, get_path('bert_vocab'))
        self.sample_counts = int(self.__len__())

        self.load_external = load_external
        self.external_elib_path = ''

        if name == 'train':
            self.char_in_vocab = []
            self.init_generator_library(load_external=load_external)
            # self.target_types = [""] + list(self.entity_case.keys())
        self.target_types = target_types

        self.dispatch = {
            'positive': ['0', '3', '4', '5', '6'],
            'negative': ['-1'],
            'match-negative': ['-1', '1', '2'],
            'empty': ['1', '2'],
        }

        self.function_dict = {
            # id: ( fn_name, label_mask[name, context, joint](-1:mask掉, 0:变成负例, 1:label不变),
            # alpha_case(-1:mask掉, 0:context_aware, 1:name_aware), function_call
            '-1': ('match-neg', [0, 0, 0], -1, self.matching_negative_sample),  # <= main conflict
            '0': ('origin-pos', [1, 1, 1], -1, self.keep_origin_in_sample),
            '1': ('insert_ent', [1, 0, -1], 1, self.fn1_insert_ent),  # we really set this as correct?
            '2': ('insert_rd', [0, 0, 0], -1, self.fn2_insert_rd),  # can be a smaller number
            '3': ('replace_ent', [1, 1, 1], -1, self.fn3_replace_ent),
            '4': ('replace_rd', [0, 1, -1], 0, self.fn4_replace_rd),  # set alpha=0 for un-seen entities
            '5': ('rd_context', [1, 0, -1], 1, self.fn5_rd_context),  # set alpha=1 for
            '6': ('shift_boundry', [0, 0, 0], -1, self.fn6_shift_boundry),
        }
        if unsupervised_alpha:
            # maybe more modifies.
            self.function_dict.update({
                # all alpha set to -1
                '1': ('insert_ent', [1, 0, -1], -1, self.fn1_insert_ent),
                '4': ('replace_rd', [0, 1, -1], -1, self.fn4_replace_rd),
                '5': ('rd_context', [1, 0, -1], -1, self.fn5_rd_context),
            })
        self.unsupervised_alpha = unsupervised_alpha

    @staticmethod
    def determine_mode(label_path=None):
        # fixed DB type for super().__init__()
        # class Folder as default, directory path without postfix
        return 'one_sample_per_file'

    def init_generator_library(self, load_external=False, external_only=False):
        # positive and negative share the same elib file.
        dp_path = self.database_path.rstrip('/').replace('_positive', '').replace('_negative', '')
        entity_library_path = dp_path + '.elib'
        if os.path.exists(entity_library_path):
            self.entity_case = pickle.load(open(entity_library_path, 'rb'))
            if load_external:
                if external_only:
                    self.entity_case = dict()  # renew
                entity_library_path = get_path('external_entity_case')
                for k, v in pickle.load(open(entity_library_path, 'rb')).items():
                    self.entity_case.setdefault(k, [])
                    self.entity_case[k].extend(v)
            for k in self.entity_case:
                if "" in self.entity_case[k]:
                    del self.entity_case[k][""]
        else:
            print("Now generating entity_library for database {}. {}".format(
                self.database_path, get_cur_time(8)))
            self.entity_case = {}
            for s_idx, sample in tqdm(enumerate(self)):
                words = sample['words']
                for ent in sample['entities']:
                    ent_str = get_string_from_entity(entity=ent, words=words)
                    ent_type = ent['type']
                    self.entity_case.setdefault(ent_type, set())
                    self.entity_case[ent_type].add(ent_str)
            self.entity_case = {k: list(v) for k, v in self.entity_case.items()}
            pickle.dump(self.entity_case, open(entity_library_path, 'wb'))

        # Generate charsets for random.choices
        if False:  # is too slow.
            self.char_in_vocab = []
            for k, v in self.entity_case.items():
                char_set = list(set(''.join(v)))
                self.char_in_vocab.extend(char_set)
            self.char_in_vocab = list(set(self.char_in_vocab))
            print("Finished generation for vocab-chars with {} characters.".format(self.char_in_vocab.__len__()))

    def is_known_entity(self, text, ent_type=None):
        if ent_type is not None:
            return text in self.entity_case[ent_type]
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

    def random_n_gram(self, min_length=2, max_length=8, method='take', foundry_form=False):
        if method.startswith('vocab'):
            # return a list of random characters
            char_list = random.choices(
                # SINGLE_CHAR_VOCAB_LIST,
                self.char_in_vocab,
                k=random.randint(min_length, max_length))
            retry = 5
            _str = ''.join(char_list)
            while (not self.is_known_entity(_str)) and (retry > 0):
                char_list = random.choices(
                    # SINGLE_CHAR_VOCAB_LIST,
                    self.char_in_vocab,
                    k=random.randint(min_length, max_length))
                _str = ''.join(char_list)
                retry -= 1
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
        # return an entity string with the same entity_type
        return random.choice(self.entity_case[entity_type])

    @staticmethod
    def get_entity(sample, generate=False, entity_type=None):
        """
        get the entity from sample
        :param sample: target sample
        :param generate: if the entity does not exist,
                         generate an empty_entity at random position.
        :param entity_type: if generate an entity, set its entity_type
        :return: the entity in target sample, or generated one (might be non-entity label).
        """
        if sample['entities']:
            return sample['entities'][0]
        if generate:  # there's no entities in this sample.
            words_length = sample['words'].__len__()
            random_index = random.randint(0, words_length - 1)
            # default dummy_entity is an empty string.
            sample['entities'].append({
                'id': 'dummy-0',
                'tokens': [],
                'type': '',
                'span': (random_index, random_index),
            })
            if entity_type is not None:
                sample['entities'][0]['type'] = entity_type
            return sample['entities'][0]
        return None

    @staticmethod
    def insert_entity_in_sample(sample, entity_str, entity_type):
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
        new_words = sample['words'][:insert_position] + entity_words + sample['words'][insert_position:]

        max_words_length = 500
        if len(new_words) > max_words_length:
            if insert_position + len(entity_words) > max_words_length:
                new_words = new_words[-max_words_length:]
            else:
                new_words = new_words[:max_words_length]

        sample['words'] = new_words
        sample['entities'] = [{  # overwrite the original entity
            'id': 'ins-ent-01',
            'type': entity_type,
            'tokens': [w['id'] for w in entity_words],
        }]
        return add_span_in_sample(sample)

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
        new_words = sample['words'][:left] + entity_words + sample['words'][right:]

        max_words_length = 500
        if len(new_words) > max_words_length:
            if left + len(entity_words) > max_words_length:
                new_words = new_words[-max_words_length:]
            else:
                new_words = new_words[:max_words_length]

        sample['words'] = new_words
        origin_entity['tokens'] = [w['id'] for w in entity_words]
        if entity_type:
            origin_entity['type'] = entity_type
        # we only modify the first entity in sample.
        if sample["entities"]:
            sample["entities"][0] = origin_entity
        else:
            sample["entities"] = [origin_entity]
        sample = add_span_in_sample(sample)
        sample, _ = re_id_func(sample)
        return sample

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
        words = surrounding_words[:index] + target_words + surrounding_words[index:]
        sample['words'] = words
        return add_span_in_sample(sample)

    def generate_random_context_sample(self, min_length=2, max_length=16, method='take'):
        random_context = self.random_n_gram(
            min_length=min_length, max_length=max_length,
            method=method, foundry_form=True)

        _sample = {
            'info': {'sid': ''},
            'words': random_context,
            'entities': [],
            'relations': [],
        }
        return _sample

    @staticmethod
    def matching_negative_sample(sample):
        return sample

    @staticmethod
    def keep_origin_in_sample(sample):
        return sample

    def fn1_insert_ent(self, sample):
        origin_entity = self.get_entity(sample)
        if origin_entity is None:
            ent_type = random.choice(self.target_types[1:])
        elif origin_entity.get('type'):
            ent_type = origin_entity['type']
        else:  # for samples with label not-entity, randomly select one
            ent_type = random.choice(self.target_types[1:])
        random_word = self.random_ent_str(
            entity_type=ent_type)
        sample = self.insert_entity_in_sample(
            sample=sample, entity_str=random_word,
            entity_type=ent_type)
        return sample

    def fn2_insert_rd(self, sample):
        # insert a random string as a negative sample.
        ent_type = self.target_types[0]
        random_word = self.random_n_gram(
            min_length=2, max_length=6,
            method='take')
        sample = self.insert_entity_in_sample(
            sample=sample, entity_str=random_word,
            entity_type=ent_type)
        return sample

    def fn3_replace_ent(self, sample):
        ent_type = self.get_entity(sample)['type']
        random_word = self.random_ent_str(
            entity_type=ent_type)
        sample = self.replace_entity_in_sample(
            sample=sample, entity_str=random_word,
            entity_type=ent_type)
        return sample

    def trim_entity_in_sample(self, sample):
        # be covered by shift-boundry method
        _sample = copy.deepcopy(sample)

        def generate_trim(_ent_str, _ent_type):
            ent_length = len(_ent_str)
            lef_offset = random.randint(0, min(2, ent_length - 1))
            rig_offset = random.randint(0, min(2, ent_length - 1 - lef_offset))
            if lef_offset + rig_offset > 0:
                _ent_type = self.target_types[0]
            # print(ent_str)
            trim_str = _ent_str[lef_offset: ent_length - rig_offset]
            if trim_str != _ent_str and trim_str in self.entity_case[_ent_type]:
                return generate_trim(_ent_str, _ent_type)
            return trim_str, _ent_type

        origin_ent = self.get_entity(_sample)
        ent_str = get_string_from_entity(
            entity=origin_ent, words=_sample['words'])
        ent_type = origin_ent['type']
        ent_str, ent_type = generate_trim(ent_str, ent_type)
        sample = self.replace_entity_in_sample(
            sample=_sample, entity_str=ent_str,
            entity_type=ent_type)
        return sample

    def fn4_replace_rd(self, sample):
        ent_type = self.target_types[0]
        random_word = self.random_n_gram(
            min_length=2, max_length=6,
            method='take')
        sample = self.replace_entity_in_sample(
            sample=sample, entity_str=random_word,
            entity_type=ent_type)
        return sample

    def fn5_rd_context(self, sample):
        origin_ent = self.get_entity(sample)
        ent_type = origin_ent['type']
        ent_str = get_string_from_entity(
            entity=origin_ent, words=sample['words'])
        dummy_sample = self.generate_random_context_sample(
            min_length=2, max_length=16,
            method='take')
        sample = self.insert_entity_in_sample(
            sample=dummy_sample, entity_str=ent_str,
            entity_type=ent_type)
        return sample

    def fn6_shift_boundry(self, sample, origin_entity=None, max_shift_offset=2):
        if origin_entity is None:
            # sample for shifting must be a one-entity sample
            origin_entity = self.get_entity(sample)
        _lef, _rig = origin_entity['span']
        span_length = _rig - _lef
        sample_words = sample['words']
        sent_length = len(sample_words)
        dist_to_lef = _lef
        dist_to_rig = sent_length - 1 - _rig

        def generate_offset():
            _offset_lef = random.randint(
                -max(0, min(max_shift_offset, dist_to_lef-1)),
                max(0, min(max_shift_offset, span_length-1)))
            _offset_rig = random.randint(
                -max(0, min(max_shift_offset, span_length-_offset_lef-1, span_length-1)),
                max(0, min(max_shift_offset, dist_to_rig-1)))
            return _offset_lef, _offset_rig

        retry = 0
        offset_lef, offset_rig = generate_offset()
        while offset_lef == 0 and offset_rig == 0 and retry < 3:
            offset_lef, offset_rig = generate_offset()
            retry += 1

        _lef += offset_lef
        _rig += offset_rig
        sample['entities'][0]['span'] = (_lef, _rig)
        if offset_lef == 0 and offset_rig == 0:
            sample['info']['shift_offset'] = None
        if _rig - _lef == 0:  # for empty entity cases
            sample['info']['shift_offset'] = None
        else:
            sample['entities'][0]['tokens'] = [
                sample_words[idx]['id'] for idx in range(_lef, _rig)]
            sample['entities'][0]['type'] = self.target_types[0]
            sample['info']['shift_offset'] = (offset_lef, offset_rig)
        return sample

    def fnb_add_blanks(self, sample):
        """
        用于应对句子中莫名出现空格的情况，数据增强的目标是随机插入空格（实体内部是否允许有空格？），
        并要求模型最终得到包含空格的正确预测。
        :param sample:
        :return:
        """
        return sample

    def post_process(self, sample, label_mask, alpha=-1):
        if sample is None:
            # we better drop all invalid samples in pre-processing
            return None
        # re-construct the ids
        sample = self.reid(sample)
        # re-generate the text
        sample = add_text_key_in_sample(sample)
        # re-calculate the span info
        sample = add_span_in_sample(sample)
        # get the label from the original entity
        current_ent = self.get_entity(sample)
        ent_type = current_ent.get('type')
        if ent_type.lower() in ['not-found', 'none']:
            ent_type = ''
        # if the negative span has known entity-text.
        if self.name == 'train':
            if sample['info']['augment_type'] in ['match-neg', 'match-negative']:
                text = get_string_from_entity(current_ent, sample['words'])
                if self.is_known_entity(text):
                    label_mask[0] = 1  # name label is 1
        # if offsets from shift_boundry are both 0
        if sample['info']['augment_type'] == 'shift_boundry':
            if sample['info'].get('shift_offset') is None:
                fn_name, label_mask, alpha, _ = self.function_dict['0']
                sample['info']['augment_type'] = fn_name
        type_label = self.target_types.index(ent_type)

        # add labels for each submodules into the sample
        sample['labels'] = np.clip(
            np.array(label_mask) * type_label,
            a_max=None, a_min=-1)  # name, context, joint
        # -1 means do not learn alpha from this sample, and 0 for context, 1 for name
        if sample.get('alpha') is None:
            sample['alpha'] = alpha
        # re-construct the ids
        sample = self.reid(sample)
        return sample

    def get_generating_method(self, _entity):
        if _entity is None:
            # this kind of samples only exist in train set
            ent_alternative = 'empty'
        elif _entity['type'] not in self.target_types[1:]:
            ent_alternative = 'match-negative'
        else:
            ent_alternative = 'positive'

        random_for_aug = random.random()
        if not self.name.startswith('train'):
            # valid / test / infer phase
            generating_methods = {
                'positive': ['0'],
                'match-negative': ['-1'],
                # 'empty': ['1', '2'],
                # if the key is "empty", something must be wrong.
            }.get(ent_alternative)
        elif random_for_aug > self.aug_line and ent_alternative != 'empty':
            # no augmentations
            generating_methods = {
                'positive': ['0'],
                'match-negative': ['-1'],
                'empty': ['1', '2'],
                # if the key is "empty", something must be wrong.
            }.get(ent_alternative)
        else:
            # do augmentations
            generating_methods = self.dispatch[ent_alternative]
        try:
            _gm = random.choice(generating_methods)
        except Exception as e:
            print(generating_methods)
            print(random_for_aug, self.aug_line, ent_alternative, _entity)
            raise ValueError(e)

        return _gm

    def generate_sample(self, sample):
        """
        label & loss masks for [name, context, joint] => Alpha
        :param sample: a foundry-like sample, zero or one entity in the sample
        :return:
        """
        _sample = copy.deepcopy(sample)

        # obtain the span for the origin entity
        _sample = add_span_in_sample(_sample)
        _entity = self.get_entity(_sample)
        if _entity is None:
            _sample['info']['augment_type'] = 'none'
            return _sample  # do nothing on empty samples
        _gm = self.get_generating_method(_entity)

        fn_name, label_mask, alpha, fn = self.function_dict[_gm]
        _sample = fn(_sample)
        _sample['info']['augment_type'] = fn_name
        _sample = self.post_process(
            sample=_sample, label_mask=label_mask, alpha=alpha)
        if _sample['info']['augment_type'].lower().startswith('replace'):
            fn_name, label_mask, alpha, fn = self.function_dict['0']
            focus_sample = copy.deepcopy(sample)
            focus_sample['info']['augment_type'] = fn_name
            origin_pos_sample = self.post_process(
                sample=focus_sample,
                label_mask=label_mask,
                alpha=alpha)
            _sample.update({"origin_sample": origin_pos_sample})
        return _sample

    def reset_augment_probability(self):
        self.aug_line = copy.copy(self.aug_init)

    def augment_probability_decay(self, decay=None):
        if decay is None:
            decay = self.aug_decay
        self.aug_line = max(self.aug_line * decay, self.min_aug_rate)

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
