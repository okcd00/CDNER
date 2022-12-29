import copy
import random
import pickle


def get_entity_length_count():
    from collections import Counter
    ct = Counter(map(len, ent_lib['人名']))
    print(sorted(ct.most_common()))
    return ct


def get_msra_entity_lib(db_train):
    msra_lib_train = {'人名': set(), '公司': set(), '地址': set()}
    for sample in db_train:
        words = sample['words']
        for ent in sample['entities']:
            et = ent['type']
            text = get_string_from_entity(entity=ent, words=words)
            msra_lib_train[et].add(text)
    return msra_lib_train


def check_length_threshold(msra_lib_train):
    per = [name for name in msra_lib['人名'] if 7 <= len(name) <= 16 and (name not in msra_lib_train)]
    loc = [name for name in msra_lib['地址'] if 7 <= len(name) <= 16 and (name not in msra_lib_train)]
    com = [name for name in msra_lib['公司'] if 12 <= len(name) <= 16 and (name not in msra_lib_train)]
    return len(per), len(loc), len(com)


def get_longest_mentions(msra_lib_train):
    long_names = sorted([nm for nm in msra_lib['人名'] if len(nm) <= 16 and (nm not in msra_lib_train)],
                        key=lambda x: -len(x))[:50]
    long_comps = sorted([nm for nm in msra_lib['公司'] if len(nm) <= 16 and (nm not in msra_lib_train)],
                        key=lambda x: -len(x))[:50]
    return long_names, long_comps


def has_entity_with_type(sp, et):
    for ent in sp['entities']:
        if ent['type'] == et:
            return True
    return False


def only_leave_first_ent_with_type(sp, et):
    _entities = []
    import copy
    span = None
    for ent in sp['entities']:
        if ent['type'] == et:
            _entities.append(ent)
            span = ent['span']
            ent_type = ent['type']
            break
    _sp = copy.deepcopy(sp)
    if span is None:
        raise ValueError()
    _sp['entities'] = _entities
    _sp['target_span'] = span
    _sp['target_type'] = ent_type
    _sp['target_text'] = get_string_from_entity(_entities[0], sp['words'])
    return _sp


def get_samples_for_inv_test(db_msra):
    tags = ['公司', '人名']
    samples_for_tags = {}
    for tag in tags:
        samples_for_x = random.sample([sample for sample in db_msra
                                       if has_entity_with_type(sample, tag)], k=100)
        samples_for_x = [only_leave_first_ent_with_type(sample, tag)
                         for sample in samples_for_x]
        samples_for_tags[tag] = samples_for_x
    return samples_for_tags


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


def replace_entity_in_sample(sample, entity_str, entity_type=None):
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

    origin_entity = get_entity(sample)
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
    sample["entities"] = [origin_entity]
    return add_span_in_sample(sample)


def generate_inv_samples(entity_lib, samples_for_tags):
    ret_samples = []

    # if entity_lib is None:
    # entity_lib = msra_long_elib
    # entity_lib = common_long_elib

    for entity_type in ['人名', '公司']:
        for s_idx, sample in enumerate(samples_for_tags[entity_type]):
            l, r = sample['entities'][0]['span']
            for e_idx, entity_str in enumerate(entity_lib[entity_type]):
                g_sample = replace_entity_in_sample(
                    copy.deepcopy(sample), entity_str, entity_type=entity_type)
                g_sample['target_type'] = entity_type
                g_sample['target_span'] = [l, l + len(entity_str)]
                g_sample['info']['row_col'] = [e_idx, s_idx]
                g_sample['info']['sid'] = '{}:{}-{}-{}'.format(g_sample['info']['sid'], entity_type, e_idx, s_idx)
                ret_samples.append(g_sample)

    from scripts import count_entities
    print(count_entities(ret_samples))
    return ret_samples


if __name__ == "__main__":
    ent_lib = pickle.load(open('./external_entity_case.elib', 'rb'))
    msra_lib = pickle.load(open('/data/chend/preprocess_data/sl_train5texts_msra_folder.elib', 'rb'))
    # get_path('sl_msra_train')
    print(msra_lib.keys())