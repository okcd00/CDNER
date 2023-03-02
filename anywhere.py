import datetime


yangjie_rich_pretrain_word_path = "TBD"
yangjie_rich_pretrain_unigram_path = "TBD"
yangjie_rich_pretrain_bigram_path = "TBD"


from shared.const import task_ner_labels
TARGET_CLASSES = task_ner_labels.get('ccks')


def get_path(key):
    return {}.get(key, "TBD")


def get_cur_time(delta=0):
    # C8 has no delta, C14 has 8 hours delta.
    cur_time = (datetime.datetime.now() +
                datetime.timedelta(hours=delta)).strftime("%m-%d %H:%M:%S")
    return cur_time
