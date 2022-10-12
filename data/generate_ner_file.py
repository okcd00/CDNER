import unicodedata


def is_whitespace(char):
    """Checks whether `chars` is a whitespace character."""
    # \t, \n, and \r are technically contorl characters but we treat them
    # as whitespace since they are generally considered as such.
    if char == " " or char == "\t" or char == "\n" or char == "\r":
        return True
    cat = unicodedata.category(char)
    if cat == "Zs":
        return True
    return False


def generate_msra():
    # the msra dataset comes from
    # https://github.com/lemonhu/NER-BERT-pytorch/tree/master/data/msra
    dir_path = '/home/chendian/PURE/data/msra/'
    for phase in ['train', 'dev', 'test']:
        sentences = [line.strip().split()
                     for line in open(dir_path + '{}/sentences.txt'.format(phase), 'r')]
        tags = [line.strip().split()
                for line in open(dir_path + '{}/tags.txt'.format(phase), 'r')]

        with open(dir_path + '{}.ner'.format(phase), 'w') as f:
            for sent, tg in zip(sentences, tags):
                for s, t in zip(sent, tg):
                    f.write('{}\t{}\n'.format(s, t))
                f.write('\n')


def generate_weibo():
    dir_path = '/data/chendian/download/weibo/'
    for phase in ['train', 'dev', 'test']:
        lines = [line.strip() for line in open(dir_path + 'weiboNER_2nd_conll.{}'.format(phase), 'r')]
        with open('./data/weibo/{}.ner'.format(phase), 'w') as f:
            for line in lines:
                if line.strip():
                    items = line.split()
                    if items[0][0].strip() and not is_whitespace(items[0][0]):
                        f.write(f"{items[0][0]}\t{items[1].split('.')[0]}\n")
                else:
                    f.write(f'{line}\n')


if __name__ == '__main__':
    generate_weibo()