# the msra dataset comes from
# https://github.com/lemonhu/NER-BERT-pytorch/tree/master/data/msra


if __name__ == '__main__':
    for phase in ['train', 'dev', 'test']:
        sentences = [line.strip().split()
                     for line in open('/home/chendian/PURE/data/msra/{}/sentences.txt'.format(phase), 'r')]
        tags = [line.strip().split()
                for line in open('/home/chendian/PURE/data/msra/{}/tags.txt'.format(phase), 'r')]

        with open('/home/chendian/PURE/data/msra/{}.ner'.format(phase), 'w') as f:
            for sent, tg in zip(sentences, tags):
                for s, t in zip(sent, tg):
                    f.write('{}\t{}\n'.format(s, t))
                f.write('\n')
