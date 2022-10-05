task_ner_labels = {
    # Chinese Corpus
    'msra':     [u"公司", u"人名", u"地址"],
    # 'onto4':    ['ORG', 'PER', 'LOC', 'GPE'],
    'onto4':    [u'公司', u'人名', u'地址', u'位置'],
    'resume':   [u"公司", u"人名", u"地址", u"学历", u"专业", u"国籍", u"民族", u"职称"],
    'findoc':   [u"公司", u"人名", u"地址", u"产品业务", u"文件"],
    'ccks':     ['cellno', 'devzone', 'distance', 'poi', 'floorno',
                 'roadno', 'intersection', 'city', 'houseno',
                 'village_group', 'town', 'subpoi', 'assist',
                 'community', 'district', 'road', 'prov'],
    'onto5':    ['CARDINAL', 'DATE', 'EVENT', 'FAC', 'GPE', 'LANGUAGE', 'LAW', 'LOC', 
                 'MONEY', 'NORP', 'ORDINAL', 'ORG', 'PERCENT', 'PERSON', 'PRODUCT', 
                 'QUANTITY', 'TIME', 'WORK_OF_ART'],
    'cluener':  [u'地址', u'书名', u'公司', u'游戏', u'政府', u'电影', u'姓名', u'机构', u'职位', u'景点'],
    'unity':    [u'实体'],  # for test training binary-judgment
    'weibo':    [u'公司', u'人名', u'地址', u'位置'],
    # Span Filtering Models
    'span': [u'实体'],  
    'bin': [u'IN-实体'], 
    'bdy': [u'B-实体', 'E-实体', 'S-实体'],

    # English Corpus
    'ace04':    ['FAC', 'WEA', 'LOC', 'VEH', 'GPE', 'ORG', 'PER'],
    'ace05':    ['FAC', 'WEA', 'LOC', 'VEH', 'GPE', 'ORG', 'PER'],
    'scierc':   ['Method', 'OtherScientificTerm', 'Task', 'Generic', 'Material', 'Metric'],
}

task_tag_dicts = {
    'msra': {
        'NR': '人名',
        'NS': '地址',
        'NT': '公司'},
    'onto4': {
        'PER': '人名',
        'LOC': '地址',
        'ORG': '公司',
        'GPE': '位置'},
    'resume': {
        'ORG': '公司',
        'EDU': '学历',
        'PRO': '专业',
        'LOC': '地址',
        'NAME': '人名',
        'CONT': '国籍',
        'RACE': '民族',
        'TITLE': '职称',
    },
    'weibo': {
        'GPE.NAM': '位置',
        'GPE.NOM': '位置',
        'LOC.NAM': '地址',
        'LOC.NOM': '地址',
        'ORG.NAM': '公司',
        'ORG.NOM': '公司',
        'PER.NAM': '人名',
        'PER.NOM': '人名',
    },
    'cluener': {
        'address': '地址',
        'book': '书名',
        'company': '公司',
        'game': '游戏',
        'goverment': '政府',
        'movie': '电影',
        'name': '姓名',
        'organization': '机构',
        'position': '职位',
        'scene': '景点'
    },
    'onto5': {},
    'findoc': None,
    'unity': None,
}

task_max_span_length = {
    # Chinese Corpus
    'msra': 25,  # PER<18, COM<36, LOC<25
    'weibo': 12,
    'onto4': 16,
    'onto5': 20,
    'resume': 16,
    'ccks': 18,
    'findoc': 40,  #
    'unity': 40,  # msra, resume, onto4, findoc

    # English Corpus
    'ace04': 8,
    'ace05': 8,
    'scierc': 8,
}


task_rel_labels = {
    'ace04': ['PER-SOC', 'OTHER-AFF', 'ART', 'GPE-AFF', 'EMP-ORG', 'PHYS'],
    'ace05': ['ART', 'ORG-AFF', 'GEN-AFF', 'PHYS', 'PER-SOC', 'PART-WHOLE'],
    'scierc': ['PART-OF', 'USED-FOR', 'FEATURE-OF', 'CONJUNCTION', 'EVALUATE-FOR', 'HYPONYM-OF', 'COMPARE'],
}


TASK_NAME = 'findoc'
CONFIG_FOR_PURE_API = {
    # basic settings for prediction
    'task': TASK_NAME,
    'max_span_length': task_max_span_length[TASK_NAME],

    # hyper-parameters
    'do_eval': True,
    'eval_test': True,
    'context_window': 0,
    'train_batch_size': 8,
    'eval_batch_size': 8,
    
    # model settings
    'take_name_module': True,
    'take_context_module': False,
    'take_width_feature': True,
    'take_context_attn': False,
    'take_alpha_loss': False,
    'boundary_token': 'both',
    'boundary_only_mode': 'none',
    'fusion_method': 'none',
    'loss_function': 'ce',

    # path to Bert
    'use_albert': False,
    'model': 'bert-base-chinese',  # use known bert
    'bert_model_dir': '/data/chend/model_files/chinese_L-12_H-768_A-12/',

    # path to the file for predicting, and path to the dump file
    'data_dir': './data/test_files/',  # put test.json here for predicting
    'output_dir': './output_dir/',  # load model files in this directory
    'test_pred_filename': 'ent_pred_test.json',  # the results file's name, in output_dir
}


use_albert = False
if use_albert:
    CONFIG_FOR_PURE_API.update({
        'use_albert': True,
        'model': 'voidful/albert-chinese-xxlarge',
        'bert_model_dir': '/home/chendian/download/albert_chinese_xxlarge',
    })


is_training = False
if is_training:
    CONFIG_FOR_PURE_API.update({
        'seed': 0,
        'num_epoch': 50,
        'warmup_proportion': 0.1,  # means first 10% epochs
        'learning_rate': 1e-5,  # for BERT's learning
        'task_learning_rate': 1e-4,  # for down-stream task learning
        'print_loss_step': 500,
        'eval_per_epoch': 1,
        'do_train': True,
        'take_context_module': False,
    })


def get_labelmap(label_list):
    label2id = {}
    id2label = {}
    # i = 0 means 'O', but not in the dict
    for i, label in enumerate(label_list):
        label2id[label] = i + 1
        id2label[i + 1] = label
    return label2id, id2label
