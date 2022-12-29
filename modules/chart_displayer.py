# coding: utf-8
# ==========================================================================
#   Copyright (C) 2016-2021 All rights reserved.
#
#   filename : ChartDisplayer.py
#   author   : chendian / okcd00@qq.com
#   date     : 2021-01-08
#   desc     : for drawing charts in Jupyter, later transform into class
# ==========================================================================
import seaborn as sns
from data_utils import *
from pylab import *  # 支持中文
import matplotlib.pyplot as plt

# 这两行代码解决 plt 中文显示的问题
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


def drawx(y_list, label_list=None, title='undefined title',
          start=0, end=None, x_list=None, x_label=None, semilog=False, axh=0):
    length = min(end, max(map(len, y_list))) if end else max(map(len, y_list))
    y_length = list(map(lambda x: min(length, x.__len__()), y_list))
    x_list = range(1, length + 1) if x_list is None else x_list[:length]
    plt.figure(figsize=(15, 5))
    if label_list is None:
        label_list = map(
            lambda x: u"Data_{}".format(x),
            range(y_list.__len__()))
    for idx, label in enumerate(label_list):
        plt.plot(
            x_list[start: y_length[idx]],
            y_list[idx][start: y_length[idx]],
            marker='o',
            label=label)
    plt.legend()  # let legends work
    plt.margins(0.05)
    plt.subplots_adjust(bottom=0.15)
    plt.xlabel(u"Epoch" if x_label is None else x_label)
    plt.title(u"Brief Figure for {}".format(title))  # 标题
    if axh:  # Add a horizontal line across the axis.
        plt.axhline(axh)
    if semilog:
        plt.semilogy()

    plt.grid(False)
    plt.show()


def generate_result_dict_from_logs(log_files, phase='train'):
    pat_rn = '^\[[0-9]{2,5} runner(_[a-z]{2,8})?\.py [0-9]{2,5}\]\:'
    pat_tv = '^Ep [0-9]{1,4} \| PRF\.V\: \[[0-9\.]+\|[0-9\.]+\] [0-9\.]+ \| Loss: [0-9\.]+'
    pat_nc = '^Ep [0-9]{1,4} \| PRF\.V\: \[[0-9\.]+\|[0-9\.]+\] [0-9\.]+ \| Loss: \[([0-9\.]+\s*){2,6}\]'

    def is_target_line(l):
        if re.match(pat_rn, l.strip()):
            return 1
        if re.match(pat_tv, l.strip()):
            return 2
        if re.match(pat_nc, l.strip()):
            return 3
        return 0

    result_dict = defaultdict(list)
    for name, _path in log_files.items():
        lines = [l for l in open(_path, 'r') if is_target_line(l) == 1]
        print(name, lines.__len__())
        ep_idx = -1
        for line in lines:
            json_str = line.split(']: ')[1]
            try:
                results = json.loads(json_str)
            except Exception as e:
                # print(e, json_str)
                continue
            target = phase
            if target not in results:
                # print(name, target, results.keys())
                continue
            ep_idx += 1
            results = results.get(target)
            result_dict[name].append({'epoch': ep_idx})
            for main_key in results:
                if not isinstance(results[main_key], dict):
                    if main_key == 'loss':
                        result_dict[name][ep_idx].update({
                            'loss_submodules': results[main_key][0],
                            'loss_alpha': results[main_key][1],
                            # 'loss_name': results[main_key][2],
                            # 'loss_context': results[main_key][3],
                            # 'loss_joint': results[main_key][4],
                        })
                        continue
                    result_dict[name][ep_idx].update({main_key: results[main_key]})
                    continue
                for sub_key in results[main_key]:
                    result_dict[name][ep_idx].update({
                        sub_key + '_' + main_key: results[main_key][sub_key]})
    return result_dict


def draw_figure_from_log_files(log_files, targets=None, phase='train', max_length=500,
                               axh=None, semilogy=False, title='training performance on nc_models'):
    plt.figure(figsize=(15, 5))

    if targets is None:
        targets = ['f1_overall']

    result_dict = generate_result_dict_from_logs(log_files, phase=phase)
    # pprint(result_dict)
    names = list(log_files.keys())[:]

    def marker_fn(target_measure):
        if 'precision' in target_measure:
            return '^'
        if 'recall' in target_measure:
            return 'v'
        if 'f1' in target_measure:
            return 'd'
        return 'o'

    for n_idx, name in enumerate(names):
        for target in targets:
            plt.plot(
                [x['epoch'] for x in result_dict[name]][:max_length],
                [x[target] for x in result_dict[name]][:max_length],
                marker=marker_fn(target),
                label='{} @{}'.format(target, name))

    plt.legend()  # let legends work
    plt.margins(0.05)
    plt.subplots_adjust(bottom=0.15)
    plt.xlabel(u"Epoch")
    plt.title(u"Brief Figure for {}".format(title))
    if axh:
        plt.axhline(axh)
    if semilogy:
        plt.semilogy()

    plt.grid(False)
    plt.show()


def draw_prob_distribution(name_aware, context_aware, joint,
                           to_percentage=False, title=None):
    # 输入统计数据
    plt.figure(figsize=(15, 5))
    headers = [str(0.1 * x)[:3] for x in range(11)]
    # print(headers)

    bar_width = 0.3  # 条形宽度
    index_n = np.arange(len(headers))  # 横坐标
    index_c = index_n + bar_width
    index_j = index_c + bar_width

    # 使用两次 bar 函数画出两组条形图
    arr_n = np.array(list(name_aware.values()))
    arr_c = np.array(list(context_aware.values()))
    arr_j = np.array(list(joint.values()))

    if to_percentage:
        arr_n = arr_n / arr_n.sum()
        arr_c = arr_c / arr_c.sum()
        arr_j = arr_j / arr_j.sum()

    plt.bar(index_n, height=arr_n,
            width=bar_width, color='b', label='name_aware')
    plt.bar(index_c, height=arr_c,
            width=bar_width, color='g', label='context_aware')
    plt.bar(index_j, height=arr_j,
            width=bar_width, color='r', label='joint')

    plt.legend()  # 显示图例
    plt.xticks(index_c, headers)
    plt.ylabel('confidence')  # 纵坐标轴标题
    if title is None:
        title = '标签为正例时，对应标签上的 confidence 分布情况'
    plt.title(title)  # 图形标题
    plt.show()


def multi_heatmap():
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

    plt.figure(figsize=(20, 9))
    plt.figure(1)

    hm_pos_case = [331, 332, 333, 334, 335, 336, 337, 338, 339]
    model_case = ['Bert-CRF', 'FLAT', 'NC']
    tag_type_case = ['PER', 'ORG', 'LOC']
    for idx, hm_pos in enumerate(hm_pos_case):
        ax = plt.subplot(hm_pos)
        cmap = sns.cubehelix_palette(start=hm_pos % 3 + 0.5, gamma=0.5, light=.9, as_cmap=True)
        # dat = pickle.load('/home/chendian/doc_ner/data/heatmap/{}.pkl'.format(hm_pos), 'rb')
        dat = (np.random.random([20, 50]) > 0.5).astype(int)
        sns.heatmap(dat,
                    cmap=cmap, linewidths=0, ax=ax, cbar=False).invert_yaxis()
        ax.set_title('INV check on ({} from {}).'.format(
            tag_type_case[(hm_pos % 10 - 1) // 3],
            model_case[(hm_pos - 1) % 3]), fontdict={'weight': 'bold', 'size': 18})
        ax.set_xlabel('sample index', fontdict={'weight': 'normal', 'size': 15})
        ax.set_ylabel('entity index', fontdict={'weight': 'normal', 'size': 15})

    plt.tight_layout(pad=0.5, w_pad=1.0, h_pad=1.0)
    # f.savefig('./sns_heatmap_normal.jpg', bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    name_aware = {
        0.0: 1192,
        0.1: 665,
        0.2: 578,
        0.3: 570,
        0.4: 686,
        0.5: 805,
        0.6: 1082,
        0.7: 1499,
        0.8: 2496,
        0.9: 5152,
        1.0: 14794}
    context_aware = {
        0.0: 1843,
        0.1: 1290,
        0.2: 1009,
        0.3: 964,
        0.4: 1051,
        0.5: 1152,
        0.6: 1384,
        0.7: 1951,
        0.8: 3095,
        0.9: 5988,
        1.0: 9792}
    joint = {
        0.0: 1072,
        0.1: 539,
        0.2: 462,
        0.3: 494,
        0.4: 618,
        0.5: 699,
        0.6: 1040,
        0.7: 1547,
        0.8: 2511,
        0.9: 4838,
        1.0: 15699}
    draw_prob_distribution(name_aware, context_aware, joint)
