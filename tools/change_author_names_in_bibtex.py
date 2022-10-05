def change(author_str):
    authors = author_str.split('and')
    for i, _str in enumerate(authors):
        _str = _str.strip()
        last_name = _str.split(' ')[-1]
        authors[i] = f'{last_name}, {_str[:-len(last_name)].strip()}'
    return ' and '.join(authors)

if __name__ == "__main__":
    text = """
    Yi Luan and Luheng He and Mari Ostendorf and Hannaneh Hajishirzi
    Luheng He and Kenton Lee and Omer Levy and Luke Zettlemoyer
    Luoxin Chen and Weitong Ruan and Xinyue Liu and Jianhua Lu
    Xiang Dai and Heike Adel
    Dou Hu and Lingwei Wei
    Yuyang Nie and Yuanhe Tian and Yan Song and Xiang Ao and Xiang Wan
    Ruiqing Zhang and Chao Pang and Chuanqiang Zhang and Shuohuan Wang and Zhongjun He and Yu Sun and Hua Wu and Haifeng Wang
    """
    
    for _line in text.split('\n'):
        _line = _line.strip()
        if _line:
            print(change(_line))
