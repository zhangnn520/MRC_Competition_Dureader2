# coding=utf-8
import os
import re
import csv
import xlrd
import json
import jieba
import paddle
import codecs
import pandas as pd
from functools import reduce
from pandas import DataFrame
from tqdm import tqdm
from nltk.tokenize import word_tokenize

base_dir = os.getcwd()


# 用户自定义词典

def word_cut(content, lan_type="ch"):
    if lan_type == "ch":
        content = content.lower().replace("lh", "左").replace("rh", "右")
        paddle.enable_static()
        seg_list = jieba.cut(content, use_paddle=True)  # 使用paddle模式进行分词
    else:
        content = content.lower().replace("lh", "LH").replace("rh", "RH")
        seg_list = word_tokenize(content)
    return ' '.join(seg_list).replace(" / ", "/")


def read_text(path):
    with open(path, "r", encoding="utf-8") as reader:
        return [i.replace("\n", "") for i in reader.readlines()]


# 用户自定义词典

def find_chinese_word(string):
    string = string.strip()
    pattern = r"[\u4e00-\u9fa5]+"
    pat = re.compile(pattern)
    result = pat.findall(string)
    if result:
        return " ".join(result)
    else:
        return ""


def find_english_word(string):
    pattern = r"[a-zA-Z]+|（[a-zA-Z]+）"
    pat = re.compile(pattern)
    result = pat.findall(string.strip())
    return " ".join(result)


def find_index(string):
    pattern = r"[A-Z0-9,，]+"
    pat = re.compile(pattern)
    result_list = pat.findall(string)
    if result_list:
        return [result.replace(",", " ").replace("，", " ") for result in result_list][0]
    else:
        return ""


def write_csv(path, fieldnames, data_dict):
    f = open(path, 'w+', encoding="utf-8",newline="")
    csv_writer = csv.DictWriter(f, fieldnames=fieldnames)
    csv_writer.writeheader()
    for data in data_dict:
        try:
            csv_writer.writerow(data)
        except:
            pass
            # print(data)
    f.close()


def get_string_index(string, string_one):
    index_num = -1
    index_list = []
    string_length = len(string_one)
    if string_length:
        b = string.count(string_one)
        for i in range(b):  # 查找所有的下标
            index_num = string.find(string_one, index_num + 1, len(string))
            index_list.append([str(index_num), str(index_num + string_length)])
    return index_list


def read_json(path):
    with open(path, "r", encoding="utf-8") as reader:
        return json.loads(reader.read())


def get_content(object_list, content_name="CONTENT"):
    return [content[f'{content_name}'] for content in object_list]


def write_text(write_path, content_line):
    with open(write_path, "w+", encoding="utf-8") as writer:
        for content in content_line:
            if content != content_line[-1]:
                writer.write(content + "\n")
            else:
                writer.write(content)


def write_json(write_path, content_line):
    with open(write_path, "w+", encoding="utf-8") as writer:
        writer.writelines(json.dumps(content_line, ensure_ascii=False, indent=4))


def write_line_json(write_path, content_line):
    with open(write_path, "w+", encoding="utf-8") as writer:
        for line in content_line:
            writer.write(json.dumps({line: content_line[line]}, ensure_ascii=False))
            writer.write("\n")


def get_maintain_clear_data(english_maintain, chinese_maintain):
    english_clear_data_list, chinses_clear_data_list = list(), list()
    for en in tqdm(english_maintain):
        for ch in chinese_maintain:
            if en['STATISTICS'] == ch['STATISTICS'] and en['HTML'].split("/")[-1] == ch['HTML'].split("/")[-1]:
                english_clear_data_list.append(en)
                chinses_clear_data_list.append(ch)
    try:
        assert len(english_clear_data_list) == len(chinses_clear_data_list)
    except Exception as e:
        print(e)
    return english_clear_data_list, chinses_clear_data_list


def fast_align_data(en_list, ch_list):
    en2ch_list = list()
    assert len(en_list) == len(ch_list)
    for num, ch_content in enumerate(ch_list):
        en_content = en_list[num]
        new_content = en_content + ' ||| ' + ch_content
        en2ch_list.append(new_content)
    return en2ch_list


def get_fast_align_data(en2chcontent, en2ch_align_content, word_align_result_path):
    """
    :param en2chcontent: 对齐原始预料
    :param en2ch_align_content: 对齐后产生的映射字典
    :param word_align_result_path: 写入对齐映射的文件目录
    :return: null
    """
    result_list = []
    for num, content in enumerate(en2chcontent):
        ens, chs = content.split("|||")
        id2en = {str(num): en for num, en in enumerate(ens.split(" "))}
        id2ch = {str(num): ch for num, ch in enumerate(chs.split(" "))}
        content_align = en2ch_align_content[num]
        en_word_mapping = [id2en[i.split("-")[0]] for i in content_align.replace("\n", "").split(" ")]
        ch_word_mapping = [id2ch[i.split("-")[1]] for i in content_align.replace("\n", "").split(" ")]
        result = dict(zip(en_word_mapping, ch_word_mapping))
        result_list.append(result)
    write_json(word_align_result_path, result_list)


def sub_word(string):
    pattern_one = u"\\(.*?\\)|\\（.*?）|\\[.*?]|-|“|”|``|''"
    result = re.sub(pattern_one, "", string)
    pattern_two = u" +"
    result = re.sub(pattern_two, " ", result)

    return result


def merge_xls_sheet(excel_name):
    # 读取excel
    wb = xlrd.open_workbook(excel_name)
    sheets = wb.sheet_names()
    # 合并sheet
    all_data = DataFrame()
    temp_dict = dict()
    for i in range(len(sheets)):
        j = 1
        df = pd.read_excel(excel_name, sheet_name=i, header=None)
        all_data = all_data.append(df)
        j += 1
        temp_dict[sheets[i]] = df.values.tolist()
    # todo 目前已经完成中英文拆分，后续还需要图片名称和零件位置写入库中
    content_list = all_data[1].values.tolist()[1:]
    return content_list, all_data, temp_dict


def write_bio(write_path, content_line):
    with open(write_path, "w+", encoding="utf-8") as writer:
        for content in content_line:
            writer.writelines(content)
            writer.write("\n")


def read_csv(read_path, model="utf-8"):
    content_list = list()
    with codecs.open(read_path, encoding=model) as f:
        for row in csv.DictReader(f, skipinitialspace=True):
            content_list.append(dict(row))
    return content_list


def delete_duplicate_elements(list_data):
    return reduce(lambda x, y: x if y in x else x + [y], [[], ] + list_data)


def write_ann_text(write_path, content_line):
    with open(write_path, "w+", encoding="utf-8") as writer:
        for content in content_line:
            writer.write(content + "\n")
