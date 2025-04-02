import os.path as osp
import random
import xml.etree.ElementTree as ET
# from .utils import list_files, replace_ext
# from paddlex.utils import is_pic, logging
import os
import os.path as osp
from PIL import Image
from flask import logging
import numpy as np
# from loguru import logger as logging


def is_pic(img_name):
    valid_suffix = ['JPEG', 'jpeg', 'JPG', 'jpg', 'BMP', 'bmp', 'PNG', 'png']
    suffix = img_name.split('.')[-1]
    if suffix not in valid_suffix:
        return False
    return True

def replace_ext(filename, new_ext):
    """ 替换文件后缀

    Args:
        filename: 文件路径
        new_ext: 需要替换的新的后缀
    """
    items = filename.split(".")
    items[-1] = new_ext
    new_filename = ".".join(items)
    return new_filename


def list_files(dirname):
    """ 列出目录下所有文件（包括所属的一级子目录下文件）

    Args:
        dirname: 目录路径
    """

    def filter_file(f):
        if f.startswith('.'):
            return True
        return False

    all_files = list()
    dirs = list()
    for f in os.listdir(dirname):
        if filter_file(f):
            continue
        if osp.isdir(osp.join(dirname, f)):
            dirs.append(f)
        else:
            all_files.append(f)
    for d in dirs:
        for f in os.listdir(osp.join(dirname, d)):
            if filter_file(f):
                continue
            if osp.isdir(osp.join(dirname, d, f)):
                continue
            all_files.append(osp.join(d, f))
    return all_files

def split_voc_dataset(dataset_dir, val_percent, test_percent, save_dir):
    if not osp.exists(osp.join(dataset_dir, "JPEGImages")):
        logging.error("\'JPEGImages\' is not found in {}!".format(dataset_dir))
    if not osp.exists(osp.join(dataset_dir, "Annotations")):
        logging.error("\'Annotations\' is not found in {}!".format(
            dataset_dir))

    all_image_files = list_files(osp.join(dataset_dir, "JPEGImages"))

    image_anno_list = list()
    label_list = list()
    for image_file in all_image_files:
        if not is_pic(image_file):
            continue
        anno_name = replace_ext(image_file, "xml")
        if osp.exists(osp.join(dataset_dir, "Annotations", anno_name)):
            image_anno_list.append([image_file, anno_name])
            try:
                tree = ET.parse(
                    osp.join(dataset_dir, "Annotations", anno_name))
            except:
                raise Exception("文件{}不是一个良构的xml文件，请检查标注文件".format(
                    osp.join(dataset_dir, "Annotations", anno_name)))
            objs = tree.findall("object")
            for i, obj in enumerate(objs):
                cname = obj.find('name').text
                if not cname in label_list:
                    label_list.append(cname)
        else:
            logging.error("The annotation file {} doesn't exist!".format(
                anno_name))

    random.shuffle(image_anno_list)
    image_num = len(image_anno_list)
    val_num = int(image_num * val_percent)
    test_num = int(image_num * test_percent)
    train_num = image_num - val_num - test_num

    train_image_anno_list = image_anno_list[:train_num]
    val_image_anno_list = image_anno_list[train_num:train_num + val_num]
    test_image_anno_list = image_anno_list[train_num + val_num:]

    with open(
            osp.join(save_dir, 'train.txt'), mode='w',
            encoding='utf-8') as f:
        for x in train_image_anno_list:
            file = osp.join("JPEGImages", x[0])
            label = osp.join("Annotations", x[1])
            f.write('{} {}\n'.format(file, label))
    with open(
            osp.join(save_dir, 'val.txt'), mode='w',
            encoding='utf-8') as f:
        for x in val_image_anno_list:
            file = osp.join("JPEGImages", x[0])
            label = osp.join("Annotations", x[1])
            f.write('{} {}\n'.format(file, label))
    if len(test_image_anno_list):
        with open(
                osp.join(save_dir, 'test.txt'), mode='w',
                encoding='utf-8') as f:
            for x in test_image_anno_list:
                file = osp.join("JPEGImages", x[0])
                label = osp.join("Annotations", x[1])  
                f.write('{} {}\n'.format(file, label))
    with open(
            osp.join(save_dir, 'labels.txt'), mode='w', encoding='utf-8') as f:
        for l in sorted(label_list):
            f.write('{}\n'.format(l))

    return train_num, val_num, test_num

if __name__ == '__main__':
    # 修改为本地路径
    dataset_dir = r"D:\dataset\train"
    save_dir = os.path.join(dataset_dir, "ImageSets", "Main") 
    
    # 确保保存目录存在
    os.makedirs(save_dir, exist_ok=True)
    
    train_num, val_num, test_num = split_voc_dataset(
        dataset_dir=dataset_dir,
        val_percent=0.0,  # 不划分验证集
        test_percent=0,   # 不划分测试集
        save_dir=save_dir)
    
    print(f"数据集分割完成！")
    print(f"训练集样本数: {train_num}")
    print(f"分割文件保存在: {save_dir}")