import os.path as osp
import random
import xml.etree.ElementTree as ET
import os
from PIL import Image
from flask import logging
import numpy as np

def is_pic(file_name):
    return file_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))

def replace_ext(file_name, new_ext):
    return osp.splitext(file_name)[0] + '.' + new_ext

def list_files(directory):
    return [f for f in os.listdir(directory) if osp.isfile(osp.join(directory, f))]

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
    
    # 生成val_list.txt
    with open(
            osp.join(save_dir, 'val.txt'), mode='w',
            encoding='utf-8') as f:
        for x in image_anno_list:
            file = osp.join("JPEGImages", x[0])
            label = osp.join("Annotations", x[1])
            f.write('{} {}\n'.format(file, label))
            
    # 保存labels.txt
    with open(
            osp.join(save_dir, 'labels.txt'), mode='w', encoding='utf-8') as f:
        for l in sorted(label_list):
            f.write('{}\n'.format(l))

    return image_num

if __name__ == '__main__':
    # 设置本地路径
    dataset_dir = r"D:\dataset\val"
    save_dir = os.path.join(dataset_dir, "ImageSets", "Main") 
    
    # 确保保存目录存在
    os.makedirs(save_dir, exist_ok=True)
    
    val_num = split_voc_dataset(
        dataset_dir=dataset_dir,
        val_percent=1.0,  # 全部作为验证集
        test_percent=0,   # 不划分测试集
        save_dir=save_dir)
    
    print(f"验证集分割完成！")
    print(f"验证集样本数: {val_num}")
    print(f"分割文件保存在: {save_dir}")
