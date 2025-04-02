# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys

# add python path of PaddleDetection to sys.path
parent_path = os.path.abspath(os.path.join(__file__, *(['..'] * 2)))
sys.path.insert(0, parent_path)

# ignore warning log
import warnings
warnings.filterwarnings('ignore')
import glob
import ast

import paddle
from ppdet.core.workspace import create, load_config, merge_config
from ppdet.engine import Trainer, Trainer_ARSL
from ppdet.utils.check import check_gpu, check_npu, check_xpu, check_mlu, check_gcu, check_version, check_config
from ppdet.utils.cli import ArgsParser, merge_args
from ppdet.slim import build_slim_model

from ppdet.utils.logger import setup_logger
logger = setup_logger('train')


def parse_args():
    parser = ArgsParser()
    parser.add_argument(
        "--infer_dir",
        type=str,
        default=r"D:\dataset\val\JPEGImages",  # 修改默认推理目录为验证集图片目录
        help="Directory for images to perform inference on.")
    parser.add_argument(
        "--infer_list",
        type=str,
        default=None,
        help="The file path containing path of image to be infered. Valid only when --infer_dir is given."
    )
    parser.add_argument(
        "--infer_img",
        type=str,
        default=None,
        help="Image path, has higher priority over --infer_dir")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="D:/dataset/work/infer_output",  # 修改默认推理输出目录
        help="Directory for storing the output visualization files.")
    parser.add_argument(
        "--draw_threshold",
        type=float,
        default=0.5,
        help="Threshold to reserve the result for visualization.")
    parser.add_argument(
        "--save_threshold",
        type=float,
        default=0.5,
        help="Threshold to reserve the result for saving.")
    parser.add_argument(
        "--slim_config",
        default=None,
        type=str,
        help="Configuration file of slim method.")
    parser.add_argument(
        "--use_vdl",
        type=bool,
        default=False,
        help="Whether to record the data to VisualDL.")
    parser.add_argument(
        "--do_eval",
        type=ast.literal_eval,
        default=False,
        help="Whether to eval after infer.")
    parser.add_argument(
        '--vdl_log_dir',
        type=str,
        default="D:/dataset/work/vdl_log_dir/image",  # 修改默认日志目录
        help='VisualDL logging directory for image.')
    parser.add_argument(
        "--save_results",
        type=bool,
        default=False,
        help="Whether to save inference results to output_dir.")
    parser.add_argument(
        "--slice_infer",
        action='store_true',
        help="Whether to slice the image and merge the inference results for small object detection."
    )
    parser.add_argument(
        '--slice_size',
        nargs='+',
        type=int,
        default=[640, 640],
        help="Height of the sliced image.")
    parser.add_argument(
        "--overlap_ratio",
        nargs='+',
        type=float,
        default=[0.25, 0.25],
        help="Overlap height ratio of the sliced image.")
    parser.add_argument(
        "--combine_method",
        type=str,
        default='nms',
        help="Combine method of the sliced images' detection results, choose in ['nms', 'nmm', 'concat']." 
    )
    parser.add_argument(
        "--match_threshold",
        type=float,
        default=0.6,
        help="Combine method matching threshold.")
    parser.add_argument(
        "--match_metric",
        type=str,
        default='ios',
        help="Combine method matching metric, choose in ['iou', 'ios'].")
    parser.add_argument(
        "--visualize",
        type=ast.literal_eval,
        default=True,
        help="Whether to save visualize results to output_dir.")
    parser.add_argument(
        "--rtn_im_file",
        type=bool,
        default=False,
        help="Whether to return image file path in Dataloader.")
    args = parser.parse_args()
    return args


def get_test_images(infer_dir, infer_img, infer_list=None):
    """
    Get image path list in TEST mode
    """
    assert infer_img is not None or infer_dir is not None, \
        "--infer_img or --infer_dir should be set"
    assert infer_img is None or os.path.isfile(infer_img), \
            "{} is not a file".format(infer_img)
    assert infer_dir is None or os.path.isdir(infer_dir), \
            "{} is not a directory".format(infer_dir)

    # infer_img has a higher priority
    if infer_img and os.path.isfile(infer_img):
        return [infer_img]

    images = set()
    infer_dir = os.path.abspath(infer_dir)
    assert os.path.isdir(infer_dir), \
        "infer_dir {} is not a directory".format(infer_dir)
    if infer_list:
        assert os.path.isfile(
            infer_list), f"infer_list {infer_list} is not a valid file path."
        with open(infer_list, 'r') as f:
            lines = f.readlines()
        for line in lines:
            images.update([os.path.join(infer_dir, line.strip())])
    else:
        exts = ['jpg', 'jpeg', 'png', 'bmp']
        exts += [ext.upper() for ext in exts]
        for ext in exts:
            images.update(glob.glob('{}/*.{}'.format(infer_dir, ext)))
    images = list(images)
    assert len(images) > 0, "no image found in {}".format(infer_dir)
    logger.info("Found {} inference images in total.".format(len(images)))

    return images


def run(FLAGS, cfg):
    print(f"使用配置文件: {FLAGS.config}")
    print(f"使用权重文件: {cfg.weights}")
    print(f"输出目录: {FLAGS.output_dir}")
    print(f"推理阈值: {FLAGS.draw_threshold}")

    if FLAGS.rtn_im_file:
        cfg['TestReader']['sample_transforms'][0]['Decode'][
            'rtn_im_file'] = FLAGS.rtn_im_file
    ssod_method = cfg.get('ssod_method', None)
    if ssod_method == 'ARSL':
        trainer = Trainer_ARSL(cfg, mode='test')
        trainer.load_weights(cfg.weights, ARSL_eval=True)
    else:
        trainer = Trainer(cfg, mode='test')
        trainer.load_weights(cfg.weights)
    # get inference images
    if FLAGS.do_eval:
        dataset = create('TestDataset')()
        images = dataset.get_images()
    else:
        images = get_test_images(FLAGS.infer_dir, FLAGS.infer_img, FLAGS.infer_list)
    print(f"共找到 {len(images)} 张待推理图片")

    # inference
    print("\n开始执行推理...")
    if FLAGS.slice_infer:
        trainer.slice_predict(
            images,
            slice_size=FLAGS.slice_size,
            overlap_ratio=FLAGS.overlap_ratio,
            combine_method=FLAGS.combine_method,
            match_threshold=FLAGS.match_threshold,
            match_metric=FLAGS.match_metric,
            draw_threshold=FLAGS.draw_threshold,
            output_dir=FLAGS.output_dir,
            save_results=FLAGS.save_results,
            visualize=FLAGS.visualize)
    else:
        trainer.predict(
            images,
            draw_threshold=FLAGS.draw_threshold,
            output_dir=FLAGS.output_dir,
            save_results=FLAGS.save_results,
            visualize=FLAGS.visualize,
            save_threshold=FLAGS.save_threshold,
            do_eval=FLAGS.do_eval)
    
    print("\n================ 推理完成 ================")
    print(f"结果已保存至: {FLAGS.output_dir}")


def infer():
    FLAGS = parse_args()
    
    # 设置配置文件路径
    if FLAGS.config is None:
        FLAGS.config = os.path.normpath(r"D:\programfiles\codes\Py_files\PaddleDetection-release-2.8.1\configs\ppq.yml")
    
    cfg = load_config(FLAGS.config)
    merge_args(cfg, FLAGS)
    merge_config(FLAGS.opt)

    # disable npu in config by default
    if 'use_npu' not in cfg:
        cfg.use_npu = False

    # disable xpu in config by default
    if 'use_xpu' not in cfg:
        cfg.use_xpu = False

    if 'use_gpu' not in cfg:
        cfg.use_gpu = False

    # disable mlu in config by default
    if 'use_mlu' not in cfg:
        cfg.use_mlu = False

    # disable gcu in config by default
    if 'use_gcu' not in cfg:
        cfg.use_gcu = False

    print("\n================ 推理环境配置 ================")
    if cfg.use_gpu:
        print("使用设备: GPU")
        place = paddle.set_device('gpu')
    elif cfg.use_npu:
        place = paddle.set_device('npu')
    elif cfg.use_xpu:
        place = paddle.set_device('xpu')
    elif cfg.use_mlu:
        place = paddle.set_device('mlu')
    elif cfg.use_gcu:
        place = paddle.set_device('gcu')
    else:
        print("使用设备: CPU")
        place = paddle.set_device('cpu')

    # 确保输出目录存在
    os.makedirs(FLAGS.output_dir, exist_ok=True)
    if FLAGS.use_vdl:
        os.makedirs(FLAGS.vdl_log_dir, exist_ok=True)

    if FLAGS.slim_config:
        cfg = build_slim_model(cfg, FLAGS.slim_config, mode='test')

    check_config(cfg)
    check_gpu(cfg.use_gpu)
    check_npu(cfg.use_npu)
    check_xpu(cfg.use_xpu)
    check_mlu(cfg.use_mlu)
    check_gcu(cfg.use_gcu)
    check_version()
    run(FLAGS, cfg)


if __name__ == '__main__':
    print("\n================ 开始推理 ================")
    infer()
