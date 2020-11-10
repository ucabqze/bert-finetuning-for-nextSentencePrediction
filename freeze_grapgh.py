import json
import os
import sys
import logging
import pickle
import argparse
import shutil
import tensorflow as tf
from bert import modeling
from enum import Enum
from termcolor import colored

from create_model import create_model


def set_logger(context, verbose=False):
    if os.name == 'nt':  # for Windows
        return NTLogger(context, verbose)

    logger = logging.getLogger(context)
    logger.setLevel(logging.DEBUG if verbose else logging.INFO)
    formatter = logging.Formatter(
        '%(levelname)-.1s:' + context + ':[%(filename).3s:%(funcName).3s:%(lineno)3d]:%(message)s', datefmt=
        '%m-%d %H:%M:%S')
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG if verbose else logging.INFO)
    console_handler.setFormatter(formatter)
    logger.handlers = []
    logger.addHandler(console_handler)
    return logger


class NTLogger:
    def __init__(self, context, verbose):
        self.context = context
        self.verbose = verbose

    def info(self, msg, **kwargs):
        print('I:%s:%s' % (self.context, msg), flush=True)

    def debug(self, msg, **kwargs):
        if self.verbose:
            print('D:%s:%s' % (self.context, msg), flush=True)

    def error(self, msg, **kwargs):
        print('E:%s:%s' % (self.context, msg), flush=True)

    def warning(self, msg, **kwargs):
        print('W:%s:%s' % (self.context, msg), flush=True)


def init_predict_var(path):
    label2id_file = os.path.join(path, 'label2id.pkl')
    if os.path.exists(label2id_file):
        with open(label2id_file, 'rb') as rf:
            label2id = pickle.load(rf)
            id2label = {value: key for key, value in label2id.items()}
            num_labels = len(label2id.items())
    return num_labels, label2id, id2label


def optimize_class_model(args, logger=None):
    if not logger:
        logger = set_logger(colored('CLASSIFICATION_MODEL, Lodding...', 'cyan'), args.verbose)
        pass
    try:
        # 如果PB文件已经存在则，返回PB文件的路径，否则将模型转化为PB文件，并且返回存储PB文件的路径
        if args.model_pb_dir is None:
            tmp_file = args.model_dir
        else:
            tmp_file = args.model_pb_dir

        pb_file = os.path.join(tmp_file, 'saved_model.pb')
        if os.path.exists(pb_file) and (args.pb_file_name == '' or args.pb_version == ''):
            print('pb_file exits', pb_file)
            return pb_file

        # 增加 从label2id.pkl中读取num_labels, 这样也可以不用指定num_labels参数； 2019/4/17
        if not args.num_labels:
            num_labels, label2id, id2label = init_predict_var()
        else:
            num_labels = args.num_labels

        graph = tf.Graph()
        with graph.as_default():
            with tf.Session() as sess:

                input_ids = tf.placeholder(tf.int32, (None, args.max_seq_len), 'input_ids')
                input_mask = tf.placeholder(tf.int32, (None, args.max_seq_len), 'input_mask')
                segment_ids = tf.placeholder(tf.int32, (None, args.max_seq_len), 'segment_ids')

                bert_config = modeling.BertConfig.from_json_file(os.path.join(args.bert_model_dir, 'bert_config.json'))

                (predicted_labels, log_probs) = create_model(
                    bert_config = bert_config,
                    is_predicting = True,
                    input_ids = input_ids,
                    input_mask = input_mask,
                    segment_ids = segment_ids,
                    labels = None,
                    num_labels = 2)


                # pred_ids = tf.argmax(probabilities, axis=-1, output_type=tf.int32, name='pred_ids')
                # pred_ids = tf.identity(pred_ids, 'pred_ids')

                probabilities = tf.identity(log_probs, 'log_prob')
                saver = tf.train.Saver()

            with tf.Session() as sess:
                sess.run(tf.global_variables_initializer())
                latest_checkpoint = tf.train.latest_checkpoint(args.model_dir)
                logger.info('loading... %s ' % latest_checkpoint)
                saver.restore(sess, latest_checkpoint)
                logger.info('freeze...')
                from tensorflow.python.framework import graph_util
                tmp_g = graph_util.convert_variables_to_constants(sess, graph.as_graph_def(), ['log_prob'])
                logger.info('predict cut finished !!!')

        # 存储二进制模型到文件中
        logger.info('write graph to a tmp file: %s' % pb_file)
        with tf.gfile.GFile(pb_file, 'wb') as f:
            f.write(tmp_g.SerializeToString())

        #如果pb_file_name pb_version存在则直接生成tfserving格式
        if args.pb_file_name == '' or args.pb_version == '':
            return pb_file

        pd_model_file = os.path.join(os.path.join(args.model_dir, args.pb_file_name), args.pb_version)
        if os.path.exists(pd_model_file):
            shutil.rmtree(pd_model_file)
            os.makedirs(pd_model_file)
        shutil.move(pb_file, pd_model_file)
        latest_checkpoint = tf.train.latest_checkpoint(args.model_dir)
        if latest_checkpoint == '':
            return pb_file
        data_path = os.path.join(os.path.join(args.model_dir, "{}.data-00000-of-00001".format(latest_checkpoint)))
        index_path = os.path.join(os.path.join(args.model_dir, "{}.index".format(latest_checkpoint)))
        pd_variables_path = os.path.join(pd_model_file, 'variables')
        os.makedirs(pd_variables_path)
        shutil.copy(data_path, os.path.join(pd_variables_path, "variables.data-00000-of-00001"))
        shutil.copy(index_path, os.path.join(pd_variables_path, "variables.index"))
        logger.info('pb dir success!!! {}'.format(pd_model_file))
        return pd_model_file
    except Exception as e:
        logger.error('fail to optimize the graph! %s' % e, exc_info=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Trans ckpt file to .pb file')
    parser.add_argument('-bert_model_dir', type=str, required=True,
                        help='chinese google bert model path')
    parser.add_argument('-model_dir', type=str, required=True,
                        help='directory of a pretrained BERT model')
    parser.add_argument('-model_pb_dir', type=str, default=None,
                        help='directory of a pretrained BERT model,default = model_dir')
    parser.add_argument('-max_seq_len', type=int, default=128,
                        help='maximum length of a sequence,default:128')
    parser.add_argument('-num_labels', type=int, default=2,
                        help='length of all labels,default=2')
    parser.add_argument('-verbose', action='store_true', default=False,
                        help='turn on tensorflow logging for debug')
    parser.add_argument('-pb_file_name', type=str, default='',
                        help='please input pb file name')
    parser.add_argument('-pb_version', type=str, default='',
                        help='please input pb version')
    args = parser.parse_args()
    optimize_class_model(args, logger=None)

# !python3 freeze_grapgh.py -bert_model_dir="/data/bisize/zengqiqi/bert-finetuning/uncased_L-12_H-768_A-12/"  -model_dir='/data/bisize/zengqiqi/bert-finetuning/results/BERT_Imdb/'


