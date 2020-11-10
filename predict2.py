import bert
import os
import tensorflow as tf
import numpy as np
import tensorflow.contrib.tensorrt as trt
from tensorflow.python.framework import graph_util
# from bert import tokenization
from tensorflow.python.platform import gfile
from bert import run_classifier
from bert_package import tokenization

from config import cfg
from read_data import get_train_test_data

os.environ["CUDA_VISIBLE_DEVICES"] = "3"

sess_config = tf.ConfigProto(device_count={'gpu': 3})
sess_config.allow_soft_placement = True
sess_config.gpu_options.per_process_gpu_memory_fraction = 0.6
sess_config.gpu_options.allow_growth = True
sess_config.log_device_placement = True


# def model_predict(id, mask, seg_id, pb_path):
#     with tf.Session() as sess:
#         tf.global_variables_initializer().run()
#         output_graph_def = tf.GraphDef()
#         with open(pb_path,"rb") as f:
#             output_graph_def.ParseFromString(f.read())
#             x = tf.import_graph_def(output_graph_def, name="fine-tuned-bert_1")
#
#         # ops = [o for o in sess.graph.get_operations()]
#         # for o in ops:
#         #     print(o.name)
#
#         input_ids = sess.graph.get_tensor_by_name('fine-tuned-bert_1/input_ids:0')
#         input_mask = sess.graph.get_tensor_by_name('fine-tuned-bert_1/input_mask:0')
#         segment_ids = sess.graph.get_tensor_by_name('fine-tuned-bert_1/segment_ids:0')
#
#         operation_ = sess.graph.get_tensor_by_name('fine-tuned-bert_1/log_prob:0')
#
#         feed_dict = {input_ids:id,input_mask:mask,segment_ids:seg_id}
#         output_ = (sess.run(operation_,feed_dict))
#
#     return output_
#
# def get_model_graph(pb_path):
#
#     graph_bertClassifier = tf.Graph()
#     sess_bert = tf.Session(graph=graph_bertClassifier, config=sess_config)
#     # tf.global_variables_initializer().run()
#     with gfile.FastGFile(pb_path, "rb") as f:
#         output_graph_def = tf.GraphDef()
#         output_graph_def.ParseFromString(f.read())
#         # x = tf.import_graph_def(output_graph_def, name="fine-tuned-bert_1")
#
#         trt_bert_graph = trt.create_inference_graph(
#             input_graph_def=output_graph_def,
#             outputs=['fine-tuned-bert_1/log_prob'],
#             # is_dynamic_op=True,
#             # maximum_cached_engines=1,
#             precision_mode="INT8")  # FP32 FP16 INT8
#
#     with graph_bertClassifier.as_default():
#         tf.import_graph_def(trt_bert_graph, name='')  # 导入计算图
#
#
#     # ops = [o for o in sess.graph.get_operations()]
#     # for o in ops:
#     #     print(o.name)
#
#     input_ids = sess_bert.graph.get_tensor_by_name('fine-tuned-bert_1/input_ids:0')
#     input_mask = sess_bert.graph.get_tensor_by_name('fine-tuned-bert_1/input_mask:0')
#     segment_ids = sess_bert.graph.get_tensor_by_name('fine-tuned-bert_1/segment_ids:0')
#
#     operation_ = sess_bert.graph.get_tensor_by_name('fine-tuned-bert_1/log_prob:0')
#
#     return sess_bert, input_ids, input_mask, segment_ids, operation_

def get_model_graph(pb_path):
    # graph_bertClassifier = tf.Graph()
    # sess = tf.Session(graph=graph_bertClassifier, config=sess_config)
    sess = tf.Session()

    with gfile.FastGFile(pb_path,"rb") as f:
        output_graph_def = tf.GraphDef()
        output_graph_def.ParseFromString(f.read())
        sess.graph.as_default()
        tf.import_graph_def(output_graph_def, name="fine-tuned-bert_1")

        # ops = [o for o in sess.graph.get_operations()]
        # for o in ops:
        #     print(o.name)
    sess.run(tf.global_variables_initializer())

    input_ids = sess.graph.get_tensor_by_name('fine-tuned-bert_1/input_ids:0')
    input_mask = sess.graph.get_tensor_by_name('fine-tuned-bert_1/input_mask:0')
    segment_ids = sess.graph.get_tensor_by_name('fine-tuned-bert_1/segment_ids:0')

    operation_ = sess.graph.get_tensor_by_name('fine-tuned-bert_1/log_prob:0')

    return sess, input_ids, input_mask, segment_ids, operation_

def model_predict(id, mask, seg_id, sess, input_ids, input_mask, segment_ids, operation_):

    feed_dict = {input_ids: id, input_mask: mask, segment_ids: seg_id}
    output_ = (sess.run(operation_, feed_dict))

    return output_


# pb_path = '/data/bisize/zengqiqi/bert-finetuning/results/BERT_Imdb/saved_model.pb'
# id = [train_features[1].input_ids]
# mask = [train_features[1].input_mask]
# seg_id = [train_features[1].segment_ids]
#


def convert_sent_to_vec(tokenizer, dataset, DATA_COLUMN='text_1', DATA_COLUMN2='text_2'):
    label_list = [1, 0]

    input_example = dataset.apply(lambda x: run_classifier.InputExample(guid=None,
                                                                        text_a=x[DATA_COLUMN],
                                                                        text_b=x[DATA_COLUMN2],
                                                                        label=1), axis=1)
    features = run_classifier.convert_examples_to_features(input_example, cfg.label_list, cfg.MAX_SEQ_LENGTH, tokenizer)

    return features


def next_sentence_predict(test, sess, input_ids, input_mask, segment_ids, operation_ ):
    tokenizer = tokenization.FullTokenizer(
        vocab_file=cfg.base_uncased_dir + 'vocab.txt', do_lower_case=True)

    features = convert_sent_to_vec(tokenizer, test)

    id = [x.input_ids for x in features]
    mask = [x.input_mask for x in features]
    seg_id = [x.segment_ids for x in features]

    # log_pro = model_predict(id, mask, seg_id, cfg.fine_tuning_pb_model_dir)
    log_pro = model_predict(id, mask, seg_id, sess, input_ids, input_mask, segment_ids, operation_)
    predicted_labels = np.argmax(log_pro, axis=1)

    return predicted_labels


train, test = get_train_test_data(cfg.NextSentence_pos_data_dir, cfg.NextSentence_neg_data_dir)
sess, input_ids, input_mask, segment_ids, operation_ = get_model_graph(cfg.fine_tuning_pb_model_dir)
predicted_labels = next_sentence_predict(test, sess, input_ids, input_mask, segment_ids, operation_ )

for i in range(test.shape[0]):
    print("-----------------------------")
    print(predicted_labels[i])
    print(test.iloc[i].text_1)
    print(test.iloc[i].text_2)
