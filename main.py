import os
import pickle
from run_dfs import run_on_dfs
from read_data import get_train_test_data
from config import sentiment_training_data_dir, fine_tuning_model_dir, \
    NextSentence_pos_data_dir, NextSentence_neg_data_dir

os.environ["CUDA_VISIBLE_DEVICES"] = "3"

# with open(sentiment_training_data_dir, 'rb') as f:
#     train, test = pickle.load(f)
# train = train.sample(len(train))
#
# myparam = {
#         "DATA_COLUMN": "text",
#         "LABEL_COLUMN": "sentiment",
#         "LEARNING_RATE": 2e-5,
#         "NUM_TRAIN_EPOCHS":10
#     }

train, test = get_train_test_data(NextSentence_pos_data_dir, NextSentence_neg_data_dir)
myparam = {
        "DATA_COLUMN": "text_1",
        "DATA_COLUMN2": "text_2",
        "LABEL_COLUMN": "label",
        "LEARNING_RATE": 2e-5,
        "NUM_TRAIN_EPOCHS":10
    }

OUTPUT_DIR = fine_tuning_model_dir  # 这里注意请写绝对路径否则会有tensorflow.python.framework.errors_impl.PermissionDeniedError
result, estimator = run_on_dfs(OUTPUT_DIR, train, test, **myparam)
# print(result)
#
# DATA_COLUMN = "text_1"
# DATA_COLUMN2= "text_2"
# LABEL_COLUMN = "label"
# MAX_SEQ_LENGTH = 128  # 最大应该可以到520
# BATCH_SIZE = 32
# LEARNING_RATE = 2e-5
# NUM_TRAIN_EPOCHS = 3.0
# WARMUP_PROPORTION = 0.1
# SAVE_SUMMARY_STEPS = 100
# SAVE_CHECKPOINTS_STEPS = 10000
# from config import base_uncased_dir
# bert_model_local = base_uncased_dir
