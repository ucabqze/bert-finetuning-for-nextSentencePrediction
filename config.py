# trainig data dir
sentiment_training_data_dir = "/data/bisize/zengqiqi/bert-finetuning/training_data/imdb-sample.pickle"
NextSentence_pos_data_dir = "/data/bisize/zengqiqi/bert-finetuning/training_data/pos_1819.txt"
NextSentence_neg_data_dir = "/data/bisize/zengqiqi/bert-finetuning/training_data/neg_1819.txt"

# pretrain_model
base_uncased_dir = "/data/bisize/zengqiqi/bert-finetuning/uncased_L-12_H-768_A-12/"

# finetuning model
fine_tuning_model_dir = '/data/bisize/zengqiqi/bert-finetuning/results/BERT_Imdb/' # 这里注意请写绝对路径否则会有tensorflow.python.framework.errors_impl.PermissionDeniedError
fine_tuning_pb_model_dir = '/data/bisize/zengqiqi/bert-finetuning/results/BERT_Imdb/saved_model.pb'

# mode params
MAX_SEQ_LENGTH = 128
label_list = [1, 0]


from easydict import EasyDict as edict

cfg = edict()
# trainig data dir
cfg.sentiment_training_data_dir = "/data/bisize/zengqiqi/bert-finetuning/training_data/imdb-sample.pickle"
cfg.NextSentence_pos_data_dir = "/data/bisize/zengqiqi/bert-finetuning/training_data/pos_1819.txt"
cfg.NextSentence_neg_data_dir = "/data/bisize/zengqiqi/bert-finetuning/training_data/neg_1819.txt"

# pretrain_model
cfg.base_uncased_dir = "/data/bisize/zengqiqi/bert-finetuning/uncased_L-12_H-768_A-12/"

# finetuning model
cfg.fine_tuning_model_dir = '/data/bisize/zengqiqi/bert-finetuning/results/BERT_Imdb/' # 这里注意请写绝对路径否则会有tensorflow.python.framework.errors_impl.PermissionDeniedError
cfg.fine_tuning_pb_model_dir = '/data/bisize/zengqiqi/bert-finetuning/results/BERT_Imdb/saved_model.pb'

# mode params
cfg.MAX_SEQ_LENGTH = 128
cfg.label_list = [1, 0]




