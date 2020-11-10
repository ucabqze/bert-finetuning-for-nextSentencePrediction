import bert
from bert import run_classifier, modeling
# from bert import tokenization
from bert_package import tokenization

from create_tokenizer import create_tokenizer_from_hub_module
from make_features import make_features
from model_fn import estimator_builder
from config import base_uncased_dir


def run_on_dfs(OUTPUT_DIR, train, test, DATA_COLUMN, LABEL_COLUMN, DATA_COLUMN2 = None,
               MAX_SEQ_LENGTH=128, # 最大应该可以到520
               BATCH_SIZE=32,
               LEARNING_RATE=2e-5,
               NUM_TRAIN_EPOCHS=3.0,
               WARMUP_PROPORTION=0.1,
               SAVE_SUMMARY_STEPS=100,
               SAVE_CHECKPOINTS_STEPS=10000,
               bert_model_hub="https://tfhub.dev/google/bert_uncased_L-12_H-768_A-12/1",
               # bert_model_local = "/Users/cengqiqi/Desktop/bert-finetuining/uncased_L-12_H-768_A-12/",
               bert_model_local = base_uncased_dir):

    label_list = train[LABEL_COLUMN].unique().tolist()

    # tokenizer = create_tokenizer_from_hub_module(bert_model_hub)
    tokenizer = tokenization.FullTokenizer(
        vocab_file= bert_model_local+'vocab.txt', do_lower_case=True)

    train_features = make_features(train, label_list, MAX_SEQ_LENGTH, tokenizer, LABEL_COLUMN, DATA_COLUMN, DATA_COLUMN2)
    test_features = make_features(test, label_list, MAX_SEQ_LENGTH, tokenizer, LABEL_COLUMN, DATA_COLUMN, DATA_COLUMN2)

    num_train_steps = int(len(train_features) / BATCH_SIZE * NUM_TRAIN_EPOCHS)
    num_warmup_steps = int(num_train_steps * WARMUP_PROPORTION)

    bert_config_file = bert_model_local+'config.json'
    bert_config = modeling.BertConfig.from_json_file(bert_config_file)

    estimator, model_fn, run_config = estimator_builder(
        bert_config,
        OUTPUT_DIR,
        SAVE_SUMMARY_STEPS,
        SAVE_CHECKPOINTS_STEPS,
        label_list,
        LEARNING_RATE,
        num_train_steps,
        num_warmup_steps,
        BATCH_SIZE)

    train_input_fn = bert.run_classifier.input_fn_builder(
        features=train_features,
        seq_length=MAX_SEQ_LENGTH,
        is_training=True,
        drop_remainder=False)

    estimator.train(input_fn='train_input_fn', max_steps=num_train_steps)   # INFO:tensorflow:Skipping training since max_steps has already saved.

    test_input_fn = run_classifier.input_fn_builder(
        features=test_features,
        seq_length=MAX_SEQ_LENGTH,
        is_training=False,
        drop_remainder=False)

    result_dict = estimator.evaluate(input_fn=test_input_fn, steps=None)

    # test_input_fn = run_classifier.input_fn_builder(
    #     features=test_features,
    #     seq_length=MAX_SEQ_LENGTH,
    #     is_training=False,
    #     drop_remainder=False)
    #
    # result_dict = estimator.predict(input_fn=test_input_fn)

    return result_dict, estimator

