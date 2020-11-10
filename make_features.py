import bert


# def make_features(dataset, label_list, MAX_SEQ_LENGTH, tokenizer, DATA_COLUMN, LABEL_COLUMN):
#     input_example = dataset.apply(lambda x: bertrun_classifier.InputExample(guid=None,
#                                                                    text_a = x[DATA_COLUMN],
#                                                                    text_b = None,
#                                                                    label = x[LABEL_COLUMN]), axis = 1)
#     features = bert.run_classifier.convert_examples_to_features(input_example, label_list, MAX_SEQ_LENGTH, tokenizer)
#     return features

def make_features(dataset, label_list, MAX_SEQ_LENGTH, tokenizer, LABEL_COLUMN, DATA_COLUMN, DATA_COLUMN2 = None):
    if DATA_COLUMN2:
        input_example = dataset.apply(lambda x: bert.run_classifier.InputExample(guid=None,
                                                                       text_a = x[DATA_COLUMN],
                                                                       text_b = x[DATA_COLUMN2],
                                                                       label = x[LABEL_COLUMN]), axis = 1)
    else:
        input_example = dataset.apply(lambda x: bert.run_classifier.InputExample(guid=None,
                                                                       text_a = x[DATA_COLUMN],
                                                                       text_b = None,
                                                                       label = x[LABEL_COLUMN]), axis = 1)

    features = bert.run_classifier.convert_examples_to_features(input_example, label_list, MAX_SEQ_LENGTH, tokenizer)

    return features