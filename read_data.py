import random
import math
import pandas as pd

def read_from_txt(filepath):
    with open(filepath,"r") as f:
        lines = f.readlines()
        sample_num = len(lines)//4
        sample_data = []
        for i in range(sample_num):
            text_1 = lines[i*4]
            text_2 = lines[i*4+1]
            label = lines[i*4+2]
            break_sign = lines[i*4+3]

            try:
                label = int(label)
            except:
                print("Error occurred while reading the data from ", NextSentence_pos_data_dir)
                print("At around the line", i*4)
            else:
                if break_sign == '\n':
                    sample_data.append((text_1, text_2, label))
    return sample_data

# Balance the pos and neg data
def balance_pos_and_neg(pos_data, neg_data):
    if len(pos_data)<len(neg_data):
        neg_data = random.sample(neg_data, len(pos_data))
        random.shuffle(pos_data)
    else:
        pos_data = random.sample(pos_data, len(neg_data))
        random.shuffle(neg_data)
    return pos_data, neg_data

def list_to_dataframe(list_data):

    sample_df = pd.DataFrame(columns=['text_1','text_2','label'])
    for t1, t2, l in list_data:
        sample_df = sample_df.append(pd.DataFrame({'text_1':[t1],
                                       'text_2':[t2],
                                       'label':[l]}),ignore_index=True)
    sample_df = sample_df.sample(len(sample_df))

    return sample_df

def divide_train_test(sample_df, tt_ratio = 0.9):

    split_idx = math.ceil(sample_df.shape[0]*0.7)
    train = sample_df.iloc[:split_idx]
    test = sample_df.iloc[split_idx:]

    return train, test

def get_train_test_data(NextSentence_pos_data_dir, NextSentence_neg_data_dir):

    pos_data = read_from_txt(NextSentence_pos_data_dir)
    neg_data = read_from_txt(NextSentence_neg_data_dir)
    pos_data, neg_data = balance_pos_and_neg(pos_data, neg_data)

    sample_df = list_to_dataframe(pos_data+neg_data)
    train, test = divide_train_test(sample_df, tt_ratio = 0.9)

    return train, test