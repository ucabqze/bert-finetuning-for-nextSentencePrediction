# bert-finetuning-for-nextSentencePrediction
# main reference: https://medium.com/serendeepia/finetuning-bert-with-tensorflow-estimators-in-only-a-few-lines-of-code-f522dfa2295c

# 'uncased_L-12_H-768_A-12' downloaded from https://github.com/google-research/bert / https://huggingface.co/models?filter=bert
# fine-tune the BERT model:  main.py - run_dfs.py
# get pb model: freeze_grapgh.py  (reference git: https://github.com/hiroLinGoing/bert/blob/master/freeze_graph.py)
# Inference: predict.py / predict2.py

# After train the model, the finetuned model would be saved in results/BERT_Imdb, the pb model would also be saved in this path by running freeze_grapgh.py
