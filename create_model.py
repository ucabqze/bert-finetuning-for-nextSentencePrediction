import tensorflow as tf
import tensorflow_hub as hub

import bert
from bert import modeling


def create_model(bert_config, is_predicting, input_ids, input_mask, segment_ids,
                 labels, num_labels, use_one_hot_embeddings = False):
  """Creates a classification model."""
  model = modeling.BertModel(
      config=bert_config,
      is_training=not is_predicting,
      input_ids=input_ids,
      input_mask=input_mask,
      token_type_ids=segment_ids,
      use_one_hot_embeddings=use_one_hot_embeddings)

  # In the demo, we are doing a simple classification task on the entire
  # segment.
  #
  # If you want to use the token-level output, use model.get_sequence_output()
  # instead.
  embedding_layer = model.get_sequence_output()
  output_layer = model.get_pooled_output()

  hidden_size = output_layer.shape[-1].value

  output_weights = tf.get_variable(
      "output_weights", [num_labels, hidden_size],
      initializer=tf.truncated_normal_initializer(stddev=0.02))

  output_bias = tf.get_variable(
      "output_bias", [num_labels], initializer=tf.zeros_initializer())

  with tf.variable_scope("loss"):
    if not is_predicting:
      # I.e., 0.1 dropout
      output_layer = tf.nn.dropout(output_layer, keep_prob=0.9)

    logits = tf.matmul(output_layer, output_weights, transpose_b=True)
    logits = tf.nn.bias_add(logits, output_bias)
    probabilities = tf.nn.softmax(logits, axis=-1)
    log_probs = tf.nn.log_softmax(logits, axis=-1)

    predicted_labels = tf.squeeze(tf.argmax(log_probs, axis=-1, output_type=tf.int32))

    # one_hot_labels = tf.one_hot(labels, depth=num_labels, dtype=tf.float32)
    #
    # per_example_loss = -tf.reduce_sum(one_hot_labels * log_probs, axis=-1)
    # loss = tf.reduce_mean(per_example_loss)
    if labels is not None:
        one_hot_labels = tf.one_hot(labels, depth=num_labels, dtype=tf.float32)

        per_example_loss = -tf.reduce_sum(one_hot_labels * log_probs, axis=-1)
        loss = tf.reduce_mean(per_example_loss)
    else:
        loss, per_example_loss = None, None

    #return (loss, per_example_loss, logits, probabilities)
    if not is_predicting:
        return (loss, predicted_labels, log_probs)
    else:
        return (predicted_labels, log_probs)



# def create_model_hub(bert_model_hub, is_predicting, input_ids, input_mask, segment_ids, labels,
#                  num_labels):
#   """Creates a classification model."""
#
#   bert_module = hub.Module(
#       bert_model_hub,
#       trainable=True)
#   bert_inputs = dict(
#       input_ids=input_ids,
#       input_mask=input_mask,
#       segment_ids=segment_ids)
#   bert_outputs = bert_module(
#       inputs=bert_inputs,
#       signature="tokens",
#       as_dict=True)
#
#   # Use "pooled_output" for classification tasks on an entire sentence.
#   # Use "sequence_outputs" for token-level output.
#   output_layer = bert_outputs["pooled_output"]
#
#   hidden_size = output_layer.shape[-1].value
#
#   # Create our own layer to tune for politeness data.
#   output_weights = tf.get_variable(
#       "output_weights", [num_labels, hidden_size],
#       initializer=tf.truncated_normal_initializer(stddev=0.02))
#
#   output_bias = tf.get_variable(
#       "output_bias", [num_labels], initializer=tf.zeros_initializer())
#
#   with tf.variable_scope("loss"):
#
#     # Dropout helps prevent overfitting
#     output_layer = tf.nn.dropout(output_layer, keep_prob=0.9)
#
#     logits = tf.matmul(output_layer, output_weights, transpose_b=True)
#     logits = tf.nn.bias_add(logits, output_bias)
#     log_probs = tf.nn.log_softmax(logits, axis=-1)
#
#     # Convert labels into one-hot encoding
#     one_hot_labels = tf.one_hot(labels, depth=num_labels, dtype=tf.float32)
#
#     predicted_labels = tf.squeeze(tf.argmax(log_probs, axis=-1, output_type=tf.int32))
#     # If we're predicting, we want predicted labels and the probabiltiies.
#     if is_predicting:
#       return (predicted_labels, log_probs)
#
#     # If we're train/eval, compute loss between predicted and actual label
#     per_example_loss = -tf.reduce_sum(one_hot_labels * log_probs, axis=-1)
#     loss = tf.reduce_mean(per_example_loss)
#     return (loss, predicted_labels, log_probs)
#
