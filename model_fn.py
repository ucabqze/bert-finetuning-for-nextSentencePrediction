import tensorflow as tf
import bert

from create_model import create_model

# def model_fn_builder(bert_model_hub, num_labels, learning_rate, num_train_steps,
#                      num_warmup_steps):
def model_fn_builder(bert_config, num_labels, learning_rate, num_train_steps,
                     num_warmup_steps):
    """Returns `model_fn` closure for TPUEstimator."""


    def model_fn(features, labels, mode, params):  # pylint: disable=unused-argument
        """The `model_fn` for TPUEstimator."""

        input_ids = features["input_ids"]
        input_mask = features["input_mask"]
        segment_ids = features["segment_ids"]
        label_ids = features["label_ids"]

        is_predicting = (mode == tf.estimator.ModeKeys.PREDICT)

        # TRAIN and EVAL
        if not is_predicting:
            with tf.Session() as sess:

                input_ids = tf.placeholder(tf.int32, (None, 128), 'input_ids')
                input_mask = tf.placeholder(tf.int32, (None, 128), 'input_mask')

                (loss, predicted_labels, log_probs) = create_model(
                    bert_config, is_predicting, input_ids, input_mask, segment_ids, label_ids, num_labels)

                train_op = bert.optimization.create_optimizer(
                    loss, learning_rate, num_train_steps, num_warmup_steps, use_tpu=False)

            # Calculate evaluation metrics.
            def metric_fn(label_ids, predicted_labels):
                accuracy = tf.metrics.accuracy(label_ids, predicted_labels)
                f1_score = tf.contrib.metrics.f1_score(
                    label_ids,
                    predicted_labels)
                auc = tf.metrics.auc(
                    label_ids,
                    predicted_labels)
                recall = tf.metrics.recall(
                    label_ids,
                    predicted_labels)
                precision = tf.metrics.precision(
                    label_ids,
                    predicted_labels)
                true_pos = tf.metrics.true_positives(
                    label_ids,
                    predicted_labels)
                true_neg = tf.metrics.true_negatives(
                    label_ids,
                    predicted_labels)
                false_pos = tf.metrics.false_positives(
                    label_ids,
                    predicted_labels)
                false_neg = tf.metrics.false_negatives(
                    label_ids,
                    predicted_labels)
                return {
                    "eval_accuracy": accuracy,
                    "f1_score": f1_score,
                    "auc": auc,
                    "precision": precision,
                    "recall": recall,
                    "true_positives": true_pos,
                    "true_negatives": true_neg,
                    "false_positives": false_pos,
                    "false_negatives": false_neg
                }

            eval_metrics = metric_fn(label_ids, predicted_labels)

            if mode == tf.estimator.ModeKeys.TRAIN:
                return tf.estimator.EstimatorSpec(mode=mode,
                                                  loss=loss,
                                                  train_op=train_op)
            else:
                return tf.estimator.EstimatorSpec(mode=mode,
                                                  loss=loss,
                                                  eval_metric_ops=eval_metrics)
        else:
            # (predicted_labels, log_probs) = create_model(
            #     bert_model_hub, is_predicting, input_ids, input_mask, segment_ids, label_ids, num_labels)
            (predicted_labels, log_probs) = create_model(
                bert_config, is_predicting, input_ids, input_mask, segment_ids, label_ids, num_labels)

            predictions = {
                'probabilities': log_probs,
                'labels': predicted_labels
            }
            return tf.estimator.EstimatorSpec(mode, predictions=predictions)

    # Return the actual model function in the closure
    return model_fn


def estimator_builder(bert_config, OUTPUT_DIR, SAVE_SUMMARY_STEPS, SAVE_CHECKPOINTS_STEPS, label_list, LEARNING_RATE,
                      num_train_steps, num_warmup_steps, BATCH_SIZE):
    # Specify outpit directory and number of checkpoint steps to save
    run_config = tf.estimator.RunConfig(
        model_dir=OUTPUT_DIR,
        save_summary_steps=SAVE_SUMMARY_STEPS,
        save_checkpoints_steps=SAVE_CHECKPOINTS_STEPS)

    model_fn = model_fn_builder(
        bert_config=bert_config,
        num_labels=len(label_list),
        learning_rate=LEARNING_RATE,
        num_train_steps=num_train_steps,
        num_warmup_steps=num_warmup_steps)

    estimator = tf.estimator.Estimator(
        model_fn=model_fn,
        config=run_config,
        params={"batch_size": BATCH_SIZE})

    return estimator, model_fn, run_config

# def estimator_builder(bert_model_hub, OUTPUT_DIR, SAVE_SUMMARY_STEPS, SAVE_CHECKPOINTS_STEPS, label_list, LEARNING_RATE,
#                       num_train_steps, num_warmup_steps, BATCH_SIZE):
#     # Specify outpit directory and number of checkpoint steps to save
#     run_config = tf.estimator.RunConfig(
#         model_dir=OUTPUT_DIR,
#         save_summary_steps=SAVE_SUMMARY_STEPS,
#         save_checkpoints_steps=SAVE_CHECKPOINTS_STEPS)
#
#     model_fn = model_fn_builder(
#         bert_model_hub=bert_model_hub,
#         num_labels=len(label_list),
#         learning_rate=LEARNING_RATE,
#         num_train_steps=num_train_steps,
#         num_warmup_steps=num_warmup_steps)
#
#     estimator = tf.estimator.Estimator(
#         model_fn=model_fn,
#         config=run_config,
#         params={"batch_size": BATCH_SIZE})
#     return estimator, model_fn, run_config