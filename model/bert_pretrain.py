from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from model import bert_base
from model import optimization
import tensorflow as tf
from tensorflow.contrib import slim

class BertPretrain:
    def __init__(self, config):
        self.config = config

    def _accuracy(self, labels, predictions, weights=None):
        if weights is None:
            weights = tf.ones_like(labels)
        weights = tf.cast(weights, tf.float32)
        equals = tf.cast(tf.equal(tf.cast(predictions, tf.int32), labels), tf.float32)
        numerator = tf.reduce_sum(weights*equals)
        denominator = tf.reduce_sum(weights) + 1e-32
        acc = numerator/denominator
        return acc*100.

    def _metric_fn(self, masked_lm_example_loss, masked_lm_log_probs, masked_lm_ids,
                  masked_lm_weights, next_sentence_example_loss,
                  next_sentence_log_probs, next_sentence_labels):
        """Computes the loss and accuracy of the model."""
        masked_lm_log_probs = tf.reshape(masked_lm_log_probs,[-1, masked_lm_log_probs.shape[-1]])
        masked_lm_predictions = tf.argmax(masked_lm_log_probs, axis=-1, output_type=tf.int32)
        masked_lm_example_loss = tf.reshape(masked_lm_example_loss, [-1])
        masked_lm_ids = tf.reshape(masked_lm_ids, [-1])
        masked_lm_weights = tf.reshape(masked_lm_weights, [-1])


        masked_lm_accuracy = self._accuracy(
            labels=masked_lm_ids,
            predictions=masked_lm_predictions,
            weights=masked_lm_weights)
        masked_lm_mean_loss = tf.reduce_mean(masked_lm_example_loss*masked_lm_weights)

        next_sentence_log_probs = tf.reshape(next_sentence_log_probs, [-1, next_sentence_log_probs.shape[-1]])
        next_sentence_predictions = tf.argmax(next_sentence_log_probs, axis=-1, output_type=tf.int32)
        next_sentence_labels = tf.reshape(next_sentence_labels, [-1])

        next_sentence_accuracy = self._accuracy(labels=next_sentence_labels, predictions=next_sentence_predictions)
        next_sentence_mean_loss = tf.reduce_mean(next_sentence_example_loss)

        return {
            "masked_lm_accuracy": masked_lm_accuracy,
            "masked_lm_loss": masked_lm_mean_loss,
            "next_sentence_accuracy": next_sentence_accuracy,
            "next_sentence_loss": next_sentence_mean_loss,
        }

    def model_fn(self, features, labels, mode):  # pylint: disable=unused-argument
        """The `model_fn` for TPUEstimator."""

        bert_config = self.config
        tf.logging.info("*** Features ***")
        for name in sorted(features.keys()):
            tf.logging.info("  name = %s, shape = %s" % (name, features[name].shape))

        qids = features['qids']
        input_ids = features["input_ids_list"]
        input_mask = features["input_mask"]
        segment_ids = features["segment_ids_list"]
        masked_lm_positions = features["masked_lm_positions"]
        masked_lm_ids = features["masked_lm_ids"]
        masked_lm_weights = tf.cast(features["masked_lm_weights"], tf.float32)
        next_sentence_labels = labels


        is_training = (mode == tf.estimator.ModeKeys.TRAIN)
        print('********************* is training : ',is_training)
        if is_training:
            assert (self.config["init_checkpoint"] is not False)

        model = bert_base.BertModel(
            config=bert_config,
            is_training=is_training,
            input_ids=input_ids,
            input_mask=input_mask,
            token_type_ids=segment_ids,
            use_one_hot_embeddings=bert_config["use_one_hot_embeddings"])

        (masked_lm_loss, masked_lm_example_loss, masked_lm_log_probs) =\
            get_masked_lm_output(
            bert_config, model.get_sequence_output(), model.get_embedding_table(),
            masked_lm_positions, masked_lm_ids, masked_lm_weights)

        (next_sentence_loss, next_sentence_example_loss,next_sentence_log_probs) =\
            get_cate_prediction_output(
            bert_config, model.get_sequence_output(), input_mask, next_sentence_labels)

        # only masked_lm_loss
        total_loss = masked_lm_loss + 2*next_sentence_loss

        eval_metrics = self._metric_fn(masked_lm_example_loss, masked_lm_log_probs, masked_lm_ids,
                                       masked_lm_weights, next_sentence_example_loss,
                                       next_sentence_log_probs, next_sentence_labels)

        tvars = tf.trainable_variables()

        initialized_variable_names = {}
        scaffold_fn = None
        if bert_config["init_checkpoint"]:
            (assignment_map, initialized_variable_names) = \
                bert_base.get_assignment_map_from_checkpoint(tvars, bert_config["init_checkpoint"])
            tf.train.init_from_checkpoint(bert_config["init_checkpoint"], assignment_map)

        tf.logging.info("**** Trainable Variables ****")
        for var in tvars:
            init_string = ""
            if var.name in initialized_variable_names:
                init_string = ", *INIT_FROM_CKPT*"
            tf.logging.info("  name = %s, shape = %s%s", var.name, var.shape,
                            init_string)

        output_spec = None
        if mode == tf.estimator.ModeKeys.TRAIN:
            optimizer = tf.train.AdamOptimizer(learning_rate=bert_config["learning_rate"])
            train_op = optimizer.minimize(total_loss, global_step=tf.train.get_global_step())

            output_spec = tf.estimator.EstimatorSpec(
                mode=mode,
                loss=total_loss,
                train_op=train_op,
                training_hooks=[
                    tf.train.LoggingTensorHook(
                    {
                        "loss": total_loss,
                        "step": tf.train.get_global_step(),
                        "masked_lm_acc": eval_metrics["masked_lm_accuracy"],
                        "masked_lm_loss" : masked_lm_loss,
                        "cate_pred_acc" : eval_metrics["next_sentence_accuracy"],
                        "cate_pred_loss" : next_sentence_loss
                    },
                    every_n_iter=100
                )
            ])
        elif mode == tf.estimator.ModeKeys.PREDICT:
            outputs = dict(oneid=qids)
            if bert_config['out_embedding_type'] == "sequence_output":
                outputs['out_embedding'] = model.get_sequence_output()
            elif bert_config['out_embedding_type'] == "pooled_output":
                outputs['out_embedding'] = model.get_pooled_output()
            else:
                raise ValueError("Not recognized output embedding")
            output_spec = tf.estimator.EstimatorSpec(
                    mode=tf.estimator.ModeKeys.PREDICT,
                    predictions=outputs
                    )
        else:
            raise ValueError("Only TRAIN and EVAL modes are supported: %s" % (mode))

        slim.model_analyzer.analyze_vars(tf.trainable_variables(), print_info=True)
        return output_spec


def get_masked_lm_output(bert_config, input_tensor, output_weights, positions,
                         label_ids, label_weights):
  """Get loss and log probs for the masked LM."""
  input_tensor = gather_indexes(input_tensor, positions)

  with tf.variable_scope("cls/predictions"):
    # We apply one more non-linear transformation before the output layer.
    # This matrix is not used after pre-training.
    with tf.variable_scope("transform"):
      input_tensor = tf.layers.dense(
          input_tensor,
          units=bert_config["hidden_size"],
          activation=bert_base.get_activation(bert_config["hidden_act"]),
          kernel_initializer=bert_base.create_initializer(bert_config["initializer_range"]))
      input_tensor = bert_base.layer_norm(input_tensor)

    # The output weights are the same as the input embeddings, but there is
    # an output-only bias for each token.
    output_bias = tf.get_variable(
        "output_bias",
        shape=[bert_config["vocab_size"]],
        initializer=tf.zeros_initializer())
    logits = tf.matmul(input_tensor, output_weights, transpose_b=True)
    logits = tf.nn.bias_add(logits, output_bias)
    log_probs = tf.nn.log_softmax(logits, dim=-1) + 1e-32
    label_ids = tf.reshape(label_ids, [-1])

    label_weights = tf.cast(label_weights, dtype=tf.float32)
    label_weights = tf.reshape(label_weights, [-1])

    one_hot_labels = tf.one_hot(label_ids, depth=bert_config["vocab_size"], dtype=tf.float32)

    # The `positions` tensor might be zero-padded (if the sequence is too
    # short to have the maximum number of predictions). The `label_weights`
    # tensor has a value of 1.0 for every real prediction and 0.0 for the
    # padding predictions.
    per_example_loss = -tf.reduce_sum(log_probs * one_hot_labels, axis=[-1])
    numerator = tf.reduce_sum(label_weights * per_example_loss)
    denominator = tf.reduce_sum(label_weights) + 1e-5
    loss = numerator / denominator

  return (loss, per_example_loss, log_probs)


def get_next_sentence_output(bert_config, input_tensor, labels):
  """Get loss and log probs for the next sentence prediction."""

  # Simple binary classification. Note that 0 is "next sentence" and 1 is
  # "random sentence". This weight matrix is not used after pre-training.
  with tf.variable_scope("cls/seq_relationship"):
    output_weights = tf.get_variable(
        "output_weights",
        shape=[2, bert_config["hidden_size"]],
        initializer=bert_base.create_initializer(bert_config["initializer_range"]))
    output_bias = tf.get_variable("output_bias", shape=[2], initializer=tf.zeros_initializer())

    logits = tf.matmul(input_tensor, output_weights, transpose_b=True)
    logits = tf.nn.bias_add(logits, output_bias)
    log_probs = tf.nn.log_softmax(logits, dim=-1)
    labels = tf.reshape(labels, [-1])
    one_hot_labels = tf.one_hot(labels, depth=2, dtype=tf.float32)
    per_example_loss = -tf.reduce_sum(one_hot_labels * log_probs, axis=-1)
    loss = tf.reduce_mean(per_example_loss)

    return (loss, per_example_loss, log_probs)

def get_cate_prediction_output(bert_config, input_tensor, input_mask, labels):
    with tf.variable_scope("cls/seq_relationship"):
        input_tensor = ave_pooling(input_tensor, tf.cast(input_mask,tf.float32))  # [b,e]
        # add one layer of dense for cate prediction
        input_tensor = tf.contrib.layers.fully_connected(input_tensor, num_outputs=bert_config["hidden_size"], activation_fn=tf.nn.relu)
        input_tensor = bert_base.layer_norm(input_tensor)

        output_weights = tf.get_variable(
            "output_weights",
            shape=[bert_config["cate_size"], bert_config["hidden_size"]],
            initializer=bert_base.create_initializer(bert_config["initializer_range"]))
        output_bias = tf.get_variable(
            "output_bias", shape=[bert_config["cate_size"]], initializer=tf.zeros_initializer())

        logits = tf.matmul(input_tensor, output_weights, transpose_b=True)
        logits = tf.nn.bias_add(logits, output_bias)
        log_probs = tf.nn.log_softmax(logits, dim =-1)
        labels = tf.reshape(labels, [-1])
        one_hot_labels = tf.one_hot(labels, depth=bert_config["cate_size"], dtype=tf.float32)
        per_example_loss = -tf.reduce_sum(one_hot_labels * log_probs, axis=-1)
        loss = tf.reduce_mean(per_example_loss)

        return (loss, per_example_loss, log_probs)


def gather_indexes(sequence_tensor, positions):
  """Gathers the vectors at the specific positions over a minibatch."""
  sequence_shape = bert_base.get_shape_list(sequence_tensor, expected_rank=3)
  batch_size = sequence_shape[0]
  seq_length = sequence_shape[1]
  width = sequence_shape[2]

  flat_offsets = tf.reshape(
      tf.range(0, batch_size, dtype=tf.int32) * seq_length, [-1, 1])
  flat_positions = tf.reshape(positions + flat_offsets, [-1])
  flat_sequence_tensor = tf.reshape(sequence_tensor,
                                    [batch_size * seq_length, width])
  output_tensor = tf.gather(flat_sequence_tensor, flat_positions)
  return output_tensor

def ave_pooling(embeddings, masks):
    # batch * length * 1
    multiplier = tf.expand_dims(masks, axis=-1)
    embeddings_sum = tf.reduce_sum(tf.multiply(multiplier, embeddings),axis=1)
    length = tf.expand_dims(tf.maximum(tf.reduce_sum(masks, axis=1), 1.0), axis=-1) + 1e-32
    embedding_avg = embeddings_sum / length
    return embedding_avg