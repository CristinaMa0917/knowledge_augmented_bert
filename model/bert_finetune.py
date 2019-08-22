from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from model import bert_base
from model import optimization
import tensorflow as tf
from tensorflow.contrib import slim
from tensorflow.contrib import layers
from argparse import Namespace


class BertFinetune:
    def __init__(self, config):
        self.max_seq_len = config["finetune_seq_length"]
        self.max_que_len = config["finetune_query_length"]
        self.layers_num = config["finetune_layers"]
        self.classes = config["finetune_classes"]
        self.dp = config["finetune_dropout"]
        self.init_ckt = config["init_checkpoint"]
        self.output_emb = config["out_embedding_type"]
        self.bert_config = config

    def _build_model(self, features, labels, mode):
        tf.logging.info("*** Features ***")
        for name in sorted(features.keys()):
            tf.logging.info("  name = %s, shape = %s" % (name, features[name].shape))

        oneid = features["oneid"]
        input_ids = features["input_ids_list"]  # [b,s*q]
        input_mask = features["input_mask"]  # [b,s*q]
        labels = labels

        input_ids = tf.reshape(input_ids, [-1, self.max_seq_len])  # [b*q, s]
        input_mask = tf.reshape(input_mask, [-1, self.max_seq_len])  # [b*q,s]
        input_mask = tf.cast(input_mask, tf.float32)
        pooled_mask = tf.reshape(tf.reduce_max(input_mask, -1), [-1, self.max_que_len])  # [b,q]

        is_training = (mode == tf.estimator.ModeKeys.TRAIN)

        model = bert_base.BertModel(
            config=self.bert_config,
            is_training=is_training,
            input_ids=input_ids,
            input_mask=input_mask,
            use_one_hot_embeddings=self.bert_config["use_one_hot_embeddings"])

        if self.output_emb == "get_sequence_output":
            output_emb = model.get_sequence_output()[:, 1:, :] # [b*q, s, h]
            word_emb = max_pooling(output_emb, input_mask[:,1:])  # [b*q, h]
            word_emb = tf.reshape(word_emb, [-1, self.max_que_len, self.bert_config["hidden_size"]])  # [b,q,h]
            values = max_pooling(word_emb, pooled_mask)  # [b,h]
        elif self.output_emb == "get_pooled_output":
            output_emb = model.get_pooled_output()  # [b*q, h]
            output_emb = tf.reshape(output_emb, [-1, self.max_que_len, self.bert_config["hidden_size"]])  # [b,q,h]
            values = max_pooling(output_emb, pooled_mask)
        else:
            raise ValueError("Not implemented output emb type")


        for i, n_units in enumerate(self.layers_num, 1):
            with tf.variable_scope("MlpLayer-%d" % i) as hidden_layer_scope:
                values = layers.fully_connected(
                    values, num_outputs=n_units, activation_fn=tf.nn.tanh,
                    scope=hidden_layer_scope, reuse=tf.AUTO_REUSE
                )
            if is_training and self.dp > 0:
                print("In training mode, use dropout")
                values = tf.nn.dropout(values, keep_prob=1 - self.dp)

        logits = layers.linear(values, self.classes, scope="Logit_layer", reuse=tf.AUTO_REUSE)
        prediction = tf.argmax(logits, dimension=1)

        # print params in std_output
        slim.model_analyzer.analyze_vars(tf.trainable_variables(), print_info=True)

        loss = None
        if labels is not None :
            loss = tf.reduce_mean(
                tf.nn.sparse_softmax_cross_entropy_with_logits(
                    logits=logits,
                    labels=labels
                ))

        #print params in std_err
        tvars = tf.trainable_variables()
        initialized_variable_names = {}
        if self.init_ckt:
            (assignment_map, initialized_variable_names) = \
                bert_base.get_assignment_map_from_checkpoint(tvars, self.init_ckt)
            tf.train.init_from_checkpoint(self.init_ckt, assignment_map)

        tf.logging.info("**** Trainable Variables ****")
        for var in tvars:
            init_string = ""
            if var.name in initialized_variable_names:
                init_string = ", *INIT_FROM_CKPT*"
            tf.logging.info("  name = %s, shape = %s%s", var.name, var.shape,
                            init_string)

        outputs = dict(oneid=oneid)
        outputs['prediction'] = prediction
        outputs["loss"] = loss
        return outputs

    def _train(self, outputs, labels):
        prediction = outputs["prediction"]
        loss = outputs["loss"]
        train_op = optimization.create_optimizer(
            loss , self.bert_config["learning_rate"], self.bert_config["num_train_steps"],
            self.bert_config["num_warmup_steps"], self.bert_config["use_tpu"])

        output_spec = tf.estimator.EstimatorSpec(
            mode=tf.estimator.ModeKeys.TRAIN,
            loss=outputs['loss'],
            train_op=train_op,
            training_hooks=[
                tf.train.LoggingTensorHook(
                    {
                        "loss": loss,
                        "step": tf.train.get_global_step(),
                        "acc" : 100. * tf.reduce_mean(
                                tf.cast(tf.equal(tf.cast(prediction, tf.int32), tf.cast(labels, tf.int32)),
                                        tf.float32))
                    },
                    every_n_iter=100
                )
            ])
        return output_spec

    def _predict(self, outputs):
        pred = dict(oneid=outputs['oneid'])
        pred['prediction'] = outputs['prediction']
        output_spec = tf.estimator.EstimatorSpec(
            mode=tf.estimator.ModeKeys.PREDICT,
            predictions=pred
        )
        return output_spec

    def model_fn(self, features, labels, mode):  # pylint: disable=unused-argument
        """The `model_fn` for TPUEstimator."""
        outputs = self._build_model(features, labels, mode)
        if mode == tf.estimator.ModeKeys.TRAIN:
            return self._train(outputs, labels)
        elif mode == tf.estimator.ModeKeys.PREDICT:
            return self._predict(outputs)
        else:
            raise ValueError("Only TRAIN and EVAL modes are supported: %s" % (mode))


def max_pooling(embeddings, masks):
    # batch * length * 1
    multiplier = tf.expand_dims(masks, axis=-1)
    embeddings_max = tf.reduce_max(tf.multiply(multiplier, embeddings),axis=1)
    return embeddings_max


def ave_pooling(embeddings, masks):
    # batch * length * 1
    multiplier = tf.expand_dims(masks, axis=-1)
    embeddings_sum = tf.reduce_sum(tf.multiply(multiplier, embeddings),axis=1)
    length = tf.expand_dims(tf.maximum(tf.reduce_sum(masks, axis=1), 1.0), axis=-1)
    embedding_avg = embeddings_sum / length
    return embedding_avg
