# -*- coding: UTF-8 -*-
from collections import namedtuple
from argparse import Namespace
import tensorflow as tf
from tensorflow.contrib import layers
import numpy as np
from tensorflow.contrib import slim

# push 0627

class _WordEmbeddingEncoder:
    def __init__(self, word_count, dimension, training, scope_name,
                 freeze_word2vec, init_with_w2v, enable_date_embedding, date_span, l2_date_embedding, **kwargs):
        training = training and not freeze_word2vec
        self._scope_name = scope_name
        self._enable_date_embedding = enable_date_embedding
        self._l2_date_embedding = l2_date_embedding

        with tf.variable_scope(scope_name, reuse=False):
            if init_with_w2v and word_count==178422:
                with open("resources/pre-trained-w2v.npz", "rb") as reader:
                    pre_trained_embedding = np.load(reader)["embedding"]

                if dimension != pre_trained_embedding.shape[1]:
                    raise KeyError(
                        "Word embedding dimension doesn't match with the pre-trained word embedding dimension"
                    )

                # Use pre-trained word2vec to initialize word embedding matrix
                self._word_embedding = tf.get_variable(
                    "WordEmbedding",
                    pre_trained_embedding.shape,
                    dtype=tf.float32,
                    initializer=tf.constant_initializer(pre_trained_embedding),
                    trainable=training
                )
            else:
                self._word_embedding = tf.get_variable(
                    "WordEmbedding",
                    [word_count, dimension],
                    dtype=tf.float32,
                    initializer=tf.contrib.layers.xavier_initializer(),
                    trainable=training
                )

            if enable_date_embedding:
                self._date_embedding = tf.get_variable(
                    "TimeDiffEmbedding",
                    [date_span + 1, int(self._word_embedding.shape[1])],
                    dtype=tf.float32,
                    initializer=tf.contrib.layers.xavier_initializer(),
                    trainable=training
                )

            print('WordEmbedding',self._word_embedding.shape)
            print('DateEmbedding',self._date_embedding.shape)

    def __call__(self, tokens, time_diff):
        """
        获得句子的embedding
        :param tokens: batch_size * max_seq_len
        :param masks: batch_size * max_seq_len
        :return:
        """
        with tf.variable_scope(self._scope_name, reuse=tf.AUTO_REUSE):
            word_embedding = tf.nn.embedding_lookup(self._word_embedding, tokens)
            if self._enable_date_embedding:
                print("Date embedding is enabled")
                time_embedding = tf.nn.embedding_lookup(self._date_embedding, time_diff)
                embedding = word_embedding + time_embedding                                     # 词嵌入和时间嵌入直接叠加

                if self._l2_date_embedding > 0:
                    print("Date embedding regularization is enabled")
                    distinct_times = tf.unique(tf.reshape(time_diff, [-1]))[0]
                    distinct_time_embedding = tf.nn.embedding_lookup(self._date_embedding, distinct_times)

                    print('distinct times', distinct_times)

                    return embedding,  self._l2_date_embedding* tf.nn.l2_loss(distinct_time_embedding)      #
                else:
                    return embedding, 0
            else:
                return word_embedding, 0


class _CnnEncoder:
    def __init__(self, cnn_kernels, training, scope_name, *args, **kwargs):
        self._scope = tf.variable_scope(scope_name, reuse=tf.AUTO_REUSE)

        cnn_kernels = [int(x) for x in cnn_kernels.split(",")]
        self._cnn_kernels = list(zip(cnn_kernels[::2], cnn_kernels[1::2]))
        self._training = training

    def __call__(self, word_embeddings):
        with self._scope:
            pooled_output = []
            for kernel_size, kernel_count in self._cnn_kernels:
                kernel_params = tf.get_variable(
                    "Conv-{}".format(kernel_size),
                    shape=[kernel_size, int(word_embeddings.shape[2]), 1, kernel_count], # 卷积核大小，输入的emb的形状，步长，卷积核个数
                    dtype=tf.float32, trainable=self._training,
                    initializer=tf.contrib.layers.xavier_initializer()
                )
                kernel_bias = tf.get_variable(
                    "Conv-bias-{}".format(kernel_size),
                    shape=[kernel_count], dtype=tf.float32,
                    trainable=self._training, initializer=tf.constant_initializer(0)
                )

                conv_output = tf.nn.conv2d(
                    tf.expand_dims(word_embeddings, axis=-1),
                    kernel_params,
                    strides=[1] * 4,
                    padding="VALID"
                )
                features = tf.nn.relu(tf.nn.bias_add(conv_output, kernel_bias))

                pooled_features = tf.squeeze(tf.reduce_max(features, axis=1), axis=1)
                pooled_output.append(pooled_features)

            all_features = tf.concat(pooled_output, axis=1)

            print('==== cnn encoder =====')
            print('input_wordemb shape ',word_embeddings.shape)
            print('output_wordemb shape pooling',all_features.shape)

            return all_features

class _MlpTransformer(object):
    '''
    input :
    from CnnEncoder,sentence embedding

    '''
    def __init__(self, layers, dropout, training, scope_name):
        self._layers = layers
        self._training = training
        self._dropout = dropout
        self._scope = tf.variable_scope(scope_name)

    def __call__(self, input):
        with self._scope as scope:
            values = input
            for i, n_units in enumerate(self._layers[:-1], 1):
                if self._training:
                    print("In training mode, use dropout")
                    values = [tf.nn.dropout(v, keep_prob=1-self._dropout) for v in values]

                with tf.variable_scope("MlpLayer-%d" % i) as hidden_layer_scope:
                    values = [layers.fully_connected(
                        v, num_outputs=n_units, activation_fn=tf.nn.relu,
                        scope=("l-{}-p-{}".format(i, j)), reuse=tf.AUTO_REUSE
                    )for j, v in enumerate(values)]

            if self._training:
                values = [tf.nn.dropout(v, keep_prob=1-self._dropout) for v in values]

            features = tf.concat(values, axis=1)

            logits = [layers.linear(
                v, self._layers[-1], scope="f-{}".format(j), reuse=tf.AUTO_REUSE
            ) for j, v in enumerate(values)]

            print('MLP input shape',input[0].shape)
            print('MLP layers',self._layers)
            print("_MLP Features shape :", features.shape)
            print("_MLP Logits shape : ", logits[0].shape)

            slim.model_analyzer.analyze_vars(tf.trainable_variables(), print_info=True)

            return logits[0], features


class _Model:
    def __init__(self, encoder, pooling, dropout, layers, training, scope_name):
        self._scope = tf.variable_scope(scope_name, reuse=tf.AUTO_REUSE)
        self._encoders = encoder
        self._poolings = pooling
        self._mlp = _MlpTransformer(layers, dropout, training=training, scope_name=scope_name + "_" + "MLP")

    def __call__(self, input, time_diff, labels):
        print('####################### _Model ',labels)
        with self._scope:
            sentence_embeddings = []
            extra_loss = tf.constant(0, dtype=tf.float32)

            for encoder, pooling, text, date in zip(self._encoders, self._poolings, input, time_diff):
                embeddings, reg_loss = encoder(text, date)
                sentence_embeddings.append(pooling(embeddings))
                extra_loss += reg_loss

            #sentence_embeddings = tf.concat(sentence_embeddings, axis=1)
            print('_Model,sentence_embeddings',sentence_embeddings)
            logits, features = self._mlp(sentence_embeddings)
            prob = tf.nn.softmax(logits)
            prediction = tf.argmax(logits, dimension=1)

            loss = None

            if labels is not None:
                loss = tf.reduce_mean(
                    tf.nn.sparse_softmax_cross_entropy_with_logits(
                        logits=logits,
                        labels=labels
                    ) + extra_loss)

            return Namespace(
                logit=logits,
                confidence=prob,
                feature=features,
                loss=loss,
                prediction=prediction
            )


class TextConvNet:
    ModelConfigs = namedtuple("ModelConfigs", ("kernels", "dropout", "enable_date_embedding",
                                               "l2_date_embedding", "date_span", "n_classes",
                                               "hidden_layers", "init_with_w2v", "dim_word_embedding", "word_count"))

    def __init__(self, model_configs, train_configs, predict_configs, run_configs):
        self._model_configs = model_configs
        self._train_configs = train_configs
        self._predict_configs = predict_configs
        self._run_configs = run_configs

    def _train(self, model_output, labels):
        # TODO: Add optimizer / reguliazer
        optimizer = tf.train.AdamOptimizer(learning_rate=self._train_configs.learning_rate)
        train_op = optimizer.minimize(model_output.loss, global_step=tf.train.get_global_step())

        return tf.estimator.EstimatorSpec(
            mode=tf.estimator.ModeKeys.TRAIN,
            loss=model_output.loss,
            train_op=train_op,
            training_hooks=[
                tf.train.LoggingTensorHook(
                    {
                        "loss" : model_output.loss,
                        "accuracy": 100.* tf.reduce_mean(tf.cast(tf.equal(tf.cast(model_output.prediction,tf.int32), tf.cast(labels,tf.int32)), tf.float32)),
                        "step": tf.train.get_global_step()
                    },
                    every_n_iter=100
                )
            ]
        )

    def _predict(self, model_output):
        outputs = dict(oneid=model_output.qid)

        if self._predict_configs.output_embedding:
            outputs["feature"] = tf.reduce_join(
                tf.as_string(model_output.feature),
                axis=1,
                separator=" "
            )

        if self._predict_configs.output_confidence:
            outputs["confidence"] = model_output.confidence

        if self._predict_configs.output_prediction:
            outputs["prediction"] = model_output.prediction

        return tf.estimator.EstimatorSpec(
            mode=tf.estimator.ModeKeys.PREDICT,
            predictions=outputs
        )

    def _evaluate(self, model_output, labels):
        # 二分类评估指标
        if self._model_configs.n_classes == 2:
            return tf.estimator.EstimatorSpec(
                mode=tf.estimator.ModeKeys.EVAL,
                loss=model_output.loss,
                eval_metric_ops={
                    "accuracy": tf.metrics.accuracy(labels, model_output.prediction),
                    "precision": tf.metrics.precision(labels, model_output.prediction),
                    "recall": tf.metrics.recall(labels, model_output.prediction),
                    "auc": tf.metrics.auc(labels, model_output.probability)
                }
            )
        # 多分类评估指标
        else:
            return tf.estimator.EstimatorSpec(
                mode=tf.estimator.ModeKeys.EVAL,
                loss=model_output.loss,
                eval_metric_ops={
                    "accuracy": tf.metrics.accuracy(labels, model_output.prediction),
                    "mean_per_class_accuracy": tf.metrics.mean_per_class_accuracy(labels,
                                                                                  model_output.prediction,
                                                                                  self._model_configs.n_classes)
                }
            )

    def _create_encoders(self, prefix, training):
        word_encoder = _WordEmbeddingEncoder(
            scope_name="{}-encoder".format(prefix),
            word_count=self._model_configs.word_count + 1,
            dimension=self._model_configs.dim_word_embedding,
            training=training,
            freeze_word2vec=False,
            init_with_w2v=self._model_configs.init_with_w2v,
            date_span=self._model_configs.date_span,
            enable_date_embedding=self._model_configs.enable_date_embedding,
            l2_date_embedding=self._model_configs.l2_date_embedding
        )

        cnn_encoder = _CnnEncoder(
            cnn_kernels=self._model_configs.kernels,
            training=training,
            scope_name="{}-pooling".format(prefix)
        )
        return word_encoder, cnn_encoder

    def _build_model(self, features, labels, mode):
        oneid = features["oneids"]
        query_text, query_date = features["query_text"], features["query_date"]

        training = mode is tf.estimator.ModeKeys.TRAIN
        query_word_encoder, query_cnn_encoder = self._create_encoders("query", training)

        model = _Model(
            encoder=[query_word_encoder],
            pooling=[query_cnn_encoder],
            dropout=self._model_configs.dropout,
            layers=[int(n) for n in self._model_configs.hidden_layers.split(",")] + [self._model_configs.n_classes],
            training=training,
            scope_name="Classification"
        )

        model_output = model([query_text], [query_date], labels)
        model_output.oneid = oneid
        return model_output

    def model_fn(self, features, labels, mode):
        model_output = self._build_model(features, labels, mode)

        if mode is tf.estimator.ModeKeys.TRAIN:
            return self._train(model_output, labels)
        elif mode is tf.estimator.ModeKeys.PREDICT:
            return self._predict(model_output)
        elif mode is tf.estimator.ModeKeys.EVAL:
            return self._evaluate(model_output, labels)
