# -*- coding: utf-8 -*-#
import tensorflow as tf

class OdpsDataLoader:
    def __init__(self, table_name, config , mode, slice_id=0, slice_count=1):
        self.config = config
        self._record_defaults = self.config['col_types']
        self._selected_cols = self.config['columns']
        self._batch_size = self.config["batch_size"]
        self._table_name = table_name
        self._max_seq_length = self.config['seq_length']
        self._slice_id = slice_id
        self._slice_count = slice_count
        self._mode = mode
        self._max_predictions_per_seq = self.config['max_predictions_per_seq']

    def _text_content_parser(self, text, max_length, shift_no=True):
        word_strs = tf.string_split([text], " ")
        return tf.string_to_number(word_strs.values, out_type=tf.int32)[:max_length] + (1 if shift_no else 0), \
               tf.ones([tf.minimum(tf.shape(word_strs)[-1], max_length)] , dtype = tf.int32)

    def _text_content_parser_clip(self, text, max_length, clip_len, shift_no=True):
        word_strs = tf.string_split([text], " ")
        word_val = tf.string_to_number(word_strs.values, out_type=tf.int32)[:max_length] + (1 if shift_no else 0)

        # to avoid the masked word index is bigger than the max sequence length
        clip = tf.cast(tf.less(word_val, clip_len), dtype=tf.int32)
        word_val = tf.multiply(word_val, clip)
        return word_val, clip

    def _train_parse_data(self, qid, input_ids, masked_pos, masked_ids, labels):
        with tf.device("/cpu:0"):
            input_ids_list, input_mask = self._text_content_parser(input_ids, self._max_seq_length)
            masked_lm_ids  ,_ = self._text_content_parser(masked_ids,self._max_predictions_per_seq)
            masked_lm_positions , masked_lm_weights = self._text_content_parser_clip(masked_pos,
                                                                                    self._max_predictions_per_seq,
                                                                                    self._max_seq_length, shift_no=False)
            segment_ids_list = tf.zeros([self._max_seq_length], dtype=tf.int32)

        return {
                "qids": qid,
                "input_ids_list":input_ids_list,
                "input_mask":input_mask,
                "masked_lm_positions": masked_lm_positions,
                "masked_lm_ids": masked_lm_ids,
                "masked_lm_weights":masked_lm_weights,
                "segment_ids_list": segment_ids_list
                 }, labels

    def _train_data_fn(self):
        with tf.name_scope('data_loader'):
            dataset = tf.data.TableRecordDataset(
                self._table_name,
                record_defaults=self._record_defaults,
                selected_cols=','.join(self._selected_cols),
                slice_id=self._slice_id,
                slice_count=self._slice_count
                )
            dataset = dataset.map(map_func=self._train_parse_data, num_parallel_calls=4)
            dataset = dataset.repeat(None)
            dataset= dataset.prefetch(40000)
            dataset = dataset.shuffle(500)
            dataset = dataset.padded_batch(
                self._batch_size,
                padded_shapes=(
                    {
                        "qids": [],
                        "input_ids_list": [self._max_seq_length],
                        "input_mask": [self._max_seq_length],
                        "masked_lm_positions": [self._max_predictions_per_seq],
                        "masked_lm_ids": [self._max_predictions_per_seq],
                        "masked_lm_weights":[self._max_predictions_per_seq],
                        "segment_ids_list":[self._max_seq_length]
                    }, [])
            )

            return dataset

    def _test_parse_data(self, qid, input_ids):
        with tf.device("/cpu:0"):
            input_ids_list,input_mask = self._text_content_parser(input_ids, self._max_seq_length)
            masked_lm_positions = None
            masked_lm_ids = None
            segment_ids_list = tf.zeros([self._max_seq_length], dtype=tf.int32)

        return {"qids": qid, "input_ids_list":input_ids_list, "input_mask":input_mask,
                "masked_lm_positions": masked_lm_positions,"masked_lm_ids": masked_lm_ids,
                "segment_ids_list": segment_ids_list}, tf.constant(0, dtype=tf.int32)

    def _test_data_fn(self):
        with tf.name_scope('data_loader'):
            dataset = tf.data.TableRecordDataset(self._table_name,
                                                 record_defaults=["", ""],
                                                 selected_cols=','.join(['qid', 'input_ids']),
                                                 slice_id=self._slice_id,
                                                 slice_count=self._slice_count
                                                 )
            # print(self.mode, self.selected_cols)
            # create a parallel parsing function based on number of cpu cores
            dataset = dataset.map(map_func=self._test_parse_data, num_parallel_calls=4)
            dataset = dataset.repeat(None)
            dataset = dataset.shuffle(500)
            dataset = dataset.padded_batch(
                self._batch_size,
                padded_shapes=(
                    {
                        "qids": [],
                        "input_ids_list": [self._max_seq_length],
                        "input_mask": [self._max_seq_length],
                        "masked_lm_positions": [],
                        "masked_lm_ids": [],
                        "segment_ids_list":[self._max_seq_length]
                    }, [])
            )

            return dataset

    def input_fn(self):
        return self._train_data_fn() if self._mode else self._test_data_fn()