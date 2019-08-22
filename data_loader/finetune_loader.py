# -*- coding: utf-8 -*-#
import tensorflow as tf

class OdpsDataLoader:
    def __init__(self, table_name, config , mode, slice_id=0, slice_count=1):
        self.config = config
        self._batch_size = self.config["finetune_batch_size"]
        self._table_name = table_name
        self._slice_id = slice_id
        self._slice_count = slice_count
        self._mode = mode
        self._max_sum_length = self.config['finetune_seq_length']*self.config['finetune_query_length']

    def _text_content_parser(self, text, max_length):
        word_strs = tf.string_split([text], " ")
        word_index = tf.string_to_number(word_strs.values, out_type=tf.int32)[:max_length]
        masks = tf.cast(tf.not_equal(word_index, 0), tf.int32)
        return word_index, masks

    def _train_parse_data(self, oneid, labels, input_ids):
        with tf.device("/cpu:0"):
            input_ids_list, input_mask = self._text_content_parser(input_ids, self._max_sum_length)

        return {
                "oneid": oneid,
                "input_ids_list":input_ids_list,
                "input_mask":input_mask,
                 }, labels

    def _train_data_fn(self):
        with tf.name_scope('data_loader'):
            dataset = tf.data.TableRecordDataset(
                self._table_name,
                record_defaults=["", 0, ""],
                selected_cols='oneid, label, content',
                slice_id=self._slice_id,
                slice_count=self._slice_count
                )
            dataset = dataset.map(map_func=self._train_parse_data, num_parallel_calls=4)
            dataset = dataset.repeat(None)
            dataset = dataset.prefetch(40000)
            dataset = dataset.shuffle(200)
            dataset = dataset.padded_batch(
                self._batch_size,
                padded_shapes=(
                    {
                        "oneid": [],
                        "input_ids_list": [self._max_sum_length],
                        "input_mask": [self._max_sum_length],
                    }, [])
            )

            return dataset.make_one_shot_iterator().get_next()

    def _test_parse_data(self, oneid, input_ids):
        with tf.device("/cpu:0"):
            input_ids_list,input_mask = self._text_content_parser(input_ids, self._max_sum_length)

        return {"oneid": oneid,
                "input_ids_list":input_ids_list,
                "input_mask":input_mask,
                }, tf.constant(0, dtype=tf.int32)

    def _test_data_fn(self):
        with tf.name_scope('data_loader'):
            dataset = tf.data.TableRecordDataset(self._table_name,
                                                 record_defaults=["", ""],
                                                 selected_cols='oneid, content',
                                                 slice_id=self._slice_id,
                                                 slice_count=self._slice_count
                                                 )
            # print(self.mode, self.selected_cols)
            # create a parallel parsing function based on number of cpu cores
            dataset = dataset.map(map_func=self._test_parse_data, num_parallel_calls=4)
            dataset = dataset.shuffle(200)
            dataset = dataset.padded_batch(
                self._batch_size,
                padded_shapes=(
                    {
                        "oneid": [],
                        "input_ids_list": [self._max_sum_length],
                        "input_mask": [self._max_sum_length],
                    }, [])
            )

            return dataset.make_one_shot_iterator().get_next()

    def input_fn(self):
        return self._train_data_fn() if self._mode else self._test_data_fn()