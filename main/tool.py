import tensorflow as tf
import time
import sys
import pathlib
import os
import json
from argparse import Namespace
from scipy.linalg import norm
import numpy as np


class CheckPointReader:
    """
    This reader extract tensor specified by a name
    """

    def __init__(self, model_path):
        from tensorflow.python import pywrap_tensorflow
        self._reader = pywrap_tensorflow.NewCheckpointReader(model_path)

    def get(self, name):
        return self._get_tensor(name)

    def _get_tensor(self, keyword):
        for name in self._reader.get_variable_to_shape_map():
            if keyword in name:
                return self._reader.get_tensor(name)


if __name__ == '__main__':
    reader = CheckPointReader(r'D:\code\query_entity_insert_3\model.ckpt-20000')
    reader = reader._reader
    print(reader.get_variable_to_dtype_map())
    w2v = reader.get_tensor("query-encoder/WordEmbedding")
    print('w2c shape', w2v.shape)
    index = [0, 1, 2, 3]
    print('entity_symbol_emb:')
    for i in index:
        print(norm(w2v[178422 + i]))
    print('Average norm')
    print(np.mean(norm(w2v[:10000], axis=1)))
