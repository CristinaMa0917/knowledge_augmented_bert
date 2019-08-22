# -*- coding: utf-8 -*-#
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__) + "/.."))
import tensorflow as tf
import argparse
import model
from data_loader import pretrain_loader
from util import env
from util import helper
from util.config import set_dist_env, parse_config
from model.bert_pretrain import BertPretrain
from tensorflow.contrib.distribute.python import cross_tower_ops as cross_tower_ops_lib

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tables", type=str, help="Kernels configuration for CNN")
    parser.add_argument("--buckets", type=str, help="Worker task index")
    parser.add_argument("--snap_shot", type=int, default=10000)
    parser.add_argument("--checkpoint_dir", type=str)
    parser.add_argument("--warm_start_up", type = int, help="0: not warm_start , 1: warm_start_from_checkpoint")
    return parser.parse_known_args()[0]


def main():
    # bert 参数初始化
    config = parse_config('MiniBERT')
    config["learning_rate"] = 1e-3

    # Parse arguments and print them
    args = parse_args()
    print("\nMain arguments:")
    for k, v in args.__dict__.items():
        print("{}={}".format(k, v))

    # Check if the model has already exisited
    model_save_dir = args.buckets + args.checkpoint_dir
    warm_start_dir = None
    if tf.gfile.Exists(model_save_dir + "/checkpoint") and not args.warm_start_up:
        raise ValueError("Model %s has already existed, please delete them and retry" % model_save_dir)
    elif tf.gfile.Exists(model_save_dir + "/checkpoint") and args.warm_start_up :
        print("Model warm start up from %s" % model_save_dir)
        warm_start_dir = model_save_dir
    else:
        print("Model init training")

    helper.dump_args(model_save_dir, args)

    bert_model = BertPretrain(config)

    cross_tower_ops = cross_tower_ops_lib.AllReduceCrossTowerOps('nccl')
    distribution = tf.contrib.distribute.MirroredStrategy(
        num_gpus=4, cross_tower_ops=cross_tower_ops,
        all_dense=True
    )

    estimator = tf.estimator.Estimator(
        model_fn=bert_model.model_fn,
        model_dir=model_save_dir,
        config=tf.estimator.RunConfig(
            session_config=tf.ConfigProto(
                gpu_options=tf.GPUOptions(allow_growth=True),
                allow_soft_placement=True),
            save_checkpoints_steps=args.snap_shot,
            keep_checkpoint_max=100,
            train_distribute=distribution
        ),
        warm_start_from = warm_start_dir,
    )

    print("Start training......")
    tf.estimator.train(
        estimator,
        train_spec=tf.estimator.TrainSpec(
            input_fn=pretrain_loader.OdpsDataLoader(
                table_name=args.tables,
                mode=1,
                config=config
            ).input_fn,
            max_steps=config["num_train_steps"]
        )
    )

if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    main()

