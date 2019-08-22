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

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tables", type=str, help="Kernels configuration for CNN")
    parser.add_argument("--buckets", type=str, help="Worker task index")
    parser.add_argument("--snap_shot", type=int, default=10000)
    parser.add_argument("--checkpoint_dir", type=str)
    parser.add_argument("--warm_start_step", type = int, default=0, help="0: not warm_start , 1: ")
    return parser.parse_known_args()[0]


def main():
    # bert 参数初始化
    config = parse_config('MiniBERT')

    # Parse arguments and print them
    args = parse_args()
    print("\nMain arguments:")
    for k, v in args.__dict__.items():
        print("{}={}".format(k, v))

    # Check if the model has already exisited
    model_save_dir = args.buckets + args.checkpoint_dir
    warm_start_settings = None
    if tf.gfile.Exists(model_save_dir + "/checkpoint") and not args.warm_start_step:
        raise ValueError("Model %s has already existed, please delete them and retry" % model_save_dir)
    elif args.warm_start_step:
        warm_start_path = model_save_dir + "/model.ckpt-{}".format(args.warm_start_step)
        warm_start_settings = tf.estimator.WarmStartSettings(warm_start_path)
        print("Model init training from %s" % warm_start_path)
    else:
        pass

    helper.dump_args(model_save_dir, args)
    bert_model = BertPretrain(config)

    estimator = tf.estimator.Estimator(
        model_fn=bert_model.model_fn,
        model_dir=model_save_dir,
        config=tf.estimator.RunConfig(
            session_config=tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True)),
            save_checkpoints_steps=args.snap_shot,
            keep_checkpoint_max=100
        ),
        warm_start_from=warm_start_settings
    )

    print("Start training......")
    estimator.train(
        pretrain_loader.OdpsDataLoader(
            table_name = args.tables,
            config = config,
            mode = 1).input_fn,
        steps=config["num_train_steps"],
    )

if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    main()

