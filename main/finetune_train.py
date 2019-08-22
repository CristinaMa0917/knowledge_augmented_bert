# -*- coding: utf-8 -*-#
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__) + "/.."))
import tensorflow as tf
import argparse
import model
from data_loader import finetune_loader
from util import env
from util import helper
from util.config import set_dist_env, parse_config
from model.bert_finetune import BertFinetune

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tables", type=str, help="Kernels configuration for CNN")
    parser.add_argument("--buckets", type=str, help="Worker task index")
    parser.add_argument("--snap_shot", type=int, default=5000)
    parser.add_argument("--checkpoint_dir", type=str) # must
    parser.add_argument("--init_ckt_dir", type=str)  # must
    parser.add_argument("--init_ckt_step", type=str)  # must
    return parser.parse_known_args()[0]

os.environ['TF_GPU_VMEM'] = 'True'

def main():
    # bert 参数初始化
    config = parse_config('MiniBERT')
    config["learning_rate"] = 5e-5

    # Parse arguments and print them
    args = parse_args()
    print("\nMain arguments:")
    for k, v in args.__dict__.items():
        print("{}={}".format(k, v))

    # Generate bert model ckt path and warm start path
    config["init_checkpoint"] = args.buckets + args.init_ckt_dir + "/model.ckpt-{}".format(args.init_ckt_step)

    # Check if the model has already exisited
    model_save_dir = args.buckets + args.checkpoint_dir
    if tf.gfile.Exists(model_save_dir + "/checkpoint") :
        raise ValueError("Model %s has already existed, please delete them and retry" % model_save_dir)

    helper.dump_args(model_save_dir, args)

    bert_model = BertFinetune(config)

    estimator = tf.estimator.Estimator(
        model_fn=bert_model.model_fn,
        model_dir=model_save_dir,
        config=tf.estimator.RunConfig(
            session_config=tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True)),
            save_checkpoints_steps=args.snap_shot,
            keep_checkpoint_max=100
        )
    )

    print("Start training......")
    estimator.train(
        finetune_loader.OdpsDataLoader(
            table_name = args.tables,
            config = config,
            mode = 1).input_fn,
        steps=config["num_train_steps"],
    )
if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    main()

