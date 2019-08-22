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
    parser.add_argument("--snap_shot", type=int, default=4000)
    parser.add_argument("--checkpoint_dir", type=str) # must
    parser.add_argument("--load_all_layers_ckt", type = int, default=None, help="0: not warm_start , 1: warm_start_from_checkpoint")
    parser.add_argument("--load_all_step", type= int, default =0)
    parser.add_argument("--load_bert_ckt", type=str, default=None, help="Load bert parameters") # must
    parser.add_argument("--load_bert_step", type=int, default =0) #must
    return parser.parse_known_args()[0]

os.environ['TF_GPU_VMEM'] = 'True'

def main():
    # bert 参数初始化
    config = parse_config('BERT')

    # Parse arguments and print them
    args = parse_args()
    print("\nMain arguments:")
    for k, v in args.__dict__.items():
        print("{}={}".format(k, v))

    # Generate bert model ckt path and warm start path
    warm_start_path = None
    if args.load_bert_ckt:
        warm_start_path = args.buckets +args.load_bert_ckt + "/model.ckpt-{}".format(args.load_bert_step)
        warm_start_settings = tf.estimator.WarmStartSettings(warm_start_path,vars_to_warm_start='bert*')
    elif args.load_all_layers_ckt:
        warm_start_path = args.buckets + args.load_all_layers_ckt + "/model.ckpt-{}".format(args.load_all_step)
        warm_start_settings = tf.estimator.WarmStartSettings(warm_start_path, vars_to_warm_start=".*")
    else:
        raise ValueError("No pretain params for finetune models")

    # Check if the model has already exisited
    model_save_dir = args.buckets + args.checkpoint_dir
    warm_start_dir = None # bert.*
    if tf.gfile.Exists(model_save_dir + "/checkpoint") and args.load_all_layers_ckt != args.checkpoint_dir :
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
        ),
        warm_start_from = warm_start_settings
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

