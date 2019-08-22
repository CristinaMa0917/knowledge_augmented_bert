# -*- coding: utf-8 -*-#

import json
import os

optimizer = ["Adagrad", "Adam", "Ftrl", "Momentum", "RMSProp", "SGD"]
task_type = ["train", "predict", "eval", "export"]


def parse_config(model, json_file='configs/HyperParameters.json'):
    """
        Get the config from a json file
        :param json_file: 
        :param model: ['DCN','DeepFM',...]
        :return: config(namespace) or config(dictionary)
        """
    # parse the configurations from the config json file provided
    # try:
    #     modules = __import__("models")
    #     getattr(modules, model)
    # except:
    #     raise ValueError('{} argument is not the subclass of BaseModel.'.format(model))

    with open(json_file, 'r') as config_file:
        config_dict = json.load(config_file)

    hp = config_dict[model]

    if hp['task_type'] not in task_type:
        raise ValueError(
            'Value of task_type must be in : {}'.format(task_type))

    if 'optimizer' in hp.keys() and hp['optimizer'] not in optimizer:
        raise ValueError(
            'Value of optimizer must be in : {}'.format(optimizer))

    # config_dict['summary_dir'] = os.path.join("../experiments", config.exp_name, "summary/")
    # config_dict['checkpoint_dir'] = os.path.join("../experiments", config.exp_name, "checkpoint/")
    return hp


# 设置分布式环境参数
def set_dist_env(dist_params):
    '''
        基于estimator下份环境参数配置
    :param dist_params: dict，分布式训练环境参数设置
    :return: 
    '''
    if dist_params['dist_mode'] and dist_params['task_type'] == 'predict':
        # 集群预测模式： 1 chief, 1 ps, n worker
        if dist_params['job_name'] == 'ps':
            exit(0)
        worker_hosts = dist_params['worker_hosts'].split(',')
        chief_hosts = worker_hosts[0:1]  # get first worker as chief
        worker_hosts = worker_hosts[1:]  # the rest as worker
        task_index = dist_params['task_index']
        job_name = dist_params['job_name']
        if job_name == "worker" and task_index == 0:
            job_name = "chief"
        # the others as worker
        if job_name == "worker" and task_index > 0:
            task_index -= 1
        tf_config = {
            'cluster': {'chief': chief_hosts, 'worker': worker_hosts},
            'task': {'type': job_name, 'index': task_index}
        }
        print(json.dumps(tf_config))
        os.environ['TF_CONFIG'] = json.dumps(tf_config)

        return len(worker_hosts) + 1, dist_params['task_index']

    elif dist_params['dist_mode'] and dist_params['task_type'] == 'train':
        # 集群训练模式: 1 chief, 1 ps, 1 evaluator, n worker
        ps_hosts = dist_params['ps_hosts'].split(',')
        worker_hosts = dist_params['worker_hosts'].split(',')
        chief_hosts = worker_hosts[0:1]  # get first worker as chief
        worker_hosts = worker_hosts[2:]  # the rest as worker
        task_index = dist_params['task_index']
        job_name = dist_params['job_name']
        # print('ps_host', ps_hosts)
        # print('worker_host', worker_hosts)
        # print('chief_hosts', chief_hosts)
        # print('job_name', job_name)
        # print('task_index', str(task_index))
        # use #worker=0 as chief
        if job_name == "worker" and task_index == 0:
            job_name = "chief"
        # use #worker=1 as evaluator
        if job_name == "worker" and task_index == 1:
            job_name = 'evaluator'
            task_index = 0
        # the others as worker
        if job_name == "worker" and task_index > 1:
            task_index -= 2

        tf_config = {
            'cluster': {'chief': chief_hosts, 'worker': worker_hosts, 'ps': ps_hosts},
            'task': {'type': job_name, 'index': task_index}
        }
        print(json.dumps(tf_config))
        os.environ['TF_CONFIG'] = json.dumps(tf_config)

        return 1, 0
    else:  # 单机环境
        return 1, 0
