# -*- coding: utf-8 -*-#
from collections import namedtuple

# 训练配置
TrainConfigs = namedtuple("TrainConfigs", (
    # 学习率
    "learning_rate",
))

# 预测配置项
PredictConfigs = namedtuple("PredictConfigs", (
    # 是否输出分类前的特征向量
    "output_embedding",
    # 是否输出概率分
    "output_confidence",
    # 是否输出预测结果
    "output_prediction"
))

# 模型配置
RunConfigs = namedtuple("RunConfigs", (
    # 输出日志间隔
    "log_every"
))