# Standard library imports
import time
import os
import sys
import math
import random
import pathlib
import argparse
from argparse import Namespace
from copy import deepcopy

# Data manipulation and analysis
import numpy as np
import pandas as pd
import scipy.stats as stats
from scipy.stats import spearmanr, pearsonr

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm, trange

# Machine learning tools
from sklearn import preprocessing
from sklearn.metrics import (
    r2_score, f1_score, roc_auc_score, 
    mean_squared_error, mean_absolute_error
)
from sklearn.model_selection import KFold

# PyTorch and related imports
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.optim.lr_scheduler import StepLR

# Configuration
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

sys.path.append("/home/yanyichu/1_UTR_Cell_Type/parade/predictor/model")
import utrdata_cl as utrdata
from legnet import LegNetClassifier
from pl_regressor import RNARegressor

import pytorch_lightning as pl
import itertools

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import json
from Bio import SeqIO
import os

seed =1337
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True

def load_model_from_checkpoint(checkpoint_path, model_class=None):
    """
    从检查点加载预训练模型
    
    参数:
    checkpoint_path: 检查点文件路径
    model_class: 模型类，如果不提供，将尝试从检查点中直接加载模型
    
    返回:
    加载的模型实例
    """
    # 检查文件是否存在
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"检查点文件不存在: {checkpoint_path}")
    
    # 加载检查点
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        print(f"检查点已加载: {checkpoint_path}")
    except Exception as e:
        raise RuntimeError(f"加载检查点失败: {str(e)}")
    
    # 检查检查点结构
    print("检查点键:", list(checkpoint.keys()))
    
    # 获取超参数
    if 'hyper_parameters' in checkpoint:
        hparams = checkpoint['hyper_parameters']
        print("找到超参数:", hparams)
    else:
        print("检查点中没有超参数，将使用默认参数")
        hparams = {}  # 使用空字典或提供必要的参数
    
    # 创建模型实例
    if model_class is None:
        # 如果没有提供模型类，尝试直接加载模型
        if 'model' in checkpoint:
            model = checkpoint['model']
            print("直接从检查点加载模型")
        else:
            raise ValueError("未提供模型类且检查点中没有模型，无法初始化模型")
    else:
        # 如果提供了模型类，使用超参数初始化模型
        model = model_class(**hparams)
        print(f"使用 {model_class.__name__} 创建模型")
    
    # 加载模型状态
    if 'state_dict' in checkpoint:
        try:
            model.load_state_dict(checkpoint['state_dict'])
            print("模型状态加载成功!")
        except Exception as e:
            raise RuntimeError(f"加载模型状态失败: {str(e)}")
    else:
        print("警告: 检查点中没有state_dict，模型参数未更新")
    
    # 设置为评估模式
    model.eval()
    
    return model
    
class UTRPredictor:
    def __init__(self, celltype_codes, model=None, checkpoint_path=None, model_class=None, 
                 batch_size=64, num_workers=0, device=None):
        """
        初始化UTR序列预测器
        
        参数:
        model: 预训练的PyTorch模型
        checkpoint_path: 检查点文件路径，如果提供则从检查点加载模型
        model_class: 模型类，用于从检查点创建模型实例
        celltype_codes: 细胞类型代码列表
        batch_size: 批处理大小
        num_workers: 数据加载器使用的工作进程数
        device: 运行模型的设备 ('cuda', 'cpu' 或特定的 torch.device)
        """
        # 设置设备
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device if isinstance(device, torch.device) else torch.device(device)
        
        # 加载模型
        if model is not None:
            self.model = model
        elif checkpoint_path is not None:
            self.model = load_model_from_checkpoint(checkpoint_path, model_class)
        else:
            raise ValueError("必须提供 model 或 checkpoint_path 参数之一")
        
        # 将模型移至指定设备
        self.model = self.model.to(self.device)
        self.celltype_codes = celltype_codes
        self.batch_size = batch_size
        self.num_workers = num_workers
        
        # 设置模型参数
        self.params = {
            "seed": 3,
            "features": ("sequence", "positional", "conditions"),
            "augment_dict": {
                "extend_left": 0,
                "extend_right": 0,
                "shift_left": 0,
                "shift_right": 0,
                "revcomp": False,
            },
            "epochs": 10
        }
        self.augment_key = any(self.params["augment_dict"].values())
        
    def prepare_dataset(self, sequences, utrdata_module):
        """
        准备数据集
        
        参数:
        sequences: 要预测的UTR序列列表
        utrdata_module: UTRData模块，用于创建数据集
        
        返回:
        数据加载器
        """
        # 创建包含所有序列和细胞类型组合的DataFrame
        df_gen = pd.DataFrame(itertools.product(sequences, self.celltype_codes),
                          columns=['sequence', 'cell_type'])
        
        # 数据集参数
        gen_ds_kws = dict(
            celltype_codes=self.celltype_codes,
            predict_cols=[],
            construct_type="utr3",
            features=self.params["features"],
            augment=False,
            augment_test_time=self.augment_key,
            augment_kws=self.params["augment_dict"],
        )
        
        # 创建数据集
        gen_set = utrdata_module.UTRData(
            df=df_gen,
            **gen_ds_kws,
        )
        
        # 创建数据加载器
        dl_gen = DataLoader(
            gen_set,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            drop_last=False
        )
        
        return dl_gen, df_gen
        
    def predict(self, sequences, utrdata_module):
        """
        预测UTR序列的翻译效率和中心偏移
        
        参数:
        sequences: 要预测的UTR序列列表
        utrdata_module: UTRData模块，用于创建数据集
        
        返回:
        带有预测结果的DataFrame
        """
        # 准备数据集
        dl_gen, df_gen = self.prepare_dataset(sequences, utrdata_module)
        
        # 设置模型为评估模式
        self.model.eval()
        
        # 进行预测
        with torch.no_grad():
            predictions = []
            
            for batch in dl_gen:
                x, _ = batch
                # 将输入移至设备
                if isinstance(x, torch.Tensor):
                    x = x.to(self.device)
                elif isinstance(x, dict):
                    x = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in x.items()}
                
                # 判断模型是否有model属性
                if hasattr(self.model, 'model'):
                    output = self.model.model(x)
                else:
                    output = self.model(x)
                    
                predictions.append(output.cpu())
            
            # 合并结果
            gen_pred = torch.cat(predictions, dim=0).numpy()
            
            # 保存到DataFrame
            gen_df = df_gen.copy()
            gen_df["pred_diff"] = gen_pred[:, 0]
            gen_df["pred_center_of_mass"] = gen_pred[:, 1]
            
        return gen_df
    
    def predict_and_summarize(self, sequences, utrdata_module):
        """
        预测UTR序列的翻译效率和中心偏移，并按序列进行汇总
        
        参数:
        sequences: 要预测的UTR序列列表
        utrdata_module: UTRData模块，用于创建数据集
        
        返回:
        汇总后的DataFrame，每个序列有一行
        """
        # 获取详细预测结果
        detailed_df = self.predict(sequences, utrdata_module)
        
        return detailed_df.sort_values('pred_diff', ascending = False)

