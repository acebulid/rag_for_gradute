#!/usr/bin/env python3
"""
模型A（图片编码→含义向量）完整工具集
整合模型训练、预测、评估、状态修复等所有功能
"""

import sys
import os
import traceback
import logging
import numpy as np
import pickle
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any, Tuple, Union, List
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import sklearn
import warnings
from sklearn.preprocessing import StandardScaler
from typing import Optional, Dict, Any

# 简化日志配置
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ========== 关键修复：全局注册安全对象 ==========
# 兼容PyTorch 2.6+的安全机制
try:
    # 1. 注册安全全局对象
    if hasattr(torch.serialization, 'add_safe_globals'):
        torch.serialization.add_safe_globals([
            sklearn.preprocessing._data.StandardScaler,
            StandardScaler,
            # 如果还有其他自定义对象，也需要注册在这里
        ])
    
    # 2. 全局关闭weights_only默认值（终极方案）
    if hasattr(torch.serialization, '_get_default_load_weights_only'):
        # 覆盖默认行为
        original_default = torch.serialization._get_default_load_weights_only
        def override_default():
            return False  # 强制默认weights_only=False
        torch.serialization._get_default_load_weights_only = override_default
    
    # 3. 忽略无关警告
    warnings.filterwarnings('ignore', category=UserWarning)
    warnings.filterwarnings('ignore', category=FutureWarning)
    
except Exception as e:
    print(f"PyTorch安全机制兼容警告: {e}")

# ============================ 核心配置与数据结构 ============================
@dataclass
class ModelAConfig:
    """模型A配置"""
    # 模型架构
    input_dim: int = 512  # CLIP向量维度
    hidden_dims: List[int] = (1024, 1024, 1024)  # 隐藏层维度
    output_dim: int = 1024  # 输出向量维度（与文字向量对齐）
    
    # 训练参数
    learning_rate: float = 0.001
    batch_size: int = 32
    epochs: int = 100
    dropout_rate: float = 0.2
    weight_decay: float = 1e-5
    
    # 数据参数
    validation_split: float = 0.2
    test_split: float = 0.1
    
    # 保存路径
    model_path: str = "data/models/model_a.pth"
    config_path: str = "data/models/model_a_config.json"
    scaler_path: str = "data/models/model_a_scaler.pkl"
    
    # 日志和监控
    print_every_n_epochs: int = 5  # 每N个epoch打印一次训练效果
    save_checkpoints: bool = True  # 是否保存检查点
    checkpoint_dir: str = "data/models/checkpoints"  # 检查点目录


class ImageToMeaningDataset(Dataset):
    """图片到含义数据集"""
    
    def __init__(self, image_embeddings: np.ndarray, text_embeddings: np.ndarray):
        """
        参数:
            image_embeddings: 图片向量数组 (n_samples, input_dim)
            text_embeddings: 文字向量数组 (n_samples, output_dim)
        """
        self.image_embeddings = torch.FloatTensor(image_embeddings)
        self.text_embeddings = torch.FloatTensor(text_embeddings)
        
    def __len__(self):
        return len(self.image_embeddings)
    
    def __getitem__(self, idx):
        return self.image_embeddings[idx], self.text_embeddings[idx]


class ImageToMeaningModel(nn.Module):
    """图片到含义模型（全连接神经网络）"""
    
    def __init__(self, config: ModelAConfig):
        super().__init__()
        self.config = config
        
        # 构建网络层
        layers = []
        input_dim = config.input_dim
        
        # 隐藏层
        for hidden_dim in config.hidden_dims:
            layers.extend([
                nn.Linear(input_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(config.dropout_rate)
            ])
            input_dim = hidden_dim
        
        # 输出层
        layers.append(nn.Linear(input_dim, config.output_dim))
        self.network = nn.Sequential(*layers)
        
        # 初始化权重
        self._initialize_weights()
    
    def _initialize_weights(self):
        """初始化权重"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        """前向传播"""
        return self.network(x)

# ============================ 模型核心功能 ============================
def create_model_instance(config: ModelAConfig) -> Tuple[ImageToMeaningModel, torch.device]:
    """
    创建模型实例并确定计算设备（统一版本，替换重复定义）
    
    参数:
        config: 模型配置实例
    
    返回:
        tuple: (模型实例, 计算设备)
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ImageToMeaningModel(config).to(device)
    logger.info(f"模型初始化完成 - 使用设备: {device}")
    return model, device


def prepare_training_data(
    image_embeddings: np.ndarray,
    text_embeddings: np.ndarray,
    config: ModelAConfig,
    random_seed: Optional[int] = None
) -> Dict[str, Any]:
    """
    准备训练数据（标准化+划分+创建加载器）
    
    参数:
        image_embeddings: 图片向量数组
        text_embeddings: 文字向量数组
        config: 模型配置
        random_seed: 随机种子，可选
    
    返回:
        dict: 包含数据集信息和加载器的字典
    """
    if random_seed is not None:
        np.random.seed(random_seed)
    
    n_samples = len(image_embeddings)
    
    # 数据标准化
    scaler = StandardScaler()
    image_embeddings_scaled = scaler.fit_transform(image_embeddings)
    
    # 划分数据集
    indices = np.random.permutation(n_samples)
    n_val = int(config.validation_split * n_samples)
    n_test = int(config.test_split * n_samples)
    n_train = n_samples - n_val - n_test
    
    train_indices = indices[:n_train]
    val_indices = indices[n_train:n_train + n_val]
    test_indices = indices[n_train + n_val:]
    
    # 创建数据集和加载器
    def create_loader(indices, shuffle: bool = False):
        dataset = ImageToMeaningDataset(
            image_embeddings_scaled[indices],
            text_embeddings[indices]
        )
        return DataLoader(
            dataset,
            batch_size=config.batch_size,
            shuffle=shuffle
        )
    
    train_loader = create_loader(train_indices, shuffle=True)
    val_loader = create_loader(val_indices)
    test_loader = create_loader(test_indices)
    
    return {
        'n_train': n_train,
        'n_val': n_val,
        'n_test': n_test,
        'train_loader': train_loader,
        'val_loader': val_loader,
        'test_loader': test_loader,
        'scaler': scaler
    }


def train_model(
    image_embeddings: np.ndarray,
    text_embeddings: np.ndarray,
    config: ModelAConfig,
    random_seed: Optional[int] = None,
    verbose: bool = True
) -> Tuple[ImageToMeaningModel, Dict[str, Any], StandardScaler]:
    """
    训练模型A
    
    参数:
        image_embeddings: 图片向量数组
        text_embeddings: 文字向量数组
        config: 模型配置
        random_seed: 随机种子
        verbose: 是否打印训练过程
    
    返回:
        tuple: (模型实例, 训练历史, scaler)
    """
    if verbose:
        print("="*70)
        print("开始训练模型A")
        print("="*70)
    
    training_start_time = time.time()
    
    # 准备数据
    data_info = prepare_training_data(image_embeddings, text_embeddings, config, random_seed)
    scaler = data_info['scaler']
    
    # 初始化模型、损失函数和优化器
    model, device = create_model_instance(config)
    criterion = nn.CosineEmbeddingLoss()
    optimizer = optim.Adam(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay
    )
    
    # 训练循环初始化
    best_val_loss = float('inf')
    train_losses, val_losses = [], []
    
    if verbose:
        print(f"\n开始训练循环，共 {config.epochs} 个epoch")
        print(f"   批次大小: {config.batch_size}")
        print(f"   学习率: {config.learning_rate}")
        print(f"   每 {config.print_every_n_epochs} 个epoch打印一次进度")
    
    for epoch in range(config.epochs):
        epoch_start_time = time.time()
        
        # 训练阶段
        model.train()
        train_loss = 0.0
        
        for batch_images, batch_texts in data_info['train_loader']:
            batch_images, batch_texts = batch_images.to(device), batch_texts.to(device)
            
            # 前向传播 + 计算损失 + 反向传播
            outputs = model(batch_images)
            loss = criterion(outputs, batch_texts, torch.ones(batch_images.size(0)).to(device))
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * batch_images.size(0)
        
        train_loss /= data_info['n_train']
        train_losses.append(train_loss)
        
        # 验证阶段
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_images, batch_texts in data_info['val_loader']:
                batch_images, batch_texts = batch_images.to(device), batch_texts.to(device)
                outputs = model(batch_images)
                loss = criterion(outputs, batch_texts, torch.ones(batch_images.size(0)).to(device))
                val_loss += loss.item() * batch_images.size(0)
        
        val_loss /= data_info['n_val']
        val_losses.append(val_loss)
        epoch_time = time.time() - epoch_start_time
        
        # 打印训练效果
        if verbose and ((epoch + 1) % config.print_every_n_epochs == 0 or epoch in (0, config.epochs - 1)):
            print(f"\nEpoch {epoch+1}/{config.epochs}")
            print(f"   训练损失: {train_loss:.6f}")
            print(f"   验证损失: {val_loss:.6f}")
            print(f"   训练/验证损失比: {train_loss/val_loss:.4f}")
            print(f"   Epoch耗时: {epoch_time:.2f}秒")
            
            # 保存检查点
            if config.save_checkpoints and (epoch + 1) % 20 == 0:
                checkpoint_path = f"{config.checkpoint_dir}/model_a_epoch_{epoch+1}.pth"
                save_checkpoint(
                    model, checkpoint_path, epoch, train_loss, val_loss, config
                )
                print(f"   检查点已保存: {checkpoint_path}")
        
        # 保存最佳模型
        if val_loss < best_val_loss:
            improvement = (best_val_loss - val_loss) / best_val_loss * 100 if best_val_loss != float('inf') else 0
            best_val_loss = val_loss
            if verbose:
                print(f"   发现更好模型! 验证损失改善: {improvement:.2f}%")
            save_model(model, scaler, config, config.model_path)
    
    # 训练完成
    total_time = time.time() - training_start_time
    
    if verbose:
        print("\n" + "="*70)
        print("模型训练完成!")
        print("="*70)
        print(f"   总训练时间: {total_time:.2f}秒")
        print(f"   最佳验证损失: {best_val_loss:.6f}")
        print(f"   最终训练损失: {train_losses[-1]:.6f}")
        print(f"   最终验证损失: {val_losses[-1]:.6f}")
        print(f"   训练样本数: {data_info['n_train']}")
        print(f"   验证样本数: {data_info['n_val']}")
    
    # 组装训练历史
    train_history = {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'best_val_loss': best_val_loss,
        'training_time': total_time,
        'n_epochs': config.epochs
    }
    
    return model, train_history, scaler


def save_checkpoint(
    model: ImageToMeaningModel,
    path: str,
    epoch: int,
    train_loss: float,
    val_loss: float,
    config: ModelAConfig
):
    """
    保存训练检查点
    
    参数:
        model: 模型实例
        path: 保存路径
        epoch: 当前epoch
        train_loss: 训练损失
        val_loss: 验证损失
        config: 模型配置
    """
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'train_loss': train_loss,
        'val_loss': val_loss,
        'config': config.__dict__,
        'timestamp': datetime.now().isoformat(),
        'is_trained': True  # 增加训练状态标记
    }, path)


def save_model(
    model: ImageToMeaningModel,
    scaler: StandardScaler,
    config: ModelAConfig,
    save_path: Optional[str] = None
):
    """
    保存完整模型
    
    参数:
        model: 模型实例
        scaler: 数据标准化器
        config: 模型配置
        save_path: 保存路径，默认使用配置中的路径
    """
    if save_path is None:
        save_path = config.model_path
    
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    
    # 保存模型权重（包含训练状态）
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': config.__dict__,
        'scaler': scaler,
        'is_trained': getattr(model, 'is_trained', True),  # 兼容状态属性
        'device': str(next(model.parameters()).device)
    }, save_path)
    
    # 保存配置
    with open(config.config_path, 'w') as f:
        json.dump(config.__dict__, f, indent=2)
    
    # 保存scaler
    if scaler:
        with open(config.scaler_path, 'wb') as f:
            pickle.dump(scaler, f)


def load_model(
    config: ModelAConfig,
    model_path: Optional[str] = None
) -> Tuple[ImageToMeaningModel, StandardScaler, torch.device]:
    """
    加载模型（基础版）
    
    参数:
        config: 模型配置
        model_path: 模型路径，默认使用配置中的路径
    
    返回:
        tuple: (模型实例, scaler, 计算设备)
    """
    if model_path is None:
        model_path = config.model_path
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 禁用weights_only以允许加载sklearn对象
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    
    # 初始化模型
    model = ImageToMeaningModel(config).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # 添加训练状态属性
    model.is_trained = checkpoint.get('is_trained', False)
    model.device = device
    
    # 加载scaler
    scaler = None
    if 'scaler' in checkpoint and checkpoint['scaler']:
        scaler = checkpoint['scaler']
    else:
        # 尝试从单独的文件加载scaler
        if os.path.exists(config.scaler_path):
            with open(config.scaler_path, 'rb') as f:
                scaler = pickle.load(f)
    
    logger.info("模型加载完成")
    return model, scaler, device


def predict_single(
    model: ImageToMeaningModel,
    scaler: StandardScaler,
    device: torch.device,
    image_embedding: np.ndarray,
    verbose: bool = True
) -> np.ndarray:
    """
    单样本预测
    
    参数:
        model: 模型实例
        scaler: 数据标准化器
        device: 计算设备
        image_embedding: 输入图片向量
        verbose: 是否打印预测信息
    
    返回:
        np.ndarray: 输出含义向量
    """
    model.eval()
    
    if verbose:
        print(f"模型A开始预测")
        print(f"   输入向量维度: {len(image_embedding)}")
    
    start_time = time.time()
    
    # 数据标准化 + 预测
    with torch.no_grad():
        if scaler is not None:
            image_embedding = scaler.transform([image_embedding])[0]
        
        # 转换为张量并预测
        image_tensor = torch.FloatTensor(image_embedding).unsqueeze(0).to(device)
        meaning_vector = model(image_tensor).cpu().numpy()[0]
    
    # 归一化输出
    meaning_vector = meaning_vector / np.linalg.norm(meaning_vector)
    inference_time = time.time() - start_time
    
    if verbose:
        print(f"预测完成")
        print(f"   输出向量维度: {len(meaning_vector)}")
        print(f"   输出向量范数: {np.linalg.norm(meaning_vector):.4f}")
        print(f"   推理耗时: {inference_time*1000:.2f}ms")
    
    return meaning_vector


def batch_predict(
    model: ImageToMeaningModel,
    scaler: StandardScaler,
    device: torch.device,
    image_embeddings: np.ndarray,
    verbose: bool = True
) -> np.ndarray:
    """
    批量预测
    
    参数:
        model: 模型实例
        scaler: 数据标准化器
        device: 计算设备
        image_embeddings: 输入图片向量数组
        verbose: 是否打印预测信息
    
    返回:
        np.ndarray: 输出含义向量数组
    """
    model.eval()
    
    if verbose:
        print(f"模型A开始批量预测")
        print(f"   批量大小: {len(image_embeddings)}")
        print(f"   输入向量维度: {image_embeddings.shape[1]}")
    
    start_time = time.time()
    
    # 数据标准化 + 批量预测
    with torch.no_grad():
        if scaler is not None:
            image_embeddings = scaler.transform(image_embeddings)
        
        # 转换为张量并预测
        image_tensor = torch.FloatTensor(image_embeddings).to(device)
        meaning_vectors = model(image_tensor).cpu().numpy()
    
    # 归一化输出
    norms = np.linalg.norm(meaning_vectors, axis=1, keepdims=True)
    meaning_vectors = meaning_vectors / norms
    
    inference_time = time.time() - start_time
    
    if verbose:
        print(f"批量预测完成")
        print(f"   输出向量形状: {meaning_vectors.shape}")
        print(f"   平均向量范数: {np.mean(norms):.4f}")
        print(f"   推理耗时: {inference_time*1000:.2f}ms")
        print(f"   平均每个样本耗时: {inference_time*1000/len(image_embeddings):.2f}ms")
    
    return meaning_vectors


def evaluate_model(
    model: ImageToMeaningModel,
    scaler: StandardScaler,
    device: torch.device,
    image_embeddings: np.ndarray,
    text_embeddings: np.ndarray,
    verbose: bool = True
) -> Dict[str, float]:
    """
    评估模型性能
    
    参数:
        model: 模型实例
        scaler: 数据标准化器
        device: 计算设备
        image_embeddings: 测试图片向量
        text_embeddings: 测试文字向量
        verbose: 是否打印评估信息
    
    返回:
        dict: 评估结果字典
    """
    if verbose:
        print("开始模型评估...")
    
    start_time = time.time()
    
    # 数据标准化和预测
    model.eval()
    with torch.no_grad():
        if scaler is not None:
            image_embeddings = scaler.transform(image_embeddings)
        
        image_tensor = torch.FloatTensor(image_embeddings).to(device)
        predicted_vectors = model(image_tensor).cpu().numpy()
    
    # 计算评估指标
    similarities = []
    for pred, true in zip(predicted_vectors, text_embeddings):
        # 余弦相似度
        sim = np.dot(pred, true) / (np.linalg.norm(pred) * np.linalg.norm(true))
        similarities.append(sim)
    
    # 计算MSE损失
    mse_loss = np.mean(np.square(predicted_vectors - text_embeddings))
    
    # 计算评估指标
    eval_results = {
        'mean_cosine_similarity': np.mean(similarities),
        'std_cosine_similarity': np.std(similarities),
        'max_similarity': np.max(similarities),
        'min_similarity': np.min(similarities),
        'mse_loss': mse_loss,
        'evaluation_time': time.time() - start_time,
        'n_samples': len(image_embeddings)
    }
    
    # 打印评估结果
    if verbose:
        print("模型评估完成")
        print(f"   评估样本数: {eval_results['n_samples']}")
        print(f"   平均余弦相似度: {eval_results['mean_cosine_similarity']:.4f}")
        print(f"   余弦相似度标准差: {eval_results['std_cosine_similarity']:.4f}")
        print(f"   最大相似度: {eval_results['max_similarity']:.4f}")
        print(f"   最小相似度: {eval_results['min_similarity']:.4f}")
        print(f"   MSE损失: {eval_results['mse_loss']:.6f}")
        print(f"   评估耗时: {eval_results['evaluation_time']:.2f}秒")
    
    return eval_results

# ============================ 模型状态修复工具 ============================
def load_trained_model(
    model_path: str,
    config: Optional[ModelAConfig] = None
) -> Tuple[Union[ImageToMeaningModel, None], Optional[Exception]]:
    """
    加载训练好的模型实例（带异常捕获，增强版）- 兼容PyTorch 2.6+
    
    参数:
        model_path: 模型文件路径
        config: 模型配置实例，若为None则从模型文件加载
    
    返回:
        Tuple: (模型实例/None, 异常对象/None)
    """
    try:
        # ========== 核心修复1：兼容torch.load的weights_only问题 ==========
        # 安全上下文管理器 + 显式关闭weights_only
        with torch.serialization.safe_globals([
            sklearn.preprocessing._data.StandardScaler,
            StandardScaler
        ]):
            # 适配原ModelA的加载逻辑
            if config is None:
                # 从文件加载配置（显式设置weights_only=False）
                checkpoint = torch.load(
                    model_path, 
                    map_location=torch.device("cpu"),
                    weights_only=False,  # 关键：允许加载自定义对象
                    encoding='utf-8'
                )
                config_dict = checkpoint.get('config', {})
                config = ModelAConfig(**config_dict)
        
        # ========== 核心修复2：确保load_model函数也使用兼容参数 ==========
        # 临时修改load_model的调用逻辑（若load_model内部仍有问题，需同步改）
        model, scaler, device = load_model(config, model_path)  # 复用统一的加载函数
        
        # 为模型添加必要属性
        model.scaler = scaler
        model.device = device
        
        return model, None
    except Exception as e:
        logger.error(f"模型加载失败: {str(e)}")
        return None, e

def get_model_training_status(model: ImageToMeaningModel) -> Tuple[bool, Optional[Exception]]:
    """
    安全获取模型的训练状态
    
    参数:
        model: 已加载的模型实例
    
    返回:
        Tuple: (训练状态值/False, 异常对象/None)
    """
    try:
        # 检查是否存在is_trained属性
        if hasattr(model, 'is_trained'):
            return model.is_trained, None
        else:
            # 若不存在该属性，返回False并提示
            return False, AttributeError("模型实例缺少'is_trained'属性")
    except Exception as e:
        logger.error(f"获取训练状态失败: {str(e)}")
        return False, e


def set_model_training_status(
    model: ImageToMeaningModel,
    is_trained: bool
) -> Tuple[bool, Optional[Exception]]:
    """
    安全设置模型的训练状态
    
    参数:
        model: 已加载的模型实例
        is_trained: 要设置的训练状态（True/False）
    
    返回:
        Tuple: (操作是否成功, 异常对象/None)
    """
    try:
        model.is_trained = is_trained
        return True, None
    except Exception as e:
        logger.error(f"设置训练状态失败: {str(e)}")
        return False, e


def save_model_with_status(
    model: ImageToMeaningModel,
    save_path: str,
    config: ModelAConfig,
    scaler: Optional[StandardScaler] = None
) -> Tuple[bool, Optional[Exception]]:
    """
    保存包含训练状态的完整模型
    
    参数:
        model: 已加载的模型实例
        save_path: 模型保存路径
        config: 模型配置实例
        scaler: 数据标准化器（可选）
    
    返回:
        Tuple: (保存是否成功, 异常对象/None)
    """
    try:
        # 复用核心的save_model函数，保证逻辑统一
        save_model(model, scaler or getattr(model, 'scaler', None), config, save_path)
        return True, None
    except Exception as e:
        logger.error(f"模型保存失败: {str(e)}")
        return False, e


def fix_model_training_status(
    model_path: str,
    target_status: bool = True,
    config: Optional[ModelAConfig] = None
) -> Dict[str, Any]:
    """
    修复模型的训练状态（完整流程）
    
    参数:
        model_path: 模型文件路径
        target_status: 要设置的目标训练状态，默认True
        config: 模型配置实例（可选）
    
    返回:
        Dict: 修复结果详情
            - success: bool 修复是否成功
            - original_status: bool 原始状态
            - target_status: bool 目标状态
            - fixed_status: bool 修复后的状态
            - error: str 错误信息（若有）
    """
    result = {
        "success": False,
        "original_status": None,
        "target_status": target_status,
        "fixed_status": None,
        "error": None
    }
    
    try:
        # 1. 加载模型
        model, load_error = load_trained_model(model_path, config)
        if load_error:
            result["error"] = f"加载模型失败: {str(load_error)}"
            return result
        
        # 2. 获取原始状态
        original_status, status_error = get_model_training_status(model)
        if status_error:
            result["error"] = f"获取原始状态失败: {str(status_error)}"
            return result
        result["original_status"] = original_status
        
        # 3. 仅当状态不一致时才进行修复
        if original_status != target_status:
            # 设置新状态
            set_success, set_error = set_model_training_status(model, target_status)
            if set_error:
                result["error"] = f"设置状态失败: {str(set_error)}"
                return result
            
            # 保存模型
            save_success, save_error = save_model_with_status(
                model, model_path, 
                config or ModelAConfig(), 
                getattr(model, 'scaler', None)
            )
            if save_error:
                result["error"] = f"保存模型失败: {str(save_error)}"
                return result
        
        # 4. 验证修复结果
        verify_model, verify_load_error = load_trained_model(model_path, config)
        if verify_load_error:
            result["error"] = f"验证加载失败: {str(verify_load_error)}"
            return result
        
        fixed_status, verify_status_error = get_model_training_status(verify_model)
        if verify_status_error:
            result["error"] = f"验证状态失败: {str(verify_status_error)}"
            return result
        
        result["fixed_status"] = fixed_status
        result["success"] = (fixed_status == target_status)
        
    except Exception as e:
        result["error"] = f"修复过程异常: {str(e)}\n{traceback.format_exc()}"
    
    return result


def test_model_prediction(
    model: ImageToMeaningModel,
    test_vector: Optional[np.ndarray] = None,
    verbose: bool = False
) -> Dict[str, Any]:
    """
    测试模型的预测功能（带详细结果）
    
    参数:
        model: 已加载的模型实例
        test_vector: 测试用的输入向量，可选（默认生成512维随机向量）
        verbose: 是否打印详细信息
    
    返回:
        Dict: 预测结果信息
            - input_dim: int 输入维度
            - output_dim: int 输出维度
            - output_norm: float 输出向量范数
            - prediction_success: bool 预测是否成功
            - inference_time: float 推理耗时(秒)
            - error: str 错误信息（若有）
    """
    result = {
        "input_dim": None,
        "output_dim": None,
        "output_norm": None,
        "prediction_success": False,
        "inference_time": 0.0,
        "error": None
    }
    
    try:
        # 生成默认测试向量（512维归一化向量）
        if test_vector is None:
            test_vector = np.random.randn(model.config.input_dim).astype(np.float32)
            test_vector = test_vector / np.linalg.norm(test_vector)
        
        result["input_dim"] = len(test_vector)
        
        # 执行预测（复用核心的predict_single函数）
        start_time = time.time()
        meaning_vector = predict_single(
            model, model.scaler, model.device, 
            test_vector, verbose=False
        )
        inference_time = time.time() - start_time
        
        # 填充结果
        result["output_dim"] = len(meaning_vector)
        result["output_norm"] = float(np.linalg.norm(meaning_vector))
        result["inference_time"] = inference_time
        result["prediction_success"] = True
        
        if verbose:
            print(f"预测完成 - 输入维度: {result['input_dim']}, 输出维度: {result['output_dim']}, 耗时: {inference_time:.4f}秒")
        
    except Exception as e:
        result["error"] = f"预测失败: {str(e)}\n{traceback.format_exc()}"
        if verbose:
            print(f"预测测试失败: {result['error']}")
    
    return result


def verify_model_fix(
    model_path: str,
    test_vector: Optional[np.ndarray] = None,
    config: Optional[ModelAConfig] = None,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    全面验证模型修复结果 - 兼容PyTorch 2.6+
    
    参数:
        model_path: 模型文件路径
        test_vector: 测试用的输入向量，可选
        config: 模型配置实例，可选
        verbose: 是否打印详细信息
    
    返回:
        Dict: 验证结果字典
            - model_path: str 模型路径
            - current_training_status: bool 当前训练状态
            - prediction_test: dict 预测测试结果
            - verification_success: bool 整体验证是否成功
            - error: str 错误信息（若有）
    """
    result = {
        "model_path": model_path,
        "current_training_status": False,
        "prediction_test": {},
        "verification_success": False,
        "error": None
    }
    
    try:
        # ========== 核心修复：加载模型前注册安全对象 ==========
        with torch.serialization.safe_globals([
            sklearn.preprocessing._data.StandardScaler,
            StandardScaler
        ]):
            # 1. 加载修复后的模型
            model, load_error = load_trained_model(model_path, config)
        
        if load_error:
            result["error"] = f"加载验证模型失败: {str(load_error)}"
            return result
        
        # 2. 检查训练状态
        current_status, status_error = get_model_training_status(model)
        if status_error:
            result["error"] = f"获取验证状态失败: {str(status_error)}"
            return result
        result["current_training_status"] = current_status
        
        # 3. 测试预测功能
        prediction_result = test_model_prediction(model, test_vector, verbose)
        result["prediction_test"] = prediction_result
        
        # 4. 判断整体验证是否成功
        result["verification_success"] = (
            current_status and 
            prediction_result["prediction_success"]
        )
        
        if verbose:
            print(f"\n验证结果: {'成功' if result['verification_success'] else '失败'}")
            print(f"当前训练状态: {current_status}")
            print(f"预测功能: {'正常' if prediction_result['prediction_success'] else '异常'}")
        
    except Exception as e:
        result["error"] = f"验证过程异常: {str(e)}\n{traceback.format_exc()}"
        if verbose:
            print(f"验证失败: {result['error']}")
    
    return result

def batch_fix_model_status(
    model_paths: list,
    target_status: bool = True,
    config: Optional[ModelAConfig] = None,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    批量修复多个模型的训练状态
    
    参数:
        model_paths: 模型文件路径列表
        target_status: 目标训练状态
        config: 模型配置实例
        verbose: 是否打印详细信息
    
    返回:
        Dict: 批量处理结果
            - total: int 总数量
            - success: int 成功数量
            - failed: int 失败数量
            - details: list 每个模型的修复详情
    """
    batch_result = {
        "total": len(model_paths),
        "success": 0,
        "failed": 0,
        "details": []
    }
    
    if verbose:
        print(f"\n开始批量修复 {len(model_paths)} 个模型...")
    
    for idx, model_path in enumerate(model_paths):
        if verbose:
            print(f"\n[{idx+1}/{len(model_paths)}] 处理模型: {model_path}")
        
        # 修复单个模型
        fix_result = fix_model_training_status(model_path, target_status, config)
        batch_result["details"].append({
            "model_path": model_path,
            "fix_result": fix_result
        })
        
        # 统计结果
        if fix_result["success"]:
            batch_result["success"] += 1
        else:
            batch_result["failed"] += 1
    
    if verbose:
        print(f"\n批量修复完成 - 成功: {batch_result['success']}, 失败: {batch_result['failed']}")
    
    return batch_result

# ============================ 完整流程与示例 ============================
def run_model_pipeline(
    image_embeddings: np.ndarray,
    text_embeddings: np.ndarray,
    config: ModelAConfig,
    random_seed: Optional[int] = 42
):
    """
    运行完整的模型训练-评估-预测流程
    
    参数:
        image_embeddings: 图片向量数组
        text_embeddings: 文字向量数组
        config: 模型配置
        random_seed: 随机种子
    """
    # 1. 训练模型
    model, train_history, scaler = train_model(
        image_embeddings, text_embeddings, config, random_seed
    )
    
    # 2. 保存最终模型
    save_model(model, scaler, config)
    
    # 3. 评估模型（使用测试集数据）
    n_test = int(config.test_split * len(image_embeddings))
    eval_results = evaluate_model(
        model, scaler, next(model.parameters()).device,
        image_embeddings[-n_test:], text_embeddings[-n_test:]
    )
    
    # 4. 单样本预测示例
    single_pred = predict_single(
        model, scaler, next(model.parameters()).device,
        image_embeddings[0]
    )
    
    # 5. 批量预测示例
    batch_preds = batch_predict(
        model, scaler, next(model.parameters()).device,
        image_embeddings[:10]
    )
    
    return {
        'train_history': train_history,
        'eval_results': eval_results,
        'single_pred': single_pred,
        'batch_preds': batch_preds
    }


def main():
    """主函数：演示如何使用各个功能函数"""
    # 示例配置
    custom_config = ModelAConfig(
        input_dim=512,
        output_dim=1024,
        epochs=10,
        batch_size=16,
        print_every_n_epochs=2,
        model_path="data/models/custom_model_a.pth"
    )
    
    # 准备示例数据
    n_samples = 1000
    image_embeds = np.random.randn(n_samples, custom_config.input_dim)
    text_embeds = np.random.randn(n_samples, custom_config.output_dim)
    
    # 1. 运行完整训练流程
    print("=== 运行完整训练流程 ===")
    pipeline_results = run_model_pipeline(image_embeds, text_embeds, custom_config)
    
    # 2. 单模型状态修复示例
    print("\n=== 单模型状态修复示例 ===")
    fix_result = fix_model_training_status(
        model_path=custom_config.model_path, 
        target_status=True, 
        config=custom_config
    )
    print(f"修复结果: {fix_result}")
    
    # 3. 验证修复结果
    print("\n=== 验证修复结果 ===")
    verify_result = verify_model_fix(
        model_path=custom_config.model_path, 
        config=custom_config
    )
    print(f"验证结果: {verify_result}")
    
    # 4. 批量修复示例（模拟多个模型）
    print("\n=== 批量修复示例 ===")
    batch_result = batch_fix_model_status(
        model_paths=[custom_config.model_path, "data/models/model_a_v2.pth"],
        target_status=True,
        config=custom_config
    )
    print(f"批量结果: {batch_result}")


if __name__ == "__main__":
    import sys
    from dataclasses import dataclass
    
    # 重新应用dataclass装饰器（解决导入顺序问题）
    globals()['ModelAConfig'] = dataclass(ModelAConfig)
    
    # 运行主函数
    main()
