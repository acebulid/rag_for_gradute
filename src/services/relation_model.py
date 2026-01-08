#!/usr/bin/env python3
"""
关联模型训练模块
训练文字向量到图片向量的映射关系
"""

import logging
import numpy as np
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass
import pickle
from pathlib import Path
import json

from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
import joblib

logger = logging.getLogger(__name__)


@dataclass
class TrainingPair:
    """训练数据对"""
    text_embedding: np.ndarray  # 文字向量
    image_embedding: np.ndarray  # 图片向量
    text_id: str  # 文字ID
    image_id: str  # 图片ID
    similarity_score: float  # 相似度分数


@dataclass
class RelationModelConfig:
    """关联模型配置"""
    model_type: str = "knn"  # 模型类型: knn, linear, mlp
    n_neighbors: int = 5  # KNN的K值
    metric: str = "cosine"  # 距离度量: cosine, euclidean
    normalize: bool = True  # 是否标准化
    model_path: str = "data/models/relation_model.pkl"  # 模型保存路径


class RelationModel:
    """关联模型：文字向量 -> 图片向量"""
    
    def __init__(self, config: Optional[RelationModelConfig] = None):
        self.config = config or RelationModelConfig()
        self.model: Optional[NearestNeighbors] = None
        self.scaler: Optional[StandardScaler] = None
        self.image_embeddings: Optional[np.ndarray] = None
        self.image_ids: List[str] = []
        self.text_embeddings: Optional[np.ndarray] = None
        self.text_ids: List[str] = []
        
    def prepare_training_data(self, training_pairs: List[TrainingPair]) -> Tuple[np.ndarray, np.ndarray]:
        """准备训练数据"""
        text_embeddings = []
        image_embeddings = []
        text_ids = []
        image_ids = []
        
        for pair in training_pairs:
            text_embeddings.append(pair.text_embedding)
            image_embeddings.append(pair.image_embedding)
            text_ids.append(pair.text_id)
            image_ids.append(pair.image_id)
        
        self.text_embeddings = np.array(text_embeddings)
        self.image_embeddings = np.array(image_embeddings)
        self.text_ids = text_ids
        self.image_ids = image_ids
        
        return self.text_embeddings, self.image_embeddings
    
    def train(self, training_pairs: List[TrainingPair]):
        """训练关联模型"""
        logger.info(f"开始训练关联模型，训练数据: {len(training_pairs)} 对")
        
        # 准备数据
        text_embeddings, image_embeddings = self.prepare_training_data(training_pairs)
        
        # 数据标准化
        if self.config.normalize:
            self.scaler = StandardScaler()
            text_embeddings = self.scaler.fit_transform(text_embeddings)
            logger.info("数据标准化完成")
        
        # 训练KNN模型
        self.model = NearestNeighbors(
            n_neighbors=self.config.n_neighbors,
            metric=self.config.metric,
            algorithm='auto'
        )
        self.model.fit(image_embeddings)
        
        logger.info(f"关联模型训练完成，模型类型: {self.config.model_type}, "
                   f"邻居数: {self.config.n_neighbors}, "
                   f"度量: {self.config.metric}")
    
    def predict(self, text_embedding: np.ndarray, top_k: int = 3) -> List[Tuple[str, float]]:
        """预测：根据文字向量找到最相似的图片向量"""
        if self.model is None or self.image_embeddings is None:
            raise ValueError("模型未训练")
        
        # 数据标准化
        if self.config.normalize and self.scaler is not None:
            text_embedding = self.scaler.transform([text_embedding])[0]
        
        # 找到最相似的图片向量
        # 注意：这里我们训练的是图片向量的KNN，所以需要找到与文字向量最相似的图片向量
        # 由于我们训练的是文字->图片的映射，这里使用文字向量作为查询
        
        # 方法1：直接使用文字向量在图片向量空间中搜索
        # 这里简化处理，假设文字向量和图片向量在同一空间
        distances, indices = self.model.kneighbors([text_embedding], n_neighbors=min(top_k, len(self.image_ids)))
        
        results = []
        for i, (dist, idx) in enumerate(zip(distances[0], indices[0])):
            image_id = self.image_ids[idx]
            similarity = 1 - dist if self.config.metric == "cosine" else 1 / (1 + dist)
            results.append((image_id, float(similarity)))
        
        return results
    
    def find_similar_texts(self, image_embedding: np.ndarray, top_k: int = 3) -> List[Tuple[str, float]]:
        """根据图片向量找到最相似的文字向量"""
        if self.model is None or self.text_embeddings is None:
            raise ValueError("模型未训练")
        
        # 找到最相似的文字向量
        # 计算图片向量与所有文字向量的相似度
        similarities = []
        for i, text_emb in enumerate(self.text_embeddings):
            if self.config.metric == "cosine":
                sim = np.dot(image_embedding, text_emb) / (np.linalg.norm(image_embedding) * np.linalg.norm(text_emb))
            else:  # euclidean
                sim = 1 / (1 + np.linalg.norm(image_embedding - text_emb))
            similarities.append((self.text_ids[i], float(sim)))
        
        # 按相似度排序
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        return similarities[:top_k]
    
    def save(self, path: Optional[str] = None):
        """保存模型"""
        save_path = path or self.config.model_path
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        
        model_data = {
            'config': self.config.__dict__,
            'model': self.model,
            'scaler': self.scaler,
            'image_embeddings': self.image_embeddings,
            'image_ids': self.image_ids,
            'text_embeddings': self.text_embeddings,
            'text_ids': self.text_ids
        }
        
        joblib.dump(model_data, save_path)
        logger.info(f"模型已保存到: {save_path}")
    
    def load(self, path: Optional[str] = None):
        """加载模型"""
        load_path = path or self.config.model_path
        
        if not Path(load_path).exists():
            raise FileNotFoundError(f"模型文件不存在: {load_path}")
        
        model_data = joblib.load(load_path)
        
        self.config = RelationModelConfig(**model_data['config'])
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.image_embeddings = model_data['image_embeddings']
        self.image_ids = model_data['image_ids']
        self.text_embeddings = model_data['text_embeddings']
        self.text_ids = model_data['text_ids']
        
        logger.info(f"模型已从 {load_path} 加载")
    
    def evaluate(self, test_pairs: List[TrainingPair]) -> Dict[str, float]:
        """评估模型性能"""
        if self.model is None:
            raise ValueError("模型未训练")
        
        correct_top1 = 0
        correct_top3 = 0
        total = len(test_pairs)
        
        for pair in test_pairs:
            # 预测最相似的图片
            predictions = self.predict(pair.text_embedding, top_k=3)
            predicted_image_ids = [pid for pid, _ in predictions]
            
            # 检查是否预测正确
            if pair.image_id == predicted_image_ids[0]:
                correct_top1 += 1
            if pair.image_id in predicted_image_ids:
                correct_top3 += 1
        
        accuracy_top1 = correct_top1 / total if total > 0 else 0
        accuracy_top3 = correct_top3 / total if total > 0 else 0
        
        metrics = {
            'total_samples': total,
            'accuracy_top1': accuracy_top1,
            'accuracy_top3': accuracy_top3,
            'correct_top1': correct_top1,
            'correct_top3': correct_top3
        }
        
        logger.info(f"模型评估结果: Top-1准确率: {accuracy_top1:.3f}, Top-3准确率: {accuracy_top3:.3f}")
        
        return metrics


class TrainingDataCollector:
    """训练数据收集器"""
    
    def __init__(self, data_path: str = "data/training_pairs.json"):
        self.data_path = data_path
        self.training_pairs: List[TrainingPair] = []
        
        # 加载已有数据
        self.load()
    
    def add_pair(self, text_embedding: np.ndarray, image_embedding: np.ndarray,
                text_id: str, image_id: str, similarity_score: float = 1.0):
        """添加训练对"""
        pair = TrainingPair(
            text_embedding=text_embedding,
            image_embedding=image_embedding,
            text_id=text_id,
            image_id=image_id,
            similarity_score=similarity_score
        )
        self.training_pairs.append(pair)
        logger.debug(f"添加训练对: 文字ID={text_id}, 图片ID={image_id}, 相似度={similarity_score}")
    
    def save(self):
        """保存训练数据"""
        Path(self.data_path).parent.mkdir(parents=True, exist_ok=True)
        
        data = []
        for pair in self.training_pairs:
            data.append({
                'text_embedding': pair.text_embedding.tolist(),
                'image_embedding': pair.image_embedding.tolist(),
                'text_id': pair.text_id,
                'image_id': pair.image_id,
                'similarity_score': pair.similarity_score
            })
        
        with open(self.data_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"训练数据已保存到: {self.data_path}, 共 {len(self.training_pairs)} 对")
    
    def load(self):
        """加载训练数据"""
        if Path(self.data_path).exists():
            with open(self.data_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            self.training_pairs = []
            for item in data:
                pair = TrainingPair(
                    text_embedding=np.array(item['text_embedding']),
                    image_embedding=np.array(item['image_embedding']),
                    text_id=item['text_id'],
                    image_id=item['image_id'],
                    similarity_score=item['similarity_score']
                )
                self.training_pairs.append(pair)
            
            logger.info(f"从 {self.data_path} 加载了 {len(self.training_pairs)} 对训练数据")
    
    def get_training_pairs(self) -> List[TrainingPair]:
        """获取所有训练对"""
        return self.training_pairs
    
    def clear(self):
        """清空训练数据"""
        self.training_pairs = []
        logger.info("训练数据已清空")


# 全局关联模型实例
def get_relation_model(config: Optional[RelationModelConfig] = None) -> RelationModel:
    """获取关联模型实例"""
    return RelationModel(config)