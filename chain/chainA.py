#!/usr/bin/env python3
"""
chainA：图片查询处理链
功能：图片 → BGE模型 → 图片编码 → 模型A → 关键词
"""

import sys
import asyncio
from pathlib import Path
from typing import Dict, Any, Optional
import numpy as np

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# 导入必要的函数
from function.api_function import generate_image_embedding
from function.model_function import (
    ModelAConfig,
    load_model,
    predict_single
)


class ChainAConfig:
    """chainA配置"""
    def __init__(self):
        # 模型A配置
        self.model_a_config = ModelAConfig()
        
        # 图片处理配置
        self.image_embedding_model = "bge-m3"  # 用于图片编码的模型
        self.image_embedding_dim = 512  # CLIP图片向量维度（512维）
        
        # 输出配置
        self.verbose = True  # 是否打印详细过程


async def process_image_query(
    image_path: str,
    config: Optional[ChainAConfig] = None
) -> Dict[str, Any]:
    """
    处理图片查询：图片 → BGE模型 → 图片编码 → 模型A → 关键词
    
    参数:
        image_path: 图片文件路径
        config: chainA配置，可选
    
    返回:
        dict: 查询结果
            - success: bool 是否成功
            - keyword: str 预测关键词
            - confidence: float 置信度
            - image_embedding: list 图片向量（可选）
            - error: str 错误信息（若有）
    """
    if config is None:
        config = ChainAConfig()
    
    result = {
        "success": False,
        "keyword": None,
        "confidence": None,
        "image_embedding": None,
        "error": None
    }
    
    try:
        # 1. 检查图片文件是否存在
        img_path = Path(image_path)
        if not img_path.exists():
            result["error"] = f"图片文件不存在: {image_path}"
            return result
        
        if config.verbose:
            print(f"[chainA] 开始处理图片查询")
            print(f"   图片路径: {image_path}")
            print(f"   图片大小: {img_path.stat().st_size} bytes")
        
        # 2. 生成图片向量（使用BGE模型）
        if config.verbose:
            print(f"[chainA] 生成图片向量...")
        
        try:
            # 调用api_function生成图片向量
            # 注意：现在需要传入text_label参数，但chainA不知道图片对应的文本标签
            # 使用默认标签"default image"
            image_embedding = await generate_image_embedding(
                image_path=image_path,
                model_name=config.image_embedding_model,
                text_label="default image"  # 添加默认文本标签
            )
            
            if not image_embedding:
                result["error"] = "图片向量生成失败"
                return result
            
            # 关键修复1：将list转换为numpy数组，方便后续维度调整（先拷贝，避免修改原列表）
            image_embedding_np = np.array(image_embedding, dtype=np.float32)
            
            # 确保向量维度正确
            if len(image_embedding_np) != config.image_embedding_dim:
                if config.verbose:
                    print(f"[chainA] 警告: 图片向量维度 {len(image_embedding_np)} != 预期 {config.image_embedding_dim}")
                # 尝试调整维度
                if len(image_embedding_np) > config.image_embedding_dim:
                    image_embedding_np = image_embedding_np[:config.image_embedding_dim]
                else:
                    # 填充到正确维度
                    padding = np.zeros(config.image_embedding_dim - len(image_embedding_np), dtype=np.float32)
                    image_embedding_np = np.concatenate([image_embedding_np, padding])
            
            # 关键修复2：numpy数组转list存入结果（list无tolist()方法，numpy数组才有）
            result["image_embedding"] = image_embedding_np.tolist()
            
            if config.verbose:
                print(f"[chainA] 图片向量生成完成")
                print(f"   向量维度: {len(image_embedding_np)}")
                print(f"   向量范数: {np.linalg.norm(image_embedding_np):.4f}")
        
        except Exception as e:
            result["error"] = f"图片向量生成异常: {str(e)}"
            return result
        
        # 3. 加载模型A（图片到关键词分类模型）
        if config.verbose:
            print(f"[chainA] 加载模型A...")
        
        try:
            model, scaler, label_encoder, device = load_model(config.model_a_config)
            
            if config.verbose:
                print(f"[chainA] 模型A加载成功")
                print(f"   设备: {device}")
                print(f"   关键词类别: {label_encoder.classes_}")
        
        except Exception as e:
            result["error"] = f"模型A加载失败: {str(e)}"
            return result
        
        # 4. 使用模型A预测关键词
        if config.verbose:
            print(f"[chainA] 预测关键词...")
        
        try:
            # 关键修复3：传入调整后的numpy数组（模型预测需要numpy数组，而非list）
            keyword, confidence = predict_single(
                model=model,
                scaler=scaler,
                label_encoder=label_encoder,
                device=device,
                image_embedding=image_embedding_np,  # 传入numpy数组
                verbose=config.verbose
            )
            
            result["keyword"] = keyword
            result["confidence"] = float(confidence)
            result["success"] = True
            
            if config.verbose:
                print(f"[chainA] 预测完成")
                print(f"   关键词: {keyword}")
                print(f"   置信度: {confidence:.4f}")
        
        except Exception as e:
            result["error"] = f"关键词预测失败: {str(e)}"
            return result
    
    except Exception as e:
        result["error"] = f"chainA处理异常: {str(e)}"
    
    return result


if __name__ == "__main__":
    print("chainA模块 - 图片查询处理链")
    print("使用方法: 作为模块导入使用，不直接运行")
    print("示例: from chain.chainA import process_image_query")
