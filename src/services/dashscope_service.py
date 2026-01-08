#!/usr/bin/env python3
"""
阿里云DashScope API服务
用于图像向量化和多模态处理
使用官方dashscope库
"""

import os
import logging
import time
import base64
from typing import List, Optional, Dict, Any
from dataclasses import dataclass
from pathlib import Path

import dashscope
from dashscope import Generation, MultiModalEmbedding
from PIL import Image
import io

logger = logging.getLogger(__name__)

# 设置DashScope API基础URL
dashscope.base_http_api_url = 'https://dashscope.aliyuncs.com/api/v1'


@dataclass
class ImageEmbeddingResult:
    """图像嵌入结果"""
    embedding: List[float]
    model: str
    dimension: int
    duration_ms: float


@dataclass
class TextEmbeddingResult:
    """文本嵌入结果"""
    embedding: List[float]
    model: str
    dimension: int
    duration_ms: float


@dataclass
class ImageDescriptionResult:
    """图像描述结果"""
    description: str
    model: str
    duration_ms: float


class DashScopeService:
    """阿里云DashScope API服务"""
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("DASHSCOPE_API_KEY")
        if not self.api_key:
            raise ValueError("DASHSCOPE_API_KEY环境变量未设置")
        
        # 设置API Key
        dashscope.api_key = self.api_key
        
        self.timeout = 30
        self.max_retries = 3
        self.retry_delay = 1
    
    def _preprocess_image(self, image_path: str) -> Optional[str]:
        """预处理图片：调整大小、转换格式"""
        try:
            with Image.open(image_path) as img:
                # 转换为RGB
                img = img.convert("RGB")
                
                # 调整大小（DashScope推荐不超过1024x1024）
                max_size = (1024, 1024)
                img.thumbnail(max_size, Image.Resampling.LANCZOS)
                
                # 保存为JPEG格式到内存
                img_byte_arr = io.BytesIO()
                img.save(img_byte_arr, format='JPEG', quality=85)
                img_byte_arr = img_byte_arr.getvalue()
                
                # 编码为base64
                return base64.b64encode(img_byte_arr).decode("utf-8")
        except Exception as e:
            logger.error(f"图片预处理失败 {image_path}: {e}")
            return None
    
    def generate_image_embedding(self, image_path: str, model: str = "qwen3-max") -> ImageEmbeddingResult:
        """生成图像向量（使用多模态模型）"""
        start_time = time.time()
        
        # 预处理图片
        image_base64 = self._preprocess_image(image_path)
        if not image_base64:
            raise ValueError(f"图片预处理失败: {image_path}")
        
        try:
            # 使用MultiModalEmbedding生成图像向量
            response = MultiModalEmbedding.call(
                model=model,
                input={
                    "image": f"data:image/jpeg;base64,{image_base64}"
                },
                timeout=self.timeout
            )
            
            if response.status_code != 200:
                raise Exception(f"DashScope API错误: {response.message}")
            
            duration_ms = (time.time() - start_time) * 1000
            embedding = response.output.embeddings[0].embedding
            dimension = len(embedding)
            
            logger.info(f"生成图像向量成功: {image_path}, "
                       f"维度: {dimension}, 耗时: {duration_ms:.2f}ms")
            
            return ImageEmbeddingResult(
                embedding=embedding,
                model=model,
                dimension=dimension,
                duration_ms=duration_ms
            )
            
        except Exception as e:
            logger.error(f"生成图像向量失败: {e}")
            raise
    
    def generate_text_embedding(self, text: str, model: str = "text-embedding-v2") -> TextEmbeddingResult:
        """生成文本向量"""
        start_time = time.time()
        
        try:
            # 使用MultiModalEmbedding生成文本向量
            response = MultiModalEmbedding.call(
                model=model,
                input={
                    "texts": [text]
                },
                timeout=self.timeout
            )
            
            if response.status_code != 200:
                raise Exception(f"DashScope API错误: {response.message}")
            
            duration_ms = (time.time() - start_time) * 1000
            embedding = response.output.embeddings[0].embedding
            dimension = len(embedding)
            
            logger.debug(f"生成文本向量成功: {len(text)}字符, "
                        f"维度: {dimension}, 耗时: {duration_ms:.2f}ms")
            
            return TextEmbeddingResult(
                embedding=embedding,
                model=model,
                dimension=dimension,
                duration_ms=duration_ms
            )
            
        except Exception as e:
            logger.error(f"生成文本向量失败: {e}")
            raise
    
    def generate_image_description(self, image_path: str, model: str = "qwen3-max") -> ImageDescriptionResult:
        """生成图像描述"""
        start_time = time.time()
        
        # 预处理图片
        image_base64 = self._preprocess_image(image_path)
        if not image_base64:
            raise ValueError(f"图片预处理失败: {image_path}")
        
        # 构建消息
        messages = [
            {
                "role": "user",
                "content": [
                    {"image": f"data:image/jpeg;base64,{image_base64}"},
                    {"text": "请用中文详细描述这张图片的内容，包括场景、物体、颜色、布局等关键信息"}
                ]
            }
        ]
        
        try:
            response = Generation.call(
                model=model,
                messages=messages,
                result_format="message",
                temperature=0.1,
                top_p=0.9,
                timeout=self.timeout
            )
            
            if response.status_code != 200:
                raise Exception(f"DashScope API错误: {response.message}")
            
            duration_ms = (time.time() - start_time) * 1000
            description = response.output.choices[0].message.content
            
            logger.info(f"生成图像描述成功: {image_path}, "
                       f"长度: {len(description)}字符, 耗时: {duration_ms:.2f}ms")
            
            return ImageDescriptionResult(
                description=description,
                model=model,
                duration_ms=duration_ms
            )
            
        except Exception as e:
            logger.error(f"生成图像描述失败: {e}")
            raise
    
    def generate_chat_response(self, messages: List[Dict[str, Any]], model: str = "qwen3-max", 
                              temperature: float = 0.7, top_p: float = 0.9) -> str:
        """生成聊天响应"""
        start_time = time.time()
        
        try:
            response = Generation.call(
                model=model,
                messages=messages,
                result_format="message",
                temperature=temperature,
                top_p=top_p,
                timeout=self.timeout
            )
            
            if response.status_code != 200:
                raise Exception(f"DashScope API错误: {response.message}")
            
            duration_ms = (time.time() - start_time) * 1000
            content = response.output.choices[0].message.content
            
            logger.info(f"生成聊天响应成功, "
                       f"长度: {len(content)}字符, 耗时: {duration_ms:.2f}ms")
            
            return content
            
        except Exception as e:
            logger.error(f"生成聊天响应失败: {e}")
            raise
    
    def batch_generate_image_embeddings(self, image_paths: List[str], 
                                       model: str = "qwen3-max") -> List[ImageEmbeddingResult]:
        """批量生成图像向量"""
        results = []
        for path in image_paths:
            try:
                result = self.generate_image_embedding(path, model)
                results.append(result)
            except Exception as e:
                logger.error(f"处理图像失败 {path}: {e}")
                results.append(e)
        return results
    
    def batch_generate_text_embeddings(self, texts: List[str], 
                                      model: str = "text-embedding-v2") -> List[TextEmbeddingResult]:
        """批量生成文本向量"""
        results = []
        for text in texts:
            try:
                result = self.generate_text_embedding(text, model)
                results.append(result)
            except Exception as e:
                logger.error(f"处理文本失败: {e}")
                results.append(e)
        return results


# 全局DashScope服务实例
def get_dashscope_service() -> DashScopeService:
    """获取DashScope服务实例"""
    return DashScopeService()
