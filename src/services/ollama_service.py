import requests
import time
import logging
import base64
import asyncio
from pathlib import Path
from typing import List, Optional, Dict, Any
from dataclasses import dataclass

from config.settings import settings

logger = logging.getLogger(__name__)


@dataclass
class EmbeddingResult:
    """嵌入结果"""
    embedding: List[float]
    model: str
    tokens: Optional[int] = None
    duration_ms: Optional[float] = None


@dataclass
class GenerationResult:
    """生成结果"""
    text: str
    model: str
    tokens: Optional[int] = None
    duration_ms: Optional[float] = None


class OllamaService:
    """Ollama服务封装 - 使用同步requests，避免会话复用问题"""
    
    def __init__(self, base_url: Optional[str] = None):
        self.base_url = base_url or settings.ollama_base_url
        self.timeout = 30  # 请求超时时间
        self.max_retries = 3  # 最大重试次数
        self.retry_delay = 1  # 重试延迟（秒）

    def _create_new_session(self) -> requests.Session:
        """创建新的会话对象（每次请求都创建新会话，避免复用已关闭的会话）"""
        session = requests.Session()
        session.headers.update({
            "Content-Type": "application/json",
            "Accept": "application/json"
        })
        return session
    
    def _preprocess_image(self, image_path: str) -> Optional[str]:
        """图片预处理：调整大小、转换格式，适配VLM模型"""
        try:
            from PIL import Image
            import io
            
            # 打开图片并转换为RGB（去除透明通道）
            with Image.open(image_path) as img:
                img = img.convert("RGB")
                
                # 调整图片大小（VLM模型推荐不超过1024x1024）
                max_size = (1024, 1024)
                img.thumbnail(max_size, Image.Resampling.LANCZOS)
                
                # 保存为JPEG格式到内存
                img_byte_arr = io.BytesIO()
                img.save(img_byte_arr, format='JPEG', quality=85)
                img_byte_arr = img_byte_arr.getvalue()
                
                # 编码为base64
                return base64.b64encode(img_byte_arr).decode("utf-8")
        except Exception as e:
            logger.error(f"图片预处理失败 {image_path}: {str(e)}")
            return None

    def generate_embedding(self, text: str, model: Optional[str] = None) -> EmbeddingResult:
        """生成文本嵌入向量"""
        start_time = time.time()
        model = model or settings.embedding_model
        
        if not text.strip():
            logger.warning("空文本无法生成嵌入")
            return None

        url = f"{self.base_url}/api/embeddings"
        payload = {
            "model": model,
            "prompt": text
        }

        for attempt in range(self.max_retries):
            # 每次重试都创建新会话
            session = self._create_new_session()
            try:
                response = session.post(
                    url,
                    json=payload,
                    timeout=self.timeout
                )
                response.raise_for_status()  # 抛出HTTP错误
                data = response.json()
                
                duration_ms = (time.time() - start_time) * 1000
                embedding = data.get("embedding", [])
                
                if not embedding:
                    raise ValueError("Empty embedding returned")
                
                logger.debug(f"Generated embedding for {len(text)} chars, "
                            f"dimension: {len(embedding)}, time: {duration_ms:.2f}ms")
                
                return EmbeddingResult(
                    embedding=embedding,
                    model=model,
                    tokens=data.get("total_duration"),
                    duration_ms=duration_ms
                )
            
            except requests.exceptions.RequestException as e:
                logger.warning(f"请求错误 (attempt {attempt+1}/{self.max_retries}): {str(e)}")
                time.sleep(self.retry_delay * (attempt + 1))  # 指数退避
                continue
            
            finally:
                # 确保会话关闭
                session.close()

        logger.error(f"Failed to generate embedding: 达到最大重试次数")
        raise Exception(f"Failed to generate embedding after {self.max_retries} attempts")

    def generate_image_description(self, image_path: str, model: Optional[str] = None) -> GenerationResult:
        """生成图片描述（多模态）- 优化版"""
        start_time = time.time()
        model = model or settings.vlm_model
        
        # 图片预处理
        image_base64 = self._preprocess_image(image_path)
        if not image_base64:
            raise ValueError(f"图片预处理失败: {image_path}")

        url = f"{self.base_url}/api/generate"
        payload = {
            "model": model,
            "prompt": "用中文详细描述这张图片的内容，包括场景、物体、颜色、布局等关键信息，语言简洁",
            "images": [image_base64],
            "stream": False,
            "temperature": 0.1  # 降低随机性，提升稳定性
        }

        for attempt in range(self.max_retries):
            # 每次重试都创建新会话
            session = self._create_new_session()
            try:
                response = session.post(
                    url,
                    json=payload,
                    timeout=60  # 图生文需要更长超时时间
                )
                
                # 打印响应状态便于排查
                logger.info(f"Ollama响应状态码: {response.status_code}")
                if response.status_code != 200:
                    logger.error(f"Ollama错误响应: {response.text}")
                    response.raise_for_status()
                
                data = response.json()
                duration_ms = (time.time() - start_time) * 1000
                description = data.get("response", "").strip()
                
                if not description:
                    raise ValueError("Empty description returned")
                
                logger.info(f"Generated image description for {image_path}, "
                           f"length: {len(description)} chars, time: {duration_ms:.2f}ms")
                
                return GenerationResult(
                    text=description,
                    model=model,
                    tokens=data.get("total_duration"),
                    duration_ms=duration_ms
                )
            
            except requests.exceptions.HTTPError as e:
                logger.warning(f"HTTP错误 (attempt {attempt+1}/{self.max_retries}): {str(e)}")
                if attempt == self.max_retries - 1:
                    logger.error(f"图生文失败 {image_path}: Ollama返回错误，可能是模型问题或资源不足")
                time.sleep(self.retry_delay * (attempt + 1))
                continue
            except requests.exceptions.RequestException as e:
                logger.warning(f"请求错误 (attempt {attempt+1}/{self.max_retries}): {str(e)}")
                time.sleep(self.retry_delay * (attempt + 1))
                continue
            finally:
                session.close()

        logger.error(f"Failed to generate image description for {image_path}: 达到最大重试次数")
        raise Exception(f"Failed to generate image description after {self.max_retries} attempts")

    def generate_text(self, prompt: str, model: Optional[str] = None, 
                     system_prompt: Optional[str] = None, **kwargs) -> GenerationResult:
        """生成文本"""
        start_time = time.time()
        model = model or settings.llm_model

        url = f"{self.base_url}/api/generate"
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.7,
                "top_p": 0.9,
                **kwargs.get("options", {})
            }
        }
        
        if system_prompt:
            payload["system"] = system_prompt

        for attempt in range(self.max_retries):
            session = self._create_new_session()
            try:
                response = session.post(
                    url,
                    json=payload,
                    timeout=self.timeout
                )
                response.raise_for_status()
                data = response.json()
                
                duration_ms = (time.time() - start_time) * 1000
                response_text = data.get("response", "").strip()
                
                logger.debug(f"Generated text response, "
                            f"prompt length: {len(prompt)}, "
                            f"response length: {len(response_text)}, "
                            f"time: {duration_ms:.2f}ms")
                
                return GenerationResult(
                    text=response_text,
                    model=model,
                    tokens=data.get("total_duration"),
                    duration_ms=duration_ms
                )
            
            except requests.exceptions.RequestException as e:
                logger.warning(f"请求错误 (attempt {attempt+1}/{self.max_retries}): {str(e)}")
                time.sleep(self.retry_delay * (attempt + 1))
                continue
            
            finally:
                session.close()

        logger.error(f"Failed to generate text: 达到最大重试次数")
        raise Exception(f"Failed to generate text after {self.max_retries} attempts")

    async def batch_generate_embeddings(self, texts: List[str], model: Optional[str] = None) -> List[EmbeddingResult]:
        """批量生成文本向量（异步包装）"""
        model = model or settings.embedding_model
        results = []
        for text in texts:
            try:
                result = self.generate_embedding(text, model)
                results.append(result)
            except Exception as e:
                results.append(e)
        return results

    async def batch_generate_image_descriptions(self, image_paths: List[str], 
                                              model: Optional[str] = None) -> List[GenerationResult]:
        """批量生成图像描述（异步包装）"""
        model = model or settings.vlm_model
        results = []
        for path in image_paths:
            try:
                result = self.generate_image_description(path, model)
                results.append(result)
            except Exception as e:
                results.append(e)
        return results

    def check_model_available(self, model: str) -> bool:
        """检查模型是否可用"""
        url = f"{self.base_url}/api/tags"
        
        session = self._create_new_session()
        try:
            response = session.get(url, timeout=self.timeout)
            if response.status_code == 200:
                data = response.json()
                models = [m["name"] for m in data.get("models", [])]
                return model in models
            return False
        except Exception as e:
            logger.error(f"Failed to check model availability: {e}")
            return False
        finally:
            session.close()


# 全局Ollama服务实例
def get_ollama_service() -> OllamaService:
    """获取Ollama服务实例"""
    return OllamaService()
