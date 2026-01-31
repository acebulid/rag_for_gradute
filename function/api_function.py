"""
Ollama API功能函数 - 实现对Ollama模型的请求
包括文本编码、文本生成、关键词提取等功能
"""

import os
import aiohttp
import json
import asyncio
import threading
from typing import List, Dict, Any, Optional
import logging
import numpy as np
import torch
from PIL import Image
from transformers import AutoModel, AutoProcessor  # 关键：替换为AutoProcessor

# 配置日志
logger = logging.getLogger(__name__)

# ========== 全局CLIP模型配置（优化：支持图文双输入） ==========
_CLIP_PROCESSOR = None  # 改名：从_CLIP_IMAGE_PROCESSOR改为_CLIP_PROCESSOR（适配图文）
_CLIP_MODEL = None
_CLIP_MODEL_NAME = "openai/clip-vit-base-patch32"
_TARGET_EMBEDDING_DIM = 512
_CLIP_MODEL_LOCK = threading.Lock()

def _init_clip_model():
    """初始化CLIP模型（单例，支持图文双输入，避免重复加载）"""
    global _CLIP_PROCESSOR, _CLIP_MODEL
    if _CLIP_PROCESSOR is None or _CLIP_MODEL is None:
        with _CLIP_MODEL_LOCK:
            if _CLIP_PROCESSOR is None or _CLIP_MODEL is None:
                try:
                    logger.info(f"正在加载CLIP模型：{_CLIP_MODEL_NAME}")
                    # 配置国内镜像，提升下载速度
                    import os
                    if not os.getenv("HF_ENDPOINT"):
                        os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
                    # 关键：使用AutoProcessor，支持同时预处理图片和文本
                    _CLIP_PROCESSOR = AutoProcessor.from_pretrained(_CLIP_MODEL_NAME)
                    _CLIP_MODEL = AutoModel.from_pretrained(_CLIP_MODEL_NAME)
                    _CLIP_MODEL.eval()
                    logger.info("CLIP模型加载完成（支持图文双输入）")
                except Exception as e:
                    logger.error(f"加载CLIP模型失败：{e}")
                    raise e

def _extract_clip_image_feature(image_path: str, text_label: str = "default image") -> np.ndarray:
    """
    同步提取CLIP图片特征（优化版：补全文本标签输入，解决input_ids报错，提升向量质量）
    
    参数:
        image_path: 图片文件路径
        text_label: 图片对应的文本标签（如章节名"校门"）
    
    返回:
        np.ndarray: 512维CLIP图片特征向量
    """
    try:
        # 1. 初始化CLIP模型
        _init_clip_model()
        
        # 2. 读取并预处理图片（转为RGB，避免格式错误）
        image = Image.open(image_path).convert("RGB")
        
        # 3. 关键：用AutoProcessor同时预处理图片和文本标签（解决input_ids报错）
        # CLIP会自动生成图片的pixel_values和文本的input_ids/attention_mask
        inputs = _CLIP_PROCESSOR(
            images=image,
            text=text_label,  # 传入图片对应的章节标签，补全文本输入
            return_tensors="pt",
            padding=True,
            truncation=True
        )
        
        # 4. 提取图文融合特征（无梯度计算，提升速度）
        with torch.no_grad():
            outputs = _CLIP_MODEL(** inputs)
            # 提取图像特征（image_embeds：纯图像特征；text_embeds：纯文本特征；可按需选择）
            # 方案A：用纯图像特征（保持原有逻辑，仅解决报错）
            image_feature = outputs.image_embeds.squeeze().numpy()
            # 方案B：用图文融合特征（推荐，语义更精准，取两者均值）
            # text_feature = outputs.text_embeds.squeeze().numpy()
            # image_feature = (image_feature + text_feature) / 2
        
        # 5. 归一化（对齐向量格式，避免数值溢出）
        image_feature = image_feature / np.linalg.norm(image_feature)
        
        return image_feature.astype(np.float32)
    except FileNotFoundError:
        logger.error(f"提取CLIP图片特征失败：图片文件不存在 {image_path}")
        return np.array([])
    except Exception as e:
        logger.error(f"提取CLIP图片特征失败：{e}")
        return np.array([])

# ========== 配置 ==========
class OllamaConfig:
    """Ollama配置"""
    
    def __init__(self):
        self.base_url = os.getenv('OLLAMA_BASE_URL', 'http://localhost:11434')
        self.embedding_model = os.getenv('EMBEDDING_MODEL', 'bge-m3:latest')
        self.llm_model = os.getenv('LLM_MODEL', 'qwen3:8b')
        self.timeout = 30  # 请求超时时间（秒）


# ========== Ollama服务类 ==========
class OllamaService:
    """Ollama服务，实现真正的API调用"""
    
    def __init__(self, config: Optional[OllamaConfig] = None):
        self.config = config or OllamaConfig()
        self.session = None
    
    async def __aenter__(self):
        """异步上下文管理器入口"""
        self.session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """异步上下文管理器出口"""
        if self.session:
            await self.session.close()
    
    async def generate_embedding(self, text: str) -> List[float]:
        """
        生成文本嵌入向量，调用Ollama API
        
        参数:
            text: 输入文本
        
        返回:
            List[float]: 嵌入向量
        """
        url = f"{self.config.base_url}/api/embeddings"
        payload = {
            "model": self.config.embedding_model,
            "prompt": text
        }
        
        try:
            async with self.session.post(url, json=payload, timeout=self.config.timeout) as response:
                if response.status == 200:
                    result = await response.json()
                    embedding = result.get("embedding", [])
                    logger.debug(f"生成嵌入向量成功，维度: {len(embedding)}")
                    return embedding
                else:
                    error_text = await response.text()
                    logger.error(f"生成嵌入失败: {response.status}, {error_text}")
                    return []
        except Exception as e:
            logger.error(f"生成嵌入异常: {e}")
            return []
    
    async def generate_image_embedding(self, image_path: str, model_name: str = "bge-m3", text_label: str = "default image") -> List[float]:
        """
        生成图像嵌入向量（优化版：传入文本标签，图文双输入生成优质向量）
        
        参数:
            image_path: 图像文件路径
            model_name: 模型名称（保留参数，保持接口兼容）
            text_label: 图片对应的文本标签（新增，用于补全CLIP文本输入）
        
        返回:
            List[float]: 512维图像嵌入向量（CLIP原始维度）
        """
        try:
            # 1. 同步提取CLIP图文特征（包装为异步，避免阻塞事件循环，传入文本标签）
            clip_feature = await asyncio.get_event_loop().run_in_executor(
                None, _extract_clip_image_feature, image_path, text_label  # 新增传入text_label
            )
            
            if clip_feature.size == 0:
                logger.error("CLIP图片特征提取结果为空")
                return []
            
            # 2. 直接返回512维向量（不进行维度扩展）
            # 归一化，保证向量规范性
            clip_feature = clip_feature / np.linalg.norm(clip_feature)
            
            # 3. 转换为List[float]返回，符合接口要求
            result_embedding = clip_feature.tolist()
            logger.debug(f"生成512维图片向量成功（标签：{text_label}），维度：{len(result_embedding)}")
            
            return result_embedding
        except Exception as e:
            logger.error(f"生成图片嵌入向量异常: {e}")
            return []
    
    async def generate_text(self, prompt: str, model: Optional[str] = None) -> str:
        """
        生成文本回复，调用Ollama API
        
        参数:
            prompt: 输入提示
            model: 模型名称，默认使用配置中的LLM模型
        
        返回:
            str: 生成的文本回复
        """
        url = f"{self.config.base_url}/api/generate"
        payload = {
            "model": model or self.config.llm_model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.7,
                "top_p": 0.9,
                "max_tokens": 1000
            }
        }
        
        try:
            async with self.session.post(url, json=payload, timeout=self.config.timeout) as response:
                if response.status == 200:
                    result = await response.json()
                    response_text = result.get("response", "")
                    logger.debug(f"生成文本成功，长度: {len(response_text)}")
                    return response_text.strip()
                else:
                    error_text = await response.text()
                    logger.error(f"生成文本失败: {response.status}, {error_text}")
                    return ""
        except Exception as e:
            logger.error(f"生成文本异常: {e}")
            return ""
    
    async def extract_keywords(self, question: str) -> List[str]:
        """
        从问题中提取关键词，调用LLM
        使用参考关键词提高准确性
        
        参数:
            question: 用户问题
        
        返回:
            List[str]: 关键词列表
        """
        import re
        
        # 参考关键词列表（校园导览相关）
        reference_keywords = [
            "首都师范大学", "校门", "校训石", "主楼", "图书馆", 
            "理科楼", "学生活动中心", "操场", "食堂", "大成广场"
        ]
        
        prompt = f"""请从以下问题中提取最重要的关键词，用逗号分隔。
        请优先选择与以下参考词相关的关键词：
        参考词：{', '.join(reference_keywords)}
        
        注意：如果没有相关关键词，可以只提取1-2个关键词，或者不提取。

        问题：{question}

        关键词："""
        
        response = None
        try:
            response = await self.generate_text(prompt)
            logger.debug(f"LLM返回关键词响应: {response}")
        except Exception as e:
            logger.error(f"调用LLM提取关键词失败: {e}")
        
        # 解析响应，提取关键词
        keywords = []
        if response:
            # 步骤1：清洗文本 - 移除前缀、换行、多余空格
            clean_response = re.sub(r'^关键词：\s*', '', response.strip(), flags=re.IGNORECASE)
            clean_response = re.sub(r'\s+', ' ', clean_response)
            
            # 步骤2：多分隔符分割（逗号、顿号、空格）
            keywords = re.split(r'[,，、\s]+', clean_response)
            
            # 步骤3：过滤处理（空值、单个字符、去重）
            keywords = [kw.strip() for kw in keywords if kw.strip() and len(kw.strip()) > 1]
            keywords = list(dict.fromkeys(keywords))  # 保持顺序去重
            
            # 步骤4：参考词匹配排序，优先保留相关关键词
            if reference_keywords:
                ref_set = set([kw.lower() for kw in reference_keywords])
                
                def _keyword_score(kw):
                    kw_lower = kw.lower()
                    if kw_lower in ref_set:
                        return 2
                    for ref_kw in ref_set:
                        if ref_kw in kw_lower or kw_lower in ref_kw:
                            return 1
                    return 0
                
                keywords = sorted(keywords, key=lambda x: _keyword_score(x), reverse=True)
        
        # 如果提取失败，使用优化后的简单分词兜底
        if not keywords:
            logger.debug("LLM提取关键词失败，使用兜底分词逻辑")
            stop_words = {'哪里', '什么', '如何', '为什么', '请问', '可以', 
                          '怎么', '是否', '能否', '我', '你', '他', '的', '了', 
                          '在', '和', '有', '是', '这', '那', '此'}
            
            # 按标点分割句子，再提取有效词汇
            sentences = re.split(r'[。，！？；：()\[\]{}"\'"《》]', question)
            for sent in sentences:
                sent = sent.strip()
                if not sent:
                    continue
                # 提取中文字符、字母、数字，过滤其他符号
                for word in re.findall(r'[\u4e00-\u9fa5a-zA-Z0-9]+', sent):
                    if len(word) > 1 and word not in stop_words:
                        keywords.append(word)
            
            # 去重
            keywords = list(dict.fromkeys(keywords))
        
        # 限制最多5个关键词，增加截断日志
        final_keywords = keywords[:5]
        if len(keywords) > 5:
            logger.debug(f"关键词数量超过5个，截断前5个，原列表：{keywords}")
        
        # 关键修改：过滤掉"首都师范大学"这个词
        filtered_keywords = [kw for kw in final_keywords if kw != "首都师范大学"]
        
        logger.debug(f"过滤前关键词: {final_keywords}")
        logger.debug(f"过滤后关键词: {filtered_keywords}")
        return filtered_keywords
    
    async def batch_generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        批量生成文本嵌入向量
        
        参数:
            texts: 文本列表
        
        返回:
            List[List[float]]: 嵌入向量列表
        """
        embeddings = []
        for text in texts:
            embedding = await self.generate_embedding(text)
            if embedding:
                embeddings.append(embedding)
            else:
                embeddings.append([])  # 失败时返回空列表
        
        return embeddings
    
    async def check_models_available(self) -> Dict[str, bool]:
        """
        检查所需模型是否可用
        
        返回:
            Dict[str, bool]: 模型可用性状态
        """
        url = f"{self.config.base_url}/api/tags"
        
        try:
            async with self.session.get(url, timeout=10) as response:
                if response.status == 200:
                    result = await response.json()
                    models = [model["name"] for model in result.get("models", [])]
                    
                    return {
                        "embedding_model": self.config.embedding_model in models,
                        "llm_model": self.config.llm_model in models,
                        "available_models": models
                    }
                else:
                    return {
                        "embedding_model": False,
                        "llm_model": False,
                        "available_models": []
                    }
        except Exception as e:
            logger.error(f"检查模型可用性异常: {e}")
            return {
                "embedding_model": False,
                "llm_model": False,
                "available_models": []
            }


# ========== 功能函数（简化接口） ==========
async def generate_embedding(text: str) -> List[float]:
    """
    生成文本嵌入向量（简化接口）
    
    参数:
        text: 输入文本
    
    返回:
        List[float]: 嵌入向量
    """
    async with OllamaService() as service:
        return await service.generate_embedding(text)


async def generate_image_embedding(image_path: str, model_name: str = "bge-m3", text_label: str = "default image") -> List[float]:
    """
    生成图像嵌入向量（简化接口）
    
    参数:
        image_path: 图像文件路径
        model_name: 模型名称，默认使用bge-m3
        text_label: 图片对应的文本标签（新增，用于补全CLIP文本输入）
    
    返回:
        List[float]: 图像嵌入向量
    """
    async with OllamaService() as service:
        return await service.generate_image_embedding(image_path, model_name, text_label)


async def generate_text(prompt: str, model: Optional[str] = None) -> str:
    """
    生成文本回复（简化接口）
    
    参数:
        prompt: 输入提示
        model: 模型名称
    
    返回:
        str: 生成的文本回复
    """
    async with OllamaService() as service:
        return await service.generate_text(prompt, model)


async def extract_keywords(question: str) -> List[str]:
    """
    从问题中提取关键词（简化接口）
    
    参数:
        question: 用户问题
    
    返回:
        List[str]: 关键词列表
    """
    async with OllamaService() as service:
        return await service.extract_keywords(question)


async def batch_generate_embeddings(texts: List[str]) -> List[List[float]]:
    """
    批量生成文本嵌入向量（简化接口）
    
    参数:
        texts: 文本列表
    
    返回:
        List[List[float]]: 嵌入向量列表
    """
    async with OllamaService() as service:
        return await service.batch_generate_embeddings(texts)


async def check_ollama_health() -> Dict[str, Any]:
    """
    检查Ollama服务健康状态
    
    返回:
        Dict[str, Any]: 健康状态信息
    """
    config = OllamaConfig()
    
    try:
        async with OllamaService(config) as service:
            # 检查服务是否可达
            async with service.session.get(f"{config.base_url}/api/tags", timeout=5) as response:
                service_reachable = response.status == 200
            
            # 检查模型可用性
            models_status = await service.check_models_available()
            
            return {
                "service_reachable": service_reachable,
                "models_status": models_status,
                "config": {
                    "base_url": config.base_url,
                    "embedding_model": config.embedding_model,
                    "llm_model": config.llm_model
                }
            }
    except Exception as e:
        return {
            "service_reachable": False,
            "models_status": {
                "embedding_model": False,
                "llm_model": False,
                "available_models": []
            },
            "error": str(e),
            "config": {
                "base_url": config.base_url,
                "embedding_model": config.embedding_model,
                "llm_model": config.llm_model
            }
        }


# ========== 测试函数 ==========
async def test_api_functions():
    """测试API功能"""
    print("=" * 70)
    print("测试Ollama API功能")
    print("=" * 70)
    
    # 1. 检查服务健康状态
    print("\n1. 检查Ollama服务健康状态...")
    health = await check_ollama_health()
    
    if health["service_reachable"]:
        print("   ✅ Ollama服务可达")
        models_status = health["models_status"]
        print(f"   嵌入模型 ({health['config']['embedding_model']}): {'可用' if models_status['embedding_model'] else '不可用'}")
        print(f"   LLM模型 ({health['config']['llm_model']}): {'可用' if models_status['llm_model'] else '不可用'}")
    else:
        print("   ❌ Ollama服务不可达")
        print(f"   请确保Ollama服务正在运行: {health['config']['base_url']}")
        return
    
    # 2. 测试文本嵌入
    print("\n2. 测试文本嵌入...")
    test_text = "首都师范大学"
    embedding = await generate_embedding(test_text)
    
    if embedding:
        print(f"   ✅ 嵌入成功，维度: {len(embedding)}")
        print(f"   前5个值: {embedding[:5]}")
    else:
        print("   ❌ 嵌入失败")
    
    # 3. 测试关键词提取
    print("\n3. 测试关键词提取...")
    test_question = "首都师范大学的校门在哪里？"
    keywords = await extract_keywords(test_question)
    
    if keywords:
        print(f"   ✅ 提取到关键词: {keywords}")
    else:
        print("   ❌ 关键词提取失败")
    
    # 4. 测试文本生成
    print("\n4. 测试文本生成...")
    test_prompt = "请用一句话介绍首都师范大学。"
    response = await generate_text(test_prompt)
    
    if response:
        print(f"   ✅ 文本生成成功")
        print(f"   响应: {response[:100]}...")
    else:
        print("   ❌ 文本生成失败")
    
    print("\n" + "=" * 70)
    print("API功能测试完成")
    print("=" * 70)


if __name__ == "__main__":
    # 运行测试
    asyncio.run(test_api_functions())