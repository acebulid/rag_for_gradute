"""
Ollama API功能函数 - 实现对Ollama模型的请求
包括文本编码、文本生成、关键词提取等功能
"""

import os
import aiohttp
import json
import asyncio
from typing import List, Dict, Any, Optional
import logging

# 配置日志
logger = logging.getLogger(__name__)

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
        
        logger.debug(f"最终提取关键词: {final_keywords}")
        return final_keywords
    
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