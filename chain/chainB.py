"""
chainB: 文本关键词提取链
输入提问，使用qwen3:8b模型提取关键词（提供建议选项）
"""

import asyncio
from typing import List, Dict, Any
from function.api_function import OllamaService, extract_keywords


class ChainB:
    """chainB: 文本关键词提取链"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
    
    async def process_question(self, question: str) -> List[str]:
        """
        处理问题，提取关键词
        
        参数:
            question: 用户提问
        
        返回:
            List[str]: 提取的关键词列表
        """
        print(f"[chainB] 开始处理问题: {question}")
        
        try:
            # 使用api_function中的extract_keywords函数
            # 该函数已经包含参考关键词选项
            keywords = await extract_keywords(question)
            
            print(f"[chainB] 提取到关键词: {keywords}")
            return keywords
            
        except Exception as e:
            print(f"[chainB] 处理失败: {e}")
            return []
    
    async def process_with_debug(self, question: str) -> Dict[str, Any]:
        """
        处理问题并返回调试信息
        
        参数:
            question: 用户提问
        
        返回:
            Dict[str, Any]: 包含关键词和调试信息的结果
        """
        print(f"[chainB调试] 开始处理问题: {question}")
        
        result = {
            "success": False,
            "question": question,
            "keywords": [],
            "error": None
        }
        
        try:
            # 使用api_function中的extract_keywords函数
            keywords = await extract_keywords(question)
            
            result["keywords"] = keywords
            result["success"] = True
            
            print(f"[chainB调试] 提取到关键词: {keywords}")
            
        except Exception as e:
            result["error"] = str(e)
            print(f"[chainB调试] 处理失败: {e}")
        
        return result


if __name__ == "__main__":
    print("chainB模块 - 文本关键词提取链")
    print("使用方法: 作为模块导入使用，不直接运行")
    print("示例: from chain.chainB import ChainB")
