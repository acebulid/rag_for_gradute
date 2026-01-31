"""
service功能函数 - 实现chain服务和历史对话管理
包含chain_service和QueryHistory_service两部分
"""

import asyncio
import logging
from typing import List, Dict, Any, Optional, Union
from datetime import datetime
import uuid

# 导入必要的函数
from function.chain_function import ChainConfig, process_chain_query, print_final_results
from function.api_function import generate_text
from function.database_function import (
    PostgreSQLVectorStore,
    add_dialogue_record,
    query_dialogue_records,
    update_dialogue_record,
    delete_dialogue_record,
    get_dialogue_record_by_id,
    get_dialogue_statistics,
    init_vector_store,
    close_vector_store
)

# 配置日志
logger = logging.getLogger(__name__)


# ========== chain_service ==========
class ChainService:
    """chain服务：调用chain_function处理查询，然后润色返回结果"""
    
    def __init__(self, config: Optional[ChainConfig] = None):
        self.config = config or ChainConfig()
        self.vector_store = None
    
    async def initialize(self):
        """初始化服务"""
        try:
            self.vector_store = await init_vector_store()
            logger.info("ChainService初始化成功")
        except Exception as e:
            logger.error(f"ChainService初始化失败: {e}")
            raise
    
    async def close(self):
        """关闭服务"""
        if self.vector_store:
            await close_vector_store(self.vector_store)
            logger.info("ChainService已关闭")
    
    async def process_query(
        self,
        query_type: str,
        query_content: Union[str, Dict[str, Any]],
        polish_response: bool = True,
        save_history: bool = True
    ) -> Dict[str, Any]:
        """
        处理查询：调用chain_function + 润色 + 保存历史
        
        参数:
            query_type: 查询类型 ('text', 'image')
            query_content: 查询内容
            polish_response: 是否润色回复
            save_history: 是否保存到历史记录
        
        返回:
            Dict: 处理结果
        """
        result = {
            "success": False,
            "query_type": query_type,
            "original_results": [],
            "polished_response": None,
            "dialogue_record_id": None,
            "error": None
        }
        
        try:
            # 1. 调用chain_function处理查询
            logger.info(f"开始处理{query_type}查询")
            
            chain_result = await process_chain_query(
                query_type=query_type,
                query_content=query_content,
                config=self.config
            )
            
            if not chain_result["success"]:
                result["error"] = chain_result.get("error", "chain处理失败")
                return result
            
            # 2. 保存原始结果
            result["original_results"] = chain_result["search_results"]
            
            # 3. 润色回复（如果需要）
            if polish_response and chain_result["search_results"]:
                polished = await self._polish_response(chain_result["search_results"])
                result["polished_response"] = polished
            
            # 4. 保存到历史记录（如果需要）
            if save_history and self.vector_store:
                record_id = await self._save_to_history(
                    query_type=query_type,
                    query_content=query_content,
                    search_results=chain_result["search_results"],
                    polished_response=result["polished_response"]
                )
                result["dialogue_record_id"] = record_id
            
            result["success"] = True
            logger.info(f"{query_type}查询处理成功")
            
        except Exception as e:
            result["error"] = str(e)
            logger.error(f"处理查询失败: {e}")
        
        return result
    
    async def _polish_response(self, search_results: List[Dict[str, Any]]) -> str:
        """
        润色回复：使用LLM优化检索结果
        
        参数:
            search_results: 检索结果列表
        
        返回:
            str: 润色后的回复
        """
        if not search_results:
            return "抱歉，没有找到相关信息。"
        
        try:
            # 构建提示词
            context = "\n\n".join([
                f"【来源{i+1}】{result['content_preview']}"
                for i, result in enumerate(search_results)
            ])
            
            prompt = f"""请根据以下检索到的信息，生成一个自然、流畅的回复：
            
检索到的信息：
{context}

请生成一个友好的校园导览回复，包含以下要点：
1. 使用亲切的语气（如"各位游客朋友"）
2. 简要介绍相关地点
3. 提供实用信息
4. 保持回复简洁（200字以内）

回复："""
            
            # 调用LLM生成回复
            response = await generate_text(prompt)
            
            if not response:
                # 如果LLM调用失败，使用默认回复
                response = self._generate_default_response(search_results)
            
            return response.strip()
            
        except Exception as e:
            logger.error(f"润色回复失败: {e}")
            return self._generate_default_response(search_results)
    
    def _generate_default_response(self, search_results: List[Dict[str, Any]]) -> str:
        """生成默认回复"""
        if not search_results:
            return "抱歉，没有找到相关信息。"
        
        # 使用第一个结果生成简单回复
        first_result = search_results[0]
        content = first_result.get('content_preview', '')
        
        # 提取前100个字符作为回复
        if len(content) > 100:
            content = content[:100] + "..."
        
        return f"根据检索到的信息：{content}"
    
    async def _save_to_history(
        self,
        query_type: str,
        query_content: Union[str, Dict[str, Any]],
        search_results: List[Dict[str, Any]],
        polished_response: Optional[str] = None,
        session_id: Optional[str] = None
    ) -> str:
        """
        保存查询到历史记录
        
        返回:
            str: 历史记录ID
        """
        try:
            # 提取查询文本或图片路径
            if query_type == 'text':
                query_text = query_content if isinstance(query_content, str) else query_content.get('question', '')
                query_image_path = None
            else:  # image
                query_text = None
                query_image_path = query_content if isinstance(query_content, str) else query_content.get('image_path', '')
            
            # 提取检索到的文档ID（如果有）
            retrieved_document_ids = []
            for result in search_results:
                if 'id' in result:
                    retrieved_document_ids.append(result['id'])
            
            # 计算响应时间（简化处理）
            response_time_ms = 100.0  # 默认值，实际应该计算
            
            # 保存到历史记录
            record_id = await add_dialogue_record(
                vector_store=self.vector_store,
                query_text=query_text,
                query_image_path=query_image_path,
                query_type=query_type,
                retrieved_document_ids=retrieved_document_ids,
                retrieved_image_ids=[],  # 图片查询暂时不保存图片ID
                response=polished_response,
                response_time_ms=response_time_ms,
                session_id=session_id
            )
            
            logger.debug(f"查询历史记录已保存: {record_id}")
            return record_id
            
        except Exception as e:
            logger.error(f"保存查询历史失败: {e}")
            return ""


# ========== QueryHistory_service ==========
class QueryHistoryService:
    """历史对话服务：封装database_function中的历史记录操作"""
    
    def __init__(self):
        self.vector_store = None
    
    async def initialize(self):
        """初始化服务"""
        try:
            self.vector_store = await init_vector_store()
            logger.info("QueryHistoryService初始化成功")
        except Exception as e:
            logger.error(f"QueryHistoryService初始化失败: {e}")
            raise
    
    async def close(self):
        """关闭服务"""
        if self.vector_store:
            await close_vector_store(self.vector_store)
            logger.info("QueryHistoryService已关闭")
    
    async def add_record(
        self,
        query_text: Optional[str] = None,
        query_image_path: Optional[str] = None,
        query_type: str = "text",
        retrieved_document_ids: Optional[List[str]] = None,
        retrieved_image_ids: Optional[List[str]] = None,
        response: Optional[str] = None,
        response_time_ms: Optional[float] = None,
        session_id: Optional[str] = None
    ) -> str:
        """
        添加对话记录
        
        参数:
            query_text: 查询文本
            query_image_path: 查询图片路径
            query_type: 查询类型
            retrieved_document_ids: 检索到的文档ID列表
            retrieved_image_ids: 检索到的图片ID列表
            response: 系统回复
            response_time_ms: 响应时间（毫秒）
            session_id: 会话ID，用于关联多轮对话
        
        返回:
            str: 新记录的ID
        """
        try:
            record_id = await add_dialogue_record(
                vector_store=self.vector_store,
                query_text=query_text,
                query_image_path=query_image_path,
                query_type=query_type,
                retrieved_document_ids=retrieved_document_ids,
                retrieved_image_ids=retrieved_image_ids,
                response=response,
                response_time_ms=response_time_ms,
                session_id=session_id
            )
            logger.info(f"对话记录添加成功: {record_id}")
            return record_id
        except Exception as e:
            logger.error(f"添加对话记录失败: {e}")
            raise
    
    async def query_records(
        self,
        limit: int = 10,
        offset: int = 0,
        query_type: Optional[str] = None,
        session_id: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        查询对话记录
        
        参数:
            limit: 返回记录数量限制
            offset: 偏移量
            query_type: 查询类型过滤
            session_id: 会话ID过滤
            start_date: 开始日期
            end_date: 结束日期
        
        返回:
            List[Dict]: 对话记录列表
        """
        try:
            records = await query_dialogue_records(
                vector_store=self.vector_store,
                limit=limit,
                offset=offset,
                query_type=query_type,
                session_id=session_id,
                start_date=start_date,
                end_date=end_date
            )
            logger.debug(f"查询到 {len(records)} 条对话记录")
            return records
        except Exception as e:
            logger.error(f"查询对话记录失败: {e}")
            raise
    
    async def update_record(
        self,
        record_id: str,
        response: Optional[str] = None,
        response_time_ms: Optional[float] = None,
        retrieved_document_ids: Optional[List[str]] = None,
        retrieved_image_ids: Optional[List[str]] = None,
        session_id: Optional[str] = None
    ) -> bool:
        """
        更新对话记录
        
        参数:
            record_id: 记录ID
            response: 更新的系统回复
            response_time_ms: 更新的响应时间
            retrieved_document_ids: 更新的检索文档ID列表
            retrieved_image_ids: 更新的检索图片ID列表
            session_id: 更新的会话ID
        
        返回:
            bool: 更新是否成功
        """
        try:
            success = await update_dialogue_record(
                vector_store=self.vector_store,
                record_id=record_id,
                response=response,
                response_time_ms=response_time_ms,
                retrieved_document_ids=retrieved_document_ids,
                retrieved_image_ids=retrieved_image_ids,
                session_id=session_id
            )
            
            if success:
                logger.info(f"对话记录更新成功: {record_id}")
            else:
                logger.warning(f"对话记录更新失败或不存在: {record_id}")
            
            return success
        except Exception as e:
            logger.error(f"更新对话记录失败: {e}")
            raise
    
    async def delete_record(self, record_id: str) -> bool:
        """
        删除对话记录
        
        参数:
            record_id: 记录ID
        
        返回:
            bool: 删除是否成功
        """
        try:
            success = await delete_dialogue_record(
                vector_store=self.vector_store,
                record_id=record_id
            )
            
            if success:
                logger.info(f"对话记录删除成功: {record_id}")
            else:
                logger.warning(f"对话记录删除失败或不存在: {record_id}")
            
            return success
        except Exception as e:
            logger.error(f"删除对话记录失败: {e}")
            raise
    
    async def get_record_by_id(self, record_id: str) -> Optional[Dict[str, Any]]:
        """
        根据ID获取单个对话记录
        
        参数:
            record_id: 记录ID
        
        返回:
            Optional[Dict]: 对话记录，如果不存在则返回None
        """
        try:
            record = await get_dialogue_record_by_id(self.vector_store, record_id)
            
            if record:
                logger.debug(f"获取对话记录成功: {record_id}")
            else:
                logger.debug(f"对话记录不存在: {record_id}")
            
            return record
        except Exception as e:
            logger.error(f"获取对话记录失败: {e}")
            raise
    
    async def get_statistics(self, days: int = 7) -> Dict[str, Any]:
        """
        获取对话统计信息
        
        参数:
            days: 统计最近多少天的数据
        
        返回:
            Dict: 统计信息
        """
        try:
            stats = await get_dialogue_statistics(self.vector_store, days)
            logger.debug(f"获取对话统计信息成功，统计周期: {days}天")
            return stats
        except Exception as e:
            logger.error(f"获取对话统计信息失败: {e}")
            raise
    
    async def search_records_by_keyword(
        self,
        keyword: str,
        limit: int = 10,
        offset: int = 0
    ) -> List[Dict[str, Any]]:
        """
        根据关键词搜索对话记录
        
        参数:
            keyword: 搜索关键词
            limit: 返回记录数量限制
            offset: 偏移量
        
        返回:
            List[Dict]: 匹配的对话记录列表
        """
        try:
            # 先获取所有记录，然后在内存中过滤（简单实现）
            all_records = await self.query_records(limit=100, offset=0)
            
            filtered_records = []
            for record in all_records:
                query_text = record.get('query_text', '')
                response = record.get('response', '')
                
                if keyword.lower() in query_text.lower() or keyword.lower() in response.lower():
                    filtered_records.append(record)
            
            # 应用分页
            start_idx = offset
            end_idx = offset + limit
            paginated_records = filtered_records[start_idx:end_idx]
            
            logger.debug(f"关键词搜索找到 {len(filtered_records)} 条记录，返回 {len(paginated_records)} 条")
            return paginated_records
            
        except Exception as e:
            logger.error(f"关键词搜索失败: {e}")
            raise


# ========== 综合服务 ==========
class ServiceManager:
    """综合服务管理器：整合chain服务和历史对话服务"""
    
    def __init__(self):
        self.chain_service = ChainService()
        self.history_service = QueryHistoryService()
        self.initialized = False
    
    async def initialize(self):
        """初始化所有服务"""
        try:
            await self.chain_service.initialize()
            await self.history_service.initialize()
            self.initialized = True
            logger.info("ServiceManager初始化成功")
        except Exception as e:
            logger.error(f"ServiceManager初始化失败: {e}")
            raise
    
    async def close(self):
        """关闭所有服务"""
        try:
            await self.chain_service.close()
            await self.history_service.close()
            self.initialized = False
            logger.info("ServiceManager已关闭")
        except Exception as e:
            logger.error(f"ServiceManager关闭失败: {e}")
    
    async def process_query_with_history(
        self,
        query_type: str,
        query_content: Union[str, Dict[str, Any]],
        polish_response: bool = True
    ) -> Dict[str, Any]:
        """
        处理查询并保存历史
        
        参数:
            query_type: 查询类型
            query_content: 查询内容
            polish_response: 是否润色回复
        
        返回:
            Dict: 处理结果
        """
        if not self.initialized:
            raise RuntimeError("ServiceManager未初始化，请先调用initialize()")
        
        return await self.chain_service.process_query(
            query_type=query_type,
            query_content=query_content,
            polish_response=polish_response,
            save_history=True
        )
    
    async def get_recent_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """获取最近的对话历史"""
        if not self.initialized:
            raise RuntimeError("ServiceManager未初始化，请先调用initialize()")
        
        return await self.history_service.query_records(limit=limit)
    
    async def get_system_statistics(self, days: int = 7) -> Dict[str, Any]:
        """获取系统统计信息"""
        if not self.initialized:
            raise RuntimeError("ServiceManager未初始化，请先调用initialize()")
        
        return await self.history_service.get_statistics(days)


# ========== 测试函数 ==========
async def test_service_functions():
    """测试service功能"""
    print("=" * 70)
    print("测试service功能")
    print("=" * 70)
    
    service_manager = ServiceManager()
    
    try:
        # 1. 初始化服务
        print("\n1. 初始化服务...")
        await service_manager.initialize()
        print("   服务初始化成功")
        
        # 2. 测试文本查询
        print("\n2. 测试文本查询...")
        result = await service_manager.process_query_with_history(
            query_type='text',
            query_content='首都师范大学的校门在哪里？',
            polish_response=True
        )
        
        if result["success"]:
            print("   查询成功")
            print(f"   原始结果数量: {len(result['original_results'])}")
            if result['polished_response']:
                print(f"   润色回复: {result['polished_response'][:100]}...")
            if result['dialogue_record_id']:
                print(f"   历史记录ID: {result['dialogue_record_id']}")
        else:
            print(f"   查询失败: {result.get('error')}")
        
        # 3. 测试历史记录查询
        print("\n3. 测试历史记录查询...")
        recent_history = await service_manager.get_recent_history(limit=5)
        print(f"   最近 {len(recent_history)} 条历史记录:")
        for i, record in enumerate(recent_history, 1):
            query_text = record.get('query_text', '无文本查询')
            response_preview = record.get('response', '无回复')[:50] + "..." if record.get('response') else '无回复'
            print(f"     {i}. 查询: {query_text[:30]}..., 回复: {response_preview}")
        
        # 4. 测试系统统计
        print("\n4. 测试系统统计...")
        stats = await service_manager.get_system_statistics(days=7)
        print(f"   总记录数: {stats['total_records']}")
        print(f"   最近7天记录数: {stats['recent_records']}")
        print(f"   平均响应时间: {stats['average_response_time_ms']}ms")
        print(f"   查询类型分布:")
        for type_stat in stats['query_type_distribution']:
            print(f"     - {type_stat['type']}: {type_stat['count']}条")
        
        # 5. 关闭服务
        print("\n5. 关闭服务...")
        await service_manager.close()
        print("   服务关闭成功")
        
    except Exception as e:
        print(f"测试失败: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "=" * 70)
    print("service功能测试完成")
    print("=" * 70)


if __name__ == "__main__":
    # 运行测试
    asyncio.run(test_service_functions())
