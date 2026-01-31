# ui/core/service_caller.py
from pathlib import Path
import sys

# 添加项目根目录到Python路径，确保能导入Service
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from function.service_function import ServiceManager

# 全局Service实例（避免重复初始化）
_service_manager = None

async def init_service():
    """初始化Service（UI启动时调用一次）"""
    global _service_manager
    if not _service_manager:
        _service_manager = ServiceManager()
        await _service_manager.initialize()

async def process_user_query(query_type, query_content):
    """处理用户查询（核心：转发给Service层，返回结果给UI）"""
    if not _service_manager:
        return {"success": False, "error": "Service未初始化，请刷新页面"}
    
    # 直接调用Service的核心方法，返回原始结果给UI
    return await _service_manager.process_query_with_history(
        query_type=query_type,
        query_content=query_content,
        polish_response=True
    )

async def close_service():
    """关闭Service（UI退出时调用，释放资源）"""
    global _service_manager
    if _service_manager:
        await _service_manager.close()
        _service_manager = None