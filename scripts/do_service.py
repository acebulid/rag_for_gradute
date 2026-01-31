#!/usr/bin/env python3
"""
service功能测试脚本 - 测试service_function中的chain服务和历史对话管理
"""

import sys
import asyncio
import argparse
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# 导入service_function中的现有函数
from function.service_function import (
    ChainService,
    QueryHistoryService,
    ServiceManager,
    test_service_functions
)


async def test_chain_service():
    """测试chain服务"""
    print("=" * 70)
    print("测试ChainService")
    print("=" * 70)
    
    service = ChainService()
    
    try:
        # 1. 初始化服务
        print("\n1. 初始化服务...")
        await service.initialize()
        print("   服务初始化成功")
        
        # 2. 测试文本查询
        print("\n2. 测试文本查询...")
        text_result = await service.process_query(
            query_type='text',
            query_content='首都师范大学的校门在哪里？',
            polish_response=True,
            save_history=True
        )
        
        if text_result["success"]:
            print("   文本查询成功")
            print(f"   原始结果数量: {len(text_result['original_results'])}")
            if text_result['polished_response']:
                print(f"   润色回复: {text_result['polished_response'][:100]}...")
            if text_result['dialogue_record_id']:
                print(f"   历史记录ID: {text_result['dialogue_record_id']}")
        else:
            print(f"   文本查询失败: {text_result.get('error')}")
        
        # 3. 测试图片查询
        print("\n3. 测试图片查询...")
        image_result = await service.process_query(
            query_type='image',
            query_content='data/image-1.png',
            polish_response=True,
            save_history=True
        )
        
        if image_result["success"]:
            print("   图片查询成功")
            print(f"   原始结果数量: {len(image_result['original_results'])}")
            if image_result['polished_response']:
                print(f"   润色回复: {image_result['polished_response'][:100]}...")
            if image_result['dialogue_record_id']:
                print(f"   历史记录ID: {image_result['dialogue_record_id']}")
        else:
            print(f"   图片查询失败: {image_result.get('error')}")
        
        # 4. 关闭服务
        print("\n4. 关闭服务...")
        await service.close()
        print("   服务关闭成功")
        
    except Exception as e:
        print(f"测试失败: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "=" * 70)
    print("ChainService测试完成")
    print("=" * 70)


async def test_query_history_service():
    """测试历史对话服务"""
    print("=" * 70)
    print("测试QueryHistoryService")
    print("=" * 70)
    
    service = QueryHistoryService()
    
    try:
        # 1. 初始化服务
        print("\n1. 初始化服务...")
        await service.initialize()
        print("   服务初始化成功")
        
        # 2. 添加测试记录
        print("\n2. 添加测试记录...")
        record_id = await service.add_record(
            query_text="测试查询文本",
            query_type="text",
            response="测试回复内容",
            response_time_ms=150.0,
            session_id="test_session_001"
        )
        print(f"   记录添加成功: {record_id}")
        
        # 3. 查询记录
        print("\n3. 查询记录...")
        records = await service.query_records(limit=5)
        print(f"   查询到 {len(records)} 条记录")
        for i, record in enumerate(records, 1):
            # 1. 处理 query_text 空值和切片
            query_text = record.get('query_text') or '无文本查询'
            query_text_preview = query_text[:30]
            
            # 2. 处理 response 空值和切片
            response_text = record.get('response') or '无回复内容'
            response_preview = response_text[:50] + "..." if len(response_text) > 50 else response_text
            
            # 3. 处理 session_id 空值
            session_id = record.get('session_id') or '无session_id'
            
            # 4. 安全打印
            print(f"     [{i}] 查询: {query_text_preview}..., 回复: {response_preview}, session_id: {session_id}")
        
        # 4. 更新记录
        print("\n4. 更新记录...")
        if record_id:
            success = await service.update_record(
                record_id=record_id,
                response="更新后的回复内容",
                session_id="updated_session_001"
            )
            print(f"   记录更新: {'成功' if success else '失败'}")
        
        # 5. 根据session_id查询
        print("\n5. 根据session_id查询...")
        session_records = await service.query_records(
            session_id="test_session_001",
            limit=10
        )
        print(f"   找到 {len(session_records)} 条session_id为'test_session_001'的记录")
        
        # 6. 删除记录
        print("\n6. 删除记录...")
        if record_id:
            success = await service.delete_record(record_id)
            print(f"   记录删除: {'成功' if success else '失败'}")
        
        # 7. 获取统计信息
        print("\n7. 获取统计信息...")
        stats = await service.get_statistics(days=7)
        print(f"   总记录数: {stats['total_records']}")
        print(f"   最近7天记录数: {stats['recent_records']}")
        print(f"   平均响应时间: {stats['average_response_time_ms']}ms")
        
        # 8. 关闭服务
        print("\n8. 关闭服务...")
        await service.close()
        print("   服务关闭成功")
        
    except Exception as e:
        print(f"测试失败: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "=" * 70)
    print("QueryHistoryService测试完成")
    print("=" * 70)


async def test_service_manager():
    """测试综合服务管理器"""
    print("=" * 70)
    print("测试ServiceManager")
    print("=" * 70)
    
    manager = ServiceManager()
    
    try:
        # 1. 初始化服务管理器
        print("\n1. 初始化服务管理器...")
        await manager.initialize()
        print("   服务管理器初始化成功")
        
        # 2. 测试文本查询并保存历史
        print("\n2. 测试文本查询并保存历史...")
        result = await manager.process_query_with_history(
            query_type='text',
            query_content='首都师范大学的图书馆在哪里？',
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
        
        # 3. 获取最近历史
        print("\n3. 获取最近历史...")
        recent_history = await manager.get_recent_history(limit=5)
        print(f"   最近 {len(recent_history)} 条历史记录:")
        for i, record in enumerate(recent_history, 1):
            # 1. 处理 query_text 空值和切片
            query_text = record.get('query_text') or '无文本查询'
            query_text_preview = query_text[:30]
            
            # 2. 处理 response 空值和切片
            response_text = record.get('response') or '无回复内容'
            response_preview = response_text[:50] + "..." if len(response_text) > 50 else response_text
            
            # 3. 处理 session_id 空值
            session_id = record.get('session_id') or '无session_id'
            
            # 4. 安全打印
            print(f"     [{i}] 查询: {query_text_preview}..., 回复: {response_preview}, session_id: {session_id}")
        
        # 4. 获取系统统计
        print("\n4. 获取系统统计...")
        stats = await manager.get_system_statistics(days=7)
        print(f"   总记录数: {stats['total_records']}")
        print(f"   最近7天记录数: {stats['recent_records']}")
        print(f"   平均响应时间: {stats['average_response_time_ms']}ms")
        
        # 5. 关闭服务管理器
        print("\n5. 关闭服务管理器...")
        await manager.close()
        print("   服务管理器关闭成功")
        
    except Exception as e:
        print(f"测试失败: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "=" * 70)
    print("ServiceManager测试完成")
    print("=" * 70)


async def test_session_id_functionality():
    """测试session_id功能"""
    print("=" * 70)
    print("测试session_id功能")
    print("=" * 70)
    
    service = QueryHistoryService()
    
    try:
        # 1. 初始化服务
        print("\n1. 初始化服务...")
        await service.initialize()
        print("   服务初始化成功")
        
        # 2. 添加多个session的记录
        print("\n2. 添加多个session的记录...")
        sessions = ["session_001", "session_002", "session_003"]
        
        for session_id in sessions:
            for i in range(2):  # 每个session添加2条记录
                record_id = await service.add_record(
                    query_text=f"{session_id}的查询{i+1}",
                    query_type="text",
                    response=f"{session_id}的回复{i+1}",
                    response_time_ms=100.0 + i*50,
                    session_id=session_id
                )
                print(f"   添加记录: session_id={session_id}, record_id={record_id}")
        
        # 3. 查询所有记录
        print("\n3. 查询所有记录...")
        all_records = await service.query_records(limit=10)
        print(f"   总记录数: {len(all_records)}")
        
        # 4. 按session_id查询
        print("\n4. 按session_id查询...")
        for session_id in sessions:
            session_records = await service.query_records(
                session_id=session_id,
                limit=5
            )
            print(f"   session_id={session_id}: {len(session_records)}条记录")
            for record in session_records:
                # 处理 query_text 空值和切片
                query_text = record.get('query_text') or '无文本查询'
                query_text_preview = query_text[:20]
                
                # 处理 session_id 空值
                session_id_val = record.get('session_id') or '无session_id'
                
                print(f"     查询: {query_text_preview}..., session_id: {session_id_val}")
        
        # 5. 更新记录的session_id
        print("\n5. 更新记录的session_id...")
        if all_records:
            first_record = all_records[0]
            record_id = first_record.get('id')
            if record_id:
                success = await service.update_record(
                    record_id=record_id,
                    session_id="updated_session_001"
                )
                print(f"   更新记录{record_id}的session_id: {'成功' if success else '失败'}")
                
                # 验证更新
                updated_record = await service.get_record_by_id(record_id)
                if updated_record:
                    print(f"   更新后的session_id: {updated_record.get('session_id')}")
        
        # 6. 关闭服务
        print("\n6. 关闭服务...")
        await service.close()
        print("   服务关闭成功")
        
    except Exception as e:
        print(f"测试失败: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "=" * 70)
    print("session_id功能测试完成")
    print("=" * 70)


def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='service功能测试脚本')
    parser.add_argument('--chain', action='store_true', help='测试ChainService')
    parser.add_argument('--history', action='store_true', help='测试QueryHistoryService')
    parser.add_argument('--manager', action='store_true', help='测试ServiceManager')
    parser.add_argument('--session', action='store_true', help='测试session_id功能')
    parser.add_argument('--all', action='store_true', help='测试所有功能')
    parser.add_argument('--original', action='store_true', help='运行原始测试函数')
    
    return parser.parse_args()


async def main():
    """主函数"""
    args = parse_arguments()
    
    if args.original:
        # 运行原始测试函数
        await test_service_functions()
        return
    
    if args.all or (not args.chain and not args.history and not args.manager and not args.session):
        # 默认运行所有测试
        print("开始运行所有service功能测试...")
        await test_chain_service()
        await test_query_history_service()
        await test_service_manager()
        await test_session_id_functionality()
        return
    
    # 运行指定测试
    if args.chain:
        await test_chain_service()
    
    if args.history:
        await test_query_history_service()
    
    if args.manager:
        await test_service_manager()
    
    if args.session:
        await test_session_id_functionality()


if __name__ == "__main__":
    # 运行异步主函数
    asyncio.run(main())