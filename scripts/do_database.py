#!/usr/bin/env python3
"""
数据库操作脚本 - 调用database_function验证数据注入
注意：本文件只包含主函数，不创建新函数
"""

import sys
import asyncio
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# 导入database_function中的现有函数
from function.database_function import (
    verify_data_injection, 
    init_vector_store, 
    clear_database, 
    close_vector_store, 
    inject_sample_data,
    add_dialogue_record,
    query_dialogue_records,
    update_dialogue_record,
    delete_dialogue_record,
    get_dialogue_record_by_id,
    get_dialogue_statistics
)


async def test_query_history_crud():
    """
    测试QueryHistory表的CRUD操作
    """
    print("=" * 70)
    print("测试QueryHistory CRUD操作")
    print("=" * 70)
    
    vector_store = None
    try:
        # 连接数据库
        vector_store = await init_vector_store()
        
        # 1. 添加对话记录
        print("\n1. 添加对话记录...")
        record_id = await add_dialogue_record(
            vector_store=vector_store,
            query_text="首都师范大学的校门在哪里？",
            query_type="text",
            retrieved_document_ids=["doc-001", "doc-002"],
            retrieved_image_ids=["img-001"],
            response="校门位于校园正门主干道核心位置",
            response_time_ms=150.5
        )
        print(f"   添加成功，记录ID: {record_id}")
        
        # 2. 根据ID查询记录
        print("\n2. 根据ID查询记录...")
        record = await get_dialogue_record_by_id(vector_store, record_id)
        if record:
            print(f"   查询成功:")
            print(f"     查询文本: {record['query_text']}")
            print(f"     查询类型: {record['query_type']}")
            print(f"     响应时间: {record['response_time_ms']}ms")
        else:
            print("   查询失败: 记录不存在")
        
        # 3. 查询对话记录列表
        print("\n3. 查询对话记录列表...")
        records = await query_dialogue_records(
            vector_store=vector_store,
            limit=5,
            offset=0,
            query_type="text"
        )
        print(f"   查询到 {len(records)} 条记录")
        for i, rec in enumerate(records, 1):
            print(f"     {i}. ID: {rec['id'][:8]}..., 查询: {rec['query_text'][:30]}..., 类型: {rec['query_type']}")
        
        # 4. 更新对话记录
        print("\n4. 更新对话记录...")
        updated = await update_dialogue_record(
            vector_store=vector_store,
            record_id=record_id,
            response="校门位于校园正门主干道核心位置，紧挨着主教学楼",
            response_time_ms=180.2,
            retrieved_document_ids=["doc-001", "doc-002", "doc-003"]
        )
        if updated:
            print("   更新成功")
            # 验证更新
            updated_record = await get_dialogue_record_by_id(vector_store, record_id)
            if updated_record:
                print(f"   更新后响应: {updated_record['response']}")
                print(f"   更新后文档ID数: {len(updated_record['retrieved_document_ids'])}")
        else:
            print("   更新失败")
        
        # 5. 获取对话统计信息
        print("\n5. 获取对话统计信息...")
        stats = await get_dialogue_statistics(vector_store, days=7)
        print(f"   总记录数: {stats['total_records']}")
        print(f"   最近7天记录数: {stats['recent_records']}")
        print(f"   平均响应时间: {stats['average_response_time_ms']}ms")
        print(f"   查询类型分布:")
        for type_stat in stats['query_type_distribution']:
            print(f"     - {type_stat['type']}: {type_stat['count']}条")
        
        # 6. 删除对话记录
        print("\n6. 删除对话记录...")
        deleted = await delete_dialogue_record(vector_store, record_id)
        if deleted:
            print("   删除成功")
            # 验证删除
            deleted_record = await get_dialogue_record_by_id(vector_store, record_id)
            if deleted_record is None:
                print("   验证: 记录已成功删除")
            else:
                print("   验证失败: 记录仍然存在")
        else:
            print("   删除失败")
        
        print("\n" + "=" * 70)
        print("QueryHistory CRUD测试完成!")
        print("=" * 70)
        
    except Exception as e:
        print(f"测试失败: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if vector_store:
            await close_vector_store(vector_store)


async def main_async():
    """
    异步主函数：彻底删除数据库表结构并验证数据注入
    """
    min_doc_count = 9      # 章节数
    min_image_count = 66   # 图片数
    min_relation_count = 66 # 关联数（假设每个图片都有一个关联）
    
    print("=" * 70)
    print("开始验证数据库数据注入（彻底清理表结构）")
    print("=" * 70)
    
    print(f"\n验证参数:")
    print(f"   最小文档数要求: {min_doc_count}")
    print(f"   最小图片数要求: {min_image_count}")
    print(f"   最小关联数要求: {min_relation_count}")
    
    # 第一步：彻底删除所有数据库表（数据+表结构，替代原有的仅清空数据）
    print("\n[第一步] 彻底删除所有数据库表（数据+表结构）...")
    vector_store = None
    try:
        vector_store = await init_vector_store()
        async with vector_store.pool.acquire() as conn:
            # 级联删除所有相关表，无需关注顺序（CASCADE处理外键依赖）
            await conn.execute("DROP TABLE IF EXISTS query_history CASCADE")
            await conn.execute("DROP TABLE IF EXISTS text_image_relations CASCADE")
            await conn.execute("DROP TABLE IF EXISTS image_descriptions CASCADE")
            await conn.execute("DROP TABLE IF EXISTS documents CASCADE")
        print("   ✅ 所有表已彻底删除（数据+表结构）")
    except Exception as e:
        print(f"   ⚠️  删除表结构时出错: {e}")
    finally:
        if vector_store:
            await close_vector_store(vector_store)
    
    # 第二步：重建表结构并注入数据（核心：删表后必须先重建表，否则注入数据报错）
    print("\n[第二步] 重建表结构并注入数据...")
    vector_store = None
    try:
        vector_store = await init_vector_store()
        # 先重建表结构（调用vector_store的create_tables方法）
        await vector_store.create_tables()
        print("   ✅ 表结构重建成功")
        
        # 再注入示例数据
        injection_result = await inject_sample_data(vector_store)
        if injection_result["status"] == "success":
            print("   ✅ 数据注入成功")
            injected = injection_result["injected"]
            actual = injection_result["actual_counts"]
            print(f"     预期注入: {injected['chapters']}章节, {injected['images']}图片, {injected['relations']}关联")
            print(f"     实际统计: {actual['documents']}文档, {actual['images']}图片, {actual['relations']}关联")
        else:
            print(f"   ⚠️  数据注入失败: {injection_result.get('error', '未知错误')}")
    except Exception as e:
        print(f"   ⚠️  重建表/注入数据时出错: {e}")
    finally:
        if vector_store:
            await close_vector_store(vector_store)
    
    # 第三步：验证数据注入
    print("\n[第三步] 开始验证数据注入...")
    
    result = await verify_data_injection(
        min_doc_count=min_doc_count,
        min_image_count=min_image_count,
        min_relation_count=min_relation_count,
        chapter_limit=5,      # 限制显示5个章节
        image_limit=10,       # 限制显示10个图片
        relation_limit=5,     # 限制显示5个关联
        verbose=True          # 显示详细输出
    )
    
    print("\n" + "=" * 70)
    print("验证完成!")
    print("=" * 70)
    
    # 显示验证结果
    if result["status"] == "success":
        print(f"验证状态: {result['status']}")
        print(f"数据注入成功: {result['validation']['injection_success']}")
        
        counts = result["counts"]
        print(f"\n数据库统计:")
        print(f"   文档数: {counts['documents']}")
        print(f"   图片数: {counts['images']}")
        print(f"   关联数: {counts['relations']}")
        
        requirements = result["validation"]["min_requirements"]
        print(f"\n验证要求:")
        print(f"   最小文档数: {requirements['documents']}")
        print(f"   最小图片数: {requirements['images']}")
        print(f"   最小关联数: {requirements['relations']}")
        
        if result["validation"]["injection_success"]:
            print("\n  数据注入验证通过!")
            
            # 检查统计量是否正好是预期值（不是两倍）
            if counts['documents'] == min_doc_count and counts['images'] == min_image_count:
                print("\n  统计量正确：正好是预期值")
            elif counts['documents'] == min_doc_count * 2 or counts['images'] == min_image_count * 2:
                print("\n   注意：统计量是预期的两倍，可能存在重复数据")
            else:
                print(f"\n   注意：统计量与预期不一致")
        else:
            print("\n   数据注入验证未通过")
    else:
        print(f"验证失败: {result.get('error', '未知错误')}")
        print(f"详细错误信息请查看上方输出")
    
    print("=" * 70)
    
    # 第四步：测试QueryHistory CRUD操作
    print("\n\n" + "=" * 70)
    print("开始测试QueryHistory CRUD操作")
    print("=" * 70)
    
    await test_query_history_crud()


if __name__ == "__main__":
    """
    主函数：验证数据库数据注入
    使用单个asyncio.run调用，避免事件循环冲突
    """
    asyncio.run(main_async())
