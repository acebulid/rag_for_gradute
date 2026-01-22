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
from function.database_function import verify_data_injection, init_vector_store, clear_database, close_vector_store, inject_sample_data


async def main_async():
    """
    异步主函数：清空数据库并验证数据注入
    """
    min_doc_count = 9      # 章节数
    min_image_count = 66   # 图片数
    min_relation_count = 66 # 关联数（假设每个图片都有一个关联）
    
    print("=" * 70)
    print("开始验证数据库数据注入")
    print("=" * 70)
    
    print(f"\n验证参数:")
    print(f"   最小文档数要求: {min_doc_count}")
    print(f"   最小图片数要求: {min_image_count}")
    print(f"   最小关联数要求: {min_relation_count}")
    
    # 第一步：清空数据库
    print("\n[第一步] 清空数据库...")
    vector_store = None
    try:
        vector_store = await init_vector_store()
        success = await clear_database(vector_store)
        
        if success:
            print("   ✅ 数据库已清空")
        else:
            print("   ⚠️  数据库清空失败，继续验证...")
    except Exception as e:
        print(f"   ⚠️  清空数据库时出错: {e}，继续验证...")
    finally:
        if vector_store:
            await close_vector_store(vector_store)
    
    # 第二步：注入数据
    print("\n[第二步] 注入数据...")
    vector_store = None
    try:
        vector_store = await init_vector_store()
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
        print(f"   ⚠️  数据注入时出错: {e}")
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


if __name__ == "__main__":
    """
    主函数：验证数据库数据注入
    使用单个asyncio.run调用，避免事件循环冲突
    """
    asyncio.run(main_async())
