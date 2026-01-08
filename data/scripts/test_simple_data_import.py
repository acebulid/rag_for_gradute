#!/usr/bin/env python3
"""
简化数据导入测试
测试基本的数据库连接和数据插入功能
"""

import sys
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import asyncpg
import uuid
from datetime import datetime
from config.settings import settings

async def test_basic_insert():
    """测试基本数据插入"""
    print("="*50)
    print("测试基本数据插入")
    print("="*50)
    
    try:
        conn = await asyncpg.connect(
            host=settings.postgres_host,
            port=settings.postgres_port,
            user=settings.postgres_user,
            password=settings.postgres_password,
            database=settings.postgres_db
        )
        
        # 1. 测试插入文档
        print("📄 测试插入文档数据...")
        doc_id = uuid.uuid4()
        await conn.execute("""
            INSERT INTO documents (id, content, doc_metadata, embedding, source)
            VALUES ($1, $2, $3, $4, $5)
        """, doc_id, "测试文档内容", {"test": True}, "[0.1,0.2,0.3]" + ",0.0"*1021, "test_source")
        print("  ✅ 文档插入成功")
        
        # 2. 测试插入图像描述
        print("\n🖼️  测试插入图像描述...")
        image_id = uuid.uuid4()
        await conn.execute("""
            INSERT INTO image_descriptions (id, image_path, vlm_description, embedding, image_metadata)
            VALUES ($1, $2, $3, $4, $5)
        """, image_id, "test_image.jpg", "测试图像描述", "[0.4,0.5,0.6]" + ",0.0"*1021, {"format": "jpg"})
        print("  ✅ 图像描述插入成功")
        
        # 3. 测试插入关联
        print("\n🔗 测试插入文本-图像关联...")
        relation_id = uuid.uuid4()
        await conn.execute("""
            INSERT INTO text_image_relations (id, document_id, image_id, similarity_score, relation_type)
            VALUES ($1, $2, $3, $4, $5)
        """, relation_id, doc_id, image_id, 0.85, "测试关联")
        print("  ✅ 关联插入成功")
        
        # 4. 验证数据
        print("\n📊 验证插入的数据...")
        
        # 统计文档数量
        doc_count = await conn.fetchval("SELECT COUNT(*) FROM documents")
        print(f"  文档数量: {doc_count}")
        
        # 统计图像描述数量
        image_count = await conn.fetchval("SELECT COUNT(*) FROM image_descriptions")
        print(f"  图像描述数量: {image_count}")
        
        # 统计关联数量
        relation_count = await conn.fetchval("SELECT COUNT(*) FROM text_image_relations")
        print(f"  关联数量: {relation_count}")
        
        # 5. 清理测试数据
        print("\n🧹 清理测试数据...")
        await conn.execute("DELETE FROM text_image_relations WHERE id = $1", relation_id)
        await conn.execute("DELETE FROM image_descriptions WHERE id = $1", image_id)
        await conn.execute("DELETE FROM documents WHERE id = $1", doc_id)
        print("  ✅ 测试数据清理完成")
        
        await conn.close()
        return True
        
    except Exception as e:
        print(f"❌ 数据插入测试失败: {e}")
        return False

async def test_vector_operations():
    """测试向量操作"""
    print("\n" + "="*50)
    print("测试向量操作")
    print("="*50)
    
    try:
        conn = await asyncpg.connect(
            host=settings.postgres_host,
            port=settings.postgres_port,
            user=settings.postgres_user,
            password=settings.postgres_password,
            database=settings.postgres_db
        )
        
        # 测试向量相似度计算
        print("🧮 测试向量相似度计算...")
        similarity = await conn.fetchval("""
            SELECT '[1,0,0]'::vector <=> '[0,1,0]'::vector
        """)
        print(f"  向量 [1,0,0] 和 [0,1,0] 的余弦相似度: {similarity}")
        
        # 测试向量维度
        print("\n📏 测试向量维度...")
        try:
            # 尝试插入1024维向量
            test_vector = "[0.1]" + ",0.1"*1023
            await conn.execute("SELECT $1::vector(1024)", test_vector)
            print("  ✅ 1024维向量支持正常")
        except Exception as e:
            print(f"  ❌ 向量维度测试失败: {e}")
        
        await conn.close()
        return True
        
    except Exception as e:
        print(f"❌ 向量操作测试失败: {e}")
        return False

async def test_table_structure():
    """测试表结构"""
    print("\n" + "="*50)
    print("测试表结构")
    print("="*50)
    
    try:
        conn = await asyncpg.connect(
            host=settings.postgres_host,
            port=settings.postgres_port,
            user=settings.postgres_user,
            password=settings.postgres_password,
            database=settings.postgres_db
        )
        
        # 检查表结构
        tables = ['documents', 'image_descriptions', 'text_image_relations', 'query_history']
        
        for table in tables:
            columns = await conn.fetch("""
                SELECT column_name, data_type, is_nullable
                FROM information_schema.columns
                WHERE table_name = $1
                ORDER BY ordinal_position
            """, table)
            
            print(f"\n{table} 表结构:")
            for col in columns:
                nullable = "NULL" if col['is_nullable'] == 'YES' else "NOT NULL"
                print(f"  - {col['column_name']}: {col['data_type']} ({nullable})")
        
        await conn.close()
        return True
        
    except Exception as e:
        print(f"❌ 表结构测试失败: {e}")
        return False

async def main():
    """主函数"""
    print("简化数据导入测试")
    print("="*60)
    
    tests = [
        ("基本数据插入", test_basic_insert),
        ("向量操作", test_vector_operations),
        ("表结构", test_table_structure),
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            success = test_func()
            results.append((test_name, success))
        except Exception as e:
            print(f"{test_name}测试异常: {e}")
            results.append((test_name, False))
    
    # 打印总结
    print("\n" + "="*60)
    print("数据导入测试总结")
    print("="*60)
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    for test_name, success in results:
        status = "✅ 通过" if success else "❌ 失败"
        print(f"{test_name}: {status}")
    
    print(f"\n总计: {passed}/{total} 个测试通过")
    
    if passed == total:
        print("\n🎉 所有数据导入测试通过！")
        print("\n数据库已准备好接收多模态RAG数据。")
        print("下一步可以运行完整的数据处理流程。")
        return 0
    else:
        print(f"\n⚠️  有 {total - passed} 个测试失败")
        return 1

if __name__ == "__main__":
    import asyncio
    sys.exit(asyncio.run(main()))