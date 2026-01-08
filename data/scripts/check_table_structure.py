#!/usr/bin/env python3
"""
检查数据库表结构
"""

import sys
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import asyncpg
from config.settings import settings

async def check_table_structure():
    """检查表结构"""
    print("="*50)
    print("检查数据库表结构")
    print("="*50)
    
    try:
        conn = await asyncpg.connect(
            host=settings.postgres_host,
            port=settings.postgres_port,
            user=settings.postgres_user,
            password=settings.postgres_password,
            database=settings.postgres_db
        )
        
        # 检查documents表
        print("\n📄 documents表结构:")
        columns = await conn.fetch("""
            SELECT column_name, data_type, is_nullable
            FROM information_schema.columns
            WHERE table_name = 'documents'
            ORDER BY ordinal_position
        """)
        
        for col in columns:
            nullable = "NULL" if col['is_nullable'] == 'YES' else "NOT NULL"
            print(f"  - {col['column_name']}: {col['data_type']} ({nullable})")
        
        # 检查image_descriptions表
        print("\n🖼️  image_descriptions表结构:")
        columns = await conn.fetch("""
            SELECT column_name, data_type, is_nullable
            FROM information_schema.columns
            WHERE table_name = 'image_descriptions'
            ORDER BY ordinal_position
        """)
        
        for col in columns:
            nullable = "NULL" if col['is_nullable'] == 'YES' else "NOT NULL"
            print(f"  - {col['column_name']}: {col['data_type']} ({nullable})")
        
        # 检查数据
        print("\n📊 数据统计:")
        
        # 文档数量
        doc_count = await conn.fetchval("SELECT COUNT(*) FROM documents")
        print(f"  文档数量: {doc_count}")
        
        # 图像描述数量
        image_count = await conn.fetchval("SELECT COUNT(*) FROM image_descriptions")
        print(f"  图像描述数量: {image_count}")
        
        # 关联数量
        relation_count = await conn.fetchval("SELECT COUNT(*) FROM text_image_relations")
        print(f"  关联数量: {relation_count}")
        
        # 查询历史数量
        query_count = await conn.fetchval("SELECT COUNT(*) FROM query_history")
        print(f"  查询历史数量: {query_count}")
        
        # 如果有数据，显示示例
        if doc_count > 0:
            print("\n📝 文档示例:")
            rows = await conn.fetch("SELECT id, LEFT(content, 50) as preview FROM documents LIMIT 3")
            for row in rows:
                print(f"  - {row['id']}: {row['preview']}...")
        
        if image_count > 0:
            print("\n🖼️  图像描述示例:")
            rows = await conn.fetch("SELECT id, image_path, LEFT(vlm_description, 50) as preview FROM image_descriptions LIMIT 3")
            for row in rows:
                print(f"  - {row['id']}: {row['image_path']}")
                print(f"    描述: {row['preview']}...")
        
        await conn.close()
        return True
        
    except Exception as e:
        print(f"❌ 检查表结构失败: {e}")
        import traceback
        traceback.print_exc()
        return False

async def main():
    """主函数"""
    print("数据库表结构检查工具")
    print("="*60)
    
    if not await check_table_structure():
        return 1
    
    print("\n" + "="*60)
    print("检查完成")
    print("="*60)
    
    return 0

if __name__ == "__main__":
    import asyncio
    sys.exit(asyncio.run(main()))