#!/usr/bin/env python3
"""
创建多模态RAG系统表结构
"""

import sys
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import asyncpg
from config.settings import settings

async def create_tables():
    """创建多模态RAG系统表结构"""
    print("="*50)
    print("创建多模态RAG系统表结构")
    print("="*50)
    
    try:
        conn = await asyncpg.connect(
            host=settings.postgres_host,
            port=settings.postgres_port,
            user=settings.postgres_user,
            password=settings.postgres_password,
            database=settings.postgres_db
        )
        
        # 1. 创建文档表
        print("📄 创建文档表 (documents)...")
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS documents (
                id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                content TEXT NOT NULL,
                doc_metadata JSONB NOT NULL DEFAULT '{}',
                embedding VECTOR(1024) NOT NULL,
                source VARCHAR(255),
                created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
            )
        """)
        print("  ✅ 文档表创建成功")
        
        # 2. 创建图像描述表
        print("\n🖼️  创建图像描述表 (image_descriptions)...")
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS image_descriptions (
                id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                image_path VARCHAR(500) NOT NULL,
                vlm_description TEXT NOT NULL,
                embedding VECTOR(1024) NOT NULL,
                image_metadata JSONB NOT NULL DEFAULT '{}',
                image_size VARCHAR(50),
                file_format VARCHAR(10),
                created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
            )
        """)
        print("  ✅ 图像描述表创建成功")
        
        # 3. 创建文本-图像关联表
        print("\n🔗 创建文本-图像关联表 (text_image_relations)...")
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS text_image_relations (
                id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                document_id UUID NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
                image_id UUID NOT NULL REFERENCES image_descriptions(id) ON DELETE CASCADE,
                similarity_score FLOAT NOT NULL,
                relation_type VARCHAR(50),
                created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(document_id, image_id)
            )
        """)
        print("  ✅ 文本-图像关联表创建成功")
        
        # 4. 创建查询历史表
        print("\n📊 创建查询历史表 (query_history)...")
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS query_history (
                id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                query_text TEXT,
                query_image_path VARCHAR(500),
                query_type VARCHAR(20) NOT NULL,
                retrieved_document_ids JSONB NOT NULL DEFAULT '[]',
                retrieved_image_ids JSONB NOT NULL DEFAULT '[]',
                response TEXT,
                response_time_ms FLOAT,
                created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
            )
        """)
        print("  ✅ 查询历史表创建成功")
        
        # 5. 创建索引以提高查询性能
        print("\n⚡ 创建性能索引...")
        
        # 文档表的向量索引
        await conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_documents_embedding 
            ON documents USING ivfflat (embedding vector_cosine_ops)
            WITH (lists = 100)
        """)
        print("  ✅ 文档向量索引创建成功")
        
        # 图像描述表的向量索引
        await conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_image_descriptions_embedding 
            ON image_descriptions USING ivfflat (embedding vector_cosine_ops)
            WITH (lists = 100)
        """)
        print("  ✅ 图像描述向量索引创建成功")
        
        # 关联表的外键索引
        await conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_relations_document_id 
            ON text_image_relations(document_id)
        """)
        await conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_relations_image_id 
            ON text_image_relations(image_id)
        """)
        print("  ✅ 关联表索引创建成功")
        
        # 查询历史表的查询类型索引
        await conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_query_history_type 
            ON query_history(query_type)
        """)
        print("  ✅ 查询历史索引创建成功")
        
        await conn.close()
        return True
        
    except Exception as e:
        print(f"❌ 创建表失败: {e}")
        return False

async def verify_tables():
    """验证表结构"""
    print("\n" + "="*50)
    print("验证表结构")
    print("="*50)
    
    try:
        conn = await asyncpg.connect(
            host=settings.postgres_host,
            port=settings.postgres_port,
            user=settings.postgres_user,
            password=settings.postgres_password,
            database=settings.postgres_db
        )
        
        # 检查表是否存在
        tables = ['documents', 'image_descriptions', 'text_image_relations', 'query_history']
        
        for table in tables:
            result = await conn.fetchval(
                "SELECT 1 FROM information_schema.tables WHERE table_name = $1",
                table
            )
            if result:
                print(f"✅ 表 '{table}' 存在")
            else:
                print(f"❌ 表 '{table}' 不存在")
        
        # 检查索引
        indexes = await conn.fetch("""
            SELECT indexname, tablename 
            FROM pg_indexes 
            WHERE schemaname = 'public'
            ORDER BY tablename, indexname
        """)
        
        if indexes:
            print(f"\n📊 找到 {len(indexes)} 个索引:")
            for idx in indexes:
                print(f"  - {idx['indexname']} (表: {idx['tablename']})")
        else:
            print("\n⚠️  未找到索引")
        
        await conn.close()
        return True
        
    except Exception as e:
        print(f"❌ 验证表失败: {e}")
        return False

async def test_connection():
    """测试数据库连接和基本功能"""
    print("\n" + "="*50)
    print("测试数据库连接和基本功能")
    print("="*50)
    
    try:
        conn = await asyncpg.connect(
            host=settings.postgres_host,
            port=settings.postgres_port,
            user=settings.postgres_user,
            password=settings.postgres_password,
            database=settings.postgres_db
        )
        
        # 测试pgvector功能
        print("🧪 测试pgvector功能...")
        try:
            # 测试向量操作
            await conn.execute("SELECT '[1,2,3]'::vector <=> '[4,5,6]'::vector")
            print("  ✅ pgvector向量操作正常")
        except Exception as e:
            print(f"  ❌ pgvector测试失败: {e}")
        
        # 测试JSONB功能
        print("📋 测试JSONB功能...")
        try:
            await conn.execute("SELECT '{\"test\": \"value\"}'::jsonb")
            print("  ✅ JSONB操作正常")
        except Exception as e:
            print(f"  ❌ JSONB测试失败: {e}")
        
        await conn.close()
        return True
        
    except Exception as e:
        print(f"❌ 测试连接失败: {e}")
        return False

async def main():
    """主函数"""
    print("多模态RAG系统数据库表结构创建工具")
    print("="*60)
    
    # 创建表
    if not await create_tables():
        return 1
    
    # 验证表
    if not await verify_tables():
        return 1
    
    # 测试连接
    if not await test_connection():
        return 1
    
    print("\n" + "="*60)
    print("🎉 数据库表结构创建完成！")
    print("="*60)
    print("\n已创建的表:")
    print("  1. 📄 documents - 文档表")
    print("  2. 🖼️  image_descriptions - 图像描述表")
    print("  3. 🔗 text_image_relations - 文本-图像关联表")
    print("  4. 📊 query_history - 查询历史表")
    print("\n下一步:")
    print("  1. 运行数据处理: python data/scripts/process_data.py")
    print("  2. 启动API服务器: python -m src.api.main")
    print("  3. 测试检索功能")
    
    return 0

if __name__ == "__main__":
    import asyncio
    sys.exit(asyncio.run(main()))
