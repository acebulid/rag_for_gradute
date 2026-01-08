#!/usr/bin/env python3
"""
非交互式数据库清理脚本
自动连接Docker PostgreSQL并清理表结构
"""

import sys
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import asyncpg
from config.settings import settings

async def check_database_connection():
    """检查数据库连接"""
    print("="*50)
    print("检查数据库连接")
    print("="*50)
    
    try:
        conn = await asyncpg.connect(
            host=settings.postgres_host,
            port=settings.postgres_port,
            user=settings.postgres_user,
            password=settings.postgres_password,
            database=settings.postgres_db
        )
        
        print("✅ 成功连接到Docker PostgreSQL数据库")
        print(f"   主机: {settings.postgres_host}:{settings.postgres_port}")
        print(f"   数据库: {settings.postgres_db}")
        print(f"   用户: {settings.postgres_user}")
        
        await conn.close()
        return True
        
    except Exception as e:
        print(f"❌ 数据库连接失败: {e}")
        return False

async def list_and_drop_tables():
    """列出并删除所有表"""
    print("\n" + "="*50)
    print("清理数据库表结构")
    print("="*50)
    
    try:
        conn = await asyncpg.connect(
            host=settings.postgres_host,
            port=settings.postgres_port,
            user=settings.postgres_user,
            password=settings.postgres_password,
            database=settings.postgres_db
        )
        
        # 获取所有表名
        tables = await conn.fetch("""
            SELECT table_name 
            FROM information_schema.tables 
            WHERE table_schema = 'public' 
            AND table_type = 'BASE TABLE'
        """)
        
        if not tables:
            print("数据库中没有需要删除的表")
            await conn.close()
            return True
        
        print(f"找到 {len(tables)} 个表，准备删除:")
        for table in tables:
            print(f"  - {table['table_name']}")
        
        # 删除所有表
        await conn.execute("SET session_replication_role = 'replica';")
        
        dropped_count = 0
        for table in tables:
            try:
                await conn.execute(f'DROP TABLE IF EXISTS "{table["table_name"]}" CASCADE')
                print(f"  ✅ 删除表: {table['table_name']}")
                dropped_count += 1
            except Exception as e:
                print(f"  ❌ 删除表 {table['table_name']} 失败: {e}")
        
        await conn.execute("SET session_replication_role = 'origin';")
        
        print(f"\n✅ 成功删除 {dropped_count}/{len(tables)} 个表")
        await conn.close()
        return True
        
    except Exception as e:
        print(f"❌ 清理表失败: {e}")
        return False

async def check_and_install_pgvector():
    """检查并安装pgvector扩展"""
    print("\n" + "="*50)
    print("检查pgvector扩展")
    print("="*50)
    
    try:
        conn = await asyncpg.connect(
            host=settings.postgres_host,
            port=settings.postgres_port,
            user=settings.postgres_user,
            password=settings.postgres_password,
            database=settings.postgres_db
        )
        
        # 检查扩展是否已安装
        result = await conn.fetchval(
            "SELECT 1 FROM pg_extension WHERE extname = 'vector'"
        )
        
        if result:
            print("✅ pgvector扩展已安装")
        else:
            print("🔄 尝试安装pgvector扩展...")
            try:
                await conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
                print("✅ pgvector扩展安装成功")
            except Exception as e:
                print(f"⚠️  pgvector扩展安装失败: {e}")
                print("   注意: 标准PostgreSQL镜像可能不包含pgvector")
                print("   如果需要向量支持，请使用pgvector/pgvector镜像")
        
        await conn.close()
        return True
        
    except Exception as e:
        print(f"❌ 检查pgvector扩展失败: {e}")
        return False

async def main():
    """主函数"""
    print("Docker PostgreSQL数据库自动清理工具")
    print("="*60)
    
    # 检查连接
    if not await check_database_connection():
        print("\n❌ 无法连接到数据库，请检查:")
        print(f"  1. Docker容器是否运行: docker ps | grep my-postgres")
        print(f"  2. 配置是否正确: {settings.postgres_host}:{settings.postgres_port}/{settings.postgres_db}")
        return 1
    
    # 清理表
    if not await list_and_drop_tables():
        return 1
    
    # 检查pgvector
    await check_and_install_pgvector()
    
    print("\n" + "="*60)
    print("数据库清理完成")
    print("="*60)
    print("\n下一步:")
    print("1. 创建多模态RAG系统表结构")
    print("2. 处理数据导入")
    print("3. 测试系统功能")
    
    return 0

if __name__ == "__main__":
    import asyncio
    sys.exit(asyncio.run(main()))