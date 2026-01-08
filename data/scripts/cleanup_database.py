#!/usr/bin/env python3
"""
清理数据库脚本
连接Docker PostgreSQL，检查并删除现有表结构
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
        
        # 检查pgvector扩展
        result = await conn.fetchval(
            "SELECT 1 FROM pg_extension WHERE extname = 'vector'"
        )
        
        if result:
            print("✅ pgvector扩展已安装")
        else:
            print("❌ pgvector扩展未安装")
            print("   需要安装pgvector扩展以支持向量存储")
            print("   在容器中执行: CREATE EXTENSION vector;")
        
        await conn.close()
        return True
        
    except Exception as e:
        print(f"❌ 数据库连接失败: {e}")
        print("\n可能的原因:")
        print("1. Docker容器未运行")
        print("2. 数据库配置错误")
        print("3. 网络连接问题")
        return False

async def list_tables():
    """列出所有表"""
    print("\n" + "="*50)
    print("列出数据库中的表")
    print("="*50)
    
    try:
        conn = await asyncpg.connect(
            host=settings.postgres_host,
            port=settings.postgres_port,
            user=settings.postgres_user,
            password=settings.postgres_password,
            database=settings.postgres_db
        )
        
        tables = await conn.fetch("""
            SELECT table_name, table_type 
            FROM information_schema.tables 
            WHERE table_schema = 'public'
            ORDER BY table_name
        """)
        
        if tables:
            print(f"找到 {len(tables)} 个表:")
            for table in tables:
                table_type = "视图" if table['table_type'] == 'VIEW' else "表"
                print(f"  - {table['table_name']} ({table_type})")
        else:
            print("数据库中没有表")
        
        await conn.close()
        return [table['table_name'] for table in tables]
        
    except Exception as e:
        print(f"❌ 列出表失败: {e}")
        return []

async def drop_all_tables():
    """删除所有表"""
    print("\n" + "="*50)
    print("删除所有表")
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
        
        print(f"准备删除 {len(tables)} 个表:")
        for table in tables:
            print(f"  - {table['table_name']}")
        
        # 删除所有表（需要禁用外键约束）
        await conn.execute("SET session_replication_role = 'replica';")
        
        for table in tables:
            try:
                await conn.execute(f'DROP TABLE IF EXISTS "{table["table_name"]}" CASCADE')
                print(f"  ✅ 删除表: {table['table_name']}")
            except Exception as e:
                print(f"  ❌ 删除表 {table['table_name']} 失败: {e}")
        
        await conn.execute("SET session_replication_role = 'origin';")
        
        print("\n✅ 所有表已删除")
        await conn.close()
        return True
        
    except Exception as e:
        print(f"❌ 删除表失败: {e}")
        return False

async def check_pgvector_extension():
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
            print("🔄 安装pgvector扩展...")
            try:
                await conn.execute("CREATE EXTENSION vector")
                print("✅ pgvector扩展安装成功")
            except Exception as e:
                print(f"❌ pgvector扩展安装失败: {e}")
                print("   可能需要手动安装:")
                print("   1. 进入Docker容器: docker exec -it my-postgres bash")
                print("   2. 安装扩展: apt-get update && apt-get install -y postgresql-16-pgvector")
                print("   3. 连接到数据库: psql -U postgres -d mydb")
                print("   4. 创建扩展: CREATE EXTENSION vector;")
        
        await conn.close()
        return True
        
    except Exception as e:
        print(f"❌ 检查pgvector扩展失败: {e}")
        return False

async def main():
    """主函数"""
    print("Docker PostgreSQL数据库清理工具")
    print("="*60)
    
    # 检查连接
    if not await check_database_connection():
        return 1
    
    # 列出表
    tables = await list_tables()
    
    if tables:
        print("\n" + "="*60)
        print("警告: 这将删除所有表！")
        print("="*60)
        
        response = input("确认删除所有表？(输入 'yes' 继续): ")
        if response.lower() != 'yes':
            print("操作已取消")
            return 0
        
        # 删除表
        if not await drop_all_tables():
            return 1
    else:
        print("\n数据库是空的，无需清理")
    
    # 检查pgvector扩展
    await check_pgvector_extension()
    
    print("\n" + "="*60)
    print("数据库清理完成")
    print("="*60)
    print("\n下一步:")
    print("1. 运行数据库迁移创建表结构")
    print("2. 处理数据导入")
    print("3. 测试系统功能")
    
    return 0

if __name__ == "__main__":
    import asyncio
    sys.exit(asyncio.run(main()))