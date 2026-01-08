from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy import text
import asyncpg
from typing import AsyncGenerator
import logging

from config.settings import settings

logger = logging.getLogger(__name__)


class Database:
    def __init__(self):
        self.engine = create_async_engine(
            settings.async_database_url,
            echo=False,
            pool_size=20,
            max_overflow=10,
            pool_pre_ping=True,
        )
        
        self.async_session = sessionmaker(
            self.engine,
            class_=AsyncSession,
            expire_on_commit=False,
        )
    
    async def get_session(self) -> AsyncGenerator[AsyncSession, None]:
        """获取数据库会话"""
        async with self.async_session() as session:
            try:
                yield session
            except Exception as e:
                await session.rollback()
                logger.error(f"Database session error: {e}")
                raise
            finally:
                await session.close()
    
    async def test_connection(self) -> bool:
        """测试数据库连接"""
        try:
            async with self.engine.connect() as conn:
                await conn.execute(text("SELECT 1"))
            logger.info("Database connection test successful")
            return True
        except Exception as e:
            logger.error(f"Database connection test failed: {e}")
            return False
    
    async def check_pgvector_extension(self) -> bool:
        """检查pgvector扩展是否已安装"""
        try:
            async with self.engine.connect() as conn:
                result = await conn.execute(
                    text("SELECT EXISTS(SELECT 1 FROM pg_extension WHERE extname = 'vector')")
                )
                exists = result.scalar()
                if exists:
                    logger.info("pgvector extension is installed")
                else:
                    logger.warning("pgvector extension is NOT installed")
                return bool(exists)
        except Exception as e:
            logger.error(f"Failed to check pgvector extension: {e}")
            return False
    
    async def create_pgvector_extension(self) -> bool:
        """创建pgvector扩展"""
        try:
            async with self.engine.connect() as conn:
                await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
                await conn.commit()
                logger.info("pgvector extension created successfully")
                return True
        except Exception as e:
            logger.error(f"Failed to create pgvector extension: {e}")
            return False


# 全局数据库实例
database = Database()


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """依赖注入：获取数据库会话"""
    async for session in database.get_session():
        yield session


async def get_asyncpg_pool() -> asyncpg.Pool:
    """获取asyncpg连接池（用于直接向量操作）"""
    return await asyncpg.create_pool(
        dsn=settings.database_url,
        min_size=5,
        max_size=20,
    )