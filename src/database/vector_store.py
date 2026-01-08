import asyncpg
import logging
import json
from typing import List, Optional, Dict, Any, Tuple
from dataclasses import dataclass
import uuid
import asyncio

from config.settings import settings
from src.database.models import Document, ImageDescription, TextImageRelation

logger = logging.getLogger(__name__)


@dataclass
class SearchResult:
    """搜索结果"""
    id: str
    content: str
    metadata: Dict[str, Any]
    score: float
    source: Optional[str] = None


@dataclass
class ImageSearchResult:
    """图像搜索结果"""
    id: str
    image_path: str
    vlm_description: str
    metadata: Dict[str, Any]
    score: float


class PostgreSQLVectorStore:
    """PostgreSQL向量存储"""
    
    def __init__(self, connection_string: Optional[str] = None):
        self.connection_string = connection_string or settings.database_url
        self.pool: Optional[asyncpg.Pool] = None
        self.embedding_dimension = settings.embedding_dimension
    
    async def connect(self):
        """创建连接池"""
        if not self.pool:
            self.pool = await asyncpg.create_pool(
                dsn=self.connection_string,
                min_size=5,
                max_size=20,
                command_timeout=60,
            )
            logger.info("PostgreSQL connection pool created")
    
    async def close(self):
        """关闭连接池"""
        if self.pool:
            await self.pool.close()
            self.pool = None
            logger.info("PostgreSQL connection pool closed")
    
    async def _vector_to_str(self, vector: List[float]) -> str:
        """将向量转换为pgvector字符串格式"""
        if len(vector) != self.embedding_dimension:
            logger.warning(f"Vector dimension mismatch: expected {self.embedding_dimension}, got {len(vector)}")
        
        return f"[{','.join(map(str, vector))}]"
    
    async def _str_to_vector(self, vector_str: str) -> List[float]:
        """将pgvector字符串转换为向量"""
        if not vector_str or vector_str == "[]":
            return []
        
        # 移除方括号并按逗号分割
        vector_str = vector_str.strip("[]")
        return [float(x) for x in vector_str.split(",")]
    
    async def create_tables(self):
        """创建数据库表"""
        async with self.pool.acquire() as conn:
            # 创建文档表
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS documents (
                    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    content TEXT NOT NULL,
                    metadata JSONB NOT NULL DEFAULT '{}',
                    embedding vector(1024) NOT NULL,
                    source TEXT,
                    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # 创建图像描述表
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS image_descriptions (
                    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    image_path TEXT NOT NULL,
                    vlm_description TEXT NOT NULL,
                    embedding vector(1024) NOT NULL,
                    metadata JSONB NOT NULL DEFAULT '{}',
                    image_size TEXT,
                    file_format TEXT,
                    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # 创建文本-图像关联表
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS text_image_relations (
                    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    document_id UUID REFERENCES documents(id) ON DELETE CASCADE,
                    image_id UUID REFERENCES image_descriptions(id) ON DELETE CASCADE,
                    similarity_score FLOAT NOT NULL,
                    relation_type TEXT,
                    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(document_id, image_id)
                )
            """)
            
            # 创建查询历史表
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS query_history (
                    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    query_text TEXT,
                    query_image_path TEXT,
                    query_type TEXT NOT NULL,
                    retrieved_document_ids JSONB NOT NULL DEFAULT '[]',
                    retrieved_image_ids JSONB NOT NULL DEFAULT '[]',
                    response TEXT,
                    response_time_ms FLOAT,
                    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # 创建向量索引
            await conn.execute("""
                CREATE INDEX IF NOT EXISTS documents_embedding_idx 
                ON documents USING ivfflat (embedding vector_cosine_ops)
                WITH (lists = 100)
            """)
            
            await conn.execute("""
                CREATE INDEX IF NOT EXISTS image_descriptions_embedding_idx 
                ON image_descriptions USING ivfflat (embedding vector_cosine_ops)
                WITH (lists = 100)
            """)
            
            logger.info("Database tables created successfully")
    
    async def insert_document(self, content: str, embedding: List[float], 
                            metadata: Optional[Dict] = None, source: Optional[str] = None) -> str:
        """插入文档"""
        async with self.pool.acquire() as conn:
            embedding_str = await self._vector_to_str(embedding)
            metadata_json = json.dumps(metadata or {})
            
            result = await conn.fetchrow("""
                INSERT INTO documents (content, embedding, doc_metadata, source)
                VALUES ($1, $2::vector, $3::jsonb, $4)
                RETURNING id
            """, content, embedding_str, metadata_json, source)
            
            doc_id = str(result['id'])
            logger.debug(f"Document inserted: {doc_id}")
            return doc_id
    
    async def insert_image_description(self, image_path: str, vlm_description: str,
                                     embedding: List[float], metadata: Optional[Dict] = None,
                                     image_size: Optional[str] = None, 
                                     file_format: Optional[str] = None) -> str:
        """插入图像描述"""
        async with self.pool.acquire() as conn:
            embedding_str = await self._vector_to_str(embedding)
            metadata_json = json.dumps(metadata or {})
            
            result = await conn.fetchrow("""
                INSERT INTO image_descriptions 
                (image_path, vlm_description, embedding, image_metadata, image_size, file_format)
                VALUES ($1, $2, $3::vector, $4::jsonb, $5, $6)
                RETURNING id
            """, image_path, vlm_description, embedding_str, metadata_json, image_size, file_format)
            
            image_id = str(result['id'])
            logger.debug(f"Image description inserted: {image_id}")
            return image_id
    
    async def create_text_image_relation(self, document_id: str, image_id: str,
                                       similarity_score: float, 
                                       relation_type: Optional[str] = None) -> str:
        """创建文本-图像关联"""
        async with self.pool.acquire() as conn:
            result = await conn.fetchrow("""
                INSERT INTO text_image_relations 
                (document_id, image_id, similarity_score, relation_type)
                VALUES ($1, $2, $3, $4)
                ON CONFLICT (document_id, image_id) 
                DO UPDATE SET similarity_score = EXCLUDED.similarity_score,
                              relation_type = EXCLUDED.relation_type
                RETURNING id
            """, uuid.UUID(document_id), uuid.UUID(image_id), similarity_score, relation_type)
            
            relation_id = str(result['id'])
            logger.debug(f"Text-image relation created: {relation_id}")
            return relation_id
    
    async def search_similar_documents(self, query_embedding: List[float], 
                                     top_k: int = 5, 
                                     threshold: float = 0.0) -> List[SearchResult]:
        """搜索相似文档"""
        async with self.pool.acquire() as conn:
            query_embedding_str = await self._vector_to_str(query_embedding)
            
            rows = await conn.fetch("""
                SELECT id, content, doc_metadata as metadata, source,
                       1 - (embedding <=> $1::vector) as similarity
                FROM documents
                WHERE 1 - (embedding <=> $1::vector) >= $2
                ORDER BY embedding <=> $1::vector
                LIMIT $3
            """, query_embedding_str, threshold, top_k)
            
            results = []
            for row in rows:
                results.append(SearchResult(
                    id=str(row['id']),
                    content=row['content'],
                    metadata=row['metadata'],
                    score=float(row['similarity']),
                    source=row['source']
                ))
            
            logger.debug(f"Found {len(results)} similar documents")
            return results
    
    async def search_similar_images(self, query_embedding: List[float],
                                  top_k: int = 5,
                                  threshold: float = 0.0) -> List[ImageSearchResult]:
        """搜索相似图像"""
        async with self.pool.acquire() as conn:
            query_embedding_str = await self._vector_to_str(query_embedding)
            
            rows = await conn.fetch("""
                SELECT id, image_path, vlm_description, image_metadata as metadata,
                       1 - (embedding <=> $1::vector) as similarity
                FROM image_descriptions
                WHERE 1 - (embedding <=> $1::vector) >= $2
                ORDER BY embedding <=> $1::vector
                LIMIT $3
            """, query_embedding_str, threshold, top_k)
            
            results = []
            for row in rows:
                results.append(ImageSearchResult(
                    id=str(row['id']),
                    image_path=row['image_path'],
                    vlm_description=row['vlm_description'],
                    metadata=row['metadata'],
                    score=float(row['similarity'])
                ))
            
            logger.debug(f"Found {len(results)} similar images")
            return results
    
    async def hybrid_search(self, text_query_embedding: Optional[List[float]] = None,
                          image_query_embedding: Optional[List[float]] = None,
                          top_k: int = 5,
                          text_weight: float = 0.5,
                          image_weight: float = 0.5) -> List[SearchResult]:
        """混合搜索（文本+图像）"""
        tasks = []
        
        if text_query_embedding:
            tasks.append(self.search_similar_documents(text_query_embedding, top_k * 2))
        
        if image_query_embedding:
            tasks.append(self.search_similar_documents(image_query_embedding, top_k * 2))
        
        if not tasks:
            return []
        
        # 并行执行搜索
        search_results = await asyncio.gather(*tasks)
        
        # 合并和加权结果
        combined_scores = {}
        for i, results in enumerate(search_results):
            weight = text_weight if i == 0 else image_weight
            
            for result in results:
                if result.id not in combined_scores:
                    combined_scores[result.id] = {
                        'result': result,
                        'weighted_score': result.score * weight
                    }
                else:
                    combined_scores[result.id]['weighted_score'] += result.score * weight
        
        # 按加权分数排序
        sorted_results = sorted(
            combined_scores.values(),
            key=lambda x: x['weighted_score'],
            reverse=True
        )
        
        final_results = []
        for item in sorted_results[:top_k]:
            result = item['result']
            # 更新分数为加权分数
            result.score = item['weighted_score']
            final_results.append(result)
        
        logger.debug(f"Hybrid search found {len(final_results)} results")
        return final_results
    
    async def get_related_images_for_document(self, document_id: str, 
                                            top_k: int = 3) -> List[ImageSearchResult]:
        """获取文档相关的图像"""
        async with self.pool.acquire() as conn:
            rows = await conn.fetch("""
                SELECT i.id, i.image_path, i.vlm_description, i.metadata, r.similarity_score
                FROM text_image_relations r
                JOIN image_descriptions i ON r.image_id = i.id
                WHERE r.document_id = $1
                ORDER BY r.similarity_score DESC
                LIMIT $2
            """, uuid.UUID(document_id), top_k)
            
            results = []
            for row in rows:
                results.append(ImageSearchResult(
                    id=str(row['id']),
                    image_path=row['image_path'],
                    vlm_description=row['vlm_description'],
                    metadata=row['metadata'],
                    score=float(row['similarity_score'])
                ))
            
            return results
    
    async def get_document_count(self) -> int:
        """获取文档数量"""
        async with self.pool.acquire() as conn:
            result = await conn.fetchrow("SELECT COUNT(*) as count FROM documents")
            return result['count']
    
    async def get_image_count(self) -> int:
        """获取图像数量"""
        async with self.pool.acquire() as conn:
            result = await conn.fetchrow("SELECT COUNT(*) as count FROM image_descriptions")
            return result['count']
    
    async def log_query_history(self, query_text: Optional[str] = None,
                              query_image_path: Optional[str] = None,
                              query_type: str = "text",
                              retrieved_document_ids: Optional[List[str]] = None,
                              retrieved_image_ids: Optional[List[str]] = None,
                              response: Optional[str] = None,
                              response_time_ms: Optional[float] = None) -> str:
        """记录查询历史"""
        async with self.pool.acquire() as conn:
            result = await conn.fetchrow("""
                INSERT INTO query_history 
                (query_text, query_image_path, query_type, 
                 retrieved_document_ids, retrieved_image_ids, 
                 response, response_time_ms)
                VALUES ($1, $2, $3, $4::jsonb, $5::jsonb, $6, $7)
                RETURNING id
            """, query_text, query_image_path, query_type,
                json.dumps(retrieved_document_ids or []),
                json.dumps(retrieved_image_ids or []),
                response, response_time_ms)
            
            return str(result['id'])


# 全局向量存储实例
vector_store = PostgreSQLVectorStore()