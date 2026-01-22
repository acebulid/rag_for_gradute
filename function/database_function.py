"""
PostgreSQL向量存储操作
包含数据库连接、数据插入、查询、验证等所有数据库操作函数
"""
import asyncpg
import logging
import json
import traceback
import asyncio
import uuid
import os
from typing import List, Optional, Dict, Any, Tuple
from dataclasses import dataclass

# 导入配置和模型
from database.models import Document, ImageDescription, TextImageRelation

# 配置日志
logger = logging.getLogger(__name__)

# 从环境变量读取配置
def get_database_url():
    """从环境变量构建数据库连接字符串"""
    host = os.getenv('POSTGRES_HOST', 'localhost')
    port = os.getenv('POSTGRES_PORT', '5432')
    db = os.getenv('POSTGRES_DB', 'mydb')
    user = os.getenv('POSTGRES_USER', 'postgres')
    password = os.getenv('POSTGRES_PASSWORD', '11111111')
    
    return f"postgresql://{user}:{password}@{host}:{port}/{db}"

def get_embedding_dimension():
    """从环境变量获取向量维度"""
    return int(os.getenv('EMBEDDING_DIMENSION', '1024'))


# ========== 数据结构定义 ==========
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


# ========== 核心向量存储类 ==========
class PostgreSQLVectorStore:
    """PostgreSQL向量存储"""
    
    def __init__(self, connection_string: Optional[str] = None):
        self.connection_string = connection_string or get_database_url()
        self.pool: Optional[asyncpg.Pool] = None
        self.embedding_dimension = get_embedding_dimension()
    
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
                    doc_metadata JSONB NOT NULL DEFAULT '{}',
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
                    embedding vector(512) NOT NULL,  -- CLIP模型使用512维向量
                    image_metadata JSONB NOT NULL DEFAULT '{}',
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
                    relation_metadata JSONB NOT NULL DEFAULT '{}',
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
                            metadata: Optional[Dict] = None, source: Optional[str] = None,
                            title: Optional[str] = None) -> str:
        """插入文档"""
        async with self.pool.acquire() as conn:
            embedding_str = await self._vector_to_str(embedding)
            
            # 构建元数据
            doc_metadata = metadata or {}
            if title:
                doc_metadata["title"] = title
            
            metadata_json = json.dumps(doc_metadata)
            
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
    
    async def insert_image(self, image_path: str, image_embedding: List[float],
                          metadata: Optional[Dict] = None, image_name: Optional[str] = None,
                          chapter_title: Optional[str] = None) -> str:
        """插入图片向量（直接存储图片编码）"""
        async with self.pool.acquire() as conn:
            embedding_str = await self._vector_to_str(image_embedding)
            
            # 构建元数据
            image_metadata = metadata or {}
            if image_name:
                image_metadata["image_name"] = image_name
            if chapter_title:
                image_metadata["chapter_title"] = chapter_title
            
            metadata_json = json.dumps(image_metadata)
            
            result = await conn.fetchrow("""
                INSERT INTO image_descriptions 
                (image_path, vlm_description, embedding, image_metadata, image_size, file_format)
                VALUES ($1, $2, $3::vector, $4::jsonb, $5, $6)
                RETURNING id
            """, image_path, "阿里云图片向量", embedding_str, metadata_json, "unknown", "png")
            
            image_id = str(result['id'])
            logger.debug(f"Image vector inserted: {image_id}")
            return image_id
    
    async def create_text_image_relation(self, document_id: str, image_id: str,
                                       similarity_score: float, 
                                       relation_type: Optional[str] = None,
                                       metadata: Optional[Dict] = None) -> str:
        """创建文本-图像关联"""
        async with self.pool.acquire() as conn:
            # 构建元数据
            relation_metadata = metadata or {}
            metadata_json = json.dumps(relation_metadata)
            
            result = await conn.fetchrow("""
                INSERT INTO text_image_relations 
                (document_id, image_id, similarity_score, relation_type, relation_metadata)
                VALUES ($1, $2, $3, $4, $5::jsonb)
                ON CONFLICT (document_id, image_id) 
                DO UPDATE SET similarity_score = EXCLUDED.similarity_score,
                              relation_type = EXCLUDED.relation_type,
                              relation_metadata = EXCLUDED.relation_metadata
                RETURNING id
            """, uuid.UUID(document_id), uuid.UUID(image_id), similarity_score, relation_type, metadata_json)
            
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
    
    async def search_by_keywords(self, keyword_embeddings: List[List[float]], 
                               top_k: int = 5, 
                               threshold: float = 0.0) -> List[SearchResult]:
        """基于多个关键词向量进行检索"""
        all_results = []
        
        for embedding in keyword_embeddings:
            results = await self.search_similar_documents(
                embedding, 
                top_k=top_k, 
                threshold=threshold
            )
            all_results.extend(results)
        
        # 去重和排序
        return self._deduplicate_and_sort_results(all_results)
    
    def _deduplicate_and_sort_results(self, results: List[SearchResult]) -> List[SearchResult]:
        """去重和按相似度排序"""
        seen_ids = set()
        unique_results = []
        
        for result in sorted(results, key=lambda x: x.score, reverse=True):
            if result.id not in seen_ids:
                seen_ids.add(result.id)
                unique_results.append(result)
        
        return unique_results
    
    async def text_search_by_keywords(self, keywords: List[str], 
                                    top_k: int = 5) -> List[SearchResult]:
        """基于关键词进行文本匹配搜索"""
        if not keywords:
            return []
            
        async with self.pool.acquire() as conn:
            # 构建SQL查询条件
            conditions = []
            params = []
            
            for i, keyword in enumerate(keywords):
                conditions.append(f"content ILIKE ${i+1}")
                params.append(f"%{keyword}%")
            
            where_clause = " OR ".join(conditions)
            
            # 构建排序条件：匹配更多关键词的文档排名更高
            order_conditions = []
            for j in range(len(keywords)):
                order_conditions.append(f"WHEN content ILIKE ${j+1} THEN {len(keywords)-j}")
            
            order_case = "CASE " + " ".join(order_conditions) + " ELSE 0 END"
            
            # 添加LIMIT参数
            params.append(top_k)
            
            rows = await conn.fetch(f"""
                SELECT id, content, doc_metadata as metadata, source, 1.0 as similarity
                FROM documents
                WHERE {where_clause}
                ORDER BY {order_case} DESC
                LIMIT ${len(keywords)+1}
            """, *params)
            
            results = []
            for row in rows:
                results.append(SearchResult(
                    id=str(row['id']),
                    content=row['content'],
                    metadata=row['metadata'],
                    score=float(row['similarity']),
                    source=row['source']
                ))
            
            return results
    
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


# ========== 数据验证工具函数 ==========
async def clear_database(vector_store: PostgreSQLVectorStore) -> bool:
    """
    清空数据库中的所有数据
    
    参数:
        vector_store: 已连接的向量存储实例
    
    返回:
        bool: 清空操作是否成功
    
    注意:
        这个函数会删除所有表中的数据，但不会删除表结构
    """
    try:
        async with vector_store.pool.acquire() as conn:
            # 先清空关联表（有外键依赖）
            await conn.execute("DELETE FROM text_image_relations")
            # 再清空其他表（按依赖顺序）
            await conn.execute("DELETE FROM query_history")
            await conn.execute("DELETE FROM image_descriptions")
            await conn.execute("DELETE FROM documents")
            # 提交事务（asyncpg自动提交，但显式确认）
            await conn.execute("COMMIT")
        return True
    except Exception as e:
        print(f"清空数据库失败: {e}")
        return False


async def inject_sample_data(vector_store: PostgreSQLVectorStore) -> Dict[str, Any]:
    """
    注入示例数据到数据库
    
    参数:
        vector_store: 已连接的向量存储实例
    
    返回:
        Dict: 注入结果统计
    """
    try:
        import json
        from pathlib import Path
        
        # 加载处理后的数据
        data_path = Path("data/processed/database_data.json")
        if not data_path.exists():
            return {
                "status": "failed",
                "error": f"数据文件不存在: {data_path}"
            }
        
        with open(data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        chapters = data.get("chapters", [])
        total_images = data.get("total_images", 0)
        
        if not chapters:
            return {
                "status": "failed",
                "error": "没有找到章节数据"
            }
        
        print(f"开始注入数据: {len(chapters)}个章节, {total_images}张图片")
        
        # 注入章节数据
        chapter_ids = {}
        for chapter in chapters:
            chapter_title = chapter.get("title", "")
            content_preview = chapter.get("content_preview", "")
            image_count = chapter.get("image_count", 0)
            image_refs = chapter.get("image_refs", [])
            
            # 创建文档元数据
            metadata = {
                "title": chapter_title,
                "category": "chapter_title",
                "image_count": str(image_count),
                "chapter_id": chapter.get("id", "")
            }
            
            # 生成简单的向量（这里使用随机向量，实际应用中应该使用模型生成）
            import random
            embedding = [random.uniform(-1, 1) for _ in range(vector_store.embedding_dimension)]
            
            # 插入文档
            doc_id = await vector_store.insert_document(
                content=content_preview,
                embedding=embedding,
                metadata=metadata,
                source="首都师范大学校园导览",
                title=chapter_title
            )
            
            chapter_ids[chapter_title] = doc_id
            print(f"  章节注入: {chapter_title} (ID: {doc_id})")
            
            # 注入图片数据
            for image_ref in image_refs:
                # 创建图片元数据
                image_metadata = {
                    "image_ref": image_ref,
                    "chapter_title": chapter_title,
                    "category": "chapter_image"
                }
                
                # 生成图片向量
                image_embedding = [random.uniform(-1, 1) for _ in range(vector_store.embedding_dimension)]
                
                # 插入图片
                image_id = await vector_store.insert_image(
                    image_path=f"data/{image_ref}",
                    image_embedding=image_embedding,
                    metadata=image_metadata,
                    image_name=image_ref,
                    chapter_title=chapter_title
                )
                
                # 创建文本-图片关联
                similarity_score = 0.9  # 默认相似度
                relation_id = await vector_store.create_text_image_relation(
                    document_id=doc_id,
                    image_id=image_id,
                    similarity_score=similarity_score,
                    relation_type="chapter_image",
                    metadata={"source": "sample_data"}
                )
                
                print(f"    图片注入: {image_ref} (关联: {relation_id})")
        
        # 统计结果
        doc_count = await vector_store.get_document_count()
        image_count = await vector_store.get_image_count()
        relation_count = await get_relation_count(vector_store)
        
        result = {
            "status": "success",
            "injected": {
                "chapters": len(chapters),
                "images": total_images,
                "relations": total_images  # 每个图片都有一个关联
            },
            "actual_counts": {
                "documents": doc_count,
                "images": image_count,
                "relations": relation_count
            },
            "chapter_ids": chapter_ids
        }
        
        print(f"数据注入完成!")
        print(f"  实际文档数: {doc_count}")
        print(f"  实际图片数: {image_count}")
        print(f"  实际关联数: {relation_count}")
        
        return result
        
    except Exception as e:
        error_msg = f"数据注入失败: {str(e)}"
        print(f" {error_msg}")
        import traceback
        traceback.print_exc()
        
        return {
            "status": "failed",
            "error": error_msg,
            "traceback": traceback.format_exc()
        }


async def init_vector_store() -> PostgreSQLVectorStore:
    """
    初始化并连接向量数据库
    
    返回:
        PostgreSQLVectorStore: 已连接的向量存储实例
    
    异常:
        Exception: 连接失败时抛出异常
    """
    vector_store = PostgreSQLVectorStore()
    await vector_store.connect()
    return vector_store


async def close_vector_store(vector_store: PostgreSQLVectorStore) -> None:
    """
    关闭数据库连接
    
    参数:
        vector_store: 已连接的向量存储实例
    """
    await vector_store.close()


async def get_relation_count(vector_store: PostgreSQLVectorStore) -> int:
    """
    获取文本-图片关联总数
    
    参数:
        vector_store: 已连接的向量存储实例
    
    返回:
        int: 关联数量
    """
    async with vector_store.pool.acquire() as conn:
        relation_count = await conn.fetchval("SELECT COUNT(*) FROM text_image_relations")
    return relation_count


async def get_data_counts(vector_store: PostgreSQLVectorStore) -> Tuple[int, int, int]:
    """
    获取文档、图片、关联的数量统计
    
    参数:
        vector_store: 已连接的向量存储实例
    
    返回:
        tuple: (文档数, 图片数, 关联数)
    """
    doc_count = await vector_store.get_document_count()
    image_count = await vector_store.get_image_count()
    relation_count = await get_relation_count(vector_store)
    return doc_count, image_count, relation_count


async def get_chapter_data(
    vector_store: PostgreSQLVectorStore,
    limit: Optional[int] = None,
    category: str = "chapter_title"
) -> List[Dict[str, Any]]:
    """
    查询章节数据
    
    参数:
        vector_store: 已连接的向量存储实例
        limit: 返回结果数量限制，None表示返回全部
        category: 章节数据的分类标识
    
    返回:
        List[Dict]: 章节数据列表，包含title、image_count、created_at等字段
    """
    query = """
        SELECT doc_metadata->>'title' as title, 
               doc_metadata->>'image_count' as image_count,
               created_at
        FROM documents 
        WHERE doc_metadata->>'category' = $1
        ORDER BY created_at
    """
    
    params = [category]
    
    if limit is not None:
        query += " LIMIT $2"
        params.append(limit)
    
    async with vector_store.pool.acquire() as conn:
        chapters = await conn.fetch(query, *params)
    
    # 转换为字典列表
    return [dict(chapter) for chapter in chapters]


async def get_image_data(
    vector_store: PostgreSQLVectorStore,
    limit: Optional[int] = None,
    category: str = "chapter_image"
) -> List[Dict[str, Any]]:
    """
    查询图片数据
    
    参数:
        vector_store: 已连接的向量存储实例
        limit: 返回结果数量限制，None表示返回全部
        category: 图片数据的分类标识
    
    返回:
        List[Dict]: 图片数据列表，包含image_ref、chapter_title、created_at等字段
    """
    query = """
        SELECT image_metadata->>'image_ref' as image_ref,
               image_metadata->>'chapter_title' as chapter_title,
               created_at
        FROM image_descriptions 
        WHERE image_metadata->>'category' = $1
        ORDER BY created_at
    """
    
    params = [category]
    
    if limit is not None:
        query += " LIMIT $2"
        params.append(limit)
    
    async with vector_store.pool.acquire() as conn:
        images = await conn.fetch(query, *params)
    
    # 转换为字典列表
    return [dict(image) for image in images]


async def get_relation_data(
    vector_store: PostgreSQLVectorStore,
    limit: Optional[int] = None,
    order_by_similarity: bool = True
) -> List[Dict[str, Any]]:
    """
    查询文本-图片关联数据
    
    参数:
        vector_store: 已连接的向量存储实例
        limit: 返回结果数量限制，None表示返回全部
        order_by_similarity: 是否按相似度降序排列
    
    返回:
        List[Dict]: 关联数据列表，包含chapter_title、image_ref、similarity_score等字段
    """
    query = """
        SELECT d.doc_metadata->>'title' as chapter_title,
               i.image_metadata->>'image_ref' as image_ref,
               r.similarity_score
        FROM text_image_relations r
        JOIN documents d ON r.document_id = d.id
        JOIN image_descriptions i ON r.image_id = i.id
    """
    
    if order_by_similarity:
        query += " ORDER BY r.similarity_score DESC"
    
    if limit is not None:
        query += " LIMIT $1"
        params = [limit]
    else:
        params = []
    
    async with vector_store.pool.acquire() as conn:
        relations = await conn.fetch(query, *params)
    
    # 转换为字典列表
    return [dict(relation) for relation in relations]


async def verify_data_injection(
    min_doc_count: int = 9,
    min_image_count: int = 66,
    min_relation_count: int = 66,
    chapter_limit: int = None,
    image_limit: int = 10,
    relation_limit: int = 5,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    验证数据注入完整性
    
    参数:
        min_doc_count: 文档数量最小值
        min_image_count: 图片数量最小值
        min_relation_count: 关联数量最小值
        chapter_limit: 章节数据返回数量限制
        image_limit: 图片数据返回数量限制
        relation_limit: 关联数据返回数量限制
        verbose: 是否打印验证过程
    
    返回:
        Dict: 验证结果字典，包含数量统计、数据列表、验证状态等
    """
    vector_store = None
    try:
        if verbose:
            print("="*60)
            print("验证数据注入")
            print("="*60)
        
        # 1. 连接数据库
        if verbose:
            print("1. 连接数据库...")
        vector_store = await init_vector_store()
        if verbose:
            print("   数据库连接成功")
        
        # 2. 统计数据数量
        if verbose:
            print("2. 检查文档数量...")
        doc_count = await vector_store.get_document_count()
        if verbose:
            print(f"   文档数量: {doc_count}")
        
        if verbose:
            print("3. 检查图片数量...")
        image_count = await vector_store.get_image_count()
        if verbose:
            print(f"   图片数量: {image_count}")
        
        if verbose:
            print("4. 检查关联数量...")
        relation_count = await get_relation_count(vector_store)
        if verbose:
            print(f"   关联数量: {relation_count}")
        
        # 3. 查询章节数据
        if verbose:
            print("5. 检查章节数据...")
        chapters = await get_chapter_data(vector_store, limit=chapter_limit)
        if verbose:
            print(f"   找到 {len(chapters)} 个章节:")
            for i, chapter in enumerate(chapters, 1):
                print(f"      {i}. {chapter['title']} (图片: {chapter['image_count']})")
        
        # 4. 查询图片数据
        if verbose:
            print("6. 检查图片数据...")
        images = await get_image_data(vector_store, limit=image_limit)
        if verbose:
            print(f"   找到 {len(images)} 个图片 (显示前{image_limit}个):")
            for i, image in enumerate(images, 1):
                print(f"      {i}. {image['image_ref']} (章节: {image['chapter_title']})")
        
        # 5. 查询关联数据
        if verbose:
            print("7. 检查关联数据...")
        relations = await get_relation_data(vector_store, limit=relation_limit)
        if verbose:
            print(f"   找到 {len(relations)} 个关联 (显示前{relation_limit}个):")
            for i, relation in enumerate(relations, 1):
                print(f"      {i}. {relation['chapter_title']} <-> {relation['image_ref']} (相似度: {relation['similarity_score']:.2f})")
        
        # 6. 验证数据完整性
        injection_success = (
            doc_count >= min_doc_count and
            image_count >= min_image_count and
            relation_count >= min_relation_count
        )
        
        # 7. 输出总结
        if verbose:
            print("\n" + "="*60)
            print("数据注入验证完成!")
            print("="*60)
            print(f"\n总结:")
            print(f"  - 章节: {doc_count} 个 (要求≥{min_doc_count})")
            print(f"  - 图片: {image_count} 个 (要求≥{min_image_count})")
            print(f"  - 关联: {relation_count} 个 (要求≥{min_relation_count})")
            
            if injection_success:
                print("\n✅ 数据注入成功!")
            else:
                print("\n⚠️  数据注入可能不完整")
        
        # 组装验证结果
        result = {
            "counts": {
                "documents": doc_count,
                "images": image_count,
                "relations": relation_count
            },
            "data": {
                "chapters": chapters,
                "images": images,
                "relations": relations
            },
            "validation": {
                "min_requirements": {
                    "documents": min_doc_count,
                    "images": min_image_count,
                    "relations": min_relation_count
                },
                "injection_success": injection_success
            },
            "status": "success"
        }
        
        return result
        
    except Exception as e:
        error_msg = f"数据注入验证失败: {str(e)}"
        if verbose:
            print(f"\n❌ {error_msg}")
            traceback.print_exc()
        
        return {
            "status": "failed",
            "error": error_msg,
            "traceback": traceback.format_exc()
        }
    
    finally:
        # 确保关闭连接
        if vector_store:
            await close_vector_store(vector_store)
            if verbose:
                print("   数据库连接关闭")


async def batch_verify_data_quality(
    vector_store: PostgreSQLVectorStore,
    check_empty_fields: bool = True,
    check_similarity_range: bool = True
) -> Dict[str, Any]:
    """
    批量验证数据质量（扩展功能）
    
    参数:
        vector_store: 已连接的向量存储实例
        check_empty_fields: 是否检查空字段
        check_similarity_range: 是否检查相似度范围（0-1）
    
    返回:
        Dict: 数据质量检查结果
    """
    quality_issues = []
    
    # 检查空字段
    if check_empty_fields:
        async with vector_store.pool.acquire() as conn:
            # 检查文档空标题
            empty_title_docs = await conn.fetchval("""
                SELECT COUNT(*) FROM documents 
                WHERE doc_metadata->>'title' IS NULL OR doc_metadata->>'title' = ''
            """)
            if empty_title_docs > 0:
                quality_issues.append(f"发现 {empty_title_docs} 个文档标题为空")
            
            # 检查图片空引用
            empty_ref_images = await conn.fetchval("""
                SELECT COUNT(*) FROM image_descriptions 
                WHERE image_metadata->>'image_ref' IS NULL OR image_metadata->>'image_ref' = ''
            """)
            if empty_ref_images > 0:
                quality_issues.append(f"发现 {empty_ref_images} 个图片引用为空")
    
    # 检查相似度范围
    if check_similarity_range:
        async with vector_store.pool.acquire() as conn:
            invalid_similarity = await conn.fetchval("""
                SELECT COUNT(*) FROM text_image_relations 
                WHERE similarity_score < 0 OR similarity_score > 1
            """)
            if invalid_similarity > 0:
                quality_issues.append(f"发现 {invalid_similarity} 个关联的相似度值超出0-1范围")
    
    return {
        "quality_check_passed": len(quality_issues) == 0,
        "issues_found": quality_issues,
        "checks_performed": {
            "empty_fields": check_empty_fields,
            "similarity_range": check_similarity_range
        }
    }


async def business_usage_example():
    """
    业务场景使用示例
    """
    # 1. 基础数据注入验证
    verify_result = await verify_data_injection(
        min_doc_count=10,
        min_image_count=70,
        min_relation_count=70,
        image_limit=15,
        relation_limit=8
    )
    
    if verify_result["status"] == "success":
        print(f"\n验证结果: {'通过' if verify_result['validation']['injection_success'] else '未通过'}")
        
        # 2. 如果基础验证通过，进行数据质量检查
        if verify_result['validation']['injection_success']:
            vector_store = await init_vector_store()
            quality_result = await batch_verify_data_quality(vector_store)
            await close_vector_store(vector_store)
            
            print(f"\n数据质量检查:")
            print(f"  - 是否通过: {quality_result['quality_check_passed']}")
            if quality_result['issues_found']:
                print(f"  - 发现问题:")
                for issue in quality_result['issues_found']:
                    print(f"    * {issue}")


# ========== 全局实例和主函数 ==========
# 全局向量存储实例
vector_store = PostgreSQLVectorStore()

# 主函数入口
if __name__ == "__main__":
    # 示例调用：使用默认参数验证
    result = asyncio.run(verify_data_injection())
    
    if result["status"] == "success" and result["validation"]["injection_success"]:
        print("\n下一步:")
        print("1. 启动API服务器: python -m src.api.main")
        print("2. 测试检索功能")
        print("3. 开始设计模型A")
    else:
        print("\n数据注入验证失败，请检查数据导入流程")
