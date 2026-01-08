from sqlalchemy import Column, String, Text, Float, DateTime, ForeignKey
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.sql import func
import uuid

Base = declarative_base()


class Document(Base):
    """文档表"""
    __tablename__ = "documents"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    content = Column(Text, nullable=False)
    doc_metadata = Column(JSONB, nullable=False, default=dict)  # 重命名，避免与SQLAlchemy保留字冲突
    embedding = Column(String, nullable=False)  # pgvector存储为字符串
    source = Column(String, nullable=True)  # 数据来源
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    
    def __repr__(self):
        return f"<Document(id={self.id}, content_length={len(self.content)})>"


class ImageDescription(Base):
    """图像描述表"""
    __tablename__ = "image_descriptions"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    image_path = Column(String, nullable=False)
    vlm_description = Column(Text, nullable=False)
    embedding = Column(String, nullable=False)  # pgvector存储为字符串
    image_metadata = Column(JSONB, nullable=False, default=dict)  # 重命名，避免与SQLAlchemy保留字冲突
    image_size = Column(String, nullable=True)  # 图像尺寸，如"1920x1080"
    file_format = Column(String, nullable=True)  # 文件格式，如"jpg", "png"
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    def __repr__(self):
        return f"<ImageDescription(id={self.id}, image_path={self.image_path})>"


class TextImageRelation(Base):
    """文本-图像关联表"""
    __tablename__ = "text_image_relations"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    document_id = Column(UUID(as_uuid=True), ForeignKey("documents.id", ondelete="CASCADE"))
    image_id = Column(UUID(as_uuid=True), ForeignKey("image_descriptions.id", ondelete="CASCADE"))
    similarity_score = Column(Float, nullable=False)
    relation_type = Column(String, nullable=True)  # 关联类型，如"描述", "对应", "相关"
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    def __repr__(self):
        return f"<TextImageRelation(doc={self.document_id}, img={self.image_id}, score={self.similarity_score})>"


class QueryHistory(Base):
    """查询历史表"""
    __tablename__ = "query_history"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    query_text = Column(Text, nullable=True)
    query_image_path = Column(String, nullable=True)
    query_type = Column(String, nullable=False)  # "text", "image", "hybrid"
    retrieved_document_ids = Column(JSONB, nullable=False, default=list)
    retrieved_image_ids = Column(JSONB, nullable=False, default=list)
    response = Column(Text, nullable=True)
    response_time_ms = Column(Float, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    def __repr__(self):
        return f"<QueryHistory(id={self.id}, type={self.query_type})>"


# 导出所有模型
__all__ = ["Base", "Document", "ImageDescription", "TextImageRelation", "QueryHistory"]