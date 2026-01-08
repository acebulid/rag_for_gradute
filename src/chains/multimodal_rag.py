# LangChain集成模块
import logging
from typing import List, Optional, Dict, Any
from dataclasses import dataclass
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.llms.base import BaseLLM
from langchain.embeddings.base import Embeddings
from langchain.schema import Document

# 日志配置
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# ===================== 核心提示词常量（可统一修改） =====================
CAMPUS_GUIDE_PROMPT_TEMPLATE = """你是一个专业的校园导览助手。基于以下上下文信息，回答用户的问题。

上下文信息：
{context}

用户问题：{question}

请根据上下文信息提供准确、详细的回答。如果上下文信息不足以回答问题，请诚实地说明。
回答时请引用相关的上下文信息，并确保回答清晰、有用。

回答："""
# ======================================================================

@dataclass
class LangChainRAGConfig:
    """ RAG配置"""
    chain_type: str = "stuff"
    temperature: float = 0.7
    top_p: float = 0.9

class MultimodalRAGChain:
    """ 多模态RAG链"""
    def __init__(self, retriever, ollama_service, config: LangChainRAGConfig = None):
        self.retriever = retriever
        self.ollama_service = ollama_service
        self.config = config or LangChainRAGConfig()
        self.prompt = PromptTemplate(
            template=CAMPUS_GUIDE_PROMPT_TEMPLATE,  # 使用提示词常量
            input_variables=["context", "question"]
        )

    async def query(self, question: str, image_path: Optional[str] = None, top_k: int = 5) -> Dict[str, Any]:
        """ 查询方法"""
        # 执行检索
        rag_response = await self.retriever.retrieve_with_rag(
            type('obj', (object,), {
                "text_query": question,
                "image_path": image_path,
                "top_k": top_k
            })()
        )

        # 构建结果
        return {
            "query": question,
            "answer": rag_response.generated_answer,
            "source_documents": [
                Document(
                    page_content=res.content,
                    metadata={"id": res.id, "score": res.score, **res.metadata}
                ) for res in rag_response.retrieval_response.hybrid_results
            ]
        }

    async def batch_query(self, questions: List[str], image_paths: Optional[List[str]] = None, top_k: int = 5) -> List[Dict[str, Any]]:
        """ 批量查询"""
        import asyncio
        image_paths = image_paths or [None] * len(questions)
        tasks = [self.query(q, p, top_k) for q, p in zip(questions, image_paths)]
        return await asyncio.gather(*tasks)

class OllamaLLM(BaseLLM):
    """ Ollama LLM包装器"""
    def __init__(self, ollama_service, model: str = "qwen2.5:7b", **kwargs):
        super().__init__(**kwargs)
        self.ollama_service = ollama_service
        self.model = model
        self.temperature = kwargs.get("temperature", 0.7)

    def _call(self, prompt: str, **kwargs) -> str:
        """同步调用"""
        import asyncio
        return asyncio.run(
            self.ollama_service.generate_text(
                prompt=prompt,
                model=self.model,
                options={"temperature": self.temperature, "top_p": self.top_p}
            )
        ).text

    @property
    def _llm_type(self) -> str:
        return "ollama"

class OllamaEmbeddings(Embeddings):
    """ Ollama嵌入包装器"""
    def __init__(self, ollama_service, model: str = "bge-m3"):
        self.ollama_service = ollama_service
        self.model = model

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        import asyncio
        results = asyncio.run(self.ollama_service.batch_generate_embeddings(texts, self.model))
        return [r.embedding for r in results]

    def embed_query(self, text: str) -> List[float]:
        import asyncio
        return asyncio.run(self.ollama_service.generate_embedding(text, self.model)).embedding

def create_langchain_rag_chain(retriever, ollama_service, config: LangChainRAGConfig = None):
    """ 创建RAG链"""
    import asyncio
    config = config or LangChainRAGConfig()
    
    # 创建LLM
    llm = OllamaLLM(ollama_service, temperature=config.temperature)
    
    # 创建RAG链（复用核心提示词常量）
    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type=config.chain_type,
        retriever=type('obj', (object,), {
            "get_relevant_documents": lambda q, **kw: asyncio.run(retriever.retrieve(q, **kw))
        })(),
        chain_type_kwargs={"prompt": PromptTemplate(
            template=CAMPUS_GUIDE_PROMPT_TEMPLATE,  # 复用提示词常量
            input_variables=["context", "question"]
        )}
    )

#  创建函数
async def create_multimodal_rag_chain():
    """ 创建RAG链实例"""
    from src.services.retrieval_service import create_retriever
    from src.services.ollama_service import OllamaService
    
    ollama_service = OllamaService()
    retriever = await create_retriever()
    return MultimodalRAGChain(retriever, ollama_service)