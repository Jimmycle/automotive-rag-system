import logging
import re
from typing import List, Dict, Any

import chromadb
from llama_index.core import StorageContext, VectorStoreIndex, Document, Settings, load_index_from_storage
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.retrievers import BaseRetriever
from llama_index.vector_stores.chroma import ChromaVectorStore

from config import Config
from src.utils.embedding_models import get_ollama_embedding

logger = logging.getLogger(__name__)

class VectorStoreManager:
    """向量数据库管理器"""

    def __init__(self, persist_db_dir: str, persist_storage_dir: str):
        self.persist_db_dir = persist_db_dir
        self.persist_storage_dir = persist_storage_dir
        self.client = chromadb.PersistentClient(path=persist_db_dir)
        self._indices = {}  # 缓存索引对象

    def create_collection(self, collection_name: str,force_recreate: bool = False) -> StorageContext:
        """创建并获取集合"""
        Settings.embed_model = get_ollama_embedding()
        # 如果强制重建，先删除现有集合
        if force_recreate:
            try:
                self.client.delete_collection(collection_name)
                logger.info(f"已删除集合: {collection_name}")
            except Exception as e:
                logger.warning(f"删除集合时出错（可能集合不存在）: {e}")

        chroma_collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"description": "汽车行业知识库"}
        )
        vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
        return StorageContext.from_defaults(vector_store=vector_store)
        # return storage_context

    def text_splitter(self, chunk_size: int = Config.CHUNK_SIZE, chunk_overlap: int = Config.CHUNK_OVERLAP):
        """文本分块器"""
        sentence_splitter = SentenceSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            paragraph_separator="\n\n",
            secondary_chunking_regex="[^,.;。]+[,.;。]?",
        )
        return sentence_splitter

    def add_documents_to_collection(self, collection_name: str, documents: List[Document], force_recreate: bool = False) -> VectorStoreIndex:
        """添加文档到向量数据库，支持强制重建集合"""
        storage_context = self.create_collection(collection_name, force_recreate)
        logger.info(f"初始化向量数据库管理器，文档数量: {len(documents)}")

        # 为原始文档生成ID
        import hashlib
        for doc in documents:
            doc_id = hashlib.md5(doc.text.encode('utf-8')).hexdigest()
            doc.id_ = doc_id

        # 检查已存在的文档（通过docstore）
        try:
            existing_doc_ids = set(storage_context.docstore.docs.keys())
        except Exception as e:
            logger.warning(f"获取现有文档ID时出错: {e}")
            existing_doc_ids = set()
        logger.info(f"已存在的文档ID: {existing_doc_ids}")

        # 过滤掉已存在的文档（使用文档ID）
        new_documents = [doc for doc in documents if doc.id_ not in existing_doc_ids]
        logger.info(f"新文档数量: {len(new_documents)}")
        if new_documents:
            logger.info(f"发现 {len(new_documents)} 个新文档，正在添加到docstore...")
            # 正确添加原始文档到docstore（关键修改）
            storage_context.docstore.add_documents(new_documents)

            logger.info("正在分块文档...")
            nodes = self.text_splitter().get_nodes_from_documents(new_documents)

            logger.info("正在构建索引...")
            index = VectorStoreIndex(
                nodes,
                storage_context=storage_context,
                show_progress=True,
                store_nodes_override=True
            )
        else:
            logger.info("没有新文档，加载现有向量数据库索引...")
            # 注意：VectorStoreIndex.from_vector_store会自动使用storage_context中的docstore
            index = VectorStoreIndex.from_vector_store(
                vector_store=storage_context.vector_store,
                storage_context=storage_context
            )
        docs = index.storage_context.docstore.docs
        logger.info(f"成功处理文档，当前存储文档数量: {len(docs)}")
        self._indices[collection_name] = index
        return index

    def get_retriever(self, collection_name: str, similarity_top_k: int = 10) -> BaseRetriever:
        """获取检索器"""
        if collection_name not in self._indices:
            # 如果索引不存在，尝试从向量存储加载
            storage_context = self.create_collection(collection_name)
            vector_store: ChromaVectorStore = storage_context.vector_store
            index = VectorStoreIndex.from_vector_store(vector_store, storage_context=storage_context)
            self._indices[collection_name] = index

        index = self._indices[collection_name]
        return index.as_retriever(similarity_top_k=similarity_top_k)

    def search_similar_documents(self, collection_name: str, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """搜索相似文档 - 适配 MultiPathRetriever"""
        retriever = self.get_retriever(collection_name, similarity_top_k=top_k)
        nodes = retriever.retrieve(query)

        similar_docs = []
        for node in nodes:
            similar_docs.append({
                "content": node.text,
                "metadata": node.metadata,
                "score": node.score,  # 相似度分数
                "id": node.node_id,
                "distance": 1 - node.score if node.score is not None else None  # 转换为距离
            })

        logger.info(f"检索到 {len(similar_docs)} 个相关文档")
        return similar_docs

    def get_corpus(self, collection_name: str) -> List[str]:
        """获取语料库文本列表 - 用于 BM25 检索"""
        if collection_name not in self._indices:
            self.get_retriever(collection_name)  # 确保索引已加载

        index = self._indices[collection_name]
        corpus = []
        # 从文档存储中获取所有文档的文本内容
        for doc_id, doc in index.storage_context.docstore.docs.items():
            if hasattr(doc, 'text'):
                corpus.append(doc.text)
        logger.info(f"获取到语料库，包含 {len(corpus)} 个文档块")
        return corpus


class AutomotiveTextProcessor:
    """汽车行业文本处理器"""

    def preprocess_text(self, text: str) -> str:
        """文本预处理"""
        # 移除多余的空格和换行
        text = re.sub(r'\s+', ' ', text)
        return text.strip()

    def clean_documents(self, documents: List[Document]) -> List[Document]:
        """清理数据"""
        cleaned_docs = []
        for doc in documents:
            # 创建新的文档对象，避免修改原始对象
            cleaned_text = self.preprocess_text(doc.text)
            cleaned_doc = Document(
                text=cleaned_text,
                metadata=doc.metadata.copy() if doc.metadata else {},
                doc_id=doc.doc_id
            )
            cleaned_docs.append(cleaned_doc)

        logger.info(f"清理了 {len(documents)} 个文档")
        return cleaned_docs



