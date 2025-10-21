# import logging
# from typing import List, Optional, Dict, Any
# from llama_index.core import VectorStoreIndex, Settings
# from llama_index.core.node_parser import SentenceSplitter
# from llama_index.core.query_engine import RetrieverQueryEngine
# from llama_index.core.retrievers import VectorIndexRetriever
# from llama_index.core.schema import Document, TextNode
# from llama_index.vector_stores.milvus import MilvusVectorStore
#
# from config import Config
# from src.utils.embedding_models import get_ollama_embedding, get_qwen_embedding
#
# logger = logging.getLogger(__name__)
#
#
# class VectorStoreManager:
#     def __init__(
#             self,
#             collection_name: str = Config.COLLECTION_NAME,
#             milvus_uri: str = "http://localhost:19530",
#             chunk_size: Optional[int] = None,
#             chunk_overlap: Optional[int] = None,
#             **kwargs
#     ):
#         """
#         初始化 Milvus 向量存储管理器
#
#         Args:
#             collection_name: Milvus 集合名称
#             milvus_uri: Milvus 连接地址
#             embedding_model: 嵌入模型名称
#             chunk_size: 文本分块大小
#             chunk_overlap: 分块重叠大小
#         """
#         self.collection_name = collection_name
#         self.chunk_size = chunk_size
#         self.chunk_overlap = chunk_overlap
#
#         # 初始化嵌入模型
#         Settings.embed_model = get_ollama_embedding()
#         # Settings.embed_model = get_qwen_embedding()
#         # 初始化 Milvus 向量存储
#         self.vector_store = MilvusVectorStore(
#             uri=milvus_uri,
#             collection_name=collection_name,
#             dim=1024,  # 自动获取维度
#             overwrite=False,  # 不覆盖已有集合
#             # 关键参数：避免内部再次尝试异步连接
#             **{k: v for k, v in kwargs.items() if k != 'disable_auto_connect'}
#         )
#
#         self.index = None
#         self.retriever = None
#         self.query_engine = None
#
#         logger.info(f"MilvusVectorStore 初始化完成，集合: {collection_name}")
#
#     def load_and_chunk_documents(
#             self,
#             documents: List[Document],
#             chunk_size: Optional[int] = None,
#             chunk_overlap: Optional[int] = None
#     ) -> List[TextNode]:
#         """
#         加载文档并进行分块处理
#
#         Args:
#             documents: 文档列表
#             chunk_size: 分块大小，如果为None则使用初始化参数
#             chunk_overlap: 分块重叠大小，如果为None则使用初始化参数
#
#         Returns:
#             List[TextNode]: 分块后的节点列表
#         """
#         chunk_size = chunk_size or self.chunk_size
#         chunk_overlap = chunk_overlap or self.chunk_overlap
#
#         # 创建文本分块器
#         text_splitter = SentenceSplitter(
#             chunk_size=chunk_size,
#             chunk_overlap=chunk_overlap,
#             paragraph_separator="\n\n",
#             secondary_chunking_regex="[^,.;。]+[,.;。]?",
#         )
#         # # 方法2：基于语义的分块
#         # semantic_splitter = SemanticSplitterNodeParser(
#         #     buffer_size=1,
#         #     breakpoint_percentile_threshold=95,
#         #     embed_model=OpenAIEmbedding()
#         # )
#         #
#         # # 方法3：基于token的分块（适合LLM上下文限制）
#         # token_splitter = TokenTextSplitter(
#         #     chunk_size=chunk_size,
#         #     chunk_overlap=chunk_overlap,
#         #     separator=" ",
#         # )
#
#         # 将文档分割成节点
#         nodes = text_splitter.get_nodes_from_documents(documents)
#         logger.info(f"文档分块完成: {len(documents)} 个文档 -> {len(nodes)} 个节点")
#         return nodes
#
#     def create_index_from_documents(
#             self,
#             documents: List[Document],
#             chunk_size: Optional[int] = None,
#             chunk_overlap: Optional[int] = None,
#             overwrite: bool = False
#     ) -> VectorStoreIndex:
#         """
#         从文档创建向量索引
#
#         Args:
#             documents: 文档列表
#             chunk_size: 分块大小
#             chunk_overlap: 分块重叠大小
#             overwrite: 是否覆盖现有集合
#
#         Returns:
#             VectorStoreIndex: 创建的向量索引
#         """
#         if overwrite:
#             self.vector_store.collection.drop()
#             logger.info(f"已删除现有集合: {self.collection_name}")
#
#         # 分块处理文档
#         nodes = self.load_and_chunk_documents(documents, chunk_size, chunk_overlap)
#         print(f"nodes ${nodes}")
#
#         # 创建向量索引
#         self.index = VectorStoreIndex(
#             nodes=nodes,
#             vector_store=self.vector_store,
#             show_progress=True
#         )
#
#         logger.info(f"向量索引创建完成，包含 {len(nodes)} 个节点")
#
#         return self.index
#
#     def add_documents(
#             self,
#             documents: List[Document],
#             chunk_size: Optional[int] = None,
#             chunk_overlap: Optional[int] = None
#     ) -> int:
#         """
#         向现有索引添加文档
#
#         Args:
#             documents: 要添加的文档列表
#             chunk_size: 分块大小
#             chunk_overlap: 分块重叠大小
#
#         Returns:
#             int: 添加的节点数量
#         """
#         if self.index is None:
#             # 如果索引不存在，先创建
#             self.create_index_from_documents(documents, chunk_size, chunk_overlap)
#             return len(documents)
#
#         # 分块处理新文档
#         nodes = self.load_and_chunk_documents(documents, chunk_size, chunk_overlap)
#
#         # 向现有索引插入节点
#         self.index.insert_nodes(nodes)
#
#         logger.info(f"成功添加 {len(nodes)} 个节点到现有索引")
#
#         return len(nodes)
#
#     def initialize_retriever(
#             self,
#             similarity_top_k: int = 5,
#             **retriever_kwargs
#     ) -> VectorIndexRetriever:
#         """
#         初始化检索器
#
#         Args:
#             similarity_top_k: 返回最相似的前K个结果
#             **retriever_kwargs: 其他检索器参数
#
#         Returns:
#             VectorIndexRetriever: 检索器实例
#         """
#         if self.index is None:
#             raise ValueError("索引未创建，请先调用 create_index_from_documents 或 add_documents")
#
#         self.retriever = VectorIndexRetriever(
#             index=self.index,
#             similarity_top_k=similarity_top_k,
#             **retriever_kwargs
#         )
#
#         logger.info(f"检索器初始化完成，similarity_top_k={similarity_top_k}")
#
#         return self.retriever
#
#     def initialize_query_engine(
#             self,
#             similarity_top_k: int = 5,
#             **query_engine_kwargs
#     ) -> RetrieverQueryEngine:
#         """
#         初始化查询引擎
#
#         Args:
#             similarity_top_k: 返回最相似的前K个结果
#             **query_engine_kwargs: 其他查询引擎参数
#
#         Returns:
#             RetrieverQueryEngine: 查询引擎实例
#         """
#         if self.retriever is None:
#             self.initialize_retriever(similarity_top_k=similarity_top_k)
#
#         self.query_engine = RetrieverQueryEngine(
#             retriever=self.retriever,
#             **query_engine_kwargs
#         )
#
#         logger.info("查询引擎初始化完成")
#
#         return self.query_engine
#
#     def search(
#             self,
#             query: str,
#             similarity_top_k: Optional[int] = None,
#             use_query_engine: bool = False
#     ) -> Any:
#         """
#         搜索相似内容
#
#         Args:
#             query: 查询文本
#             similarity_top_k: 返回结果数量
#             use_query_engine: 是否使用完整的查询引擎（带LLM）
#
#         Returns:
#             搜索结果
#         """
#         if use_query_engine:
#             if self.query_engine is None:
#                 self.initialize_query_engine()
#             return self.query_engine.query(query)
#         else:
#             if self.retriever is None:
#                 if similarity_top_k:
#                     self.initialize_retriever(similarity_top_k=similarity_top_k)
#                 else:
#                     self.initialize_retriever()
#
#             return self.retriever.retrieve(query)
#
#     def get_collection_info(self) -> Dict[str, Any]:
#         """
#         获取集合信息
#
#         Returns:
#             Dict[str, Any]: 集合信息
#         """
#         if not self.vector_store.collection:
#             return {"error": "集合未初始化"}
#
#         collection = self.vector_store.collection
#         return {
#             "collection_name": collection.name,
#             "num_entities": collection.num_entities,
#             "schema": str(collection.schema),
#             "indexes": [index.params for index in collection.indexes] if collection.indexes else []
#         }
#
#     def delete_collection(self) -> bool:
#         """
#         删除集合
#
#         Returns:
#             bool: 是否删除成功
#         """
#         try:
#             self.vector_store.collection.drop()
#             logger.info(f"集合 {self.collection_name} 已删除")
#             self.index = None
#             self.retriever = None
#             self.query_engine = None
#             return True
#         except Exception as e:
#             logger.error(f"删除集合失败: {e}")
#             return False
#
#     def close(self):
#         """关闭连接"""
#         # Milvus 客户端连接通常会自动管理，这里可以添加必要的清理工作
#         pass
