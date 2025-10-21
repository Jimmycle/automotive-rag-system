from typing import List

from llama_index.core.base.base_retriever import BaseRetriever
from llama_index.core.schema import BaseNode, TextNode

from src.utils.text_processor import VectorStoreManager


class LlamaIndexMultiPathRetriever(BaseRetriever):
    """适配 LlamaIndex 的多路检索器"""

    def __init__(self, vector_store_manager: VectorStoreManager, collection_name: str):
        self.vector_store_manager = vector_store_manager
        self.collection_name = collection_name

        # 获取语料库
        self.corpus = self.vector_store_manager.get_corpus(collection_name)

        # 初始化多路检索器
        from multi_retriever import MultiPathRetriever
        self.multi_retriever = MultiPathRetriever(vector_store_manager, self.corpus)

        super().__init__()

    def _retrieve(self, query: str) -> List[BaseNode]:
        """实现 LlamaIndex 检索接口"""
        results = self.multi_retriever.retrieve_with_confidence(query)

        nodes = []
        for i, (doc_content, score) in enumerate(results["documents"]):
            node = TextNode(
                text=doc_content,
                metadata={
                    "score": score,
                    "confidence": results["confidence_scores"][i] if i < len(results["confidence_scores"]) else score,
                    "retrieval_method": results["retrieval_method"]
                }
            )
            nodes.append(node)

        return nodes
