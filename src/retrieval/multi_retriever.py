import logging
from typing import List, Dict, Any, Tuple

import numpy as np
from llama_index.core.schema import TextNode, NodeWithScore, QueryBundle
from llama_index.postprocessor.flag_embedding_reranker import FlagEmbeddingReranker
from rank_bm25 import BM25Okapi

from src.utils.text_processor import VectorStoreManager
from rank_bm25 import BM25Okapi
from FlagEmbedding import FlagReranker

logger = logging.getLogger(__name__)


class MultiPathRetriever:
    """多路检索器"""

    def __init__(self, vector_store_manager: VectorStoreManager, corpus: List[str]):
        self.vector_store = vector_store_manager
        self.corpus = corpus

        # 初始化BM25
        tokenized_corpus = [self._tokenize(doc) for doc in corpus]
        self.bm25 = BM25Okapi(tokenized_corpus)

        # 初始化中文重排模型
        try:
            self.reranker = FlagEmbeddingReranker('BAAI/bge-reranker-base', use_fp16=True)
            self.has_reranker = True
            logger.info("成功加载中文重排模型")
        except Exception as e:
            logger.warning(f"加载重排模型失败: {e}，将使用简单重排")
            self.has_reranker = False
            # self.simple_reranker = LightweightReranker()

    def _tokenize(self, text: str) -> List[str]:
        """简单的分词函数"""
        return text.lower().split()

    def dense_retrieval(self, query: str, top_k: int = 10) -> List[Tuple[str, float]]:
        """密集检索"""
        results = self.vector_store.search_similar_documents(
            "automotive_knowledge", query, top_k * 2
        )

        scored_results = []
        for result in results:
            # 使用1-距离作为分数（距离越小，相似度越高）
            score = 1 - result['distance'] if result['distance'] is not None else 0.5
            scored_results.append((result['content'], score))

        return scored_results[:top_k]

    def sparse_retrieval(self, query: str, top_k: int = 10) -> List[Tuple[str, float]]:
        """稀疏检索（BM25）"""
        tokenized_query = self._tokenize(query)
        doc_scores = self.bm25.get_scores(tokenized_query)

        # 获取top_k结果
        top_indices = np.argsort(doc_scores)[::-1][:top_k]

        results = []
        for idx in top_indices:
            if doc_scores[idx] > 0:
                results.append((self.corpus[idx], float(doc_scores[idx])))

        return results

    def hybrid_retrieval(
            self,
            query: str,
            top_k: int = 10,
            dense_weight: float = 0.7,
            sparse_weight: float = 0.3
    ) -> List[Tuple[str, float]]:
        """混合检索"""
        # 执行两种检索
        dense_results = self.dense_retrieval(query, top_k * 2)
        sparse_results = self.sparse_retrieval(query, top_k * 2)
        # 创建结果字典
        result_scores = {}
        # 添加密集检索结果
        for content, score in dense_results:
            normalized_score = self._normalize_score(score, method='minmax')
            result_scores[content] = dense_weight * normalized_score
        # 添加稀疏检索结果
        for content, score in sparse_results:
            normalized_score = self._normalize_score(score, method='minmax')
            if content in result_scores:
                result_scores[content] += sparse_weight * normalized_score
            else:
                result_scores[content] = sparse_weight * normalized_score
        # 排序并返回top_k结果
        sorted_results = sorted(result_scores.items(), key=lambda x: x[1], reverse=True)
        return sorted_results[:top_k]

    def _normalize_score(self, score: float, method: str = 'minmax') -> float:
        """分数归一化"""
        if method == 'minmax':
            # 简单的min-max归一化到[0,1]
            return min(max(score, 0), 1)
        return score

    def rerank_documents(self, query: str, documents: List[Tuple[str, float]], top_k: int = 5) -> List[Tuple[str, float]]:
        """重排序文档"""
        if not documents:
            return []

        # 1. 将文档转换为 NodeWithScore
        nodes_with_score = []
        for doc_content, orig_score in documents:
            node = TextNode(text=doc_content)
            node_with_score = NodeWithScore(node=node, score=orig_score)
            nodes_with_score.append(node_with_score)

        # 2. 创建 QueryBundle
        query_bundle = QueryBundle(query_str=query)

        # 3. 使用官方 Reranker 进行重排序
        # 注意：postprocess_nodes 返回的是排序后的 NodeWithScore 列表，按相关性降序
        try:
            reranked_nodes: List[NodeWithScore] = self.reranker.postprocess_nodes(
                nodes=nodes_with_score,
                query_bundle=query_bundle
            )
        except Exception as e:
            logger.error(f"重排序失败: {e}")
            # 失败时返回原始结果
            return documents[:top_k]

        # 4. 提取重排序分数（注意：reranker 输出的 score 是相关性分数，通常越大越好）
        reranked_results = []
        for node_with_score in reranked_nodes:
            doc_content = node_with_score.node.get_content()
            rerank_score = node_with_score.score  # 这是 reranker 给出的相关性分数

            # 找到原始分数（可能需要映射，这里假设顺序或内容唯一）
            # 更安全的方式：用内容做 key（如果内容太长可哈希）
            orig_score = next((score for content, score in documents if content == doc_content), 0.0)

            # 融合分数（示例：加权平均）
            combined_score = 0.3 * orig_score + 0.7 * rerank_score
            reranked_results.append((doc_content, combined_score))

        # 5. 按综合分数排序并返回 top_k
        reranked_results.sort(key=lambda x: x[1], reverse=True)
        return reranked_results[:top_k]
        # """重排序文档"""
        # if not documents:
        #     return []
        #
        # # 准备重排序数据
        # pairs = [(query, doc[0]) for doc in documents]
        #
        # # 执行重排序
        # # rerank_scores = self.reranker.predict(pairs)
        # rerank_scores = self.reranker.postprocess_nodes(nodes, query_bundle)
        #
        # # 组合原始分数和重排序分数
        # reranked_results = []
        # for (doc_content, orig_score), rerank_score in zip(documents, rerank_scores):
        #     # 组合分数（可以调整权重）
        #     combined_score = 0.3 * orig_score + 0.7 * rerank_score
        #     reranked_results.append((doc_content, combined_score))
        #
        # # 按新分数排序
        # reranked_results.sort(key=lambda x: x[1], reverse=True)
        # return reranked_results[:top_k]

    def retrieve_with_confidence(self, query: str, top_k: int = 5) -> Dict[str, Any]:
        """带置信度评估的检索"""
        # 混合检索
        hybrid_results = self.hybrid_retrieval(query, top_k * 3)
        # 重排序
        final_results = self.rerank_documents(query, hybrid_results, top_k)

        # 计算置信度
        confidence_scores = []
        for content, score in final_results:
            confidence = self._calculate_confidence(content, query, score)
            confidence_scores.append(confidence)

        # 返回结果和元数据
        return {
            "documents": final_results,
            "confidence_scores": confidence_scores,
            "average_confidence": np.mean(confidence_scores) if confidence_scores else 0,
            "retrieval_method": "hybrid_with_reranking"
        }

    def _calculate_confidence(self, document: str, query: str, score: float) -> float:
        """计算置信度"""
        # 1. 相似度分数，基于多个因素计算置信度
        factors = [score]

        # 2. 查询词覆盖度
        query_terms = set(self._tokenize(query))
        doc_terms = set(self._tokenize(document))
        coverage = len(query_terms.intersection(doc_terms)) / len(query_terms) if query_terms else 0
        factors.append(coverage)

        # 3. 文档长度因子（适中的长度通常更好）
        doc_length = len(document.split())
        length_factor = 1 - abs(doc_length - 200) / 500  # 假设200词左右最优
        factors.append(max(0, min(1, length_factor)))

        return np.mean(factors)
