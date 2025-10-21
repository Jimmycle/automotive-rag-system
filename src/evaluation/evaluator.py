import json
import pandas as pd
from typing import List, Dict, Any
import logging
from sklearn.metrics import precision_score, recall_score, f1_score
import numpy as np

logger = logging.getLogger(__name__)


class QAEvaluator:
    """问答系统评估器"""

    def __init__(self, qa_system):
        self.qa_system = qa_system

    def load_evaluation_data(self, file_path: str) -> List[Dict[str, Any]]:
        """加载评估数据"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            return data
        except Exception as e:
            logger.error(f"加载评估数据失败: {str(e)}")
            return []

    def evaluate_single_question(self, question: str, ground_truth: str) -> Dict[str, Any]:
        """评估单个问题"""
        result = self.qa_system.generate_answer(question)

        # 这里可以添加更复杂的评估逻辑
        # 比如使用另一个LLM来评估答案质量
        evaluation = {
            "question": question,
            "generated_answer": result["answer"],
            "ground_truth": ground_truth,
            "confidence": result["confidence"],
            "retrieved_docs_count": len(result["retrieved_documents"]),
            "retrieval_scores": [score for _, score in result["retrieved_documents"]]
        }

        return evaluation

    def evaluate_batch(self, test_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """批量评估"""
        results = []

        for item in test_data:
            evaluation = self.evaluate_single_question(
                item["question"],
                item.get("answer", "")
            )
            results.append(evaluation)

        # 计算总体指标
        avg_confidence = np.mean([r["confidence"] for r in results])
        avg_retrieved_docs = np.mean([r["retrieved_docs_count"] for r in results])

        metrics = {
            "total_questions": len(results),
            "average_confidence": avg_confidence,
            "average_retrieved_docs": avg_retrieved_docs,
            "evaluation_results": results
        }

        return metrics

    def compare_without_rag(self, questions: List[str]) -> pd.DataFrame:
        """对比有RAG和无RAG的效果"""
        comparison_results = []

        for question in questions:
            # 有RAG的回答
            rag_result = self.qa_system.generate_answer(question)

            # 无RAG的回答（直接问模型）
            # 这里需要实现直接调用LLM的逻辑

            comparison_results.append({
                "question": question,
                "rag_answer": rag_result["answer"],
                "rag_confidence": rag_result["confidence"],
                # "direct_answer": direct_result,
                # 可以添加人工评估分数等
            })

        return pd.DataFrame(comparison_results)