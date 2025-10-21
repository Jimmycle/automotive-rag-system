import logging
from typing import List, Dict, Any

from langchain_core.messages import HumanMessage, SystemMessage

from src.retrieval.multi_retriever import MultiPathRetriever
from src.utils.llm_models import LLMManager

logger = logging.getLogger(__name__)


class AutomotiveQASystem:
    """汽车行业问答系统"""

    def __init__(self, retriever: MultiPathRetriever, model_type: str, model_name: str):
        self.retriever = retriever
        self.llm = LLMManager(
            model_type=model_type,
            model_name=model_name,
            temperature=0.6
        ).get_llm()
        # 系统提示词
        self.system_prompt = """你是一个专业的汽车行业专家助手。请根据提供的上下文信息，准确、专业地回答用户的问题。
        
            回答要求：
            1. 基于提供的上下文信息回答，不要编造不知道的信息
            2. 如果上下文信息不足，请明确说明哪些方面缺乏信息
            3. 使用专业、准确的汽车行业术语
            4. 回答要结构清晰，重点突出
            5. 如果涉及技术参数或规格，请确保数据准确
            
            请开始回答用户的问题："""

    def generate_answer(self, question: str, max_retrieved_docs: int = 5) -> Dict[str, Any]:
        """生成答案"""
        try:
            # 检索相关文档
            retrieval_result = self.retriever.retrieve_with_confidence(
                question, max_retrieved_docs
            )

            # 准备上下文
            context = self._prepare_context(retrieval_result["documents"])

            # 生成回答
            answer = self._call_llm(question, context)

            return {
                "question": question,
                "answer": answer,
                "retrieved_documents": retrieval_result["documents"],
                "confidence": retrieval_result["average_confidence"],
                "context_used": context,
                "retrieval_metadata": {
                    "method": retrieval_result["retrieval_method"],
                    "documents_count": len(retrieval_result["documents"])
                }
            }

        except Exception as e:
            logger.error(f"生成答案时出错: {str(e)}")
            return {
                "question": question,
                "answer": f"抱歉，回答问题时出现错误: {str(e)}",
                "retrieved_documents": [],
                "confidence": 0.0,
                "context_used": "",
                "error": str(e)
            }

    def _prepare_context(self, documents: List[tuple]) -> str:
        """准备上下文"""
        if not documents:
            return "没有找到相关的上下文信息。"

        context_parts = []
        for i, (doc_content, score) in enumerate(documents, 1):
            context_parts.append(f"[文档{i} - 相关度: {score:.3f}]\n{doc_content}\n")

        return "\n".join(context_parts)

    def _call_llm(self, question: str, context: str) -> str:
        """调用语言模型"""
        messages = [
            SystemMessage(content=self.system_prompt),
            HumanMessage(content=f"上下文信息:\n{context}\n\n用户问题: {question}")
        ]

        response = self.llm.invoke(messages)
        return response.content

    def batch_answer(self, questions: List[str]) -> List[Dict[str, Any]]:
        """批量回答问题"""
        results = []
        for question in questions:
            result = self.generate_answer(question)
            results.append(result)
        return results
