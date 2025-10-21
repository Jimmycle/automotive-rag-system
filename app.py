import os
import sys

import plotly.graph_objects as go
import streamlit as st
import pandas as pd

from src.qa_system import AutomotiveQASystem
from src.retrieval.multi_retriever import MultiPathRetriever
from src.utils.text_processor import VectorStoreManager

# 添加项目路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import config, Config
from src.utils.data_loader import AutomotiveLoader


class AutomotiveQAApp:
    """汽车行业问答应用"""

    def __init__(self):
        self.setup_page()
        self.init_session_state()
        # 手动连接

    def setup_page(self):
        """设置页面配置"""
        st.set_page_config(
            page_title="汽车行业专业知识问答系统",
            page_icon="🚗",
            layout="wide",
            initial_sidebar_state="expanded"
        )
        st.title("🚗 汽车行业专业知识问答系统")
        st.markdown("基于RAG技术的专业汽车知识问答平台")

    def init_session_state(self):
        """初始化会话状态"""
        if 'qa_system' not in st.session_state:
            st.session_state.qa_system = None
        if 'vector_manager' not in st.session_state:
            st.session_state.vector_manager = None
        if 'knowledge_loaded' not in st.session_state:
            st.session_state.knowledge_loaded = False
        if 'chat_history' not in st.session_state:
            st.session_state.chat_history = []
        if 'comparison_results' not in st.session_state:
            st.session_state.comparison_results = []

    def load_knowledge_base(self):
        """加载知识库"""
        st.sidebar.header("📚 知识库管理")
        if st.sidebar.button("🔄 初始化知识库", type="primary"):
            with st.spinner("正在加载知识库，这可能需要几分钟..."):
                try:
                    # 加载文档
                    data_loader = AutomotiveLoader(config.RAW_DATA_DIR)
                    documents = data_loader.load_data()

                    # 创建向量数据库
                    vector_store = VectorStoreManager(persist_db_dir=config.VECTOR_DB_PATH, persist_storage_dir=config.VECTOR_STORAGE_PATH)
                    index = vector_store.add_documents_to_collection(
                        collection_name=config.COLLECTION_NAME,
                        documents=documents,
                    )
                    # 初始化检索器和问答系统
                    multi_retriever = MultiPathRetriever(vector_store, vector_store.get_corpus(config.COLLECTION_NAME))
                    qa_system = AutomotiveQASystem(
                        retriever=multi_retriever,
                        model_type=Config.MODEL_TYPE_QWEN,
                        model_name=Config.QWEN3_MAX_MODEL
                    )
                    # 保存到会话状态
                    st.session_state.vector_store = vector_store
                    st.session_state.qa_system = qa_system
                    st.session_state.knowledge_loaded = True
                    st.success(f"✅ 知识库加载完成！共处理 {len(documents)} 个文档，{len(index.storage_context.docstore.docs)} 个文本块")
                except Exception as e:
                    st.error(f"加载知识库失败: {str(e)}")

    def chat_interface(self):
        """聊天界面"""
        st.header("💬 专业问答")

        # 显示聊天历史
        for chat in st.session_state.chat_history:
            with st.chat_message("user"):
                st.write(chat["question"])
            with st.chat_message("assistant"):
                st.write(chat["answer"])
                if chat.get("confidence"):
                    st.progress(chat["confidence"], text=f"置信度: {chat['confidence']:.2%}")

        # 用户输入
        if prompt := st.chat_input("请输入您的汽车行业问题..."):
            # 添加用户消息
            with st.chat_message("user"):
                st.write(prompt)

            # 生成回答
            if st.session_state.qa_system:
                with st.chat_message("assistant"):
                    with st.spinner("正在检索信息并生成回答..."):
                        result = st.session_state.qa_system.generate_answer(prompt)

                        st.write(result["answer"])
                        st.progress(result["confidence"], text=f"置信度: {result['confidence']:.2%}")

                        # 显示检索信息
                        with st.expander("📄 查看检索到的参考文档"):
                            for i, (doc, score) in enumerate(result["retrieved_documents"]):
                                st.markdown(f"**文档 {i + 1} (相关度: {score:.3f})**")
                                st.text(doc[:200] + "..." if len(doc) > 200 else doc)
                                st.divider()

                # 保存到聊天历史
                st.session_state.chat_history.append({
                    "question": prompt,
                    "answer": result["answer"],
                    "confidence": result["confidence"]
                })
            else:
                st.error("请先初始化知识库")

    def generate_direct_answer(self, question: str) -> str:
        """生成无RAG的直接回答"""
        try:
            # 这里实现直接调用LLM的逻辑，不使用检索
            # 使用相同的LLM，但不提供检索的上下文
            from langchain_core.messages import HumanMessage, SystemMessage

            system_prompt = """你是一个专业的汽车行业专家助手。请根据你的知识准确、专业地回答用户的问题。

            回答要求：
            1. 使用专业、准确的汽车行业术语
            2. 回答要结构清晰，重点突出
            3. 如果涉及技术参数或规格，请确保数据准确
            4. 如果不知道确切信息，请明确说明

            请开始回答用户的问题："""

            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=f"用户问题: {question}")
            ]

            response = st.session_state.qa_system.llm.invoke(messages)
            return response.content
        except Exception as e:
            return f"生成直接回答时出错: {str(e)}"

    def compare_rag_vs_direct(self, question: str):
        """对比有RAG和无RAG的回答"""
        if not st.session_state.qa_system:
            st.error("请先初始化知识库")
            return

        with st.spinner("正在生成对比分析..."):
            # 有RAG的回答
            rag_result = st.session_state.qa_system.generate_answer(question)

            # 无RAG的回答
            direct_answer = self.generate_direct_answer(question)

            # 计算评估指标（简化版）
            rag_answer_length = len(rag_result["answer"])
            direct_answer_length = len(direct_answer)

            # 评估指标（这里可以添加更复杂的评估逻辑）
            rag_score = self.evaluate_answer_quality(rag_result["answer"], question)
            direct_score = self.evaluate_answer_quality(direct_answer, question)

            # 保存对比结果
            comparison_result = {
                "question": question,
                "rag_answer": rag_result["answer"],
                "direct_answer": direct_answer,
                "rag_confidence": rag_result["confidence"],
                "rag_retrieved_docs": len(rag_result["retrieved_documents"]),
                "rag_answer_length": rag_answer_length,
                "direct_answer_length": direct_answer_length,
                "rag_quality_score": rag_score,
                "direct_quality_score": direct_score
            }

            st.session_state.comparison_results.append(comparison_result)

            return comparison_result

    def evaluate_answer_quality(self, answer: str, question: str) -> float:
        """评估回答质量（简化版）"""
        # 这里可以实现更复杂的质量评估逻辑
        # 目前使用简单的启发式规则

        score = 0.5  # 基础分

        # 根据回答长度加分
        if len(answer) > 100:
            score += 0.1
        if len(answer) > 200:
            score += 0.1

        # 根据专业术语加分
        professional_terms = ["发动机", "变速箱", "底盘", "悬挂", "制动", "新能源", "电动车", "混动"]
        found_terms = sum(1 for term in professional_terms if term in answer)
        score += min(found_terms * 0.05, 0.2)

        # 根据结构加分（包含数字、列表等）
        if any(char.isdigit() for char in answer):
            score += 0.1
        if "、" in answer or ";" in answer or "：" in answer:
            score += 0.1

        return min(score, 1.0)

    def evaluation_interface(self):
        """评估界面"""
        st.header("📊 系统评估")

        # 使用选项卡组织不同的评估功能
        tab1, tab2, tab3 = st.tabs(["🔍 检索效果测试", "🔄 RAG对比测试", "📈 性能指标"])

        with tab1:
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("检索效果测试")
                test_question = st.text_input("测试问题", key="retrieval_test")
                if st.button("测试检索效果"):
                    if st.session_state.qa_system:
                        result = st.session_state.qa_system.generate_answer(test_question)
                        # 显示检索结果分析
                        if result["retrieved_documents"]:
                            fig = go.Figure(data=[
                                go.Bar(name='相关度分数',
                                       x=[f"文档{i + 1}" for i in range(len(result["retrieved_documents"]))],
                                       y=[score for _, score in result["retrieved_documents"]])
                            ])
                            fig.update_layout(title="检索文档相关度分布")
                            st.plotly_chart(fig)
                        else:
                            st.warning("未检索到相关文档")

            with col2:
                st.subheader("性能指标")
                if st.button("生成评估报告"):
                    # 这里可以添加自动评估逻辑
                    st.info("评估功能开发中...")

        with tab2:
            st.subheader("RAG与无RAG对比测试")

            # 测试问题输入
            col1, col2 = st.columns([3, 1])
            with col1:
                compare_question = st.text_input("输入对比测试问题",
                                                 placeholder="请输入一个汽车行业相关的问题...",
                                                 key="compare_question")
            with col2:
                st.write("")  # 占位
                st.write("")
                run_comparison = st.button("🚀 运行对比测试", width='stretch')

            if run_comparison and compare_question:
                result = self.compare_rag_vs_direct(compare_question)

                if result:
                    # 显示对比结果
                    st.success("对比测试完成！")

                    # 使用列布局展示结果
                    col1, col2 = st.columns(2)

                    with col1:
                        st.markdown("### 🔗 有RAG回答")
                        st.info(f"**置信度:** {result['rag_confidence']:.3f}")
                        st.info(f"**检索文档数:** {result['rag_retrieved_docs']}")
                        st.info(f"**回答质量分:** {result['rag_quality_score']:.3f}")
                        st.markdown("**回答内容:**")
                        st.write(result["rag_answer"])

                        with st.expander("查看检索文档"):
                            rag_full_result = st.session_state.qa_system.generate_answer(compare_question)
                            for i, (doc, score) in enumerate(rag_full_result["retrieved_documents"]):
                                st.markdown(f"**文档 {i + 1} (相关度: {score:.3f})**")
                                st.text(doc[:300] + "..." if len(doc) > 300 else doc)
                                st.divider()

                    with col2:
                        st.markdown("### 🎯 无RAG回答")
                        st.info(f"**回答质量分:** {result['direct_quality_score']:.3f}")
                        st.info(f"**回答长度:** {result['direct_answer_length']} 字符")
                        st.markdown("**回答内容:**")
                        st.write(result["direct_answer"])

                    # 显示评估指标对比
                    st.markdown("---")
                    st.subheader("📊 评估指标对比")

                    metrics_data = {
                        "指标": ["置信度", "回答质量", "回答长度", "检索文档数"],
                        "有RAG": [
                            f"{result['rag_confidence']:.3f}",
                            f"{result['rag_quality_score']:.3f}",
                            f"{result['rag_answer_length']}",
                            f"{result['rag_retrieved_docs']}"
                        ],
                        "无RAG": [
                            "N/A",
                            f"{result['direct_quality_score']:.3f}",
                            f"{result['direct_answer_length']}",
                            "N/A"
                        ]
                    }

                    metrics_df = pd.DataFrame(metrics_data)
                    st.dataframe(metrics_df,  width='stretch', hide_index=True)

                    # 质量分数对比图
                    quality_data = {
                        "类型": ["有RAG", "无RAG"],
                        "质量分数": [result['rag_quality_score'], result['direct_quality_score']]
                    }
                    quality_df = pd.DataFrame(quality_data)

                    fig = go.Figure(data=[
                        go.Bar(name='质量分数', x=quality_df['类型'], y=quality_df['质量分数'])
                    ])
                    fig.update_layout(title="回答质量对比", yaxis_range=[0, 1])
                    st.plotly_chart(fig)

            # 显示历史对比记录
            if st.session_state.comparison_results:
                st.markdown("---")
                st.subheader("历史对比记录")

                history_data = []
                for i, comp in enumerate(st.session_state.comparison_results[-5:]):  # 只显示最近5条
                    history_data.append({
                        "序号": i + 1,
                        "问题": comp["question"][:50] + "..." if len(comp["question"]) > 50 else comp["question"],
                        "RAG质量分": f"{comp['rag_quality_score']:.3f}",
                        "无RAG质量分": f"{comp['direct_quality_score']:.3f}",
                        "质量差异": f"{(comp['rag_quality_score'] - comp['direct_quality_score']):.3f}"
                    })

                if history_data:
                    history_df = pd.DataFrame(history_data)
                    st.dataframe(history_df,  width='stretch', hide_index=True)

                    # 清空历史记录按钮
                    if st.button("清空对比记录"):
                        st.session_state.comparison_results = []
                        st.rerun()

        with tab3:
            st.subheader("系统性能指标")
            if st.button("生成详细评估报告"):
                # 这里可以添加更全面的评估逻辑
                st.info("详细评估报告功能开发中...")

                # 模拟一些性能指标
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("平均响应时间", "2.3s", "-0.2s")
                with col2:
                    st.metric("检索准确率", "87%", "3%")
                with col3:
                    st.metric("用户满意度", "92%", "5%")

    def run(self):
        """运行应用"""
        # 侧边栏
        self.load_knowledge_base()
        # 主界面标签页
        tab1, tab2, tab3 = st.tabs(["💬 问答", "📊 评估", "ℹ️ 关于"])
        with tab1:
            self.chat_interface()
        with tab2:
            self.evaluation_interface()
        with tab3:
            st.header("关于本项目")
            # st.markdown("""
            # ### 基于RAG的汽车行业专业知识问答系统
            #
            # **技术栈:**
            # - LangChain: 框架集成
            # - Chroma: 向量数据库
            # - Sentence Transformers: 文本嵌入
            # - OpenAI GPT: 答案生成
            # - Streamlit: Web界面
            #
            # **核心功能:**
            # - 多路检索（密集检索 + 关键词检索）
            # - 重排序优化
            # - 置信度评估
            # - 实时问答
            #
            # **数据源:**
            # - 汽车技术手册
            # - 维修指南
            # - 行业报告
            # - 技术文档
            # """)


if __name__ == "__main__":
    app = AutomotiveQAApp()
    app.run()