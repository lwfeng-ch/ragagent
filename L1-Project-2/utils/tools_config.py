# tools_config.py - LangChain v1 迁移重构版本
# =============================================================================
# 要求：Python 3.10+
# 安装：
#   pip install -U langchain>=1.0 langchain-chroma
#   pip install -U langchain-classic  # 如果 create_retriever_tool 在 v1 中被移除
# =============================================================================

import logging
from typing import Any
# === LangChain v1 导入 ===
from langchain.tools import tool

# [v1 变更] create_retriever_tool 导入路径
# v1 中 retrievers 相关功能已移至 langchain-classic
# 但 create_retriever_tool 在 langchain_core.tools 中仍有定义（向后兼容）
# 优先尝试从 langchain_core.tools 导入，失败则从 langchain-classic 导入
try:
    from langchain_core.tools.retriever import create_retriever_tool
except ImportError:
    try:
        from langchain.tools.retriever import create_retriever_tool
    except ImportError:
        from langchain_classic.tools.retriever import create_retriever_tool

# [v1 新特性] ToolRuntime 允许工具访问 agent 状态和上下文
# 用于需要读取对话历史或 context 配置的高级工具
from langchain.tools import ToolRuntime
# [v1 新特性] Command 允许工具更新 agent 状态
from langgraph.types import Command
from langchain_chroma import Chroma
from config import Config

logger = logging.getLogger(__name__)


def _create_retriever_tool(llm_embedding) -> Any:
    """创建 Chroma 检索工具。

    Args:
        llm_embedding: 嵌入模型实例

    Returns:
        检索工具实例
    """
    try:
        # 创建 Chroma 向量存储实例
        vectorstore = Chroma(
            persist_directory=Config.CHROMADB_DIRECTORY,
            collection_name=Config.CHROMADB_COLLECTION_NAME,
            embedding_function=llm_embedding,
        )
        # 将向量存储转换为检索器
        retriever = vectorstore.as_retriever()

        # 创建检索工具
        # [v1 说明] create_retriever_tool 返回的是 BaseTool 实例，
        # 与 create_agent 的 tools 参数完全兼容
        retriever_tool = create_retriever_tool(
            retriever,
            name="retrieve",
            description="这是健康档案查询工具，搜索并返回有关用户的健康档案信息。",
        )
        logger.info("Retriever tool created successfully")
        return retriever_tool

    except Exception as e:
        logger.error(f"Failed to create retriever tool: {e}")
        raise


def get_tools(llm_embedding) -> list:
    """创建并返回工具列表。

    [v1 工具兼容性说明]:
    create_agent 的 tools 参数接受：
    - LangChain BaseTool 实例（@tool 装饰的函数）     ← multiply
    - create_retriever_tool 返回的 Tool               ← retriever_tool
    - Callable 对象（带类型提示和 docstring 的函数）
    - dict（表示内置 provider 工具）
    不再接受 ToolNode 实例。

    [v1 @tool 装饰器新特性]:
    1. ToolRuntime：工具可通过 runtime 参数访问 agent 状态
       @tool
       def my_tool(query: str, runtime: ToolRuntime) -> str:
           messages = runtime.state["messages"]  # 访问对话历史
           user_id = runtime.context.get("user_id")  # 访问调用上下文
    2. Command：工具可通过返回 Command 更新 agent 状态
       return Command(update={"key": "value"})
    3. 类型提示是必需的，因为它们定义了工具的输入 schema

    Args:
        llm_embedding: 嵌入模型实例

    Returns:
        list: 工具列表
    """
    # 创建检索工具
    retriever_tool = _create_retriever_tool(llm_embedding)

    @tool
    def multiply(a: float, b: float) -> float:
        """这是计算两个数的乘积的工具，返回最终的计算结果。

        Args:
            a: 第一个数字
            b: 第二个数字
        """
        return a * b

    # =================================================================
    # [v1 新特性示例] 使用 ToolRuntime 访问 agent 状态的高级工具
    # 取消注释以启用
    # =================================================================

    @tool
    def get_conversation_summary(runtime: ToolRuntime) -> str:
        """获取当前对话的摘要信息。
    
        此工具可以访问完整的对话历史来生成摘要。
        runtime 参数对模型不可见，由框架自动注入。
        """
        from langchain_core.messages import HumanMessage
        messages = runtime.state.get("messages", [])
        human_msgs = [m for m in messages if isinstance(m, HumanMessage)]
        return f"对话中共有 {len(messages)} 条消息，其中用户消息 {len(human_msgs)} 条。"

    @tool
    def get_user_context(query: str, runtime: ToolRuntime) -> str:
        """根据用户上下文信息回答问题。
    
        Args:
            query: 用户的查询内容
        """
        # 通过 runtime.context 访问调用时传入的不可变配置
        user_id = runtime.context.get("user_id", "unknown")
        return f"用户 {user_id} 的查询: {query}"


    # [v1 新特性示例] 使用 Command 更新 agent 状态的工具
    # 取消注释以启用（需要在 state 中定义对应字段 + reducer）

    @tool
    def set_user_preference(key: str, value: str) -> Command:
        """设置用户偏好。
    
        Args:
            key: 偏好名称
            value: 偏好值
        """
        return Command(update={"user_preferences": {key: value}})

    # 返回工具列表
    tools = [retriever_tool, multiply, get_conversation_summary, get_user_context, set_user_preference]
    logger.info(f"Tools initialized: {[t.name for t in tools]}")
    return tools