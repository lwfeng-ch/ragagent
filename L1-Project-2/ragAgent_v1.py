# ragAgent.py - LangChain v1 Middleware 集成完整版
import logging
from concurrent_log_handler import ConcurrentRotatingFileHandler
import os
import sys
import threading
import time
import uuid
from html import escape
from typing import Literal, Annotated, Sequence, Optional, Any, Callable
from typing_extensions import TypedDict, NotRequired
from collections import defaultdict
from dotenv import load_dotenv
load_dotenv()

# === LangChain Core ===
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain_core.messages import BaseMessage, AIMessage, HumanMessage, ToolMessage, SystemMessage
from langchain_core.runnables import RunnableConfig

# === LangChain v1 Agent + Middleware ===
from langchain.agents import create_agent
from langchain.agents.middleware import (
    # 基类和类型
    AgentMiddleware,
    AgentState,
    ModelRequest,
    ModelResponse,
    ExtendedModelResponse,
    # 装饰器
    before_agent,
    after_agent,
    before_model,
    after_model,
    wrap_model_call,
    wrap_tool_call,
    dynamic_prompt,
    # 内置中间件
    SummarizationMiddleware,
    HumanInTheLoopMiddleware,
    PIIMiddleware,
    ModelCallLimitMiddleware,
    ToolCallLimitMiddleware,
    ToolRetryMiddleware,
)

# === LangGraph ===
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, START, END
from langgraph.store.base import BaseStore
from langgraph.store.postgres import PostgresStore
from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.runtime import Runtime

# === 数据库与工具 ===
from concurrent.futures import ThreadPoolExecutor, as_completed
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from psycopg2 import OperationalError
from psycopg_pool import ConnectionPool
from pydantic import BaseModel, Field
from utils.llms_v1 import get_llm
from utils.tools_config import get_tools
from utils.config import Config

# 日志配置
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.handlers = []

handler = ConcurrentRotatingFileHandler(
    Config.LOG_FILE,
    maxBytes=Config.MAX_BYTES,
    backupCount=Config.BACKUP_COUNT
)
handler.setLevel(logging.DEBUG)
handler.setFormatter(logging.Formatter(
    "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
))
logger.addHandler(handler)

# 状态类型定义
class MessagesState(TypedDict):
    """对话状态类型定义"""
    messages: Annotated[Sequence[BaseMessage], add_messages]
    relevance_score: Annotated[Optional[str], "Relevance score: 'yes' or 'no'"]
    rewrite_count: Annotated[int, "Number of query rewrites"]


# v1 消息内容安全提取
def get_message_text(message: BaseMessage) -> str:
    """v1 兼容：安全提取消息的文本内容（支持 standard content blocks）"""
    content = message.content
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        text_parts = []
        for block in content:
            if isinstance(block, dict):
                if block.get("type") == "text" and "text" in block:
                    text_parts.append(block["text"])
                elif "text" in block:
                    text_parts.append(block["text"])
            elif isinstance(block, str):
                text_parts.append(block)
        return "\n".join(text_parts) if text_parts else str(content)
    return str(content)


# 工具配置管理类
class ToolConfig:
    """管理工具及其路由配置"""
    def __init__(self, tools):
        self.tools = tools
        self.tool_names = {tool.name for tool in tools}
        self.tool_routing_config = self._build_routing_config(tools)
        logger.info(f"Initialized ToolConfig: {self.tool_names}")

    def _build_routing_config(self, tools):
        routing_config = {}
        for tool in tools:
            tool_name = tool.name.lower()
            if "retrieve" in tool_name:
                routing_config[tool_name] = "grade_documents"
            else:
                routing_config[tool_name] = "generate"
        return routing_config

    def get_tools(self):
        return self.tools

    def get_tool_names(self):
        return self.tool_names

    def get_tool_routing_config(self):
        return self.tool_routing_config


# Pydantic 模型
class DocumentRelevanceScore(BaseModel):
    """文档相关性评分"""
    binary_score: Literal["yes", "no"]

class ConnectionPoolError(Exception):
    """数据库连接池异常"""
    pass



# ======================== 中间件定义区域 ============================
def _get_user_id(runtime: Runtime) -> str:
    """安全获取 user_id，处理 context 为 None 的情况。"""
    if runtime.context is None:
        return "unknown"
    return runtime.context.get("user_id", "unknown")


# 方式一：装饰器风格中间件（适合简单的单钩子逻辑）
@before_model
def log_before_model_call(state: AgentState, runtime: Runtime):
    messages = state.get("messages", [])
    msg_count = len(messages)

    user_id = _get_user_id(runtime)

    logger.info(
        f"[Middleware] user={user_id} "
        f"messages={msg_count}"
    )


@after_model
def log_after_model_call(state: AgentState, runtime: Runtime):
    last_msg = state["messages"][-1] if state.get("messages") else None
    if not last_msg:
        return None

    text = get_message_text(last_msg)
    preview = text[:100]

    logger.info(
        f"[Middleware] "
        f"user={_get_user_id(runtime)} "
        f"output={preview}"
    )


@before_agent
def log_agent_start(state: AgentState, runtime: Runtime):
    messages = state.get("messages", [])
    question = messages[-1].content if messages else ""

    user_id = _get_user_id(runtime)

    logger.info(
        f"[Agent Start] user={user_id} "
        f"question={question[:100]}"
    )


@after_agent
def log_agent_end(state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
    """agent 调用结束时执行（每次 invoke 只执行一次）"""
    msg_count = len(state.get("messages", []))
    user_id = _get_user_id(runtime)

    logger.info(
        f"[Agent End] user={user_id} "
        f"messages={msg_count}"
    )


# wrap 风格装饰器中间件（拦截实际的模型/工具调用）
# 模型调用失败时自动重试（最多3次）
@wrap_model_call
def retry_model_on_error(
    request: ModelRequest,
    handler: Callable[[ModelRequest], ModelResponse],
) -> ModelResponse:
    """模型调用失败时自动重试（最多3次）。

    wrap-style hook：拦截模型调用本身，可实现重试、缓存、降级等逻辑。
    handler 可被多次调用以实现重试。
    """
    max_retries = 3
    for attempt in range(max_retries):
        try:
            # 调用模型
            return handler(request)
        except Exception as e:
            if attempt == max_retries - 1:
                logger.error(f"[Middleware] Model call failed after {max_retries} retries: {e}")
                raise
            logger.warning(f"[Middleware] Model call retry {attempt + 1}/{max_retries}: {e}")


# 给每一次工具执行加上 日志 + 执行时间统计 + 错误记录
@wrap_tool_call
def log_tool_execution(request, handler):
    """记录每次工具调用的执行过程。

    wrap_tool_call：拦截每个工具调用，可用于监控、错误处理、限流等。
    """
    tool_name = request.tool_call.get("name", "unknown")
    logger.info(f"[Middleware] Executing tool: {tool_name}")
    start_time = time.time()
    try:
        result = handler(request)
        elapsed = time.time() - start_time
        logger.info(f"[Middleware] Tool {tool_name} completed in {elapsed:.2f}s")
        return result
    except Exception as e:
        elapsed = time.time() - start_time
        logger.error(f"[Middleware] Tool {tool_name} failed after {elapsed:.2f}s: {e}")
        raise


# 方式三：类风格中间件（适合复杂的多钩子 + 带配置的中间件）
# 记忆管理中间件：在模型调用前，从 memory store 中检索用户记忆并注入到状态。
class MemoryMiddleware(AgentMiddleware):
    """自定义记忆管理中间件：在模型调用前，从 memory store 中检索用户记忆并注入到状态。

    类风格中间件优势：
    - 可以同时定义多个 hook（before_model + after_model 等）
    - 可以携带状态、参数或配置（如 store 实例）
    - 可以定义 state_schema 扩展 agent 状态
    """

    # 扩展状态：添加自定义字段
    class MemoryState(AgentState):
        user_info: NotRequired[str]

    state_schema = MemoryState

    def __init__(self, store: BaseStore, user_id: str = "default"):
        super().__init__()
        self.store = store
        self.user_id = user_id

    MAX_MEMORY = 200
    TOP_K = 3

    # 记忆检索中间件：在模型调用前，从 memory store 中检索用户记忆并注入到状态。
    def before_model(self, state: AgentState, runtime: Runtime):
        if state.get("user_info"):
            return None

        messages = state.get("messages", [])
        if not messages:
            return None

        last_msg = messages[-1]
        if last_msg.type != "human":
            return None

        query = get_message_text(last_msg)

        namespace = ("memories", self.user_id)

        memories = self.store.search(
            namespace,
            query=query,
            limit=self.TOP_K
        )

        user_info = "\n".join(
            [m.value["data"] for m in memories]
        )
        
        if user_info:
            return {"user_info": user_info}

    # 记忆存储中间件：在模型调用后，根据用户输入存储记忆。
    def after_agent(self, state: AgentState, runtime: Runtime):
        messages = state.get("messages", [])
        if not messages:
            return None

        namespace = ("memories", self.user_id)

        try:
            # 只检查最后一条用户消息
            last_msg = messages[-1]

            if not isinstance(last_msg, HumanMessage):
                return None

            text = get_message_text(last_msg)

            if "记住" not in text:
                return None

            memory = escape(text.replace("记住", "").strip())

            # 检查是否存在相同记忆
            existing = self.store.search(namespace,query=memory,limit=2)

            for m in existing:
                if m.value["data"] == memory:
                    return None

            self.store.put(namespace,str(uuid.uuid4()),{"data": memory})

            logger.info(f"[MemoryMiddleware] Stored memory: {memory[:50]}")

        except Exception as e:
            logger.error(f"[MemoryMiddleware] Error storing memory: {e}")
        return None

# 消息过滤中间件：在模型调用前过滤消息历史，控制上下文窗口。
class MessageFilterMiddleware(AgentMiddleware):
    """消息过滤中间件：在模型调用前过滤消息历史，控制上下文窗口。
    
    通过 modify_model_request 钩子在不改变全局状态的情况下，
    只修改传给模型的请求。
    """

    def __init__(self, max_messages: int = 10):
        super().__init__()
        self.max_messages = max_messages

    def modify_model_request(
        self,
        request: ModelRequest,
        runtime: Runtime,
    ) -> ModelRequest:
        """修改传给模型的消息列表（不影响全局状态）。

        modify_model_request 允许修改（仅针对该次模型请求）：
        - tools, prompt, message list, model, model settings, output format, tool choice
        """
        messages = request.messages
        
        system_msgs = [msg for msg in messages if isinstance(msg, SystemMessage)]
        # 保留系统消息和对话消息
        conversation_msgs = [msg for msg in messages if isinstance(msg, (AIMessage, HumanMessage, ToolMessage))]        

        # 截断到最大数量
        if len(conversation_msgs) > self.max_messages:
            conversation_msgs = conversation_msgs[-self.max_messages:]
            logger.info(f"[MessageFilterMiddleware] Truncated to {self.max_messages} messages")

        return request.override(messages=system_msgs + conversation_msgs)

"""
class DynamicModelMiddleware(AgentMiddleware):
    # 动态模型选择中间件：根据消息复杂度选择不同的模型。

    示例：简单问题用小模型，复杂问题用大模型。


    def __init__(self, simple_model: str, complex_model: str, threshold: int = 100):
        super().__init__()
        self.simple_model = simple_model
        self.complex_model = complex_model
        self.threshold = threshold

    def modify_model_request(
        self,
        request: ModelRequest,
        runtime: Runtime,
    ) -> ModelRequest:
    # 根据最新消息长度动态选择模型
        messages = request.messages
        if messages:
            last_text = get_message_text(messages[-1])
            if len(last_text) > self.threshold:
                logger.info(f"[DynamicModelMiddleware] Using complex model")
                return request.override(model=self.complex_model)
            else:
                logger.info(f"[DynamicModelMiddleware] Using simple model")
                return request.override(model=self.simple_model)
        return request
"""

# 输出审查中间件：在模型响应后检查输出内容。
class SafetyGuardMiddleware(AgentMiddleware):
    """ 输出审查中间件：在模型响应后检查输出内容。

    使用 after_model + jump_to 实现条件中断。
    """
    # 声明可以跳转到的节点
    # 需要配合 @hook_config(can_jump_to=["end"]) 或在类级别声明
    def __init__(self, blocked_keywords: list[str] | None = None):
        super().__init__()
        self.blocked_keywords = blocked_keywords or []

    def after_model(self, state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
        """检查模型输出中是否包含被禁止的关键词"""
        messages = state.get("messages", [])
        if not messages:
            return None

        last_msg = messages[-1]
        if isinstance(last_msg, AIMessage):
            text = get_message_text(last_msg)
            for keyword in self.blocked_keywords:
                if keyword.lower() in text.lower():
                    logger.warning(f"[SafetyGuardMiddleware] Blocked keyword detected: {keyword}")
                    return {
                        "messages": [AIMessage(content="抱歉，我无法提供此类信息。")],
                        # 跳转到 end 节点
                        # 可以根据需要修改为其他节点
                        "jump_to": "end"
                    }
        return None


# 方式四：dynamic_prompt 便利装饰器
@dynamic_prompt
def inject_user_context(request: ModelRequest) -> str:
    """动态生成 system prompt，注入用户上下文信息。

    @dynamic_prompt 是便利装饰器，返回的字符串会被添加到 system message 中。
    """
    user_info = request.state.get("user_info", "") if request.state else ""
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")

    prompt_parts = [f"当前时间：{timestamp}"]
    if user_info:
        lines = user_info.split("\n")[:5]
        user_info = "\n".join(lines)
        prompt_parts.append(f"已知用户信息：\n{user_info}")

    return "\n".join(prompt_parts)


# ====================== 并行工具节点（保持不变）=======================
class ParallelToolNode:
    """自定义并行工具执行节点"""
    
    def __init__(self, tools, max_workers: int | None = None):
        self.tools = tools
        tool_count = len(tools)
        cpu_limit = os.cpu_count() or 4
        self.max_workers = min(tool_count, cpu_limit * 2, 8) if max_workers is None else max_workers
        self._tool_map = {tool.name: tool for tool in tools}
        self.tool_fail_count = defaultdict(int)
        self.tool_disabled_until = {}
        self.max_failures = 3
        self.reset_timeout = 60

    def _run_single_tool(self, tool_call: dict) -> ToolMessage:
        try:
            tool_name = tool_call["name"]
            
            # 检查是否熔断
            disabled_until = self.tool_disabled_until.get(tool_name)
            if disabled_until and time.time() < disabled_until:
                return ToolMessage(
                    content=f"Tool {tool_name} temporarily unavailable",
                    tool_call_id=tool_call["id"],
                    name=tool_name,
                )
            
            tool = self._tool_map.get(tool_name)
            if not tool:
                raise ValueError(f"Tool {tool_name} not found")
            
            start = time.time()
            result = tool.invoke(tool_call["args"])
            duration = time.time() - start
            # 成功 → 重置失败次数
            self.tool_fail_count[tool_name] = 0
            logger.info(f"[Tool] {tool_name} completed in {duration:.2f}s")

            return ToolMessage(content=str(result), tool_call_id=tool_call["id"], name=tool_name)
        # 处理异常, 增加失败次数，检查是否熔断，达到阈值 → 熔断，返回错误信息
        except Exception as e:
            logger.error(f"Error executing tool {tool_call.get('name', 'unknown')}: {e}")
            # 失败 → 增加失败次数
            self.tool_fail_count[tool_name] += 1
            logger.error(f"[Tool] {tool_name} failed {self.tool_fail_count[tool_name]} times")
            
            # 检查是否熔断，达到阈值 → 熔断
            if self.tool_fail_count[tool_name] >= self.max_failures:
                self.tool_disabled_until[tool_name] = time.time() + self.reset_timeout
                logger.info(f"[Tool] {tool_name} temporarily disabled for {self.reset_timeout} seconds")
                
            return ToolMessage(
                    content=f"Error: {str(e)},Tool {tool_name} temporarily unavailable",
                    tool_call_id=tool_call["id"],
                    name=tool_name,
                )
            

    def __call__(self, state: dict) -> dict:
        last_message = state["messages"][-1]
        tool_calls = getattr(last_message, "tool_calls", [])
        if not tool_calls:
            return {"messages": []}

        results = []
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_tool = {
                executor.submit(self._run_single_tool, tc): tc for tc in tool_calls
            }
            for future in as_completed(future_to_tool):
                try:
                    results.append(future.result(timeout=5))
                except Exception as e:
                    tc = future_to_tool[future]
                    results.append(ToolMessage(content=f"Error: {str(e)}", tool_call_id=tc["id"], name=tc.get("name", "unknown")))
        return {"messages": results}


# ====================== 辅助函数 =======================
def get_latest_question(state: MessagesState) -> Optional[str]:
    """安全获取最新用户问题"""
    try:
        if not state.get("messages"):
            return None
        for message in reversed(state["messages"]):
            if isinstance(message, HumanMessage):
                return get_message_text(message)
        return None
    except Exception as e:
        logger.error(f"Error getting latest question: {e}")
        return None

# ====================== 链工具 =======================
# 可缓存的 LLM 链构建函数，避免重复解析模板文件
def create_chain(llm_chat, template_file: str, structured_output=None):
    """创建 LLM 处理链（用于非 agent 节点 grade/rewrite/generate）"""
    if not hasattr(create_chain, "_cache"):
        create_chain._cache = {}
        create_chain._lock = threading.Lock()

    try:
        # 获取文件修改时间
        file_mtime = os.path.getmtime(template_file)
        cached = create_chain._cache.get(template_file)
        
        # 缓存过期，重新解析模板
        if not cached or cached["mtime"] != file_mtime:
            with create_chain._lock:
                cached = create_chain._cache.get(template_file)
                if not cached or cached["mtime"] != file_mtime:
                    prompt_template = PromptTemplate.from_file(template_file, encoding="utf-8")
                    create_chain._cache[template_file] = {"template": prompt_template, "mtime": file_mtime}
                    cached = create_chain._cache[template_file]

        prompt = ChatPromptTemplate.from_messages([("human", cached["template"].template)])
        if structured_output:
            return prompt | llm_chat.with_structured_output(structured_output)
        return prompt | llm_chat

    except FileNotFoundError:
        logger.error(f"Template file {template_file} not found")
        raise


# ====================== 数据库工具 =======================
@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10), retry=retry_if_exception_type(OperationalError))
def test_connection(db_connection_pool: ConnectionPool) -> bool:
    with db_connection_pool.getconn() as conn:
        with conn.cursor() as cursor:
            cursor.execute("SELECT 1")
            if cursor.fetchone() != (1,):
                raise ConnectionPoolError("连接池测试失败")
    return True


# 
def monitor_connection_pool(db_connection_pool: ConnectionPool, interval: int = 60, warning_threshold: float = 0.8):
    """监控数据库连接池状态
    Args:
        db_connection_pool: 数据库连接池。
        interval: 监控间隔，单位秒。
    
    Returns:
        threading.Thread: 监控线程。
    """
    def _monitor():
        while not db_connection_pool.closed:
            try:
                stats = db_connection_pool.get_stats()
                active = stats.get("connections_in_use", 0)
                total = db_connection_pool.max_size
                logger.info(f"Connection pool: {active}/{total} in use")
                if active >= total * warning_threshold:
                    logger.warning(f"Pool nearing capacity: {active}/{total}")
            except Exception as e:
                logger.error(f"Monitor error: {e}")
            time.sleep(interval)

    t = threading.Thread(target=_monitor, daemon=True)
    t.start()
    return t


# ====================== 非 Agent 节点函数 =======================
# （grade_documents / rewrite / generate 保持 LangGraph 自定义节点）
def grade_documents(state: MessagesState, llm_chat) -> dict:
    """评估文档相关性
    Args:
        state: 当前对话状态，包含消息历史。

    Returns:
        dict: 更新后的状态，包含评分结果。
    """
    logger.info("Grading documents")
    if not state.get("messages"):
        return {"messages": [AIMessage(content="状态为空")], "relevance_score": None}
    try:
        question = get_latest_question(state)
        context = get_message_text(state["messages"][-1])
        grade_chain = create_chain(llm_chat, Config.PROMPT_TEMPLATE_TXT_GRADE, DocumentRelevanceScore)
        scored_result = grade_chain.invoke({"question": question, "context": context}, timeout=10)
        logger.info(f"Relevance score: {scored_result.binary_score}")
        return {"messages": state["messages"], "relevance_score": scored_result.binary_score}
    except Exception as e:
        logger.error(f"Grading error: {e}")
        return {"messages": [AIMessage(content="评分失败")], "relevance_score": None}


def rewrite(state: MessagesState, llm_chat) -> dict:
    """重写查询
    Args:
        state: 当前对话状态。

    Returns:
        dict: 更新后的消息状态。
    """
    logger.info("Rewriting query")
    try:
        question = get_latest_question(state)
        rewrite_chain = create_chain(llm_chat, Config.PROMPT_TEMPLATE_TXT_REWRITE)
        response = rewrite_chain.invoke({"question": question}, timeout=10)
        rewrite_count = state.get("rewrite_count", 0) + 1
        return {"messages": [response], "rewrite_count": rewrite_count}
    except Exception as e:
        logger.error(f"Rewrite error: {e}")
        return {"messages": [AIMessage(content="重写失败")]}


def generate(state: MessagesState, llm_chat) -> dict:
    """生成最终回复
    Args:
        state: 当前对话状态。

    Returns:
        dict: 更新后的消息状态。
    """
    logger.info("Generating response")
    try:
        question = get_latest_question(state)
        context = get_message_text(state["messages"][-1])
        generate_chain = create_chain(llm_chat, Config.PROMPT_TEMPLATE_TXT_GENERATE)
        response = generate_chain.invoke({"context": context, "question": question}, timeout=10)
        return {"messages": [response]}
    except Exception as e:
        logger.error(f"Generate error: {e}")
        return {"messages": [AIMessage(content="生成失败")]}


# ====================== Edge 路由函数 =======================
def route_after_tools(state: MessagesState, tool_config: ToolConfig) -> Literal["generate", "grade_documents"]:
    """根据工具调用路由到下一个节点
    Args:
        state: 当前对话状态，包含消息历史。
        tool_config: 工具配置。
    
    Returns:
        Literal["generate", "grade_documents"]: 下一个节点的名称。
    """
    
    if not state.get("messages"):
        return "generate"
    try:
        last_message = state["messages"][-1]
        if not hasattr(last_message, "name") or last_message.name is None:
            return "generate"
        tool_name = last_message.name
        if tool_name not in tool_config.get_tool_names():
            return "generate"
        return tool_config.get_tool_routing_config().get(tool_name, "generate")
    except Exception as e:
        logger.error(f"route_after_tools error: {e}")
        return "generate"


def route_after_grade(state: MessagesState) -> Literal["generate", "rewrite"]:
    """根据文档相关性评分路由到下一个节点
    Args:
        state: 当前对话状态，包含消息历史。
    
    Returns:
        Literal["generate", "rewrite"]: 下一个节点的名称。
    """
    
    relevance_score = state.get("relevance_score")
    rewrite_count = state.get("rewrite_count", 0)
    if rewrite_count >= 3:
        return "generate"
    if isinstance(relevance_score, str) and relevance_score.lower() == "yes":
        return "generate"
    return "rewrite"


# =========== 中间件组装函数：按需组合不同的中间件栈 ===========
def build_middleware_stack(
    store: BaseStore | None = None,
    user_id: str = "default",
    llm_chat = None,
    enable_summarization: bool = True,
    enable_hitl: bool = False,
    enable_pii: bool = False,
    enable_memory: bool = True,
    enable_safety: bool = True,
    enable_logging: bool = True,
    enable_retry: bool = True,
    hitl_tools: dict | None = None,
    blocked_keywords: list[str] | None = None,
    max_messages: int = 10,
) -> list:
    """根据配置动态构建中间件栈。

    中间件执行顺序：
    - before_* hooks：按列表顺序执行（第一个最先执行）
    - wrap_* hooks：按列表顺序嵌套（第一个是最外层）
    - after_* hooks：按列表**逆序**执行（最后一个最先执行）

    推荐顺序：
    1. 状态扩展类中间件（如 MemoryMiddleware）
    2. 消息过滤/转换类中间件（如 MessageFilterMiddleware）
    3. 安全/合规类中间件（如 PIIMiddleware, SafetyGuard）
    4. 日志/监控类中间件
    5. 内置中间件（Summarization, HITL）
    """
    middleware = []

    # --- 1. 自定义记忆中间件（before_model 注入记忆 + after_agent 存储记忆）---
    if enable_memory and store is not None:
        middleware.append(MemoryMiddleware(store=store, user_id=user_id))
        logger.info("[MiddlewareStack] MemoryMiddleware enabled")

    # --- 2. 消息过滤中间件（modify_model_request 截断消息）---
    middleware.append(MessageFilterMiddleware(max_messages=max_messages))
    logger.info(f"[MiddlewareStack] MessageFilterMiddleware enabled (max={max_messages})")

    # --- 3. 动态 prompt 注入 ---
    middleware.append(inject_user_context)
    logger.info("[MiddlewareStack] DynamicPrompt (inject_user_context) enabled")

    # --- 4. PII 脱敏中间件 ---
    if enable_pii:
        middleware.append(PIIMiddleware("email", strategy="redact", apply_to_input=True))
        middleware.append(PIIMiddleware("phone_number", strategy="redact", apply_to_input=True))
        logger.info("[MiddlewareStack] PIIMiddleware enabled")

    # --- 5. 安全守护中间件 ---
    if enable_safety and blocked_keywords:
        middleware.append(SafetyGuardMiddleware(blocked_keywords=blocked_keywords))
        logger.info(f"[MiddlewareStack] SafetyGuardMiddleware enabled: {blocked_keywords}")

    # --- 6. 日志中间件（装饰器风格）---
    if enable_logging:
        middleware.extend([
            log_agent_start,       # before_agent
            log_before_model_call, # before_model
            log_after_model_call,  # after_model
            log_agent_end,         # after_agent
            log_tool_execution,    # wrap_tool_call
        ])
        logger.info("[MiddlewareStack] Logging middleware enabled")

    # --- 7. 模型调用重试 ---
    if enable_retry:
        middleware.append(retry_model_on_error)  # wrap_model_call
        logger.info("[MiddlewareStack] RetryModelOnError middleware enabled")


    # --- 8. 内置 Summarization 中间件（token 溢出时自动摘要）---
    if enable_summarization and llm_chat is not None:
        middleware.append(
            SummarizationMiddleware(
                model=llm_chat,
                trigger=("tokens", 4000),
                keep=("messages", 20),
            )
        )
        logger.info("[MiddlewareStack] SummarizationMiddleware enabled")

    # --- 9. 内置 Human-in-the-Loop 中间件 ---
    if enable_hitl and hitl_tools:
        middleware.append(
            HumanInTheLoopMiddleware(
                interrupt_on=hitl_tools
            )
        )
        logger.info(f"[MiddlewareStack] HumanInTheLoopMiddleware enabled: {hitl_tools}")


    # --- 10. 内置调用次数限制中间件 ---
    middleware.append(ModelCallLimitMiddleware(run_limit=25, exit_behavior="end"))
    logger.info("[MiddlewareStack] ModelCallLimitMiddleware enabled (run_limit=25)")

    logger.info(f"[MiddlewareStack] Total middleware count: {len(middleware)}")
    return middleware


# ============ 创建并配置状态图（混合架构：create_agent + LangGraph）============
def create_graph(
    db_connection_pool: ConnectionPool,
    llm_chat,
    llm_embedding,
    tool_config: ToolConfig,
    middleware_config: dict | None = None,
) -> StateGraph:
    """创建混合架构状态图：
    - agent 节点：使用 create_agent + middleware 享受中间件能力
    - 其余节点：使用 LangGraph 自定义路由 grade/rewrite/generate

    架构图：
    START → agent_subgraph → [tools_condition]
                                ├── call_tools → [route_after_tools]
                                │                  ├── generate → END
                                │                  └── grade_documents → [route_after_grade]
                                │                                         ├── generate → END
                                │                                         └── rewrite → agent_subgraph
                                └── END
    """
    # --- 验证连接池 ---
    if db_connection_pool is None or db_connection_pool.closed:
        raise ConnectionPoolError("数据库连接池未初始化或已关闭")

    try:
        if not test_connection(db_connection_pool):
            raise ConnectionPoolError("连接池测试失败")
        logger.info("Connection pool: OK")
    except OperationalError as e:
        raise ConnectionPoolError(f"连接池测试失败: {e}")

    # --- 持久化存储 ---
    checkpointer = PostgresSaver(db_connection_pool)
    checkpointer.setup()

    store = PostgresStore(db_connection_pool, index={"dims": 1024, "embed": llm_embedding})
    store.setup()

    # --- 构建中间件栈 ---
    mw_config = middleware_config or {}
    middleware_stack = build_middleware_stack(
        store=store,
        user_id=mw_config.get("user_id", "default"),
        llm_chat=llm_chat,
        enable_summarization=mw_config.get("enable_summarization", True),
        enable_hitl=mw_config.get("enable_hitl", False),
        enable_pii=mw_config.get("enable_pii", False),
        enable_memory=mw_config.get("enable_memory", True),
        enable_safety=mw_config.get("enable_safety", True),
        enable_logging=mw_config.get("enable_logging", True),
        enable_retry=mw_config.get("enable_retry", True),
        hitl_tools=mw_config.get("hitl_tools", None),
        blocked_keywords=mw_config.get("blocked_keywords", None),
        max_messages=mw_config.get("max_messages", 10),
    )

    # --- 读取 system prompt ---
    try:
        system_prompt_template = PromptTemplate.from_file(
            Config.PROMPT_TEMPLATE_TXT_AGENT, encoding="utf-8"
        )
        system_prompt = system_prompt_template.template
    except FileNotFoundError:
        logger.warning("Agent prompt template not found, using default")
        system_prompt = "你是一个有帮助的AI助手。请根据用户的问题和上下文回答。"

    # =========================================================================
    # create_agent 作为 agent 子图（推荐）
    # create_agent 返回的是一个编译后的 LangGraph 图，可以作为子图嵌入
    # =========================================================================
    agent_subgraph = create_agent(
        model=llm_chat,
        tools=tool_config.get_tools(),
        system_prompt=system_prompt,
        middleware=middleware_stack,
        # 不在子图设 checkpointer，由外层图统一管理
    )

    # --- 构建外层 StateGraph ---
    workflow = StateGraph(MessagesState)

    # agent 节点：使用 create_agent 返回的子图
    # create_agent 内部已封装 tool calling 循环 + 中间件
    workflow.add_node("agent", agent_subgraph)

    # 由于 create_agent 内部已处理 tool calling，
    # 外层需要根据 agent 输出判断是否需要 grade/rewrite
    # 策略：agent 完成后，检查最后一条消息来源决定路由
    workflow.add_node("call_tools", ParallelToolNode(tool_config.get_tools()))
    workflow.add_node("grade_documents", lambda state: grade_documents(state, llm_chat=llm_chat))
    workflow.add_node("rewrite", lambda state: rewrite(state, llm_chat=llm_chat))
    workflow.add_node("generate", lambda state: generate(state, llm_chat=llm_chat))

    # 路由逻辑
    def route_after_agent(state: MessagesState) -> str:
        """agent 子图完成后，判断是否有 retrieval 工具输出需要评分"""
        messages = state.get("messages", [])
        if not messages:
            return END

        # 检查是否有来自 retrieval 工具的输出
        for msg in reversed(messages):
            if isinstance(msg, ToolMessage) and hasattr(msg, "name"):
                tool_name = msg.name.lower() if msg.name else ""
                if "retrieve" in tool_name:
                    logger.info(f"Retrieval tool detected: {msg.name}, routing to grade")
                    return "grade_documents"
            # 一旦遇到 AIMessage 就停止往前找
            if isinstance(msg, AIMessage) and not getattr(msg, "tool_calls", None):
                break

        return END

    workflow.add_edge(START, "agent")
    workflow.add_conditional_edges(
        source="agent",
        path=route_after_agent,
        path_map={"grade_documents": "grade_documents", END: END}
    )
    workflow.add_conditional_edges(
        source="grade_documents",
        path=route_after_grade,
        path_map={"generate": "generate", "rewrite": "rewrite"}
    )
    workflow.add_edge("generate", END)
    workflow.add_edge("rewrite", "agent")

    return workflow.compile(checkpointer=checkpointer, store=store)


# ====================== 响应处理 =======================
def graph_response(graph, user_input: str, config: dict, tool_config: ToolConfig) -> None:
    """处理用户输入并输出响应"""
    try:
        events = graph.stream(
            {"messages": [{"role": "user", "content": user_input}], "rewrite_count": 0},
            config,
            stream_mode="values",
        )
        for chunk in events:
            if not chunk or not isinstance(chunk, dict):
                continue
            if "messages" not in chunk or not isinstance(chunk["messages"], list):
                continue

            last_message = chunk["messages"][-1]

            if hasattr(last_message, "tool_calls") and last_message.tool_calls:
                for tc in last_message.tool_calls:
                    if isinstance(tc, dict) and "name" in tc:
                        logger.info(f"Calling tool: {tc['name']}")
                continue

            if hasattr(last_message, "content"):
                content = get_message_text(last_message)
                if hasattr(last_message, "name") and last_message.name in tool_config.get_tool_names():
                    print(f"Tool Output [{last_message.name}]: {content}")
                else:
                    print(f"Assistant: {content}")

    except Exception as e:
        logger.error(f"Response error: {e}", exc_info=True)
        print("Assistant: 处理响应时发生错误")


# ====================== 主函数 =======================
def main():
    """主函数"""
    db_connection_pool = None
    try:
        llm_chat, llm_embedding = get_llm(Config.LLM_TYPE)
        tools = get_tools(llm_embedding)
        tool_config = ToolConfig(tools)

        connection_kwargs = {"autocommit": True, "prepare_threshold": 0, "connect_timeout": 5}
        db_connection_pool = ConnectionPool(
            conninfo=Config.DB_URI, max_size=20, min_size=2,
            kwargs=connection_kwargs, timeout=10
        )

        try:
            db_connection_pool.open()
            logger.info("Connection pool initialized")
        except Exception as e:
            raise ConnectionPoolError(f"无法打开连接池: {e}")

        monitor_connection_pool(db_connection_pool, interval=60)


        # 中间件配置 - 在这里控制启用哪些中间件
        middleware_config = {
            "user_id": "1",
            "enable_summarization": True,
            "enable_hitl": False,
            "hitl_tools": {
                "dangerous_tool": {
                    "allowed_decisions": ["approve", "edit", "reject"]
                }
            },
            "enable_pii": False,
            "enable_memory": True,
            "enable_safety": False,
            "blocked_keywords": ["暴力", "违法"],
            "enable_logging": True,
            "enable_retry": True,
            "max_messages": 10,
        }

        try:
            graph = create_graph(
                db_connection_pool, llm_chat, llm_embedding,
                tool_config, middleware_config
            )
        except ConnectionPoolError as e:
            logger.error(f"Graph creation failed: {e}")
            print(f"错误: {e}")
            sys.exit(1)

        print("聊天机器人准备就绪！输入 'quit'、'exit' 或 'q' 结束对话。")
        config = {"configurable": {"thread_id": "1", "user_id": "1"}}

        while True:
            user_input = input("User: ").strip()
            if user_input.lower() in {"quit", "exit", "q"}:
                print("拜拜!")
                break
            if not user_input:
                print("请输入聊天内容！")
                continue
            graph_response(graph, user_input, config, tool_config)

    except ConnectionPoolError as e:
        logger.error(f"Connection pool error: {e}")
        print(f"错误: {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n再见！")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        print(f"错误: {e}")
        sys.exit(1)
    finally:
        if db_connection_pool and not db_connection_pool.closed:
            db_connection_pool.close()
            logger.info("Connection pool closed")


if __name__ == "__main__":
    main()