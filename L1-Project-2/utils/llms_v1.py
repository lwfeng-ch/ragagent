import os
import logging
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.embeddings import Embeddings
from langchain_core.rate_limiters import InMemoryRateLimiter

# === Provider 类 ===
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

# [可选] 原生 Qwen 支持（如安装了 langchain-qwq）
try:
    from langchain_qwq import ChatQwen
    HAS_LANGCHAIN_QWQ = True
except ImportError:
    HAS_LANGCHAIN_QWQ = False

# [可选] DashScope 原生 Embedding
try:
    from langchain_community.embeddings import DashScopeEmbeddings
    HAS_DASHSCOPE_EMBEDDINGS = True
except ImportError:
    HAS_DASHSCOPE_EMBEDDINGS = False

# [仅用于支持 init_chat_model 的 provider，如 openai]
from langchain.chat_models import init_chat_model

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# =============================================================================
# 模型配置
# =============================================================================
MODEL_CONFIGS = {
    "openai": {
        "provider": "openai",
        "base_url": os.getenv("OPENAI_BASE_URL"),
        "api_key": os.getenv("OPENAI_API_KEY"),
        "chat_model": "gpt-4o",
        "embedding_model": "text-embedding-3-small",
        # openai 支持 init_chat_model 字符串模式
        "supports_init_chat_model": True,
        "model_string": "openai:gpt-4o",
        "embedding_string": "openai:text-embedding-3-small",
        # 推荐使用的 Chat 类
        "chat_class": "ChatOpenAI",
        # 推荐使用的 Embedding 类
        "embedding_class": "OpenAIEmbeddings",
    },
    "qwen": {
        "provider": "qwen",
        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "api_key": os.getenv("DASHSCOPE_API_KEY"),
        "chat_model": "qwen3.5-plus",
        "embedding_model": "qwen3-vl-embedding",
        # qwen 不支持 init_chat_model 字符串模式
        "supports_init_chat_model": False,
        "model_string": None,
        "embedding_string": None,
        # 优先使用 ChatQwen（如安装了 langchain-qwq），否则回退到 ChatOpenAI 兼容模式
        "chat_class": "ChatQwen" if HAS_LANGCHAIN_QWQ else "ChatOpenAI",
        # 优先使用 DashScopeEmbeddings，否则回退到 OpenAIEmbeddings 兼容模式
        "embedding_class": "DashScopeEmbeddings" if HAS_DASHSCOPE_EMBEDDINGS else "OpenAIEmbeddings",
    },
    "oneapi": {
        "provider": "openai",  # oneapi 使用 OpenAI 兼容 API
        "base_url": "http://139.224.72.218:3000/v1",
        "api_key": os.getenv("DASHSCOPE_API_KEY"),
        "chat_model": "qwen-max",
        "embedding_model": "text-embedding-v1",
        "supports_init_chat_model": False,
        "model_string": None,
        "embedding_string": None,
        "chat_class": "ChatOpenAI",
        "embedding_class": "OpenAIEmbeddings",
    },
    "ollama": {
        "provider": "ollama",
        "base_url": "http://localhost:11434/v1",
        "api_key": "ollama",
        "chat_model": "qwen2.5:32b",
        "embedding_model": "bge-m3:latest",
        # ollama 支持 init_chat_model（需 langchain-ollama 包）
        "supports_init_chat_model": True,
        "model_string": "ollama:qwen2.5:32b",
        "embedding_string": None,  # ollama embedding 仍需显式初始化
        "chat_class": "ChatOpenAI",  # 也可用 ChatOllama
        "embedding_class": "OpenAIEmbeddings",
    },
}

DEFAULT_LLM_TYPE = "qwen"
DEFAULT_TEMPERATURE = 0.0


class LLMInitializationError(Exception):
    """自定义异常类用于LLM初始化错误"""
    pass


# =============================================================================
# Chat 模型初始化
# =============================================================================

def _create_chat_model(llm_type: str, config: dict[str, Any]) -> BaseChatModel:
    """根据配置创建 Chat 模型实例。

    策略：
    1. 如果 provider 支持 init_chat_model 且无自定义 base_url → 使用统一初始化器
    2. 如果是 qwen 且安装了 langchain-qwq → 使用 ChatQwen
    3. 其他情况 → 使用 ChatOpenAI（OpenAI 兼容模式）
    """
    chat_class = config.get("chat_class", "ChatOpenAI")
    api_key = config["api_key"]

    # ollama 不需要真实 key
    if llm_type == "ollama" and not api_key:
        api_key = "ollama"

    # --- 策略 1：使用 init_chat_model（仅限支持的 provider 且无自定义 base_url）---
    if config.get("supports_init_chat_model") and config.get("model_string"):
        # 仅当不需要自定义 base_url 时才使用
        # openai 默认 base_url 为 None，不需要覆盖
        if config.get("base_url") is None or llm_type == "openai":
            try:
                logger.info(f"Using init_chat_model for {llm_type}: {config['model_string']}")
                return init_chat_model(
                    config["model_string"],
                    temperature=DEFAULT_TEMPERATURE,
                    timeout=30,
                    max_retries=2,
                )
            except Exception as e:
                logger.warning(f"init_chat_model failed for {llm_type}, falling back: {e}")

    # --- 策略 2：使用 ChatQwen 原生包 ---
    if chat_class == "ChatQwen" and HAS_LANGCHAIN_QWQ:
        logger.info(f"Using ChatQwen for {llm_type}")
        return ChatQwen(
            model=config["chat_model"],
            temperature=DEFAULT_TEMPERATURE,
            max_retries=3,
            # enable_thinking=True,  # 取消注释以启用 Qwen3 推理模式
        )

    # --- 策略 3：使用 ChatOpenAI（OpenAI 兼容模式）---
    logger.info(f"Using ChatOpenAI (compat mode) for {llm_type}")
    return ChatOpenAI(
        base_url=config["base_url"],
        api_key=api_key,
        model=config["chat_model"],
        temperature=DEFAULT_TEMPERATURE,
        timeout=30,
        max_retries=2,
    )


# =============================================================================
# Embedding 模型初始化
# =============================================================================

def _create_embedding_model(llm_type: str, config: dict[str, Any]) -> Embeddings:
    """根据配置创建 Embedding 模型实例。

    策略：
    1. 如果是 qwen 且安装了 DashScopeEmbeddings → 使用原生 DashScope
    2. 其他情况 → 使用 OpenAIEmbeddings（兼容模式）
    """
    embedding_class = config.get("embedding_class", "OpenAIEmbeddings")
    api_key = config["api_key"]

    if llm_type == "ollama" and not api_key:
        api_key = "ollama"

    # --- 策略 1：使用 DashScopeEmbeddings 原生包 ---
    if embedding_class == "DashScopeEmbeddings" and HAS_DASHSCOPE_EMBEDDINGS:
        logger.info(f"Using DashScopeEmbeddings for {llm_type}")
        return DashScopeEmbeddings(
            model=config["embedding_model"],
            dashscope_api_key=api_key,
        )

    # --- 策略 2：使用 OpenAIEmbeddings（兼容模式）---
    logger.info(f"Using OpenAIEmbeddings (compat mode) for {llm_type}")
    return OpenAIEmbeddings(
        base_url=config["base_url"],
        api_key=api_key,
        model=config["embedding_model"],
        check_embedding_ctx_length=False,
    )


# =============================================================================
# 主入口
# =============================================================================

def initialize_llm(llm_type: str = DEFAULT_LLM_TYPE) -> tuple[BaseChatModel, Embeddings]:
    """初始化 LLM 和 Embedding 模型实例。

    自动选择最优初始化方式：
    - openai → 优先 init_chat_model
    - qwen → 优先 ChatQwen (langchain-qwq)，回退到 ChatOpenAI 兼容模式
    - oneapi → ChatOpenAI 兼容模式
    - ollama → ChatOpenAI 兼容模式（或 init_chat_model + ChatOllama）

    Args:
        llm_type: LLM类型

    Returns:
        tuple[BaseChatModel, Embeddings]

    Raises:
        LLMInitializationError: 初始化失败时
    """
    try:
        if llm_type not in MODEL_CONFIGS:
            raise ValueError(
                f"不支持的LLM类型: {llm_type}. 可用类型: {list(MODEL_CONFIGS.keys())}"
            )

        config = MODEL_CONFIGS[llm_type]

        llm_chat = _create_chat_model(llm_type, config)
        llm_embedding = _create_embedding_model(llm_type, config)

        logger.info(f"成功初始化 {llm_type}: chat={type(llm_chat).__name__}, "
                     f"embedding={type(llm_embedding).__name__}")
        return llm_chat, llm_embedding

    except ValueError as ve:
        logger.error(f"LLM配置错误: {str(ve)}")
        raise LLMInitializationError(f"LLM配置错误: {str(ve)}")
    except Exception as e:
        logger.error(f"初始化LLM失败: {str(e)}")
        raise LLMInitializationError(f"初始化LLM失败: {str(e)}")


# =============================================================================
# [可选] 带速率限制的初始化
# =============================================================================
def initialize_llm_with_rate_limiter(
    llm_type: str = DEFAULT_LLM_TYPE,
    requests_per_second: float = 1.0,
    max_bucket_size: int = 10,
) -> tuple[BaseChatModel, Embeddings]:
    """初始化带速率限制的模型。"""
    if llm_type not in MODEL_CONFIGS:
        raise LLMInitializationError(f"不支持的LLM类型: {llm_type}")

    config = MODEL_CONFIGS[llm_type]
    # 初始化速率限制器
    logger.info(f"初始化速率限制器，速率限制器: {rate_limiter}")
    rate_limiter = InMemoryRateLimiter(
        requests_per_second=requests_per_second,
        check_every_n_seconds=0.1,
        max_bucket_size=max_bucket_size,
    )

    api_key = config["api_key"]
    if llm_type == "ollama" and not api_key:
        api_key = "ollama"

    try:
        # 带 rate_limiter 只能通过显式初始化
        llm_chat = ChatOpenAI(
            base_url=config["base_url"],
            api_key=api_key,
            model=config["chat_model"],
            temperature=DEFAULT_TEMPERATURE,
            timeout=30,
            max_retries=2,
            rate_limiter=rate_limiter,
        )
        llm_embedding = _create_embedding_model(llm_type, config)

        logger.info(f"成功初始化带速率限制的 {llm_type} ({requests_per_second} req/s)")
        return llm_chat, llm_embedding

    except Exception as e:
        raise LLMInitializationError(f"带速率限制的初始化失败: {str(e)}")

# 多层兜底机制
def get_llm(
    llm_type: str = DEFAULT_LLM_TYPE,
    requests_per_second: float = 1.0,
    max_bucket_size: int = 10,
) -> tuple[BaseChatModel, Embeddings]:
    """获取模型实例（带多级兜底机制）：

    优先级：
    1. 带限流初始化（生产推荐）
    2. 普通初始化（兜底）
    3. 默认模型 + 限流
    4. 默认模型 + 普通初始化（最终兜底）
    """
    # -------- 第一层：当前模型 + 限流 --------
    try:
        return initialize_llm_with_rate_limiter(
            llm_type,
            requests_per_second=requests_per_second,
            max_bucket_size=max_bucket_size,
        )
    except LLMInitializationError as e:
        logger.warning(f"[限流模式失败] {llm_type}: {str(e)}")

    # -------- 第二层：当前模型 + 普通模式 --------
    try:
        return initialize_llm(llm_type)
    except LLMInitializationError as e:
        logger.warning(f"[普通模式失败] {llm_type}: {str(e)}")

    # -------- 第三层：默认模型 + 限流 --------
    if llm_type != DEFAULT_LLM_TYPE:
        try:
            logger.warning(f"尝试默认模型（限流模式）: {DEFAULT_LLM_TYPE}")
            return initialize_llm_with_rate_limiter(
                DEFAULT_LLM_TYPE,
                requests_per_second=requests_per_second,
                max_bucket_size=max_bucket_size,
            )
        except LLMInitializationError as e:
            logger.warning(f"[默认限流失败] {str(e)}")

    # -------- 第四层：默认模型 + 普通模式（最终兜底）--------
    try:
        logger.warning(f"尝试默认模型（普通模式）: {DEFAULT_LLM_TYPE}")
        return initialize_llm(DEFAULT_LLM_TYPE)
    except LLMInitializationError as e:
        logger.error("所有LLM初始化方式均失败")
        raise LLMInitializationError(f"所有初始化策略失败: {str(e)}")


# =============================================================================
# 辅助函数
# =============================================================================

def get_model_string(llm_type: str = DEFAULT_LLM_TYPE) -> str | None:
    """获取 init_chat_model 兼容的模型字符串。仅对部分 provider 有效。"""
    if llm_type not in MODEL_CONFIGS:
        return None
    return MODEL_CONFIGS[llm_type].get("model_string")


def get_available_providers() -> list[str]:
    """获取所有可用的 provider 列表。"""
    return list(MODEL_CONFIGS.keys())


def get_model_info(llm_type: str = DEFAULT_LLM_TYPE) -> dict[str, Any]:
    """获取模型配置信息（api_key 脱敏）。"""
    if llm_type not in MODEL_CONFIGS:
        return {}
    config = MODEL_CONFIGS[llm_type].copy()
    if config.get("api_key"):
        key = str(config["api_key"])
        config["api_key"] = f"{key[:4]}...{key[-4:]}" if len(key) > 8 else "***"
    return config


# =============================================================================
# 测试
# =============================================================================

if __name__ == "__main__":
    try:
        print("=" * 60)
        print(f"可用 Providers: {get_available_providers()}")
        print(f"langchain-qwq 可用: {HAS_LANGCHAIN_QWQ}")
        print(f"DashScopeEmbeddings 可用: {HAS_DASHSCOPE_EMBEDDINGS}")
        print("=" * 60)

        # 测试 Qwen 初始化
        print("\n[测试] 初始化 qwen:")
        llm_chat, llm_embedding = get_llm("qwen")
        print(f"  Chat 类型: {type(llm_chat).__name__}")
        print(f"  Embedding 类型: {type(llm_embedding).__name__}")

        # 显示模型信息
        print(f"\n[信息] qwen 配置:")
        for k, v in get_model_info("qwen").items():
            print(f"  {k}: {v}")

        print("\n所有测试通过 ✅")

    except LLMInitializationError as e:
        logger.error(f"程序终止: {str(e)}")