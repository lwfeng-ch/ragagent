# 基于 LangGraph 的 RAG 智能中医系统

一个基于 LangGraph 构建的检索增强生成（RAG）智能分诊系统，支持多模型后端、向量检索、文档相关性评分和问题重写优化。

## 功能特性

- **LangGraph 状态图工作流**：使用状态图构建复杂的 Agent 决策流程
- **RAG 检索增强**：基于 ChromaDB 的向量检索，支持文档相关性评分
- **问题重写优化**：当检索文档不相关时自动重写问题，提升检索质量
- **并行工具执行**：自定义 ParallelToolNode 支持多工具并发调用
- **多模型支持**：统一接口支持 OpenAI、Qwen（通义千问）、OneAPI、Ollama
- **状态持久化**：PostgreSQL 存储对话状态，支持跨会话记忆
- **流式响应**：支持 SSE 流式输出，提升用户体验
- **Web UI**：集成 Gradio 界面，快速体验系统功能

## 技术架构

```mermaid
%%{init: {'theme': 'dark'}}%%
flowchart TB
    subgraph 用户层
        A[用户请求] --> B[FastAPI /v1/chat/completions]
        A --> C[Gradio Web UI]
    end
    
    subgraph Agent层
        B --> D{LangGraph StateGraph}
        D --> E[agent 节点<br/>分诊决策]
        E --> F{tools_condition}
        F -->|需要工具| G[ParallelToolNode]
        F -->|直接回答| H[generate 节点]
        G --> I{工具路由}
        I -->|检索工具| J[grade_documents]
        I -->|其他工具| H
        J --> K{相关性判断}
        K -->|yes| H
        K -->|no| L[rewrite 节点]
        L --> E
        H --> M[END]
    end
    
    subgraph 数据层
        G --> N[(ChromaDB)]
        D --> O[(PostgreSQL)]
    end
    
    subgraph 模型层
        E --> P[LLM Chat]
        G --> Q[Embedding]
    end
    
    style A fill:#2ecc71,color:#fff
    style M fill:#2ecc71,color:#fff
    style E fill:#3498db,color:#fff
    style G fill:#3498db,color:#fff
    style H fill:#3498db,color:#fff
    style J fill:#3498db,color:#fff
    style L fill:#3498db,color:#fff
    style F fill:#f1c40f,color:#fff
    style I fill:#f1c40f,color:#fff
    style K fill:#f1c40f,color:#fff
```

## 项目结构

```
基于LangGraph的RAG智能中医系统/
├── L1-Project-2/                    # 核心项目目录
│   ├── main.py                      # FastAPI 服务入口
│   ├── ragAgent.py                  # LangGraph Agent 核心逻辑
│   ├── webUI.py                     # Gradio Web 界面
│   ├── vectorSave.py                # 向量存储脚本
│   ├── apiTest.py                   # API 测试脚本
│   ├── chromaDB/                    # Chroma 向量数据库
│   ├── input/                       # 输入文档（PDF）
│   │   ├── deepseek-v3-1-4.pdf
│   │   └── 健康档案.pdf
│   ├── output/                      # 日志输出
│   │   └── app.log
│   ├── prompts/                     # Prompt 模板文件
│   │   ├── prompt_template_agent.txt    # Agent 分诊提示
│   │   ├── prompt_template_generate.txt # 生成回答提示
│   │   ├── prompt_template_grade.txt    # 相关性评分提示
│   │   └── prompt_template_rewrite.txt  # 问题重写提示
│   └── utils/                       # 工具模块
│       ├── config.py                # 统一配置类
│       ├── llms.py                  # LLM 模型初始化
│       └── tools_config.py          # 工具配置
├── assets/                          # 资源文件
├── punkt_tab/                       # NLTK 分词数据
├── requirements.txt                 # 依赖清单
└── README.md                        # 项目说明
```

## 快速开始

### 环境要求

- Python 3.10+
- PostgreSQL 12+（用于状态持久化）
- 可选：Ollama（本地模型）

### 安装步骤

1. **克隆项目**

```bash
git clone <repository-url>
cd 基于LangGraph的RAG智能中医系统
```

2. **创建虚拟环境**

```bash
python -m venv venv
# Windows
venv\Scripts\activate
# Linux/macOS
source venv/bin/activate
```

3. **安装依赖**

```bash
pip install -r requirements.txt
```

4. **配置环境变量**

在 `L1-Project-2` 目录下创建 `.env` 文件：

```env
# 阿里通义千问 API Key（使用 qwen 模型时必需）
DASHSCOPE_API_KEY=your_dashscope_api_key

# OpenAI API Key（使用 openai 模型时必需）
OPENAI_API_KEY=your_openai_api_key
OPENAI_BASE_URL=https://api.openai.com/v1

# PostgreSQL 数据库连接（状态持久化）
DB_URI=postgresql://username:password@localhost:5432/database_name?sslmode=disable
```

5. **初始化向量数据库**

将 PDF 文档放入 `input/` 目录，然后运行：

```bash
cd L1-Project-2
python vectorSave.py
```

6. **启动服务**

```bash
python main.py
```

服务将在 `http://0.0.0.0:8012` 启动。

### 使用 Gradio Web UI

```bash
python webUI.py
```

## 配置说明

### 模型配置

在 `utils/config.py` 中修改 `LLM_TYPE` 选择模型后端：

| LLM_TYPE | 说明 | API Key 环境变量 |
|----------|------|------------------|
| `qwen` | 阿里通义千问（默认） | `DASHSCOPE_API_KEY` |
| `openai` | OpenAI GPT 系列 | `OPENAI_API_KEY` |
| `oneapi` | OneAPI 中转服务 | `DASHSCOPE_API_KEY` |
| `ollama` | 本地 Ollama 模型 | 无需 |

### 服务配置

```python
# utils/config.py
class Config:
    # API 服务地址和端口
    HOST = "0.0.0.0"
    PORT = 8012
    
    # Chroma 向量数据库配置
    CHROMADB_DIRECTORY = "chromaDB"
    CHROMADB_COLLECTION_NAME = "demo001"
    
    # 日志配置
    LOG_FILE = "output/app.log"
    MAX_BYTES = 5 * 1024 * 1024  # 5MB
    BACKUP_COUNT = 3
```

## API 文档

### POST /v1/chat/completions

兼容 OpenAI Chat Completions API 格式。

**请求体：**

```json
{
  "messages": [
    {"role": "user", "content": "请根据健康档案分析我的身体状况"}
  ],
  "stream": false,
  "userId": "user123",
  "conversationId": "conv456"
}
```

**参数说明：**

| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| messages | array | 是 | 对话消息列表 |
| stream | boolean | 否 | 是否流式输出，默认 false |
| userId | string | 否 | 用户标识，用于记忆存储 |
| conversationId | string | 否 | 会话标识，用于状态持久化 |

**响应示例（非流式）：**

```json
{
  "id": "chatcmpl-xxx",
  "object": "chat.completion",
  "created": 1234567890,
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "根据您的健康档案分析..."
      },
      "finish_reason": "stop"
    }
  ]
}
```

**流式响应：**

设置 `stream: true`，返回 SSE 格式数据流：

```
data: {"id":"chatcmpl-xxx","object":"chat.completion.chunk","choices":[{"delta":{"content":"根据"},"finish_reason":null}]}

data: {"id":"chatcmpl-xxx","object":"chat.completion.chunk","choices":[{"delta":{"content":"您的"},"finish_reason":null}]}

data: {"id":"chatcmpl-xxx","object":"chat.completion.chunk","choices":[{"delta":{},"finish_reason":"stop"}]}
```

## 核心工作流程

```mermaid
%%{init: {'theme': 'dark'}}%%
sequenceDiagram
    participant U as 用户
    participant F as FastAPI
    participant A as Agent
    participant T as Tools
    participant V as ChromaDB
    participant L as LLM

    U->>F: POST /v1/chat/completions
    F->>A: 创建状态图执行
    
    loop LangGraph 状态流转
        A->>L: 用户问题 + Prompt
        L-->>A: 决策：调用工具/直接回答
        
        alt 需要检索
            A->>T: 执行 retrieve 工具
            T->>V: 向量相似度搜索
            V-->>T: 返回相关文档
            T-->>A: ToolMessage
            A->>L: 文档相关性评分
            L-->>A: yes/no
            
            alt 相关性=no
                A->>L: 重写问题
                L-->>A: 新问题
            end
        end
        
        A->>L: 生成最终回答
        L-->>A: AIMessage
    end
    
    A-->>F: 最终状态
    F-->>U: JSON/Stream 响应
```

## 工具说明

系统内置以下工具：

| 工具名称 | 功能 | 路由目标 |
|----------|------|----------|
| `retrieve` | ChromaDB 向量检索 | grade_documents（相关性评分） |
| `multiply` | 两数乘法计算 | generate（直接生成） |
| `get_conversation_summary` | 获取对话摘要 | generate |
| `get_user_context` | 获取用户上下文 | generate |
| `set_user_preference` | 设置用户偏好 | generate |

## 开发指南

### 添加新工具

在 `utils/tools_config.py` 中添加：

```python
@tool
def my_new_tool(param: str) -> str:
    """工具描述，用于 LLM 理解工具用途。
    
    Args:
        param: 参数说明
    """
    return f"处理结果: {param}"

# 在 get_tools 函数中添加到工具列表
tools = [..., my_new_tool]
```

### 自定义 Prompt 模板

修改 `prompts/` 目录下的模板文件：

- `prompt_template_agent.txt` - Agent 分诊决策
- `prompt_template_generate.txt` - 回答生成
- `prompt_template_grade.txt` - 文档相关性评分
- `prompt_template_rewrite.txt` - 问题重写

## 技术栈

| 类别 | 技术 |
|------|------|
| 框架 | LangChain 0.3.x, LangGraph 0.2.x |
| Web | FastAPI, Uvicorn, Gradio |
| 向量数据库 | ChromaDB |
| 关系数据库 | PostgreSQL |
| LLM | OpenAI / Qwen / Ollama |
| 文档处理 | pdfminer.six, NLTK |

## 常见问题

**Q: 如何切换到本地模型？**

A: 修改 `utils/config.py` 中的 `LLM_TYPE = "ollama"`，确保 Ollama 服务已启动并下载了对应模型。

**Q: 向量检索返回空结果？**

A: 检查 `chromaDB/` 目录是否存在数据，运行 `python vectorSave.py` 重新构建向量索引。

**Q: 数据库连接失败？**

A: 检查 PostgreSQL 服务是否运行，`.env` 中的 `DB_URI` 配置是否正确。

## License

MIT License
