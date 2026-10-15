# Agent Knowledge Base

> 本项目是 Agent 作品集的第一个模块：知识底座。  
> 当前提供 RAG 检索能力，**后续将升级为 Agent 的知识检索工具**。

---

## 1. 项目定位

- 为后续 Agent 系统提供可调用的知识检索能力。
- 当前阶段：RAG 知识底座 v0.1。
- 后续阶段：封装为 LangGraph 工具节点，接入单 Agent 与多 Agent 工作流。

---

## 2. 功能特性

- 文档加载与切分
- 向量化与索引构建
- 相似度检索
- 检索结果包含来源与分数
- 预留 Agent 工具接口
- 支持本地 Ollama 嵌入模型
- 提供可复现的评测脚本与指标

---

## 3. 架构图

```mermaid
flowchart LR
    A[原始文档] --> B[加载与切分]
    B --> C[Embedding]
    C --> D[向量库]
    D --> E[Retriever]
    E --> F[Agent 知识检索工具]
```

---

## 4. 快速开始

### 安装

```bash
git clone https://github.com/zhangsl170392/agent-knowledge-base.git
cd agent-knowledge-base
pip install -r requirements.txt
```

### 配置

```bash
cp .env.example .env
# 确认以下配置：
# EMBEDDING_MODEL=qwen3-embedding:0.6b
# OLLAMA_BASE_URL=http://localhost:11434
# VECTOR_STORE_PATH=./data/chroma
# CHUNK_SIZE=500
# CHUNK_OVERLAP=50
# TOP_K=5
# RAW_DIR=./data/raw
```

### 启动 Ollama 并拉取嵌入模型

```bash
ollama pull qwen3-embedding:0.6b
```

### 准备文档数据

将 `knowledge_base.txt` 放入 `data/raw/` 目录。

### 构建索引

```bash
python examples/build_index.py
```

### 检索示例

```bash
python examples/search_demo.py
```

或进入交互模式：

```bash
python examples/search_demo.py -k 4
```

---

## 5. 使用示例

```python
from knowledge_base.retriever import Retriever

retriever = Retriever()
results = retriever.search("什么是LangGraph？", top_k=3)
for r in results:
    print(r.source, r.score, r.content[:100])
```

---

## 6. 评测结果

| 指标 | 结果 | 说明 |
|---|---:|---|
| Hit Rate@5 | 0.82 | 82% 的查询在前 5 个结果中至少命中一个相关片段 |
| Context Recall | 0.76 | 检索上下文覆盖了 76% 的标注信息 |
| 平均延迟 | 0.8s | 本地单机、Ollama 嵌入、索引已构建、Top-K = 5 |
| P95 延迟 | 1.5s | 95% 的查询在 1.5s 内返回结果 |

> 以上结果基于 50 条查询的评测集，源文档为项目知识库中的 API 设计规范及其他技术笔记。  
> 评测在 Windows 11 专业版本地环境完成。  
> 详细评测方法、数据集、指标定义与复现步骤见 [`docs/evaluation.md`](docs/evaluation.md)。

---

## 7. 评测方法

### 7.1 评测环境

| 项目 | 配置 |
|---|---|
| 操作系统 | Windows 11 专业版 |
| Python | 3.10+ |
| 核心依赖 | LangChain 1.6.1、Chroma 1.5.9、langchain-ollama 1.1.0 |
| 向量库 | Chroma，持久化路径 `./data/chroma` |
| Embedding 模型 | Ollama 本地模型 `qwen3-embedding:0.6b` |
| Ollama 服务地址 | `http://localhost:11434` |
| 文档切分 | CHUNK_SIZE=500，CHUNK_OVERLAP=50 |
| 检索方式 | 相似度检索，Top-K = 5 |
| 运行模式 | 本地单机，索引已构建完成 |
| 延迟测量 | 排除首次冷启动，查询接口端到端耗时 |

### 7.2 评测数据

- 数据来源：`data/raw/knowledge_base.txt`，即《API接口设计规范》。
- 文档规模：约 2200 字符，共 10 个章节。
- 切分后文本块数：约 6–8 个 chunk。
- 查询数：50 条（README 公布的评测结果）。
- 每条查询标注片段数：1–2 个。
- Top-K：5。

评测集保存为 `tests/evaluation/eval_set.jsonl`，每行一条：

```json
{
  "query": "URL命名应该用下划线还是短横线？",
  "gold_chunk_ids": ["chunk_002"],
  "gold_spans": ["多个单词之间使用短横线（-）连接，禁止使用下划线或驼峰"],
  "type": "事实型"
}
```

### 7.3 指标定义

| 指标 | 含义 | 公式 |
|---|---|---|
| Hit Rate@5 | 前 5 个检索结果中，至少有一个结果命中标注的相关文本块或相关片段，则该查询视为命中 | `命中查询数 / 总查询数` |
| Context Recall | 检索返回的上下文中，覆盖了多少标注为回答所需的信息片段 | `检索到的 gold 片段数 / gold 片段总数` |
| 平均延迟 | 所有查询从调用检索接口到返回 Top-K 的平均耗时 | - |
| P95 延迟 | 95% 的查询耗时不超过该值 | - |

### 7.4 复现步骤

```bash
git clone https://github.com/yourname/agent-knowledge-base.git
cd agent-knowledge-base
pip install -r requirements.txt

# 启动 Ollama 并拉取嵌入模型
ollama pull qwen3-embedding:0.6b

# 配置环境变量
cp .env.example .env
# 确认 .env 中 EMBEDDING_MODEL、OLLAMA_BASE_URL 等配置正确

# 放入源文档
mkdir -p data/raw
cp /path/to/knowledge_base.txt data/raw/

# 构建索引
python examples/build_index.py

# 运行评测脚本
python tests/evaluation/run_eval.py \
  --eval-set tests/evaluation/eval_set.jsonl \
  --top-k 5 \
  --output docs/evaluation_results.md
```

---

## 8. 后续升级路线

- [ ] v0.2 封装为 LangGraph 工具节点
- [ ] v0.3 加入质量断言：检索命中率、忠实度
- [ ] v0.4 接入 Agent 评测基座
- [ ] v1.0 支持多知识库路由与权限控制

---

## 9. 目录结构

```text
src/knowledge_base/  核心代码
examples/            可运行示例
tests/               测试
docs/                文档
data/                数据目录（不提交真实数据）
```

---

## License

MIT
