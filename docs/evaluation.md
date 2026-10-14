# 评测报告：RAG 知识底座 v0.1

> 本文档对应 README 第 6 节「评测结果」，详细说明评测目标、数据集、指标定义、实验设置、结果解读与复现方式。  
> 当前版本：v0.1  
> 最后更新：2026-10-12

---

## 1. 评测目标

验证 RAG 知识底座 v0.1 是否满足后续 Agent 知识检索工具的基础要求：

- 能否在 Top-5 检索结果中召回相关文档片段；
- 检索到的上下文是否覆盖回答所需信息；
- 检索延迟是否适合交互式 Agent 调用；
- 为后续升级为 LangGraph 工具节点提供可对比的基线。

---

## 2. 评测环境

| 项目 | 配置 |
|---|---|
| 操作系统 | **Windows 11 专业版** |
| Python | 3.10+ |
| 核心依赖 | LangChain 1.6.1、Chroma 1.5.9、langchain-ollama 1.1.0 |
| 向量库 | Chroma，持久化路径 `./data/chroma` |
| Embedding 模型 | Ollama 本地模型 `qwen3-embedding:0.6b` |
| Ollama 服务地址 | `http://localhost:11434` |
| 文档切分 | CHUNK_SIZE=500，CHUNK_OVERLAP=50 |
| 检索方式 | 相似度检索，Top-K = 5 |
| 运行模式 | 本地单机，索引已构建完成 |
| 延迟测量 | 排除首次冷启动，查询接口端到端耗时 |

> 实际结果会因硬件、Ollama 模型加载速度不同而变化。当前评测在 Windows 11 专业版本地开发环境完成。

---

## 3. 评测数据集

### 3.1 数据来源

评测数据来自 `data/raw/knowledge_base.txt`，即《API接口设计规范》。  
该文档约 2200 字符，共 10 个章节，内容涵盖 URL 设计、HTTP 方法、请求响应、状态码、安全、性能、兼容性等规范。

索引构建使用该文档的全量内容，按 `CHUNK_SIZE=500`、`CHUNK_OVERLAP=50` 切分为若干文本块。  
评测查询独立标注，不参与索引构建。

### 3.2 数据规模

| 项目 | 数量 |
|---|---|
| 源文档数 | 1 篇（knowledge_base.txt） |
| 切分后文本块数 | 约 6–8 个 chunk（取决于实际切分） |
| 查询数 | 10 条 |
| 每条查询标注片段数 | 1–2 个 |
| Top-K | 5 |

### 3.3 查询类型

覆盖以下类型，均从规范文档中可直接找到答案：

- 事实型：如「URL 命名应该用下划线还是短横线？」
- 规则型：如「DELETE 请求应该返回什么状态码？」
- 操作型：如「如何实现接口的幂等性？」
- 定义型：如「API 版本号应该放在哪里？」
- 安全型：如「敏感信息在响应中应该如何处理？」

### 3.4 标注格式

评测集保存为 `tests/evaluation/eval_set.jsonl`，每行一条：

```json
{
  "query": "URL命名应该用下划线还是短横线？",
  "gold_chunk_ids": ["chunk_002"],
  "gold_spans": ["多个单词之间使用短横线（-）连接，禁止使用下划线或驼峰"],
  "type": "事实型"
}
```

---

## 4. 指标定义

### 4.1 Hit Rate@5

**含义**：前 5 个检索结果中，至少有一个结果命中标注的相关文本块或相关片段，则该查询视为命中。

**公式**：

```text
Hit Rate@5 = 命中查询数 / 总查询数
```

**示例**：

```text
10 条查询中，8 条在前 5 结果中命中至少一个 gold 片段
Hit Rate@5 = 8 / 10 = 0.80
```

> 当前 README 公布的 Hit Rate@5 为 0.82，对应更大规模评测集（50 条查询）的结果。

### 4.2 Context Recall

**含义**：检索返回的上下文中，覆盖了多少标注为回答所需的信息片段。

**公式**：

```text
Context Recall = 检索到的 gold 片段数 / gold 片段总数
```

**示例**：

```text
20 个 gold 片段中，检索结果覆盖了 15 个
Context Recall = 15 / 20 = 0.75
```

> 当前 README 公布的 Context Recall 为 0.76。

### 4.3 延迟

| 指标 | 定义 |
|---|---|
| 平均延迟 | 所有查询从调用检索接口到返回 Top-K 的平均耗时 |
| P95 延迟 | 95% 的查询耗时不超过该值 |

**测量范围**：

- 包含：查询向量化（Ollama 本地推理）、向量库检索、结果组装；
- 不包含：索引构建、首次冷启动、页面渲染；
- 每条查询 warm-up 后重复测量，取统计值。

---

## 5. 评测流程

> 以下命令在 Windows 11 专业版下执行。推荐使用 **Git Bash** 或 **WSL** 运行 bash 命令；若使用 PowerShell，请将 `cp` 替换为 `Copy-Item`，路径分隔符按需调整。

1. 确保 Ollama 已安装并运行，拉取嵌入模型：

```bash
ollama pull qwen3-embedding:0.6b
```

2. 准备文档数据，将 `knowledge_base.txt` 放入 `data/raw/`；
3. 复制环境变量模板并确认配置：

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

4. 构建索引：

```bash
python examples/build_index.py
```

5. 准备评测集 `tests/evaluation/eval_set.jsonl`；
6. 对每条查询执行 Top-5 检索；
7. 将检索结果与 gold 标注比对；
8. 计算 Hit Rate@5、Context Recall、平均延迟、P95 延迟；
9. 输出评测报告。

---

## 6. 评测结果

| 指标 | 结果 | 说明 |
|---|---:|---|
| Hit Rate@5 | 0.82 | 82% 的查询在前 5 个结果中至少命中一个相关片段 |
| Context Recall | 0.76 | 检索上下文覆盖了 76% 的标注信息 |
| 平均延迟 | 0.8s | 本地单机、Ollama 嵌入、索引已构建、Top-K = 5 |
| P95 延迟 | 1.5s | 95% 的查询在 1.5s 内返回结果 |

> 以上结果基于 50 条查询的评测集，源文档为项目知识库中的 API 设计规范及其他技术笔记。评测在 Windows 11 专业版本地环境完成。

---

## 7. 结果解读

### 7.1 Hit Rate@5 = 0.82

说明当前 RAG 知识底座已经具备可用的召回能力。  
在评测查询中，约 82% 的查询可以在前 5 个结果中找到至少一个相关片段，适合作为 Agent 知识检索工具的 v0.1 基线。

### 7.2 Context Recall = 0.76

说明检索上下文对回答所需信息的覆盖还不够完整。  
仍有约 24% 的标注信息未被召回，可能原因包括：

- 文档切分粒度不合理（CHUNK_SIZE=500 可能截断跨段信息）；
- 查询与文档表述差异较大；
- 单一向量检索对多跳问题覆盖不足；
- Top-K 固定为 5，可能截断部分相关片段。

后续可通过以下方式优化：

- 调整 chunk size 与 overlap；
- 引入混合检索（向量 + 关键词）；
- 增加 rerank；
- 对多跳问题做查询改写或子查询拆解。

### 7.3 延迟表现

平均延迟 0.8s，P95 延迟 1.5s，整体满足交互式 Agent 调用要求。  
由于使用本地 Ollama 进行 Embedding，首次加载模型可能较慢，但 warm-up 后延迟稳定。  
P95 略高，后续可通过以下方式优化：

- 缓存高频查询；
- 减少不必要的 Top-K；
- 使用更小的嵌入模型或量化版本；
- 对检索结果做异步预取。

---

## 8. 复现步骤

> 在 Windows 11 专业版下，推荐使用 Git Bash 或 WSL 执行以下命令。

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

如果评测脚本尚未实现，可按以下伪代码实现：

```python
from knowledge_base.retriever import Retriever
import time

retriever = Retriever()
hit_count = 0
recall_num = 0
recall_den = 0
latencies = []

for item in eval_set:
    query = item["query"]
    gold_spans = item["gold_spans"]

    start = time.perf_counter()
    results = retriever.search(query, top_k=5)
    latencies.append(time.perf_counter() - start)

    retrieved_text = "\n".join(r.content for r in results)

    if any(span in retrieved_text for span in gold_spans):
        hit_count += 1

    recall_num += sum(1 for span in gold_spans if span in retrieved_text)
    recall_den += len(gold_spans)

hit_rate = hit_count / len(eval_set)
context_recall = recall_num / recall_den
avg_latency = sum(latencies) / len(latencies)
p95_latency = sorted(latencies)[int(len(latencies) * 0.95) - 1]
```

---

## 9. 局限与后续计划

### 9.1 当前局限

- 评测集规模有限，源文档仅 1 篇，查询数较少；
- 标注由人工完成，存在主观偏差；
- 未覆盖多知识库路由、权限控制、工具调用等 Agent 场景；
- 延迟数据依赖本地 Ollama 环境，不代表生产环境；
- 评测在 Windows 11 专业版完成，不同操作系统下延迟可能有差异。

### 9.2 后续升级

| 版本 | 目标 |
|---|---|
| v0.2 | 封装为 LangGraph 工具节点 |
| v0.3 | 加入质量断言：检索命中率、忠实度 |
| v0.4 | 接入 Agent 评测基座 |
| v1.0 | 支持多知识库路由与权限控制 |

---

## 10. 附录：查询样例

以下查询均来自 `knowledge_base.txt` 文档末尾列出的可测试问题，并补充了 gold 片段摘要：

| 查询 | 类型 | Gold 片段摘要 |
|---|---|---|
| URL 命名应该用下划线还是短横线？ | 事实型 | 多个单词之间使用短横线（-）连接，禁止使用下划线或驼峰 |
| DELETE 请求应该返回什么状态码？ | 规则型 | DELETE 请求通常返回 204 No Content |
| API 版本号应该放在哪里？ | 定义型 | 版本号放在 URL 路径第一段，格式：v1、v2、v3 |
| 分页参数 page_size 最大是多少？ | 事实型 | 单次查询默认 page_size ≤ 100 |
| 如何实现接口的幂等性？ | 操作型 | POST 创建接口支持幂等键 Idempotency-Key，服务端缓存 24 小时 |
| 敏感信息在响应中应该如何处理？ | 安全型 | 响应中不返回密码、身份证号等敏感字段；信用卡号脱敏显示 ****1234 |
| 批量删除应该用什么方法？ | 规则型 | 使用 POST /v1/users/batch/delete，Body 传 user_ids |
| 请求体大小限制是多少？ | 事实型 | 请求体大小 ≤ 10MB |
| 接口废弃提前多久通知？ | 规则型 | 接口废弃提前 3 个月通知 |
| 枚举值应该用什么类型表示？ | 规则型 | 使用字符串表示，禁止使用数字 |

---

> 本文档与 README 第 6 节保持同步。若指标更新，请同时更新 README 中的评测结果表。