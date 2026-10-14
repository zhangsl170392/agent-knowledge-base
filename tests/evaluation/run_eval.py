"""
RAG 检索评测脚本

用法：
    python tests/evaluation/run_eval.py \
        --eval-file tests/evaluation/eval_set.jsonl \
        --top-k 5 \
        --output tests/evaluation/eval_result.json \
        --save-details
"""

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

# 保证可以从项目根目录导入 knowledge_base
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from knowledge_base.retriever import Retriever

def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    """加载评测集 JSONL，兼容 query/question 与 gold_spans/answer 字段。"""
    items: List[Dict[str, Any]] = []

    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                raw = json.loads(line)
            except json.decoder.JSONDecodeError as e:
                raise ValueError(f"{path}:{line_no} JSON 解析失败: {e}") from e

            query = raw.get("query") or raw.get("question")

            gold_spans = raw.get("gold_spans")
            if gold_spans is None:
                answer = raw.get("answer")
                if isinstance(answer, str):
                    gold_spans = [answer]
                elif isinstance(answer, list):
                    gold_spans = answer

            if not query:
                raise ValueError(f"{path}:{line_no} 缺少 query/question 字段")

            if not isinstance(gold_spans, list) or not gold_spans:
                raise ValueError(f"{path}:{line_no} 缺少非空 gold_spans/answer 字段")

            item = dict(raw)
            item["query"] = query
            item["gold_spans"] = [str(s) for s in gold_spans]
            items.append(item)

    return items

def get_content(result: Any) -> str:
    """兼容不同检索结果对象格式，统一取出 content。"""
    if hasattr(result, "content"):
        return str(result.content)
    if isinstance(result, dict):
        return str(result.get("content", ""))
    return str(result)

def percentile(values: List[float], p: float) -> float:
    """最近秩法计算分位数，例如 p=0.95。"""
    if not values:
        return 0.0

    sorted_values = sorted(values)
    idx = math.ceil(len(sorted_values) * p) - 1
    idx = max(0, min(idx, len(sorted_values) - 1))

    return sorted_values[idx]

def main() -> None:
    parser = argparse.ArgumentParser(description="RAG 检索评测脚本")
    parser.add_argument(
        "--eval-file",
        default="tests/evaluation/eval_set.jsonl",
        help="评测集 JSONL 路径"
    )
    parser.add_argument("--top-k", type=int, default=3, help="检索返回条数")
    parser.add_argument("--output", default=None, help="指标结果输出 JSON 路径")
    parser.add_argument(
        "--save-details",
        action="store_true",
        help="保存每条查询的明细到 <eval_file>_details.jsonl"
    )
    args = parser.parse_args()

    eval_path = Path(args.eval_file)
    if not eval_path.exists():
        raise FileNotFoundError(f"评测集不存在: {eval_path}")
    eval_set = load_jsonl(eval_path)
    if not eval_set:
        raise ValueError(f"评测集为空: {eval_path}")

    retriever = Retriever()
    hit_count = 0
    recall_num = 0
    recall_den = 0
    latencies: List[float] = []
    details: List[Dict[str, Any]] = []
    error_count = 0

    total = len(eval_set)

    for idx, item in enumerate(eval_set, 1):
        query = item["query"]
        gold_spans = item["gold_spans"]
        start = time.perf_counter()
        error = None

        try:
            results = retriever.search(query, top_k=args.top_k)
        except Exception as e:
            results = []
            error = f"{type(e).__name__}: {e}"
            error_count += 1

        latency = time.perf_counter() - start
        latencies.append(latency)

        retrieved_text = "\n".join(get_content(result) for result in results)
        matched_spans = [span for span in gold_spans if span in retrieved_text]
        missed_spans = [span for span in gold_spans if span not in retrieved_text]
        hit = len(matched_spans) > 0

        if hit:
            hit_count += 1
        recall_num += len(matched_spans)
        recall_den += len(gold_spans)

        details.append(
            {
                "id": item.get("id", idx),
                "query": query,
                "gold_spans": gold_spans,
                "matched_spans": matched_spans,
                "missed_spans": missed_spans,
                "hit": hit,
                "latency_ms": latency * 1000,
                "error": error,
                "retireved": [
                    {
                        "content": get_content(result),
                    }
                    for result in results
                ]

            }
        )

        if idx % 10 == 0 or idx == total:
            print(f"[{idx}/{total}] 已完成", file=sys.stderr)

    hit_rate = hit_count / total
    context_recall = recall_num / recall_den if recall_den else 0.0
    avg_latency = sum(latencies) / len(latencies) if latencies else 0.0
    p95_latency = percentile(latencies, 0.95)

    metrics = {
        "eval_file": str(eval_path),
        "top_k": args.top_k,
        "num_queries": total,
        "hit_count": hit_count,
        "hit_rate": hit_rate,
        "context_recall": context_recall,
        "recall_num": recall_num,
        "recall_den": recall_den,
        "avg_latency_ms": avg_latency * 1000,
        "p95_latency_ms": p95_latency * 1000,
        "error_count": error_count,
    }

    print(json.dumps(metrics, ensure_ascii=False, indent=2))

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(metrics, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"指标已写入: {output_path}", file=sys.stderr)

    if args.save_details:
        details_path = eval_path.with_name(eval_path.stem + "_details.jsonl")
        with details_path.open("w", encoding="utf-8") as f:
            for detail in details:
                f.write(json.dumps(detail, ensure_ascii=False) + "\n")
        print(f"明细已写入: {details_path}", file=sys.stderr)

if __name__ == "__main__":
    main()

