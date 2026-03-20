"""
eval_agent_parallel.py —— Agentic RAG（create_react_agent + @tool）8卡并行评估主控
与 eval_rag_parallel.py 的区别：
  - 调用 eval_agent_worker.py（真正的 Agent，LLM 自主决定是否检索）
  - 结果保存到 eval_results_agent/
  - 汇总报告额外统计 tool_call_count（工具调用次数分布）

用法：
  python eval_agent_parallel.py [--num_gpus 8] [--save_dir ./eval_results_agent]
"""

import argparse
import json
import os
import subprocess
import sys
import threading
import datetime
import time
from collections import Counter

from eval_questions import QUESTIONS

# ============================================================
# 参数
# ============================================================
parser = argparse.ArgumentParser()
parser.add_argument("--num_gpus", type=int, default=8,  help="使用的 GPU 数量")
parser.add_argument("--save_dir", type=str, default="./eval_results_agent")
args = parser.parse_args()

NUM_GPUS = args.num_gpus
SAVE_DIR = args.save_dir
os.makedirs(SAVE_DIR, exist_ok=True)

timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

# ============================================================
# 1. 展开所有题目，按顺序编号
# ============================================================
all_questions = []
for category, qs in QUESTIONS.items():
    for idx, q in enumerate(qs, 1):
        all_questions.append({
            "category": category,
            "idx":      idx,
            "question": q,
        })

total = len(all_questions)
print(f"[Agentic RAG 评估] 共 {total} 道题，分配给 {NUM_GPUS} 张 GPU")

# ============================================================
# 2. 均匀分片
# ============================================================
shards = [[] for _ in range(NUM_GPUS)]
for i, item in enumerate(all_questions):
    shards[i % NUM_GPUS].append(item)

for i, shard in enumerate(shards):
    labels = ", ".join(f"{s['category']}Q{s['idx']}" for s in shard)
    print(f"  GPU {i}: {len(shard)} 题  ({labels})")

# ============================================================
# 3. 实时打印子进程输出
# ============================================================
def stream_output(proc, shard_id, log_path):
    with open(log_path, "w", encoding="utf-8") as log_f:
        for line in iter(proc.stdout.readline, b""):
            text = line.decode("utf-8", errors="replace").rstrip()
            print(f"[GPU{shard_id}] {text}", flush=True)
            log_f.write(text + "\n")
            log_f.flush()

# ============================================================
# 4. 启动所有子进程
# ============================================================
processes = []
threads   = []

print(f"\n{'='*60}")
print(f"启动 {NUM_GPUS} 个并行 Worker（Agentic RAG 模式）...")
print(f"{'='*60}\n")

for shard_id in range(NUM_GPUS):
    shard_data    = shards[shard_id]
    questions_str = json.dumps(shard_data, ensure_ascii=False)
    log_path      = os.path.join(SAVE_DIR, f"worker_{shard_id:02d}_{timestamp}.log")

    cmd = [
        sys.executable, "eval_agent_worker.py",
        "--shard_id",       str(shard_id),
        "--questions_json", questions_str,
        "--save_dir",       SAVE_DIR,
        "--timestamp",      timestamp,
    ]

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(shard_id)

    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        env=env,
        cwd=os.path.dirname(os.path.abspath(__file__)),
    )
    processes.append(proc)

    t = threading.Thread(
        target=stream_output,
        args=(proc, shard_id, log_path),
        daemon=True
    )
    t.start()
    threads.append(t)

    print(f"  ✅ Worker {shard_id} 已启动  PID={proc.pid}  GPU={shard_id}  "
          f"题数={len(shard_data)}  日志={log_path}", flush=True)

# ============================================================
# 5. 等待所有子进程完成
# ============================================================
print(f"\n等待所有 Worker 完成...\n")
t_start = time.time()

return_codes = []
for shard_id, proc in enumerate(processes):
    rc = proc.wait()
    return_codes.append(rc)
    status = "✅ 成功" if rc == 0 else f"❌ 失败(rc={rc})"
    print(f"  Worker {shard_id} {status}", flush=True)

for t in threads:
    t.join(timeout=5)

total_elapsed = time.time() - t_start
print(f"\n所有 Worker 完成，总耗时: {total_elapsed:.1f}s", flush=True)

# ============================================================
# 6. 合并分片结果
# ============================================================
print(f"\n{'='*60}")
print("合并分片结果...")
print(f"{'='*60}")

all_records = []
for shard_id in range(NUM_GPUS):
    shard_json = os.path.join(SAVE_DIR, f"shard_{shard_id:02d}_{timestamp}.json")
    if not os.path.exists(shard_json):
        print(f"  ⚠️  Worker {shard_id} 的结果文件不存在: {shard_json}")
        continue
    try:
        with open(shard_json, "r", encoding="utf-8") as f:
            records = json.load(f)
        all_records.extend(records)
        print(f"  Worker {shard_id}: 读取 {len(records)} 条记录")
    except Exception as e:
        print(f"  ⚠️  Worker {shard_id} 结果解析失败: {e}")

# 按 category + question_idx 排序
cat_order = {cat: i for i, cat in enumerate(QUESTIONS.keys())}
all_records.sort(key=lambda r: (cat_order.get(r["category"], 99), r["question_idx"]))

# 写入合并后的完整 JSON
merged_json = os.path.join(SAVE_DIR, f"eval_agent_detail_{timestamp}.json")
with open(merged_json, "w", encoding="utf-8") as f:
    json.dump(all_records, f, ensure_ascii=False, indent=2)
print(f"\n合并 JSON 已保存: {merged_json}  (共 {len(all_records)} 条)")

# ============================================================
# 7. 生成汇总报告
# ============================================================
total_ok   = sum(1 for r in all_records if r["error_type"] == "OK")
global_err = Counter(r["error_type"] for r in all_records)

# 工具调用次数分布（Agentic RAG 特有统计）
tool_call_dist = Counter(r.get("tool_call_count", 0) for r in all_records)
triggered_cnt  = sum(1 for r in all_records if r.get("retrieval_triggered", False))
not_triggered  = len(all_records) - triggered_cnt

category_stats = {}
for cat in QUESTIONS:
    cat_records = [r for r in all_records if r["category"] == cat]
    ok_cnt      = sum(1 for r in cat_records if r["error_type"] == "OK")
    err_dist    = Counter(r["error_type"] for r in cat_records)
    avg_elapsed = (sum(r["elapsed_sec"] for r in cat_records) / len(cat_records)
                   if cat_records else 0)
    avg_tool_calls = (sum(r.get("tool_call_count", 0) for r in cat_records) / len(cat_records)
                      if cat_records else 0)
    triggered_in_cat = sum(1 for r in cat_records if r.get("retrieval_triggered", False))
    category_stats[cat] = {
        "total":           len(cat_records),
        "ok":              ok_cnt,
        "ok_rate":         round(ok_cnt / len(cat_records), 3) if cat_records else 0,
        "error_dist":      dict(err_dist),
        "avg_elapsed":     round(avg_elapsed, 2),
        "avg_tool_calls":  round(avg_tool_calls, 2),
        "triggered_rate":  round(triggered_in_cat / len(cat_records), 3) if cat_records else 0,
    }

report_path = os.path.join(SAVE_DIR, f"eval_agent_report_{timestamp}.txt")
with open(report_path, "w", encoding="utf-8") as f:
    f.write("=" * 70 + "\n")
    f.write(f"Agentic RAG 评估报告（8卡并行）  生成时间: {timestamp}\n")
    f.write(f"总耗时: {total_elapsed:.1f}s  (串行预估: {total_elapsed * NUM_GPUS:.0f}s)\n")
    f.write("=" * 70 + "\n\n")

    f.write(f"总题数: {len(all_records)}   总体 OK 率: {total_ok}/{len(all_records)} "
            f"({100*total_ok/len(all_records):.1f}%)\n\n")

    f.write("【Agentic RAG 特有统计】\n")
    f.write(f"  检索触发: {triggered_cnt}/{len(all_records)}  "
            f"未触发（LLM 自主跳过）: {not_triggered}\n")
    f.write(f"  工具调用次数分布: {dict(sorted(tool_call_dist.items()))}\n\n")

    f.write("全局错误类型分布:\n")
    for et, cnt in global_err.most_common():
        bar = "█" * cnt
        f.write(f"  {et:<25} : {cnt:>3}  {bar}\n")
    f.write("\n")

    f.write("各类别统计:\n")
    f.write(f"  {'类别':<25} {'题数':>4} {'OK':>4} {'OK率':>7} "
            f"{'触发率':>7} {'平均工具调用':>10} {'平均耗时':>8}  错误分布\n")
    f.write("  " + "-" * 80 + "\n")
    for cat, stat in category_stats.items():
        f.write(f"  {cat:<25} {stat['total']:>4} {stat['ok']:>4} "
                f"{stat['ok_rate']*100:>6.1f}% {stat['triggered_rate']*100:>6.1f}% "
                f"{stat['avg_tool_calls']:>10.2f} {stat['avg_elapsed']:>7.1f}s  "
                f"{stat['error_dist']}\n")
    f.write("\n")

    f.write("=" * 70 + "\n")
    f.write("逐题明细:\n")
    f.write("=" * 70 + "\n")
    for r in all_records:
        icon = "✅" if r["error_type"] == "OK" else "❌"
        f.write(f"\n{icon} [{r['category']}] Q{r['question_idx']}: {r['question']}\n")
        f.write(f"  检索触发: {r['retrieval_triggered']}  工具调用次数: {r.get('tool_call_count', 0)}\n")
        f.write(f"  检索 queries: {r.get('tool_queries', [])}\n")
        f.write(f"  相似度: {r['retrieval_score']}  命中(≥0.5): {r['retrieval_hit']}\n")
        f.write(f"  回答类型: {r['answer_type']}  错误类型: {r['error_type']}\n")
        f.write(f"  来源数量: {r['unique_source_count']}  耗时: {r['elapsed_sec']}s\n")
        f.write(f"  完整回答:\n")
        for line in r['raw_answer'].splitlines():
            f.write(f"    {line}\n")
        f.write(f"  来源:\n")
        for s in r['sources']:
            f.write(f"    - [{s['title']}] {s['source']}\n")
            f.write(f"      摘要: {s['snippet']}\n")
        f.write("-" * 70 + "\n")

print(f"汇总报告已保存: {report_path}")

# ============================================================
# 8. 控制台总结
# ============================================================
print(f"\n{'='*60}")
print("Agentic RAG 评估完成！总体统计：")
print(f"{'='*60}")
print(f"  {'类别':<25} {'OK率':>7} {'触发率':>7} {'平均工具调用':>10}  错误分布")
print("  " + "-" * 70)
for cat, stat in category_stats.items():
    print(f"  {cat:<25} {stat['ok_rate']*100:>6.1f}% "
          f"{stat['triggered_rate']*100:>6.1f}% "
          f"{stat['avg_tool_calls']:>10.2f}  {stat['error_dist']}")
print(f"\n总体 OK 率: {total_ok}/{len(all_records)} ({100*total_ok/len(all_records):.1f}%)")
print(f"检索触发: {triggered_cnt}/{len(all_records)}  未触发（LLM 自主跳过）: {not_triggered}")
print(f"工具调用次数分布: {dict(sorted(tool_call_dist.items()))}")
print(f"总耗时: {total_elapsed:.1f}s  (串行预估: {total_elapsed * NUM_GPUS:.0f}s)")
print(f"\n结果文件:")
print(f"  {merged_json}")
print(f"  {report_path}")
print(f"  日志目录: {SAVE_DIR}/worker_XX_{timestamp}.log")
