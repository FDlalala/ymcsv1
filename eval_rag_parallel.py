"""
eval_rag_parallel.py —— 8卡并行评估主控脚本
功能：
  1. 将 40 道题均匀分配给 8 个 GPU（每卡 5 题）
  2. 用 subprocess 启动 8 个 eval_rag_worker.py 进程，每个绑定一张 GPU
  3. 实时打印每个进程的 stdout/stderr（带 Worker 前缀）
  4. 全部完成后合并 8 个分片 JSON → 一个完整 JSON + 一份汇总报告

用法：
  python eval_rag_parallel.py [--num_gpus 8] [--save_dir ./eval_results]
"""

import argparse
import json
import os
import subprocess
import sys
import threading
import datetime
import time

from eval_questions import QUESTIONS

# ============================================================
# 参数
# ============================================================
parser = argparse.ArgumentParser()
parser.add_argument("--num_gpus", type=int, default=8,  help="使用的 GPU 数量")
parser.add_argument("--save_dir", type=str, default="./eval_results")
args = parser.parse_args()

NUM_GPUS  = args.num_gpus
SAVE_DIR  = args.save_dir
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
print(f"共 {total} 道题，分配给 {NUM_GPUS} 张 GPU")

# ============================================================
# 2. 均匀分片
# ============================================================
shards = [[] for _ in range(NUM_GPUS)]
for i, item in enumerate(all_questions):
    shards[i % NUM_GPUS].append(item)

for i, shard in enumerate(shards):
    shard_labels = ', '.join(s['category'] + 'Q' + str(s['idx']) for s in shard)
    print(f"  GPU {i}: {len(shard)} 题  ({shard_labels})")

# ============================================================
# 3. 实时打印子进程输出的辅助函数
# ============================================================
def stream_output(proc, shard_id, log_path):
    """将子进程的 stdout 实时打印到控制台，并同时写入 log 文件"""
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
print(f"启动 {NUM_GPUS} 个并行 Worker...")
print(f"{'='*60}\n")

for shard_id in range(NUM_GPUS):
    shard_data    = shards[shard_id]
    questions_str = json.dumps(shard_data, ensure_ascii=False)
    log_path      = os.path.join(SAVE_DIR, f"worker_{shard_id:02d}_{timestamp}.log")

    cmd = [
        sys.executable, "eval_rag_worker.py",
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
        stderr=subprocess.STDOUT,   # 合并 stderr 到 stdout
        env=env,
        cwd=os.path.dirname(os.path.abspath(__file__)),
    )
    processes.append(proc)

    # 每个进程一个线程负责实时打印
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

# 等待所有打印线程结束
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

# 按 category + question_idx 排序，恢复原始顺序
cat_order = {cat: i for i, cat in enumerate(QUESTIONS.keys())}
all_records.sort(key=lambda r: (cat_order.get(r["category"], 99), r["question_idx"]))

# 写入合并后的完整 JSON
merged_json = os.path.join(SAVE_DIR, f"eval_detail_{timestamp}.json")
with open(merged_json, "w", encoding="utf-8") as f:
    json.dump(all_records, f, ensure_ascii=False, indent=2)
print(f"\n合并 JSON 已保存: {merged_json}  (共 {len(all_records)} 条)")

# ============================================================
# 7. 生成汇总报告
# ============================================================
REFUSE_MARKER = "知识库中没有找到相关信息"

total_ok   = sum(1 for r in all_records if r["error_type"] == "OK")
global_err = {}
for r in all_records:
    et = r["error_type"]
    global_err[et] = global_err.get(et, 0) + 1

category_stats = {}
for cat in QUESTIONS:
    cat_records = [r for r in all_records if r["category"] == cat]
    ok_cnt = sum(1 for r in cat_records if r["error_type"] == "OK")
    err_dist = {}
    for r in cat_records:
        et = r["error_type"]
        err_dist[et] = err_dist.get(et, 0) + 1
    category_stats[cat] = {
        "total":      len(cat_records),
        "ok":         ok_cnt,
        "ok_rate":    round(ok_cnt / len(cat_records), 3) if cat_records else 0,
        "error_dist": err_dist,
    }

report_path = os.path.join(SAVE_DIR, f"eval_report_{timestamp}.txt")
with open(report_path, "w", encoding="utf-8") as f:
    f.write("=" * 70 + "\n")
    f.write(f"RAG 评估报告（8卡并行）  生成时间: {timestamp}\n")
    f.write(f"总耗时: {total_elapsed:.1f}s  (串行预估: {total_elapsed * NUM_GPUS:.0f}s)\n")
    f.write("=" * 70 + "\n\n")

    f.write(f"总题数: {len(all_records)}   总体 OK 率: {total_ok}/{len(all_records)} "
            f"({100*total_ok/len(all_records):.1f}%)\n\n")

    f.write("全局错误类型分布:\n")
    for et, cnt in sorted(global_err.items(), key=lambda x: -x[1]):
        f.write(f"  {et:<25} : {cnt}\n")
    f.write("\n")

    f.write("各类别统计:\n")
    for cat, stat in category_stats.items():
        f.write(f"  {cat:<25} OK率: {stat['ok_rate']*100:.1f}%  "
                f"错误: {stat['error_dist']}\n")
    f.write("\n")

    f.write("=" * 70 + "\n")
    f.write("逐题明细:\n")
    f.write("=" * 70 + "\n")
    for r in all_records:
        f.write(f"\n[{r['category']}] Q{r['question_idx']}: {r['question']}\n")
        f.write(f"  检索触发: {r['retrieval_triggered']}  命中: {r['retrieval_hit']}\n")
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
print("评估完成！总体统计：")
print(f"{'='*60}")
for cat, stat in category_stats.items():
    print(f"  {cat:<25} OK率: {stat['ok_rate']*100:.1f}%  错误: {stat['error_dist']}")
print(f"\n总体 OK 率: {total_ok}/{len(all_records)} ({100*total_ok/len(all_records):.1f}%)")
print(f"总耗时: {total_elapsed:.1f}s  (串行预估: {total_elapsed * NUM_GPUS:.0f}s)")
print(f"\n结果文件:\n  {merged_json}\n  {report_path}")
