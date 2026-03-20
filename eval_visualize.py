"""
eval_visualize.py —— RAG 评估结果统计分析 + 可视化
用法：
  # 单次分析
  python eval_visualize.py [--json eval_results/eval_detail_xxx.json] [--out_dir eval_results]

  # 对比两次评估（agent vs local）
  python eval_visualize.py \
      --json      eval_results/eval_detail_xxx.json \
      --compare   eval_results_local/eval_detail_yyy.json \
      --label_a   "Agent" --label_b "Local" \
      --out_dir   eval_results

功能：
  1. 统计各类别 OK 率、错误类型分布
  2. 检测  2. 检测"幻觉续写"（回答中出现 Human:/assistant: 等对话标记）
  3. 检测"来源重复"（所有来源指向同一 URL）
  4. 生成 6 张图表保存为 PNG
  5. 打印详细文字分析报告
"""

import argparse
import json
import os
import glob
import re
import textwrap
from collections import Counter, defaultdict

import matplotlib
matplotlib.use("Agg")   # 无显示器环境
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# ============================================================
# 中文字体（服务器上常见路径，找不到则用英文）
# ============================================================
import matplotlib.font_manager as fm
_FONT_CANDIDATES = [
    "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/noto-cjk/NotoSansCJK-Regular.ttc",
]
_font_path = next((p for p in _FONT_CANDIDATES if os.path.exists(p)), None)
if _font_path:
    fm.fontManager.addfont(_font_path)
    plt.rcParams["font.family"] = fm.FontProperties(fname=_font_path).get_name()
else:
    plt.rcParams["font.family"] = "DejaVu Sans"
plt.rcParams["axes.unicode_minus"] = False

# ============================================================
# 参数
# ============================================================
parser = argparse.ArgumentParser()
parser.add_argument("--json",     type=str, default=None,
                    help="指定 eval_detail_*.json 路径，默认自动找最新的")
parser.add_argument("--compare",  type=str, default=None,
                    help="第二份评估结果 JSON，用于对比")
parser.add_argument("--label_a",  type=str, default="Agent",
                    help="第一份结果的标签名")
parser.add_argument("--label_b",  type=str, default="Local",
                    help="第二份结果的标签名")
parser.add_argument("--out_dir",  type=str, default="./eval_results")
args = parser.parse_args()

# 自动找最新 JSON（兼容 eval_detail_*.json 和 eval_local_detail_*.json）
if args.json:
    json_path = args.json
else:
    candidates = (
        glob.glob(os.path.join(args.out_dir, "eval_detail_*.json")) +
        glob.glob(os.path.join(args.out_dir, "eval_local_detail_*.json"))
    )
    if not candidates:
        raise FileNotFoundError(
            f"在 {args.out_dir} 下找不到 eval_detail_*.json 或 eval_local_detail_*.json\n"
            f"请用 --json 参数手动指定文件路径"
        )
    json_path = sorted(candidates)[-1]

print(f"读取评估结果: {json_path}")
with open(json_path, "r", encoding="utf-8") as f:
    records = json.load(f)

ts = re.search(r"eval_(?:local_)?detail_(\d+_\d+)", json_path)
ts = ts.group(1) if ts else "unknown"
os.makedirs(args.out_dir, exist_ok=True)

# ============================================================
# 辅助：检测幻觉续写（模型把对话历史当内容继续生成）
# ============================================================
HALLUCINATION_PATTERNS = [
    r"\bHuman\s*:",
    r"\bassistant\s*\n",
    r"请根据以下问题进行翻译",
    r"用英语回复上述内容",
    r"翻译：",
]
def has_hallucination_continuation(text: str) -> bool:
    for pat in HALLUCINATION_PATTERNS:
        if re.search(pat, text):
            return True
    return False

# ============================================================
# 辅助：检测来源是否全部重复（所有 source URL 相同）
# ============================================================
def all_sources_same(sources: list) -> bool:
    urls = [s.get("source", "") for s in sources]
    return len(set(urls)) == 1 and len(urls) > 1

# ============================================================
# 数据增强：补充细粒度标签
# ============================================================
CATEGORY_LABELS = {
    "A_exact_grounding": "A Grounding",
    "B_reasoning":       "B Reasoning",
    "C_boundary":        "C Boundary",
    "D_out_of_domain":   "D OutOfDomain",
}
CATEGORY_EXPECTED = {
    "A_exact_grounding": True,
    "B_reasoning":       True,
    "C_boundary":        True,
    "D_out_of_domain":   False,
}

for r in records:
    r["hallucination_continuation"] = has_hallucination_continuation(r.get("raw_answer", ""))
    r["all_sources_same"]           = all_sources_same(r.get("sources", []))
    r["expected_to_answer"]         = CATEGORY_EXPECTED.get(r["category"], True)
    r["cat_label"]                  = CATEGORY_LABELS.get(r["category"], r["category"])

# ============================================================
# 统计
# ============================================================
total = len(records)
categories = list(CATEGORY_LABELS.keys())

# 1. 各类别 OK 率
cat_ok    = {c: 0 for c in categories}
cat_total = {c: 0 for c in categories}
for r in records:
    cat_total[r["category"]] += 1
    if r["error_type"] == "OK":
        cat_ok[r["category"]] += 1
cat_ok_rate = {c: cat_ok[c] / cat_total[c] if cat_total[c] else 0 for c in categories}

# 2. 全局错误类型分布
error_counter = Counter(r["error_type"] for r in records)

# 3. 检索触发 & 命中（基于余弦相似度阈值）
retrieval_triggered = sum(1 for r in records if r["retrieval_triggered"])
retrieval_hit       = sum(1 for r in records if r.get("retrieval_hit", False))
retrieval_miss      = retrieval_triggered - retrieval_hit
no_retrieval        = total - retrieval_triggered

# 3b. retrieval_score 分布（连续值）
all_scores = [r.get("retrieval_score", 0.0) for r in records if r["retrieval_triggered"]]
avg_retrieval_score = float(np.mean(all_scores)) if all_scores else 0.0

# 4. 幻觉续写
hallucination_cont  = sum(1 for r in records if r["hallucination_continuation"])

# 5. 来源全重复
all_same_src        = sum(1 for r in records if r["all_sources_same"])

# 6. 回答类型
answered_cnt = sum(1 for r in records if r["answer_type"] == "answered")
refused_cnt  = sum(1 for r in records if r["answer_type"] == "refused")

# 9. 拒绝质量分析
# 好拒绝：D 类（领域外）且拒绝了 → 正确行为
# 坏拒绝：A/B/C 类（应回答）但拒绝了 → SHOULD_ANSWER
# 好回答：A/B/C 类且回答了 → 正确行为
# 坏回答（幻觉）：D 类但回答了 → HALLUCINATION
good_refuse  = sum(1 for r in records
                   if r["answer_type"] == "refused" and not r["expected_to_answer"])
bad_refuse   = sum(1 for r in records
                   if r["answer_type"] == "refused" and r["expected_to_answer"])
good_answer  = sum(1 for r in records
                   if r["answer_type"] == "answered" and r["expected_to_answer"])
bad_answer   = sum(1 for r in records
                   if r["answer_type"] == "answered" and not r["expected_to_answer"])
# 拒绝精度：在所有拒绝中，有多少是「该拒绝的」
refuse_precision = good_refuse / refused_cnt if refused_cnt > 0 else 0.0
# 拒绝召回：在所有「该拒绝的题」中，有多少被正确拒绝了
d_total = sum(1 for r in records if not r["expected_to_answer"])
refuse_recall    = good_refuse / d_total if d_total > 0 else 0.0

# 7. 各类别错误分布
cat_err_dist = defaultdict(Counter)
for r in records:
    cat_err_dist[r["category"]][r["error_type"]] += 1

# 8. 平均耗时
avg_elapsed = {c: np.mean([r["elapsed_sec"] for r in records if r["category"] == c])
               for c in categories}

# ============================================================
# 文字报告
# ============================================================
SEP = "=" * 70
sep = "-" * 70

print(f"\n{SEP}")
print(f"  RAG 评估结果分析报告  ({ts})")
print(SEP)
print(f"\n总题数: {total}   总体 OK 率: {cat_ok['A_exact_grounding']+cat_ok['B_reasoning']+cat_ok['C_boundary']+cat_ok['D_out_of_domain']}/{total} "
      f"({100*sum(cat_ok.values())/total:.1f}%)")
print(f"检索触发: {retrieval_triggered}/{total}  命中(≥0.5): {retrieval_hit}/{retrieval_triggered}  "
      f"平均相似度: {avg_retrieval_score:.4f}  未触发: {no_retrieval}")
print(f"幻觉续写（模型把对话历史当内容）: {hallucination_cont} 条")
print(f"来源全部重复（所有来源同一URL）: {all_same_src} 条")
print(f"回答: {answered_cnt}  拒绝: {refused_cnt}")
print(f"拒绝质量 → 好拒绝(D类正确拒绝): {good_refuse}  坏拒绝(A/B/C类误拒): {bad_refuse}  "
      f"拒绝精度: {refuse_precision*100:.1f}%  拒绝召回: {refuse_recall*100:.1f}%")
print(f"回答质量 → 好回答(A/B/C类正确答): {good_answer}  坏回答(D类幻觉): {bad_answer}")

print(f"\n{sep}")
print("各类别统计:")
print(f"{'类别':<20} {'题数':>4} {'OK':>4} {'OK率':>7} {'平均耗时':>8}  错误分布")
print(sep)
for c in categories:
    label = CATEGORY_LABELS[c]
    print(f"{label:<20} {cat_total[c]:>4} {cat_ok[c]:>4} {cat_ok_rate[c]*100:>6.1f}%"
          f" {avg_elapsed[c]:>7.1f}s  {dict(cat_err_dist[c])}")

print(f"\n{sep}")
print("全局错误类型分布:")
for et, cnt in error_counter.most_common():
    bar = "█" * cnt
    print(f"  {et:<25} {cnt:>3}  {bar}")

print(f"\n{sep}")
print("关键问题诊断:")
print()

# 诊断1：RETRIEVAL_MISS 分析
miss_records = [r for r in records if r["error_type"] == "RETRIEVAL_MISS"]
print(f"【1】RETRIEVAL_MISS（检索触发但余弦相似度 < 0.5）: {len(miss_records)} 条")
if miss_records:
    miss_scores = [r.get("retrieval_score", 0.0) for r in miss_records]
    print(f"    MISS 题的平均相似度: {np.mean(miss_scores):.4f}  "
          f"最高: {max(miss_scores):.4f}  最低: {min(miss_scores):.4f}")
print("    原因：向量库中缺少对应内容，或 bge-small 语义能力不足导致相似度偏低。")
print("    典型案例:")
for r in miss_records[:3]:
    print(f"      Q: {r['question'][:50]}...")
    print(f"         相似度: {r.get('retrieval_score', '?')}  各文档: {r.get('doc_scores', [])}")
    print(f"         来源: {r['sources'][0]['title'][:40] if r['sources'] else '无'}...")

print()
# 诊断2：SHOULD_ANSWER 分析
should_ans = [r for r in records if r["error_type"] == "SHOULD_ANSWER"]
print(f"【2】SHOULD_ANSWER（应回答但拒绝了，坏拒绝）: {len(should_ans)} 条")
print(f"    拒绝精度: {refuse_precision*100:.1f}%  拒绝召回: {refuse_recall*100:.1f}%")
print("    原因：检索到的内容与问题语义相关，但 LLM 判断为无关，过度保守。")
for r in should_ans[:3]:
    print(f"      Q: {r['question'][:50]}...")
    print(f"         检索相似度: {r.get('retrieval_score', '?')}  "
          f"来源: {r['sources'][0]['title'][:40] if r['sources'] else '无'}...")

print()
# 诊断2b：拒绝质量汇总
print(f"【2b】拒绝质量汇总:")
print(f"    好拒绝（D类正确拒绝）: {good_refuse}/{d_total}  "
      f"拒绝召回率: {refuse_recall*100:.1f}%")
print(f"    坏拒绝（A/B/C类误拒）: {bad_refuse}  "
      f"拒绝精度: {refuse_precision*100:.1f}%")
print(f"    坏回答（D类幻觉）: {bad_answer}")

print()
# 诊断3：幻觉续写
hall_records = [r for r in records if r["hallucination_continuation"]]
print(f"【3】幻觉续写（模型在回答后继续生成对话/翻译等无关内容）: {len(hall_records)} 条")
print("    原因：LLM 的 stop token 未正确设置，或 max_new_tokens 过大。")
print("    建议：在 pipeline 中添加 stop sequences，或后处理截断。")
for r in hall_records[:2]:
    snippet = r["raw_answer"][:200].replace("\n", " ")
    print(f"      Q: {r['question'][:40]}...")
    print(f"         回答片段: {snippet}...")

print()
# 诊断4：D 类（领域外）
d_records = [r for r in records if r["category"] == "D_out_of_domain"]
d_hallucination = [r for r in d_records if r["answer_type"] == "answered"]
print(f"【4】D 类（领域外）应拒绝 {len(d_records)} 题，实际回答了 {len(d_hallucination)} 题（幻觉）")
for r in d_hallucination[:3]:
    print(f"      Q: {r['question'][:50]}...")

print(f"\n{SEP}\n")

# ============================================================
# 可视化：6 张图
# ============================================================
fig, axes = plt.subplots(2, 3, figsize=(18, 11))
fig.suptitle(f"RAG Evaluation Report  ({ts})", fontsize=13, fontweight="bold", y=0.98)

cat_short = [CATEGORY_LABELS[c] for c in categories]
colors_ok  = ["#4CAF50", "#2196F3", "#FF9800", "#9C27B0"]
colors_err = ["#F44336", "#FF9800", "#2196F3", "#9C27B0", "#00BCD4", "#795548"]

# ---- 图1：各类别 OK 率柱状图 ----
ax = axes[0, 0]
ok_rates = [cat_ok_rate[c] * 100 for c in categories]
bars = ax.bar(cat_short, ok_rates, color=colors_ok, edgecolor="white", linewidth=1.2)
ax.set_ylim(0, 110)
ax.set_ylabel("OK Rate (%)")
ax.set_title("OK Rate by Category")
for bar, v in zip(bars, ok_rates):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
            f"{v:.0f}%", ha="center", va="bottom", fontsize=10, fontweight="bold")
ax.axhline(y=sum(ok_rates)/len(ok_rates), color="gray", linestyle="--", linewidth=1, label="Avg")
ax.legend(fontsize=9)

# ---- 图2：全局错误类型饼图 ----
ax = axes[0, 1]
err_labels = list(error_counter.keys())
err_values = list(error_counter.values())
wedge_colors = colors_err[:len(err_labels)]
wedges, texts, autotexts = ax.pie(
    err_values, labels=err_labels, autopct="%1.0f%%",
    colors=wedge_colors, startangle=140,
    textprops={"fontsize": 8},
    wedgeprops={"edgecolor": "white", "linewidth": 1.5}
)
for at in autotexts:
    at.set_fontsize(8)
ax.set_title("Global Error Type Distribution")

# ---- 图3：retrieval_score 分布直方图 ----
ax = axes[0, 2]
if all_scores:
    n_bins = min(20, len(all_scores))
    counts, bin_edges, patches = ax.hist(
        all_scores, bins=n_bins, range=(0, 1),
        color="#2196F3", edgecolor="white", linewidth=0.8, alpha=0.85
    )
    # 阈值线
    ax.axvline(x=0.5, color="#F44336", linestyle="--", linewidth=1.8,
               label=f"Thr=0.5 (hit {retrieval_hit}/{retrieval_triggered})")
    # mean line
    ax.axvline(x=avg_retrieval_score, color="#FF9800", linestyle="-.", linewidth=1.5,
               label=f"Mean={avg_retrieval_score:.3f}")
    ax.set_xlabel("Cosine Similarity")
    ax.set_ylabel("Count")
    ax.set_title("Retrieval Similarity Distribution")
    ax.legend(fontsize=8)
else:
    ax.text(0.5, 0.5, "No retrieval_score data",
            ha="center", va="center", transform=ax.transAxes, fontsize=10)
    ax.set_title("Retrieval Similarity Distribution")

# ---- 图4：各类别错误类型堆叠柱 ----
ax = axes[1, 0]
all_err_types = sorted(set(r["error_type"] for r in records))
err_type_colors = {et: colors_err[i % len(colors_err)] for i, et in enumerate(all_err_types)}
bottom = np.zeros(len(categories))
for et in all_err_types:
    vals = [cat_err_dist[c].get(et, 0) for c in categories]
    ax.bar(cat_short, vals, bottom=bottom,
           label=et, color=err_type_colors[et], edgecolor="white", linewidth=0.8)
    bottom += np.array(vals)
ax.set_ylabel("Count")
ax.set_title("Error Type Stack by Category")
ax.legend(fontsize=7, loc="upper right")

# ---- 图5：拒绝质量分析（好拒绝/坏拒绝/好回答/坏回答） ----
ax = axes[1, 1]
quality_labels = ["Good Refuse\n(D correct)", "Bad Refuse\n(A/B/C wrong)", "Good Answer\n(A/B/C correct)", "Bad Answer\n(D halluc)"]
quality_values = [good_refuse, bad_refuse, good_answer, bad_answer]
quality_colors = ["#4CAF50", "#F44336", "#2196F3", "#E91E63"]
bars = ax.bar(quality_labels, quality_values, color=quality_colors,
              edgecolor="white", linewidth=1.2)
ax.set_ylabel("Count")
ax.set_title(f"Refuse Quality\nPrec={refuse_precision*100:.0f}%  Recall={refuse_recall*100:.0f}%")
for bar, v in zip(bars, quality_values):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
            str(v), ha="center", va="bottom", fontsize=11, fontweight="bold")

# ---- 图6：各类别平均耗时 ----
ax = axes[1, 2]
elapsed_vals = [avg_elapsed[c] for c in categories]
bars = ax.bar(cat_short, elapsed_vals, color=colors_ok, edgecolor="white", linewidth=1.2)
ax.set_ylabel("Avg Elapsed (s)")
ax.set_title("Avg Inference Time by Category")
for bar, v in zip(bars, elapsed_vals):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
            f"{v:.1f}s", ha="center", va="bottom", fontsize=10)

plt.tight_layout(rect=[0, 0, 1, 0.96])
chart_path = os.path.join(args.out_dir, f"eval_chart_{ts}.png")
plt.savefig(chart_path, dpi=150, bbox_inches="tight")
print(f"图表已保存: {chart_path}")

# ============================================================
# 额外：逐题细节热力图（OK=绿，各错误=红色系）
# ============================================================
fig2, ax2 = plt.subplots(figsize=(16, 8))

# 按 category 分组，每行一个类别
ERROR_COLOR_MAP = {
    "OK":               "#4CAF50",
    "RETRIEVAL_MISS":   "#FF5722",
    "SHOULD_ANSWER":    "#F44336",
    "HALLUCINATION":    "#E91E63",
    "DUPLICATE_SOURCE": "#FF9800",
    "NO_RETRIEVAL":     "#9E9E9E",
}

n_cols = max(cat_total.values())
n_rows = len(categories)
cell_w, cell_h = 1.0, 0.8

for row_i, cat in enumerate(categories):
    cat_recs = sorted([r for r in records if r["category"] == cat],
                      key=lambda x: x["question_idx"])
    for col_i, r in enumerate(cat_recs):
        color = ERROR_COLOR_MAP.get(r["error_type"], "#BDBDBD")
        rect = mpatches.FancyBboxPatch(
            (col_i * cell_w, row_i * cell_h),
            cell_w * 0.9, cell_h * 0.85,
            boxstyle="round,pad=0.05",
            facecolor=color, edgecolor="white", linewidth=1.5
        )
        ax2.add_patch(rect)
        # 题号
        ax2.text(col_i * cell_w + cell_w * 0.45,
                 row_i * cell_h + cell_h * 0.42,
                 f"Q{r['question_idx']}", ha="center", va="center",
                 fontsize=8, color="white", fontweight="bold")
        # 幻觉续写标记
        if r["hallucination_continuation"]:
            ax2.text(col_i * cell_w + cell_w * 0.8,
                     row_i * cell_h + cell_h * 0.7,
                     "!", ha="center", va="center",
                     fontsize=9, color="yellow", fontweight="bold")

# Y 轴标签
ax2.set_yticks([i * cell_h + cell_h * 0.4 for i in range(n_rows)])
ax2.set_yticklabels([CATEGORY_LABELS[c] for c in categories], fontsize=10)
ax2.set_xlim(-0.1, n_cols * cell_w)
ax2.set_ylim(-0.1, n_rows * cell_h)
ax2.set_xlabel("Question Index", fontsize=10)
ax2.set_title("Per-Question Evaluation Heatmap  (! = hallucination)", fontsize=12, fontweight="bold")
ax2.axis("off")
ax2.set_axis_on()
ax2.set_xticks([])

# 图例
legend_patches = [mpatches.Patch(color=v, label=k) for k, v in ERROR_COLOR_MAP.items()]
ax2.legend(handles=legend_patches, loc="lower right", fontsize=9,
           framealpha=0.9, ncol=3)

plt.tight_layout()
heatmap_path = os.path.join(args.out_dir, f"eval_heatmap_{ts}.png")
plt.savefig(heatmap_path, dpi=150, bbox_inches="tight")
print(f"热力图已保存: {heatmap_path}")

print(f"\n所有图表已保存到: {args.out_dir}")
print("  eval_chart_*.png   —— 6 张统计图")
print("  eval_heatmap_*.png —— 逐题热力图")

# ============================================================
# 对比图（仅当 --compare 指定时生成）
# ============================================================
if args.compare:
    if not os.path.exists(args.compare):
        print(f"\n⚠️  对比文件不存在: {args.compare}，跳过对比图")
    else:
        print(f"\n读取对比评估结果: {args.compare}")
        with open(args.compare, "r", encoding="utf-8") as f:
            records_b = json.load(f)

        ts_b = re.search(r"eval_(?:local_)?detail_(\d+_\d+)", args.compare)
        ts_b = ts_b.group(1) if ts_b else "compare"

        # 对 records_b 补充标签
        for r in records_b:
            r["hallucination_continuation"] = has_hallucination_continuation(r.get("raw_answer", ""))
            r["all_sources_same"]           = all_sources_same(r.get("sources", []))

        def compute_stats(recs):
            """计算一组记录的统计数据"""
            # 补充标签（compare 文件可能没有）
            for r in recs:
                if "expected_to_answer" not in r:
                    r["expected_to_answer"] = CATEGORY_EXPECTED.get(r["category"], True)
            cat_ok_r    = {c: 0 for c in categories}
            cat_tot_r   = {c: 0 for c in categories}
            for r in recs:
                cat_tot_r[r["category"]] += 1
                if r["error_type"] == "OK":
                    cat_ok_r[r["category"]] += 1
            ok_rate_r = {c: cat_ok_r[c] / cat_tot_r[c] if cat_tot_r[c] else 0
                         for c in categories}
            err_cnt_r = Counter(r["error_type"] for r in recs)
            hall_r    = sum(1 for r in recs if r["hallucination_continuation"])
            d_recs    = [r for r in recs if r["category"] == "D_out_of_domain"]
            d_hall_r  = sum(1 for r in d_recs if r["answer_type"] == "answered")
            total_ok_r = sum(cat_ok_r.values())
            # 拒绝质量
            good_ref_r = sum(1 for r in recs
                             if r["answer_type"] == "refused" and not r["expected_to_answer"])
            bad_ref_r  = sum(1 for r in recs
                             if r["answer_type"] == "refused" and r["expected_to_answer"])
            ref_total_r = sum(1 for r in recs if r["answer_type"] == "refused")
            d_tot_r     = sum(1 for r in recs if not r["expected_to_answer"])
            ref_prec_r  = good_ref_r / ref_total_r if ref_total_r > 0 else 0.0
            ref_rec_r   = good_ref_r / d_tot_r if d_tot_r > 0 else 0.0
            # 平均检索相似度
            scores_r = [r.get("retrieval_score", 0.0) for r in recs if r.get("retrieval_triggered")]
            avg_score_r = float(np.mean(scores_r)) if scores_r else 0.0
            return (ok_rate_r, err_cnt_r, hall_r, d_hall_r, total_ok_r,
                    good_ref_r, bad_ref_r, ref_prec_r, ref_rec_r, avg_score_r)
        (ok_a, err_a, hall_a, d_hall_a, total_ok_a,
         good_ref_a, bad_ref_a, ref_prec_a, ref_rec_a, avg_score_a) = compute_stats(records)
        (ok_b, err_b, hall_b, d_hall_b, total_ok_b,
         good_ref_b, bad_ref_b, ref_prec_b, ref_rec_b, avg_score_b) = compute_stats(records_b)

        label_a = args.label_a
        label_b = args.label_b

        # ── 对比图：2行3列 ──
        fig3, axes3 = plt.subplots(2, 3, figsize=(18, 10))
        fig3.suptitle(f"{label_a} vs {label_b}  Comparison Report",
                      fontsize=14, fontweight="bold", y=0.98)

        cat_short_cmp = [CATEGORY_LABELS[c] for c in categories]
        x = np.arange(len(categories))
        w = 0.35

        # ---- 对比图1：各类别 OK 率 ----
        ax = axes3[0, 0]
        vals_a = [ok_a[c] * 100 for c in categories]
        vals_b = [ok_b[c] * 100 for c in categories]
        bars_a = ax.bar(x - w/2, vals_a, w, label=label_a, color="#2196F3", edgecolor="white")
        bars_b = ax.bar(x + w/2, vals_b, w, label=label_b, color="#FF9800", edgecolor="white")
        ax.set_xticks(x)
        ax.set_xticklabels(cat_short_cmp, fontsize=9)
        ax.set_ylim(0, 120)
        ax.set_ylabel("OK Rate (%)")
        ax.set_title("OK Rate by Category")
        ax.legend(fontsize=9)
        for bar, v in zip(bars_a, vals_a):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                    f"{v:.0f}%", ha="center", va="bottom", fontsize=8)
        for bar, v in zip(bars_b, vals_b):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                    f"{v:.0f}%", ha="center", va="bottom", fontsize=8)

        # ---- 对比图2：总体 OK 率 ----
        ax = axes3[0, 1]
        total_a = len(records)
        total_b = len(records_b)
        ok_pct_a = total_ok_a / total_a * 100
        ok_pct_b = total_ok_b / total_b * 100
        bars = ax.bar([label_a, label_b], [ok_pct_a, ok_pct_b],
                      color=["#2196F3", "#FF9800"], edgecolor="white", linewidth=1.2)
        ax.set_ylim(0, 110)
        ax.set_ylabel("Overall OK Rate (%)")
        ax.set_title("Overall OK Rate Comparison")
        for bar, v in zip(bars, [ok_pct_a, ok_pct_b]):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                    f"{v:.1f}%", ha="center", va="bottom", fontsize=12, fontweight="bold")

        # ---- 对比图3：错误类型分布对比（分组柱） ----
        ax = axes3[0, 2]
        all_et = sorted(set(list(err_a.keys()) + list(err_b.keys())))
        x3 = np.arange(len(all_et))
        vals3_a = [err_a.get(et, 0) for et in all_et]
        vals3_b = [err_b.get(et, 0) for et in all_et]
        ax.bar(x3 - w/2, vals3_a, w, label=label_a, color="#2196F3", edgecolor="white")
        ax.bar(x3 + w/2, vals3_b, w, label=label_b, color="#FF9800", edgecolor="white")
        ax.set_xticks(x3)
        ax.set_xticklabels(all_et, rotation=20, ha="right", fontsize=8)
        ax.set_ylabel("Count")
        ax.set_title("Error Type Distribution Comparison")
        ax.legend(fontsize=9)

        # ---- 对比图4：拒绝质量 & 幻觉对比 ----
        ax = axes3[1, 0]
        metrics = ["Good Refuse", "Bad Refuse", "D Halluc", "Halluc Cont"]
        vals4_a = [good_ref_a, bad_ref_a, d_hall_a, hall_a]
        vals4_b = [good_ref_b, bad_ref_b, d_hall_b, hall_b]
        x4 = np.arange(len(metrics))
        ax.bar(x4 - w/2, vals4_a, w, label=label_a, color="#2196F3", edgecolor="white")
        ax.bar(x4 + w/2, vals4_b, w, label=label_b, color="#FF9800", edgecolor="white")
        ax.set_xticks(x4)
        ax.set_xticklabels(metrics, fontsize=9)
        ax.set_ylabel("Count")
        ax.set_title(f"Refuse Quality & Hallucination\n"
                     f"{label_a} Prec={ref_prec_a*100:.0f}% Rec={ref_rec_a*100:.0f}%  "
                     f"{label_b} Prec={ref_prec_b*100:.0f}% Rec={ref_rec_b*100:.0f}%")
        ax.legend(fontsize=9)
        for i, (va, vb) in enumerate(zip(vals4_a, vals4_b)):
            ax.text(i - w/2, va + 0.1, str(va), ha="center", fontsize=10, fontweight="bold")
            ax.text(i + w/2, vb + 0.1, str(vb), ha="center", fontsize=10, fontweight="bold")

        # ---- 对比图5：逐题 OK/FAIL 对比热力图（双行） ----
        ax = axes3[1, 1]
        ax.axis("off")
        # 用文字表格展示逐类别对比
        table_data = [["Category", f"{label_a} OK%", f"{label_b} OK%", "Diff"]]
        for c in categories:
            diff = (ok_b[c] - ok_a[c]) * 100
            sign = "+" if diff >= 0 else ""
            table_data.append([
                CATEGORY_LABELS[c],
                f"{ok_a[c]*100:.0f}%",
                f"{ok_b[c]*100:.0f}%",
                f"{sign}{diff:.0f}%"
            ])
        table_data.append([
            "Overall",
            f"{ok_pct_a:.1f}%",
            f"{ok_pct_b:.1f}%",
            f"{'+' if ok_pct_b-ok_pct_a>=0 else ''}{ok_pct_b-ok_pct_a:.1f}%"
        ])
        table_data.append([
            "Refuse Prec",
            f"{ref_prec_a*100:.0f}%",
            f"{ref_prec_b*100:.0f}%",
            f"{'+' if ref_prec_b-ref_prec_a>=0 else ''}{(ref_prec_b-ref_prec_a)*100:.0f}%"
        ])
        table_data.append([
            "Refuse Recall",
            f"{ref_rec_a*100:.0f}%",
            f"{ref_rec_b*100:.0f}%",
            f"{'+' if ref_rec_b-ref_rec_a>=0 else ''}{(ref_rec_b-ref_rec_a)*100:.0f}%"
        ])
        table_data.append([
            "Avg Retr Sim",
            f"{avg_score_a:.3f}",
            f"{avg_score_b:.3f}",
            f"{'+' if avg_score_b-avg_score_a>=0 else ''}{avg_score_b-avg_score_a:.3f}"
        ])
        tbl = ax.table(
            cellText=table_data[1:],
            colLabels=table_data[0],
            loc="center",
            cellLoc="center"
        )
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(10)
        tbl.scale(1.2, 1.8)
        ax.set_title("OK Rate Summary Table", pad=20)

        # ---- 对比图6：雷达图 ----
        ax = axes3[1, 2]
        ax.remove()
        ax = fig3.add_subplot(2, 3, 6, polar=True)
        radar_labels = [CATEGORY_LABELS[c] for c in categories]
        N = len(radar_labels)
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
        angles += angles[:1]  # 闭合
        vals_ra = [ok_a[c] * 100 for c in categories] + [ok_a[categories[0]] * 100]
        vals_rb = [ok_b[c] * 100 for c in categories] + [ok_b[categories[0]] * 100]
        ax.plot(angles, vals_ra, "o-", linewidth=2, color="#2196F3", label=label_a)
        ax.fill(angles, vals_ra, alpha=0.15, color="#2196F3")
        ax.plot(angles, vals_rb, "o-", linewidth=2, color="#FF9800", label=label_b)
        ax.fill(angles, vals_rb, alpha=0.15, color="#FF9800")
        ax.set_thetagrids(np.degrees(angles[:-1]), radar_labels, fontsize=9)
        ax.set_ylim(0, 100)
        ax.set_title("OK Rate Radar Chart", pad=15)
        ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize=9)

        plt.tight_layout(rect=[0, 0, 1, 0.96])
        compare_path = os.path.join(args.out_dir, f"eval_compare_{ts}_vs_{ts_b}.png")
        plt.savefig(compare_path, dpi=150, bbox_inches="tight")
        print(f"对比图已保存: {compare_path}")

        # 打印对比文字报告
        print(f"\n{'='*70}")
        print(f"  {label_a} vs {label_b}  对比报告")
        print(f"{'='*70}")
        print(f"{'类别':<20} {label_a:>10} {label_b:>10} {'差值':>8}")
        print("-" * 55)
        for c in categories:
            diff = (ok_b[c] - ok_a[c]) * 100
            sign = "+" if diff >= 0 else ""
            print(f"{CATEGORY_LABELS[c]:<20} {ok_a[c]*100:>9.1f}% {ok_b[c]*100:>9.1f}% "
                  f"{sign}{diff:>6.1f}%")
        print("-" * 55)
        diff_total = ok_pct_b - ok_pct_a
        sign = "+" if diff_total >= 0 else ""
        print(f"{'总体':<20} {ok_pct_a:>9.1f}% {ok_pct_b:>9.1f}% "
              f"{sign}{diff_total:>6.1f}%")
        print(f"\n幻觉续写:  {label_a}={hall_a}  {label_b}={hall_b}")
        print(f"D类误答:   {label_a}={d_hall_a}  {label_b}={d_hall_b}")
        print(f"拒绝精度:  {label_a}={ref_prec_a*100:.1f}%  {label_b}={ref_prec_b*100:.1f}%")
        print(f"拒绝召回:  {label_a}={ref_rec_a*100:.1f}%  {label_b}={ref_rec_b*100:.1f}%")
        print(f"平均检索相似度: {label_a}={avg_score_a:.4f}  {label_b}={avg_score_b:.4f}")
        print(f"{'='*70}\n")
