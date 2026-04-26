"""Generate the contents of reports/ from real evaluation artifacts.

Outputs:
  reports/final_metrics.csv           — leaderboard across all 8 models
  reports/classification_report.txt   — per-class precision/recall/F1 for every model
  reports/summary_results.xlsx        — multi-sheet workbook (overview + per-class)
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
EXTRACTED = ROOT / "_extracted"
REPORTS = ROOT / "reports"
REPORTS.mkdir(exist_ok=True)

TAG_MAP = {
    "BERT": "bert",
    "DistilRoBERTa": "distilroberta",
    "GCN": "gcn",
    "GraphSAGE": "graphsage",
    "GAT": "gat",
    "GETE (static α=0.608)": "gete_static_alpha_0608",
    "GETE (dynamic fusion, mixed)": "gete_dynamic_fusion_mixed",
    "GETE (dynamic fusion, head)": "gete_dynamic_fusion_head",
}


def main() -> None:
    summary = pd.read_csv(EXTRACTED / "tables" / "metrics_summary_all.csv")
    summary = summary.sort_values("accuracy", ascending=False).reset_index(drop=True)
    summary.insert(0, "rank", summary.index + 1)
    summary.to_csv(REPORTS / "final_metrics.csv", index=False)

    metrics = json.loads((EXTRACTED / "metrics" / "metrics_all_full.json").read_text(encoding="utf-8"))

    # classification_report.txt: one block per model
    lines = []
    lines.append("Classification Reports — LIAR binary test split (n=1267)")
    lines.append("=" * 68)
    lines.append(f"Class balance: FAKE={metrics['class_balance']['FAKE']}  REAL={metrics['class_balance']['REAL']}")
    lines.append(f"Static ensemble α (learned): {metrics['static_alpha']:.4f}")
    lines.append(f"Dynamic α: mean={metrics['dynamic_alpha_text_mean']:.4f}  std={metrics['dynamic_alpha_text_std']:.4f}")
    lines.append("=" * 68)
    lines.append("")

    for display, tag in TAG_MAP.items():
        per = pd.read_csv(EXTRACTED / "tables" / f"per_class_report_{tag}.csv", index_col=0)
        lines.append(f"## {display}")
        lines.append("-" * len(display))
        m = metrics["per_model"][display]
        lines.append(f"Accuracy={m['accuracy']:.4f}  Precision={m['precision']:.4f}  "
                     f"Recall={m['recall']:.4f}  F1={m['f1']:.4f}  "
                     f"ROC-AUC={m['roc_auc']:.4f}  PR-AUC={m['pr_auc']:.4f}")
        lines.append("")
        lines.append(f"{'class':<14}{'precision':>11}{'recall':>11}{'f1-score':>11}{'support':>10}")
        for cls in ["FAKE", "REAL"]:
            row = per.loc[cls]
            lines.append(f"{cls:<14}{row['precision']:>11.4f}{row['recall']:>11.4f}"
                         f"{row['f1-score']:>11.4f}{int(row['support']):>10d}")
        for agg in ["accuracy", "macro avg", "weighted avg"]:
            row = per.loc[agg]
            if agg == "accuracy":
                lines.append(f"{'accuracy':<14}{'':>11}{'':>11}{row['f1-score']:>11.4f}"
                             f"{int(row['support']):>10d}")
            else:
                lines.append(f"{agg:<14}{row['precision']:>11.4f}{row['recall']:>11.4f}"
                             f"{row['f1-score']:>11.4f}{int(row['support']):>10d}")
        lines.append("")

    (REPORTS / "classification_report.txt").write_text("\n".join(lines), encoding="utf-8")

    # summary_results.xlsx with multiple sheets
    xlsx_path = REPORTS / "summary_results.xlsx"
    with pd.ExcelWriter(xlsx_path, engine="openpyxl") as w:
        summary.to_excel(w, sheet_name="leaderboard", index=False)
        meta = pd.DataFrame({
            "key": ["test_size", "fake_count", "real_count",
                    "static_alpha", "dynamic_alpha_mean", "dynamic_alpha_std"],
            "value": [metrics["test_size"], metrics["class_balance"]["FAKE"],
                      metrics["class_balance"]["REAL"], metrics["static_alpha"],
                      metrics["dynamic_alpha_text_mean"], metrics["dynamic_alpha_text_std"]],
        })
        meta.to_excel(w, sheet_name="dataset_meta", index=False)
        for display, tag in TAG_MAP.items():
            per = pd.read_csv(EXTRACTED / "tables" / f"per_class_report_{tag}.csv", index_col=0)
            sheet_name = display.replace("(", "").replace(")", "").replace(" ", "_")[:31]
            per.to_excel(w, sheet_name=sheet_name)

    print(f"Wrote: {REPORTS / 'final_metrics.csv'}")
    print(f"Wrote: {REPORTS / 'classification_report.txt'}")
    print(f"Wrote: {xlsx_path}")


if __name__ == "__main__":
    main()
