"""Gradio web app for the GETE fake-news detection project.

Tabs
----
1. Single-text prediction: every trained model + ensemble in a side-by-side card.
2. Batch CSV: upload a CSV with a 'text' column, download predictions.
3. Leaderboard: reads results/tables/metrics_summary_all.csv (falls back to
   metrics_summary.csv).
4. Figures gallery: every PNG under results/figures/ is displayed.
5. Training logs: one collapsible panel per JSON file.

Launch
------
    python app_gradio.py                # local
    python app_gradio.py --share        # public link via Gradio tunnel
"""

from __future__ import annotations

import argparse
import io
import json
import os
import sys
from pathlib import Path
from typing import List

# disable gradio analytics phone-home (no network on Windows default)
os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

import gradio as gr
import matplotlib
matplotlib.use("Agg")
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils.predictor import Predictor


OUT = Path("results")
FIGS = OUT / "figures"
TABS = OUT / "tables"
LOGS = OUT / "logs"
METS = OUT / "metrics"

# lazy singleton
PREDICTOR = None


def get_predictor() -> Predictor:
    global PREDICTOR
    if PREDICTOR is None:
        PREDICTOR = Predictor()
    return PREDICTOR


# ---------- prediction helpers ------------------------------------------------
def predict_text(text: str):
    if not text or not text.strip():
        return (
            "Enter a news article in the box above.",
            pd.DataFrame(),
            None,
            None,
        )
    pr = get_predictor()
    result = pr.predict([text])[0]
    rows = []
    for model, pred in result["per_model"].items():
        rows.append({
            "Model": model,
            "Prediction": pred["pred"],
            "Confidence": round(pred["confidence"], 4),
            "P(FAKE)": round(pred["prob_fake"], 4),
            "P(REAL)": round(pred["prob_real"], 4),
            **{k: round(v, 4) if isinstance(v, float) else v
               for k, v in pred.items()
               if k not in {"pred", "confidence", "prob_fake", "prob_real"}},
        })
    df = pd.DataFrame(rows)

    # markdown summary card
    fake_models = [r for r in rows if r["Prediction"] == "FAKE"]
    real_models = [r for r in rows if r["Prediction"] == "REAL"]
    summary_md = "## Predictions\n\n"
    summary_md += f"- **FAKE** votes: {len(fake_models)} / {len(rows)}\n"
    summary_md += f"- **REAL** votes: {len(real_models)} / {len(rows)}\n\n"
    # highlight ensembles
    for r in rows:
        if "GETE" in r["Model"]:
            emoji = "🚨" if r["Prediction"] == "FAKE" else "✅"
            summary_md += (
                f"- {emoji} **{r['Model']}**: {r['Prediction']} "
                f"(conf={r['Confidence']:.3f})\n"
            )

    # plot per-model P(REAL)
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, max(3, 0.4 * len(rows))))
    models = [r["Model"] for r in rows]
    p_real = [r["P(REAL)"] for r in rows]
    colors = ["#2ca02c" if p >= 0.5 else "#d62728" for p in p_real]
    ax.barh(models, p_real, color=colors)
    ax.axvline(0.5, ls="--", c="gray", alpha=0.5)
    ax.set_xlim(0, 1)
    ax.set_xlabel("P(REAL)")
    ax.set_title("Probability article is REAL, per model")
    fig.tight_layout()

    # text-only attention weights (fusion)
    fusion_plot = None
    for r in rows:
        if r["Model"] == "GETE (dynamic fusion)":
            fig2, ax2 = plt.subplots(figsize=(5, 3.5))
            ax2.bar(["text branch (α)", "graph branch (1-α)"],
                    [r.get("alpha_text", 0), r.get("alpha_graph", 0)],
                    color=["#1f77b4", "#ff7f0e"])
            ax2.set_ylim(0, 1)
            ax2.set_ylabel("attention weight")
            ax2.set_title("Dynamic fusion attention for this article")
            fig2.tight_layout()
            fusion_plot = fig2
            break

    return summary_md, df, fig, fusion_plot


def batch_predict(csv_file, text_column: str = "text"):
    if csv_file is None:
        return (
            pd.DataFrame({"error": ["upload a CSV with a 'text' column"]}),
            None,
        )
    df = pd.read_csv(csv_file.name)
    if text_column not in df.columns:
        return (
            pd.DataFrame({"error": [f"CSV must contain column '{text_column}' "
                                     f"(found: {list(df.columns)})"]}),
            None,
        )
    pr = get_predictor()
    texts = df[text_column].astype(str).tolist()
    print(f"[gradio] batch of {len(texts)} texts")
    results = pr.predict(texts)
    rows = []
    for text, res in zip(texts, results):
        base = {"text": text}
        for model, pred in res["per_model"].items():
            base[f"{model}__pred"] = pred["pred"]
            base[f"{model}__P(REAL)"] = round(pred["prob_real"], 4)
            base[f"{model}__confidence"] = round(pred["confidence"], 4)
        rows.append(base)
    out = pd.DataFrame(rows)
    tmp = "results/batch_predictions.csv"
    out.to_csv(tmp, index=False)
    return out, tmp


# ---------- leaderboard / metrics --------------------------------------------
def load_leaderboard() -> pd.DataFrame:
    candidates = ["metrics_summary_all.csv", "metrics_summary.csv"]
    for c in candidates:
        p = TABS / c
        if p.exists():
            df = pd.read_csv(p)
            return df
    return pd.DataFrame({"warning": ["No metrics CSV found. Run evaluate.py."]})


def load_all_json_logs() -> str:
    if not LOGS.exists():
        return "no training logs found"
    parts = []
    for p in sorted(LOGS.glob("*.json")):
        with open(p) as f:
            data = json.load(f)
        parts.append(f"### `{p.name}`\n```json\n{json.dumps(data, indent=2)[:4000]}\n```\n")
    return "\n".join(parts) if parts else "no JSON logs"


# ---------- figures gallery ---------------------------------------------------
def load_figures() -> List[tuple]:
    if not FIGS.exists():
        return []
    return [(str(p), p.stem) for p in sorted(FIGS.glob("*.png"))]


# ---------- UI ----------------------------------------------------------------
def build_app():
    with gr.Blocks(title="GETE Fake-News Detector",
                   theme=gr.themes.Soft(primary_hue="blue")) as demo:
        gr.Markdown(
            """# 📰 Fake-News Detector — GETE
Graph-augmented transformer ensemble (Kumar et al., 2025) reproduction,
trained on the LIAR dataset. Below you can run single-text predictions,
batch CSV predictions, view the full model leaderboard, and browse every
figure produced by the evaluation pipeline."""
        )
        with gr.Tabs():
            # ----- Single text ----
            with gr.Tab("🧠 Single prediction"):
                with gr.Row():
                    with gr.Column(scale=3):
                        text_in = gr.Textbox(
                            label="News article / statement",
                            placeholder="Paste any news statement here...",
                            lines=6,
                        )
                        btn = gr.Button("Predict", variant="primary")
                        examples = gr.Examples(
                            examples=[
                                ["Says the Annies List political group supports "
                                 "third-trimester abortions on demand."],
                                ["When did the decline of coal start? It started "
                                 "when natural gas took off that started to begin "
                                 "in Bush's administration."],
                                ["Hillary Clinton agrees with John McCain by "
                                 "voting to give George Bush the benefit of the "
                                 "doubt on Iran."],
                                ["Breaking: scientists confirm aliens are living "
                                 "on the moon."],
                            ],
                            inputs=text_in,
                        )
                    with gr.Column(scale=2):
                        summary_md = gr.Markdown()
                with gr.Row():
                    per_model_plot = gr.Plot(label="P(REAL) per model")
                    fusion_plot = gr.Plot(label="Dynamic fusion attention")
                per_model_table = gr.Dataframe(
                    label="Per-model predictions",
                    interactive=False, wrap=True,
                )
                btn.click(
                    predict_text,
                    inputs=[text_in],
                    outputs=[summary_md, per_model_table, per_model_plot, fusion_plot],
                )

            # ----- Batch CSV ----
            with gr.Tab("📂 Batch CSV"):
                gr.Markdown(
                    "Upload a CSV with a `text` column. The app runs every "
                    "model and returns a downloadable CSV of predictions."
                )
                csv_in = gr.File(label="CSV file", file_types=[".csv"])
                col_name = gr.Textbox(label="Text column name", value="text")
                batch_btn = gr.Button("Run batch prediction", variant="primary")
                batch_out = gr.Dataframe(label="Results", interactive=False, wrap=True)
                batch_file = gr.File(label="Download predictions")
                batch_btn.click(batch_predict, inputs=[csv_in, col_name],
                                outputs=[batch_out, batch_file])

            # ----- Leaderboard ----
            with gr.Tab("🏆 Leaderboard"):
                gr.Markdown("Saved metrics from `results/tables/` (latest run).")
                lb = gr.Dataframe(value=load_leaderboard(), interactive=False,
                                  label="Model comparison", wrap=True)
                refresh_lb = gr.Button("Refresh")
                refresh_lb.click(lambda: load_leaderboard(), outputs=lb)

            # ----- Figures ----
            with gr.Tab("🖼 Figures"):
                gr.Markdown("Every PNG under `results/figures/`.")
                gallery = gr.Gallery(value=load_figures(), columns=2,
                                     height="auto", label="Figures")
                refresh_g = gr.Button("Refresh gallery")
                refresh_g.click(lambda: load_figures(), outputs=gallery)

            # ----- Logs ----
            with gr.Tab("📝 Training logs"):
                gr.Markdown(value=load_all_json_logs())

        gr.Markdown(
            f"GPU available: **{torch.cuda.is_available()}** "
            f"({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU-only'})"
        )
    return demo


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--share", action="store_true", help="create Gradio public link")
    p.add_argument("--port", type=int, default=7860)
    p.add_argument("--host", type=str, default="127.0.0.1")
    args = p.parse_args()

    demo = build_app()
    demo.queue().launch(server_name=args.host, server_port=args.port,
                        share=args.share)


if __name__ == "__main__":
    main()
