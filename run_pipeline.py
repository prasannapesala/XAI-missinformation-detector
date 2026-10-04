import sys, os
sys.path.insert(0, os.path.abspath("."))

print("=" * 60)
print("  XAI MISINFORMATION DETECTION SYSTEM")
print("  Full Pipeline Launcher")
print("=" * 60)

# ── Step 1: Data ──────────────────────────────────────────
print("\n[1/5] Loading dataset...")
from src.collection.data_loader import create_synthetic_dataset, split_and_save
df = create_synthetic_dataset()
train_df, val_df, test_df = split_and_save(df)
print("      ✅ Dataset ready")

# ── Step 2: Semantic Analysis ─────────────────────────────
print("\n[2/5] Running semantic analysis...")
from src.analysis.semantic_analyzer import analyze_dataframe
analyzed = analyze_dataframe(train_df)
analyzed.to_csv("data/train_analyzed.csv", index=False)
print("      ✅ Semantic analysis complete")

# ── Step 3: Train RoBERTa ─────────────────────────────────
print("\n[3/5] Training RoBERTa classifier...")
from src.detection.roberta_classifier import train_model, evaluate_model, predict
model, tokenizer = train_model()
evaluate_model(model, tokenizer)
print("      ✅ Model trained and evaluated")

# ── Step 4: Risk Classification ───────────────────────────
print("\n[4/5] Running risk classification on test set...")
from src.classification.risk_classifier import (
    score_dataframe, plot_risk_distribution, plot_confidence_scatter
)
import pandas as pd
test_df = pd.read_csv("data/test.csv")
result_df = score_dataframe(test_df, model, tokenizer, predict)
plot_risk_distribution(result_df)
plot_confidence_scatter(result_df)
print("      ✅ Risk scores saved")

# ── Step 5: Generate PDF Reports ─────────────────────────
print("\n[5/5] Generating PDF reports...")
from src.reporting.report_generator import generate_report

sample_texts = [
    "5G towers are secretly spreading viruses to control the population.",
    "The Federal Reserve raised interest rates by 0.25 percent.",
    "Scientists confirmed water ice deposits near the lunar south pole.",
]

for i, text in enumerate(sample_texts):
    path = f"reports/pipeline_report_{i+1}.pdf"
    generate_report(text, model, tokenizer, output_path=path)

print("      ✅ PDF reports generated")

# ── Summary ───────────────────────────────────────────────
print("\n" + "=" * 60)
print("  PIPELINE COMPLETE!")
print("=" * 60)
print("\nOutputs:")
print("  data/train_analyzed.csv     — semantic features")
print("  reports/risk_scores.csv     — risk scores for test set")
print("  reports/risk_distribution.png")
print("  reports/confidence_scatter.png")
print("  reports/pipeline_report_1.pdf  — 5G towers")
print("  reports/pipeline_report_2.pdf  — Federal Reserve")
print("  reports/pipeline_report_3.pdf  — Lunar south pole")
print("\nTo launch the dashboard run:")
print("  streamlit run src/dashboard/app.py")
print("=" * 60) 