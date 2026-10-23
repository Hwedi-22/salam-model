"""SALAM baseline: TF-IDF + Logistic Regression.

الاستخدام:
    python train_baseline.py --data sample_data.xlsx
    python train_baseline.py --data dataset.xlsx --text-col text --label-col label

لو مسبتش أسماء الأعمدة، السكريبت بيحاول يخمّنها.
"""
import argparse
import json
import os
import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, confusion_matrix, f1_score, precision_score,
    recall_score, roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline

from preprocess import normalize_arabic

SEED = 42
MODEL_VERSION = "baseline-tfidf-lr-v1"


# ---------------------------------------------------------------- loading
def load_table(path):
    if path.lower().endswith((".xlsx", ".xls")):
        return pd.read_excel(path)
    return pd.read_csv(path)


def guess_columns(df, text_col, label_col):
    if text_col and label_col:
        return text_col, label_col
    obj_cols = [c for c in df.columns if not pd.api.types.is_numeric_dtype(df[c])]
    if not obj_cols and not text_col:
        sys.exit("Could not find a text column. Use --text-col and --label-col.")
    if not text_col:
        text_col = max(obj_cols, key=lambda c: df[c].astype(str).str.len().mean())
    if not label_col:
        others = [c for c in df.columns if c != text_col]
        label_col = min(others, key=lambda c: df[c].nunique())
    return text_col, label_col


# ---------------------------------------------------------------- metrics
def compute_metrics(y_true, y_pred, y_score, binary):
    m = {"accuracy": accuracy_score(y_true, y_pred)}
    if binary:
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        m.update(
            precision=precision_score(y_true, y_pred, zero_division=0),
            recall=recall_score(y_true, y_pred, zero_division=0),
            f1=f1_score(y_true, y_pred, zero_division=0),
            false_negative_rate=fn / (fn + tp) if (fn + tp) else 0.0,
            roc_auc=roc_auc_score(y_true, y_score) if len(set(y_true)) > 1 else None,
            tn=int(tn), fp=int(fp), fn=int(fn), tp=int(tp),
        )
    else:
        m.update(
            precision_macro=precision_score(y_true, y_pred, average="macro", zero_division=0),
            recall_macro=recall_score(y_true, y_pred, average="macro", zero_division=0),
            f1_macro=f1_score(y_true, y_pred, average="macro", zero_division=0),
        )
    return {k: (round(float(v), 4) if isinstance(v, (float, np.floating)) else v) for k, v in m.items()}


def build_pipeline(analyzer, ngram, C):
    vec = TfidfVectorizer(
        preprocessor=normalize_arabic, analyzer=analyzer, ngram_range=ngram,
        min_df=2, sublinear_tf=True,
    )
    clf = LogisticRegression(C=C, max_iter=2000, class_weight="balanced")
    return make_pipeline(vec, clf)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--text-col")
    ap.add_argument("--label-col")
    ap.add_argument("--positive-label", help="قيمة الـ label اللي معناها 'مؤشرات ضيق/اكتئاب'")
    ap.add_argument("--out", default="outputs")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    os.makedirs(f"{args.out}/splits", exist_ok=True)

    # 1) تحميل
    df = load_table(args.data)
    print("Columns found:", list(df.columns))
    text_col, label_col = guess_columns(df, args.text_col, args.label_col)
    print(f"Using text column = '{text_col}' | label column = '{label_col}'")
    df = df[[text_col, label_col]].rename(columns={text_col: "text", label_col: "label"})

    # 2) تنظيف + حذف التكرارات
    n0 = len(df)
    df = df.dropna()
    df["clean"] = df["text"].map(normalize_arabic)
    df = df[df["clean"].str.len() > 0]
    conflict = df.groupby("clean")["label"].nunique()
    conflict = conflict[conflict > 1].index
    df = df[~df["clean"].isin(conflict)]               # نفس النص بلابلين مختلفين -> نحذفه
    df = df.drop_duplicates(subset="clean").reset_index(drop=True)
    print(f"Rows: {n0} -> {len(df)} after cleaning/dedup (conflicting-label texts removed: {len(conflict)})")

    # 3) تحديد الكلاس الموجب (لو الـ labels ثنائية)
    classes = sorted(df["label"].unique(), key=str)
    binary = len(classes) == 2
    pos = None
    if binary:
        if args.positive_label is not None:
            pos = next(c for c in classes if str(c) == args.positive_label)
        elif set(map(str, classes)) == {"0", "1"}:
            pos = next(c for c in classes if str(c) == "1")
        else:
            pos = df["label"].value_counts().idxmin()
            print(f"WARNING: positive label not given; using minority class '{pos}'. Use --positive-label to set it.")
        df["y"] = (df["label"] == pos).astype(int)
        print("Positive class (distress indicators):", pos)
    else:
        df["y"] = df["label"]
    print("Class distribution:\n", df["label"].value_counts().to_string())

    # 4) تقسيم 70/15/15 ثابت
    train, rest = train_test_split(df, test_size=0.30, stratify=df["y"], random_state=SEED)
    val, test = train_test_split(rest, test_size=0.50, stratify=rest["y"], random_state=SEED)
    train.to_csv(f"{args.out}/splits/train.csv", index=False, encoding="utf-8-sig")
    val.to_csv(f"{args.out}/splits/val.csv", index=False, encoding="utf-8-sig")
    test.to_csv(f"{args.out}/splits/test.csv", index=False, encoding="utf-8-sig")
    print(f"Split sizes: train={len(train)} val={len(val)} test={len(test)}")
    if set(train["clean"]) & set(test["clean"]):
        sys.exit("LEAKAGE: same text found in train and test!")

    # 5) اختيار أحسن إعدادات على الـ validation فقط
    grid = [(a, ng, C) for a, ng in [("word", (1, 2)), ("char_wb", (2, 5))] for C in (0.1, 1, 10)]
    best, best_score = None, -1
    for analyzer, ngram, C in grid:
        p = build_pipeline(analyzer, ngram, C).fit(train["text"], train["y"])
        pred = p.predict(val["text"])
        s = f1_score(val["y"], pred) if binary else f1_score(val["y"], pred, average="macro")
        print(f"  val F1  analyzer={analyzer:8s} ngram={ngram} C={C:<4} -> {s:.4f}")
        if s > best_score:
            best, best_score = (analyzer, ngram, C), s
    print("Best settings:", best, "| val F1 =", round(best_score, 4))

    # 6) التدريب النهائي + التقييم على الـ test (مرة واحدة)
    model = build_pipeline(*best).fit(train["text"], train["y"])
    pred = model.predict(test["text"])
    score = model.predict_proba(test["text"])[:, 1] if binary else None
    metrics = compute_metrics(test["y"], pred, score, binary)
    print("\n=== TEST RESULTS ===")
    for k, v in metrics.items():
        print(f"  {k}: {v}")

    # 7) حفظ النتائج
    if binary:
        cm = confusion_matrix(test["y"], pred, labels=[0, 1])
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(4, 4))
            ax.imshow(cm, cmap="Blues")
            ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
            ax.set_xticklabels(["pred 0", "pred 1"]); ax.set_yticklabels(["true 0", "true 1"])
            for i in range(2):
                for j in range(2):
                    ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=14)
            ax.set_title("Confusion matrix (test)")
            fig.tight_layout(); fig.savefig(f"{args.out}/confusion_matrix.png", dpi=150)
        except Exception as e:
            print("Could not draw confusion matrix:", e)
        fn_rows = test[(test["y"] == 1) & (pred == 0)][["text", "label"]]
        fn_rows.to_csv(f"{args.out}/false_negatives.csv", index=False, encoding="utf-8-sig")
        print(f"False negatives saved: {len(fn_rows)} rows -> {args.out}/false_negatives.csv")

    meta = {
        "model_version": MODEL_VERSION,
        "best_settings": {"analyzer": best[0], "ngram_range": list(best[1]), "C": best[2]},
        "positive_label": str(pos) if binary else None,
        "classes": [str(c) for c in classes],
        "split_seed": SEED,
        "n_train": len(train), "n_val": len(val), "n_test": len(test),
        "test_metrics": metrics,
    }
    with open(f"{args.out}/metrics.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    joblib.dump({"pipeline": model, "meta": meta}, f"{args.out}/baseline_model.joblib")
    print(f"\nSaved model + metrics in '{args.out}/'")


if __name__ == "__main__":
    main()
