import argparse
import logging
import os
import re

import joblib
import kagglehub
import matplotlib.pyplot as plt
import nltk
import pandas as pd
import seaborn as sns
from datasets import load_dataset
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.svm import SVC

import wandb


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--cv_folds", type=int, default=5, help="Number of CV folds (0 = skip CV)")
    parser.add_argument("--wandb_project", type=str, default="fake-news-detection")
    parser.add_argument("--wandb_entity", type=str, default=None)
    parser.add_argument("--dataset", type=str, default="welfake", choices=["welfake", "liar"])
    args = parser.parse_args()

    os.makedirs("outputs/svm_finetuned", exist_ok=True)
    logging.basicConfig(
        filename="message.log",
        format="%(asctime)s: %(levelname)s: %(message)s",
        level=logging.INFO,
    )

    run = wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=f"tfidf-svm-{args.dataset}",
        tags=["svm", "tfidf", args.dataset],
        config={
            "model": "TF-IDF + SVM",
            "dataset": args.dataset,
            "kernel": "linear",
            "C": 1.0,
            "max_features": 10000,
            "ngram_range": "(1,2)",
            "cv_folds": args.cv_folds,
        },
    )

    # =============================================================================
    # 1. DATA LOADING
    # =============================================================================

    logging.info(f"Downloading {args.dataset} dataset via kagglehub...")

    if args.dataset == "welfake":
        path = kagglehub.dataset_download("saurabhshahane/fake-news-classification")
        df = pd.read_csv(path + "/WELFake_Dataset.csv")
        df = df.dropna(subset=["title", "text", "label"]).reset_index(drop=True)
        df["content_raw"] = df["title"].fillna("") + " " + df["text"].fillna("")
    else:
        # LIAR from HuggingFace Hub — no Kaggle consent required
        # label ints: 0=false, 1=half-true, 2=mostly-true, 3=true, 4=barely-true, 5=pants-fire
        ds = load_dataset("liar")
        df = pd.concat(
            [pd.DataFrame(ds[split]) for split in ["train", "validation", "test"]],
            ignore_index=True,
        )
        fake_label_ids = {0, 4, 5}  # false, barely-true, pants-fire
        df["label"] = df["label"].apply(lambda x: 0 if x in fake_label_ids else 1)
        df = df.dropna(subset=["statement", "label"]).reset_index(drop=True)
        df["content_raw"] = df["statement"].fillna("")

    logging.info(f"Dataset loaded: {df.shape[0]} rows")
    wandb.log({"dataset/n_samples": df.shape[0]})

    # =============================================================================
    # 2. PREPROCESSING
    # =============================================================================

    nltk.download("stopwords", quiet=True)
    stop_words = set(stopwords.words("english"))
    ps = PorterStemmer()

    def preprocess(text: str) -> str:
        tokens = re.sub("[^a-zA-Z]", " ", str(text)).lower().split()
        return " ".join([ps.stem(w) for w in tokens if w not in stop_words])

    logging.info("Preprocessing text...")
    df["content"] = df["content_raw"].apply(preprocess)

    # =============================================================================
    # 3. VECTORISATION
    # =============================================================================

    tfidf = TfidfVectorizer(max_features=10000, ngram_range=(1, 2))
    X = tfidf.fit_transform(df["content"])
    y = df["label"]

    # =============================================================================
    # 4. TRAIN / TEST SPLIT
    # =============================================================================

    y_np = y.to_numpy()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y_np
    )

    # =============================================================================
    # 5. CROSS-VALIDATION
    # =============================================================================

    if args.cv_folds > 1:
        logging.info(f"Running {args.cv_folds}-fold stratified CV...")
        cv = StratifiedKFold(n_splits=args.cv_folds, shuffle=True, random_state=42)
        svm_cv = SVC(kernel="linear", C=1.0, random_state=42)
        cv_f1_scores = cross_val_score(svm_cv, X, y_np, cv=cv, scoring="f1_macro", n_jobs=-1)
        cv_acc_scores = cross_val_score(svm_cv, X, y_np, cv=cv, scoring="accuracy", n_jobs=-1)

        logging.info(f"CV F1  : {cv_f1_scores.mean():.4f} ± {cv_f1_scores.std():.4f}")
        logging.info(f"CV Acc : {cv_acc_scores.mean():.4f} ± {cv_acc_scores.std():.4f}")

        for fold_i, (f1_val, acc_val) in enumerate(zip(cv_f1_scores, cv_acc_scores)):
            wandb.log({f"cv/fold_{fold_i + 1}_f1": f1_val, f"cv/fold_{fold_i + 1}_acc": acc_val})

        wandb.log(
            {
                "cv/mean_f1": cv_f1_scores.mean(),
                "cv/std_f1": cv_f1_scores.std(),
                "cv/mean_acc": cv_acc_scores.mean(),
                "cv/std_acc": cv_acc_scores.std(),
            }
        )

    # =============================================================================
    # 6. TRAINING (HELD-OUT TEST SET)
    # =============================================================================

    logging.info("Training SVM (kernel=linear, C=1.0) on train split...")
    svm = SVC(kernel="linear", C=1.0, random_state=42)
    svm.fit(X_train, y_train)
    joblib.dump(svm, "outputs/svm_finetuned/svm_model.pkl")

    # =============================================================================
    # 7. PREDICTION & EVALUATION
    # =============================================================================

    y_pred = svm.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="macro")
    report = classification_report(y_test, y_pred)

    logging.info(f"Accuracy : {acc:.4f}")
    logging.info(f"Macro F1 : {f1:.4f}")
    logging.info(f"Classification report:\n{report}")

    wandb.log({"test/accuracy": acc, "test/macro_f1": f1})

    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=svm.classes_,
        yticklabels=svm.classes_,
    )
    ax.set_title("TF-IDF + SVM — Confusion Matrix")
    ax.set_ylabel("True label")
    ax.set_xlabel("Predicted label")
    plt.tight_layout()
    plt.savefig("outputs/svm_confusion_matrix.png", dpi=150, bbox_inches="tight")
    wandb.log({"charts/confusion_matrix": wandb.Image("outputs/svm_confusion_matrix.png")})
    logging.info("Confusion matrix saved to outputs/svm_confusion_matrix.png")
    if args.plot:
        plt.show()
    plt.close()

    feature_names = tfidf.get_feature_names_out()
    coefs = svm.coef_.toarray()[0]
    top_n = 20
    top_fake = pd.Series(coefs, index=feature_names).nsmallest(top_n)
    top_real = pd.Series(coefs, index=feature_names).nlargest(top_n)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    top_fake.plot(kind="barh", ax=axes[0], title="Top FAKE features", color="salmon")
    top_real.plot(kind="barh", ax=axes[1], title="Top REAL features", color="steelblue")
    plt.suptitle("TF-IDF + SVM — Feature Importance", fontsize=12)
    plt.tight_layout()
    plt.savefig("outputs/svm_top_features.png", dpi=150, bbox_inches="tight")
    wandb.log({"charts/top_features": wandb.Image("outputs/svm_top_features.png")})
    logging.info("Top features saved to outputs/svm_top_features.png")
    if args.plot:
        plt.show()
    plt.close()

    # CV bar chart (fold-level F1)
    if args.cv_folds > 1:
        fig, ax = plt.subplots(figsize=(8, 4))
        fold_labels = [f"Fold {i + 1}" for i in range(args.cv_folds)]
        bars = ax.bar(fold_labels, cv_f1_scores, color="steelblue", alpha=0.8)
        ax.axhline(
            cv_f1_scores.mean(),
            color="red",
            linestyle="--",
            label=f"Mean {cv_f1_scores.mean():.4f}",
        )
        ax.fill_between(
            range(args.cv_folds),
            cv_f1_scores.mean() - cv_f1_scores.std(),
            cv_f1_scores.mean() + cv_f1_scores.std(),
            alpha=0.15,
            color="red",
            label=f"±1 std ({cv_f1_scores.std():.4f})",
        )
        for bar, val in zip(bars, cv_f1_scores):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.002,
                f"{val:.4f}",
                ha="center",
                va="bottom",
                fontsize=9,
            )
        ax.set_title(f"TF-IDF + SVM — {args.cv_folds}-Fold CV Macro F1 ({args.dataset.upper()})")
        ax.set_ylabel("Macro F1")
        ax.set_ylim(max(0, cv_f1_scores.min() - 0.05), min(1.0, cv_f1_scores.max() + 0.05))
        ax.legend()
        plt.tight_layout()
        cv_plot_path = f"outputs/svm_cv_f1_{args.dataset}.png"
        plt.savefig(cv_plot_path, dpi=150, bbox_inches="tight")
        wandb.log({"charts/cv_f1_folds": wandb.Image(cv_plot_path)})
        logging.info(f"CV fold chart saved to {cv_plot_path}")
        if args.plot:
            plt.show()
        plt.close()

    # Log CV summary table to W&B
    if args.cv_folds > 1:
        cv_table = wandb.Table(
            columns=["fold", "f1_macro", "accuracy"],
            data=[
                [i + 1, float(cv_f1_scores[i]), float(cv_acc_scores[i])]
                for i in range(args.cv_folds)
            ],
        )
        wandb.log({"cv/fold_results": cv_table})

    logging.info("SVM - TF-IDF run complete. Outputs written to outputs/")
    run.finish()


if __name__ == "__main__":
    main()
