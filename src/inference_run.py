import argparse
import logging
import os
import re
import time

import joblib
import kagglehub
import nltk
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from transformers import BertForSequenceClassification, BertModel, BertTokenizerFast

MODEL_NAME = "bert-base-uncased"
KERNEL_SIZE = 4


class NewsDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {k: v[idx] for k, v in self.encodings.items()}
        item["labels"] = self.labels[idx]
        return item


class FakeBERT(nn.Module):
    def __init__(self, num_classes, num_filters, kernel_size):
        super().__init__()
        self.bert = BertModel.from_pretrained(MODEL_NAME)
        hidden_size = self.bert.config.hidden_size
        self.cnn = nn.Conv1d(hidden_size, num_filters, kernel_size, padding=0)
        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Linear(hidden_size + num_filters, num_classes)

    def forward(self, input_ids, attention_mask, token_type_ids):
        outputs = self.bert(
            input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids
        )
        cls_output = outputs.pooler_output
        x = F.relu(self.cnn(outputs.last_hidden_state.transpose(1, 2)))
        x = F.max_pool1d(x, x.size(2)).squeeze(2)
        return self.classifier(self.dropout(torch.cat([cls_output, x], dim=1)))


def measure(fn, runs: int) -> tuple[float, float]:
    times = []
    for _ in range(runs):
        start = time.perf_counter()
        fn()
        times.append(time.perf_counter() - start)
    return float(np.mean(times)), float(np.std(times))


def report(name: str, elapsed_mean: float, elapsed_std: float, n: int) -> dict:
    ms_per_sample = elapsed_mean / n * 1000
    logging.info(
        f"{name:<20} | total {elapsed_mean:.3f}s ±{elapsed_std:.3f} "
        f"| {ms_per_sample:.3f} ms/sample "
        f"| {n / elapsed_mean:.0f} samples/sec"
    )
    return {
        "model": name,
        "total_s": round(elapsed_mean, 3),
        "std_s": round(elapsed_std, 3),
        "ms_per_sample": round(ms_per_sample, 3),
        "samples_per_sec": round(n / elapsed_mean, 1),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Inference timings")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_len", type=int, default=256)
    parser.add_argument("--num_filters", type=int, default=128)
    parser.add_argument("--runs", type=int, default=3, help="Number of timing runs to average")
    args = parser.parse_args()

    logging.basicConfig(
        filename="message.log",
        format="%(asctime)s: %(levelname)s: %(message)s",
        level=logging.INFO,
    )
    os.makedirs("outputs", exist_ok=True)
    nltk.download("stopwords", quiet=True)

    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    logging.info(f"Device: {device}")

    # =============================================================================
    # 1. DATA LOADING
    # =============================================================================

    path = kagglehub.dataset_download("saurabhshahane/fake-news-classification")
    df = pd.read_csv(path + "/WELFake_Dataset.csv")
    df = df.rename(columns={"Unnamed: 0": "id"})
    df["label"] = df["label"].map({0: "fake", 1: "real"})
    df = df.dropna(subset=["title", "text", "label"]).reset_index(drop=True)
    df["text_input"] = df["title"].fillna("") + " " + df["text"].fillna("")

    label_idx = {"fake": 0, "real": 1}
    df["label_idx"] = df["label"].map(label_idx)

    _, X_test, _, y_test = train_test_split(
        df["text_input"].to_numpy(dtype=str),
        df["label_idx"].to_numpy(dtype=int),
        test_size=0.2,
        random_state=42,
        stratify=df["label_idx"].to_numpy(dtype=int),
    )

    # =============================================================================
    # 2. SVM INFERENCE
    # =============================================================================

    logging.info("--- SVM ---")

    stop_words = set(stopwords.words("english"))
    ps = PorterStemmer()

    def preprocess(text: str) -> str:
        tokens = re.sub("[^a-zA-Z]", " ", str(text)).lower().split()
        return " ".join([ps.stem(w) for w in tokens if w not in stop_words])

    logging.info("Preprocessing for TF-IDF...")
    df["content"] = df["text_input"].apply(preprocess)
    tfidf = TfidfVectorizer(max_features=10000, ngram_range=(1, 2))

    _, X_test_tfidf, _, _ = train_test_split(
        tfidf.fit_transform(df["content"]),
        df["label_idx"].to_numpy(dtype=int),
        test_size=0.2,
        random_state=42,
        stratify=df["label_idx"].to_numpy(dtype=int),
    )

    svm = joblib.load("outputs/svm_finetuned/svm_model.pkl")
    svm_mean, svm_std = measure(lambda: svm.predict(X_test_tfidf), args.runs)
    svm_result = report("TF-IDF + SVM", svm_mean, svm_std, X_test_tfidf.shape[0])

    # =============================================================================
    # 3. BERT / FAKEBERT TOKENISATION
    # =============================================================================

    logging.info("Tokenising test set for transformer models...")
    tokenizer = BertTokenizerFast.from_pretrained(MODEL_NAME)
    enc = tokenizer(
        X_test.tolist(),
        max_length=args.max_len,
        truncation=True,
        padding="max_length",
        return_tensors="pt",
    )

    test_loader = DataLoader(NewsDataset(enc, y_test), batch_size=args.batch_size, shuffle=False)

    def run_inference(model):
        model.eval()
        with torch.no_grad():
            for batch in test_loader:
                model(
                    input_ids=batch["input_ids"].to(device),
                    attention_mask=batch["attention_mask"].to(device),
                    token_type_ids=batch["token_type_ids"].to(device),
                )

    # =============================================================================
    # 4. BERT INFERENCE
    # =============================================================================

    logging.info("--- BERT ---")
    bert_model = BertForSequenceClassification.from_pretrained(
        "outputs/bert_finetuned", num_labels=2
    ).to(device)
    bert_mean, bert_std = measure(lambda m=bert_model: run_inference(m), args.runs)
    bert_result = report("BERT-base", bert_mean, bert_std, len(X_test))
    del bert_model

    # =============================================================================
    # 5. FAKEBERT INFERENCE
    # =============================================================================

    logging.info("--- FakeBERT ---")
    fakebert_model = FakeBERT(
        num_classes=2, num_filters=args.num_filters, kernel_size=KERNEL_SIZE
    ).to(device)
    fakebert_model.load_state_dict(
        torch.load(
            "outputs/fakebert_finetuned/fakebert_weights.pt",
            map_location=device,
            weights_only=True,
        )
    )
    fakebert_mean, fakebert_std = measure(lambda m=fakebert_model: run_inference(m), args.runs)
    fakebert_result = report("FakeBERT", fakebert_mean, fakebert_std, len(X_test))
    del fakebert_model

    # =============================================================================
    # 6. SUMMARY
    # =============================================================================

    results = pd.DataFrame([svm_result, bert_result, fakebert_result])
    logging.info("\n" + "=" * 70)
    logging.info("INFERENCE SUMMARY")
    logging.info("=" * 70)
    logging.info(f"\n{results.to_string(index=False)}")
    logging.info("=" * 70)

    results.to_csv("outputs/inference_times.csv", index=False)
    logging.info("Saved: outputs/inference_times.csv")


if __name__ == "__main__":
    main()
