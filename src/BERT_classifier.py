import argparse
import logging
import os

import kagglehub
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
from datasets import load_dataset
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from transformers import (
    BertForSequenceClassification,
    BertTokenizerFast,
    get_linear_schedule_with_warmup,
)

import wandb

MODEL_NAME = "bert-base-uncased"


class NewsDataset(Dataset):
    def __init__(self, encodings: dict, labels: np.ndarray):
        self.encodings = encodings
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {k: v[idx] for k, v in self.encodings.items()}
        item["labels"] = self.labels[idx]
        return item


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_len", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--dataset", type=str, default="welfake", choices=["welfake", "liar"])
    parser.add_argument("--wandb_project", type=str, default="fake-news-detection")
    parser.add_argument("--wandb_entity", type=str, default=None)
    args = parser.parse_args()

    # Seed everything
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    logging.basicConfig(
        filename="message.log",
        format="%(asctime)s: %(levelname)s: %(message)s",
        level=logging.INFO,
    )
    os.makedirs("outputs", exist_ok=True)

    run = wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=f"bert-{args.dataset}-seed{args.seed}",
        tags=["bert", args.dataset, f"seed{args.seed}"],
        config={
            "model": MODEL_NAME,
            "dataset": args.dataset,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "max_len": args.max_len,
            "lr": args.lr,
            "seed": args.seed,
        },
    )

    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    logging.info(f"Using device: {device}")
    wandb.config.update({"device": str(device)})

    # =============================================================================
    # 1. DATA LOADING
    # =============================================================================

    logging.info(f"Downloading {args.dataset} dataset via kagglehub...")

    if args.dataset == "welfake":
        path = kagglehub.dataset_download("saurabhshahane/fake-news-classification")
        df = pd.read_csv(path + "/WELFake_Dataset.csv")
        df = df.rename(columns={"Unnamed: 0": "id"})
        df["label"] = df["label"].map({0: "fake", 1: "real"})
        df = df.dropna(subset=["title", "text", "label"]).reset_index(drop=True)
        df["text_input"] = df["title"].fillna("") + " " + df["text"].fillna("")
    else:
        # LIAR from HuggingFace Hub — no Kaggle consent required
        # label ints: 0=false, 1=half-true, 2=mostly-true, 3=true, 4=barely-true, 5=pants-fire
        ds = load_dataset("liar")
        df = pd.concat(
            [pd.DataFrame(ds[split]) for split in ["train", "validation", "test"]],
            ignore_index=True,
        )
        fake_label_ids = {0, 4, 5}  # false, barely-true, pants-fire
        df["label"] = df["label"].apply(lambda x: "fake" if x in fake_label_ids else "real")
        df = df.dropna(subset=["statement", "label"]).reset_index(drop=True)
        df["text_input"] = df["statement"].fillna("")

    logging.info(f"Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")
    logging.info(f"Label distribution:\n{df['label'].value_counts().to_string()}")
    wandb.log({"dataset/n_samples": df.shape[0]})

    label_idx = {"fake": 0, "real": 1}
    idx_label = {0: "fake", 1: "real"}
    df["label_idx"] = df["label"].map(label_idx)

    X_train, X_test, y_train, y_test = train_test_split(
        df["text_input"].to_numpy(dtype=str),
        df["label_idx"].to_numpy(dtype=int),
        test_size=0.2,
        random_state=args.seed,
        stratify=df["label_idx"].to_numpy(dtype=int),
    )
    logging.info(f"Train: {len(X_train)}, Test: {len(X_test)}")

    # =============================================================================
    # 2. TOKENISATION
    # =============================================================================

    tokenizer = BertTokenizerFast.from_pretrained(MODEL_NAME)

    def tokenize(texts: np.ndarray) -> dict:
        return tokenizer(
            texts.tolist(),
            max_length=args.max_len,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )

    train_enc = tokenize(X_train)
    test_enc = tokenize(X_test)

    # =============================================================================
    # 3. DATASET & DATALOADERS
    # =============================================================================

    train_loader = DataLoader(
        NewsDataset(train_enc, y_train), batch_size=args.batch_size, shuffle=True
    )
    test_loader = DataLoader(
        NewsDataset(test_enc, y_test), batch_size=args.batch_size, shuffle=False
    )

    # =============================================================================
    # 4. MODEL & OPTIMISER
    # =============================================================================

    model = BertForSequenceClassification.from_pretrained(
        MODEL_NAME,
        ignore_mismatched_sizes=True,
        num_labels=len(label_idx),
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
    )
    model.to(device)
    logging.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    no_decay = ["bias", "LayerNorm.weight"]
    optimizer = AdamW(
        [
            {
                "params": [
                    p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.01,
            },
            {
                "params": [
                    p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
            },
        ],
        lr=args.lr,
    )

    total_steps = len(train_loader) * args.epochs
    warmup_steps = int(0.1 * total_steps)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )
    logging.info(f"Total steps: {total_steps}, Warmup steps: {warmup_steps}")

    # =============================================================================
    # 5. TRAINING
    # =============================================================================

    global_step = 0

    def run_epoch(loader, train: bool):
        nonlocal global_step
        model.train() if train else model.eval()
        total_loss, all_preds, all_labels = 0.0, [], []

        ctx = torch.enable_grad() if train else torch.no_grad()
        with ctx:
            for batch in loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                token_type_ids = batch["token_type_ids"].to(device)
                labels = batch["labels"].to(device)

                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids,
                )
                loss = criterion(outputs.logits, labels)

                if train:
                    optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
                    scheduler.step()
                    wandb.log({"train/step_loss": loss.item(), "train/step": global_step})
                    global_step += 1

                total_loss += loss.item() * len(labels)
                preds = outputs.logits.argmax(dim=1).cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(labels.cpu().numpy())

        avg_loss = total_loss / len(loader.dataset)
        acc = accuracy_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
        return avg_loss, acc, f1, all_preds, all_labels

    history = {"train_loss": [], "val_loss": [], "train_f1": [], "val_f1": []}
    logging.info(f"Starting fine-tuning for {args.epochs} epochs...")

    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_acc, tr_f1, _, _ = run_epoch(train_loader, train=True)
        vl_loss, vl_acc, vl_f1, preds, labels = run_epoch(test_loader, train=False)

        history["train_loss"].append(tr_loss)
        history["val_loss"].append(vl_loss)
        history["train_f1"].append(tr_f1)
        history["val_f1"].append(vl_f1)

        wandb.log(
            {
                "epoch": epoch,
                "train/loss": tr_loss,
                "train/accuracy": tr_acc,
                "train/macro_f1": tr_f1,
                "val/loss": vl_loss,
                "val/accuracy": vl_acc,
                "val/macro_f1": vl_f1,
            }
        )

        logging.info(
            f"Epoch {epoch}/{args.epochs} | "
            f"Train loss {tr_loss:.4f} acc {tr_acc:.4f} F1 {tr_f1:.4f} | "
            f"Val   loss {vl_loss:.4f} acc {vl_acc:.4f} F1 {vl_f1:.4f}"
        )

    logging.info("Fine-tuning complete.")

    save_path = "outputs/bert_finetuned"
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
    logging.info(f"Model saved to {save_path}/")

    # =============================================================================
    # 6. EVALUATION
    # =============================================================================

    _, _, _, final_preds, final_labels = run_epoch(test_loader, train=False)
    target_names = [idx_label[i] for i in range(len(label_idx))]

    final_acc = accuracy_score(final_labels, final_preds)
    final_f1 = f1_score(final_labels, final_preds, average="macro")
    logging.info(f"Accuracy : {final_acc:.4f}")
    logging.info(f"Macro F1 : {final_f1:.4f}")
    logging.info(
        f"Classification report:\n"
        f"{classification_report(final_labels, final_preds, target_names=target_names)}"
    )
    wandb.log({"test/accuracy": final_acc, "test/macro_f1": final_f1})

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    epochs_range = range(1, args.epochs + 1)

    axes[0].plot(epochs_range, history["train_loss"], label="Train")
    axes[0].plot(epochs_range, history["val_loss"], label="Val")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()

    axes[1].plot(epochs_range, history["train_f1"], label="Train")
    axes[1].plot(epochs_range, history["val_f1"], label="Val")
    axes[1].set_title("Macro F1")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()

    plt.suptitle("BERT — Training", fontsize=13)
    plt.tight_layout()
    curves_path = "outputs/bert_training_curves.png"
    plt.savefig(curves_path, dpi=150, bbox_inches="tight")
    wandb.log({"charts/training_curves": wandb.Image(curves_path)})
    logging.info(f"Training curves saved to {curves_path}")
    if args.plot:
        plt.show()
    plt.close()

    cm = confusion_matrix(final_labels, final_preds)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=target_names,
        yticklabels=target_names,
        ax=ax,
    )
    ax.set_title("BERT — Confusion Matrix")
    ax.set_ylabel("True label")
    ax.set_xlabel("Predicted label")
    plt.tight_layout()
    cm_path = "outputs/bert_confusion_matrix.png"
    plt.savefig(cm_path, dpi=150, bbox_inches="tight")
    wandb.log({"charts/confusion_matrix": wandb.Image(cm_path)})
    logging.info(f"Confusion matrix saved to {cm_path}")
    if args.plot:
        plt.show()
    plt.close()

    logging.info("BERT run complete. Outputs written to outputs/")
    run.finish()


if __name__ == "__main__":
    main()
