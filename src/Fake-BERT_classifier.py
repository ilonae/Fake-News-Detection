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
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from transformers import BertModel, BertTokenizerFast, get_linear_schedule_with_warmup

MODEL_NAME = "bert-base-uncased"
KERNEL_SIZE = 4  # 4-gram local patterns via CNN — as described in the FakeBERT paper


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


class FakeBERT(nn.Module):
    """
    FakeBERT: BERT encoder with a parallel CNN block.

    Rather than relying solely on the [CLS] token, FakeBERT also passes the
    full token sequence through a CNN block that detects local n-gram patterns
    (sensationalist phrases, hedging language) that BERT's global attention may
    underweight. The CNN output and [CLS] embedding are concatenated before
    classification, giving the model both global context and local pattern signal.
    """

    def __init__(self, num_classes: int, num_filters: int, kernel_size: int):
        super().__init__()
        self.bert = BertModel.from_pretrained(MODEL_NAME)
        hidden_size = self.bert.config.hidden_size  # 768 for bert-base
        self.cnn = nn.Conv1d(
            in_channels=hidden_size,
            out_channels=num_filters,
            kernel_size=kernel_size,
            padding=0,
        )
        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Linear(hidden_size + num_filters, num_classes)

    def forward(self, input_ids, attention_mask, token_type_ids):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        cls_output = outputs.pooler_output  # (B, 768)
        x = outputs.last_hidden_state.transpose(1, 2)  # (B, 768, seq_len)
        x = F.relu(self.cnn(x))  # (B, num_filters, seq_len - kernel + 1)
        x = F.max_pool1d(x, kernel_size=x.size(2)).squeeze(2)  # (B, num_filters)
        fused = self.dropout(torch.cat([cls_output, x], dim=1))
        return self.classifier(fused)  # (B, num_classes)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_len", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--num_filters", type=int, default=128, help="CNN filters per kernel size")
    args = parser.parse_args()

    logging.basicConfig(
        filename="message.log",
        format="%(asctime)s: %(levelname)s: %(message)s",
        level=logging.INFO,
    )
    os.makedirs("outputs", exist_ok=True)

    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    logging.info(f"Using device: {device}")

    # =============================================================================
    # 1. DATA LOADING
    # =============================================================================

    logging.info("Downloading WELFake dataset via kagglehub...")
    path = kagglehub.dataset_download("saurabhshahane/fake-news-classification")
    df = pd.read_csv(path + "/WELFake_Dataset.csv")
    df = df.rename(columns={"Unnamed: 0": "id"})
    df["label"] = df["label"].map({0: "fake", 1: "real"})
    df = df.dropna(subset=["title", "text", "label"]).reset_index(drop=True)

    logging.info(f"Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")
    logging.info(f"Label distribution:\n{df['label'].value_counts().to_string()}")

    label_idx = {"fake": 0, "real": 1}
    idx_label = {0: "fake", 1: "real"}
    df["label_idx"] = df["label"].map(label_idx)
    df["text_input"] = df["title"].fillna("") + " " + df["text"].fillna("")

    X_train, X_test, y_train, y_test = train_test_split(
        df["text_input"].to_numpy(dtype=str),
        df["label_idx"].to_numpy(dtype=int),
        test_size=0.2,
        random_state=42,
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

    model = FakeBERT(
        num_classes=len(label_idx), num_filters=args.num_filters, kernel_size=KERNEL_SIZE
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logging.info(f"Total parameters:     {total_params:,}")
    logging.info(f"Trainable parameters: {trainable_params:,}")

    # Lower LR for pretrained BERT to avoid catastrophic forgetting;
    # higher LR for the untrained CNN + classifier head.
    optimizer = AdamW(
        [
            {"params": model.bert.parameters(), "lr": args.lr, "weight_decay": 0.01},
            {"params": model.cnn.parameters(), "lr": args.lr * 10, "weight_decay": 0.01},
            {"params": model.classifier.parameters(), "lr": args.lr * 10, "weight_decay": 0.01},
        ],
        lr=args.lr,
    )

    total_steps = len(train_loader) * args.epochs
    warmup_steps = int(0.1 * total_steps)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )
    logging.info(f"Total steps: {total_steps}, Warmup steps: {warmup_steps}")

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    # =============================================================================
    # 5. TRAINING
    # =============================================================================

    def run_epoch(loader, train: bool):
        model.train() if train else model.eval()
        total_loss, all_preds, all_labels = 0.0, [], []

        ctx = torch.enable_grad() if train else torch.no_grad()
        with ctx:
            for step, batch in enumerate(loader):
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                token_type_ids = batch["token_type_ids"].to(device)
                labels = batch["labels"].to(device)

                logits = model(input_ids, attention_mask, token_type_ids)
                loss = criterion(logits, labels)

                if train:
                    optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
                    scheduler.step()

                    if step % 50 == 0:
                        logging.info(f"  Step {step}/{len(loader)} | loss {loss.item():.4f}")

                total_loss += loss.item() * len(labels)
                all_preds.extend(logits.argmax(dim=1).cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        avg_loss = total_loss / len(loader.dataset)
        acc = accuracy_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
        return avg_loss, acc, f1, all_preds, all_labels

    history = {"train_loss": [], "val_loss": [], "train_f1": [], "val_f1": []}
    logging.info(f"Starting FakeBERT fine-tuning for {args.epochs} epochs...")

    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_acc, tr_f1, _, _ = run_epoch(train_loader, train=True)
        vl_loss, vl_acc, vl_f1, preds, labels = run_epoch(test_loader, train=False)

        history["train_loss"].append(tr_loss)
        history["val_loss"].append(vl_loss)
        history["train_f1"].append(tr_f1)
        history["val_f1"].append(vl_f1)

        logging.info(
            f"Epoch {epoch}/{args.epochs} | "
            f"Train loss {tr_loss:.4f} acc {tr_acc:.4f} F1 {tr_f1:.4f} | "
            f"Val   loss {vl_loss:.4f} acc {vl_acc:.4f} F1 {vl_f1:.4f}"
        )

    logging.info("Fine-tuning complete.")

    save_path = "outputs/fakebert_finetuned"
    os.makedirs(save_path, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(save_path, "fakebert_weights.pt"))
    tokenizer.save_pretrained(save_path)
    logging.info(f"Model saved to {save_path}/")

    # =============================================================================
    # 6. EVALUATION
    # =============================================================================

    _, _, _, final_preds, final_labels = run_epoch(test_loader, train=False)
    target_names = [idx_label[i] for i in range(len(label_idx))]

    logging.info(f"Accuracy : {accuracy_score(final_labels, final_preds):.4f}")
    logging.info(f"Macro F1 : {f1_score(final_labels, final_preds, average='macro'):.4f}")
    logging.info(
        f"Classification report:\n"
        f"{classification_report(final_labels, final_preds, target_names=target_names)}"
    )

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

    plt.suptitle("FakeBERT (BERT + CNN k=4) — Training Curves", fontsize=13)
    plt.tight_layout()
    plt.savefig("outputs/fakebert_training_curves.png", dpi=150, bbox_inches="tight")
    logging.info("Saved: outputs/fakebert_training_curves.png")
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
    ax.set_title("FakeBERT (BERT + CNN k=4) — Confusion Matrix")
    ax.set_ylabel("True label")
    ax.set_xlabel("Predicted label")
    plt.tight_layout()
    plt.savefig("outputs/fakebert_confusion_matrix.png", dpi=150, bbox_inches="tight")
    logging.info("Saved: outputs/fakebert_confusion_matrix.png")
    if args.plot:
        plt.show()
    plt.close()

    logging.info("FakeBERT run complete. Outputs written to outputs/")


if __name__ == "__main__":
    main()
