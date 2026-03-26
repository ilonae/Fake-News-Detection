import os
import re
import logging
import argparse
import torch
import torch.nn as nn

import numpy as np
import pandas as pd
import kagglehub
import matplotlib.pyplot as plt
import seaborn as sns

from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import ( BertTokenizerFast, BertForSequenceClassification, get_linear_schedule_with_warmup)
from sklearn.model_selection import train_test_split
from sklearn.metrics import ( accuracy_score, f1_score, classification_report, confusion_matrix)

parser = argparse.ArgumentParser()
parser.add_argument('--plot', action='store_true')
parser.add_argument('--epochs',     type=int,   default=3)
parser.add_argument('--batch_size', type=int,   default=16)
parser.add_argument('--max_len',    type=int,   default=256)
parser.add_argument('--lr',         type=float, default=1e-5)
args = parser.parse_args()

logging.basicConfig(filename="message.log",
                    format='%(asctime)s: %(levelname)s: %(message)s',
                    level=logging.INFO)
os.makedirs('outputs', exist_ok=True)

# working with MPS = Apple Silicon GPU (M4)
if torch.backends.mps.is_available():
    DEVICE = torch.device('mps')
elif torch.cuda.is_available():
    DEVICE = torch.device('cuda')
else:
    DEVICE = torch.device('cpu')
logging.info(f"Using device: {DEVICE}")

MODEL_NAME = 'bert-base-uncased'

# =============================================================================
# 1. DATA LOADING
# =============================================================================

logging.info("Downloading dataset via kagglehub...")
#FakeNewsNet
path = kagglehub.dataset_download("mahdimashayekhi/fake-news-detection-dataset")
df = pd.read_csv(path+'/fake_news_dataset.csv')

logging.info(f"Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")
logging.info(f"Label distribution:\n{df['label'].value_counts().to_string()}")

# FIX 1: Hardcoded binary mapping — avoids ArrowStringArray sorting
# inconsistencies across pandas/pyarrow versions
label_idx = {'fake': 0, 'real': 1}
idx_label  = {0: 'fake', 1: 'real'}
df['label_idx'] = df['label'].map(label_idx)
logging.info(f"Label mapping: {label_idx}")

# BERT handles its own tokenisation so we skip manual preprocessing
df['text_input'] = df['title'].fillna('') + ' ' + df['text'].fillna('')

X_train, X_test, y_train, y_test = train_test_split(
    df['text_input'].to_numpy(dtype=str),
    df['label_idx'].to_numpy(dtype=int),
    test_size=0.2,
    random_state=42,
    stratify=df['label_idx'].to_numpy(dtype=int)
)
logging.info(f"Train: {len(X_train)}, Test: {len(X_test)}")

# =============================================================================
# 2. TOKENISATION
# =============================================================================

tokenizer = BertTokenizerFast.from_pretrained(MODEL_NAME)

def tokenize(texts: np.ndarray) -> dict:
    # truncation=True cuts at max_len, padding='max_length' pads shorter 
    # so batches are uniform and return_tensors='pt' returns tensors directly
    return tokenizer(
        texts.tolist(),
        max_length=args.max_len,
        truncation=True,
        padding='max_length',
        return_tensors='pt'
    )

train_enc = tokenize(X_train)
test_enc  = tokenize(X_test)

# =============================================================================
# 3. DATASET & SPLIT
# =============================================================================

class NewsDataset(Dataset):
    def __init__(self, encodings: dict, labels: np.ndarray):
        self.encodings = encodings
        self.labels    = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        # Return a dict that the BertForSequenceClassification obj expects:
        # input_ids, attention_mask, token_type_ids
        item = {k: v[idx] for k, v in self.encodings.items()}
        item['labels'] = self.labels[idx]
        return item

train_loader = DataLoader(
    NewsDataset(train_enc, y_train),
    batch_size=args.batch_size,
    shuffle=True
)
test_loader = DataLoader(
    NewsDataset(test_enc, y_test),
    batch_size=args.batch_size,
        shuffle=False
)

# =============================================================================
# 4. MODEL & SETTINGS
# =============================================================================

# BertForSequenceClassification = BERT + a linear classification head

model = BertForSequenceClassification.from_pretrained(
    MODEL_NAME,
    ignore_mismatched_sizes=True,
    num_labels=len(label_idx)
)
model.to(DEVICE)
logging.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

# AdamW = standard choice for BERT fine-tuning, excluding bias and LayerNorm weights from decay (no progress)
no_decay = ['bias', 'LayerNorm.weight']
optimizer_grouped_parameters = [
    {
        'params': [p for n, p in model.named_parameters()
                   if not any(nd in n for nd in no_decay)],
        'weight_decay': 0.01
    },
    {
        'params': [p for n, p in model.named_parameters()
                   if any(nd in n for nd in no_decay)],
        'weight_decay': 0.0
    }
]
optimizer = AdamW(optimizer_grouped_parameters, lr=args.lr)

# Linear warmup + decay is the usual BERT fine-tuning schedule.
# Warmup over 6% of total avoids destabilising pretrained weights early

total_steps  = len(train_loader) * args.epochs
warmup_steps = int(0.1 * total_steps)
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=warmup_steps,
    num_training_steps=total_steps
)
logging.info(f"Total steps: {total_steps}, Warmup steps: {warmup_steps}")


# =============================================================================
# 6. TRAINING
# =============================================================================

def run_epoch(loader, train: bool):
    model.train() if train else model.eval()
    total_loss, all_preds, all_labels = 0.0, [], []

    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for batch in loader:
            input_ids      = batch['input_ids'].to(DEVICE)
            attention_mask = batch['attention_mask'].to(DEVICE)
            token_type_ids = batch['token_type_ids'].to(DEVICE)
            labels         = batch['labels'].to(DEVICE)

            # BertForSequenceClassification returns a SequenceClassifierOutput namedtuple
            # outputs.loss = cross-entropy, computed internally
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                labels=labels
            )
            loss = outputs.loss

            if train:
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()

            total_loss += loss.item() * len(labels)
            preds = outputs.logits.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(loader.dataset)
    acc      = accuracy_score(all_labels, all_preds)
    f1       = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    return avg_loss, acc, f1, all_preds, all_labels

history = {'train_loss': [], 'val_loss': [], 'train_f1': [], 'val_f1': []}
logging.info(f"Starting fine-tuning for {args.epochs} epochs...")

for epoch in range(1, args.epochs + 1):
    tr_loss, tr_acc, tr_f1, _, _         = run_epoch(train_loader, train=True)
    vl_loss, vl_acc, vl_f1, preds, labels = run_epoch(test_loader,  train=False)

    history['train_loss'].append(tr_loss)
    history['val_loss'].append(vl_loss)
    history['train_f1'].append(tr_f1)
    history['val_f1'].append(vl_f1)

    logging.info(
        f"Epoch {epoch}/{args.epochs} | "
        f"Train loss {tr_loss:.4f} acc {tr_acc:.4f} F1 {tr_f1:.4f} | "
        f"Val   loss {vl_loss:.4f} acc {vl_acc:.4f} F1 {vl_f1:.4f}"
    )

logging.info("Fine-tuning complete.")

# Saving the full model + tokenizer to reload with no re-downloading or re-running
save_path = 'outputs/bert_finetuned'
model.save_pretrained(save_path)
tokenizer.save_pretrained(save_path)
logging.info(f"Model saved to {save_path}/")

# =============================================================================
# 7. PREDICTION & EVALUATION
# =============================================================================

_, _, _, final_preds, final_labels = run_epoch(test_loader, train=False)
target_names = [idx_label[i] for i in range(len(label_idx))]

logging.info(f"Accuracy : {accuracy_score(final_labels, final_preds):.4f}")
logging.info(f"Macro F1 : {f1_score(final_labels, final_preds, average='macro'):.4f}")
logging.info(f"Classification report:\n"
         f"{classification_report(final_labels, final_preds, target_names=target_names)}")


fig, axes = plt.subplots(1, 2, figsize=(12, 4))
epochs_range = range(1, args.epochs + 1)

axes[0].plot(epochs_range, history['train_loss'], label='Train')
axes[0].plot(epochs_range, history['val_loss'],   label='Val')
axes[0].set_title('Loss')
axes[0].set_xlabel('Epoch')
axes[0].legend()

axes[1].plot(epochs_range, history['train_f1'], label='Train')
axes[1].plot(epochs_range, history['val_f1'],   label='Val')
axes[1].set_title('Macro F1')
axes[1].set_xlabel('Epoch')
axes[1].legend()

plt.suptitle('BERT — Training', fontsize=13)
plt.tight_layout()
plt.savefig('outputs/bert_training_curves.png', dpi=150, bbox_inches='tight')
logging.info("Traning evaluation saved to outputs/bert_training_curves.png")
if args.plot:
    plt.show()
plt.close()

cm = confusion_matrix(final_labels, final_preds)
fig, ax = plt.subplots(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=target_names, yticklabels=target_names, ax=ax)
ax.set_title('BERT — Confusion Matrix')
ax.set_ylabel('True label')
ax.set_xlabel('Predicted label')
plt.tight_layout()
plt.savefig('outputs/bert_confusion_matrix.png', dpi=150, bbox_inches='tight')
logging.info("Confusion matrix saved to outputs/bert_confusion_matrix.png")
if args.plot:
    plt.show()
plt.close()

logging.info("BERT run complete. Outputs written to outputs/")

