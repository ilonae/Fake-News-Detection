# Section 1: Introduction

With the currently ongoing increasing spread of information, both of verified and unverified resources, differentiating in between fake and real news becomes a pressing issue. Especially different socio-economical groups perceive and deal with resources differently, which can quickly lead to misinformation, if not even conspiracy theory culture.

With that being stated, the aim of this project is (in the first step) to quickly implement an out-of-the-box Machine Learning approach which correctly classifies textual data into real and fake news.

# Section 2: Literature Review

While various studies exist, mostly focusing on the capacities of CNN and LSTM architectures, more recent examples highlight the benefits of using BERT models, or - more specifically - their newer variants such as DeBERTa or RoBERTa (moreover, BERT-FND or FakeBERT).
Generally, one can differentiate into the following subsections:

## Transformer-Based Models (BERT, RoBERTa, DeBERTa, etc.)

These pre-trained models yield best results as they have gained a contextual understanding - enabling differentiation between sarcasm, bias and misleading information. While they can be fine-tuned to more specific use cases, they can become computationally expensive for real-time applications.

| Model     | Key Feature                                                    | Best For                                                     |
| :-------- | :------------------------------------------------------------- | :----------------------------------------------------------- |
| BERT      | Baseline transformer, bidirectional context                    | General fake news classification                             |
| RoBERTa   | More training data, longer sequences                           | Higher accuracy on nuanced text                              |
| DeBERTa   | Disentangled attention (separates content & position)          | Better at detecting subtle misinformation                    |
| BERT-FND  | Fine-tuned BERT which uses explainability tools                | Best for platforms needing interpretable fake news detection |
| Fake-BERT | Combines BERT with parallel CNN blocks (local n-gram patterns) | High accuracy on ambiguity in language                       |

## Sequential & Hybrid Models (LSTM, CNN, BiGRU)

Sequential or hybrid models are generally less compute-heavy than transformer models, and effective when training data is limited. Yet, they tend to struggle with long-range dependencies compared to transformers - which can be
prominent in heavy contextual data. The following can be pointed out:

* LSTMs/GRUs: Capture sequential dependencies in text (especially temporal patterns in fake news)
* CNNs: Detect local n-gram patterns (e.g., sensational phrases like "SHOCKING TRUTH!")
* Hybrids, such as CNN-LSTM, BiGRU-Attention: Combine strengths of both

## Large Language Models (LLMs) for Few-Shot & Zero-Shot Detection

Using pre-trained LLMs, such as GPT-4, LLaMA, or Mistral have the advantage that they don't require fine-tuning and can be used in a zero-shot approach. Similarly, RAGs can be useful for few-shot learning, as in providing a few labeled examples to guide detection, to then compare with the input.
Both are easy to use out of the box, but the disadvantage being that false explanations can be resulting, as well as requiring high latency which becomes problematic for real-time analysis.
Hence, they're not preferred in the usage of fake news detection.

# Section 3: Data

Commonly used datasets namedly are FakeNewsNet (PolitiFact, GossipCop), LIAR (PolitiFact-based) and FEVER (Fact Extraction and Verification). They can be accessed via their respective official repositories and Kaggle mirrors, and are compatible with the HuggingFace `datasets` library for direct loading.

| Dataset                                                                                    | Description                                                                                                                      |
| :----------------------------------------------------------------------------------------- | :------------------------------------------------------------------------------------------------------------------------------- |
| [FakeNewsNet](https://www.kaggle.com/datasets/mdepak/fakenewsnet)                             | PolitiFact + GossipCop articles with social engagement metadata. Unique because it includes sharing/comment signals.             |
| [PolitiFact Fact Check](https://www.kaggle.com/datasets/rmisra/politifact-fact-check-dataset) | Fact-checking outlet covering political statements. Labels range across 6 levels.                                                |
| [GossipCop](https://www.kaggle.com/datasets/akshaynarayananb/gossipcop)                       | Entertainment news fact-checker. Useful for capturing sensationalist writing patterns that differ from political misinformation. |
| [LIAR](https://www.kaggle.com/datasets/doanquanvietnamca/liar-dataset)                        | PolitiFact statements with 6-class labels, plus speaker metadata. Self-contained, benchmarked, rich enough for analysis.         |
| [FEVER](https://fever.ai/dataset/fever.html)                                                  | Wikipedia-derived claims labeled. Requires evidence retrieval, so it's more of an NLI task than classification.                  |

# Section 6: Architecture

This project implements and compares different news text classification architectures of increasing complexity, each representing a distinct paradigm in NLP:

## TF-IDF  and SVM

(\`src/TF-IDF_SVM_classifierbaseline\_svm.py\`) - a bag-of-words pipeline.
Text is vectorized using TF-IDF (unigrams + bigrams) and classified with a linear SVM. Fast to train, fully interpretable via coefficient weights, and establishes the performance floor for comparison.

## BERT

(\`src/BERT_classifier.py\`)  - Executing fine-tuning of a pre-trained transformer.
BERT's bidirectional attention mechanism captures global context across the entire input, giving it a structural advantage over the SVM (and LSTMs generally) on longer, nuanced articles. Fine-tuned for 3 epochs using AdamW with a linear warmup schedule.

# Section 7: Usage

## Installation

```
git clone https://github.com/ilonae/Fake-News-Detection.git 
cd Fake-News-Detection
python -m venv .venv && source .venv/bin/activate 
pip install -r requirements.txt
```

## Device support

Scripts automatically detect and use either Apple MPS (M-series), CUDA, or CPU — no configuration needed

## Running the models

Each model is self-contained. All scripts download the dataset automatically via `kagglehub` on first run.

```
# TF-IDF + SVM
python src/baseline_svm.py
python src/baseline_svm.py --plot          # show plots interactively, optional arg

# BERT-base-uncased
python src/baseline_bert.py
python src/baseline_bert.py --epochs 4 --batch_size 8   # use epochs and batch_soze args
```

## Output

All scripts write to `outputs/`:

```
outputs/
├── ..._confusion_matrix.png
├── ..._training_curves.png
└── bert_finetuned/          - saved weights and tokenizer
```

# Section 8: Results

All models evaluated on an 80/20 stratified train/test split of WELFake (72k articles).
Primary metric is macro F1. Inference time measured on the test set.

## Confusion matrices

| Model        | Accuracy | Macro F1 | Inference time |
| :----------- | :------- | :------- | :------------- |
| TF-IDF + SVM | 96.14%   | 0.9613   | —             |
| BERT.        | 99.56%   | 0.9956   | —             |
| FakeBERT     | 99.49%   | 0.9949   | —             |

<p align="center">
  <img src="outputs/svm_confusion_matrix.png" width="30%"/>
  <img src="outputs/bert_confusion_matrix.png" width="30%"/>
  <img src="outputs/fakebert_confusion_matrix.png" width="30%"/>
</p>

## Learning curves — BERT vs FakeBERT

<p align="center">
  <img src="outputs/bert_training_curves.png" width="48%"/>
  <img src="outputs/fakebert_training_curves.png" width="48%"/>
</p>

## Discussion

All three models achieve high accuracy on WELFake, reflecting the dataset's relatively clean binary signal. Meaningful differentiation emerges on three axes:

**Complexity vs. accuracy trade-off** — TF-IDF + SVM achieves competitive F1 at a fraction of the compute cost, which is the expected finding from the literature. The transformer models offer no accuracy gain on this dataset but demonstrate substantially different convergence behavior (see learning curves).

**Convergence speed** — FakeBERT's parallel CNN branch accelerates early learning by capturing local n-gram patterns that BERT's attention mechanism
requires more steps to weight appropriately. This advantage is most visible in epoch 1 of the training curves.

# Section 10: Conclusion

# Section 11: Contributing

Contributions are welcome, especially around additional model architectures or dataset integrations.

1. Fork the repo
2. Create feature branch: `git checkout -b feature/your-feature`
3. Follow the structure: `logging` over `print`, plots saved to `outputs/`, `--args` flag for adjustment
4. Submit a PR for that feature, with a brief description

# Section 12: References

- Devlin, J. et al. (2019). *BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding.* NAACL. https://arxiv.org/abs/1810.04805
- Liu, Y. et al. (2019). *RoBERTa: A Robustly Optimized BERT Pretraining Approach.* https://arxiv.org/abs/1907.11692
- Shu, K. et al. (2018). *FakeNewsNet: A Data Repository with News Content, Social Context and Spatialtemporal Information.* https://arxiv.org/abs/1809.01286
- Wang, W. Y. (2017). *"Liar, Liar Pants on Fire": A New Benchmark Dataset for Fake News Detection.* ACL. https://arxiv.org/abs/1705.00648
- Thorne, J. et al. (2018). *FEVER: a Large-scale Dataset for Fact Extraction and VERification.* NAACL. https://arxiv.org/abs/1803.05355
