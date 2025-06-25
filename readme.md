# Section 1: Introduction
With the currently ongoing increasing spread of information, both of verified and unverified ressources, differentiating in between fake and real news becomes a pressing issue. Especially different socio-economical groups perceive and deal with ressources differently, which can quickly lead to misinformation, if not even conspiracy theory culture.

With that being stated, the aim of this project is (in the first step) to quickly implement an out-of-the-box Machine Learning approach which correctly classifies textual data into real and fake news.

# Section 2: Literature Review
While various studies exist, mostly focusing on the capacities of CNN and LSTM architectures, more recent examples highlight the benefits of using BERT models, or - more specifically - their newer variants such as DeBERTa or RoBERTa (moreover, BERT-FND or FakeBERT).
Generally, one can differentiate into the following subsections:

## Transformer-Based Models (BERT, RoBERTa, DeBERTa, etc.)
These pre-trained models yield best results as they have gained a contextual understanding - enabling differentiation between sarcasm, bias and misleading information. While they can be finetuned to more specific use cases, they can become computationally expensive for real-time applications.

| Model   | Key Feature                                | Best For                          |
|:--------|:-------------------------------------------|:-----------------------------------|
| BERT    | Baseline transformer, bidirectional context | General fake news classification  |
| RoBERTa | More training data, longer sequences       | Higher accuracy on nuanced text   |
| DeBERTa | Disentangled attention (separates content & position) | Better at detecting subtle misinformation |
| BERT-FND | Fine-tuned BERT which uses explainability tools | Best for platforms needing interpretable fake news detection |
| Fake-BERT | Combines BERT with parallel CNN blocks (local n-gram patterns) |  High accuracy on ambiguity in language |



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
Commonly used datasets namedly are FakeNewsNet (PolitiFact, GossipCop), LIAR (PolitiFact-based) and FEVER (Fact Extraction and Verification). They can be 

# Section 4: Architecture

# Section 5: Installation
Explain how to set up the project. Use lists (`-` or `*`) for steps.

## Example:
1. Clone the repository: `git clone <repo_url>`
2. Install dependencies: `pip install -r requirements.txt`


# Section 6: Usage

# Section 7: Results

# Section 9: Conclusion

# Section 10: Contributing
Provide guidelines for contributing. Use links (`[text](url)`) for references.

## Example:
- Fork the repository.
- Submit a pull request following the [contribution guidelines](CONTRIBUTING.md).

# Section 11: References