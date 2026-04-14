# Spam Email Detector

## Team Members
| Name | Student ID | Email |
|------|-----------|-------|
| Andrew Lin | 016880760 | andrew.r.lin@sjsu.edu |

---

## Problem Statement
Email spam remains a persistent cybersecurity and productivity problem. Traditional rule-based filters struggle to keep up with evolving spam tactics. This project trains and evaluates multiple machine learning models — including a PyTorch RNN — to accurately classify emails as spam or ham (legitimate), using natural language processing techniques on real-world datasets.

---

## Dataset / Data Sources
| Dataset | Source | Description |
|---------|--------|-------------|
| Apache SpamAssassin | [spamassassin.apache.org](https://spamassassin.apache.org/old/publiccorpus/) | Public corpus of labeled spam and ham emails |
| Kaggle Spam Email Dataset | [kaggle.com](https://www.kaggle.com/datasets/purusinghvi/email-spam-classification-dataset) | CSV-formatted dataset with email text and labels |

**Features used:** Email body, email subject, email sender

## Current Implementation Progress
- [x] Project structure initialized
- [x] PyTorch RNN model defined (`src/rnn.py`)
- [x] Train script implemented
- [x] Train RNN implemented
- [ ] LLM Embeddings + Neural Network
- [ ] LLM Scoring
- [ ] Custom Transformer Architecture
---

## References
- [BERT (Devlin et al., 2019)](https://arxiv.org/abs/1810.04805)
- [LLaMA (Meta AI)](https://ai.meta.com/llama/)
- [HuggingFace Transformers](https://huggingface.co/docs/transformers)
- Logistic Regression — scikit-learn documentation
- Word Embeddings — GloVe / nn.Embedding (PyTorch)
- Sentiment Analysis as auxiliary signal — VADER / TextBlob

---

## Setup
```bash
pip install -r requirements.txt
python scripts/preprocess.py
python scripts/train_rnn.py
# Launch demo
python webapp/app.py
```
