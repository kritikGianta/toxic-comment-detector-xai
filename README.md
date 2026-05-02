# Toxic Comment Detector

A local-first toxic comment moderation app built with a fine-tuned DistilBERT model, explainability tooling, batch CSV screening, and rewrite suggestions for safer phrasing.

## Overview

This project is designed to help review user-generated comments before they are posted or approved. It combines:

- binary toxicity detection with a fine-tuned transformer model
- token-level explanation using Integrated Gradients
- respectful rewrite suggestions driven by the trained classifier
- CSV batch screening for moderation workflows
- a lightweight fairness check across identity templates

The application runs locally in Streamlit and loads the exported model directly from this repository.

## Highlights

- Local inference: no external API required for scoring comments
- Explainable predictions: word-level attribution to show what drove the score
- Practical moderation flow: single review, batch screening, fairness check, and session history
- Tuned threshold: `0.61`, selected from validation data for the best F1 score
- Git LFS support: large model weights are tracked correctly for GitHub

## Model

- Base model: `distilbert-base-uncased`
- Architecture: `DistilBertForSequenceClassification`
- Output: single-logit binary toxicity score
- Max sequence length: `128`
- Runtime device in app: `cpu`

The app converts the model's logit into a probability with a sigmoid and compares it against the configured threshold.

## Training Configuration

Training settings are defined in [training_config.json](./training_config.json):

- Epochs: `2`
- Batch size: `32`
- Learning rate: `3e-5`
- Weight decay: `0.01`
- Warmup steps: `500`
- Validation split: `10%`
- Random seed: `42`
- Mixed precision: `fp16=true`

The current training script fine-tunes on `train.csv`, creates an internal validation split, evaluates on that split, then exports the model and threshold metadata.

## Threshold Selection

The default decision threshold is `0.61`.

It was selected by:

1. generating validation probabilities after training
2. sweeping thresholds from `0.10` to `0.90`
3. computing F1 at each threshold
4. saving the threshold with the best F1 score

Saved threshold metrics in [exported_model/threshold_info.json](./exported_model/threshold_info.json):

- F1: `0.8388`
- ROC-AUC: `0.9856`
- PR-AUC: `0.9120`

This makes `0.61` a better moderation cutoff than using `0.50` by default.

## App Features

### 1. Single-comment review

- score a comment as toxic or non-toxic
- show severity and recommendation
- display token-level attribution
- explain why the model flagged the text
- suggest a calmer rewrite when it improves the score

### 2. CSV screening

- upload a CSV file
- auto-detect a likely text column
- score comments in batches for better performance
- export the reviewed CSV with scores and labels

### 3. Fairness check

- test neutral identity-based templates
- compare toxicity scores across identity groups
- surface obvious bias drift as a quick smoke test

## Repository Structure

```text
.
|-- app.py
|-- utils.py
|-- config.py
|-- styles.css
|-- requirements.txt
|-- training_config.json
|-- train_model.py
|-- train_model_gpu.py
|-- kaggle_training.ipynb
|-- toxic-comment-detect-model.ipynb
|-- toxic-comment-kaggle-model.ipynb
|-- assets/
|   `-- sample_comments.csv
`-- exported_model/
    |-- config.json
    |-- model.safetensors
    |-- threshold_info.json
    |-- tokenizer.json
    `-- tokenizer_config.json
```

## Run Locally

### 1. Clone the repository

```bash
git clone https://github.com/kritikGianta/toxic-comment-detector-xai.git
cd toxic-comment-detector-xai
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Start the app

```bash
streamlit run app.py
```

Then open [http://localhost:8501](http://localhost:8501).

## Training and Retraining

There are two main training paths in this repository:

- [train_model.py](./train_model.py): standard training/export flow
- [train_model_gpu.py](./train_model_gpu.py): GPU-oriented training variant

You can also work from the included notebooks if you prefer an interactive workflow.

Notes:

- `train.csv` and `test.csv` are intentionally ignored because they are too large for Git
- the exported model is already included, so retraining is not required to run the app
- model weights are stored through Git LFS

## Known Limitations

- The current classifier is binary, not a full multi-label toxicity taxonomy
- Validation metrics come from a split of `train.csv`, not a separate held-out benchmark in this repo
- Rewrite suggestions are heuristic and classifier-guided, not LLM-generated
- Fairness analysis is lightweight and should not be treated as a complete bias audit
- Long comments may lose context because inference is capped at `128` tokens

## Tech Stack

- Python
- PyTorch
- Hugging Face Transformers
- Captum
- Streamlit
- Plotly
- pandas
- scikit-learn

## Author

Kritik
