# Toxic Comment Detection Model Training Script
# Auto-generated from notebook

import os
import json
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments
)
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    f1_score,
    precision_recall_curve,
    confusion_matrix,
    classification_report,
    precision_score,
    recall_score
)

# Disable warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["WANDB_DISABLED"] = "true"

# Fix Windows console encoding for emojis
import sys
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

print("=" * 80)
print("TOXIC COMMENT DETECTOR - MODEL TRAINING")
print("=" * 80)

# Load config
print("\n📋 Loading configuration...")
with open("training_config.json", "r") as f:
    config = json.load(f)

print("✅ Configuration loaded")

# File paths
TRAIN_PATH = config["data"]["train_path"]
TEST_PATH = config["data"]["test_path"]
MODEL_NAME = config["model"]["name"]
SAVE_DIR = config["output"]["model_dir"]

print(f"\n📁 Training data: {TRAIN_PATH}")
print(f"📁 Test data: {TEST_PATH}")
print(f"🤖<brace> Model: {MODEL_NAME}")
print(f"💾 Output: {SAVE_DIR}")

# Check files
if not os.path.exists(TRAIN_PATH):
    raise FileNotFoundError(f"Training file not found: {TRAIN_PATH}")
if not os.path.exists(TEST_PATH):
    raise FileNotFoundError(f"Test file not found: {TEST_PATH}")

print("✅ All files found\n")

# Load training data
print("=" * 80)
print("STEP 1: LOADING DATA")
print("=" * 80)
print("\n📊 Loading training data...")
train_df = pd.read_csv(TRAIN_PATH)

print(f"Shape: {train_df.shape}")
print(f"Columns: {train_df.columns.tolist()}")

# Check for required columns
if "comment_text" not in train_df.columns:
    raise ValueError("'comment_text' column not found!")
if "toxic" not in train_df.columns:
    raise ValueError("'toxic' column not found!")

print("\n✅ Required columns found")

# Data statistics
print(f"\n📈 Data Statistics:")
print(f"Total comments: {len(train_df):,}")
print(f"Toxic comments: {train_df['toxic'].sum():,}")
print(f"Non-toxic comments: {(len(train_df) - train_df['toxic'].sum()):,}")
print(f"Toxicity rate: {train_df['toxic'].mean():.2%}")

# Clean data
print("\n🧹 Preparing data...")
train_df = train_df[["comment_text", "toxic"]].copy()
train_df["toxic"] = train_df["toxic"].astype(float)

# Remove NaN values
initial_len = len(train_df)
train_df = train_df.dropna()
if initial_len > len(train_df):
    print(f"Removed {initial_len - len(train_df)} rows with missing values")

# Split into train and validation
dataset = Dataset.from_pandas(train_df)
dataset = dataset.train_test_split(
    test_size=config["data"]["validation_split"],
    seed=config["data"]["random_seed"]
)

train_ds = dataset["train"]
val_ds = dataset["test"]

print(f"\n✅ Dataset split complete:")
print(f"   Training: {len(train_ds):,} samples")
print(f"   Validation: {len(val_ds):,} samples")

# Load tokenizer
print(f"\n🔤 Loading tokenizer: {MODEL_NAME}")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# Tokenization function
def preprocess(batch):
    return tokenizer(
        batch["comment_text"],
        truncation=True,
        padding="max_length",
        max_length=config["model"]["max_length"]
    )

# Tokenize datasets
print("🔄 Tokenizing datasets (this may take a few minutes)...")
train_ds = train_ds.map(preprocess, batched=True)
val_ds = val_ds.map(preprocess, batched=True)

# Rename label column
train_ds = train_ds.rename_column("toxic", "labels")
val_ds = val_ds.rename_column("toxic", "labels")

# Set format for PyTorch
train_ds.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
val_ds.set_format("torch", columns=["input_ids", "attention_mask", "labels"])

print("✅ Tokenization complete")

# Load model
print("\n" + "=" * 80)
print("STEP 2: MODEL TRAINING")
print("=" * 80)
print("\n🤖 Loading model...")
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=config["model"]["num_labels"]
)
print(f"✅ Model loaded: {MODEL_NAME}")

# Metrics function
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    logits = np.squeeze(logits)
    labels = labels.astype(int)
    probs = 1 / (1 + np.exp(-logits))  # sigmoid

    return {
        "roc_auc": roc_auc_score(labels, probs),
        "pr_auc": average_precision_score(labels, probs),
    }

# Training arguments
training_args = TrainingArguments(
    output_dir="./toxicity_model",
    per_device_train_batch_size=config["training"]["batch_size"],
    per_device_eval_batch_size=config["training"]["batch_size"],
    num_train_epochs=config["training"]["epochs"],
    learning_rate=config["training"]["learning_rate"],
    weight_decay=config["training"]["weight_decay"],
    warmup_steps=config["training"]["warmup_steps"],
    logging_steps=config["training"]["logging_steps"],
    eval_steps=config["training"]["eval_steps"],
    save_steps=config["training"]["save_steps"],
    eval_strategy="steps",
    save_strategy="steps",
    load_best_model_at_end=True,
    metric_for_best_model="roc_auc",
    fp16=config["training"]["fp16"],
    report_to="none"
)

# Create trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

# Train
print("\n🚀 Starting training...")
print(f"   Epochs: {config['training']['epochs']}")
print(f"   Batch size: {config['training']['batch_size']}")
print(f"   Learning rate: {config['training']['learning_rate']}")
print("\n⏳ This will take 15-30 minutes...\n")

trainer.train()

print("\n✅ Training completed!")

# Evaluate
print("\n" + "=" * 80)
print("STEP 3: EVALUATION & THRESHOLD TUNING")
print("=" * 80)

print("\n📊 Evaluating model on validation set...")
val_metrics = trainer.evaluate()

print("\n✅ Validation Metrics:")
for key, value in val_metrics.items():
    if key.startswith("eval_"):
        metric_name = key.replace("eval_", "").upper()
        print(f"   {metric_name}: {value:.4f}")

# Get predictions
print("\n🔮 Generating predictions...")
preds = trainer.predict(val_ds)
logits = np.squeeze(preds.predictions)
labels = preds.label_ids.astype(int)
probs = 1 / (1 + np.exp(-logits))

print(f"✅ Generated {len(probs):,} predictions")

# Threshold tuning
print("\n🎯 Finding optimal decision threshold...")

thresholds = np.linspace(
    config["threshold"]["range_start"],
    config["threshold"]["range_end"],
    config["threshold"]["num_thresholds"]
)

f1_scores = []
precision_scores = []
recall_scores = []

for t in thresholds:
    predictions = (probs >= t).astype(int)
    f1 = f1_score(labels, predictions)
    f1_scores.append(f1)

    prec = precision_score(labels, predictions, zero_division=0)
    rec = recall_score(labels, predictions, zero_division=0)
    precision_scores.append(prec)
    recall_scores.append(rec)

# Find best threshold
best_idx = np.argmax(f1_scores)
best_threshold = thresholds[best_idx]
best_f1 = f1_scores[best_idx]

print(f"\n✅ Optimal Threshold: {best_threshold:.3f}")
print(f"✅ Best F1 Score: {best_f1:.4f}")

# Confusion matrix at optimal threshold
print("\n📊 Performance at optimal threshold:")
optimal_preds = (probs >= best_threshold).astype(int)
cm = confusion_matrix(labels, optimal_preds)

print("\nConfusion Matrix:")
print(f"                Predicted")
print(f"              Non-toxic  Toxic")
print(f"Actual Non-toxic  {cm[0][0]:6d}  {cm[0][1]:6d}")
print(f"       Toxic      {cm[1][0]:6d}  {cm[1][1]:6d}")

# Classification report
print("\n📋 Classification Report:")
print(classification_report(labels, optimal_preds, target_names=['Non-toxic', 'Toxic']))

# Save model
print("\n" + "=" * 80)
print("STEP 4: SAVING MODEL")
print("=" * 80)

print(f"\n💾 Saving model to {SAVE_DIR}...")
trainer.save_model(SAVE_DIR)
tokenizer.save_pretrained(SAVE_DIR)

# Save optimal threshold
threshold_info = {
    "optimal_threshold": float(best_threshold),
    "f1_score": float(best_f1),
    "roc_auc": float(val_metrics["eval_roc_auc"]),
    "pr_auc": float(val_metrics["eval_pr_auc"])
}

with open(os.path.join(SAVE_DIR, "threshold_info.json"), "w") as f:
    json.dump(threshold_info, f, indent=2)

print(f"✅ Model saved successfully!")
print(f"✅ Threshold info saved to {SAVE_DIR}/threshold_info.json")

# Sanity check
print("\n" + "=" * 80)
print("STEP 5: SANITY CHECK")
print("=" * 80)

print("\n🧪 Testing model with sample comments...\n")

test_samples = [
    "You are a stupid idiot",
    "Thank you so much for your help",
    "Go die, nobody likes you",
    "This is a wonderful community",
    "Fuck you asshole",
    "I really appreciate your thoughtful response",
    "You're pathetic and useless",
    "Have a great day!"
]

device = model.device

print("=" * 60)
for sample in test_samples:
    inputs = tokenizer(sample, return_tensors="pt", truncation=True, padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        logit = model(**inputs).logits.item()
        prob = 1 / (1 + torch.exp(-torch.tensor(logit))).item()

    label = "Toxic" if prob >= best_threshold else "Non-toxic"
    emoji = "🔴" if prob >= best_threshold else "🟢"

    print(f"{emoji} {prob:.3f} | {label:12s} | {sample}")

print("=" * 60)

print("\n" + "=" * 80)
print("✅ TRAINING COMPLETE!")
print("=" * 80)
print(f"\n📌 Model saved to: {SAVE_DIR}")
print(f"📌 Use threshold: {best_threshold:.3f} in the Streamlit app")
print(f"📌 Run the app with: streamlit run app.py")
print("\n" + "=" * 80)
