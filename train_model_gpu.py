# GPU-Optimized Toxic Comment Detection Model Training
# Optimized for GTX 1650 (4GB VRAM)

import os
import json
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
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
    precision_score,
    recall_score,
    confusion_matrix,
    classification_report
)

# Disable warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["WANDB_DISABLED"] = "true"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

# Fix Windows console encoding
import sys
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

print("=" * 80)
print("TOXIC COMMENT DETECTOR - GPU TRAINING")
print("=" * 80)

# Check CUDA
print("\n🖥️ GPU Information:")
print(f"   CUDA Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"   GPU Name: {torch.cuda.get_device_name(0)}")
    print(f"   GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    print(f"   PyTorch CUDA Version: {torch.version.cuda}")
else:
    print("   ⚠️ WARNING: CUDA not available. Training will be VERY slow on CPU.")
    print("   Install CUDA-enabled PyTorch: pip install torch --index-url https://download.pytorch.org/whl/cu118")

# Load config
print("\n📋 Loading configuration...")
with open("training_config.json", "r") as f:
    config = json.load(f)

# GPU-optimized settings for GTX 1650 (4GB VRAM)
if torch.cuda.is_available():
    # Reduce batch size for 4GB VRAM
    config["training"]["batch_size"] = 16  # Reduced from 32
    # Disable FP16 if it causes issues
    config["training"]["fp16"] = False  # More stable on older GPUs
    print("✅ Using GPU-optimized settings for GTX 1650")
else:
    config["training"]["batch_size"] = 8  # Even smaller for CPU
    config["training"]["fp16"] = False

TRAIN_PATH = config["data"]["train_path"]
TEST_PATH = config["data"]["test_path"]
MODEL_NAME = config["model"]["name"]
SAVE_DIR = config["output"]["model_dir"]

print(f"\n📁 Training data: {TRAIN_PATH}")
print(f"📁 Test data: {TEST_PATH}")
print(f"🤖 Model: {MODEL_NAME}")
print(f"💾 Output: {SAVE_DIR}")
print(f"🔢 Batch size: {config['training']['batch_size']}")

# Check files
if not os.path.exists(TRAIN_PATH):
    raise FileNotFoundError(f"Training file not found: {TRAIN_PATH}")

print("✅ Training file found\n")

# Load and prepare data
print("=" * 80)
print("STEP 1: LOADING DATA")
print("=" * 80)

print("\n📊 Loading training data...")
train_df = pd.read_csv(TRAIN_PATH)

print(f"Shape: {train_df.shape}")
print(f"\n📈 Data Statistics:")
print(f"Total comments: {len(train_df):,}")
print(f"Toxic comments: {train_df['toxic'].sum():,}")
print(f"Toxicity rate: {train_df['toxic'].mean():.2%}")

# Prepare data
print("\n🧹 Preparing data...")
train_df = train_df[["comment_text", "toxic"]].copy()
train_df["toxic"] = train_df["toxic"].astype(float)
train_df = train_df.dropna()

# Split
dataset = Dataset.from_pandas(train_df)
dataset = dataset.train_test_split(
    test_size=config["data"]["validation_split"],
    seed=config["data"]["random_seed"]
)

train_ds = dataset["train"]
val_ds = dataset["test"]

print(f"\n✅ Dataset split:")
print(f"   Training: {len(train_ds):,} samples")
print(f"   Validation: {len(val_ds):,} samples")

# Tokenization
print(f"\n🔤 Loading tokenizer: {MODEL_NAME}")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def preprocess(batch):
    return tokenizer(
        batch["comment_text"],
        truncation=True,
        padding="max_length",
        max_length=config["model"]["max_length"]
    )

print("🔄 Tokenizing datasets...")
train_ds = train_ds.map(preprocess, batched=True, desc="Tokenizing train")
val_ds = val_ds.map(preprocess, batched=True, desc="Tokenizing validation")

train_ds = train_ds.rename_column("toxic", "labels")
val_ds = val_ds.rename_column("toxic", "labels")

train_ds.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
val_ds.set_format("torch", columns=["input_ids", "attention_mask", "labels"])

print("✅ Tokenization complete\n")

# Load model
print("=" * 80)
print("STEP 2: MODEL TRAINING")
print("=" * 80)

print("\n🤖 Loading model...")
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=config["model"]["num_labels"]
)

# Move model to GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
print(f"✅ Model loaded on: {device}")

# Metrics
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    logits = np.squeeze(logits)
    labels = labels.astype(int)
    probs = 1 / (1 + np.exp(-logits))

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
    report_to="none",
    dataloader_pin_memory=True,
    gradient_accumulation_steps=2  # Effective batch size = 16*2 = 32
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    processing_class=tokenizer,
    compute_metrics=compute_metrics
)

# Train
print("\n🚀 Starting training...")
print(f"   Epochs: {config['training']['epochs']}")
print(f"   Batch size: {config['training']['batch_size']}")
print(f"   Gradient accumulation: 2 steps (effective batch size: {config['training']['batch_size']*2})")
print(f"   Learning rate: {config['training']['learning_rate']}")
print(f"   Device: {device}")
print("\n⏳ Estimated time: 20-30 minutes on GPU, 8-9 hours on CPU\n")

trainer.train()

print("\n✅ Training completed!")

# Evaluation
print("\n" + "=" * 80)
print("STEP 3: EVALUATION")
print("=" * 80)

print("\n📊 Evaluating...")
val_metrics = trainer.evaluate()

print("\n✅ Validation Metrics:")
for key, value in val_metrics.items():
    if key.startswith("eval_"):
        print(f"   {key.replace('eval_', '').upper()}: {value:.4f}")

# Predictions
print("\n🔮 Generating predictions...")
preds = trainer.predict(val_ds)
logits = np.squeeze(preds.predictions)
labels = preds.label_ids.astype(int)
probs = 1 / (1 + np.exp(-logits))

# Threshold tuning
print("\n🎯 Finding optimal threshold...")

thresholds = np.linspace(0.1, 0.9, 81)
f1_scores = []

for t in thresholds:
    predictions = (probs >= t).astype(int)
    f1 = f1_score(labels, predictions)
    f1_scores.append(f1)

best_idx = np.argmax(f1_scores)
best_threshold = thresholds[best_idx]
best_f1 = f1_scores[best_idx]

print(f"✅ Optimal Threshold: {best_threshold:.3f}")
print(f"✅ Best F1 Score: {best_f1:.4f}")

# Confusion matrix
optimal_preds = (probs >= best_threshold).astype(int)
cm = confusion_matrix(labels, optimal_preds)

print("\n📊 Confusion Matrix:")
print(f"              Predicted")
print(f"            Non-toxic  Toxic")
print(f"Actual")
print(f"Non-toxic    {cm[0][0]:6d}  {cm[0][1]:6d}")
print(f"Toxic        {cm[1][0]:6d}  {cm[1][1]:6d}")

print("\n📋 Classification Report:")
print(classification_report(labels, optimal_preds, target_names=['Non-toxic', 'Toxic']))

# Save model
print("\n" + "=" * 80)
print("STEP 4: SAVING MODEL")
print("=" * 80)

print(f"\n💾 Saving model to {SAVE_DIR}...")
trainer.save_model(SAVE_DIR)
tokenizer.save_pretrained(SAVE_DIR)

threshold_info = {
    "optimal_threshold": float(best_threshold),
    "f1_score": float(best_f1),
    "roc_auc": float(val_metrics["eval_roc_auc"]),
    "pr_auc": float(val_metrics["eval_pr_auc"])
}

with open(os.path.join(SAVE_DIR, "threshold_info.json"), "w") as f:
    json.dump(threshold_info, f, indent=2)

print(f"✅ Model saved!")

# Sanity check
print("\n" + "=" * 80)
print("STEP 5: SANITY CHECK")
print("=" * 80)

test_samples = [
    "You are a stupid idiot",
    "Thank you so much for your help",
    "Go die, nobody likes you",
    "This is a wonderful community",
    "Fuck you asshole",
    "Have a great day!"
]

print("\n🧪 Testing model:\n")
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
print(f"\n📌 Model: {SAVE_DIR}")
print(f"📌 Threshold: {best_threshold:.3f}")
print(f"📌 Run app: streamlit run app.py")
print("\n" + "=" * 80)
