🧠 Toxic Comment Detector with Explainable AI

An NLP-based toxic comment detection system built using DistilBERT, enhanced with Explainable AI (XAI), counterfactual reasoning, bias analysis, and an experimental rewrite assistant.

This project focuses not only on what the model predicts, but why it makes those predictions.

🚀 Features
🔹 Toxicity Classification

Binary classification: Toxic / Non-toxic

Outputs a probability score

Adjustable decision threshold

🔹 Explainable AI (Integrated Gradients)

Token-level attribution using Integrated Gradients

Visual heatmap showing words that increase/decrease toxicity

🔹 Counterfactual Explanations

Identifies the most influential word

Shows how removing it changes toxicity score

🔹 “Why This Is Toxic” (Natural Language Explanation)

Human-readable explanation combining:

Profanity & insult detection

Aggressive language patterns

XAI token importance

🔹 Bias & Fairness Analysis ⚖️

Evaluates model behavior on neutral identity-based sentences

Helps surface potential bias across gender, religion, nationality, etc.

🔹 Experimental Rewrite Assistant ✍️

Uses XAI to reduce toxicity via phrase and word replacement
⚠️ Limitation: The rewrite focuses on toxicity reduction and may alter original meaning.

🧠 NLP Techniques Used

Transformer-based sequence classification

Contextual embeddings

Token attribution (XAI)

Counterfactual reasoning

Bias evaluation in NLP models

🛠 Tech Stack

Python, PyTorch

HuggingFace Transformers

Captum (XAI)

Streamlit

Plotly

📦 Project Structure
├── app.py              # Complete Streamlit app (self-contained)
├── exported_model/     # Trained model & tokenizer
├── README.md

🎯 Use Cases

Content moderation research

Explainable AI demos

NLP + Responsible AI portfolio project