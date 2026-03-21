# Toxic Comment Detector with XAI

Detects toxic comments and shows you which words made it toxic. Simple as that.

## What does it do?

- Tells you if a comment is toxic or not (98.6% accuracy)
- Shows a heatmap of words - red ones are toxic, green ones are fine
- Suggests how to rewrite toxic comments nicely
- Can analyze multiple comments at once from a CSV
- Tests if the model is biased against certain groups

## How to run it

**1. Clone and install**

```bash
git clone https://github.com/kritikGianta/toxic-detect-xai.git
cd toxic-detect-xai
pip install -r requirements.txt
```

**2. Start the app**

```bash
streamlit run app.py
```

**3. Open in browser**

Go to http://localhost:8501

That's it. Type a comment and hit Analyze.

## Files

| File | What it does |
|------|--------------|
| `app.py` | The main web app |
| `utils.py` | All the ML functions |
| `config.py` | Settings and word lists |
| `styles.css` | Makes it look nice |
| `exported_model/` | The trained model |
| `kaggle_training.ipynb` | Training code (run on Kaggle with GPU) |

## How it works (simple version)

1. You type a comment
2. DistilBERT (a language model) reads it
3. It gives a toxicity score from 0 to 1
4. Integrated Gradients figures out which words mattered
5. You see red/green highlighting on words
6. If toxic, it suggests a nicer way to say it

## Training

Model was trained on Kaggle (free T4 GPU). Takes about 15 mins.

If you want to retrain:
1. Go to Kaggle, upload `kaggle_training.ipynb`
2. Add the Jigsaw Toxic Comment dataset
3. Turn on GPU
4. Run it
5. Download the model folder

## Results

- ROC-AUC: 0.986
- F1 Score: 0.84
- Threshold: 0.61

## Built with

- PyTorch
- HuggingFace Transformers
- Captum (for XAI)
- Streamlit
- Plotly

## Author

Kritik
