import streamlit as st
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from captum.attr import IntegratedGradients
import plotly.graph_objects as go
# CONFIG
MODEL_PATH = "./exported_model"
MAX_LEN = 128
DEFAULT_THRESHOLD = 0.59

device = torch.device("cpu")
torch.set_default_device("cpu")

# REWRITE VOCABULARY

PROFANITY_REPLACEMENTS = {
    "fuck": "",
    "fucking": "",
    "shit": "",
    "bullshit": "nonsense",
    "asshole": "person",
    "bitch": "person",
    "bastard": "person",
    "dumbass": "person",
    "jerk": "person",
    "moron": "person",
    "retard": "person",
    "hell": "",
    "damn": "",
}

INSULT_REPLACEMENTS = {
    "idiot": "person",
    "stupid": "unwise",
    "dumb": "uninformed",
    "pathetic": "unhelpful",
    "useless": "not effective",
    "terrible": "not good",
    "awful": "unpleasant",
    "horrible": "poor",
    "worst": "not ideal",
    "lazy": "unmotivated",
}

HATE_REPLACEMENTS = {
    "hate": "dislike",
    "despise": "strongly dislike",
    "loathe": "do not like",
}

SOFTENING_PHRASES = [
    "I think",
    "it seems that",
    "in my opinion",
    "from my perspective",
    "I feel that",
]

SAFE_PHRASE_REPLACEMENTS = [
    "please leave",
    "this is inappropriate",
    "I strongly disagree",
    "let’s keep this respectful",
    "I don’t agree with this",
    "this could be handled better",
    ""
]

# LOAD MODEL

@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_PATH,
        local_files_only=True
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_PATH,
        local_files_only=True,
        torch_dtype=torch.float32
    )

    model.eval()
    model.to("cpu")
    return tokenizer, model


tokenizer, model = load_model()

# FORWARD FUNCTION (XAI)
def forward_func(inputs_embeds, attention_mask):
    outputs = model(
        inputs_embeds=inputs_embeds,
        attention_mask=attention_mask
    )
    return outputs.logits.squeeze(-1)


ig = IntegratedGradients(forward_func)

# TOXICITY SCORE
def score_toxicity(text):
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True
    )

    with torch.no_grad():
        logits = model(**inputs).logits
        prob = torch.sigmoid(logits)[0][0].item()

    return prob
# BIAS & FAIRNESS ANALYSIS
def bias_fairness_analysis():
    """
    Returns toxicity scores for neutral identity-based sentences
    """

    templates = [
        "I am a {}",
        "This person is a {}",
        "The individual identifies as {}",
        "They are a {}"
    ]

    identity_groups = {
        "Gender": ["man", "woman", "non-binary person"],
        "Religion": ["Christian", "Muslim", "Hindu", "Jewish", "Buddhist"],
        "Nationality": ["Indian", "American", "British", "Chinese"],
        "Sexual Orientation": ["gay person", "straight person", "bisexual person"],
        "Other": ["student", "engineer", "teacher"]
    }

    results = []

    for category, identities in identity_groups.items():
        for identity in identities:
            for template in templates:
                sentence = template.format(identity)
                score = score_toxicity(sentence)

                results.append({
                    "Category": category,
                    "Identity": identity,
                    "Sentence": sentence,
                    "Toxicity": round(score, 4)
                })

    return results

# PREDICTION
def predict(text, threshold):
    prob = score_toxicity(text)
    label = "Toxic" if prob >= threshold else "Non-toxic"
    return prob, label

# INTEGRATED GRADIENTS
def explain_text(text):
    encoded = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_LEN
    )

    input_ids = encoded["input_ids"]
    attention_mask = encoded["attention_mask"]

    embeddings = model.distilbert.embeddings(input_ids)
    baseline = torch.zeros_like(embeddings)

    attributions = ig.attribute(
        inputs=embeddings,
        baselines=baseline,
        additional_forward_args=(attention_mask,),
        n_steps=50
    )

    scores = attributions.sum(dim=-1).squeeze(0)
    scores = scores / torch.norm(scores)

    tokens = tokenizer.convert_ids_to_tokens(input_ids.squeeze(0))
    return tokens, scores.detach().numpy()

# COUNTERFACTUAL EXPLANATION
def counterfactual_analysis(text, tokens, scores):
    original_prob = score_toxicity(text)

    best_word = None
    best_prob = original_prob

    for tok, score in zip(tokens, scores):
        if score <= 0 or tok in ["[CLS]", "[SEP]", "[PAD]"]:
            continue

        clean = tok.replace("##", "")
        modified = text.replace(clean, "", 1).strip()

        if not modified:
            continue

        new_prob = score_toxicity(modified)
        if new_prob < best_prob:
            best_prob = new_prob
            best_word = clean

    return {
        "best_word": best_word,
        "original_prob": original_prob,
        "new_prob": best_prob,
    }
def explain_why_toxic(text, tokens, scores, threshold, prob):
    """
    Generates natural language explanation for toxicity
    """

    PROFANITY = {
        "fuck", "shit", "asshole", "bitch", "bastard",
        "damn", "hell", "bullshit", "jerk"
    }

    INSULTS = {
        "idiot", "stupid", "dumb", "moron",
        "pathetic", "useless", "lazy", "worst"
    }

    COMMANDS = {
        "get", "leave", "go", "shut"
    }

    reasons = []

    text_lower = text.lower()

    # Rule-based detection
    if any(w in text_lower for w in PROFANITY):
        reasons.append("strong profanity")

    if any(w in text_lower for w in INSULTS):
        reasons.append("direct insult")

    if "you" in text_lower and any(w in text_lower for w in INSULTS | PROFANITY):
        reasons.append("personal attack")

    if any(w in text_lower.split()[:2] for w in COMMANDS):
        reasons.append("aggressive command")

    # XAI-based reinforcement
    top_tokens = [
        tok.replace("##", "")
        for tok, score in zip(tokens, scores)
        if score > 0.15 and tok not in ["[CLS]", "[SEP]", "[PAD]"]
    ]

    if top_tokens:
        reasons.append(
            f"key words like {', '.join(set(top_tokens[:3]))}"
        )
    # Final explanation
    if prob < threshold:
        return "The model does not find strong toxic signals in this comment."

    if not reasons:
        return "The comment contains subtle language patterns associated with toxicity."

    explanation = (
        "This comment is considered toxic because it contains "
        + ", ".join(reasons[:-1])
        + (" and " + reasons[-1] if len(reasons) > 1 else reasons[0])
        + "."
    )

    return explanation

def extract_toxic_phrase(text, toxic_word, window=3):
    words = text.split()
    for i, w in enumerate(words):
        if toxic_word.lower() in w.lower():
            start = max(0, i - window)
            end = min(len(words), i + window + 1)
            return " ".join(words[start:end])
    return toxic_word
def replace_individual_words(text):
    words = text.split()
    new_words = []

    for w in words:
        clean = w.lower().strip(".,!?")

        if clean in PROFANITY_REPLACEMENTS:
            new_words.append(PROFANITY_REPLACEMENTS[clean])
        elif clean in INSULT_REPLACEMENTS:
            new_words.append(INSULT_REPLACEMENTS[clean])
        elif clean in HATE_REPLACEMENTS:
            new_words.append(HATE_REPLACEMENTS[clean])
        else:
            new_words.append(w)

    return " ".join(new_words)


def rewrite_text_model_in_loop(text, tokens, scores, max_passes=2):
    original_prob = score_toxicity(text)

    current_text = text
    current_prob = original_prob

    ranked = sorted(
        zip(tokens, scores),
        key=lambda x: x[1],
        reverse=True
    )

    toxic_tokens = [
        tok.replace("##", "")
        for tok, score in ranked
        if score > 0 and tok not in ["[CLS]", "[SEP]", "[PAD]"]
    ]

    if not toxic_tokens:
        return None, original_prob, original_prob

    SAFE_PHRASE_REPLACEMENTS = [
        "please leave",
        "this is inappropriate",
        "I strongly disagree",
        "let’s keep this respectful",
        "I don’t agree with this",
        "this feels unfair",
        "I think this could be handled better",
        ""
    ]
    PROFANITY_REPLACEMENTS = {
        "fuck": "",
        "fucking": "",
        "shit": "",
        "bullshit": "nonsense",
        "asshole": "person",
        "bitch": "person",
        "bastard": "person",
        "dumbass": "person",
        "jerk": "person",
        "moron": "person",
        "retard": "person",
        "hell": "",
        "damn": "",
    }
    INSULT_REPLACEMENTS = {
        "idiot": "person",
        "stupid": "unwise",
        "dumb": "uninformed",
        "pathetic": "unhelpful",
        "useless": "not effective",
        "terrible": "not good",
        "awful": "unpleasant",
        "horrible": "poor",
        "worst": "not ideal",
        "lazy": "unmotivated",
    }
    HATE_REPLACEMENTS = {
        "hate": "dislike",
        "despise": "strongly dislike",
        "loathe": "do not like",
        "can't stand": "find difficult",
        "sick of": "tired of",
    }
    COMMAND_PHRASE_REPLACEMENTS = [
        "please leave",
        "let’s keep this respectful",
        "I strongly disagree",
        "this feels inappropriate",
        "this conversation should stay respectful",
        "let’s discuss this calmly",
        "I don’t agree with this",
        "this could be handled better",
        "I think this is unfair",
        "I’m not comfortable with this",
    ]

    PERSONAL_ATTACK_REPLACEMENTS = [
        "I disagree with this behavior",
        "this does not seem appropriate",
        "I think this approach is problematic",
        "this comes across as unfair",
        "I don’t think this is constructive",
    ]
    SOFTENING_PHRASES = [
        "I think",
        "it seems that",
        "in my opinion",
        "from my perspective",
        "I feel that",
        "it might be better to",
    ]
    SAFE_FALLBACKS = [
        "let’s keep this respectful",
        "this discussion could be more constructive",
        "I disagree, but let’s stay civil",
        "this feels inappropriate",
        "let’s handle this calmly",
        ""
    ]

    for _ in range(max_passes):

        toxic_word = extract_toxic_phrase(current_text, toxic_tokens[0]).split()[0]
        toxic_phrase = extract_toxic_phrase(current_text, toxic_word)

        best_candidate = current_text
        best_candidate_score = current_prob

        #  Phrase replacement
        for rep in SAFE_PHRASE_REPLACEMENTS:
            candidate = current_text.replace(toxic_phrase, rep, 1).strip()
            if not candidate:
                continue

            score = score_toxicity(candidate)
            if score < best_candidate_score:
                best_candidate = candidate
                best_candidate_score = score

        #  Word-level replacement
        candidate_words = replace_individual_words(current_text)
        score = score_toxicity(candidate_words)
        if score < best_candidate_score:
            best_candidate = candidate_words
            best_candidate_score = score

        #  Tone softening
        for prefix in SOFTENING_PHRASES:
            candidate = f"{prefix} {current_text}"
            score = score_toxicity(candidate)
            if score < best_candidate_score:
                best_candidate = candidate
                best_candidate_score = score

        # Stop if no improvement
        if best_candidate_score >= current_prob:
            break

        current_text = best_candidate
        current_prob = best_candidate_score

        if current_prob < DEFAULT_THRESHOLD:
            break


    return current_text, original_prob, current_prob


# CONFIDENCE GAUGE
def confidence_gauge(prob, threshold):
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=prob * 100,
            number={"suffix": "%"},
            title={"text": "Toxicity Confidence"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": "darkred"},
                "steps": [
                    {"range": [0, threshold * 100], "color": "lightgreen"},
                    {"range": [threshold * 100, 70], "color": "khaki"},
                    {"range": [70, 100], "color": "salmon"},
                ],
                "threshold": {
                    "line": {"color": "black", "width": 4},
                    "value": threshold * 100,
                },
            },
        )
    )
    fig.update_layout(height=300, margin=dict(t=40, b=0))
    return fig

# STREAMLIT UI
st.set_page_config(page_title="Toxic Comment Detector + XAI", layout="wide")
st.title("🧠 Toxic Comment Detector with Explainable AI")

threshold = st.slider(
    "🎚 Decision Threshold",
    0.1, 0.9, DEFAULT_THRESHOLD, 0.01
)

mode = st.radio(
    "🧭 Select Mode",
    ["Standard Analysis", "⚖️ Bias & Fairness Analysis"]
)

# BIAS & FAIRNESS MODE
if mode == "⚖️ Bias & Fairness Analysis":

    st.subheader("⚖️ Bias & Fairness Analysis")

    with st.spinner("Analyzing model fairness..."):
        bias_results = bias_fairness_analysis()

    import pandas as pd
    df = pd.DataFrame(
        bias_results,
        columns=["Category", "Identity", "Sentence", "Toxicity"]
    )

    st.dataframe(df, use_container_width=True)

    st.info(
        "ℹ️ These are neutral identity-based sentences. "
        "Large score differences may indicate potential model bias."
    )

# STANDARD ANALYSIS MODE
else:
    text = st.text_area("Enter a comment:", height=120)

    if st.button("Analyze"):
        if text.strip() == "":
            st.warning("Please enter text")
        else:
            prob, label = predict(text, threshold)
            tokens, scores = explain_text(text)

            # Counterfactual
            st.subheader("🧪 Counterfactual Explanation")
            cf = counterfactual_analysis(text, tokens, scores)

            if cf["best_word"] is None:
                st.info("No impactful single-word removal found.")
            else:
                st.write(f"🧨 **Most influential word:** `{cf['best_word']}`")
                st.write(
                    f"📉 Removing it drops toxicity "
                    f"`{cf['original_prob']:.3f}` → `{cf['new_prob']:.3f}`"
                )

            # Rewrite
            st.subheader("✍️ Rewrite Assistant")
            rewrite, old, new = rewrite_text_model_in_loop(text, tokens, scores)

            if rewrite is None or rewrite == text:
                st.info("No safer rewrite found.")
            else:
                st.success(rewrite)
                st.write(
                    f"📉 Toxicity reduced `{old:.3f}` → `{new:.3f}`"
                )

            # Prediction
            st.subheader("📊 Prediction")
            st.write(f"**Probability:** `{prob:.3f}`")
            st.write(f"**Label:** {label}")

            st.plotly_chart(
                confidence_gauge(prob, threshold),
                use_container_width=True
            )
            # WHY THIS IS TOXIC
            st.subheader("🧠 Why This Is Toxic")

            reasoning = explain_why_toxic(
                text=text,
                tokens=tokens,
                scores=scores,
                threshold=threshold,
                prob=prob
            )

            st.info(reasoning)

            # Explanation
            st.subheader("🔍 Explanation")
            html = ""
            for tok, score in zip(tokens, scores):
                if tok in ["[CLS]", "[SEP]", "[PAD]"]:
                    continue
                color = (
                    f"rgba(255,0,0,{abs(score):.2f})"
                    if score > 0
                    else f"rgba(0,200,0,{abs(score):.2f})"
                )
                html += f"""
                <span style="
                    background:{color};
                    padding:4px;
                    margin:2px;
                    border-radius:4px;
                    display:inline-block;">
                    {tok.replace("##","")}
                </span>
                """
            st.markdown(html, unsafe_allow_html=True)

