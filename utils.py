"""
Utility functions for toxic comment detection and analysis
"""

import torch
import numpy as np
import re
from config import *


def score_toxicity(text, model, tokenizer):
    """Calculate toxicity probability for given text"""
    if not text or not text.strip():
        return 0.0

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=MAX_LENGTH
    )

    # Remove token_type_ids for DistilBERT (it doesn't use them)
    inputs.pop("token_type_ids", None)

    with torch.no_grad():
        logits = model(**inputs).logits
        prob = torch.sigmoid(logits)[0][0].item()

    return prob


def predict(text, model, tokenizer, threshold):
    """Predict if text is toxic and return probability"""
    prob = score_toxicity(text, model, tokenizer)
    label = "Toxic" if prob >= threshold else "Non-toxic"
    return prob, label


def explain_text(text, model, tokenizer, ig):
    """Generate token-level attributions using Integrated Gradients"""
    if not text or not text.strip():
        return [], np.array([])

    encoded = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_LENGTH
    )

    input_ids = encoded["input_ids"]
    attention_mask = encoded["attention_mask"]

    embeddings = model.distilbert.embeddings(input_ids)
    baseline = torch.zeros_like(embeddings)

    attributions = ig.attribute(
        inputs=embeddings,
        baselines=baseline,
        additional_forward_args=(attention_mask,),
        n_steps=IG_STEPS
    )

    scores = attributions.sum(dim=-1).squeeze(0)
    scores = scores / torch.norm(scores)

    tokens = tokenizer.convert_ids_to_tokens(input_ids.squeeze(0))
    return tokens, scores.detach().numpy()


def explain_why_toxic(text, tokens, scores, threshold, prob):
    """Generate natural language explanation for toxicity"""
    reasons = []
    text_lower = text.lower()

    # Check for profanity
    if any(word in text_lower for word in PROFANITY_WORDS):
        reasons.append("strong profanity")

    # Check for insults
    if any(word in text_lower for word in INSULT_WORDS):
        reasons.append("direct insult")

    # Check for personal attacks
    if "you" in text_lower and any(word in text_lower for word in INSULT_WORDS | PROFANITY_WORDS):
        reasons.append("personal attack")

    # Check for aggressive commands
    words_at_start = text_lower.split()[:2]
    if any(word in words_at_start for word in COMMAND_WORDS):
        reasons.append("aggressive command")

    # Add XAI insights
    top_tokens = [
        tok.replace("##", "")
        for tok, score in zip(tokens, scores)
        if score > ATTRIBUTION_THRESHOLD and tok not in ["[CLS]", "[SEP]", "[PAD]"]
    ]

    if top_tokens and len(top_tokens) > 0:
        top_3 = ", ".join(set(top_tokens[:3]))
        reasons.append(f"words like '{top_3}' contribute strongly")

    # Build explanation
    if prob < threshold:
        return "The model does not find strong toxic signals in this comment."

    if not reasons:
        return "The comment contains subtle language patterns associated with toxicity."

    if len(reasons) == 1:
        explanation = f"This comment is toxic because it contains {reasons[0]}."
    else:
        explanation = (
            "This comment is toxic because it contains "
            + ", ".join(reasons[:-1])
            + f" and {reasons[-1]}."
        )

    return explanation


def counterfactual_analysis(text, tokens, scores, model, tokenizer):
    """Find top influential words and show impact of removing them"""
    original_prob = score_toxicity(text, model, tokenizer)

    # Get top toxic tokens (sorted by attribution score)
    token_impacts = []

    for tok, score in zip(tokens, scores):
        if score <= 0 or tok in ["[CLS]", "[SEP]", "[PAD]"]:
            continue

        clean_tok = tok.replace("##", "")

        # Try removing this token
        modified = re.sub(r'\b' + re.escape(clean_tok) + r'\b', '', text, count=1, flags=re.IGNORECASE)
        modified = re.sub(r'\s+', ' ', modified).strip()

        if not modified or modified == text:
            continue

        new_prob = score_toxicity(modified, model, tokenizer)
        impact = original_prob - new_prob

        if impact > 0:
            token_impacts.append({
                "word": clean_tok,
                "original_prob": original_prob,
                "new_prob": new_prob,
                "impact": impact
            })

    # Sort by impact and return top 3
    token_impacts.sort(key=lambda x: x["impact"], reverse=True)
    return token_impacts[:3]


def replace_word_with_punctuation(word, replacement):
    """Replace a word while preserving its punctuation"""
    punctuation = ""
    clean_word = word

    # Extract trailing punctuation
    while clean_word and clean_word[-1] in ".,!?;:":
        punctuation = clean_word[-1] + punctuation
        clean_word = clean_word[:-1]

    if not replacement:
        return punctuation.lstrip() if punctuation else ""

    return replacement + punctuation


def replace_individual_words(text):
    """Replace toxic words with neutral alternatives"""
    words = text.split()
    new_words = []

    for word in words:
        clean = word.lower().strip(".,!?;:")

        # Check each replacement dictionary
        if clean in PROFANITY_REPLACEMENTS:
            new_words.append(replace_word_with_punctuation(word, PROFANITY_REPLACEMENTS[clean]))
        elif clean in INSULT_REPLACEMENTS:
            new_words.append(replace_word_with_punctuation(word, INSULT_REPLACEMENTS[clean]))
        elif clean in HATE_REPLACEMENTS:
            new_words.append(replace_word_with_punctuation(word, HATE_REPLACEMENTS[clean]))
        else:
            new_words.append(word)

    # Clean up extra spaces
    result = " ".join(new_words)
    result = re.sub(r'\s+', ' ', result).strip()
    return result


def rewrite_text(text, model, tokenizer):
    """
    Attempt to reduce toxicity through word/phrase replacement.
    Returns (rewritten_text, original_score, new_score)
    """
    original_prob = score_toxicity(text, model, tokenizer)

    # If already non-toxic, don't rewrite
    if original_prob < DEFAULT_THRESHOLD * 0.8:
        return None, original_prob, original_prob

    current_text = text
    current_prob = original_prob

    for iteration in range(MAX_REWRITE_PASSES):
        best_candidate = current_text
        best_score = current_prob

        # Strategy 1: Word-level replacement
        candidate = replace_individual_words(current_text)
        if candidate != current_text:
            new_score = score_toxicity(candidate, model, tokenizer)
            if new_score < best_score:
                best_candidate = candidate
                best_score = new_score

        # Strategy 2: Find and replace toxic phrases
        words = current_text.split()
        for i, word in enumerate(words):
            word_lower = word.lower().strip(".,!?;:")

            # Check if this word is toxic
            is_toxic = (word_lower in PROFANITY_WORDS or
                       word_lower in INSULT_WORDS or
                       word_lower in HATE_REPLACEMENTS)

            if is_toxic:
                # Extract phrase around this word
                start = max(0, i - 2)
                end = min(len(words), i + 3)
                toxic_phrase = " ".join(words[start:end])

                # Try replacing with safe phrases
                for replacement in SAFE_PHRASE_REPLACEMENTS:
                    candidate = current_text.replace(toxic_phrase, replacement, 1)
                    candidate = re.sub(r'\s+', ' ', candidate).strip()

                    if candidate and candidate != current_text:
                        new_score = score_toxicity(candidate, model, tokenizer)
                        if new_score < best_score:
                            best_candidate = candidate
                            best_score = new_score
                break

        # Strategy 3: Tone softening (only if still quite toxic)
        if best_score > DEFAULT_THRESHOLD:
            for prefix in SOFTENING_PHRASES:
                candidate = f"{prefix}, {current_text}".capitalize()
                new_score = score_toxicity(candidate, model, tokenizer)
                if new_score < best_score:
                    best_candidate = candidate
                    best_score = new_score

        # Check if we made improvement
        if best_score >= current_prob - MIN_IMPROVEMENT:
            break

        current_text = best_candidate
        current_prob = best_score

        # Stop if we've made it non-toxic
        if current_prob < DEFAULT_THRESHOLD:
            break

    # Only return rewrite if it's actually better
    if current_prob < original_prob - MIN_IMPROVEMENT:
        return current_text, original_prob, current_prob

    return None, original_prob, original_prob


def bias_analysis(model, tokenizer):
    """Run bias and fairness analysis on identity groups"""
    results = []

    for category, identities in IDENTITY_GROUPS.items():
        for identity in identities:
            for template in BIAS_TEMPLATES:
                sentence = template.format(identity)
                score = score_toxicity(sentence, model, tokenizer)

                results.append({
                    "Category": category,
                    "Identity": identity,
                    "Sentence": sentence,
                    "Toxicity": round(score, 4)
                })

    return results


def get_toxicity_category(text_lower, tokens, scores):
    """Categorize type of toxicity present"""
    categories = []

    if any(word in text_lower for word in PROFANITY_WORDS):
        categories.append("Profanity")

    if any(word in text_lower for word in INSULT_WORDS):
        categories.append("Insults")

    if any(word in text_lower for word in ["hate", "despise", "loathe"]):
        categories.append("Hate Speech")

    words = text_lower.split()
    if len(words) > 0 and words[0] in COMMAND_WORDS:
        categories.append("Threats/Commands")

    if "you" in text_lower and any(word in text_lower for word in INSULT_WORDS | PROFANITY_WORDS):
        categories.append("Personal Attack")

    return categories if categories else ["Other"]
