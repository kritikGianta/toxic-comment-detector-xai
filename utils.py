"""Utility functions for toxic comment detection and analysis."""

import re

import numpy as np
import torch

from config import *

SPECIAL_TOKENS = {"[CLS]", "[SEP]", "[PAD]"}


def score_toxicity(text, model, tokenizer):
    """Calculate toxicity probability for given text"""
    cleaned_text = normalize_text(text)
    if not cleaned_text:
        return 0.0

    inputs = tokenizer(
        cleaned_text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=MAX_LENGTH
    )

    # Remove token_type_ids for DistilBERT (it doesn't use them)
    inputs.pop("token_type_ids", None)
    inputs = {key: value.to(DEVICE) for key, value in inputs.items()}

    with torch.no_grad():
        logits = model(**inputs).logits
        prob = torch.sigmoid(logits)[0][0].item()

    return prob


def batch_score_toxicity(texts, model, tokenizer, batch_size=32):
    """Score many texts in batches to keep batch analysis responsive."""
    if not texts:
        return []

    normalized = [normalize_text(text) for text in texts]
    scores = []

    for start in range(0, len(normalized), batch_size):
        batch = normalized[start:start + batch_size]
        empty_positions = [index for index, text in enumerate(batch) if not text]
        safe_batch = [text if text else " " for text in batch]

        inputs = tokenizer(
            safe_batch,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=MAX_LENGTH
        )
        inputs.pop("token_type_ids", None)
        inputs = {key: value.to(DEVICE) for key, value in inputs.items()}

        with torch.no_grad():
            logits = model(**inputs).logits
            probs = torch.sigmoid(logits).squeeze(-1).detach().cpu().tolist()

        if not isinstance(probs, list):
            probs = [float(probs)]

        for position in empty_positions:
            probs[position] = 0.0

        scores.extend(float(prob) for prob in probs)

    return scores


def predict(text, model, tokenizer, threshold):
    """Predict if text is toxic and return probability"""
    prob = score_toxicity(text, model, tokenizer)
    label = "Toxic" if prob >= threshold else "Non-toxic"
    return prob, label


def explain_text(text, model, tokenizer, ig):
    """Generate token-level attributions using Integrated Gradients"""
    cleaned_text = normalize_text(text)
    if not cleaned_text:
        return [], np.array([])

    encoded = tokenizer(
        cleaned_text,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_LENGTH
    )
    encoded.pop("token_type_ids", None)
    encoded = {key: value.to(DEVICE) for key, value in encoded.items()}

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
        if score > ATTRIBUTION_THRESHOLD and tok not in SPECIAL_TOKENS
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
        if score <= 0 or tok in SPECIAL_TOKENS:
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


def normalize_text(text):
    """Collapse whitespace and convert non-string inputs safely."""
    if text is None:
        return ""

    cleaned = str(text).replace("\r", " ").replace("\n", " ")
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def _match_replacement_style(source, replacement):
    if not replacement:
        return replacement
    if source.isupper():
        return replacement.upper()
    if source.istitle():
        return replacement.capitalize()
    return replacement


def replace_phrase_case_aware(text, phrase, replacement):
    """Replace whole phrases while preserving simple casing."""
    pattern = re.compile(rf"\b{re.escape(phrase)}\b", flags=re.IGNORECASE)

    def repl(match):
        matched_text = match.group(0)
        return _match_replacement_style(matched_text, replacement)

    updated = pattern.sub(repl, text)
    updated = re.sub(r"\s+", " ", updated).strip()
    updated = re.sub(r"\s+([,.!?;:])", r"\1", updated)
    return updated


def _dedupe_words(words):
    seen = set()
    ordered = []
    for word in words:
        key = word.lower()
        if key in seen:
            continue
        seen.add(key)
        ordered.append(word)
    return ordered


def _top_flagged_words(tokens, scores, limit=3):
    flagged = []
    for token, score in sorted(zip(tokens, scores), key=lambda item: item[1], reverse=True):
        if score <= ATTRIBUTION_THRESHOLD or token in SPECIAL_TOKENS:
            continue
        cleaned = token.replace("##", "").strip()
        if len(cleaned) < 2 or not re.search(r"[A-Za-z]", cleaned):
            continue
        flagged.append(cleaned)
    return _dedupe_words(flagged)[:limit]


def _apply_replacement_maps(text):
    updated = text
    replacement_groups = [
        PROFANITY_REPLACEMENTS,
        INSULT_REPLACEMENTS,
        HATE_REPLACEMENTS,
    ]

    for replacement_group in replacement_groups:
        for source, target in replacement_group.items():
            updated = replace_phrase_case_aware(updated, source, target)

    updated = re.sub(r"\s+", " ", updated).strip()
    updated = re.sub(r"\s+([,.!?;:])", r"\1", updated)
    return updated


def _soften_opening(text):
    if not text:
        return text

    first_word = text.split()[0].lower()
    if first_word in COMMAND_WORDS:
        return f"Please {text[0].lower() + text[1:]}"

    if re.match(r"^(you\s+are|you're)\b", text, flags=re.IGNORECASE):
        return re.sub(
            r"^(you\s+are|you're)\b",
            "This comes across as",
            text,
            count=1,
            flags=re.IGNORECASE
        )

    return text


def _cleanup_rewrite(text):
    text = re.sub(r"\s+", " ", text).strip(" ,")
    text = re.sub(r"\s+([,.!?;:])", r"\1", text)
    if text and text[-1] not in ".!?":
        text += "."
    if text:
        text = text[0].upper() + text[1:]
    return text


def build_rewrite_rationale(prob, new_prob, flagged_words):
    """Explain what changed in a concise, human-friendly way."""
    if not flagged_words:
        return "The suggestion softens the tone and removes language the model treats as hostile."

    changed_words = ", ".join(f"'{word}'" for word in flagged_words)
    if new_prob < DEFAULT_THRESHOLD:
        return f"The suggestion replaces or softens {changed_words} and brings the comment back into a safer range."
    return f"The suggestion mainly targets {changed_words}, which are the biggest drivers of the score."


def rewrite_text(text, model, tokenizer, tokens=None, scores=None):
    """Generate a more natural respectful rewrite when the score is high enough."""
    original_text = normalize_text(text)
    original_prob = score_toxicity(original_text, model, tokenizer)

    if not original_text or original_prob < DEFAULT_THRESHOLD * 0.8:
        return None, original_prob, original_prob, None

    candidate_pool = []

    direct_replacement = _cleanup_rewrite(_apply_replacement_maps(original_text))
    if direct_replacement and direct_replacement != _cleanup_rewrite(original_text):
        candidate_pool.append(direct_replacement)

    softened = _cleanup_rewrite(_soften_opening(direct_replacement or original_text))
    if softened and softened != _cleanup_rewrite(original_text):
        candidate_pool.append(softened)

    if tokens is not None and len(tokens) and scores is not None and len(scores):
        highlighted_words = _top_flagged_words(tokens, scores)
        trimmed_candidate = original_text
        for word in highlighted_words:
            trimmed_candidate = replace_phrase_case_aware(trimmed_candidate, word, "")
        trimmed_candidate = _cleanup_rewrite(trimmed_candidate)
        if trimmed_candidate and trimmed_candidate != _cleanup_rewrite(original_text):
            candidate_pool.append(trimmed_candidate)
    else:
        highlighted_words = []

    for phrase in SAFE_PHRASE_REPLACEMENTS:
        phrase_candidate = _cleanup_rewrite(phrase)
        if phrase_candidate:
            candidate_pool.append(phrase_candidate)

    scored_candidates = []
    seen = set()
    for candidate in candidate_pool:
        if not candidate:
            continue
        key = candidate.lower()
        if key in seen:
            continue
        seen.add(key)
        candidate_score = score_toxicity(candidate, model, tokenizer)
        scored_candidates.append((candidate_score, candidate))

    if not scored_candidates:
        return None, original_prob, original_prob, None

    new_prob, best_candidate = min(scored_candidates, key=lambda item: item[0])
    if new_prob >= original_prob - MIN_IMPROVEMENT:
        return None, original_prob, original_prob, None

    rationale = build_rewrite_rationale(original_prob, new_prob, highlighted_words)
    return best_candidate, original_prob, new_prob, rationale


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
