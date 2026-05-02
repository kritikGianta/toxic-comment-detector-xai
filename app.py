import os

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import torch
from captum.attr import IntegratedGradients
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from config import DEFAULT_THRESHOLD, DEVICE, EXAMPLE_TEXTS, MODEL_PATH
from utils import (
    batch_score_toxicity,
    bias_analysis,
    counterfactual_analysis,
    explain_text,
    explain_why_toxic,
    get_severity,
    get_toxicity_category,
    normalize_text,
    predict,
    rewrite_text,
)

st.set_page_config(
    page_title="Comment Safety Review",
    page_icon="C",
    layout="wide",
    initial_sidebar_state="expanded",
)


def load_css():
    css_path = os.path.join(os.path.dirname(__file__), "styles.css")
    if os.path.exists(css_path):
        with open(css_path, encoding="utf-8") as css_file:
            st.markdown(f"<style>{css_file.read()}</style>", unsafe_allow_html=True)


load_css()

if "analysis_history" not in st.session_state:
    st.session_state.analysis_history = []
if "selected_example" not in st.session_state:
    st.session_state.selected_example = "Custom"
if "main_text_input" not in st.session_state:
    st.session_state.main_text_input = ""


@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        st.error(f"Model not found at {MODEL_PATH}. Train or export the model first.")
        st.stop()

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_PATH,
        local_files_only=True,
        torch_dtype=torch.float32,
    )
    model.to(DEVICE)
    model.eval()
    return tokenizer, model


def forward_func(inputs_embeds, attention_mask):
    outputs = model(inputs_embeds=inputs_embeds, attention_mask=attention_mask)
    return outputs.logits.squeeze(-1)


tokenizer, model = load_model()
ig = IntegratedGradients(forward_func)


def render_token_heatmap(tokens, scores):
    html = ['<div class="token-heatmap">']
    for token, score in zip(tokens, scores):
        if token in {"[CLS]", "[SEP]", "[PAD]"}:
            continue

        clean_token = token.replace("##", "")
        strength = min(abs(float(score)), 1.0)
        if score >= 0:
            color = f"rgba(185, 28, 28, {0.14 + strength * 0.34:.2f})"
            border = "rgba(185, 28, 28, 0.24)"
            tip = f"Raises score by {score:.3f}"
        else:
            color = f"rgba(22, 101, 52, {0.12 + strength * 0.28:.2f})"
            border = "rgba(22, 101, 52, 0.22)"
            tip = f"Lowers score by {score:.3f}"

        html.append(
            f'<span class="token" style="background:{color}; border-color:{border};" title="{tip}">{clean_token}</span>'
        )
    html.append("</div>")
    return "".join(html)


def render_score_gauge(probability, threshold):
    severity, color = get_severity(probability)
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=probability * 100,
            number={"suffix": "%", "font": {"size": 42}},
            title={"text": f"Risk score<br><span style='font-size:16px'>{severity}</span>"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": color},
                "bgcolor": "#f6efe5",
                "borderwidth": 0,
                "steps": [
                    {"range": [0, 30], "color": "#dbead8"},
                    {"range": [30, 50], "color": "#efe6b8"},
                    {"range": [50, 70], "color": "#f6d3ad"},
                    {"range": [70, 100], "color": "#f2c3c1"},
                ],
                "threshold": {
                    "line": {"color": "#2e2a26", "width": 3},
                    "thickness": 0.7,
                    "value": threshold * 100,
                },
            },
        )
    )
    fig.update_layout(
        height=280,
        margin=dict(t=50, b=10, l=10, r=10),
        paper_bgcolor="rgba(0,0,0,0)",
        font={"family": "Aptos, Segoe UI, sans-serif", "color": "#2e2a26"},
    )
    return fig


def render_summary_card(title, value, helper, tone_class="neutral"):
    st.markdown(
        f"""
        <div class="summary-card {tone_class}">
            <div class="summary-label">{title}</div>
            <div class="summary-value">{value}</div>
            <div class="summary-helper">{helper}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def get_recommendation(probability, threshold):
    if probability >= max(threshold + 0.15, 0.8):
        return "Block or rewrite before posting"
    if probability >= threshold:
        return "Needs review before posting"
    if probability >= threshold * 0.7:
        return "Borderline tone, worth a second look"
    return "Safe to post as written"


def detect_comment_column(dataframe):
    candidates = ["comment_text", "text", "comment", "message", "content"]
    lower_map = {column.lower(): column for column in dataframe.columns}
    for candidate in candidates:
        if candidate in lower_map:
            return lower_map[candidate]
    for column in dataframe.columns:
        if dataframe[column].dtype == object:
            return column
    return None


def analyze_single_text(text, threshold):
    probability, label = predict(text, model, tokenizer, threshold)
    tokens, scores = explain_text(text, model, tokenizer, ig)
    explanation = explain_why_toxic(text, tokens, scores, threshold, probability)
    categories = get_toxicity_category(text.lower(), tokens, scores)
    rewrite, old_score, new_score, rationale = rewrite_text(text, model, tokenizer, tokens, scores)
    counterfactuals = counterfactual_analysis(text, tokens, scores, model, tokenizer)
    return {
        "probability": probability,
        "label": label,
        "tokens": tokens,
        "scores": scores,
        "explanation": explanation,
        "categories": categories,
        "rewrite": rewrite,
        "old_score": old_score,
        "new_score": new_score,
        "rewrite_rationale": rationale,
        "counterfactuals": counterfactuals,
    }


with st.sidebar:
    st.markdown("### Review settings")
    threshold = st.slider(
        "Decision threshold",
        min_value=0.10,
        max_value=0.90,
        value=DEFAULT_THRESHOLD,
        step=0.01,
        help="Scores above this line are treated as toxic.",
    )
    st.caption("Use a lower threshold for stricter moderation and a higher one for fewer false alarms.")

    st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
    st.markdown("### Session")
    total_checks = len(st.session_state.analysis_history)
    toxic_checks = sum(1 for item in st.session_state.analysis_history if item["label"] == "Toxic")
    st.metric("Comments reviewed", total_checks)
    st.metric("Flagged", toxic_checks)
    if st.button("Clear session history", use_container_width=True):
        st.session_state.analysis_history = []
        st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
    st.markdown("### Notes")
    st.markdown(
        "- Uses the trained local model already in this project\n"
        "- Runs fully offline from the exported model folder\n"
        "- Focuses on moderation workflow instead of AI branding"
    )
    st.markdown("</div>", unsafe_allow_html=True)


st.markdown(
    """
    <section class="hero-shell">
        <div class="hero-copy">
            <div class="eyebrow">Comment Safety Review</div>
            <h1>Review language the way a moderator would.</h1>
            <p>
                Check whether a comment crosses the line, see which words drive the score,
                and get a calmer rewrite when the tone is too sharp.
            </p>
        </div>
        <div class="hero-panel">
            <div class="hero-panel-label">What this app is best for</div>
            <ul>
                <li>Single-comment review before posting</li>
                <li>CSV screening for moderation queues</li>
                <li>Quick fairness checks on neutral identity statements</li>
            </ul>
        </div>
    </section>
    """,
    unsafe_allow_html=True,
)

tabs = st.tabs(["Review comment", "Scan CSV", "Fairness check", "Recent checks"])

with tabs[0]:
    sample_names = ["Custom"] + list(EXAMPLE_TEXTS.keys())
    selected_example = st.selectbox("Start from a sample or write your own", sample_names)
    if selected_example != "Custom":
        st.session_state.main_text_input = EXAMPLE_TEXTS[selected_example]

    st.text_area(
        "Comment",
        key="main_text_input",
        height=180,
        placeholder="Paste a comment here to review tone, risk, and rewrite suggestions.",
    )
    current_text = normalize_text(st.session_state.main_text_input)

    info_col, action_col = st.columns([3, 1])
    with info_col:
        st.caption(f"{len(current_text)} characters | {len(current_text.split()) if current_text else 0} words")
    with action_col:
        run_single = st.button("Review comment", type="primary", use_container_width=True)

    if run_single:
        if not current_text:
            st.warning("Enter a comment before running a review.")
        else:
            with st.spinner("Reviewing comment..."):
                result = analyze_single_text(current_text, threshold)

            st.session_state.analysis_history.append(
                {
                    "text": current_text,
                    "prob": result["probability"],
                    "label": result["label"],
                    "timestamp": pd.Timestamp.now(),
                }
            )

            severity, _ = get_severity(result["probability"])
            recommendation = get_recommendation(result["probability"], threshold)
            tone_class = "danger" if result["label"] == "Toxic" else "safe"

            top_left, top_right = st.columns([1.2, 1])
            with top_left:
                st.plotly_chart(render_score_gauge(result["probability"], threshold), use_container_width=True)
            with top_right:
                st.markdown(
                    f"""
                    <div class="decision-panel {tone_class}">
                        <div class="decision-label">Decision</div>
                        <div class="decision-value">{result["label"]}</div>
                        <div class="decision-subtext">Severity: {severity}</div>
                        <div class="decision-subtext">Recommendation: {recommendation}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            card_one, card_two, card_three = st.columns(3)
            with card_one:
                render_summary_card("Risk score", f"{result['probability']:.1%}", "Model confidence", tone_class)
            with card_two:
                render_summary_card("Threshold", f"{threshold:.0%}", "Current moderation line")
            with card_three:
                render_summary_card("Categories", ", ".join(result["categories"]), "Detected language pattern")

            insight_col, rewrite_col = st.columns(2)
            with insight_col:
                st.markdown("### Why it was flagged")
                st.markdown(
                    f"""
                    <div class="content-card">
                        <p>{result["explanation"]}</p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                st.markdown("### Word-level impact")
                st.caption("Warmer red tokens push the score up. Green tokens pull it down.")
                st.markdown(render_token_heatmap(result["tokens"], result["scores"]), unsafe_allow_html=True)

            with rewrite_col:
                st.markdown("### Suggested rewrite")
                if result["rewrite"]:
                    score_drop = result["old_score"] - result["new_score"]
                    st.markdown(
                        f"""
                        <div class="content-card">
                            <div class="rewrite-text">{result["rewrite"]}</div>
                            <p>{result["rewrite_rationale"]}</p>
                            <div class="rewrite-metrics">
                                <span>Original: {result["old_score"]:.3f}</span>
                                <span>Rewritten: {result["new_score"]:.3f}</span>
                                <span>Drop: {score_drop:.3f}</span>
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                else:
                    st.markdown(
                        """
                        <div class="content-card">
                            <p>No stronger rewrite was found. The original wording is already close to the safe range or the model did not improve on simple edits.</p>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                st.markdown("### Highest-impact removals")
                if result["counterfactuals"]:
                    for item in result["counterfactuals"]:
                        st.markdown(
                            f"""
                            <div class="mini-row">
                                <span class="pill">{item["word"]}</span>
                                <span>Score change: {item["impact"]:.3f}</span>
                                <span>New score: {item["new_prob"]:.3f}</span>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )
                else:
                    st.caption("No single token stood out enough to produce a meaningful drop on its own.")

with tabs[1]:
    st.markdown("### CSV screening")
    st.caption("Upload a CSV. The app will try to find a text column automatically and score comments in batches.")

    uploaded_file = st.file_uploader("Upload CSV", type="csv")
    if uploaded_file is not None:
        try:
            dataframe = pd.read_csv(uploaded_file)
            comment_column = detect_comment_column(dataframe)

            if comment_column is None:
                st.error("No text-like column was found. Add a column such as comment_text, comment, or text.")
            else:
                st.markdown(
                    f"""
                    <div class="content-card">
                        <p><strong>Detected text column:</strong> {comment_column}</p>
                        <p><strong>Rows loaded:</strong> {len(dataframe)}</p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                if st.button("Run CSV screening", type="primary"):
                    texts = [normalize_text(value) for value in dataframe[comment_column].tolist()]
                    with st.spinner("Scoring comments in batches..."):
                        scores = batch_score_toxicity(texts, model, tokenizer)

                    labels = ["Toxic" if score >= threshold else "Non-toxic" for score in scores]
                    severities = [get_severity(score)[0] for score in scores]
                    results_df = dataframe.copy()
                    results_df["toxicity_score"] = [round(score, 4) for score in scores]
                    results_df["label"] = labels
                    results_df["severity"] = severities

                    total = len(results_df)
                    toxic_total = sum(1 for label in labels if label == "Toxic")
                    avg_score = sum(scores) / total if total else 0.0

                    first, second, third = st.columns(3)
                    with first:
                        render_summary_card("Rows reviewed", str(total), "CSV rows scored")
                    with second:
                        render_summary_card("Flagged", str(toxic_total), "Above current threshold", "danger" if toxic_total else "neutral")
                    with third:
                        render_summary_card("Average score", f"{avg_score:.3f}", "Across the uploaded file")

                    st.dataframe(results_df, use_container_width=True, height=420)

                    histogram = go.Figure()
                    histogram.add_trace(
                        go.Histogram(
                            x=results_df["toxicity_score"],
                            nbinsx=20,
                            marker_color="#6f7f6b",
                            opacity=0.85,
                        )
                    )
                    histogram.update_layout(
                        height=320,
                        margin=dict(t=20, b=20, l=20, r=20),
                        paper_bgcolor="rgba(0,0,0,0)",
                        plot_bgcolor="#fbf7f1",
                        xaxis_title="Toxicity score",
                        yaxis_title="Count",
                        showlegend=False,
                    )
                    st.plotly_chart(histogram, use_container_width=True)

                    csv_data = results_df.to_csv(index=False).encode("utf-8")
                    st.download_button(
                        "Download scored CSV",
                        data=csv_data,
                        file_name="comment_safety_results.csv",
                        mime="text/csv",
                        use_container_width=True,
                    )

        except Exception as exc:
            st.error(f"Could not process the CSV: {exc}")

with tabs[2]:
    st.markdown("### Fairness check")
    st.caption("This compares neutral identity statements and highlights large score swings.")

    if st.button("Run fairness check", type="primary"):
        with st.spinner("Scoring identity templates..."):
            bias_results = pd.DataFrame(bias_analysis(model, tokenizer))

        grouped = (
            bias_results.groupby("Category")["Toxicity"]
            .agg(["mean", "max", "min"])
            .reset_index()
            .round(4)
        )
        grouped["spread"] = (grouped["max"] - grouped["min"]).round(4)

        st.dataframe(grouped, use_container_width=True, height=260)
        st.dataframe(bias_results, use_container_width=True, height=360)

with tabs[3]:
    st.markdown("### Recent checks")
    if not st.session_state.analysis_history:
        st.caption("No comments reviewed in this session yet.")
    else:
        history_df = pd.DataFrame(st.session_state.analysis_history)
        history_df["timestamp"] = history_df["timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S")
        history_df["prob"] = history_df["prob"].map(lambda value: f"{value:.3f}")
        st.dataframe(
            history_df.rename(
                columns={
                    "text": "comment",
                    "prob": "toxicity_score",
                    "label": "decision",
                    "timestamp": "reviewed_at",
                }
            ),
            use_container_width=True,
            height=420,
        )
