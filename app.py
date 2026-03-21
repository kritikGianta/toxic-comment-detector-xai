import streamlit as st
import torch
import pandas as pd
import plotly.graph_objects as go
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from captum.attr import IntegratedGradients
import os

from config import *
from utils import *

# Page config
st.set_page_config(
    page_title="Toxic Comment Detector + XAI",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load custom CSS
def load_css():
    css_path = os.path.join(os.path.dirname(__file__), "styles.css")
    if os.path.exists(css_path):
        with open(css_path) as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_css()

# Initialize session state
if "analysis_history" not in st.session_state:
    st.session_state.analysis_history = []

if "current_text" not in st.session_state:
    st.session_state.current_text = ""


@st.cache_resource
def load_model():
    """Load model and tokenizer with error handling"""
    if not os.path.exists(MODEL_PATH):
        st.error(f"Model not found at {MODEL_PATH}. Please train the model first using the notebook.")
        st.stop()

    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)
        model = AutoModelForSequenceClassification.from_pretrained(
            MODEL_PATH,
            local_files_only=True,
            torch_dtype=torch.float32
        )
        model.eval()
        model.to(DEVICE)
        return tokenizer, model
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        st.stop()


try:
    tokenizer, model = load_model()

    # Setup Integrated Gradients
    def forward_func(inputs_embeds, attention_mask):
        outputs = model(inputs_embeds=inputs_embeds, attention_mask=attention_mask)
        return outputs.logits.squeeze(-1)

    ig = IntegratedGradients(forward_func)

except Exception as e:
    st.error(f"Failed to initialize model: {str(e)}")
    st.stop()


def render_token_heatmap(tokens, scores):
    """Render token attribution heatmap"""
    html = '<div class="token-heatmap">'

    for tok, score in zip(tokens, scores):
        if tok in ["[CLS]", "[SEP]", "[PAD]"]:
            continue

        # Calculate color intensity
        if score > 0:
            intensity = min(abs(score), 1.0)
            color = f"rgba(255, 50, 50, {intensity:.2f})"
            tooltip = f"Increases toxicity: +{score:.3f}"
        else:
            intensity = min(abs(score), 1.0)
            color = f"rgba(50, 200, 50, {intensity:.2f})"
            tooltip = f"Decreases toxicity: {score:.3f}"

        clean_tok = tok.replace("##", "")
        html += f'<span class="token" style="background:{color};" title="{tooltip}">{clean_tok}</span>'

    html += '</div>'
    return html


def render_confidence_gauge(prob, threshold):
    """Create confidence gauge visualization"""
    severity, color = get_severity(prob)

    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=prob * 100,
            number={"suffix": "%", "font": {"size": 50}},
            title={"text": f"Toxicity Score<br><span style='font-size:20px;color:{color}'>{severity}</span>",
                   "font": {"size": 20}},
            gauge={
                "axis": {"range": [0, 100], "tickwidth": 2},
                "bar": {"color": color, "thickness": 0.75},
                "bgcolor": "white",
                "borderwidth": 2,
                "bordercolor": "#e2e8f0",
                "steps": [
                    {"range": [0, 30], "color": "#c6f6d5"},
                    {"range": [30, 50], "color": "#fef08a"},
                    {"range": [50, 70], "color": "#fed7aa"},
                    {"range": [70, 85], "color": "#fecaca"},
                    {"range": [85, 100], "color": "#fca5a5"},
                ],
                "threshold": {
                    "line": {"color": "black", "width": 3},
                    "thickness": 0.8,
                    "value": threshold * 100,
                },
            },
        )
    )

    fig.update_layout(
        height=350,
        margin=dict(t=50, b=20, l=20, r=20),
        paper_bgcolor="rgba(0,0,0,0)",
        font={"family": "Arial, sans-serif"}
    )

    return fig


# SIDEBAR
with st.sidebar:
    st.markdown("## ⚙️ Settings")

    threshold = st.slider(
        "Decision Threshold",
        min_value=0.1,
        max_value=0.9,
        value=DEFAULT_THRESHOLD,
        step=0.01,
        help="Scores above this threshold are classified as toxic"
    )

    st.markdown("---")

    st.markdown("## 📖 About")
    st.markdown("""
    This tool uses **DistilBERT** with **Explainable AI** to:
    - Detect toxic comments
    - Explain why they're toxic
    - Suggest improvements
    - Analyze bias

    Built with PyTorch, Transformers, and Captum.
    """)

    st.markdown("---")

    st.markdown("## 📊 Quick Stats")
    if st.session_state.analysis_history:
        total = len(st.session_state.analysis_history)
        toxic_count = sum(1 for h in st.session_state.analysis_history if h["label"] == "Toxic")
        st.metric("Analyzed", total)
        st.metric("Toxic", toxic_count)
        st.metric("Non-toxic", total - toxic_count)

        if st.button("Clear History"):
            st.session_state.analysis_history = []
            st.rerun()

# HEADER
st.markdown("""
<div style='text-align: center; padding: 60px 40px; background: #FFF546;
border-radius: 24px; margin-bottom: 40px; border: 2px solid #66640F;
box-shadow: 0 8px 32px rgba(0, 0, 0, 0.08);'>
    <h1 style='color: #000000; margin: 0; font-size: 4rem; font-family: "Source Serif Pro", serif;
    font-weight: 700; letter-spacing: -2px;'>
        🧠 Toxic Comment Detector
    </h1>
    <p style='font-size: 22px; margin: 20px 0 12px 0; color: #66640F; font-weight: 600;
    font-family: "Inter", sans-serif; letter-spacing: -0.5px;'>
        AI-Powered Analysis with Explainable Insights
    </p>
    <p style='font-size: 15px; margin: 0; color: #000000; font-family: "Space Mono", monospace;'>
        DistilBERT • ROC-AUC: 0.986 • F1: 0.84
    </p>
</div>
""", unsafe_allow_html=True)

# FEATURES SHOWCASE
st.markdown("<h2 style='text-align: center; margin: 40px 0 32px 0;'>✨ Key Features</h2>", unsafe_allow_html=True)

feat_col1, feat_col2, feat_col3, feat_col4 = st.columns(4)

with feat_col1:
    st.markdown("""
    <div class='feature-card'>
        <div class='feature-icon'>🎯</div>
        <div class='feature-title'>Smart Detection</div>
        <div class='feature-desc'>
            Advanced AI model with 98.6% accuracy classifies comments into toxic/non-toxic with confidence scores
        </div>
    </div>
    """, unsafe_allow_html=True)

with feat_col2:
    st.markdown("""
    <div class='feature-card'>
        <div class='feature-icon'>🔍</div>
        <div class='feature-title'>Explainable AI</div>
        <div class='feature-desc'>
            Visual heatmaps show which words contribute to toxicity using Integrated Gradients
        </div>
    </div>
    """, unsafe_allow_html=True)

with feat_col3:
    st.markdown("""
    <div class='feature-card'>
        <div class='feature-icon'>✍️</div>
        <div class='feature-title'>Rewrite Assistant</div>
        <div class='feature-desc'>
            Get intelligent suggestions to rephrase toxic comments into respectful alternatives
        </div>
    </div>
    """, unsafe_allow_html=True)

with feat_col4:
    st.markdown("""
    <div class='feature-card'>
        <div class='feature-icon'>⚖️</div>
        <div class='feature-title'>Bias Analysis</div>
        <div class='feature-desc'>
            Test model fairness across different demographics and identity groups
        </div>
    </div>
    """, unsafe_allow_html=True)

# INFO BANNER
st.markdown("""
<div class='info-banner'>
    <div class='info-banner-title'>📌 How It Works</div>
    <div class='info-banner-text'>
        <strong>1. Detection:</strong> AI analyzes text and assigns toxicity score (0-100%) •
        <strong>2. Explanation:</strong> Visual heatmap highlights problematic words •
        <strong>3. Insights:</strong> See why content is toxic with natural language explanations •
        <strong>4. Improvement:</strong> Get suggestions to make comments more respectful
    </div>
</div>
""", unsafe_allow_html=True)

st.markdown("---")


# MAIN TABS
tab1, tab2, tab3, tab4 = st.tabs(["📝 Single Analysis", "📊 Batch Analysis", "⚖️ Bias Analysis", "📚 History"])

# TAB 1: Single Analysis
with tab1:
    text_input = st.text_area(
        "Enter comment to analyze:",
        value=st.session_state.current_text,
        height=150,
        placeholder="Type or paste a comment here...",
        key="main_text_input"
    )

    # Character count
    if text_input:
        st.caption(f"📏 {len(text_input)} characters, {len(text_input.split())} words")

    if st.button("🔍 Analyze", type="primary", use_container_width=True):
        if not text_input or not text_input.strip():
            st.warning("⚠️ Please enter some text to analyze")
        else:
            with st.spinner("Analyzing comment..."):
                try:
                    # Get prediction
                    prob, label = predict(text_input, model, tokenizer, threshold)

                    # Get explanations
                    tokens, scores = explain_text(text_input, model, tokenizer, ig)

                    # Store in history
                    st.session_state.analysis_history.append({
                        "text": text_input,
                        "prob": prob,
                        "label": label,
                        "timestamp": pd.Timestamp.now()
                    })

                    # Display results in organized sections
                    st.markdown("---")
                    st.markdown("## 🎯 Analysis Results")

                    # Main prediction section
                    col1, col2 = st.columns([1, 1])

                    with col1:
                        st.plotly_chart(
                            render_confidence_gauge(prob, threshold),
                            use_container_width=True
                        )

                    with col2:
                        st.markdown("### 🏷️ Classification")
                        severity, color = get_severity(prob)

                        st.markdown(f"""
                        <div style='padding: 20px; background: {color}15; border-left: 4px solid {color}; border-radius: 10px; margin: 20px 0;'>
                            <h2 style='color: {color}; margin: 0;'>{label}</h2>
                            <p style='font-size: 24px; margin: 10px 0 0 0;'>Severity: <strong>{severity}</strong></p>
                            <p style='font-size: 18px; margin: 5px 0 0 0;'>Confidence: <strong>{prob:.1%}</strong></p>
                        </div>
                        """, unsafe_allow_html=True)

                        # Show categories
                        categories = get_toxicity_category(text_input.lower(), tokens, scores)
                        st.markdown("**Categories:**")
                        for cat in categories:
                            st.markdown(f"- {cat}")

                    # Explanation section
                    st.markdown("### 💭 Why is this toxic?")
                    reasoning = explain_why_toxic(text_input, tokens, scores, threshold, prob)
                    st.info(reasoning)

                    # Token heatmap
                    st.markdown("### 🔍 Word-Level Attribution")
                    st.markdown("**Red** = increases toxicity | **Green** = decreases toxicity")
                    heatmap_html = render_token_heatmap(tokens, scores)
                    st.markdown(heatmap_html, unsafe_allow_html=True)

                    # Counterfactual analysis
                    st.markdown("### 🧪 Counterfactual Analysis")
                    st.markdown("What if we removed key words?")

                    cf_results = counterfactual_analysis(text_input, tokens, scores, model, tokenizer)

                    if not cf_results:
                        st.info("No significant single-word removals found.")
                    else:
                        for i, result in enumerate(cf_results, 1):
                            col1, col2, col3 = st.columns([1, 1, 1])
                            with col1:
                                st.markdown(f"**#{i} Word:** `{result['word']}`")
                            with col2:
                                st.markdown(f"**Impact:** {result['impact']:.3f}")
                            with col3:
                                st.markdown(f"**New Score:** {result['new_prob']:.3f}")

                    # Rewrite suggestions
                    st.markdown("### ✍️ Suggested Rewrite")

                    with st.spinner("Generating suggestions..."):
                        rewrite, old_score, new_score = rewrite_text(text_input, model, tokenizer)

                    if rewrite is None or rewrite == text_input:
                        st.info("✅ No rewrite needed or no better alternative found.")
                    else:
                        improvement = ((old_score - new_score) / old_score) * 100

                        st.success(f"**Suggested version:**\n\n{rewrite}")

                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Original", f"{old_score:.3f}")
                        with col2:
                            st.metric("Improved", f"{new_score:.3f}")
                        with col3:
                            st.metric("Improvement", f"{improvement:.1f}%")

                        # Show side-by-side comparison
                        with st.expander("📋 View Comparison"):
                            comp_col1, comp_col2 = st.columns(2)
                            with comp_col1:
                                st.markdown("**Original:**")
                                st.code(text_input)
                            with comp_col2:
                                st.markdown("**Rewritten:**")
                                st.code(rewrite)

                except Exception as e:
                    st.error(f"❌ Error during analysis: {str(e)}")

# TAB 2: Batch Analysis
with tab2:
    st.markdown("## 📊 Batch Analysis")
    st.markdown("Upload a CSV file with a column named `comment_text` to analyze multiple comments at once.")

    uploaded_file = st.file_uploader(
        "Choose a CSV file",
        type="csv",
        help="CSV should have a 'comment_text' column"
    )

    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)

            if "comment_text" not in df.columns:
                st.error("❌ CSV must contain a 'comment_text' column")
            else:
                st.success(f"✅ Loaded {len(df)} comments")

                if st.button("🚀 Analyze All", type="primary"):
                    progress_bar = st.progress(0)
                    status_text = st.empty()

                    results = []

                    for idx, row in df.iterrows():
                        status_text.text(f"Analyzing comment {idx + 1}/{len(df)}...")
                        progress_bar.progress((idx + 1) / len(df))

                        text = row["comment_text"]

                        if pd.isna(text) or not str(text).strip():
                            results.append({
                                "comment_text": text,
                                "toxicity_score": 0.0,
                                "label": "Empty",
                                "severity": "N/A"
                            })
                            continue

                        prob, label = predict(str(text), model, tokenizer, threshold)
                        severity, _ = get_severity(prob)

                        results.append({
                            "comment_text": text,
                            "toxicity_score": round(prob, 4),
                            "label": label,
                            "severity": severity
                        })

                    progress_bar.empty()
                    status_text.empty()

                    results_df = pd.DataFrame(results)

                    st.markdown("### 📈 Results")

                    # Summary stats
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Total", len(results_df))
                    with col2:
                        toxic_count = len(results_df[results_df["label"] == "Toxic"])
                        st.metric("Toxic", toxic_count)
                    with col3:
                        safe_count = len(results_df[results_df["label"] == "Non-toxic"])
                        st.metric("Non-toxic", safe_count)
                    with col4:
                        avg_score = results_df["toxicity_score"].mean()
                        st.metric("Avg Score", f"{avg_score:.3f}")

                    # Show results table
                    st.dataframe(results_df, use_container_width=True, height=400)

                    # Download button
                    csv = results_df.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="📥 Download Results",
                        data=csv,
                        file_name="toxicity_analysis_results.csv",
                        mime="text/csv",
                        use_container_width=True
                    )

                    # Distribution chart
                    st.markdown("### 📊 Score Distribution")

                    fig = go.Figure()
                    fig.add_trace(go.Histogram(
                        x=results_df["toxicity_score"],
                        nbinsx=20,
                        marker_color="#667eea",
                        opacity=0.75
                    ))

                    fig.update_layout(
                        title="Toxicity Score Distribution",
                        xaxis_title="Toxicity Score",
                        yaxis_title="Count",
                        showlegend=False,
                        height=400
                    )

                    st.plotly_chart(fig, use_container_width=True)

        except Exception as e:
            st.error(f"❌ Error processing file: {str(e)}")

# TAB 3: Bias Analysis
with tab3:
    st.markdown("## ⚖️ Bias & Fairness Analysis")
    st.markdown("""
    This test evaluates whether the model assigns different toxicity scores to neutral statements
    about different identity groups. Large variations may indicate bias.
    """)

    if st.button("🧪 Run Bias Analysis", type="primary"):
        with st.spinner("Running fairness evaluation..."):
            try:
                results = bias_analysis(model, tokenizer)
                df = pd.DataFrame(results)

                st.markdown("### 📊 Results")

                # Summary by category
                summary = df.groupby("Category")["Toxicity"].agg(["mean", "std", "min", "max"]).reset_index()
                summary.columns = ["Category", "Mean", "Std Dev", "Min", "Max"]

                st.dataframe(summary, use_container_width=True)

                # Full results
                with st.expander("📋 View All Results"):
                    st.dataframe(df, use_container_width=True, height=400)

                # Highlight high scores
                high_scores = df[df["Toxicity"] > threshold]
                if not high_scores.empty:
                    st.warning(f"⚠️ {len(high_scores)} neutral statements scored above threshold:")
                    st.dataframe(high_scores, use_container_width=True)
                else:
                    st.success("✅ All neutral statements scored below threshold")

                # Download option
                csv = df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Download Bias Analysis",
                    data=csv,
                    file_name="bias_analysis_results.csv",
                    mime="text/csv"
                )

            except Exception as e:
                st.error(f"❌ Error during bias analysis: {str(e)}")

# TAB 4: History
with tab4:
    st.markdown("## 📚 Analysis History")

    if not st.session_state.analysis_history:
        st.info("No analyses yet. Start by analyzing some comments in the Single Analysis tab!")
    else:
        # Convert history to dataframe
        history_df = pd.DataFrame(st.session_state.analysis_history)

        # Summary metrics
        st.markdown("### 📈 Session Summary")
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Total Analyzed", len(history_df))
        with col2:
            toxic = len(history_df[history_df["label"] == "Toxic"])
            st.metric("Toxic", toxic)
        with col3:
            safe = len(history_df[history_df["label"] == "Non-toxic"])
            st.metric("Non-toxic", safe)
        with col4:
            avg = history_df["prob"].mean()
            st.metric("Avg Score", f"{avg:.3f}")

        # History table
        st.markdown("### 📝 Recent Analyses")

        display_df = history_df[["text", "prob", "label"]].copy()
        display_df.columns = ["Comment", "Score", "Label"]
        display_df["Score"] = display_df["Score"].round(3)

        st.dataframe(display_df, use_container_width=True, height=400)

        # Trend chart
        if len(history_df) > 1:
            st.markdown("### 📉 Toxicity Trend")

            fig = go.Figure()
            fig.add_trace(go.Scatter(
                y=history_df["prob"],
                mode='lines+markers',
                name='Toxicity Score',
                line=dict(color='#667eea', width=3),
                marker=dict(size=8)
            ))

            fig.add_hline(
                y=threshold,
                line_dash="dash",
                line_color="red",
                annotation_text="Threshold"
            )

            fig.update_layout(
                title="Toxicity Over Time",
                xaxis_title="Analysis #",
                yaxis_title="Toxicity Score",
                height=400,
                showlegend=False
            )

            st.plotly_chart(fig, use_container_width=True)

        # Export options
        col1, col2 = st.columns(2)
        with col1:
            csv = history_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download History (CSV)",
                data=csv,
                file_name="analysis_history.csv",
                mime="text/csv",
                use_container_width=True
            )

        with col2:
            if st.button("🗑️ Clear History", use_container_width=True):
                st.session_state.analysis_history = []
                st.rerun()

# FOOTER
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #718096; padding: 20px;'>
    <p>Built with DistilBERT, PyTorch, Transformers & Captum</p>
    <p>🧠 Explainable AI for Responsible Content Moderation</p>
</div>
""", unsafe_allow_html=True)
