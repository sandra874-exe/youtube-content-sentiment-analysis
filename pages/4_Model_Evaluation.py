import sys
from pathlib import Path

import streamlit as st
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.evaluation import load_labelled_dataset, run_evaluation
from src.sentiment_models import load_transformer
from src.charts import (
    model_metrics,
    model_f1,
    cross_validation_chart,
    confusion_matrix_figure,
)

st.set_page_config(page_title="Model Evaluation", page_icon="🧠", layout="wide")

st.markdown(
    """
    <div style="
        padding:35px 40px;
        border-radius:26px;
        background:linear-gradient(135deg,#eee5ff,#e7f8f0);
        border:1px solid #eadff4;
        margin-bottom:28px;
    ">
        <div style="font-size:12px;font-weight:800;letter-spacing:1.5px;color:#82758d;">
            SUPERVISED MACHINE LEARNING
        </div>
        <div style="font-size:38px;font-weight:800;color:#292638;margin-top:8px;">
            🧠 Model Evaluation
        </div>
        <div style="font-size:15px;line-height:1.7;color:#706a77;max-width:950px;margin-top:10px;">
            The internal human-labelled comments are used as ground truth.
            TF-IDF + Logistic Regression and TF-IDF + Linear SVM are trained on
            the training split, while VADER and pretrained RoBERTa are evaluated
            on the same holdout test set.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

try:
    preview_df, dataset_path = load_labelled_dataset()
except Exception as error:
    st.error(str(error))
    st.stop()

st.markdown("### Internal evaluation dataset")

c1, c2, c3 = st.columns(3)
c1.metric("Labelled comments", f"{len(preview_df):,}")
c2.metric("Positive", int((preview_df["human-sentiment"] == "positive").sum()))
c3.metric("Neutral / Negative", int((preview_df["human-sentiment"].isin(["neutral", "negative"])).sum()))

st.caption(f"Dataset used: `{dataset_path.name}`")

label_counts = (
    preview_df["human-sentiment"]
    .value_counts()
    .reindex(["positive", "neutral", "negative"], fill_value=0)
    .reset_index()
)
label_counts.columns = ["sentiment", "comments"]

import plotly.express as px
fig = px.bar(
    label_counts,
    x="sentiment",
    y="comments",
    text="comments",
    title="Human-labelled class distribution",
)
fig.update_traces(textposition="outside")
fig.update_layout(template="plotly_white", height=400)
st.plotly_chart(fig, use_container_width=True)


if st.button("Train and evaluate all models", type="primary", use_container_width=True):
    try:
        with st.spinner("Loading pretrained RoBERTa..."):
            roberta = load_transformer()

        with st.spinner("Training and evaluating Logistic Regression, SVM, VADER and RoBERTa..."):
            evaluation = run_evaluation(roberta)

        st.session_state["evaluation"] = evaluation
        st.success("Evaluation completed.")
    except Exception as error:
        st.error(f"Evaluation failed: {error}")

if "evaluation" not in st.session_state:
    st.info("Run the evaluation above to generate the model-comparison results.")
    st.stop()

evaluation = st.session_state["evaluation"]
results_df = evaluation["results"]

st.markdown("---")
st.markdown("## Holdout-test performance")
st.caption(
    f"Training comments: {evaluation['train_size']} • "
    f"Test comments: {evaluation['test_size']} • "
    "All models are scored against the same human-labelled test subset."
)

st.dataframe(
    results_df.round(3),
    use_container_width=True,
    hide_index=True,
)

st.plotly_chart(model_metrics(results_df), use_container_width=True)
st.plotly_chart(model_f1(results_df), use_container_width=True)

cv_fig = cross_validation_chart(results_df)
if cv_fig is not None:
    st.plotly_chart(cv_fig, use_container_width=True)

st.markdown("## Confusion matrices")
for model_name, detail in evaluation["details"].items():
    st.plotly_chart(
        confusion_matrix_figure(
            detail["confusion_matrix"],
            model_name,
        ),
        use_container_width=True,
    )

st.markdown("## Per-class report")
selected_model = st.selectbox(
    "Model",
    list(evaluation["details"].keys()),
)
report = pd.DataFrame(evaluation["details"][selected_model]["report"]).T
st.dataframe(report.round(3), use_container_width=True)

st.markdown("## Methodology")
st.markdown(
    """
    **TF-IDF** converts comments into numerical features using term frequency and
    inverse document frequency. Unigrams and bigrams are used.

    **Logistic Regression** learns class weights from the TF-IDF feature matrix.

    **Linear SVM** learns a separating decision boundary from the same TF-IDF representation.

    **VADER** is a lexicon and rule-based baseline and is not trained on the labelled dataset.

    **RoBERTa** is a pretrained transformer model and is used for inference in this version.

    **Important:** VADER and RoBERTa are evaluated on the same held-out comments as the
    supervised models, while the supervised models themselves only fit on the training split.
    """
)
