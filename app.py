import streamlit as st
import joblib
import pandas as pd

# -------------------------------------------------
# Page Configuration
# -------------------------------------------------
st.set_page_config(
    page_title="AutoJudge – Problem Difficulty Predictor",
    layout="centered"
)

# -------------------------------------------------
# Load Trained Models
# -------------------------------------------------
@st.cache_resource
def load_models():
    clf = joblib.load("difficulty_classifier.pkl")
    reg = joblib.load("difficulty_regressor.pkl")
    return clf, reg

clf_pipeline, reg_pipeline = load_models()

# -------------------------------------------------
# Feature Engineering Function (MUST MATCH TRAINING)
# -------------------------------------------------
def build_features(description, input_desc, output_desc):
    combined_text = description + " " + input_desc + " " + output_desc
    text_lower = combined_text.lower()

    data = {
        "combined_text": combined_text,
        "text_length": len(combined_text),
        "digit_count": sum(c.isdigit() for c in combined_text),
        "symbol_count": len([c for c in combined_text if c in "+-*/=<>" ]),
        "kw_graph": int("graph" in text_lower),
        "kw_tree": int("tree" in text_lower),
        "kw_dp": int("dp" in text_lower or "dynamic programming" in text_lower),
        "kw_recursion": int("recursion" in text_lower)
    }

    return pd.DataFrame([data])

# -------------------------------------------------
# App UI
# -------------------------------------------------
st.title("AutoJudge")
st.subheader("Predict Programming Problem Difficulty")

st.write(
    """
    Enter a programming problem description and this system will predict:
    - **Difficulty Class** (Easy / Medium / Hard)
    - **Difficulty Score** (Numerical)
    """
)

st.divider()

# -------------------------------------------------
# User Inputs
# -------------------------------------------------
description = st.text_area(
    "Problem Description",
    height=200,
    placeholder="Enter the problem description here..."
)

input_description = st.text_area(
    "Input Description",
    height=120,
    placeholder="Describe the input format..."
)

output_description = st.text_area(
    "Output Description",
    height=120,
    placeholder="Describe the output format..."
)

# -------------------------------------------------
# Prediction
# -------------------------------------------------
if st.button("🔍 Predict Difficulty"):
    if description.strip() == "":
        st.warning("Please enter at least the problem description.")
    else:
        # Build feature dataframe
        X_input = build_features(description, input_description, output_description)

        # Predictions
        pred_class = clf_pipeline.predict(X_input)[0]
        pred_score = reg_pipeline.predict(X_input)[0]

        # -------------------------------------------------
        # Display Results
        # -------------------------------------------------
        st.divider()
        st.subheader("Prediction Results")

        col1, col2 = st.columns(2)

        with col1:
            st.metric(
                label="Difficulty Class",
                value=str(pred_class).capitalize()
            )

        with col2:
            st.metric(
                label="Difficulty Score",
                value=f"{pred_score:.2f}"
            )

        # Interpretation
        if pred_class.lower() == "easy":
            st.success("Suitable for beginners.")
        elif pred_class.lower() == "medium":
            st.info("Requires intermediate problem-solving skills.")
        else:
            st.error("Challenging problem (advanced level).")

# -------------------------------------------------
# Footer
# -------------------------------------------------
st.divider()
st.caption("AutoJudge • ML-based Programming Difficulty Predictor")

