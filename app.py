
import streamlit as st
import torch

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    pipeline
)

# -----------------------------
# Application configuration
# -----------------------------

st.set_page_config(
    page_title="BERT Sentiment Analysis",
    page_icon="🎬",
    layout="centered"
)

MODEL_NAME = "distilbert-base-uncased-finetuned-sst-2-english"


# -----------------------------
# Load model once
# -----------------------------

@st.cache_resource
def load_sentiment_model():

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME
    )

    classifier = pipeline(
        "sentiment-analysis",
        model=model,
        tokenizer=tokenizer,
        device=0 if torch.cuda.is_available() else -1
    )

    return classifier


# -----------------------------
# Application interface
# -----------------------------

st.title("🎬 BERT Sentiment Analysis")

st.write(
    "Analyze movie reviews using a pretrained "
    "BERT-family sentiment classification model."
)

st.info(
    "Enter a movie review below to predict whether "
    "its sentiment is positive or negative."
)

review = st.text_area(
    "Enter your movie review",
    placeholder="Example: This movie was amazing!",
    height=160
)

if st.button("Analyze Sentiment", type="primary"):

    if not review.strip():

        st.warning("Please enter a movie review first.")

    elif len(review.strip()) < 3:

        st.warning("Please enter a longer review.")

    else:

        try:
            with st.spinner("Analyzing your review..."):

                classifier = load_sentiment_model()

                result = classifier(
                    review.strip(),
                    truncation=True,
                    max_length=512
                )[0]

            raw_label = result["label"].upper()
            confidence = float(result["score"])

            # This pretrained model uses POSITIVE/NEGATIVE labels.
            if raw_label not in {"POSITIVE", "NEGATIVE"}:
                st.error(
                    f"Unexpected model label: {raw_label}. "
                    "Verify the model's label configuration."
                )
            else:
                st.subheader("Prediction Result")

                if raw_label == "POSITIVE":
                    st.success("😊 Sentiment: Positive")
                else:
                    st.error("😞 Sentiment: Negative")

                st.metric(
                    "Model Confidence",
                    f"{confidence * 100:.2f}%"
                )

                st.progress(confidence)

                st.caption(
                    "Confidence represents the model's score "
                    "for its selected class, not a guarantee "
                    "that the prediction is correct."
                )

        except Exception as error:
            st.error(
                "Unable to analyze the review. "
                "Please try again."
            )

            st.exception(error)


st.divider()

st.caption(
    "BERT Sentiment Analysis Project | "
    "Powered by Hugging Face Transformers and Streamlit"
)
# Fallback app
