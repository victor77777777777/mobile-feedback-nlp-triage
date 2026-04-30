import requests
import streamlit as st


API_URL = "http://127.0.0.1:8000/predict"


st.set_page_config(
    page_title="Mobile Feedback NLP Triage",
    page_icon="📱",
    layout="centered",
)


st.title("📱 Mobile Feedback NLP Triage")
st.write(
    "This demo predicts the sentiment and issue category of a mobile phone review."
)

st.markdown("---")

example_reviews = {
    "Battery issue": "The phone battery dies very quickly and the charging port is loose.",
    "Performance issue": "This phone is very slow and freezes all the time.",
    "Positive review": "Great phone for the price. It works well and arrived quickly.",
    "Screen issue": "The screen cracked after one day and the touch display stopped working.",
    "Network issue": "The phone could not connect to the carrier network and the SIM card was not recognized.",
}

selected_example = st.selectbox(
    "Choose an example review:",
    options=[""] + list(example_reviews.keys()),
)

default_text = example_reviews.get(selected_example, "")

review_text = st.text_area(
    "Enter a mobile phone review:",
    value=default_text,
    height=160,
)

if st.button("Predict"):
    if not review_text.strip():
        st.warning("Please enter a review text.")
    else:
        try:
            response = requests.post(
                API_URL,
                json={"review_text": review_text},
                timeout=10,
            )

            if response.status_code == 200:
                result = response.json()

                st.success("Prediction completed!")

                col1, col2 = st.columns(2)

                with col1:
                    st.metric(
                        label="Sentiment",
                        value=result["sentiment_label"],
                    )

                with col2:
                    st.metric(
                        label="Issue Category",
                        value=result["issue_label"],
                    )

                st.markdown("### Full API Response")
                st.json(result)

            else:
                st.error(f"API error: {response.status_code}")
                st.text(response.text)

        except requests.exceptions.ConnectionError:
            st.error(
                "Could not connect to the FastAPI backend. "
                "Please make sure the backend is running at http://127.0.0.1:8000."
            )

        except requests.exceptions.Timeout:
            st.error("The API request timed out. Please try again.")

        except Exception as e:
            st.error(f"Unexpected error: {e}")


st.markdown("---")
st.caption(
    "Model: TF-IDF + Logistic Regression baseline. "
    "Issue labels are generated using rule-based keyword matching."
)