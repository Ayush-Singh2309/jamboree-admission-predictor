import streamlit as st
from frontend.utils import predict, get_model_info


st.set_page_config(
    page_title="Jamboree Admission Predictor",
    page_icon="🎓",
    layout="centered"
)


try:
    model_info = get_model_info()
except Exception as e:
    model_info = None
    st.error(f"Unable to connect to API\n\n{e}")


st.title("🎓 Jamboree Admission Predictor")

st.write(
    """
Predict your probability of graduate admission based on
your academic profile.
"""
)

st.divider()


gre = st.number_input(
    "GRE Score",
    min_value=260,
    max_value=340,
    value=320,
    step=1,
)

toefl = st.number_input(
    "TOEFL Score",
    min_value=0,
    max_value=120,
    value=110,
    step=1,
)

university_rating = st.slider(
    "University Rating",
    min_value=1,
    max_value=5,
    value=4,
)

sop = st.slider(
    "Statement of Purpose (SOP)",
    min_value=1.0,
    max_value=5.0,
    value=4.0,
    step=0.5,
)

lor = st.slider(
    "Letter of Recommendation (LOR)",
    min_value=1.0,
    max_value=5.0,
    value=4.0,
    step=0.5,
)

cgpa = st.number_input(
    "CGPA",
    min_value=0.0,
    max_value=10.0,
    value=9.0,
    step=0.01,
)

research = st.checkbox("Research Experience")

st.divider()

if st.button("Predict Admission Chance", use_container_width=True):

    payload = {
        "GRE_Score": gre,
        "TOEFL_Score": toefl,
        "University_Rating": university_rating,
        "SOP": sop,
        "LOR": lor,
        "CGPA": cgpa,
        "Research": int(research),
    }

    with st.spinner("Please wait while the API becomes live..."):
        try:
            result = predict(payload)
            probability = result["chance_of_admit"]
        except Exception as e:
            st.error(f"Prediction failed.\n\n{e}")
            probability = None

    if probability is not None:
        st.metric(
            label="Chance of Admission",
            value=f"{probability:.2%}"
        )

        st.progress(probability)

        if probability >= 0.80:
            st.success("Excellent chance of admission.")
        elif probability >= 0.60:
            st.warning("Good chance of admission.")
        else:
            st.error("Admission may be difficult. Consider improving your profile.")


st.divider()

with st.sidebar:

    st.header("Model Information")

    if model_info:

        st.write(f"**Model:** {model_info["algorithm"]}")

        st.write(f"**Framework:** {model_info['framework']}")

        st.write(f"**Version:** {model_info['version']}")

    else:

        st.error("Unable to connect to API")

    st.markdown("---")

    st.write("**Backend:** FastAPI")

    st.write("**Tracking:** MLflow")

    st.write("**Deployment:** Render")

    st.markdown("---")

    st.caption(
        "This application communicates with a FastAPI "
        "backend which serves the trained machine learning model."
    )