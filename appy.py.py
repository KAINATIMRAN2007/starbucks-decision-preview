
import pickle
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Starbucks Calorie Predictor",
    page_icon="☕",
    layout="centered"
)

st.title("☕ Starbucks Calorie Predictor")
st.write("Predict estimated calories for a Starbucks beverage.")

# Load the trained Decision Tree model
try:
    with open("starbucks_calorie_model.pkl", "rb") as file:
        bundle = pickle.load(file)

    model = bundle["model"]
    feature_cols = bundle["feature_cols"]

    # Get available beverage choices from the model
    encoder = model.named_steps["preprocessor"].named_transformers_["text"]
    categories = encoder.categories_

    choices = {
        col: list(values)
        for col, values in zip(feature_cols, categories)
    }

    beverage = st.selectbox(
        "Choose a beverage",
        choices["Beverage"]
    )

    selected = {"Beverage": beverage}

    if "Beverage_prep" in feature_cols:
        selected["Beverage_prep"] = st.selectbox(
            "Choose preparation type",
            choices["Beverage_prep"]
        )

    if st.button("Predict Calories"):
        input_data = pd.DataFrame([selected])[feature_cols]
        input_data = input_data.astype(str)
        prediction = model.predict(input_data)[0]

        st.success(f"Estimated Calories: {prediction:.1f}")
        st.caption(
            "This is a model estimate, not an official nutrition result."
        )

except FileNotFoundError:
    st.error(
        "Model file not found. Place starbucks_calorie_model.pkl "
        "in the same folder as app.py."
    )
except Exception as error:
    st.error(f"Something went wrong: {error}")
