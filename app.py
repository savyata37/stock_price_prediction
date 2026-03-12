# app.py
import os
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
import streamlit as st

from cleaning import fill_missing_values, handle_skewness, remove_outliers, run_initial_cleaning
from eda import engineer_features


st.set_page_config(page_title="S&P 500 Predictor", layout="wide")
st.title("S&P 500 Close Price Predictor")
st.write("Enter a standard daily market row and the app will calculate the model features automatically.")
st.caption("Required inputs: Date, Open, High, Low, Close, Volume, policy_change, and fedrete.")


def load_bundle():
    bundle = None
    scaler = None
    pca = None

    if os.path.exists("model_bundle.pkl"):
        bundle = joblib.load("model_bundle.pkl")
        scaler = bundle.get("scaler")
        pca = bundle.get("pca")

    if scaler is None:
        scaler = joblib.load("scaler.pkl")
    if pca is None:
        pca = joblib.load("pca.pkl")

    regressor = bundle.get("regressor") if bundle else None
    classifier = bundle.get("classifier") if bundle else None
    feature_columns = bundle.get("feature_columns") if bundle else None
    train_medians = bundle.get("train_medians") if bundle else None

    if feature_columns is None and hasattr(scaler, "feature_names_in_"):
        feature_columns = list(scaler.feature_names_in_)

    if train_medians is not None and not isinstance(train_medians, pd.Series):
        train_medians = pd.Series(train_medians)

    return regressor, classifier, scaler, pca, feature_columns, train_medians


def load_history():
    raw_history = pd.read_csv("sp500_data.csv")
    return run_initial_cleaning(raw_history)


def build_prediction_features(history_df, prediction_date, raw_inputs, feature_columns, train_medians):
    history_df = history_df.loc[history_df.index != prediction_date].copy()

    prediction_row = pd.DataFrame(
        [
            {
                "Date": prediction_date.strftime("%m/%d/%Y"),
                "Close": raw_inputs["Close"],
                "High": raw_inputs["High"],
                "Low": raw_inputs["Low"],
                "Open": raw_inputs["Open"],
                "Volume": raw_inputs["Volume"],
                "policy_change": raw_inputs["policy_change"],
                "fedrete": raw_inputs["fedrete"],
            }
        ]
    )

    combined_raw = pd.concat([history_df.reset_index(), prediction_row], ignore_index=True)
    combined_clean = run_initial_cleaning(combined_raw)
    combined_engineered = engineer_features(combined_clean.copy())
    combined_processed = fill_missing_values(combined_engineered.copy())
    combined_processed = handle_skewness(combined_processed)
    combined_processed = remove_outliers(combined_processed)

    if prediction_date not in combined_processed.index:
        raise ValueError("The entered date could not be aligned in preprocessing.")

    features = combined_processed.loc[[prediction_date]].copy().reindex(columns=feature_columns)
    fallback_medians = (
        combined_processed.reindex(columns=feature_columns).median(numeric_only=True)
        if train_medians is None
        else train_medians.reindex(feature_columns)
    )
    features = features.replace([float("inf"), float("-inf")], pd.NA)
    return features.fillna(fallback_medians).fillna(0)


try:
    regressor, classifier, scaler, pca, feature_columns, train_medians = load_bundle()
    history_df = load_history()
    st.success("Models and historical data loaded successfully.")
except FileNotFoundError as exc:
    st.error(f"Missing required file: {exc}")
    st.info("Run the notebook through the model-save section so the Streamlit artifacts exist.")
    st.stop()

best_models_available = all(
    item is not None for item in [regressor, classifier, scaler, pca, feature_columns]
)

with st.sidebar:
    st.header("Inputs")
    st.write("Use the raw fields users usually know: Date, Open, High, Low, Close, Volume, policy_change, and fedrete.")
    if feature_columns is not None:
        st.caption(f"Loaded training schema with {len(feature_columns)} features")

st.header("Enter Daily Market Data")

col1, col2, col3 = st.columns(3)
with col1:
    market_date = st.date_input(
        "Date",
        value=datetime.now().date(),
        help="Enter the market date for the OHLCV row you are providing."
    )
with col2:
    open_price = st.number_input("Open", min_value=100.0, value=5000.0, step=1.0)
with col3:
    high_price = st.number_input("High", min_value=100.0, value=5100.0, step=1.0)

col4, col5, col6 = st.columns(3)
with col4:
    low_price = st.number_input("Low", min_value=100.0, value=4950.0, step=1.0)
with col5:
    close_price = st.number_input("Close", min_value=100.0, value=5050.0, step=1.0)
with col6:
    volume = st.number_input("Volume", min_value=1_000_000, value=4_000_000_000, step=10_000_000)

col7, col8 = st.columns(2)
with col7:
    policy_change = st.checkbox("policy_change", value=False)
with col8:
    fed_rate = st.number_input("fedrete", min_value=0.0, max_value=10.0, value=4.5, step=0.25)

if st.button("Predict Tomorrow's Market", use_container_width=True):
    if not best_models_available:
        st.warning("Trained models are not available. Run the notebook through the model export cells first.")
    else:
        try:
            with st.spinner("Preparing features and making prediction..."):
                prediction_ts = pd.Timestamp(market_date)
                today_features = build_prediction_features(
                    history_df,
                    prediction_ts,
                    {
                        "Close": close_price,
                        "High": high_price,
                        "Low": low_price,
                        "Open": open_price,
                        "Volume": volume,
                        "policy_change": int(policy_change),
                        "fedrete": fed_rate,
                    },
                    feature_columns,
                    train_medians,
                )

                input_scaled = scaler.transform(today_features)
                input_pca = pca.transform(input_scaled)
                price_pred_log = float(regressor.predict(today_features)[0])
                price_pred = float(np.expm1(price_pred_log))
                direction_proba = classifier.predict_proba(input_pca)[0]
                direction = "Up" if direction_proba[1] >= 0.5 else "Down"
                confidence = min(max(direction_proba) * 100, 99.9)

            st.info(f"Prediction based on market data entered for {prediction_ts.strftime('%Y-%m-%d')}")

            result_col1, result_col2, result_col3 = st.columns(3)
            with result_col1:
                st.metric("Predicted Close", f"{price_pred:,.2f}")
            with result_col2:
                st.metric("Direction", direction)
            with result_col3:
                st.metric("Confidence", f"{confidence:.1f}%")

            with st.expander("Calculated Model Inputs"):
                st.dataframe(today_features.transpose(), use_container_width=True)

            st.caption("Predicted Close is converted back from the model's log-transformed scale. Confidence is the classifier probability, not a guarantee.")

        except Exception as exc:
            st.error(f"Prediction failed: {exc}")