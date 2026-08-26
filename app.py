

import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import joblib
import shap

# ============================================================
# AEROGUARDIAN — STREAMLIT APP + SHAP
# ============================================================

MODEL_PATH = "fd001_final_deployable.keras"
SCALER_PATH = "fd001_final_deployable_scaler.pkl"
FEATURES_PATH = "fd001_final_deployable_features.pkl"
SHAP_BACKGROUND_PATH = "shap_background.npy"

SEQ_LEN = 30


# ============================================================
# 1. DEPLOYMENT-SAFE CUSTOM LAYER
# ============================================================

@tf.keras.utils.register_keras_serializable(
    package="AeroGuardian"
)
class SumOverTime(tf.keras.layers.Layer):

    def call(self, inputs):
        return tf.reduce_sum(inputs, axis=1)

    def compute_output_shape(self, input_shape):
        return (
            input_shape[0],
            input_shape[2]
        )


# ============================================================
# 2. LOAD MODEL + PREPROCESSING + SHAP BACKGROUND
# ============================================================

@st.cache_resource
def load_artifacts():

    model = tf.keras.models.load_model(
        MODEL_PATH,
        compile=False,
        custom_objects={
            "SumOverTime": SumOverTime
        }
    )

    scaler = joblib.load(
        SCALER_PATH
    )

    features = joblib.load(
        FEATURES_PATH
    )

    shap_background = np.load(
        SHAP_BACKGROUND_PATH
    ).astype(np.float32)

    return (
        model,
        scaler,
        features,
        shap_background
    )


# ============================================================
# 3. PAGE
# ============================================================

st.set_page_config(
    page_title="AeroGuardian",
    page_icon="✈️",
    layout="wide"
)

st.title("✈️ AeroGuardian")
st.markdown(
    "### Aircraft Remaining Useful Life Prediction"
)

st.caption(
    "Predictive maintenance using NASA C-MAPSS FD001"
)


# ============================================================
# 4. LOAD ARTIFACTS
# ============================================================

try:

    (
        model,
        scaler,
        features,
        shap_background
    ) = load_artifacts()

except Exception as e:

    st.error(
        "Could not load the model artifacts."
    )

    st.exception(e)
    st.stop()


# ============================================================
# 5. SIDEBAR
# ============================================================

with st.sidebar:

    st.header("Model")

    st.write(
        "**Architecture:** CNN + LSTM + Attention"
    )

    st.write(
        "**Sequence:** 30 cycles"
    )

    st.write(
        "**RUL cap:** 125 cycles"
    )

    st.write(
        "**Task:** Regression"
    )

    st.divider()

    st.caption(
        "Official FD001 test performance"
    )

    st.metric(
        "MAE",
        "11.06 cycles"
    )

    st.metric(
        "RMSE",
        "14.91 cycles"
    )

    st.metric(
        "R²",
        "0.871"
    )


# ============================================================
# 6. UPLOAD
# ============================================================

uploaded_file = st.file_uploader(
    "Upload engine CSV",
    type=["csv"],
    help="Upload one engine's FD001-style sensor history."
)

if uploaded_file is None:

    st.info(
        "Upload an FD001-style engine CSV to generate a prediction."
    )

    st.stop()


# ============================================================
# 7. READ CSV
# ============================================================

try:

    df = pd.read_csv(
        uploaded_file
    )

except Exception as e:

    st.error(
        "Could not read the uploaded CSV."
    )

    st.exception(e)
    st.stop()


# ============================================================
# 8. VALIDATE COLUMNS
# ============================================================

required_columns = [
    "unit",
    "cycle",
    "setting_1",
    "setting_2",
    "setting_3"
] + [
    f"sensor_{i}"
    for i in range(1, 22)
]


missing_columns = [
    col
    for col in required_columns
    if col not in df.columns
]


if missing_columns:

    st.error(
        "This file is missing required FD001 columns."
    )

    st.code(
        ", ".join(missing_columns)
    )

    st.stop()


df = df.sort_values(
    "cycle"
).reset_index(
    drop=True
)


# ============================================================
# 9. CHECK ENOUGH HISTORY
# ============================================================

if len(df) < SEQ_LEN:

    st.error(
        f"This engine has {len(df)} observed cycles. "
        f"The model needs at least {SEQ_LEN} cycles."
    )

    st.stop()


# ============================================================
# 10. LATEST 30 CYCLES
# ============================================================

latest_30 = df.tail(
    SEQ_LEN
).copy()


# ============================================================
# 11. CHECK FEATURES
# ============================================================

missing_features = [
    col
    for col in features
    if col not in latest_30.columns
]


if missing_features:

    st.error(
        "The uploaded CSV does not contain all model features."
    )

    st.code(
        ", ".join(missing_features)
    )

    st.stop()


model_input = latest_30[
    features
].apply(
    pd.to_numeric,
    errors="coerce"
)


if model_input.isnull().any().any():

    bad_columns = model_input.columns[
        model_input.isnull().any()
    ].tolist()

    st.error(
        "Missing or non-numeric values were found "
        "in the model input."
    )

    st.code(
        ", ".join(bad_columns)
    )

    st.stop()


# ============================================================
# 12. SCALE
# ============================================================

try:

    scaled_input = scaler.transform(
        model_input.to_numpy(
            dtype=np.float32
        )
    ).astype(
        np.float32
    )

    X_input = np.expand_dims(
        scaled_input,
        axis=0
    )

except Exception as e:

    st.error(
        "Preprocessing failed."
    )

    st.exception(e)
    st.stop()


expected_shape = (
    1,
    SEQ_LEN,
    len(features)
)


if X_input.shape != expected_shape:

    st.error(
        f"Input shape mismatch. "
        f"Expected {expected_shape}, got {X_input.shape}."
    )

    st.stop()


# ============================================================
# 13. PREDICTION
# ============================================================

try:

    predicted_rul = float(
        model.predict(
            X_input,
            verbose=0
        )[0][0]
    )

except Exception as e:

    st.error(
        "Prediction failed."
    )

    st.exception(e)
    st.stop()


predicted_rul = max(
    0.0,
    predicted_rul
)


latest_cycle = int(
    df["cycle"].iloc[-1]
)

estimated_failure_cycle = (
    latest_cycle
    + predicted_rul
)


# ============================================================
# 14. HEALTH STATUS
# ============================================================

if predicted_rul > 100:

    status = "LOW RISK"

    status_text = (
        "The estimated remaining life is relatively high."
    )

    action = (
        "Continue routine monitoring."
    )

    display_function = st.success

elif predicted_rul > 50:

    status = "MONITOR"

    status_text = (
        "The engine has a moderate estimated remaining life."
    )

    action = (
        "Plan maintenance monitoring for upcoming cycles."
    )

    display_function = st.warning

elif predicted_rul > 20:

    status = "MAINTENANCE PLANNED"

    status_text = (
        "The predicted remaining life is becoming limited."
    )

    action = (
        "Schedule maintenance before the estimated failure point."
    )

    display_function = st.warning

else:

    status = "HIGH RISK"

    status_text = (
        "The predicted remaining life is low."
    )

    action = (
        "Prioritize maintenance inspection."
    )

    display_function = st.error


# ============================================================
# 15. MAIN RESULT
# ============================================================

st.subheader(
    "Engine Health Assessment"
)


c1, c2, c3 = st.columns(3)


with c1:

    st.metric(
        "Estimated Remaining Useful Life",
        f"{predicted_rul:.1f} cycles"
    )


with c2:

    st.metric(
        "Estimated Failure Cycle",
        f"{estimated_failure_cycle:.0f}"
    )


with c3:

    st.metric(
        "Latest Observed Cycle",
        f"{latest_cycle}"
    )


display_function(
    f"**{status}** — {status_text} {action}"
)


# ============================================================
# 16. RUL TIMELINE
# ============================================================

st.subheader("Engine Life Timeline")

timeline_left, timeline_right = st.columns([1, 2])

with timeline_left:
    st.write(f"**Current cycle:** {latest_cycle}")
    st.write(f"**Predicted RUL:** {predicted_rul:.1f} cycles")
    st.write(
        f"**Estimated failure point:** "
        f"cycle {estimated_failure_cycle:.0f}"
    )

with timeline_right:
    timeline_df = pd.DataFrame({
        "Engine Cycle": [
            latest_cycle,
            estimated_failure_cycle
        ],
        "Predicted Remaining Useful Life (cycles)": [
            predicted_rul,
            0.0
        ]
    })

    st.line_chart(
        timeline_df.set_index(
            "Engine Cycle"
        )
    )

    st.caption(
        "X-axis: engine operating cycle • "
        "Y-axis: predicted remaining useful life (cycles)"
    )


# ============================================================
# 17. SHAP EXPLANATION
# ============================================================

st.divider()

st.subheader("🔍 Prediction Explanation")

st.caption(
    "SHAP shows which sensor features had the strongest influence "
    "on this engine's RUL prediction."
)

try:
    with st.spinner("Calculating explanation..."):

        explainer = shap.GradientExplainer(
            model,
            shap_background
        )

        shap_values = explainer.shap_values(
            X_input
        )

    if isinstance(shap_values, list):
        shap_values = shap_values[0]

    shap_values = np.asarray(shap_values)

    if shap_values.ndim == 4:
        shap_values = shap_values[..., 0]

    if shap_values.ndim != 3:
        raise ValueError(
            f"Unexpected SHAP shape: {shap_values.shape}"
        )

    sample_shap = shap_values[0]

    mean_abs_importance = np.mean(
        np.abs(sample_shap),
        axis=0
    )

    mean_signed_contribution = np.mean(
        sample_shap,
        axis=0
    )

    shap_df = pd.DataFrame({
        "Feature": features,
        "Importance": mean_abs_importance,
        "Contribution": mean_signed_contribution
    }).sort_values(
        "Importance",
        ascending=False
    )

    top_features = shap_df.head(8).copy()

    # --------------------------------------------------------
    # Top contributors chart
    # --------------------------------------------------------

    st.write("### Top contributing sensors")

    chart_df = (
        top_features[
            ["Feature", "Importance"]
        ]
        .sort_values(
            "Importance",
            ascending=True
        )
        .set_index("Feature")
    )

    st.bar_chart(chart_df)

    st.caption(
        "Larger bars mean stronger influence on this prediction. "
        "Bar length shows importance; direction is summarized below."
    )

    # --------------------------------------------------------
    # Directional summary — compact, not repetitive
    # --------------------------------------------------------

    st.write("### What influenced the prediction?")

    contributors = top_features.copy()
    contributors["Effect"] = np.where(
        contributors["Contribution"] > 0,
        "↑ Higher RUL",
        "↓ Lower RUL"
    )

    display_df = contributors[
        ["Feature", "Effect", "Importance"]
    ].copy()

    display_df["Importance"] = display_df[
        "Importance"
    ].round(3)

    st.dataframe(
        display_df,
        hide_index=True,
        use_container_width=True
    )

    # --------------------------------------------------------
    # One-sentence takeaway
    # --------------------------------------------------------

    strongest = top_features.iloc[0]

    strongest_direction = (
        "higher"
        if strongest["Contribution"] > 0
        else "lower"
    )

    st.info(
        f"**Main takeaway:** {strongest['Feature']} had the "
        f"largest overall influence and pushed the estimated "
        f"RUL toward a **{strongest_direction}** value."
    )

except Exception as e:

    st.warning(
        "The RUL prediction worked, but the SHAP explanation "
        "could not be generated."
    )

    st.exception(e)


# ============================================================
# 18. INPUT DETAILS
# ============================================================

with st.expander(
    "Input details"
):

    st.write(
        f"**Engine ID:** {df['unit'].iloc[0]}"
    )

    st.write(
        f"**History available:** {len(df)} cycles"
    )

    st.write(
        f"**Latest observed cycle:** {latest_cycle}"
    )

    st.write(
        f"**Cycles used:** latest {SEQ_LEN}"
    )

    st.write(
        f"**Model features:** {len(features)}"
    )


# ============================================================
# 19. FOOTER
# ============================================================

st.divider()

st.caption(
    "AeroGuardian • NASA C-MAPSS FD001 • "
    "CNN + LSTM + Attention • SHAP"
)
