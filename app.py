
import streamlit as st
import pandas as pd
import numpy as np

from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

from tensorflow import keras
from tensorflow.keras import layers


# ==========================================
# 1. PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="PowerPredict AI",
    page_icon="⚡",
    layout="wide"
)



# ==========================================
# CUSTOM GREEN THEME
# ==========================================

st.markdown("""
<style>

/* Main Background */
.stApp {
    background: linear-gradient(
        135deg,
        #F5FBF6 0%,
        #E8F5E9 100%
    );
}

/* Main Titles */
h1, h2, h3 {
    color: #14532D !important;
    font-weight: 750 !important;
}

/* Normal Text */
p, label {
    color: #244B35;
}

/* All Success Messages - Light Green */
div[data-testid="stAlert"]:has([data-testid="stIconSuccess"]) {
    background-color: #DCFCE7 !important;
    border: 1px solid #86EFAC !important;
    border-radius: 10px !important;
}

/* Success Message Text */
div[data-testid="stAlert"]:has([data-testid="stIconSuccess"]) p {
    color: #166534 !important;
    font-weight: 600 !important;
}

/* Success Icon */
div[data-testid="stAlert"]:has([data-testid="stIconSuccess"]) svg {
    color: #16A34A !important;
    fill: #16A34A !important;
}

/* Info Message - Sky Blue */
div[data-testid="stAlert"]:has([data-testid="stIconInfo"]) {
    background-color: #E0F2FE !important;
    border: 1px solid #7DD3FC !important;
}

div[data-testid="stAlert"]:has([data-testid="stIconInfo"]) p {
    color: #075985 !important;
}

div[data-testid="stAlert"]:has([data-testid="stIconInfo"]) svg {
    fill: #0284C7 !important;
    color: #0284C7 !important;
}

/* Warning Message */
div[data-testid="stAlert"]:has([data-testid="stIconWarning"]) {
    background-color: #FEF3C7 !important;
}

div[data-testid="stAlert"]:has([data-testid="stIconWarning"]) p {
    color: #92400E !important;
}

/* Primary Buttons */
.stButton > button,
.stFormSubmitButton > button {
    background-color: #15803D !important;
    color: #FFFFFF !important;
    border: 1px solid #15803D !important;
    border-radius: 9px !important;
    font-weight: 700 !important;
}

/* Button Text */
.stButton > button p,
.stFormSubmitButton > button p {
    color: #FFFFFF !important;
}

/* Button Hover */
.stButton > button:hover,
.stFormSubmitButton > button:hover {
    background-color: #166534 !important;
    border-color: #166534 !important;
    color: #FFFFFF !important;
}

/* Multiselect Selected Tags */
.stMultiSelect [data-baseweb="tag"] {
    background-color: #15803D !important;
    color: #FFFFFF !important;
}

.stMultiSelect [data-baseweb="tag"] span {
    color: #FFFFFF !important;
}

/* Metric Cards */
[data-testid="stMetric"] {
    background-color: #DCFCE7;
    padding: 18px;
    border-radius: 12px;
    border: 1px solid #BBDFC4;
}

/* Input Fields */
.stNumberInput input {
    border-radius: 8px;
}

/* Dataframe */
.stDataFrame {
    border: 1px solid #CDE7D2;
    border-radius: 10px;
}

/* Divider */
hr {
    border-color: #B7DCC0;
}

</style>
""", unsafe_allow_html=True)


st.title("⚡ PowerPredict AI")
st.subheader("Power Plant Energy Output Prediction")

st.write(
    "Compare Machine Learning and Deep Learning "
    "models to predict electrical power output."
)

# ==========================================
# CUSTOM LIGHT GREEN SUCCESS MESSAGE
# ==========================================

def green_success(message):

    st.markdown(
        f"""
        <div style="
            background-color: #DCFCE7;
            border: 1px solid #86EFAC;
            border-radius: 10px;
            padding: 13px 18px;
            color: #166534;
            font-weight: 600;
            margin: 8px 0;
        ">
            ✅ {message}
        </div>
        """,
        unsafe_allow_html=True
    )

# ==========================================
# 2. LOAD DATASET
# ==========================================

@st.cache_data
def load_data():

    file_path = Path("data/power_plant_cleaned.csv")

    df = pd.read_csv(file_path)

    return df


try:
    df = load_data()

    green_success("Dataset loaded successfully!")

    st.write("Dataset Shape:", df.shape)

    with st.expander("View Dataset"):
        st.dataframe(df)

except Exception as e:
    st.error(f"Dataset loading error: {e}")
    st.stop()


# ==========================================
# 3. FEATURES AND TARGET
# ==========================================

X = df[['AT', 'V', 'AP', 'RH']]
y = df['PE']


# ==========================================
# 4. MODEL SELECTION
# ==========================================

st.header("Model Selection")

model_options = [
    "Linear Regression",
    "Decision Tree",
    "Random Forest",
    "ANN"
]

selected_models = st.multiselect(
    "Select at least 2 models to compare:",
    model_options,
    default=["Linear Regression", "Random Forest"]
)

st.write("Selected Models:", len(selected_models))


# ==========================================
# 5. METRIC FUNCTION
# ==========================================

def calculate_metrics(y_true, y_pred):

    mae = mean_absolute_error(y_true, y_pred)

    mse = mean_squared_error(y_true, y_pred)

    rmse = np.sqrt(mse)

    r2 = r2_score(y_true, y_pred)

    return {
        "R2": r2,
        "RMSE": rmse,
        "MAE": mae
    }


# ==========================================
# 6. TRAIN MODELS
# ==========================================

def train_models(selected_models):

    # Split 1: 80% development data, 20% test data
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42
    )

    # Split 2: 80% training, 20% validation
    # Final proportions: 64% train, 16% validation, 20% test
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val,
        y_train_val,
        test_size=0.20,
        random_state=42
    )

    results = []
    trained_models = {}

    progress = st.progress(0)

    status = st.empty()

    total_models = len(selected_models)

    for index, model_name in enumerate(selected_models):

        status.info(f"Training {model_name}...")

        # ----------------------------------
        # Linear Regression
        # ----------------------------------

        if model_name == "Linear Regression":

            model = Pipeline([
                ("scaler", StandardScaler()),
                ("regressor", LinearRegression())
            ])

            model.fit(X_train, y_train)

            val_pred = model.predict(X_val)

            test_pred = model.predict(X_test)

            trained_models[model_name] = model

        # ----------------------------------
        # Decision Tree
        # ----------------------------------

        elif model_name == "Decision Tree":

            model = DecisionTreeRegressor(
                random_state=42
            )

            model.fit(X_train, y_train)

            val_pred = model.predict(X_val)

            test_pred = model.predict(X_test)

            trained_models[model_name] = model

        # ----------------------------------
        # Random Forest
        # ----------------------------------

        elif model_name == "Random Forest":

            model = RandomForestRegressor(
                n_estimators=200,
                random_state=42,
                n_jobs=-1
            )

            model.fit(X_train, y_train)

            val_pred = model.predict(X_val)

            test_pred = model.predict(X_test)

            trained_models[model_name] = model

        # ----------------------------------
        # Artificial Neural Network
        # ----------------------------------

        elif model_name == "ANN":

            # Scale features for ANN
            scaler = StandardScaler()

            X_train_scaled = scaler.fit_transform(X_train)

            X_val_scaled = scaler.transform(X_val)

            X_test_scaled = scaler.transform(X_test)

            keras.utils.set_random_seed(42)

            model = keras.Sequential([

                layers.Input(shape=(4,)),

                layers.Dense(64, activation="relu"),

                layers.Dropout(0.2),

                layers.Dense(32, activation="relu"),

                layers.Dense(1)

            ])

            model.compile(
                optimizer="adam",
                loss="mse",
                metrics=["mae"]
            )

            early_stop = keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=10,
                restore_best_weights=True
            )

            model.fit(
                X_train_scaled,
                y_train,
                validation_data=(X_val_scaled, y_val),
                epochs=100,
                batch_size=32,
                callbacks=[early_stop],
                verbose=0
            )

            val_pred = model.predict(
                X_val_scaled,
                verbose=0
            ).flatten()

            test_pred = model.predict(
                X_test_scaled,
                verbose=0
            ).flatten()

            trained_models[model_name] = {
                "model": model,
                "scaler": scaler
            }

        # ----------------------------------
        # Calculate Performance
        # ----------------------------------

        val_metrics = calculate_metrics(y_val, val_pred)

        test_metrics = calculate_metrics(y_test, test_pred)

        results.append({

            "Model": model_name,

            "Validation R2": val_metrics["R2"],

            "Validation RMSE": val_metrics["RMSE"],

            "Validation MAE": val_metrics["MAE"],

            "Test R2": test_metrics["R2"],

            "Test RMSE": test_metrics["RMSE"],

            "Test MAE": test_metrics["MAE"]

        })

        progress.progress((index + 1) / total_models)

    status.markdown("""
<div style="
    background-color: #DCFCE7;
    border: 1px solid #86EFAC;
    border-radius: 10px;
    padding: 13px 18px;
    color: #166534;
    font-weight: 600;
">
    ✅ All selected models trained successfully!
</div>
""", unsafe_allow_html=True)
    results_df = pd.DataFrame(results)

    # Select best model using validation RMSE
    results_df = results_df.sort_values(
        by="Validation RMSE",
        ascending=True
    ).reset_index(drop=True)

    best_model_name = results_df.iloc[0]["Model"]

    return results_df, trained_models, best_model_name


# ==========================================
# 7. TRAINING BUTTON
# ==========================================

st.divider()

if st.button(
    "🚀 Train and Compare Models",
    type="primary",
    disabled=len(selected_models) < 2
):

    with st.spinner("Training selected models..."):

        results_df, trained_models, best_model_name = train_models(
            selected_models
        )

        st.session_state["results"] = results_df

        st.session_state["trained_models"] = trained_models

        st.session_state["best_model"] = best_model_name


if len(selected_models) < 2:

    st.warning("Please select at least 2 models.")


# ==========================================
# 8. DISPLAY MODEL COMPARISON
# ==========================================

if "results" in st.session_state:

    st.header("📊 Model Performance Comparison")

    results_df = st.session_state["results"]

    st.dataframe(
        results_df,
        use_container_width=True
    )

    st.subheader("Validation RMSE Comparison")

    chart_data = results_df.set_index("Model")[
        ["Validation RMSE"]
    ]

    st.bar_chart(chart_data)

    # ----------------------------------
    # Best Model
    # ----------------------------------

    best_model_name = st.session_state["best_model"]

    green_success(f"🏆 Best Model: {best_model_name}")

    st.write(
        "The best model is selected based on the lowest "
        "Validation RMSE."
    )

# ==========================================
# 9. UNSEEN DATA PREDICTION
# ==========================================

st.divider()

st.header("🔮 Test with Unseen Data")

st.write(
    "Enter new environmental conditions to predict "
    "electrical power output using a trained model."
)

# Check whether models have been trained
if "trained_models" in st.session_state:

    trained_models = st.session_state["trained_models"]

    best_model_name = st.session_state["best_model"]

    # Model selection
    prediction_model = st.selectbox(
        "Choose a trained model:",
        list(trained_models.keys()),
        index=list(trained_models.keys()).index(best_model_name)
    )

    st.subheader("Enter Environmental Features")

    # Input form
    with st.form("prediction_form"):

        col1, col2 = st.columns(2)

        with col1:

            AT = st.number_input(
                "Ambient Temperature (AT) - °C",
                value=float(df["AT"].mean()),
                format="%.2f"
            )

            V = st.number_input(
                "Exhaust Vacuum (V) - cm Hg",
                value=float(df["V"].mean()),
                format="%.2f"
            )

        with col2:

            AP = st.number_input(
                "Ambient Pressure (AP) - millibar",
                value=float(df["AP"].mean()),
                format="%.2f"
            )

            RH = st.number_input(
                "Relative Humidity (RH) - %",
                value=float(df["RH"].mean()),
                format="%.2f"
            )

        predict_button = st.form_submit_button(
            "🔍 Predict Power Output",
            type="primary"
        )

    # Make prediction
    if predict_button:

        input_data = pd.DataFrame(
            [[AT, V, AP, RH]],
            columns=["AT", "V", "AP", "RH"]
        )

        # Check input ranges
        outside_range = []

        for feature in ["AT", "V", "AP", "RH"]:

            if (
                input_data[feature].iloc[0] < df[feature].min()
                or
                input_data[feature].iloc[0] > df[feature].max()
            ):

                outside_range.append(feature)

        if outside_range:

            st.warning(
                "Some input values are outside the original "
                "dataset range: " + ", ".join(outside_range)
                + ". Prediction may be less reliable."
            )

        # Get selected model
        selected_model = trained_models[prediction_model]

        # ANN prediction
        if prediction_model == "ANN":

            ann_model = selected_model["model"]

            scaler = selected_model["scaler"]

            input_scaled = scaler.transform(input_data)

            prediction = ann_model.predict(
                input_scaled,
                verbose=0
            )[0][0]

        # Machine Learning prediction
        else:

            prediction = selected_model.predict(input_data)[0]

        # Display result
        green_success("Prediction completed successfully!")

        st.metric(
            label="Predicted Electrical Power Output",
            value=f"{prediction:.2f} MW"
        )

        st.subheader("Your Input Values")

        st.dataframe(
            input_data,
            use_container_width=True
        )

else:

    
    st.markdown("""
        <div style="
            background-color: #E0F2FE;
            border: 1px solid #7DD3FC;
            border-radius: 10px;
            padding: 15px 18px;
            color: #075985;
            font-weight: 600;
            margin-top: 10px;
        ">
            ℹ️ Please train at least 2 models first
            before testing unseen data.
        </div>
        """, unsafe_allow_html=True)

