import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.metrics import classification_report, confusion_matrix, recall_score, precision_score, f1_score
import plotly.express as px
import time

# --- Page Configuration ---
st.set_page_config(
    page_title="Credit Card Fraud Detection System",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- Helper Function to Load Model ---
@st.cache_resource
def load_model_artifacts():
    """
    Loads the pre-trained model and test data from the pickle file.
    The @st.cache_resource decorator ensures this function runs only once.
    """
    try:
        with open('fraud_model.pkl', 'rb') as f:
            artifacts = pickle.load(f)
        return artifacts
    except FileNotFoundError:
        st.error("Model file not found. Please run `python model/model.py` first to train and save the model.")
        return None

# --- Load Artifacts ---
artifacts = load_model_artifacts()

if artifacts:
    model = artifacts['model']
    X_test = artifacts['X_test']
    y_test = artifacts['y_test']

    # --- Main Application ---

    # --- Sidebar ---
    st.sidebar.title("Fraud Detection Control Panel")
    st.sidebar.info(
        "This dashboard uses a pre-trained Random Forest model to detect fraudulent credit card transactions."
    )

    # --- Predictions and Metrics ---
    y_pred = model.predict(X_test)
    recall = recall_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)

    # --- Dashboard Title ---
    st.title("💳 Real-Time Credit Card Fraud Detection Dashboard")
    st.markdown("---")

    # --- Key Performance Indicators (KPIs) ---
    st.header("Model Performance Metrics")
    st.write(
        "These metrics evaluate the model's ability to correctly identify fraudulent transactions. "
        "**Recall** is critical for minimizing missed fraud cases."
    )

    kpi_cols = st.columns(3)
    kpi_cols[0].metric(label="**Recall Score**", value=f"{recall:.2%}",
                       help="Percentage of actual frauds correctly identified. Higher is better.")
    kpi_cols[1].metric(label="**Precision Score**", value=f"{precision:.2%}",
                       help="Percentage of flagged transactions that were actually fraud. Higher means fewer false positives.")
    kpi_cols[2].metric(label="**F1-Score**", value=f"{f1:.2%}",
                       help="A balance between Recall and Precision.")

    # --- Visualizations ---
    st.markdown("---")
    st.header("Data Visualization & Analysis")

    # Confusion Matrix
    fig_cm = px.imshow(conf_matrix,
                       labels=dict(x="Predicted Label", y="True Label", color="Count"),
                       x=['Not Fraud', 'Fraud'],
                       y=['Not Fraud', 'Fraud'],
                       text_auto=True,
                       color_continuous_scale='Blues',
                       title="<b>Confusion Matrix</b>")
    fig_cm.update_layout(title_x=0.5)

    # Feature Importance
    feature_importances = pd.DataFrame({'feature': X_test.columns, 'importance': model.feature_importances_})
    feature_importances = feature_importances.sort_values('importance', ascending=False).head(15)

    fig_fi = px.bar(feature_importances, x='importance', y='feature', orientation='h',
                    title="<b>Top 15 Most Important Features</b>",
                    labels={'importance': 'Importance Score', 'feature': 'Feature Name'})
    fig_fi.update_layout(title_x=0.5, yaxis={'categoryorder':'total ascending'})

    viz_cols = st.columns(2)
    with viz_cols[0]:
        st.plotly_chart(fig_cm, use_container_width=True)
    with viz_cols[1]:
        st.plotly_chart(fig_fi, use_container_width=True)

    # --- Real-Time Transaction Monitoring Simulation ---
    st.markdown("---")
    st.header("Live Transaction Monitoring")
    st.write("This section simulates a live feed of incoming transactions and their fraud predictions.")

    if st.sidebar.button("▶️ Start Real-Time Simulation"):
        placeholder = st.empty()
        live_df = pd.DataFrame(columns=['Time', 'Amount', 'Prediction', 'Status'])

        for i in range(20):
            idx = np.random.randint(0, len(X_test))
            transaction_data = X_test.iloc[[idx]]
            prediction = model.predict(transaction_data)[0]
            prediction_proba = model.predict_proba(transaction_data)[0][1]

            status_icon = "🚨" if prediction == 1 else "✅"
            status_text = "FRAUD" if prediction == 1 else "Normal"

            new_row = pd.DataFrame({
                'Time': [pd.to_datetime('now').strftime("%H:%M:%S")],
                'Amount': [f"${transaction_data['Amount'].values[0]:,.2f}"],
                'Prediction': [f"{prediction_proba:.2%} Confidence"],
                'Status': [f"{status_icon} {status_text}"]
            })
            live_df = pd.concat([new_row, live_df])
            with placeholder.container():
                st.dataframe(live_df, use_container_width=True)
            time.sleep(1.5)
    else:
        st.info("Click the 'Start Real-Time Simulation' button in the sidebar to begin monitoring.")

    # --- Raw Data Explorer ---
    st.markdown("---")
    with st.expander("Explore Test Data"):
        st.dataframe(X_test.sample(100), use_container_width=True)
