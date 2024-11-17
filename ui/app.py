# app.py
import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
import joblib
from math import pi

# Load the dataset and model
DATA = load_breast_cancer()
MODEL_PATH = "../models/best_model.pkl"

def main():
    st.set_page_config(page_title="Breast Cancer Detection", layout="wide")

    # Create a custom sidebar navbar for navigation
    st.sidebar.title("Navigation")
    menu = ["Home", "Dataset Exploration", "Model Testing", "Model Metrics"]
    choice = st.sidebar.selectbox("Select a Page", menu)

    # Home Page with About Me
    if choice == "Home":
        homepage()

    # Dataset Exploration
    elif choice == "Dataset Exploration":
        dataset_exploration()

    # Model Testing
    elif choice == "Model Testing":
        model_testing()

    # Model Metrics
    elif choice == "Model Metrics":
        model_metrics()

# Homepage with About Me Section
def homepage():
    st.title("Welcome to Breast Cancer Detection App")
    st.subheader("About Me")
    st.write("""
    Hello! I'm Halleluyah Darasimi Oludele, the creator of this Breast Cancer Detection application.
    This project aims to provide an interactive platform to explore a machine learning model
    built on the Wisconsin Breast Cancer dataset for tumor classification (benign or malignant).
    Through this app, you can:

    - Explore the dataset and its features
    - Test the model with new inputs and see predictions
    - View various performance metrics of the model

    ### Goals of this Project:
    - To provide a simple interface for cancer diagnosis prediction.
    - To understand and visualize how machine learning models work in real-world applications.

    I'm passionate about leveraging machine learning in healthcare and making data science more accessible!
    """)

    # Add an image or project logo here
    # st.image("path_to_image", caption="Project Logo", width=300)

# Dataset Exploration
def dataset_exploration():
    st.title("Dataset Exploration")
    st.write("Explore the Breast Cancer Wisconsin dataset used for training the model.")

    # Dataset overview
    data_df = pd.DataFrame(DATA.data, columns=DATA.feature_names)
    data_df['target'] = DATA.target
    target_mapping = dict(enumerate(DATA.target_names))
    data_df['target_name'] = data_df['target'].map(target_mapping)

    # Display the dataset
    st.subheader("Dataset Preview")
    st.dataframe(data_df.head())

    # Feature summary
    st.subheader("Feature Summary")
    st.write(data_df.describe())

    # Target class distribution
    st.subheader("Target Class Distribution")
    fig, ax = plt.subplots()
    data_df['target_name'].value_counts().plot(kind='bar', ax=ax, color=['lightblue', 'orange'])
    ax.set_title("Distribution of Target Classes")
    ax.set_xlabel("Class")
    ax.set_ylabel("Count")
    st.pyplot(fig)

    # Correlation heatmap
    st.subheader("Correlation Heatmap")
    if st.checkbox("Show Heatmap"):
        fig, ax = plt.subplots(figsize=(12, 8))
        sns.heatmap(data_df.iloc[:, :-2].corr(), cmap="coolwarm", annot=False, ax=ax)
        st.pyplot(fig)

# Model Testing
def model_testing():
    st.title("Model Testing")
    st.write("Test the trained model by providing input values for each feature.")

    # Load the saved model and scaler
    try:
        model = joblib.load(MODEL_PATH)
        scaler = StandardScaler()
        scaler.fit(DATA.data)  # Fit scaler to training data

        # Input fields
        st.sidebar.header("Input Features")
        user_input = {}
        for feature in DATA.feature_names:
            user_input[feature] = st.sidebar.number_input(
                f"{feature}", min_value=float(np.min(DATA.data)), max_value=float(np.max(DATA.data)), step=0.01
            )

        # Predict button
        if st.sidebar.button("Predict"):
            # Preprocess input
            input_data = np.array([list(user_input.values())])
            input_scaled = scaler.transform(input_data)

            # Make prediction
            prediction = model.predict(input_scaled)[0]
            prediction_proba = model.predict_proba(input_scaled)[0]

            # Display results
            st.subheader("Prediction Results")
            result = "Malignant" if prediction == 1 else "Benign"
            confidence = prediction_proba[prediction] * 100
            st.write(f"The tumor is predicted to be **{result}** with a confidence of **{confidence:.2f}%**.")

            # Display Radar Chart for visualization
            st.subheader("Radar Chart: Tumor Features")
            plot_radar_chart(user_input, DATA.feature_names)

    except Exception as e:
        st.error(f"Error loading model or processing input: {e}")

# Radar Chart for Visualizing Tumor Features
def plot_radar_chart(user_input, features):
    # Calculate the average values for benign and malignant tumors (from the dataset)
    benign_mean = DATA.data[DATA.target == 0].mean(axis=0)
    malignant_mean = DATA.data[DATA.target == 1].mean(axis=0)

    # Prepare data for the radar chart
    categories = features
    user_values = list(user_input.values())

    # Number of features
    num_vars = len(categories)

    # Compute angle for each feature on the radar chart
    angles = [n / float(num_vars) * 2 * pi for n in range(num_vars)]
    angles += angles[:1]  # Closing the circle

    # Radar chart setup
    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))

    # User input data
    user_values += user_values[:1]
    ax.plot(angles, user_values, color='r', linewidth=2, label='User Input')
    ax.fill(angles, user_values, color='r', alpha=0.25)

    # Benign mean data
    benign_mean = np.append(benign_mean, benign_mean[0])
    ax.plot(angles, benign_mean, color='b', linewidth=2, label='Benign Mean')
    ax.fill(angles, benign_mean, color='b', alpha=0.25)

    # Malignant mean data
    malignant_mean = np.append(malignant_mean, malignant_mean[0])
    ax.plot(angles, malignant_mean, color='g', linewidth=2, label='Malignant Mean')
    ax.fill(angles, malignant_mean, color='g', alpha=0.25)

    # Customize the radar chart
    ax.set_yticklabels([])
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, rotation=45, ha='right')
    ax.set_title("Radar Chart: Tumor Features", size=16, color='black', y=1.1)

    # Add legend
    ax.legend(loc='upper right', bbox_to_anchor=(1.2, 1.2))

    # Display chart
    st.pyplot(fig)

# Model Metrics
def model_metrics():
    st.title("Model Metrics")
    st.write("Explore the performance metrics of the trained model.")

    # Load the saved model
    try:
        model = joblib.load(MODEL_PATH)
        scaler = StandardScaler()
        scaler.fit(DATA.data)  # Fit scaler to training data

        # Split data
        X = DATA.data
        y = DATA.target
        X_scaled = scaler.transform(X)

        # Predictions
        y_pred = model.predict(X_scaled)
        y_proba = model.predict_proba(X_scaled)[:, 1]

        # Classification Report
        st.subheader("Classification Report")
        report = classification_report(y, y_pred, target_names=DATA.target_names, output_dict=True)
        st.table(pd.DataFrame(report).transpose())

        # Confusion Matrix
        st.subheader("Confusion Matrix")
        conf_matrix = confusion_matrix(y, y_pred)
        fig, ax = plt.subplots()
        sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=DATA.target_names, yticklabels=DATA.target_names, ax=ax)
        ax.set_xlabel("Predicted Label")
        ax.set_ylabel("True Label")
        st.pyplot(fig)

        # ROC Curve and AUC
        st.subheader("ROC Curve and AUC")
        fpr, tpr, _ = roc_curve(y, y_proba)
        roc_auc = auc(fpr, tpr)
        fig, ax = plt.subplots()
        ax.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (AUC = {roc_auc:.2f})")
        ax.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title("Receiver Operating Characteristic (ROC)")
        ax.legend(loc="lower right")
        st.pyplot(fig)

        # Top 5 Predictions
        st.subheader("Top 5 Predictions")
        top_5_idx = np.argsort(y_proba)[-5:]
        top_5_preds = [(i, DATA.target_names[y[i]], y_proba[i]) for i in top_5_idx]
        st.write("Top 5 predictions based on probability:")
        for i, (idx, true_label, prob) in enumerate(top_5_preds, 1):
            st.write(f"{i}. Sample {idx}: True Label: {true_label}, Predicted Proba: {prob:.2f}")

    except Exception as e:
        st.error(f"Error loading model or processing metrics: {e}")

# Run the app
if __name__ == "__main__":
    main()
