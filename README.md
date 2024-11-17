# Breast Cancer Detection Using Machine Learning

## Overview

This project is a **Breast Cancer Detection** web application built using **Streamlit** and **Machine Learning** techniques. It provides a user-friendly interface to interact with a machine learning model that classifies breast tumors as either **Benign** or **Malignant** using the well-known **Breast Cancer Wisconsin dataset**.

The model leverages advanced algorithms to classify the tumor type based on various diagnostic features, and the application allows users to explore the dataset, test the model with real-time inputs, and view detailed performance metrics.

### **Tech Stack:**
- **Streamlit** for building the web interface
- **Python** for data processing and machine learning
- **scikit-learn** for model training and evaluation
- **Matplotlib**, **Seaborn** for visualizations
- **joblib** for model persistence

## Features

### 1. **Homepage and About Me**
   - The homepage serves as an introduction to the project and the creator. It provides details about the project goals, methodology, and how users can interact with the model.
   - It includes a personalized "About Me" section where the creator's background, motivation, and goals for the project are highlighted.

### 2. **Interactive Model Testing**
   - Users can input values for tumor features through a user-friendly sidebar.
   - The model predicts whether the tumor is benign or malignant based on the input data.
   - The prediction includes a confidence score, providing transparency about how confident the model is in its prediction.

### 3. **Radar Chart Visualization**
   - For better understanding, the app provides a **Radar Chart** comparing the input features of a tumor to the average feature values for **Benign** and **Malignant** tumors. This visualization helps users intuitively understand how their input compares to typical tumor profiles.

### 4. **Dataset Exploration**
   - Users can explore the **Breast Cancer Wisconsin dataset** used to train the model. They can view the dataset, check out summary statistics, and analyze feature distributions.
   - The app includes a **correlation heatmap** to show how various features relate to each other and the target variable, enhancing data comprehension.

### 5. **Comprehensive Model Metrics**
   - The app provides a detailed performance evaluation of the machine learning model:
     - **Classification Report**: Precision, Recall, F1-Score, and Support for each class.
     - **Confusion Matrix**: A heatmap visualizing the true vs. predicted class distribution.
     - **ROC Curve**: A plot showing the performance of the classifier at various thresholds.
     - **AUC (Area Under the Curve)**: A numeric evaluation of model performance.
     - **Top 5 Predictions**: Displays the top 5 predictions for test data based on model probabilities.

### 6. **Easy-to-Navigate Interface**
   - The app is designed with a **navbar-style sidebar** for easy navigation between different sections:
     - Home
     - Dataset Exploration
     - Model Testing
     - Model Metrics
   - The navigation is intuitive, allowing users to jump between pages seamlessly without any confusion.

### 7. **Scalability**
   - The underlying model can be retrained with other datasets, and the app can be expanded to support more models and different prediction tasks, making it scalable for various use cases beyond breast cancer detection.

## Standout Features

### **1. User-Centric Interface**
   - The app has been designed to be intuitive and interactive. Even users with minimal technical background can easily understand how the model works and visualize its predictions.

### **2. Real-Time Model Testing**
   - The real-time prediction feature allows users to input custom tumor features and receive immediate feedback. This makes the app an ideal tool for educational purposes and real-world usage scenarios.

### **3. Comprehensive Data Exploration**
   - Unlike most applications that focus solely on the model's output, this app emphasizes **data understanding**. The inclusion of dataset summaries, correlations, and feature visualizations ensures users can appreciate the broader context of machine learning in healthcare.

### **4. Performance Metrics Visualization**
   - The inclusion of multiple model evaluation metrics (like ROC and AUC) goes beyond basic accuracy and provides in-depth insights into model performance. This is crucial for making informed decisions about the reliability of the model.

### **5. Radar Chart Feature**
   - The innovative **Radar Chart** visualization adds a unique aspect to model testing. It not only provides users with a prediction but also shows how their inputs compare to average benign and malignant tumors, offering a more holistic view of the prediction.

## Installation

To run this project locally, follow these steps:

### **1. Clone the Repository:**

```bash
git clone https://github.com/hallelx2/breast-cancer-detection.git
cd breast-cancer-detection
```

### **2. Install Dependencies:**

```bash
pipenv shell
pipenv install
```

### **3. Run the App:**

```bash
streamlit run ui/app.py
```

The app will open in your default web browser.

## Example Screenshots

**1. Homepage:**
- A welcoming page introducing the project and the creator.

**2. Dataset Exploration:**
- A detailed preview of the dataset with feature distributions and correlation heatmap.

**3. Model Testing:**
- Real-time input form for tumor features and model prediction output with a radar chart.

**4. Model Metrics:**
- Performance evaluation with confusion matrix, classification report, and ROC curve.

## Contributing

Feel free to contribute to this project by submitting issues, suggestions, or pull requests. If you have ideas for new features or improvements, don't hesitate to reach out!

### **To Contribute:**

1. Fork the repository
2. Create a new branch (`git checkout -b feature-name`)
3. Make your changes
4. Commit your changes (`git commit -am 'Add new feature'`)
5. Push to the branch (`git push origin feature-name`)
6. Create a pull request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details
