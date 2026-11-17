# ⚡ Power Plant Predict
### A Comparative Analysis of Machine Learning and Deep Learning Models for Electrical Power Output Prediction

## 📌 Project Overview

Power Plant Predict is a web-based machine learning application developed using Streamlit. It predicts the electrical power output of a Combined Cycle Power Plant (CCPP) based on environmental conditions.
The main purpose of this project is to compare different Machine Learning and Deep Learning regression models, evaluate their performance, and identify the best-performing model for electrical power output prediction.
Users can select multiple models, compare their performance, and use the selected trained model to predict power output using unseen input data.

## 🎯 Project Objectives

- Predict electrical power output using environmental input features.
- Compare Machine Learning and Deep Learning regression models.
- Evaluate model performance using different evaluation metrics.
- Identify the best-performing model.
- Allow users to test predictions with unseen input values through a web interface.

## 📊 Dataset

**Dataset Name:** Combined Cycle Power Plant (CCPP)
**Source:** UCI Machine Learning Repository
**Original Dataset:** 9,568 records
**Cleaned Dataset:** 9,527 records
**Target Variable:** PE (Net Electrical Energy Output)

### Dataset Features

| Feature | Description | Unit |
|---|---|---|
| AT | Ambient Temperature | °C |
| V | Exhaust Vacuum | cm Hg |
| AP | Ambient Pressure | millibar |
| RH | Relative Humidity | % |
| PE | Net Electrical Energy Output | MW |

The dataset contains four input features (AT, V, AP, RH) and one target variable (PE).

Dataset Source: https://archive.ics.uci.edu/dataset/294/combined+cycle+power+plant

## 🤖 Machine Learning Models

This application provides four regression models for comparison:

1. **Linear Regression** – A basic regression model that estimates the relationship between input features and power output.
2. **Decision Tree Regressor** – A tree-based model that learns decision rules from the dataset.
3. **Random Forest Regressor** – An ensemble learning model that combines multiple decision trees.
4. **Artificial Neural Network (ANN)** – A deep learning model that learns complex patterns from input data.

## 📈 Model Evaluation Metrics

The models are evaluated using three performance metrics:

- **R² Score:** Measures how well the model explains the variation in the target variable. Higher values are better.
- **RMSE (Root Mean Squared Error):** Measures the average prediction error while penalizing larger errors. Lower values are better.
- **MAE (Mean Absolute Error):** Measures the average absolute difference between actual and predicted values. Lower values are better.

The best model is selected based on the lowest validation RMSE.

## ✨ Key Features

- Interactive Streamlit web application.
- Selection of at least two regression models for comparison.
- Model performance comparison using evaluation metrics.
- Model comparison visualization.
- Automatic identification of the best-performing model.
- Unseen data prediction using manually entered input values.
- Green-themed user interface.

## 🛠️ Technologies Used

- Python
- Streamlit
- Pandas
- NumPy
- Scikit-learn
- TensorFlow / Keras
- Matplotlib
- Seaborn

## 📂 Project Structure
```text
Power-Plant-Prediction/
│
├── app.py
├── requirements.txt
│
├── data/
│   └── power_plant_cleaned.csv
│
└── .streamlit/
    └── config.toml
```

## ⚙️ Installation and Setup
### 1. Clone the Repository
```bash
git clone https://github.com/nwenisan539-spec/Power-Plant-Prediction.git
```

### 2. Navigate to the Project Folder
```bash
cd Power-Plant-Prediction
```

### 3. Create a Virtual Environment
```bash
python -m venv .venv
```

### 4. Activate the Virtual Environment
For Windows PowerShell:
```powershell
.venv\Scripts\Activate.ps1
```

### 5. Install Required Libraries
```bash
pip install -r requirements.txt
```

### 6. Run the Application

```bash
streamlit run app.py
```

The application will open in your web browser.

## 🚀 How to Use

1. Open the PowerPredict AI web application.
2. View the dataset information.
3. Select at least two regression models or all.
4. Click the Train and Compare Models button.
5. Review the model comparison results and evaluation metrics.
6. Identify the best-performing model.
7. Select a trained model for prediction.
8. Enter unseen input values for AT, V, AP, and RH.
9. Click the Predict button to obtain the estimated electrical power output in MW.

## 🌐 Live Application

[PowerPredict AI – Streamlit Web App](https://power-plant-prediction-cjkjw6ptbmki2lc5brxiby.streamlit.app/)
