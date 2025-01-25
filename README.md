# Urban Air Quality Prediction Using Deep Learning

## Project Overview

This project focuses on the prediction of urban air quality, specifically nitrogen dioxide (NO2) levels, using advanced deep learning techniques. By leveraging Long Short-Term Memory (LSTM) networks with an attention mechanism, the system aims to provide accurate and interpretable forecasts of pollutant concentrations. The model utilizes historical air quality data from Los Angeles and is designed to aid city planners and public health officials in proactive decision-making.

---

## Features

- **Dynamic Prediction**: Predicts daily NO2 concentrations with enhanced accuracy using LSTM networks.
- **Temporal Insight**: Integrates an attention mechanism to identify key temporal patterns influencing pollutant levels.
- **Data Visualization**: Provides clear visualizations of actual vs. predicted NO2 concentrations for interpretability.
- **Performance Metrics**: Evaluates predictions using robust metrics such as Mean Absolute Error (MAE) and Mean Squared Error (MSE).
- **Real-Time Deployment**: Includes a Flask API for real-time predictions and a Google Colab deployment for ease of use.

---

## Objectives

1. Develop a deep learning model to predict daily NO2 concentrations dynamically.
2. Enhance model interpretability through attention mechanisms.
3. Provide actionable insights for urban planners and public health officials.
4. Ensure robust evaluation using metrics like MAE and MSE.

---

## Technologies Used

### Frameworks and Libraries
- **Deep Learning**: TensorFlow, Keras
- **Data Processing**: Pandas, NumPy
- **Visualization**: Matplotlib
- **Hyperparameter Tuning**: Keras Tuner (Hyperband algorithm)

### Model Architecture
- **Base Model**: Long Short-Term Memory (LSTM) network
- **Enhancements**: Attention mechanism for improved interpretability
- **Training**: Adam optimizer, Min-Max scaling, Early stopping

### Deployment
- **Platform**: Google Colab (model execution)
- **API**: Flask (for user interaction and real-time predictions)

---

## Dataset and Preprocessing

- **Data Source**: Historical NO2 concentration data from Los Angeles.
- **Preprocessing**:
  - Missing values handled with imputation techniques.
  - Data normalized using Min-Max scaling.
  - Time-series sequences created with 7-day intervals.
- **Train-Test Split**: 80:20 ratio.

---

## Training Details

- **Hyperparameters**:
  - LSTM units: 64
  - Dropout rate: 0.2
  - Batch size: 32
  - Learning rate: 0.001
  - Epochs: 50
- **Optimizer**: Adam
- **Loss Function**: Mean Squared Error (MSE)
- **Regularization**: Dropout and early stopping
- **Evaluation Metrics**: MAE = 0.1498, MSE = 0.0313

---

## Deployment Instructions

1. **Google Colab Setup**:
   - Upload the trained model in TensorFlow’s SavedModel format.
   - Run the provided Colab notebook for real-time predictions.

2. **Flask API Setup**:
   - Install Flask (`pip install flask`).
   - Use the provided `app.py` to start the API locally.
   - Send HTTP POST requests with historical NO2 data to receive predictions.

---

## Challenges and Solutions

### Challenges
- **Data Variability**: Seasonal and temporal fluctuations in NO2 levels.
- **Model Optimization**: Balancing accuracy and computational efficiency.
- **Interpretability**: Visualizing attention weights to identify key temporal patterns.

### Solutions
- Imputation techniques for missing values and Min-Max scaling for normalization.
- Keras Tuner for optimal hyperparameter selection.
- Attention heatmaps for enhanced model interpretability.

---

## Future Work

1. **Data Expansion**: Include additional features like weather conditions and traffic data.
2. **Cloud Deployment**: Deploy the system on platforms like AWS or Google Cloud.
3. **Extended Applications**: Adapt the model for other pollutants (e.g., PM2.5, PM10).
4. **Real-Time Forecasting**: Integrate continuous data pipelines for live predictions.

---

## How to Use

1. Clone the repository.
2. Set up the environment with the required dependencies.
3. Run the Google Colab notebook for model execution or the Flask API for local hosting.
4. Input historical NO2 data for real-time predictions.

---

## Contact Information

- **Author**: Atharva Talegaonkar  
- **Email**: atale014@ucr.edu
