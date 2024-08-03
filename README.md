# Stroke Prediction App

This repository contains a Stroke Prediction application built using Streamlit. The application predicts the risk of stroke based on various health and demographic factors. The prediction model is based on the K-Nearest Neighbors (KNN) algorithm.

## Overview
The Stroke Prediction App utilizes a K-Nearest Neighbors (KNN) model to predict whether a person is at risk of a stroke. It takes into account factors such as age, gender, hypertension, heart disease, marital status, work type, residence type, average glucose level, BMI, and smoking status.

## Model Training
The models were trained using the following process:

#### **Data Preprocessing:**

- Loaded the dataset and performed exploratory data analysis.
- Handled missing values and performed one-hot encoding for categorical variables.
#### **Feature Scaling:**

- Standardized the numerical features using StandardScaler.
- Model Training:

#### **Split the dataset into training and test sets.**
- Trained a K-Nearest Neighbors model on the training set.
- Evaluated the model's performance using accuracy, precision, recall, and F1-score.
- **Model Accuracy** : 95%
#### **Saving the Model:**

- The trained KNN model and scaler are saved using pickle for use in the Streamlit app.
  
