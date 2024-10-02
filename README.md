# House-Price-Prediction-Using-Linear-Regression
# Project Overview
This project implements a linear regression model to predict house prices based on features from the Boston Housing Dataset. The goal is to train the model to predict the price of a house given various features such as the number of rooms, crime rate, and more. This project serves as a demonstration of machine learning fundamentals using Python and the Scikit-learn library.

# Dataset
The dataset used in this project is the Boston Housing Dataset, which includes the following features:

# CRIM: Crime rate by town
# ZN: Proportion of residential land zoned for lots over 25,000 sq. ft.
# INDUS: Proportion of non-retail business acres per town
# CHAS: Charles River dummy variable (1 if tract bounds river, 0 otherwise)
# NOX: Nitric oxides concentration
# RM: Average number of rooms per dwelling
# AGE: Proportion of owner-occupied units built prior to 1940
# DIS: Distances to five Boston employment centers
# RAD: Index of accessibility to radial highways
# TAX: Full-value property-tax rate per $10,000
# PTRATIO: Pupil-teacher ratio by town
# B: 1000(Bk - 0.63)^2, where Bk is the proportion of Black people by town
# LSTAT: Percentage of lower status population
# MEDV: Median value of owner-occupied homes (in $1000s, the target variable)
# Project Workflow
# Data Preprocessing:

# Load the dataset.
Split the data into features (X) and target (y).
Divide the dataset into training and testing sets.
# Model Building:

Initialize the Linear Regression model from Scikit-learn.
Train the model using the training data.
Model Evaluation:

Use Mean Squared Error (MSE) and R-squared values to evaluate the model's performance on the test data.
Visualize the predicted vs actual house prices.
Making Predictions:

The model can predict house prices based on user input for various features.
Installation & Setup
To run this project locally, follow these steps:

# Prerequisites
Make sure you have the following libraries installed:

Python 3.x
NumPy
Pandas
Scikit-learn
Matplotlib
You can install these dependencies using pip:
pip install numpy pandas scikit-learn matplotlib
# Running the Project
1. Clone this repository
   git clone https://github.com/your-username/House-Price-Prediction-Using-Linear-Regression.git
2. Open the notebook or script.
3. Run the script in your preferred Python environment, such as Jupyter Notebook, Google Colab, or any Python IDE.
Results
# The linear regression model achieved the following performance metrics:

Mean Squared Error (MSE): X.XX
R-squared (R²) Value: 0.XX
The model provides a reasonable prediction for house prices based on the input features.
