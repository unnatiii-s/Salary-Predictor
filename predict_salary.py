# Import the libraries

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
import joblib 
# Loading the dataset into the dataframe
df = pd.read_csv("salary_data.csv")

#print(df.info())

# split the data into target variable and independent variables
X = df[["YearsExperience"]]
y = df[["Salary"]]

# Train-test split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.2,random_state = 1)

# Scaling down the data
# Creating an Object of StandardScaler Module present in Sklearn Library
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.fit_transform(X_test)

# Train the Model
model = LinearRegression()
model.fit(X_train_scaled,y_train)

# Save the model and Scaler
joblib.dump(model,"predict_salary.pkl")
joblib.dump(scaler,"scaler.pkl")
print("Model and Scaler are saved")