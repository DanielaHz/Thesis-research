#MULTIVARIABLE LOGISTIC REGRESSION APPLIED TO THE PREDICTING OF THE RELEASE DATE OF COLOMBIAN FILMS  

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# Opening the dataset
FILE = 'Dataset_releasedate.csv'
DATA = pd.read_csv(FILE, engine= 'python', index_col=0)

# ------------------------ DATA PREPROCESSING PART -----------------------------------#

"""
In this part was converted the original qualitative variables to dummies. This is an essential 
part of the process to apply classification models.
"""
# Converting qualitative variables to dummies

Genre_dummy = pd.get_dummies(DATA["Genre"], prefix = "Genre")
Age_rating_dummy = pd.get_dummies(DATA["Age_raiting"], prefix = "Age_raiting")
Director_dummy = pd.get_dummies(DATA["Director"], prefix = "Director")
Producer_dummy = pd.get_dummies(DATA["Producer"], prefix = "Producer")
Sequel_dummy = pd.get_dummies(DATA["Sequel"], prefix = "Sequel")
Distributor_dummy = pd.get_dummies(DATA["Distributor"], prefix = "Distributor")
Screens_dummy = pd.get_dummies(DATA["Screens"], prefix = "Screens")
Release_date_dummy = pd.get_dummies(DATA["Release_date"], prefix = "Release_date")


# Concatenating the dummy variables with the original file

DATA = pd.concat([DATA, Genre_dummy] , axis=1)
DATA = pd.concat([DATA, Age_rating_dummy] , axis=1)
DATA = pd.concat([DATA, Director_dummy] , axis=1)
DATA = pd.concat([DATA, Producer_dummy] , axis=1)
DATA = pd.concat([DATA, Sequel_dummy] , axis=1)
DATA = pd.concat([DATA, Distributor_dummy] , axis=1)
DATA = pd.concat([DATA, Screens_dummy] , axis=1)
DATA = pd.concat([DATA, Release_date_dummy] , axis=1)

# Deleting the original columns (No dummy variables) and creating our DATA_dummies

DATA_dummies = DATA.drop(['Genre', 'Age_raiting', 'Producer','Director', 'Sequel', 'Distributor', 'Screens', 'Release_date'], axis = 1)

# ------------------------ VARIABLE ASIGNATION PART -----------------------------------#

"""
Here we assign the variables of the DATA_dummies to the variables X and y.
We now have around 180 columns after creating the DATA_dummies, so we need to assign using the columm 
range.
"""

# Checink the column range

print(DATA_dummies.columns) 
# -> length = 188, the last concatenation that we did were the release date and those are 12 columns,
# so 188 - 12 = 176.

#Asignation of dependent(y) and independment (X) variables.

X= DATA_dummies.iloc[:,0:177].values # Converting the list to a pd vector 

y = DATA_dummies.iloc[:,177:189].values # Converting the list to a pd vector 

# Split data into training and test sets

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05, random_state=42)

# Theta's inicialization 
theta = np.zeros(X_train.shape[1])

# Hiperparmaters
alpha = 0.01
num_iterations = 1000

# ------------------------ MATHEMATICAL FUNCTIONS PART -----------------------------------#

def sigmoid_function(z):
    """
    Return the value of the aplication of a sigmoid function.
    Use: The sigmoid function then takes the linear combination (z) as input and maps it to
    a value between 0 and 1, representing the estimated probability of the event occurring.
    """
    return 1/(1 + np.exp(-z))

def cost_function(X, y, theta):
    """
    Return the value of the cost_function
    Use: The cost function is used to measure the error or mismatch between the predicted 
    probabilities and the actual class labels. The goal is to minimize this cost function 
    during the model training process.
    """
    m = len(y)
    h = sigmoid_function(X.dot(theta))
    cost = (~y * np.log(h) - (1 - y) * np.log(1 - h))
    cost = np.mean(cost)
    return cost

def gradient_descent_function(X, y, theta, alpha, num_iteration):
    """
    Return the value of the gradient_descent_function.
    Use: It's an algorithm used to minimize the cost function.
    """
    m = len(y)
    n = X.shape[1]
    theta = np.zeros((n, y.shape[1]))

    costs = []
    for i in range(num_iteration):
        h = sigmoid_function(X.dot(theta))
        error = h - y
        gradient = X.T.dot(error) / m
        theta -= alpha * gradient
        cost = cost_function(X, y, theta)
        costs.append(cost)
    return theta, costs

def class_prediction(X, theta, threshold = 0.5):
    """
    Return the value of the class_prediction.
    Use: It's used for classifiyng the probability equal or greater that 0.5.
    A positive class was labeled as 1 and the opossite ladaled as 0.
    """
    h = sigmoid_function(X.dot(theta))
    predictions = (h >= threshold).astype(int)
    return predictions

# ------------------------ MODEL TRAINING PART -----------------------------------#

# Training model 
theta, costs = gradient_descent_function(X_train, y_train, theta, alpha, num_iterations)

y_pred = class_prediction(X_test, theta)

# ------------------------ EVALUATION MODELO PART -----------------------------------#

from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test, y_pred)
print("Precition:", accuracy)
