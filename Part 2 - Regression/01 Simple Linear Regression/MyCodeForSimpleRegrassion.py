# Simple Linear regrassion is y = b0 + b1 * x1

# Regrassion : regression analysis is a set of statistical processes for estimating the relationships among variables.
# It includes many techniques for modeling and analyzing several variables
# when the focus is on the relationship between a dependent variable(output) and one or more independent variables(input)
# regression analysis helps one understand how the typical value of the dependent variable changes when any one of the independent variables is varied.

# linear regression:  is a linear approach to modelling the relationship between a scalar response (or dependent variable) 
# and one or more explanatory variables (or independent variables). 

# The case of one independent variable is called simple linear regression.
# For more than one independent variable, the process is called multiple linear regression.

# Linear regression: Y = a + bX + u
# Multiple regression: Y = a + b1X1 + b2X2 + b3X3 + ... + btXt + u
# Where:
# Y = the variable that you are trying to predict (dependent variable).
# X = the variable that you are using to predict Y (independent variable).
# a = the intercept. (Constant)
# b = the slope. (Coefficient)
# u = the regression residual.

# https://www.khanacademy.org/math/statistics-probability/describing-relationships-quantitative-data/more-on-regression/v/regression-line-example

#In Our case : Salary = a + b * Exprience
# What is constant (i.e. a) point where liner line cross vartical axis (i.e. Exprience = 0 : X-axis)
# What is Coefficient (i.e. b) is slope of liner line, If slope is less that menas salary will increase less vice versa.

# what is Best Fitting Line?
# finding the best-fitting straight line through the points. The best-fitting line is called a regression line.
# the regression line and consists of the predicted score on Y for each possible value of X.
# The vertical lines from the points to the regression line represent the errors of prediction.
# The error of prediction for a point is the value of the point minus the predicted value (the value on the line).
# error of prediction = (Y-Y')
# most commonly-used criterion for the best-fitting line is the line that minimizes the sum of the squared errors of prediction. i.e. (Y-Y')2
# In m/n leanrning, finds the line with less error of prediction (Sum of (Y-Y')2)
#                                                              y2-y1
# then find Scope / Coefficient (a) using best fitting line = -------- 
#                                                              x2-x1
# Use formala 
# Y = a + bX ; a = best fitting line touches on x-axis where y-axis's value

# http://onlinestatbook.com/2/regression/intro.html
# https://www.youtube.com/watch?v=gtXAC3DQxC8

#==============================================================================#

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Salary_Data.csv')
X = dataset.iloc[:, :-1].values # independent variable ; Exprience
y = dataset.iloc[:, 1].values # dependent variable ; Salary

# Splitting the dataset into the Training set and Test set
# IMP: In all time, Always divide dataset from train to test
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size = 1/3, random_state = 0)

# Feature Scaling
# Dont need to do Feature Scaling because in simple liner regression library will take care of scaling.

#Fitting Simple Liner Regression to the training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train, y_train) # fit train set to LinearRegression

#Predicating the Test set results
y_pred = regressor.predict(x_test) # Predicated salary

# Visualising the Training set results
# To plot we will use matplotlib library
plt.scatter(x_train, y_train, color = 'red') # Plot train data
plt.plot(x_train, regressor.predict(x_train), color = 'blue') # Plot regrassion line : best fit line : henece y-axis is regressor.predict(x_train)
plt.title('Salary vs Expreience (Trainig set)')
plt.xlabel('Years of Expreience')
plt.ylabel('Salary')
plt.show()

# regresssion line (blue line) is predicated salary
# Red dot is actual salary

# Visualising the Test set results
plt.scatter(x_test, y_test, color = 'red') # Plot train data
plt.plot(x_train, regressor.predict(x_train), color = 'blue') # Same regession line 
plt.title('Salary vs Expreience (Test set)')
plt.xlabel('Years of Expreience')
plt.ylabel('Salary')
plt.show()
