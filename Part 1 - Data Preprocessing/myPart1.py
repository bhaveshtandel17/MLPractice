#Must DO Things

#=========================================================================#
#import the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#=========================================================================#
#import the datasets
dataset = pd.read_csv('Data.csv')
#Feature
x = dataset.iloc[:, :-1].values
#Depended
y = dataset.iloc[:, 3].values

#=========================================================================#
#taking care of missing data
#missing data is replaced by mean-value i.e. avg
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values = np.nan, strategy = "mean")
imputer = imputer.fit(x[:,1:3])
x[:,1:3] = imputer.transform(x[:,1:3]) 

#=========================================================================#
# Encoding categorical Data. (Text - Number because we nned number for maths in m/n learning)
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
#Encode country code in 0, 1, 2 ....
x[:, 0] = labelencoder_X.fit_transform(x[:, 0])# Country coloum is zero
# Now X first cloumn is encoded by 0, 1, 2 
#Before
#[France' 44.0 72000.0]
# ['Spain' 27.0 48000.0]
# ['Germany' 30.0 54000.0]
# ['Spain' 38.0 61000.0]
# ['Germany' 40.0 63777.77777777778]
# ['France' 35.0 58000.0]
# ['Spain' 38.77777777777778 52000.0]
# ['France' 48.0 79000.0]
# ['Germany' 50.0 83000.0]
# ['France' 37.0 67000.0]]
#After Encode 
#[[0 44.0 72000.0]
# [2 27.0 48000.0]
# [1 30.0 54000.0]
# [2 38.0 61000.0]
# [1 40.0 63777.77777777778]
# [0 35.0 58000.0]
# [2 38.77777777777778 52000.0]
# [0 48.0 79000.0]
# [1 50.0 83000.0]
# [0 37.0 67000.0]]

# But Problem is : M/n leaning model are based on equestions, Since
#So we encode categorical data and can include it in equestions.
# However, Since 0 is greater than zero and two is greater than one.
#the equestion in models thinks that spain(2)has higher value than Germany(1) and France(0)
#but that not a case, thoses are only simple category and not realtional order b/w them

# if we have variable like size : large, small, medium. That case we can order that in size (0,1,2)
#Large is greater that meduim and small

# so we have to pretend m/n learning equation to thinking THAT spain(2)has greater than Germany(1) and France(0)
#For we use DUMMY VARIABLE : Insteand of having one coulumn we can have number of coulmn equals to number of category
#We have three categories(spain, Germany and France) so we will have new 3 coulmb
# each of three coulmb coresponding to one country

# *Country*                   *France*  *Germany*  *Spain*
#France                         1          0         0
#Germany                        0          1         0
#Spain                          0          0         1

# OneHotEncoder : will help in that.

onehotencoder = OneHotEncoder(categorical_features = [0])
x = onehotencoder.fit_transform(x).toarray()

# Purchase (Categry) variable encoded.
labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)

#=========================================================================#
#Spliting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split  
#x_train : traing part of matrix of features
#x_test : TEST part of matrix of features
#y_train : traing part of matrix of depending variable of x_train
#y_test : TEST part of matrix of depending variable of x_test
 #initial declare all 4 at time
 #20% of data for test from datasetc (should have smaler than train %)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)

#=========================================================================#
# Feature Scaling : 
#Why? : We have two age and salary that contains numberical numbers, You notice that
#AGE and SALARY dont have in same scale because age is in b/w (27-50)
#Salary is b/w (48000 - 83000). This will cause issue in m/n model.
#Becoz lots of m/n leanining modele based on "Euclidean Distance"
#                                      ___________________________
# Euclidean Distance b/w two point is √((x2 - x1)^2 + (y2 - y1)^2)
# Assume Age is x coordinate and Salary is y coordinate.
#Because age and salary doesn't have in same scale .
#So Salary is "dominating" over age in m/n leanining equation.
# Several Way to Scale data : STANDARDISATION & NORMALISATION
# STANDARDISATION:
#               x - mean(x)
# x(stand) = ---------------------
#            Standard deviation(x)

# NORMALISATION: 
#               x - min(x)
# x(norm) = ---------------------
#             max(x) - min(x)

# Use StandardScaler lib

from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
x_train = sc_x.fit_transform(x_train)
#Dont need to fit sc_x to x_test, beacuse it alerady fit with train set. 
#(x-test and x-train scale on same bases)
x_test = sc_x.transform(x_test)
# Do we need to scale DUMMY VARIABLE(Encoded Categororical data)?
#Its depent on context, Here, If we do scale Dummy variable scale, 
#we will loose interpation  knowing which obervation below which country.
# But we do scale dummy variable for this example to understand concept of scale.
# Afer scale all data is in b/w -1 to 3
# If algo doest depend of Euclidean Distance still we should scale because of performance.

 










