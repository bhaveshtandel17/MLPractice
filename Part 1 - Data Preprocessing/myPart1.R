
#Must DO Things


#import datasets
dataset = read.csv('data.csv')
#In R index starts from 1.

#Taking care of missing value
#is.na : missing value
#ifelse(condtion , if true value, if false value)
dataset$Age = ifelse(is.na(dataset$Age),
                     ave(dataset$Age, FUN = function(x) mean(x, na.rm = TRUE)),
                     dataset$Age)

dataset$Salary = ifelse(is.na(dataset$Salary),
                     ave(dataset$Salary, FUN = function(x) mean(x, na.rm = TRUE)),
                     dataset$Salary)
 
#Encoding categorical data
#In R, Encoding categorical data is simplnar than python (Using OneHardEncoder)
# Here we have factor fuction, Which transfer it into numberical categoeries

#Country columb encoding
# c is vector
dataset$Country  = factor(dataset$Country, 
                          levels = c('France', 'Spain', 'Germany'),
                          labels = c(1, 2, 3))

dataset$Purchased  = factor(dataset$Purchased, 
                          levels = c('No', 'Yes'),
                          labels = c(0, 1))



#Spliting the dataset into the Training set and Test set

#First import lib from packages : caTools
#install.packages('caTools') -- Uncommented after install lib
# select caTools from packages or use below command
#library(caTools)

set.seed(123) 
#Unlike python we have to set only dependending variable(i.e. Purchased)
#SplitRatio : train %
split = sample.split(dataset$Purchased, SplitRatio = 0.8)
# returns if observation choose for training set true otherwise false

trainig_set = subset(dataset, split == TRUE)
test_set = subset(dataset, split == FALSE)

# Feature Scaling : 
# Before wo did factor for Encoding categorical data, In R factor doesnt mean numerical data
# So we exclude Country and Purchased 
trainig_set[,2:3] = scale(trainig_set[,2:3])
test_set[,2:3] = scale(test_set[,2:3])




