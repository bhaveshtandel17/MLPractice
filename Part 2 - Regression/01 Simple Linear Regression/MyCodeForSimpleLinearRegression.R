# Simple linear regression

# Importing the dataset
dataset = read.csv('Salary_data.csv')

# Splitting the dataset into the Training set and Test set
# install.packages('caTools')
library(caTools)
set.seed(123)
split = sample.split(dataset$Salary, SplitRatio = 2/3)
training_set = subset(dataset, split == TRUE)
test_set = subset(dataset, split == FALSE)

# Feature Scaling
# We dont need to care about scaling because we are using regression library which handles scaling

# Fitting Simple Linear Regression to Training set 
# lm : linear model
regressor = lm(formula = Salary ~ YearsExperience,
               data = training_set)
# Salary ~ YearsExprience means Salary directlt progession to YearsExprience # Simpler liner formula
# If you want to see info type "summary(regressor)" in console
y_pred = predict(regressor, newdata = test_set) # to see type "y_pred" in console

#visualising the training set
# ggplot2 library to plot data
#install.packages('ggplot2') # to install # one time on one machine
library(ggplot2) # Or select from packges
# geom_point : plot observation / training points
# geom_line :  plot regression line
ggplot() + 
  geom_point(aes(x = training_set$YearsExperience, y = training_set$Salary),
             color = 'red') +
  geom_line(aes(x = training_set$YearsExperience, y = predict(regressor, newdata = training_set)),
            color = 'blue') +
  ggtitle('Salary vs Experience (Training set)') +
  xlab('Years of Experience') +
  ylab('Salary')

#visualising the Test set
ggplot() + 
  geom_point(aes(x = test_set$YearsExperience, y = test_set$Salary),
             color = 'red') +
  geom_line(aes(x = training_set$YearsExperience, y = predict(regressor, newdata = training_set)),
            color = 'blue') +
  ggtitle('Salary vs Experience (Test set)') +
  xlab('Years of Experience') +
  ylab('Salary')
