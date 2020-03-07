library(tidyverse)

# 1. Define Problem.

# Your goal is to predict the total_cases label for each (city, year, weekofyear) in the test set. There are two cities, San Juan and Iquitos, with test data for each city spanning 5 and 3 years respectively. You will make one submission that contains predictions for both cities. 

# 2. Prepare Data.

# Create train dataset with labels
train <- left_join(dengue_features_train, dengue_labels_train, by = c("city", "year", "weekofyear"))

# Store X and Y for later use.
x = train[, 5:24] 
y = train$total_cases

# 2.1 Descriptive Statistics

# Dimensions of Dataset
dim(train)

# list types for each attribute
sapply(train, class)

# Look at summary of dataset, the labels are all filled out but there are some NAs in other variables
summary(train, digits = 1)

# Look at distribution of labels, 75% of the data is under 28 cases per day. There could be outliers.
summary(train$total_cases)

# 2.2 Missing values

# Create the knn imputation model on the training data
library(caret) # required for preProcess
preProcess_missingdata_model <- preProcess(train, method='knnImpute')
preProcess_missingdata_model

# Use the imputation model to predict the values of missing data points. 
library(RANN)  # required for knnInpute
trainData <- predict(preProcess_missingdata_model, newdata = train)
anyNA(trainData) # no missing values identified

# 2.3 Transform data

# 2.3.1 Dummy Variables (One-hot encoding)
# Creating dummy variables is converting a categorical variable to as many binary variables as here are categories.
trainData <- subset(trainData,select = -c(week_start_date))
dummies_model <- dummyVars(total_cases ~ ., data=trainData)

# Create the dummy variables using predict. The Y variable will not be present in trainData_mat.
trainData_mat <- predict(dummies_model, newdata = trainData)

# # Convert to dataframe
trainData <- data.frame(trainData_mat)

# # See the structure of the new dataset
str(trainData)

# 2.3.2 Normalization
# Normalization is applied because ranges of variables can differ significantly and thus influence in the prediction.
preProcess_range_model <- preProcess(trainData, method='range')
trainData <- predict(preProcess_range_model, newdata = trainData)

# Append the Y variable
trainData$total_cases <- as.numeric(y)

# See if dataset was normalized
apply(trainData[, 1:24], 2, FUN=function(x){c('min'=min(x), 'max'=max(x))}) # !!! Customize based on number o variables

# 3. Modeling

# Now that the preprocessing is complete, let's visually examine how the predictors influence the Y.

# 3.1 Boxplot for each attribute on one image
par(mfrow=c(1,4))
for(i in 5:8) {
  boxplot(x[,i], main=names(trainData)[i])
}
par(mfrow=c(1,4))
for(i in 9:12) {
  boxplot(x[,i], main=names(trainData)[i])
}
par(mfrow=c(1,4))
for(i in 13:16) {
  boxplot(x[,i], main=names(trainData)[i])
}
par(mfrow=c(1,4))
for(i in 17:20) {
  boxplot(x[,i], main=names(trainData)[i])
}
par(mfrow=c(1,4))
for(i in 21:24) {
  boxplot(x[,i], main=names(trainData)[i])
}

# barplot for class breakdown
par(mfrow=c(1,1))
plot(y)

# 3.2 Correlation plot - Check, did not work well
library(corrplot)
M<-cor(trainData[,5:25])
head(round(M,2))
corrplot(M, method="color")

# Data can contain attributes that are highly correlated with each other. Many methods perform better if highly correlated attributes are removed.

# ensure the results are repeatable
set.seed(7)
# calculate correlation matrix
correlationMatrix <- cor(trainData[,5:24])
# summarize the correlation matrix
print(correlationMatrix)
# find attributes that are highly corrected (ideally >0.75)
highlyCorrelated <- findCorrelation(correlationMatrix, cutoff=0.75)
# print indexes of highly correlated attributes
print(highlyCorrelated)


# 4 Training and Tuning the model

# 4.1 Build models

# Setting up the trainControl()
# Inside trainControl() you can control how the train() will:
# Cross validation method to use.

# Define the training control
control <- trainControl(method="cv", number=10)

# Train the model and predict on the training data itself.

## Linear regression, since use of full data generates error, second model removes highly correlated variables
#set.seed(100) # Set the seed for reproducibility
#fit.lm <- train(total_cases~., data=trainData, method="lm", trControl=control)
set.seed(100) # Set the seed for reproducibility
fit.lm2 <- train(total_cases~., data=trainData[,-highlyCorrelated], method="lm", trControl=control)

## Random forest, used also both sets of data, with and without highly correlated variables
set.seed(100) # Set the seed for reproducibility
fit.rf <- train(total_cases~., data=trainData, method="rf", trControl=control)
set.seed(100) # Set the seed for reproducibility
fit.rf2 <- train(total_cases~., data=trainData[,-highlyCorrelated], method="rf", trControl=control)

## Support vector machine, , since use of full data generates error, second model removes highly correlated variables
#set.seed(100) # Set the seed for reproducibility
#fit.svm <- train(total_cases~., data=trainData, method="svmRadial", trControl=control)
set.seed(100) # Set the seed for reproducibility
fit.svm2 <- train(total_cases~., data=trainData[,-highlyCorrelated], method="svmRadial", trControl=control)

## K-nearest neighbor, used also both sets of data, with and without highly correlated variables
set.seed(100) # Set the seed for reproducibility
fit.knn <- train(total_cases~., data=trainData, method="knn", trControl=control)
set.seed(100) # Set the seed for reproducibility
fit.knn2 <- train(total_cases~., data=trainData[,-highlyCorrelated], method="knn", trControl=control)

## Gradient b0osting (similar to random forest), used also both sets of data, with and without highly correlated variables
set.seed(100) # Set the seed for reproducibility
fit.gbm <- train(total_cases~., data=trainData, method="gbm", trControl=control)
set.seed(100) # Set the seed for reproducibility
fit.gbm2 <- train(total_cases~., data=trainData[,-highlyCorrelated], method="gbm", trControl=control)

# Summarize accuracy of models
results <- resamples(list(lm2=fit.lm2, rf=fit.rf, rf2 = fit.rf2, svm2 = fit.svm2, knn=fit.knn, knn2 = fit.knn2, gbm = fit.gbm, gbm2 = fit.gbm2))

# compare accuracy of models. 
summary(results)
dotplot(results)

# compare two best models that have the lowest RMSE and highest R-squared. At a 95% confidence level, the rf model is statistically the same as the gbm. Since the random forest model is simpler than gradient boosting, we will consider this as the best model.
compare_models(fit.gbm, fit.rf)

# Tunning parameters for best algorithm. Final model selected with mtry of 22.
set.seed(100) # Set the seed for reproducibility
finalModel <- train(total_cases~., data=trainData, method="rf", trControl=control, tuneLength  = 15)


#5. Present Results.

# Prepare test data

## Step 1: Impute missing values 
preProcess_missingdata_model <- preProcess(dengue_features_test, method='knnImpute')
testData <- predict(preProcess_missingdata_model, newdata = dengue_features_test)
anyNA(testData)

## Step 2: Create one-hot encodings (dummy variables)
dummies_model2 <- dummyVars(~ ., data=testData)
testData2 <- predict(dummies_model2, testData)

### Convert to dataframe
testData2 <- data.frame(testData2)

## Step 3: Transform the features to range between 0 and 1
testData3 <- predict(preProcess_range_model, testData2)

# Make predictions
predictions <- predict(finalModel, testData3)
head(predictions)

# Output
testDataFinal <- cbind(dengue_features_test[,1:3], total_cases = round(predictions))
write.csv(testDataFinal,"Submission.csv", row.names = FALSE)

# Compare results with train data
trainTemp <- cbind(train, type = "train")
testTemp <- cbind(dengue_features_test, total_cases = round(predictions), type = "test")
Temp <- rbind(trainTemp, testTemp)
  
ggplot(Temp, aes(weekofyear, total_cases, group = as.factor(year), color = as.factor(year))) +
  geom_line() + 
  facet_grid(rows = vars(as.factor(city),as.factor(type))) +
  theme(legend.position = "none") +
  ggtitle("Total cases in training and test data")


