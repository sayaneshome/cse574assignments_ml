#set the path to working directory containing the files
setwd('coms573_lab4/')
install.packages('randomForest')
install.packages('data.table')
install.packages('caretEnsemble')
install.packages('caret')
library(randomForest)
library(caret)
library(caretEnsemble)
library(data.table)
#include all output in .txt file
sink(file = 'Output_final_April18_2019.txt', append = FALSE, type = c("output","message"),split = FALSE)
#Data-reading and processing
train <- read.csv('lab4-train.csv')
test <- read.csv('lab4-test.csv')
train$Class[train$Class == "0"] <- "N"
train$Class[train$Class == "1"] <- "Y"
test$Class[test$Class == "1"] <- "Y"
test$Class[test$Class == "0"] <- "N"
names(train)<-c("R", "F", "M", "T","Class")
names(test)<-c("R", "F", "M", "T","Class")
predictors<-c("R", "F", "M", "T")
outcomeName<-'Class'

print("----------TASK 1----------")
print("Random Forest Classifier:")

#Function for RandomTree Classifier
MyRFclassifier <- function(num,num1){
  set.seed(200)
  fitControl <- trainControl(
    method = "repeatedcv",
    number = 10,
    repeats = 10)
  model_rf<-train(train[,predictors],train[,outcomeName],method='rf',trControl=fitControl,tuneLength=num,ntree=num1)
  model_rf
  test$pred_rf<-predict(object = model_rf,test[,predictors])
  confusionMatrix(as.factor(test$Class),as.factor(test$pred_rf))
}

MyRFclassifier(2,50)
MyRFclassifier(2,100)
MyRFclassifier(2,175)
MyRFclassifier(2,200)
MyRFclassifier(2,250)
MyRFclassifier(3,200)
MyRFclassifier(3,250)

#Function for Adaboost Classifier
set.seed(200)
control <- trainControl(method = "repeatedcv", number = 10, repeats = 3, search = "grid", savePredictions = "final", classProbs = TRUE, verboseIter = TRUE)
model_adaboost = train(train[,predictors],train[,outcomeName],method = "adaboost",trControl = control)
test$pred_ada1<-predict(object = model_adaboost,test[,predictors])
confusionMatrix(as.factor(test$Class),as.factor(test$pred_ada1))

print("----------TASK 2----------")
print("Ensemble Classifier:")

#10-fold crossvalidation
fitControl <- trainControl(method = "cv",number = 10,savePredictions = TRUE)
#Function for Linear Regression
model_lreg<-train(train[,predictors],train[,outcomeName],method="glm",family=binomial(),trControl=fitControl)
test$pred_lreg<-predict(object = model_lreg,test[,predictors])
print('Results for Logistic Regression')
model_lreg
confusionMatrix(as.factor(test$Class),as.factor(test$pred_lreg))

#Function for decision tree classifier
model_dtree<-train(train[,predictors],train[,outcomeName],method="ctree",trControl=fitControl)
test$pred_dtree<-predict(object = model_dtree,test[,predictors])
print('Results for Decision tree classifier')
model_dtree
confusionMatrix(as.factor(test$Class),as.factor(test$pred_dtree))

#Function for KNN
My_KNN <- function(num){
model_knn<-train(train[,predictors],train[,outcomeName],method="knn",trControl=fitControl,tuneLength=num)
test$pred_knn<-predict(object = model_knn,test[,predictors])
confusionMatrix(as.factor(test$Class),as.factor(test$pred_knn))
}
print('Results for KNN models')
My_KNN(2)
My_KNN(8)

#Function for Naive Bayes classifier
print('Results for NB Classifier')
model_nb<-train(train[,predictors],train[,outcomeName],method="nb",trControl=fitControl,tuneLength=num)
test$pred_nb<-predict(object = model_nb,test[,predictors])
model_nb
confusionMatrix(as.factor(test$Class),as.factor(test$pred_nb))

#Function for Neural network
My_NN <- function(num){
model_nn<-train(train[,predictors],train[,outcomeName],method='nnet',trControl=fitControl,tuneLength=num)
test$pred_nn<-predict(object = model_nn,test[,predictors])
model_nn
confusionMatrix(as.factor(test$Class),as.factor(test$pred_nn))
}
print('Results for Neural Networks')
My_NN(3)
My_NN(4)

#Function for classification error
classification_error <- function(conf_mat) {
  conf_mat = as.matrix(conf_mat)
  error = 1 - sum(diag(conf_mat)) / sum(conf_mat)
  return (error)
}
#Ensemble creation of all the five discussed in Task 2-a

control <- trainControl(method = "repeatedcv", number = 10, repeats = 3, search = "grid", savePredictions = "final", classProbs = TRUE, verboseIter = TRUE)
# List of algorithms to use in ensemble
alg_list <- c("glm", "ctree", "nb", "nnet", "knn")
multi_mod <- caretList(train[,predictors],train[,outcomeName], trControl = control, methodList = alg_list, metric = "Accuracy")
res <- resamples(multi_mod)
summary(res)
#unweighted classifier based on ensemble

stackControl1 <- trainControl(method = "repeatedcv", number = 10, repeats = 3, savePredictions = TRUE, verboseIter = TRUE)
stack1 <- caretStack(multi_mod, metric = "Accuracy",method ='rf', trControl = stackControl1)
stack_train_preds <- predict(stack1, train)
stack_test_preds <- predict(stack1, test)
train$total_preds <- stack_train_preds1
test$total_preds <- stack_test_preds1
print('Unweighted ensemble classifier results')
confusionMatrix(as.factor(test$Class),as.factor(test$total_preds))

# Stacking for testing the ensemble
stackControl <- trainControl(method = "repeatedcv", number = 10, repeats = 3, savePredictions = TRUE, classProbs = TRUE, verboseIter = TRUE)
#Stacking the ensemble based on Random Forest 
stack <- caretStack(multi_mod,method = 'rf', metric = "Accuracy", trControl = stackControl)
# Predicting the probability values for training and testing test based on the stack
stack_train_preds <- data.frame(predict(stack, train, type = "prob"))
stack_test_preds <- data.frame(predict(stack, test, type = "prob"))
#Finding the threshold value for classifying decision
thresholds <- seq(0, 1, .05)
num_thresh <- length(thresholds)
errors <- rep(0, num_thresh)
iter <- 1
for (i in thresholds) {
  threshold_value <- i
  train_pred <- ifelse(stack_train_preds > threshold_value, "Y", "N")
  conf_mat <- table(true = train$Class, pred = train_pred)
  errors[iter]<- classification_error(conf_mat) 
  iter <- iter + 1
}
result <- data.table(cbind(thresholds, errors))
final_value <- result[which(result$error == min(result$errors))]
test_pred <- ifelse(stack_test_preds >= final_value$thresholds, 1, 0)
test$total_pred <- as.factor(test_pred)

#To make RF perform classification instead of regression
train$Class <- as.character(train$Class)
train$Class <- as.factor(train$Class)
rf <- randomForest(Class ~ ., data = train, importance = TRUE)
rf_test_pred <- predict(rf, test)
rf_conf_mat <- table(true = test[,outcomeName], pred = test$total_pred)
#print the random forest model details on training set
print("Performance details of the ensemble on training set:")
print(rf)
print("Confusion matrix of stacked Ensemble Classifier on training dataset:")
print(rf_conf_mat)

print("----------TASK 3----------")
print("Ensemble Classifier on all 7 models:")
control <- trainControl(method = "repeatedcv", number = 10, repeats = 3, search = "grid", savePredictions = "final", classProbs = TRUE, verboseIter = TRUE)
# List of algorithms to use in ensemble
alg_list <- c("glm", "ctree", "nb", "nnet", "knn","rf","adaboost")
multi_mod <- caretList(train[,predictors],train[,outcomeName], trControl = control, methodList = alg_list, metric = "Accuracy")
# Results on unweighted dataset
res <- resamples(multi_mod)
summary(res)
# Stacking for testing the ensemble
stackControl1 <- trainControl(method = "repeatedcv", number = 10, repeats = 3, savePredictions = TRUE, classProbs = FALSE, verboseIter = TRUE)
#Stacking the ensemble
stack1 <- caretStack(multi_mod, metric = "Accuracy",method = 'rf',trControl = stackControl1)
stackControl <- trainControl(method = "repeatedcv", number = 10, repeats = 3, savePredictions = TRUE, classProbs = TRUE, verboseIter = TRUE)
stack <- caretStack(multi_mod, metric = "Accuracy",method = 'rf', trControl = stackControl)
stack_train_preds1 <- predict(stack1, train)
stack_test_preds1 <- predict(stack1, test)
test$total_preds <- stack_test_preds1
train$total_preds <- stack_train_preds1
#print the random forest model details on training set
print("Confusion matrix of unweighted stacked Ensemble Classifier on test dataset:")
print('Unweighted ensemble classifier results')
confusionMatrix(as.factor(test$Class),as.factor(test$total_preds))

# Predicting the probability values for training and testing test based on the stack
stack_train_preds <- data.frame(predict(stack, train, type = "prob"))
stack_test_preds <- data.frame(predict(stack, test, type = "prob"))
train$total_preds <- stack_train_preds1
test$total_preds <- stack_test_preds1
confusionMatrix(as.factor(test$Class),as.factor(test$total_preds))
#Finding the threshold value for classifying decision
thresholds <- seq(0, 1, .05)
num_thresh <- length(thresholds)
errors <- rep(0, num_thresh)
iter <- 1

for (i in thresholds) {
  threshold_value <- i
  train_pred <- ifelse(stack_train_preds > threshold_value, "Y", "N")
  conf_mat <- table(true = train$Class, pred = train_pred)
  errors[iter]<- classification_error(conf_mat) 
  iter <- iter + 1
}

result <- data.table(cbind(thresholds, errors))
final_value <- result[which(result$error == min(result$errors))]
test_pred <- ifelse(stack_test_preds >= final_value$thresholds, 1, 0)
test$total_pred <- as.factor(test_pred)

#To make RF perform classification instead of regression
train$Class <- as.character(train$Class)
train$Class <- as.factor(train$Class)
rf <- randomForest(Class ~ ., data = train, importance = TRUE)
rf_test_pred <- predict(rf, test)
rf_conf_mat <- table(true = test[,outcomeName], pred = test$total_pred)

#print the random forest model details on training set
print("Performance details of the ensemble on training set:")
print(rf)
print("Confusion matrix of stacked Ensemble Classifier on test dataset:")
print(rf_conf_mat)


